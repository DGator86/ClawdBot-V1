// Kalshi Crypto Trading Bot - Main Entry Point
import 'dotenv/config';
import express from 'express';
import { KalshiClient } from './kalshi-client.js';
import { TechnicalAnalysis } from './analysis.js';
import { RiskManager } from './risk-manager.js';
import { ProposalManager } from './proposals.js';
import { CONFIG } from './config.js';

// Validate environment
if (!process.env.KALSHI_API_KEY_ID || !process.env.KALSHI_PRIVATE_KEY) {
  console.error('Missing KALSHI_API_KEY_ID or KALSHI_PRIVATE_KEY in .env');
  process.exit(1);
}

// Initialize components
const kalshi = new KalshiClient(
  process.env.KALSHI_API_KEY_ID,
  process.env.KALSHI_PRIVATE_KEY
);

const analysis = new TechnicalAnalysis();
const risk = new RiskManager(kalshi);
const proposals = new ProposalManager(kalshi, analysis, risk);

// Express API server for moltbot integration
const app = express();
app.use(express.json());

// === API ENDPOINTS ===

// Health check
app.get('/health', (req, res) => {
  res.json({ status: 'ok', timestamp: Date.now() });
});

// Get account status and risk metrics
app.get('/status', async (req, res) => {
  try {
    const riskStatus = await risk.getStatus();
    const positions = await kalshi.getPositions();
    res.json({
      risk: riskStatus,
      positions,
      allowedCryptos: CONFIG.ALLOWED_CRYPTOS,
    });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// Get available hourly crypto markets
app.get('/markets', async (req, res) => {
  try {
    const markets = await kalshi.getCryptoHourlyMarkets();
    res.json({
      count: markets.length,
      markets: markets.map(m => ({
        ticker: m.ticker,
        title: m.title,
        yes_price: m.yes_price,
        no_price: m.no_price,
        volume: m.volume,
        close_time: m.close_time,
      })),
    });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// Run technical analysis for a crypto
app.get('/analyze/:symbol', async (req, res) => {
  try {
    const symbol = req.params.symbol.toUpperCase();
    if (!CONFIG.ALLOWED_CRYPTOS.includes(symbol)) {
      return res.status(400).json({ error: `Symbol must be one of: ${CONFIG.ALLOWED_CRYPTOS.join(', ')}` });
    }
    const result = await analysis.analyze(symbol);
    res.json(result);
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// Generate trade proposals
app.post('/propose', async (req, res) => {
  try {
    // Check kill switch
    if (risk.isKilled) {
      return res.status(403).json({ error: `KILL SWITCH ACTIVE: ${risk.state.killReason}` });
    }

    // Check drawdown before proposing
    const drawdownOk = await risk.checkDrawdown();
    if (!drawdownOk) {
      return res.status(403).json({ error: 'Kill switch triggered due to drawdown' });
    }

    const newProposals = await proposals.generateProposals();
    res.json({
      count: newProposals.length,
      proposals: newProposals.map(p => proposals.formatProposal(p)),
      raw: newProposals,
    });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// Get active proposals
app.get('/proposals', (req, res) => {
  const active = proposals.getActiveProposals();
  res.json({
    count: active.length,
    proposals: active.map(p => proposals.formatProposal(p)),
    raw: active,
  });
});

// Confirm and execute a proposal
app.post('/confirm/:id', async (req, res) => {
  try {
    const id = req.params.id;

    // Check kill switch
    if (risk.isKilled) {
      return res.status(403).json({ error: `KILL SWITCH ACTIVE: ${risk.state.killReason}` });
    }

    // Check drawdown
    const drawdownOk = await risk.checkDrawdown();
    if (!drawdownOk) {
      return res.status(403).json({ error: 'Kill switch triggered due to drawdown' });
    }

    // Get and confirm proposal
    const proposal = proposals.confirmProposal(id);

    // Validate trade
    const validation = risk.validateTrade(proposal);
    if (!validation.valid) {
      return res.status(400).json({ error: validation.errors.join('; ') });
    }

    // Execute the trade
    const order = await kalshi.createOrder({
      ticker: proposal.ticker,
      action: proposal.action,
      side: proposal.side,
      type: 'limit',
      count: proposal.contracts,
      price: proposal.price,
    });

    // Remove executed proposal
    proposals.rejectProposal(id);

    res.json({
      success: true,
      message: `Order placed: BUY ${proposal.contracts} ${proposal.side.toUpperCase()} @ ${proposal.price}¢`,
      order,
      proposal,
    });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// Reject a proposal
app.delete('/reject/:id', (req, res) => {
  const rejected = proposals.rejectProposal(req.params.id);
  res.json({ rejected, id: req.params.id });
});

// Trigger kill switch manually
app.post('/kill', async (req, res) => {
  try {
    const result = await risk.triggerKillSwitch(req.body.reason || 'Manual kill switch');
    res.json(result);
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// Reset kill switch (requires confirmation code)
app.post('/reset-kill', (req, res) => {
  try {
    const code = req.body.code;
    const result = risk.resetKillSwitch(code);
    res.json(result);
  } catch (e) {
    res.status(400).json({ error: e.message });
  }
});

// Get trade history
app.get('/history', async (req, res) => {
  try {
    const history = await kalshi.getPortfolioHistory();
    res.json(history);
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// === STARTUP ===

async function start() {
  console.log('='.repeat(50));
  console.log('  KALSHI CRYPTO TRADING BOT');
  console.log('='.repeat(50));
  console.log(`Allowed cryptos: ${CONFIG.ALLOWED_CRYPTOS.join(', ')}`);
  console.log(`Max trade cost: $${CONFIG.MAX_TRADE_COST}`);
  console.log(`Max drawdown: ${CONFIG.MAX_DRAWDOWN_PERCENT}%`);
  console.log(`Require confirmation: ${CONFIG.REQUIRE_CONFIRMATION}`);
  console.log('='.repeat(50));

  try {
    // Initialize risk manager with current balance
    await risk.initialize();

    // Start API server
    app.listen(CONFIG.API_PORT, CONFIG.API_HOST, () => {
      console.log(`\nAPI server running at http://${CONFIG.API_HOST}:${CONFIG.API_PORT}`);
      console.log('\nEndpoints:');
      console.log('  GET  /status          - Account status & risk metrics');
      console.log('  GET  /markets         - Available hourly crypto markets');
      console.log('  GET  /analyze/:symbol - Technical analysis (BTC/ETH/SOL)');
      console.log('  POST /propose         - Generate trade proposals');
      console.log('  GET  /proposals       - View active proposals');
      console.log('  POST /confirm/:id     - Execute a proposal');
      console.log('  DEL  /reject/:id      - Reject a proposal');
      console.log('  POST /kill            - Trigger kill switch');
      console.log('  POST /reset-kill      - Reset kill switch');
      console.log('  GET  /history         - Trade history');
      console.log('\nReady for moltbot integration.');
    });
  } catch (e) {
    console.error('Failed to start:', e.message);
    process.exit(1);
  }
}

start();
