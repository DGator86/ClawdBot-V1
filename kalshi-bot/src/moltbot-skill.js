// Moltbot Skill: Kalshi Trading Controller
// This skill adds /kalshi commands to your Telegram bot
import axios from 'axios';
import { CONFIG } from './config.js';

const BOT_URL = `http://${CONFIG.API_HOST}:${CONFIG.API_PORT}`;

// Skill definition for moltbot
export const kalshiSkill = {
  name: 'kalshi-trading',
  version: '1.0.0',
  description: 'Kalshi hourly crypto trading with technical analysis',

  // Commands this skill handles
  commands: [
    { command: 'kalshi', description: 'Kalshi trading commands' },
    { command: 'k', description: 'Kalshi shortcut' },
  ],

  // Handle incoming messages
  async handle(message, context) {
    const text = message.text?.trim() || '';
    const parts = text.split(/\s+/);
    const cmd = parts[0]?.toLowerCase();

    // Check if this is a kalshi command
    if (cmd !== '/kalshi' && cmd !== '/k') {
      return null; // Not for us
    }

    const subCmd = parts[1]?.toLowerCase();
    const args = parts.slice(2);

    try {
      switch (subCmd) {
        case 'status':
        case 's':
          return await this.getStatus();

        case 'markets':
        case 'm':
          return await this.getMarkets();

        case 'analyze':
        case 'a':
          return await this.analyze(args[0]);

        case 'propose':
        case 'p':
          return await this.propose();

        case 'list':
        case 'l':
          return await this.listProposals();

        case 'confirm':
        case 'c':
          return await this.confirm(args[0]);

        case 'reject':
        case 'r':
          return await this.reject(args[0]);

        case 'kill':
          return await this.kill(args.join(' '));

        case 'reset':
          return await this.resetKill(args[0]);

        case 'history':
        case 'h':
          return await this.history();

        default:
          return this.help();
      }
    } catch (error) {
      return `Error: ${error.message}`;
    }
  },

  // Help text
  help() {
    return `
**Kalshi Trading Commands**

/kalshi status - Portfolio & risk metrics
/kalshi markets - Available crypto markets
/kalshi analyze <BTC|ETH|SOL> - Technical analysis
/kalshi propose - Generate trade proposals
/kalshi list - View active proposals
/kalshi confirm <id> - Execute a proposal
/kalshi reject <id> - Discard a proposal
/kalshi kill [reason] - Emergency stop
/kalshi reset <code> - Reset kill switch
/kalshi history - Recent trades

**Shortcuts:** /k s, /k m, /k a eth, etc.

**Safety Limits (hardcoded):**
- Max trade: $${CONFIG.MAX_TRADE_COST}
- Max drawdown: ${CONFIG.MAX_DRAWDOWN_PERCENT}%
- Confirmation required: Yes
    `.trim();
  },

  // Get account status
  async getStatus() {
    const { data } = await axios.get(`${BOT_URL}/status`);
    const r = data.risk;

    let status = `
**Account Status**

Balance: $${r.currentEquity?.toFixed(2) || 'N/A'}
Starting: $${r.startingEquity?.toFixed(2) || 'N/A'}
High Water: $${r.highWaterMark?.toFixed(2) || 'N/A'}

**Drawdown**
From Start: ${r.drawdownFromStart}%
From High: ${r.drawdownFromHigh}%
Max Allowed: ${r.maxAllowedDrawdown}%

**Stats**
Trades: ${r.stats.totalTrades}
Win Rate: ${r.stats.winRate}%
P&L: $${r.stats.totalPnL}
    `.trim();

    if (r.isKilled) {
      status = `⚠️ **KILL SWITCH ACTIVE**\nReason: ${r.killReason}\n\n${status}`;
    }

    if (data.positions?.length > 0) {
      status += '\n\n**Open Positions**\n';
      for (const p of data.positions.slice(0, 5)) {
        status += `- ${p.ticker}: ${p.position} contracts\n`;
      }
    }

    return status;
  },

  // Get available markets
  async getMarkets() {
    const { data } = await axios.get(`${BOT_URL}/markets`);

    if (data.count === 0) {
      return 'No hourly crypto markets currently open.';
    }

    let msg = `**Hourly Crypto Markets (${data.count})**\n\n`;

    for (const m of data.markets.slice(0, 10)) {
      msg += `**${m.ticker}**\n`;
      msg += `${m.title}\n`;
      msg += `YES: ${m.yes_price}¢ | NO: ${m.no_price}¢\n\n`;
    }

    return msg;
  },

  // Technical analysis
  async analyze(symbol) {
    if (!symbol) {
      return 'Usage: /kalshi analyze <BTC|ETH|SOL>';
    }

    symbol = symbol.toUpperCase();
    if (!CONFIG.ALLOWED_CRYPTOS.includes(symbol)) {
      return `Symbol must be one of: ${CONFIG.ALLOWED_CRYPTOS.join(', ')}`;
    }

    const { data } = await axios.get(`${BOT_URL}/analyze/${symbol}`);

    return `
**${symbol} Technical Analysis**

Price: $${data.price?.toLocaleString()}

**Trend:** ${data.trend.direction} (strength ${data.trend.strength}/4)
${data.trend.signals.map(s => `- ${s}`).join('\n')}

**Velocity:** ${data.velocity.direction} ${data.velocity.magnitude?.toFixed(2)}%
- ${data.velocity.interpretation}
- Volatility: ${data.velocity.volatility?.toFixed(2)}%

**RSI:** ${data.indicators.rsi?.toFixed(1)}

**Pinning:** ${data.pinning.isPinned ? 'Yes' : 'No'}
${data.pinning.pins?.map(p => `- Near ${p.type}: $${p.level?.toLocaleString()}`).join('\n') || ''}

**Levels**
Support: $${data.indicators.levels.support[0]?.toLocaleString() || 'N/A'}
Resistance: $${data.indicators.levels.resistance[0]?.toLocaleString() || 'N/A'}
    `.trim();
  },

  // Generate proposals
  async propose() {
    const { data } = await axios.post(`${BOT_URL}/propose`);

    if (data.count === 0) {
      return 'No trade proposals generated. Markets may not meet confidence threshold.';
    }

    let msg = `**Generated ${data.count} Proposal(s)**\n\n`;

    for (const p of data.raw) {
      msg += `**[${p.id}] ${p.crypto}**\n`;
      msg += `${p.title}\n`;
      msg += `BUY ${p.side.toUpperCase()} x${p.contracts} @ ${p.price}¢\n`;
      msg += `Cost: $${p.cost} | Confidence: ${(p.confidence * 100).toFixed(0)}%\n`;
      msg += `→ CONFIRM ${p.id} or REJECT ${p.id}\n\n`;
    }

    return msg;
  },

  // List active proposals
  async listProposals() {
    const { data } = await axios.get(`${BOT_URL}/proposals`);

    if (data.count === 0) {
      return 'No active proposals. Use /kalshi propose to generate.';
    }

    let msg = `**Active Proposals (${data.count})**\n\n`;

    for (const p of data.raw) {
      const expires = new Date(p.expiry).toLocaleTimeString();
      msg += `**[${p.id}] ${p.crypto}**\n`;
      msg += `BUY ${p.side.toUpperCase()} x${p.contracts} @ ${p.price}¢\n`;
      msg += `Cost: $${p.cost} | Conf: ${(p.confidence * 100).toFixed(0)}%\n`;
      msg += `Expires: ${expires}\n\n`;
    }

    msg += 'Use /kalshi confirm <id> to execute';
    return msg;
  },

  // Confirm a proposal
  async confirm(id) {
    if (!id) {
      return 'Usage: /kalshi confirm <proposal-id>';
    }

    const { data } = await axios.post(`${BOT_URL}/confirm/${id}`);

    return `
✅ **Order Executed**

${data.message}

Ticker: ${data.proposal.ticker}
Side: ${data.proposal.side.toUpperCase()}
Contracts: ${data.proposal.contracts}
Price: ${data.proposal.price}¢
Cost: $${data.proposal.cost}
    `.trim();
  },

  // Reject a proposal
  async reject(id) {
    if (!id) {
      return 'Usage: /kalshi reject <proposal-id>';
    }

    const { data } = await axios.delete(`${BOT_URL}/reject/${id}`);

    return data.rejected
      ? `❌ Proposal ${id} rejected`
      : `Proposal ${id} not found (may have expired)`;
  },

  // Trigger kill switch
  async kill(reason) {
    const { data } = await axios.post(`${BOT_URL}/kill`, {
      reason: reason || 'Manual kill via Telegram',
    });

    return `
🛑 **KILL SWITCH ACTIVATED**

Reason: ${data.reason}
Time: ${new Date(data.timestamp).toLocaleString()}

All open orders cancelled.
Trading halted until reset.

To reset: /kalshi reset RESET-RISK-CONFIRM
    `.trim();
  },

  // Reset kill switch
  async resetKill(code) {
    if (code !== 'RESET-RISK-CONFIRM') {
      return `
To reset the kill switch, use:
/kalshi reset RESET-RISK-CONFIRM

⚠️ This will reset your starting equity to current balance.
      `.trim();
    }

    const { data } = await axios.post(`${BOT_URL}/reset-kill`, { code });

    return data.reset
      ? '✅ Kill switch reset. Trading re-enabled.'
      : 'Failed to reset kill switch.';
  },

  // Trade history
  async history() {
    const { data } = await axios.get(`${BOT_URL}/history`);

    if (!data.settlements?.length) {
      return 'No trade history yet.';
    }

    let msg = '**Recent Trades**\n\n';

    for (const t of data.settlements.slice(0, 10)) {
      const pnl = t.revenue - t.cost;
      const emoji = pnl >= 0 ? '🟢' : '🔴';
      msg += `${emoji} ${t.ticker}\n`;
      msg += `P&L: $${(pnl / 100).toFixed(2)}\n\n`;
    }

    return msg;
  },
};

export default kalshiSkill;
