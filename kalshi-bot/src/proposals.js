// Trade Proposal Generator
import { v4 as uuidv4 } from 'uuid';
import { CONFIG } from './config.js';

export class ProposalManager {
  constructor(kalshiClient, analysisEngine, riskManager) {
    this.client = kalshiClient;
    this.analysis = analysisEngine;
    this.risk = riskManager;
    this.proposals = new Map(); // Active proposals
  }

  // Generate trade proposals based on analysis
  async generateProposals() {
    const proposals = [];

    // Get available hourly crypto markets
    const markets = await this.client.getCryptoHourlyMarkets();
    console.log(`Found ${markets.length} hourly crypto markets`);

    for (const market of markets) {
      // Determine which crypto this market is for
      const crypto = this.extractCrypto(market.ticker);
      if (!crypto) continue;

      try {
        // Run technical analysis
        const analysis = await this.analysis.analyze(crypto);

        // Get market details
        const orderbook = await this.client.getOrderbook(market.ticker);

        // Generate proposal if confidence is high enough
        const proposal = this.evaluateMarket(market, analysis, orderbook);

        if (proposal && proposal.confidence >= CONFIG.MIN_CONFIDENCE) {
          proposals.push(proposal);
        }
      } catch (e) {
        console.error(`Error analyzing ${market.ticker}: ${e.message}`);
      }
    }

    // Store proposals
    for (const p of proposals) {
      this.proposals.set(p.id, p);
    }

    // Clean expired proposals
    this.cleanExpired();

    return proposals;
  }

  // Extract crypto symbol from market ticker
  extractCrypto(ticker) {
    ticker = ticker.toUpperCase();
    for (const crypto of CONFIG.ALLOWED_CRYPTOS) {
      if (ticker.includes(crypto)) return crypto;
    }
    return null;
  }

  // Evaluate a market and generate a proposal
  evaluateMarket(market, analysis, orderbook) {
    const { trend, velocity, pinning, price, indicators } = analysis;

    // Parse market strike price from rules/title
    const strikePrice = this.parseStrikePrice(market);
    if (!strikePrice) {
      console.log(`Could not parse strike price for ${market.ticker}`);
      return null;
    }

    // Calculate probability based on analysis
    const prediction = this.predictOutcome(price, strikePrice, trend, velocity, pinning, indicators);

    if (!prediction) return null;

    // Calculate optimal position size (respecting max trade cost)
    const bestPrice = prediction.side === 'yes'
      ? orderbook?.yes?.[0]?.price || 50
      : orderbook?.no?.[0]?.price || 50;

    const maxContracts = Math.floor((CONFIG.MAX_TRADE_COST * 100) / bestPrice);
    const contracts = Math.min(maxContracts, 10); // Cap at 10 contracts

    if (contracts < 1) return null;

    const cost = (contracts * bestPrice) / 100;

    return {
      id: uuidv4().slice(0, 8),
      timestamp: Date.now(),
      expiry: Date.now() + (CONFIG.PROPOSAL_EXPIRY_MINUTES * 60 * 1000),
      confirmed: false,

      // Market info
      ticker: market.ticker,
      crypto: this.extractCrypto(market.ticker),
      title: market.title || market.ticker,
      strikePrice,
      currentPrice: price,

      // Trade details
      action: 'buy',
      side: prediction.side,
      contracts,
      price: bestPrice,
      cost: cost.toFixed(2),

      // Analysis
      confidence: prediction.confidence,
      reasoning: prediction.reasoning,
      trend: trend.direction,
      velocity: velocity.direction,

      // Potential outcomes
      maxProfit: ((100 - bestPrice) * contracts / 100).toFixed(2),
      maxLoss: cost.toFixed(2),
    };
  }

  // Parse strike price from market data
  parseStrikePrice(market) {
    // Try to extract from title like "ETH above $3028.24"
    const patterns = [
      /above\s*\$?([\d,]+\.?\d*)/i,
      /below\s*\$?([\d,]+\.?\d*)/i,
      /at least\s*\$?([\d,]+\.?\d*)/i,
      /([\d,]+\.?\d*)/,
    ];

    const text = `${market.title} ${market.subtitle || ''} ${market.rules_primary || ''}`;

    for (const pattern of patterns) {
      const match = text.match(pattern);
      if (match) {
        return parseFloat(match[1].replace(/,/g, ''));
      }
    }

    return null;
  }

  // Predict market outcome based on analysis
  predictOutcome(currentPrice, strikePrice, trend, velocity, pinning, indicators) {
    const isAboveMarket = strikePrice > currentPrice;
    const distance = Math.abs(strikePrice - currentPrice);
    const distancePercent = (distance / currentPrice) * 100;

    let confidence = 0.5; // Start neutral
    const reasoning = [];

    // Trend analysis
    if (trend.direction === 'BULLISH') {
      if (isAboveMarket) {
        confidence += 0.1 * trend.strength;
        reasoning.push(`Bullish trend supports price rising to $${strikePrice}`);
      } else {
        confidence += 0.15 * trend.strength; // Even more confident it stays above
        reasoning.push(`Bullish trend supports staying above $${strikePrice}`);
      }
    } else if (trend.direction === 'BEARISH') {
      if (!isAboveMarket) {
        confidence += 0.1 * trend.strength;
        reasoning.push(`Bearish trend supports price falling to $${strikePrice}`);
      } else {
        confidence -= 0.1 * trend.strength;
        reasoning.push(`Bearish trend works against reaching $${strikePrice}`);
      }
    }

    // Velocity analysis
    if (velocity.magnitude > 1) {
      if (velocity.direction === 'UP' && isAboveMarket) {
        confidence += 0.1;
        reasoning.push(`Strong upward velocity (${velocity.magnitude.toFixed(1)}%)`);
      } else if (velocity.direction === 'DOWN' && !isAboveMarket) {
        confidence += 0.1;
        reasoning.push(`Strong downward velocity (${velocity.magnitude.toFixed(1)}%)`);
      }
    }

    // Distance penalty - harder to reach farther strikes
    if (distancePercent > 2) {
      confidence -= distancePercent * 0.02;
      reasoning.push(`Strike is ${distancePercent.toFixed(1)}% away - distance penalty`);
    } else if (distancePercent < 0.5) {
      confidence += 0.1;
      reasoning.push(`Strike is very close (${distancePercent.toFixed(2)}%)`);
    }

    // RSI considerations
    if (indicators.rsi) {
      if (indicators.rsi > 70 && isAboveMarket) {
        confidence -= 0.1;
        reasoning.push(`RSI overbought (${indicators.rsi.toFixed(0)}) - may reverse`);
      } else if (indicators.rsi < 30 && !isAboveMarket) {
        confidence -= 0.1;
        reasoning.push(`RSI oversold (${indicators.rsi.toFixed(0)}) - may bounce`);
      }
    }

    // Pinning near strike
    if (pinning.isPinned) {
      const nearStrike = pinning.pins.some(p => Math.abs(p.level - strikePrice) < currentPrice * 0.005);
      if (nearStrike) {
        confidence -= 0.05;
        reasoning.push('Price pinned near strike - outcome uncertain');
      }
    }

    // Clamp confidence
    confidence = Math.max(0.1, Math.min(0.9, confidence));

    // Determine side
    // If we're confident price will be ABOVE strike, buy YES
    // If we're confident price will be BELOW strike, buy NO
    const predictAbove = (trend.direction === 'BULLISH' && confidence > 0.5) ||
                         (!isAboveMarket && confidence > 0.6);

    return {
      side: predictAbove ? 'yes' : 'no',
      confidence: confidence,
      reasoning,
    };
  }

  // Get a specific proposal
  getProposal(id) {
    return this.proposals.get(id);
  }

  // Get all active proposals
  getActiveProposals() {
    this.cleanExpired();
    return Array.from(this.proposals.values());
  }

  // Confirm a proposal for execution
  confirmProposal(id) {
    const proposal = this.proposals.get(id);
    if (!proposal) {
      throw new Error(`Proposal ${id} not found`);
    }
    if (Date.now() > proposal.expiry) {
      this.proposals.delete(id);
      throw new Error(`Proposal ${id} has expired`);
    }
    proposal.confirmed = true;
    proposal.confirmedAt = Date.now();
    return proposal;
  }

  // Reject/delete a proposal
  rejectProposal(id) {
    const had = this.proposals.has(id);
    this.proposals.delete(id);
    return had;
  }

  // Clean up expired proposals
  cleanExpired() {
    const now = Date.now();
    for (const [id, proposal] of this.proposals) {
      if (now > proposal.expiry) {
        this.proposals.delete(id);
      }
    }
  }

  // Format proposal for display
  formatProposal(proposal) {
    const lines = [
      `=== TRADE PROPOSAL ${proposal.id} ===`,
      ``,
      `Market: ${proposal.title}`,
      `Crypto: ${proposal.crypto}`,
      ``,
      `Current Price: $${proposal.currentPrice.toLocaleString()}`,
      `Strike Price: $${proposal.strikePrice.toLocaleString()}`,
      ``,
      `ACTION: BUY ${proposal.side.toUpperCase()}`,
      `Contracts: ${proposal.contracts}`,
      `Price: ${proposal.price}¢`,
      `Cost: $${proposal.cost}`,
      ``,
      `Max Profit: $${proposal.maxProfit}`,
      `Max Loss: $${proposal.maxLoss}`,
      ``,
      `Confidence: ${(proposal.confidence * 100).toFixed(0)}%`,
      `Trend: ${proposal.trend}`,
      `Velocity: ${proposal.velocity}`,
      ``,
      `Reasoning:`,
      ...proposal.reasoning.map(r => `  - ${r}`),
      ``,
      `Expires: ${new Date(proposal.expiry).toLocaleTimeString()}`,
      ``,
      `To execute: CONFIRM ${proposal.id}`,
      `To reject: REJECT ${proposal.id}`,
    ];

    return lines.join('\n');
  }
}

export default ProposalManager;
