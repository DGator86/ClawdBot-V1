// Risk Management System - SAFETY CRITICAL
// All limits are HARDCODED and cannot be overridden
import fs from 'fs';
import path from 'path';
import { CONFIG } from './config.js';

const STATE_FILE = path.join(process.cwd(), '.risk-state.json');

export class RiskManager {
  constructor(kalshiClient) {
    this.client = kalshiClient;
    this.state = this.loadState();
    this.isKilled = false;
  }

  // Load persisted state
  loadState() {
    try {
      if (fs.existsSync(STATE_FILE)) {
        return JSON.parse(fs.readFileSync(STATE_FILE, 'utf8'));
      }
    } catch (e) {
      console.error('Failed to load risk state:', e.message);
    }

    return {
      startingEquity: null,
      highWaterMark: null,
      totalTrades: 0,
      wins: 0,
      losses: 0,
      totalPnL: 0,
      lastCheck: null,
      killTriggered: false,
      killReason: null,
    };
  }

  // Save state to disk
  saveState() {
    try {
      fs.writeFileSync(STATE_FILE, JSON.stringify(this.state, null, 2));
    } catch (e) {
      console.error('Failed to save risk state:', e.message);
    }
  }

  // Initialize with current account balance
  async initialize() {
    const balance = await this.client.getBalance();
    const currentEquity = balance.balance;

    if (!this.state.startingEquity) {
      this.state.startingEquity = currentEquity;
      this.state.highWaterMark = currentEquity;
      console.log(`Risk Manager initialized. Starting equity: $${currentEquity.toFixed(2)}`);
    }

    // Update high water mark if equity increased
    if (currentEquity > this.state.highWaterMark) {
      this.state.highWaterMark = currentEquity;
    }

    this.state.lastCheck = Date.now();
    this.saveState();

    // Check if already killed
    if (this.state.killTriggered) {
      this.isKilled = true;
      console.error(`KILL SWITCH ACTIVE: ${this.state.killReason}`);
    }

    return this.state;
  }

  // Check drawdown and trigger kill switch if needed
  async checkDrawdown() {
    const balance = await this.client.getBalance();
    const currentEquity = balance.balance;

    const drawdownFromStart = ((this.state.startingEquity - currentEquity) / this.state.startingEquity) * 100;
    const drawdownFromHigh = ((this.state.highWaterMark - currentEquity) / this.state.highWaterMark) * 100;

    const maxDrawdown = Math.max(drawdownFromStart, drawdownFromHigh);

    console.log(`Drawdown check: ${maxDrawdown.toFixed(2)}% (limit: ${CONFIG.MAX_DRAWDOWN_PERCENT}%)`);

    if (maxDrawdown >= CONFIG.MAX_DRAWDOWN_PERCENT) {
      await this.triggerKillSwitch(`Drawdown exceeded ${CONFIG.MAX_DRAWDOWN_PERCENT}%: ${maxDrawdown.toFixed(2)}%`);
      return false;
    }

    return true;
  }

  // KILL SWITCH - stops all trading immediately
  async triggerKillSwitch(reason) {
    console.error(`\n!!! KILL SWITCH TRIGGERED !!!\nReason: ${reason}\n`);

    this.isKilled = true;
    this.state.killTriggered = true;
    this.state.killReason = reason;
    this.state.killTime = Date.now();
    this.saveState();

    // Cancel all open orders
    try {
      const orders = await this.client.getOrders({ status: 'open' });
      for (const order of orders) {
        try {
          await this.client.cancelOrder(order.order_id);
          console.log(`Cancelled order: ${order.order_id}`);
        } catch (e) {
          console.error(`Failed to cancel order ${order.order_id}: ${e.message}`);
        }
      }
    } catch (e) {
      console.error(`Failed to fetch/cancel orders: ${e.message}`);
    }

    return {
      killed: true,
      reason,
      timestamp: this.state.killTime,
    };
  }

  // Reset kill switch (manual override required)
  resetKillSwitch(confirmationCode) {
    // Require specific confirmation to prevent accidental reset
    if (confirmationCode !== 'RESET-RISK-CONFIRM') {
      throw new Error('Invalid confirmation code. Use: RESET-RISK-CONFIRM');
    }

    console.log('Kill switch reset by user');
    this.isKilled = false;
    this.state.killTriggered = false;
    this.state.killReason = null;
    this.state.killTime = null;
    // Reset starting equity to current
    this.state.startingEquity = null;
    this.state.highWaterMark = null;
    this.saveState();

    return { reset: true };
  }

  // Validate a trade before execution
  validateTrade(proposal) {
    const errors = [];

    // Check if killed
    if (this.isKilled) {
      errors.push(`KILL SWITCH ACTIVE: ${this.state.killReason}`);
      return { valid: false, errors };
    }

    // Check max trade cost (HARDCODED LIMIT)
    const cost = proposal.contracts * (proposal.price / 100);
    if (cost > CONFIG.MAX_TRADE_COST) {
      errors.push(`Trade cost $${cost.toFixed(2)} exceeds max $${CONFIG.MAX_TRADE_COST}`);
    }

    // Check if confirmation required
    if (CONFIG.REQUIRE_CONFIRMATION && !proposal.confirmed) {
      errors.push('Trade requires confirmation. Use: CONFIRM <proposal-id>');
    }

    // Check allowed markets
    const crypto = proposal.crypto?.toUpperCase();
    if (!CONFIG.ALLOWED_CRYPTOS.includes(crypto)) {
      errors.push(`${crypto} not in allowed list: ${CONFIG.ALLOWED_CRYPTOS.join(', ')}`);
    }

    return {
      valid: errors.length === 0,
      errors,
    };
  }

  // Record a completed trade
  recordTrade(trade) {
    this.state.totalTrades++;

    if (trade.pnl > 0) {
      this.state.wins++;
    } else if (trade.pnl < 0) {
      this.state.losses++;
    }

    this.state.totalPnL += trade.pnl;
    this.saveState();
  }

  // Get risk status report
  async getStatus() {
    const balance = await this.client.getBalance();
    const currentEquity = balance.balance;

    const drawdownFromStart = this.state.startingEquity
      ? ((this.state.startingEquity - currentEquity) / this.state.startingEquity) * 100
      : 0;

    const drawdownFromHigh = this.state.highWaterMark
      ? ((this.state.highWaterMark - currentEquity) / this.state.highWaterMark) * 100
      : 0;

    const winRate = this.state.totalTrades > 0
      ? (this.state.wins / this.state.totalTrades) * 100
      : 0;

    return {
      currentEquity,
      startingEquity: this.state.startingEquity,
      highWaterMark: this.state.highWaterMark,
      drawdownFromStart: drawdownFromStart.toFixed(2),
      drawdownFromHigh: drawdownFromHigh.toFixed(2),
      maxAllowedDrawdown: CONFIG.MAX_DRAWDOWN_PERCENT,
      isKilled: this.isKilled,
      killReason: this.state.killReason,
      stats: {
        totalTrades: this.state.totalTrades,
        wins: this.state.wins,
        losses: this.state.losses,
        winRate: winRate.toFixed(1),
        totalPnL: this.state.totalPnL.toFixed(2),
      },
      limits: {
        maxTradeCost: CONFIG.MAX_TRADE_COST,
        maxDrawdown: CONFIG.MAX_DRAWDOWN_PERCENT,
        requireConfirmation: CONFIG.REQUIRE_CONFIRMATION,
      },
    };
  }
}

export default RiskManager;
