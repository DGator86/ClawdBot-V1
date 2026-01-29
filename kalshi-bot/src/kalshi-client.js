// Kalshi API Client with RSA Authentication
import crypto from 'crypto';
import axios from 'axios';
import { CONFIG } from './config.js';

export class KalshiClient {
  constructor(apiKeyId, privateKey) {
    this.apiKeyId = apiKeyId;
    this.privateKey = privateKey;
    this.baseUrl = CONFIG.KALSHI_API_BASE;
  }

  // Generate RSA signature for authentication
  signRequest(method, path, timestamp) {
    const message = `${timestamp}${method}${path}`;
    const sign = crypto.createSign('RSA-SHA256');
    sign.update(message);
    sign.end();
    return sign.sign(this.privateKey, 'base64');
  }

  // Make authenticated request to Kalshi API
  async request(method, path, data = null) {
    const timestamp = Date.now().toString();
    const signature = this.signRequest(method.toUpperCase(), path, timestamp);

    const headers = {
      'KALSHI-ACCESS-KEY': this.apiKeyId,
      'KALSHI-ACCESS-SIGNATURE': signature,
      'KALSHI-ACCESS-TIMESTAMP': timestamp,
      'Content-Type': 'application/json',
    };

    try {
      const response = await axios({
        method,
        url: `${this.baseUrl}${path}`,
        headers,
        data,
      });
      return response.data;
    } catch (error) {
      console.error(`Kalshi API Error: ${error.response?.data?.message || error.message}`);
      throw error;
    }
  }

  // === ACCOUNT ===
  async getBalance() {
    const data = await this.request('GET', '/portfolio/balance');
    return {
      balance: data.balance / 100,           // Convert cents to dollars
      available: data.portfolio_value / 100,
    };
  }

  async getPositions() {
    const data = await this.request('GET', '/portfolio/positions');
    return data.market_positions || [];
  }

  // === MARKETS ===
  async getMarkets(params = {}) {
    const query = new URLSearchParams(params).toString();
    const path = query ? `/markets?${query}` : '/markets';
    const data = await this.request('GET', path);
    return data.markets || [];
  }

  async getMarket(ticker) {
    const data = await this.request('GET', `/markets/${ticker}`);
    return data.market;
  }

  async getCryptoHourlyMarkets() {
    // Filter for hourly crypto markets (BTC, ETH, SOL)
    const allMarkets = await this.getMarkets({ status: 'open', limit: 200 });

    return allMarkets.filter(market => {
      const ticker = market.ticker.toUpperCase();
      const isCrypto = CONFIG.ALLOWED_CRYPTOS.some(c => ticker.includes(c));
      const isHourly = ticker.includes('1H') || market.subtitle?.toLowerCase().includes('hour');
      return isCrypto && isHourly;
    });
  }

  // === ORDERBOOK ===
  async getOrderbook(ticker) {
    const data = await this.request('GET', `/markets/${ticker}/orderbook`);
    return data.orderbook;
  }

  // === ORDERS ===
  async createOrder(params) {
    // Safety check: enforce max trade cost
    if (params.count * (params.price || 100) / 100 > CONFIG.MAX_TRADE_COST) {
      throw new Error(`Order exceeds max trade cost of $${CONFIG.MAX_TRADE_COST}`);
    }

    return await this.request('POST', '/portfolio/orders', {
      ticker: params.ticker,
      action: params.action,           // 'buy' or 'sell'
      side: params.side,               // 'yes' or 'no'
      type: params.type || 'limit',    // 'limit' or 'market'
      count: params.count,             // Number of contracts
      ...(params.type === 'limit' && { yes_price: params.price, no_price: 100 - params.price }),
    });
  }

  async cancelOrder(orderId) {
    return await this.request('DELETE', `/portfolio/orders/${orderId}`);
  }

  async getOrders(params = {}) {
    const query = new URLSearchParams(params).toString();
    const path = query ? `/portfolio/orders?${query}` : '/portfolio/orders';
    const data = await this.request('GET', path);
    return data.orders || [];
  }

  // === TRADES ===
  async getTrades(ticker) {
    const data = await this.request('GET', `/markets/${ticker}/trades`);
    return data.trades || [];
  }

  async getPortfolioHistory() {
    const data = await this.request('GET', '/portfolio/history');
    return data;
  }
}

export default KalshiClient;
