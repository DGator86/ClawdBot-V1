// Technical Analysis Engine for Crypto Price Prediction
import { EMA, RSI, MACD, BollingerBands, ATR, SMA } from 'technicalindicators';
import axios from 'axios';
import { CONFIG } from './config.js';

export class TechnicalAnalysis {
  constructor() {
    this.priceCache = new Map();
    this.cacheExpiry = 60000; // 1 minute cache
  }

  // Fetch historical price data from CoinGecko
  async fetchPriceHistory(symbol, hours = CONFIG.ANALYSIS.LOOKBACK_HOURS) {
    const cacheKey = `${symbol}-${hours}`;
    const cached = this.priceCache.get(cacheKey);

    if (cached && Date.now() - cached.timestamp < this.cacheExpiry) {
      return cached.data;
    }

    const coinIds = {
      BTC: 'bitcoin',
      ETH: 'ethereum',
      SOL: 'solana',
    };

    const coinId = coinIds[symbol.toUpperCase()];
    if (!coinId) throw new Error(`Unknown symbol: ${symbol}`);

    try {
      // CoinGecko API for hourly data
      const days = Math.ceil(hours / 24);
      const response = await axios.get(
        `https://api.coingecko.com/api/v3/coins/${coinId}/market_chart`,
        { params: { vs_currency: 'usd', days, interval: 'hourly' } }
      );

      const prices = response.data.prices.map(([timestamp, price]) => ({
        timestamp,
        price,
        high: price * 1.001,  // Approximate OHLC from price
        low: price * 0.999,
        close: price,
        open: price,
      }));

      this.priceCache.set(cacheKey, { data: prices, timestamp: Date.now() });
      return prices;
    } catch (error) {
      console.error(`Failed to fetch price data: ${error.message}`);
      throw error;
    }
  }

  // Get current price
  async getCurrentPrice(symbol) {
    const history = await this.fetchPriceHistory(symbol, 2);
    return history[history.length - 1]?.price || 0;
  }

  // Calculate all technical indicators
  calculateIndicators(priceData) {
    const closes = priceData.map(d => d.close);
    const highs = priceData.map(d => d.high);
    const lows = priceData.map(d => d.low);

    // EMAs
    const ema9 = EMA.calculate({ period: 9, values: closes });
    const ema21 = EMA.calculate({ period: 21, values: closes });
    const ema50 = EMA.calculate({ period: 50, values: closes });

    // RSI
    const rsi = RSI.calculate({ period: CONFIG.ANALYSIS.RSI_PERIOD, values: closes });

    // MACD
    const macd = MACD.calculate({
      values: closes,
      fastPeriod: CONFIG.ANALYSIS.MACD_FAST,
      slowPeriod: CONFIG.ANALYSIS.MACD_SLOW,
      signalPeriod: CONFIG.ANALYSIS.MACD_SIGNAL,
      SimpleMAOscillator: false,
      SimpleMASignal: false,
    });

    // Bollinger Bands
    const bb = BollingerBands.calculate({
      period: CONFIG.ANALYSIS.BOLLINGER_PERIOD,
      values: closes,
      stdDev: CONFIG.ANALYSIS.BOLLINGER_STD,
    });

    // ATR (Average True Range)
    const atr = ATR.calculate({
      period: CONFIG.ANALYSIS.ATR_PERIOD,
      high: highs,
      low: lows,
      close: closes,
    });

    // Rate of Change (Velocity)
    const roc = this.calculateROC(closes, 12);

    // Support/Resistance levels
    const levels = this.findSupportResistance(priceData);

    return {
      current: closes[closes.length - 1],
      ema: {
        ema9: ema9[ema9.length - 1],
        ema21: ema21[ema21.length - 1],
        ema50: ema50[ema50.length - 1],
      },
      rsi: rsi[rsi.length - 1],
      macd: macd[macd.length - 1],
      bollingerBands: bb[bb.length - 1],
      atr: atr[atr.length - 1],
      roc: roc[roc.length - 1],
      levels,
    };
  }

  // Rate of Change (momentum/velocity)
  calculateROC(prices, period = 12) {
    const roc = [];
    for (let i = period; i < prices.length; i++) {
      const change = ((prices[i] - prices[i - period]) / prices[i - period]) * 100;
      roc.push(change);
    }
    return roc;
  }

  // Find support and resistance levels
  findSupportResistance(priceData, lookback = 48) {
    const recent = priceData.slice(-lookback);
    const highs = recent.map(d => d.high);
    const lows = recent.map(d => d.low);

    // Simple pivot points
    const highestHigh = Math.max(...highs);
    const lowestLow = Math.min(...lows);
    const currentPrice = recent[recent.length - 1].close;

    // Find local maxima/minima for pinning levels
    const resistance = [];
    const support = [];

    for (let i = 2; i < recent.length - 2; i++) {
      // Local maximum
      if (recent[i].high > recent[i - 1].high &&
          recent[i].high > recent[i - 2].high &&
          recent[i].high > recent[i + 1].high &&
          recent[i].high > recent[i + 2].high) {
        resistance.push(recent[i].high);
      }
      // Local minimum
      if (recent[i].low < recent[i - 1].low &&
          recent[i].low < recent[i - 2].low &&
          recent[i].low < recent[i + 1].low &&
          recent[i].low < recent[i + 2].low) {
        support.push(recent[i].low);
      }
    }

    return {
      resistance: [...new Set(resistance)].sort((a, b) => a - b).slice(-3),
      support: [...new Set(support)].sort((a, b) => b - a).slice(-3),
      highestHigh,
      lowestLow,
      range: highestHigh - lowestLow,
    };
  }

  // Analyze trend direction and strength
  analyzeTrend(indicators) {
    const { ema, macd, rsi, current } = indicators;

    let trendScore = 0;
    let signals = [];

    // EMA alignment
    if (current > ema.ema9 && ema.ema9 > ema.ema21 && ema.ema21 > ema.ema50) {
      trendScore += 2;
      signals.push('Strong uptrend (EMA aligned bullish)');
    } else if (current < ema.ema9 && ema.ema9 < ema.ema21 && ema.ema21 < ema.ema50) {
      trendScore -= 2;
      signals.push('Strong downtrend (EMA aligned bearish)');
    } else if (current > ema.ema21) {
      trendScore += 1;
      signals.push('Moderate uptrend (above EMA21)');
    } else {
      trendScore -= 1;
      signals.push('Moderate downtrend (below EMA21)');
    }

    // MACD
    if (macd) {
      if (macd.MACD > macd.signal && macd.histogram > 0) {
        trendScore += 1;
        signals.push('MACD bullish crossover');
      } else if (macd.MACD < macd.signal && macd.histogram < 0) {
        trendScore -= 1;
        signals.push('MACD bearish crossover');
      }
    }

    // RSI
    if (rsi > CONFIG.ANALYSIS.RSI_OVERBOUGHT) {
      signals.push(`RSI overbought (${rsi.toFixed(1)})`);
    } else if (rsi < CONFIG.ANALYSIS.RSI_OVERSOLD) {
      signals.push(`RSI oversold (${rsi.toFixed(1)})`);
    }

    return {
      direction: trendScore > 0 ? 'BULLISH' : trendScore < 0 ? 'BEARISH' : 'NEUTRAL',
      strength: Math.abs(trendScore),
      score: trendScore,
      signals,
    };
  }

  // Calculate velocity (rate of price change)
  analyzeVelocity(indicators) {
    const { roc, atr, current } = indicators;

    const velocityPercent = roc || 0;
    const volatilityPercent = (atr / current) * 100;

    return {
      velocity: velocityPercent,
      direction: velocityPercent > 0 ? 'UP' : velocityPercent < 0 ? 'DOWN' : 'FLAT',
      magnitude: Math.abs(velocityPercent),
      volatility: volatilityPercent,
      interpretation: Math.abs(velocityPercent) > 2 ? 'STRONG_MOVE' :
                      Math.abs(velocityPercent) > 0.5 ? 'MODERATE_MOVE' : 'CONSOLIDATING',
    };
  }

  // Analyze pinning (price gravitating toward levels)
  analyzePinning(indicators) {
    const { current, levels, bollingerBands } = indicators;
    const pins = [];

    // Check proximity to support/resistance
    const threshold = current * 0.005; // 0.5% proximity

    for (const resistance of levels.resistance) {
      if (Math.abs(current - resistance) < threshold) {
        pins.push({ level: resistance, type: 'RESISTANCE', distance: current - resistance });
      }
    }

    for (const support of levels.support) {
      if (Math.abs(current - support) < threshold) {
        pins.push({ level: support, type: 'SUPPORT', distance: current - support });
      }
    }

    // Bollinger Band pinning
    if (bollingerBands) {
      if (Math.abs(current - bollingerBands.upper) < threshold) {
        pins.push({ level: bollingerBands.upper, type: 'BB_UPPER', distance: current - bollingerBands.upper });
      }
      if (Math.abs(current - bollingerBands.lower) < threshold) {
        pins.push({ level: bollingerBands.lower, type: 'BB_LOWER', distance: current - bollingerBands.lower });
      }
      if (Math.abs(current - bollingerBands.middle) < threshold) {
        pins.push({ level: bollingerBands.middle, type: 'BB_MIDDLE', distance: current - bollingerBands.middle });
      }
    }

    return {
      isPinned: pins.length > 0,
      pins,
      nearestLevel: pins.length > 0 ? pins.reduce((a, b) =>
        Math.abs(a.distance) < Math.abs(b.distance) ? a : b
      ) : null,
    };
  }

  // Full analysis for a symbol
  async analyze(symbol) {
    console.log(`Analyzing ${symbol}...`);

    const priceData = await this.fetchPriceHistory(symbol);
    const indicators = this.calculateIndicators(priceData);

    const trend = this.analyzeTrend(indicators);
    const velocity = this.analyzeVelocity(indicators);
    const pinning = this.analyzePinning(indicators);

    return {
      symbol,
      timestamp: Date.now(),
      price: indicators.current,
      indicators,
      trend,
      velocity,
      pinning,
      summary: this.generateSummary(symbol, indicators, trend, velocity, pinning),
    };
  }

  // Generate human-readable summary
  generateSummary(symbol, indicators, trend, velocity, pinning) {
    const lines = [
      `=== ${symbol} Analysis ===`,
      `Price: $${indicators.current.toLocaleString()}`,
      ``,
      `TREND: ${trend.direction} (strength: ${trend.strength}/4)`,
      ...trend.signals.map(s => `  - ${s}`),
      ``,
      `VELOCITY: ${velocity.direction} ${velocity.magnitude.toFixed(2)}%`,
      `  - ${velocity.interpretation}`,
      `  - Volatility: ${velocity.volatility.toFixed(2)}%`,
      ``,
      `RSI: ${indicators.rsi?.toFixed(1) || 'N/A'}`,
      ``,
      `PINNING: ${pinning.isPinned ? 'YES' : 'NO'}`,
    ];

    if (pinning.pins.length > 0) {
      lines.push(...pinning.pins.map(p => `  - Near ${p.type}: $${p.level.toLocaleString()}`));
    }

    lines.push(``, `Support: $${indicators.levels.support[0]?.toLocaleString() || 'N/A'}`);
    lines.push(`Resistance: $${indicators.levels.resistance[0]?.toLocaleString() || 'N/A'}`);

    return lines.join('\n');
  }
}

export default TechnicalAnalysis;
