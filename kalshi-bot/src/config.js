// Kalshi Crypto Bot Configuration
// All safety limits are HARDCODED - do not make configurable

export const CONFIG = {
  // === SAFETY LIMITS (HARDCODED) ===
  MAX_TRADE_COST: 20,           // Maximum $20 per trade
  MAX_DRAWDOWN_PERCENT: 20,     // Kill switch at 20% drawdown
  REQUIRE_CONFIRMATION: true,   // Must confirm every trade

  // === ALLOWED MARKETS ===
  ALLOWED_CRYPTOS: ['BTC', 'ETH', 'SOL'],
  MARKET_TYPE: 'hourly',        // Only hourly markets

  // === KALSHI API ===
  KALSHI_API_BASE: 'https://api.elections.kalshi.com/trade-api/v2',
  KALSHI_WS_URL: 'wss://api.elections.kalshi.com/trade-api/ws/v2',

  // === PRICE FEEDS ===
  PRICE_SOURCES: {
    primary: 'coingecko',
    fallback: 'coinbase'
  },

  // === TECHNICAL ANALYSIS ===
  ANALYSIS: {
    EMA_PERIODS: [9, 21, 50],
    RSI_PERIOD: 14,
    RSI_OVERBOUGHT: 70,
    RSI_OVERSOLD: 30,
    MACD_FAST: 12,
    MACD_SLOW: 26,
    MACD_SIGNAL: 9,
    BOLLINGER_PERIOD: 20,
    BOLLINGER_STD: 2,
    ATR_PERIOD: 14,
    LOOKBACK_HOURS: 168,        // 7 days of hourly data
  },

  // === PROPOSAL SETTINGS ===
  PROPOSAL_EXPIRY_MINUTES: 15,  // Proposals expire after 15 min
  MIN_CONFIDENCE: 0.6,          // Minimum 60% confidence to propose

  // === LOCAL API SERVER ===
  API_PORT: 3456,
  API_HOST: '127.0.0.1',
};

export default CONFIG;
