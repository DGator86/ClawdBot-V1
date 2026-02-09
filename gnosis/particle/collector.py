"""Unified Data Collection Pipeline.

Collects all data needed for the physics engine:
- Multi-timeframe OHLCV data
- Cross-exchange funding rates
- Order book snapshots
- Liquidation data (when available)
- Macro asset prices (SPX, DXY, etc.)

All collection parameters are exposed as hyperparameters for ML tuning.
"""
import time
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


@dataclass
class CollectorConfig:
    """Hyperparameters for data collection.

    All parameters can be tuned by the improvement loop.
    """
    # Timeframes to collect
    timeframes: List[str] = field(default_factory=lambda: ['1m', '5m', '15m', '1h'])

    # Lookback periods (in bars for each timeframe)
    lookback_1m: int = 1440  # 24 hours
    lookback_5m: int = 576   # 48 hours
    lookback_15m: int = 384  # 4 days
    lookback_1h: int = 336   # 2 weeks

    # Exchanges for funding rate aggregation
    funding_exchanges: List[str] = field(default_factory=lambda: ['binance', 'bybit', 'okx'])

    # Order book settings
    orderbook_depth: int = 100  # Levels to fetch
    orderbook_refresh_seconds: float = 1.0

    # Retry settings
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    timeout_seconds: float = 30.0

    # Cache settings
    cache_ttl_seconds: float = 60.0
    enable_caching: bool = True

    # Rate limiting
    rate_limit_per_second: float = 10.0


@dataclass
class MarketState:
    """Complete market state at a point in time."""
    timestamp: datetime
    symbol: str

    # OHLCV data by timeframe
    ohlcv_1m: Optional[pd.DataFrame] = None
    ohlcv_5m: Optional[pd.DataFrame] = None
    ohlcv_15m: Optional[pd.DataFrame] = None
    ohlcv_1h: Optional[pd.DataFrame] = None

    # Order book
    bids: Optional[List[List[float]]] = None
    asks: Optional[List[List[float]]] = None
    mid_price: float = 0.0

    # Funding rates by exchange
    funding_rates: Dict[str, float] = field(default_factory=dict)

    # Liquidation data
    liquidation_levels: List[Tuple[float, float, float]] = field(default_factory=list)  # (price, long_vol, short_vol)

    # Macro data
    macro_prices: Dict[str, float] = field(default_factory=dict)  # SPX, DXY, etc.
    macro_returns: Dict[str, float] = field(default_factory=dict)

    # Derived
    current_price: float = 0.0
    spread_bps: float = 0.0
    total_bid_volume: float = 0.0
    total_ask_volume: float = 0.0


class DataCollectorBase(ABC):
    """Abstract base class for data collectors."""

    @abstractmethod
    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        limit: int,
    ) -> pd.DataFrame:
        """Fetch OHLCV data."""
        pass

    @abstractmethod
    def fetch_orderbook(
        self,
        symbol: str,
        limit: int,
    ) -> Dict[str, List]:
        """Fetch order book."""
        pass

    @abstractmethod
    def fetch_funding_rate(
        self,
        symbol: str,
    ) -> float:
        """Fetch funding rate."""
        pass


class CCXTDataCollector(DataCollectorBase):
    """Data collector using CCXT library.

    Supports multiple exchanges for funding rate aggregation.
    """

    def __init__(self, config: Optional[CollectorConfig] = None):
        """Initialize collector with config.

        Args:
            config: CollectorConfig with hyperparameters
        """
        self.config = config or CollectorConfig()
        self._exchanges: Dict[str, Any] = {}
        self._cache: Dict[str, Tuple[float, Any]] = {}
        self._last_request_time: float = 0.0

        # Initialize primary exchange (Binance)
        self._init_exchanges()

    def _init_exchanges(self):
        """Initialize CCXT exchange connections."""
        try:
            import ccxt

            # Primary exchange for OHLCV and order book
            self._exchanges['binance'] = ccxt.binance({
                'enableRateLimit': True,
                'options': {'defaultType': 'future'}
            })

            # Additional exchanges for funding rates
            for exchange_id in self.config.funding_exchanges:
                if exchange_id not in self._exchanges:
                    try:
                        exchange_class = getattr(ccxt, exchange_id)
                        self._exchanges[exchange_id] = exchange_class({
                            'enableRateLimit': True,
                            'options': {'defaultType': 'future'}
                        })
                    except Exception as e:
                        logger.warning(f"Failed to init {exchange_id}: {e}")

        except ImportError:
            logger.warning("CCXT not installed. Data collection will be simulated.")

    def _rate_limit(self):
        """Enforce rate limiting."""
        now = time.time()
        min_interval = 1.0 / self.config.rate_limit_per_second
        elapsed = now - self._last_request_time

        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)

        self._last_request_time = time.time()

    def _get_cached(self, key: str) -> Optional[Any]:
        """Get cached value if not expired."""
        if not self.config.enable_caching:
            return None

        if key in self._cache:
            timestamp, value = self._cache[key]
            if time.time() - timestamp < self.config.cache_ttl_seconds:
                return value

        return None

    def _set_cached(self, key: str, value: Any):
        """Set cached value."""
        if self.config.enable_caching:
            self._cache[key] = (time.time(), value)

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        limit: int,
    ) -> pd.DataFrame:
        """Fetch OHLCV data from primary exchange.

        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Timeframe (e.g., '1m', '5m', '1h')
            limit: Number of bars to fetch

        Returns:
            DataFrame with OHLCV columns
        """
        cache_key = f"ohlcv_{symbol}_{timeframe}_{limit}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        self._rate_limit()

        exchange = self._exchanges.get('binance')
        if exchange is None:
            return self._simulate_ohlcv(limit)

        for attempt in range(self.config.max_retries):
            try:
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

                df = pd.DataFrame(
                    ohlcv,
                    columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                )
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df['returns'] = df['close'].pct_change()

                self._set_cached(cache_key, df)
                return df

            except Exception as e:
                logger.warning(f"OHLCV fetch attempt {attempt+1} failed: {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay_seconds * (2 ** attempt))

        return self._simulate_ohlcv(limit)

    def fetch_orderbook(
        self,
        symbol: str,
        limit: int,
    ) -> Dict[str, List]:
        """Fetch order book from primary exchange.

        Args:
            symbol: Trading pair
            limit: Depth levels to fetch

        Returns:
            Dict with 'bids' and 'asks' lists
        """
        cache_key = f"orderbook_{symbol}_{limit}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        self._rate_limit()

        exchange = self._exchanges.get('binance')
        if exchange is None:
            return self._simulate_orderbook(limit)

        for attempt in range(self.config.max_retries):
            try:
                orderbook = exchange.fetch_order_book(symbol, limit=limit)
                result = {
                    'bids': orderbook['bids'],
                    'asks': orderbook['asks'],
                    'timestamp': orderbook.get('timestamp', time.time() * 1000),
                }

                self._set_cached(cache_key, result)
                return result

            except Exception as e:
                logger.warning(f"Order book fetch attempt {attempt+1} failed: {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay_seconds * (2 ** attempt))

        return self._simulate_orderbook(limit)

    def fetch_funding_rate(
        self,
        symbol: str,
        exchange_id: str = 'binance',
    ) -> float:
        """Fetch funding rate from an exchange.

        Args:
            symbol: Trading pair (perpetual)
            exchange_id: Exchange to fetch from

        Returns:
            Funding rate as decimal
        """
        cache_key = f"funding_{exchange_id}_{symbol}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        self._rate_limit()

        exchange = self._exchanges.get(exchange_id)
        if exchange is None:
            return 0.0001  # Default funding

        # Map symbol to perpetual format
        perp_symbol = symbol
        if ':' not in symbol:
            perp_symbol = f"{symbol}:USDT"

        for attempt in range(self.config.max_retries):
            try:
                funding = exchange.fetch_funding_rate(perp_symbol)
                rate = funding.get('fundingRate', 0.0)

                self._set_cached(cache_key, rate)
                return rate

            except Exception as e:
                logger.debug(f"Funding rate fetch failed ({exchange_id}): {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay_seconds * (2 ** attempt))

        return 0.0

    def fetch_all_funding_rates(
        self,
        symbol: str,
    ) -> Dict[str, float]:
        """Fetch funding rates from all configured exchanges.

        Args:
            symbol: Trading pair

        Returns:
            Dict mapping exchange name to funding rate
        """
        rates = {}

        for exchange_id in self.config.funding_exchanges:
            rate = self.fetch_funding_rate(symbol, exchange_id)
            if rate != 0.0:
                rates[exchange_id] = rate

        return rates

    def collect_market_state(
        self,
        symbol: str = 'BTC/USDT',
    ) -> MarketState:
        """Collect complete market state.

        Args:
            symbol: Trading pair

        Returns:
            MarketState with all available data
        """
        now = datetime.now()

        state = MarketState(
            timestamp=now,
            symbol=symbol,
        )

        # Fetch OHLCV for each timeframe
        lookbacks = {
            '1m': self.config.lookback_1m,
            '5m': self.config.lookback_5m,
            '15m': self.config.lookback_15m,
            '1h': self.config.lookback_1h,
        }

        for tf in self.config.timeframes:
            if tf in lookbacks:
                df = self.fetch_ohlcv(symbol, tf, lookbacks[tf])
                setattr(state, f'ohlcv_{tf.replace("m", "m").replace("h", "h")}', df)

                # Set current price from most recent data
                if len(df) > 0 and state.current_price == 0:
                    state.current_price = float(df['close'].iloc[-1])

        # Fetch order book
        orderbook = self.fetch_orderbook(symbol, self.config.orderbook_depth)
        state.bids = orderbook.get('bids', [])
        state.asks = orderbook.get('asks', [])

        if state.bids and state.asks:
            best_bid = state.bids[0][0] if state.bids else 0
            best_ask = state.asks[0][0] if state.asks else 0

            if best_bid > 0 and best_ask > 0:
                state.mid_price = (best_bid + best_ask) / 2
                state.spread_bps = (best_ask - best_bid) / state.mid_price * 10000

            state.total_bid_volume = sum(b[1] for b in state.bids[:20])
            state.total_ask_volume = sum(a[1] for a in state.asks[:20])

        # Fetch funding rates
        state.funding_rates = self.fetch_all_funding_rates(symbol)

        return state

    def _simulate_ohlcv(self, limit: int) -> pd.DataFrame:
        """Generate simulated OHLCV data for testing."""
        now = datetime.now()
        timestamps = [now - timedelta(minutes=i) for i in range(limit)][::-1]

        np.random.seed(42)
        price = 50000.0
        data = []

        for ts in timestamps:
            ret = np.random.normal(0, 0.001)
            price *= (1 + ret)
            high = price * (1 + abs(np.random.normal(0, 0.0005)))
            low = price * (1 - abs(np.random.normal(0, 0.0005)))
            volume = np.random.exponential(100)

            data.append({
                'timestamp': ts,
                'open': price * (1 - ret/2),
                'high': high,
                'low': low,
                'close': price,
                'volume': volume,
            })

        df = pd.DataFrame(data)
        df['returns'] = df['close'].pct_change()
        return df

    def _simulate_orderbook(self, limit: int) -> Dict[str, List]:
        """Generate simulated order book for testing."""
        mid_price = 50000.0

        bids = []
        asks = []

        for i in range(limit):
            bid_price = mid_price * (1 - 0.0001 * (i + 1))
            ask_price = mid_price * (1 + 0.0001 * (i + 1))

            bid_vol = np.random.exponential(10)
            ask_vol = np.random.exponential(10)

            bids.append([bid_price, bid_vol])
            asks.append([ask_price, ask_vol])

        return {
            'bids': bids,
            'asks': asks,
            'timestamp': time.time() * 1000,
        }


class MacroDataCollector:
    """Collector for macro asset data (SPX, DXY, etc.)."""

    def __init__(self, config: Optional[CollectorConfig] = None):
        """Initialize macro collector."""
        self.config = config or CollectorConfig()
        self._cache: Dict[str, Tuple[float, pd.DataFrame]] = {}

    def fetch_macro_prices(
        self,
        symbols: List[str] = None,
        period: str = '7d',
    ) -> Dict[str, pd.DataFrame]:
        """Fetch macro asset prices.

        Args:
            symbols: List of symbols (default: SPX, DXY, GOLD, VIX)
            period: Lookback period

        Returns:
            Dict mapping symbol to price DataFrame
        """
        symbols = symbols or ['SPY', 'UUP', 'GLD', 'VIX']

        results = {}

        try:
            import yfinance as yf

            for symbol in symbols:
                cache_key = f"macro_{symbol}_{period}"

                # Check cache
                if cache_key in self._cache:
                    timestamp, data = self._cache[cache_key]
                    if time.time() - timestamp < 300:  # 5 min cache
                        results[symbol] = data
                        continue

                try:
                    ticker = yf.Ticker(symbol)
                    df = ticker.history(period=period)

                    if len(df) > 0:
                        df = df.reset_index()
                        df['returns'] = df['Close'].pct_change()
                        results[symbol] = df
                        self._cache[cache_key] = (time.time(), df)

                except Exception as e:
                    logger.debug(f"Failed to fetch {symbol}: {e}")

        except ImportError:
            logger.warning("yfinance not installed. Macro data will be simulated.")
            results = self._simulate_macro_data(symbols)

        return results

    def _simulate_macro_data(
        self,
        symbols: List[str],
    ) -> Dict[str, pd.DataFrame]:
        """Generate simulated macro data."""
        results = {}
        now = datetime.now()

        base_prices = {
            'SPY': 500.0,
            'UUP': 28.0,
            'GLD': 200.0,
            'VIX': 18.0,
        }

        for symbol in symbols:
            timestamps = [now - timedelta(hours=i) for i in range(168)][::-1]
            price = base_prices.get(symbol, 100.0)

            data = []
            for ts in timestamps:
                ret = np.random.normal(0, 0.005)
                price *= (1 + ret)
                data.append({
                    'Date': ts,
                    'Close': price,
                    'returns': ret,
                })

            results[symbol] = pd.DataFrame(data)

        return results


def get_collector_hyperparameters() -> List[Dict]:
    """Get hyperparameter definitions for improvement loop."""
    return [
        {
            'name': 'collector_lookback_1m',
            'path': 'particle.collector.lookback_1m',
            'current_value': 1440,
            'candidates': [720, 1440, 2880],
            'variable_type': 'discrete',
        },
        {
            'name': 'collector_lookback_5m',
            'path': 'particle.collector.lookback_5m',
            'current_value': 576,
            'candidates': [288, 576, 1152],
            'variable_type': 'discrete',
        },
        {
            'name': 'collector_orderbook_depth',
            'path': 'particle.collector.orderbook_depth',
            'current_value': 100,
            'candidates': [50, 100, 200],
            'variable_type': 'discrete',
        },
        {
            'name': 'collector_cache_ttl_seconds',
            'path': 'particle.collector.cache_ttl_seconds',
            'current_value': 60.0,
            'candidates': [30.0, 60.0, 120.0],
            'variable_type': 'continuous',
        },
        {
            'name': 'collector_rate_limit_per_second',
            'path': 'particle.collector.rate_limit_per_second',
            'current_value': 10.0,
            'candidates': [5.0, 10.0, 20.0],
            'variable_type': 'continuous',
        },
    ]
