"""Time-of-Day and Seasonal Effects.

Models intraday volatility patterns and directional biases:
- Hourly volatility multipliers (London open, US session, etc.)
- Day-of-week effects
- Monthly/seasonal patterns
- Event-based adjustments (FOMC, CPI, etc.)

All parameters are exposed as hyperparameters for ML tuning.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum


class TradingSession(Enum):
    """Trading session classification."""
    ASIA = "asia"
    EUROPE = "europe"
    US = "us"
    OVERLAP_EU_US = "overlap_eu_us"
    WEEKEND = "weekend"


@dataclass
class TemporalConfig:
    """Hyperparameters for time-of-day effects.

    All parameters can be tuned by the improvement loop.
    """
    # Hourly volatility multipliers (24 hours UTC)
    # These are baseline values that can be learned/adjusted
    vol_hour_0: float = 0.85   # Midnight UTC
    vol_hour_1: float = 0.80
    vol_hour_2: float = 0.80
    vol_hour_3: float = 1.10   # Tokyo open
    vol_hour_4: float = 1.15
    vol_hour_5: float = 1.10
    vol_hour_6: float = 1.00
    vol_hour_7: float = 1.05   # London pre-open
    vol_hour_8: float = 1.20   # London open
    vol_hour_9: float = 1.25
    vol_hour_10: float = 1.20
    vol_hour_11: float = 1.15
    vol_hour_12: float = 1.10
    vol_hour_13: float = 1.25  # US pre-market
    vol_hour_14: float = 1.40  # US open (9:30 ET)
    vol_hour_15: float = 1.35
    vol_hour_16: float = 1.25
    vol_hour_17: float = 1.15
    vol_hour_18: float = 1.10
    vol_hour_19: float = 1.05
    vol_hour_20: float = 1.00  # US close
    vol_hour_21: float = 0.95
    vol_hour_22: float = 0.90
    vol_hour_23: float = 0.85

    # Day-of-week volatility multipliers
    vol_monday: float = 1.10    # Higher vol on Monday (weekend gap)
    vol_tuesday: float = 1.00
    vol_wednesday: float = 1.05  # FOMC days often Wednesday
    vol_thursday: float = 1.00
    vol_friday: float = 1.10    # Position squaring
    vol_saturday: float = 0.70  # Weekend lower vol
    vol_sunday: float = 0.75

    # Session-based adjustments
    asia_vol_mult: float = 0.9
    europe_vol_mult: float = 1.1
    us_vol_mult: float = 1.3
    overlap_vol_mult: float = 1.4  # EU/US overlap highest vol

    # Directional biases (subtle, should be near zero unless evidence)
    # Positive = bullish bias, Negative = bearish bias
    bias_hour_14: float = 0.0001  # Slight bullish at US open (historically)
    bias_hour_20: float = -0.0001  # Slight bearish at US close
    bias_monday: float = 0.0      # No day bias by default
    bias_friday: float = -0.00005  # Slight risk-off Friday

    # Volatility clustering adjustment
    recent_vol_weight: float = 0.3  # Weight for recent realized vol
    historical_pattern_weight: float = 0.7  # Weight for historical pattern

    # Event calendar effects (multipliers for known event days)
    fomc_vol_mult: float = 1.5
    cpi_vol_mult: float = 1.3
    nfp_vol_mult: float = 1.2  # Non-farm payrolls
    quadruple_witching_vol_mult: float = 1.4

    # Month-end effects
    month_end_days: int = 3  # Last N days of month
    month_end_vol_mult: float = 1.15

    # Force scaling
    temporal_force_strength: float = 1.0
    max_vol_multiplier: float = 2.0
    min_vol_multiplier: float = 0.5


class TimeOfDayEffects:
    """Models intraday and calendar effects on volatility and direction.

    Bitcoin shows distinct patterns:
    - Higher volatility during US trading hours
    - Lower volatility on weekends
    - Volatility spikes at key session opens
    - Subtle directional biases at certain times

    These patterns can be exploited for better calibrated predictions.
    """

    def __init__(self, config: Optional[TemporalConfig] = None):
        """Initialize with config.

        Args:
            config: TemporalConfig with hyperparameters
        """
        self.config = config or TemporalConfig()
        self._event_calendar: Dict[str, str] = {}  # date -> event type
        self._historical_vol: Dict[int, float] = {}  # hour -> historical vol

    def get_hourly_volatility_multiplier(self, hour: int) -> float:
        """Get volatility multiplier for a specific hour.

        Args:
            hour: Hour of day (0-23 UTC)

        Returns:
            Volatility multiplier
        """
        attr_name = f'vol_hour_{hour}'
        return getattr(self.config, attr_name, 1.0)

    def get_day_of_week_multiplier(self, day: int) -> float:
        """Get volatility multiplier for day of week.

        Args:
            day: Day of week (0=Monday, 6=Sunday)

        Returns:
            Volatility multiplier
        """
        day_map = {
            0: self.config.vol_monday,
            1: self.config.vol_tuesday,
            2: self.config.vol_wednesday,
            3: self.config.vol_thursday,
            4: self.config.vol_friday,
            5: self.config.vol_saturday,
            6: self.config.vol_sunday,
        }
        return day_map.get(day, 1.0)

    def get_session(self, hour: int, day: int) -> TradingSession:
        """Determine current trading session.

        Args:
            hour: Hour of day (UTC)
            day: Day of week (0=Monday)

        Returns:
            TradingSession enum
        """
        if day >= 5:  # Saturday or Sunday
            return TradingSession.WEEKEND

        # Session times (approximate, UTC)
        # Asia: 0-8 UTC (Tokyo 9-17 JST)
        # Europe: 7-16 UTC (London 8-17 BST)
        # US: 13-21 UTC (NY 9-17 ET)

        if 13 <= hour < 16:  # EU/US overlap
            return TradingSession.OVERLAP_EU_US
        elif 13 <= hour < 21:  # US session
            return TradingSession.US
        elif 7 <= hour < 16:  # Europe session
            return TradingSession.EUROPE
        else:  # Asia/off-hours
            return TradingSession.ASIA

    def get_session_multiplier(self, session: TradingSession) -> float:
        """Get volatility multiplier for trading session.

        Args:
            session: TradingSession enum

        Returns:
            Volatility multiplier
        """
        session_map = {
            TradingSession.ASIA: self.config.asia_vol_mult,
            TradingSession.EUROPE: self.config.europe_vol_mult,
            TradingSession.US: self.config.us_vol_mult,
            TradingSession.OVERLAP_EU_US: self.config.overlap_vol_mult,
            TradingSession.WEEKEND: 0.7,
        }
        return session_map.get(session, 1.0)

    def get_directional_bias(self, hour: int, day: int) -> float:
        """Get subtle directional bias for time.

        Args:
            hour: Hour of day (UTC)
            day: Day of week

        Returns:
            Directional bias (positive = bullish)
        """
        # Hour-based bias
        hour_bias = 0.0
        if hour == 14:
            hour_bias = self.config.bias_hour_14
        elif hour == 20:
            hour_bias = self.config.bias_hour_20

        # Day-based bias
        day_bias = 0.0
        if day == 0:  # Monday
            day_bias = self.config.bias_monday
        elif day == 4:  # Friday
            day_bias = self.config.bias_friday

        return hour_bias + day_bias

    def add_event(self, date: str, event_type: str):
        """Add an event to the calendar.

        Args:
            date: Date string (YYYY-MM-DD)
            event_type: Event type (fomc, cpi, nfp, quadruple_witching)
        """
        self._event_calendar[date] = event_type

    def get_event_multiplier(self, current_time: datetime) -> float:
        """Get volatility multiplier for calendar events.

        Args:
            current_time: Current datetime

        Returns:
            Event volatility multiplier
        """
        date_str = current_time.strftime('%Y-%m-%d')

        if date_str in self._event_calendar:
            event = self._event_calendar[date_str]
            event_map = {
                'fomc': self.config.fomc_vol_mult,
                'cpi': self.config.cpi_vol_mult,
                'nfp': self.config.nfp_vol_mult,
                'quadruple_witching': self.config.quadruple_witching_vol_mult,
            }
            return event_map.get(event, 1.0)

        # Check for month-end
        next_month = (current_time.replace(day=1) + timedelta(days=32)).replace(day=1)
        days_to_month_end = (next_month - current_time).days

        if days_to_month_end <= self.config.month_end_days:
            return self.config.month_end_vol_mult

        return 1.0

    def calculate_combined_multiplier(
        self,
        current_time: datetime,
        recent_realized_vol: Optional[float] = None,
    ) -> Dict[str, float]:
        """Calculate combined volatility multiplier.

        Args:
            current_time: Current datetime
            recent_realized_vol: Recent realized volatility (optional)

        Returns:
            Dict with multiplier components
        """
        hour = current_time.hour
        day = current_time.weekday()

        # Component multipliers
        hourly_mult = self.get_hourly_volatility_multiplier(hour)
        day_mult = self.get_day_of_week_multiplier(day)
        session = self.get_session(hour, day)
        session_mult = self.get_session_multiplier(session)
        event_mult = self.get_event_multiplier(current_time)

        # Combine multipliers (geometric mean to avoid extreme values)
        combined = (hourly_mult * day_mult * session_mult * event_mult) ** 0.25
        combined *= 1.5  # Scale back up after geometric mean

        # Blend with recent realized vol if available
        if recent_realized_vol is not None:
            # Adjust based on whether recent vol is above/below expectation
            expected_vol = combined * 0.02  # Assume 2% base hourly vol
            vol_ratio = recent_realized_vol / (expected_vol + 1e-9)
            vol_adjustment = 1 + (vol_ratio - 1) * self.config.recent_vol_weight
            combined *= vol_adjustment

        # Clamp to reasonable range
        combined = np.clip(
            combined,
            self.config.min_vol_multiplier,
            self.config.max_vol_multiplier
        )

        # Get directional bias
        bias = self.get_directional_bias(hour, day)

        return {
            'combined_multiplier': combined,
            'hourly_mult': hourly_mult,
            'day_mult': day_mult,
            'session_mult': session_mult,
            'event_mult': event_mult,
            'session': session.value,
            'directional_bias': bias,
        }

    def get_temporal_features(
        self,
        current_time: datetime,
        recent_realized_vol: Optional[float] = None,
    ) -> Dict[str, float]:
        """Get time-related features for ML model."""
        result = self.calculate_combined_multiplier(current_time, recent_realized_vol)

        hour = current_time.hour
        day = current_time.weekday()
        session = self.get_session(hour, day)

        features = {
            'temporal_vol_mult': result['combined_multiplier'],
            'temporal_hourly_mult': result['hourly_mult'],
            'temporal_day_mult': result['day_mult'],
            'temporal_session_mult': result['session_mult'],
            'temporal_event_mult': result['event_mult'],
            'temporal_bias': result['directional_bias'],

            # One-hot encoded session
            'temporal_session_asia': 1.0 if session == TradingSession.ASIA else 0.0,
            'temporal_session_europe': 1.0 if session == TradingSession.EUROPE else 0.0,
            'temporal_session_us': 1.0 if session == TradingSession.US else 0.0,
            'temporal_session_overlap': 1.0 if session == TradingSession.OVERLAP_EU_US else 0.0,
            'temporal_session_weekend': 1.0 if session == TradingSession.WEEKEND else 0.0,

            # Cyclical encoding of hour (sin/cos for continuity)
            'temporal_hour_sin': np.sin(2 * np.pi * hour / 24),
            'temporal_hour_cos': np.cos(2 * np.pi * hour / 24),

            # Cyclical encoding of day
            'temporal_day_sin': np.sin(2 * np.pi * day / 7),
            'temporal_day_cos': np.cos(2 * np.pi * day / 7),

            # Minutes to key events
            'temporal_minutes_to_us_open': self._minutes_to_hour(current_time, 14),
            'temporal_minutes_to_us_close': self._minutes_to_hour(current_time, 21),
            'temporal_minutes_to_london_open': self._minutes_to_hour(current_time, 8),
        }

        return features

    def _minutes_to_hour(self, current_time: datetime, target_hour: int) -> float:
        """Calculate minutes until target hour (0-720 range)."""
        current_minutes = current_time.hour * 60 + current_time.minute
        target_minutes = target_hour * 60

        diff = target_minutes - current_minutes
        if diff < 0:
            diff += 24 * 60  # Next day

        return min(diff, 720)  # Cap at 12 hours

    def update_historical_volatility(self, hourly_vol_data: Dict[int, float]):
        """Update historical volatility patterns.

        Args:
            hourly_vol_data: Dict mapping hour -> average volatility
        """
        self._historical_vol = hourly_vol_data

    def calibrate_from_data(
        self,
        df: pd.DataFrame,
        price_col: str = 'close',
        timestamp_col: str = 'timestamp',
    ):
        """Calibrate hourly volatility patterns from historical data.

        Args:
            df: DataFrame with price data
            price_col: Name of price column
            timestamp_col: Name of timestamp column
        """
        if len(df) < 168:  # Need at least a week of hourly data
            return

        df = df.copy()
        df['returns'] = df[price_col].pct_change()
        df['hour'] = pd.to_datetime(df[timestamp_col]).dt.hour

        # Calculate average volatility by hour
        hourly_vol = df.groupby('hour')['returns'].std()

        if len(hourly_vol) == 24:
            # Normalize to have mean 1.0
            mean_vol = hourly_vol.mean()
            if mean_vol > 0:
                normalized = hourly_vol / mean_vol

                # Update config
                for hour in range(24):
                    if hour in normalized.index:
                        setattr(self.config, f'vol_hour_{hour}', float(normalized[hour]))


def get_temporal_hyperparameters() -> List[Dict]:
    """Get hyperparameter definitions for improvement loop."""
    hyperparams = []

    # Key hourly multipliers (not all 24, just important ones)
    key_hours = [8, 14, 15, 20, 21]  # London open, US open, US close
    for hour in key_hours:
        hyperparams.append({
            'name': f'temporal_vol_hour_{hour}',
            'path': f'particle.temporal.vol_hour_{hour}',
            'current_value': getattr(TemporalConfig(), f'vol_hour_{hour}'),
            'candidates': [0.8, 1.0, 1.2, 1.4, 1.6],
            'variable_type': 'continuous',
        })

    # Session multipliers
    hyperparams.extend([
        {
            'name': 'temporal_asia_vol_mult',
            'path': 'particle.temporal.asia_vol_mult',
            'current_value': 0.9,
            'candidates': [0.7, 0.8, 0.9, 1.0, 1.1],
            'variable_type': 'continuous',
        },
        {
            'name': 'temporal_us_vol_mult',
            'path': 'particle.temporal.us_vol_mult',
            'current_value': 1.3,
            'candidates': [1.0, 1.2, 1.3, 1.5, 1.7],
            'variable_type': 'continuous',
        },
        {
            'name': 'temporal_overlap_vol_mult',
            'path': 'particle.temporal.overlap_vol_mult',
            'current_value': 1.4,
            'candidates': [1.2, 1.4, 1.6, 1.8],
            'variable_type': 'continuous',
        },
    ])

    # Day multipliers
    hyperparams.extend([
        {
            'name': 'temporal_vol_monday',
            'path': 'particle.temporal.vol_monday',
            'current_value': 1.1,
            'candidates': [0.9, 1.0, 1.1, 1.2],
            'variable_type': 'continuous',
        },
        {
            'name': 'temporal_vol_friday',
            'path': 'particle.temporal.vol_friday',
            'current_value': 1.1,
            'candidates': [0.9, 1.0, 1.1, 1.2],
            'variable_type': 'continuous',
        },
        {
            'name': 'temporal_vol_weekend',
            'path': 'particle.temporal.vol_saturday',
            'current_value': 0.7,
            'candidates': [0.5, 0.6, 0.7, 0.8],
            'variable_type': 'continuous',
        },
    ])

    # Event multipliers
    hyperparams.extend([
        {
            'name': 'temporal_fomc_vol_mult',
            'path': 'particle.temporal.fomc_vol_mult',
            'current_value': 1.5,
            'candidates': [1.2, 1.4, 1.5, 1.7, 2.0],
            'variable_type': 'continuous',
        },
        {
            'name': 'temporal_recent_vol_weight',
            'path': 'particle.temporal.recent_vol_weight',
            'current_value': 0.3,
            'candidates': [0.1, 0.2, 0.3, 0.4, 0.5],
            'variable_type': 'continuous',
        },
    ])

    return hyperparams
