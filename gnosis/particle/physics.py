"""Price Particle Physics Framework.

Models price as a particle moving through a potential field with:
- Position: Current price level relative to key references
- Velocity: Rate of price change (momentum)
- Acceleration: Change in momentum (second derivative)
- Force: Order flow imbalance, buying/selling pressure
- Mass: Inverse of volatility (higher vol = easier to move)
- Potential Energy: Distance from equilibrium levels
- Kinetic Energy: Momentum strength
- Friction: Mean reversion damping

The physics model provides features that capture market microstructure
dynamics beyond simple technical indicators.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


class PriceParticle:
    """Full physics-based price particle model.

    Treats price as a particle in a force field where:
    - Order flow creates forces (F = ma)
    - Support/resistance create potential wells
    - Volatility determines mass (inertia)
    - Mean reversion acts as friction/damping
    """

    def __init__(self, config: Optional[Dict] = None):
        """Initialize particle physics model.

        Args:
            config: Configuration dict with optional parameters:
                - velocity_span: EWM span for velocity smoothing (default: 5)
                - acceleration_span: EWM span for acceleration (default: 3)
                - mass_lookback: Bars for mass calculation (default: 20)
                - potential_lookback: Bars for potential field (default: 50)
                - friction_coefficient: Mean reversion strength (default: 0.1)
                - volume_profile_bins: Number of price bins for volume profile (default: 20)
        """
        config = config or {}
        self.velocity_span = config.get("velocity_span", 5)
        self.acceleration_span = config.get("acceleration_span", 3)
        self.mass_lookback = config.get("mass_lookback", 20)
        self.potential_lookback = config.get("potential_lookback", 50)
        self.friction_coefficient = config.get("friction_coefficient", 0.1)
        self.volume_profile_bins = config.get("volume_profile_bins", 20)
        self.energy_lookback = config.get("energy_lookback", 10)
        self.bb_lookback = config.get("bb_lookback", 20)
        self.bb_std_mult = config.get("bb_std_mult", 2.0)
        self.rsi_period = config.get("rsi_period", 14)
        self.ichimoku_tenkan = config.get("ichimoku_tenkan", 9)
        self.ichimoku_kijun = config.get("ichimoku_kijun", 26)
        self.ichimoku_senkou = config.get("ichimoku_senkou", 52)
        self.avwap_lookback = config.get("avwap_lookback", 200)
        self.avwap_impulse_std = config.get("avwap_impulse_std", 2.0)
        self.avwap_volume_std = config.get("avwap_volume_std", 2.0)

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all particle physics features.

        Args:
            df: DataFrame with OHLCV data and basic features
                Required: close, high, low, volume, returns, ofi

        Returns:
            DataFrame with additional particle physics features
        """
        result = df.copy()

        # Core kinematics
        result = self._compute_velocity(result)
        result = self._compute_acceleration(result)
        result = self._compute_jerk(result)

        # Mass and inertia
        result = self._compute_mass(result)

        # Forces
        result = self._compute_forces(result)

        # Energy
        result = self._compute_kinetic_energy(result)
        result = self._compute_potential_energy(result)
        result = self._compute_total_energy(result)

        # Potential field (support/resistance)
        result = self._compute_potential_field(result)

        # Friction and damping
        result = self._compute_friction(result)

        # Volume profile features
        result = self._compute_volume_profile(result)

        # Anchored VWAP wells (crypto anchor points)
        result = self._compute_anchored_vwap(result)

        # Bollinger geometry (volatility curvature)
        result = self._compute_bollinger_geometry(result)

        # RSI throttle (impulse efficiency)
        result = self._compute_rsi_throttle(result)

        # Ichimoku regime topology
        result = self._compute_ichimoku_regime(result)

        # Crypto-specific forces (funding, OI, liquidations, CVD)
        result = self._compute_crypto_forces(result)

        # Composite particle state
        result = self._compute_particle_state(result)

        return result

    def _compute_velocity(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute price velocity (rate of change).

        v = dx/dt = returns (smoothed)
        """
        # Raw velocity is just returns
        df["velocity_raw"] = df["returns"]

        # Smoothed velocity using EWM
        df["velocity"] = df.groupby("symbol")["returns"].transform(
            lambda x: x.ewm(span=self.velocity_span, adjust=False).mean()
        )

        # Velocity magnitude (absolute)
        df["velocity_abs"] = df["velocity"].abs()

        # Velocity direction (-1, 0, +1)
        df["velocity_direction"] = np.sign(df["velocity"])

        return df

    def _compute_acceleration(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute price acceleration (change in velocity).

        a = dv/dt = d²x/dt²
        """
        # Raw acceleration is change in velocity
        df["acceleration_raw"] = df.groupby("symbol")["velocity"].diff()

        # Smoothed acceleration
        df["acceleration"] = df.groupby("symbol")["acceleration_raw"].transform(
            lambda x: x.ewm(span=self.acceleration_span, adjust=False).mean()
        )

        # Acceleration magnitude
        df["acceleration_abs"] = df["acceleration"].abs()

        # Acceleration direction
        df["acceleration_direction"] = np.sign(df["acceleration"])

        # Velocity-acceleration alignment (same direction = momentum building)
        df["momentum_alignment"] = df["velocity_direction"] * df["acceleration_direction"]

        return df

    def _compute_jerk(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute jerk (rate of change of acceleration).

        j = da/dt = d³x/dt³

        High jerk indicates sudden changes in market dynamics.
        """
        df["jerk"] = df.groupby("symbol")["acceleration"].diff()
        df["jerk_abs"] = df["jerk"].abs()

        return df

    def _compute_mass(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute effective mass (inverse volatility).

        Higher volatility = lower mass = easier to move
        Lower volatility = higher mass = more inertia

        m = 1 / σ (normalized)
        """
        # Realized volatility as inverse mass proxy
        vol = df.groupby("symbol")["returns"].transform(
            lambda x: x.rolling(self.mass_lookback, min_periods=5).std()
        )

        # Mass is inverse of volatility (normalized to avoid extremes)
        df["mass"] = 1.0 / (vol + 1e-6)

        # Normalize mass to reasonable range [0.1, 10]
        df["mass"] = df.groupby("symbol")["mass"].transform(
            lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9) * 9.9 + 0.1
        )

        # Mass change (indicates volatility regime shifts)
        df["mass_change"] = df.groupby("symbol")["mass"].pct_change()

        return df

    def _compute_forces(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute forces acting on the price particle.

        F = m * a (Newton's second law)

        Forces come from:
        - Order flow imbalance (buy/sell pressure)
        - Volume (intensity of trading)
        """
        # Net force from order flow: F = OFI * volume
        if "ofi" in df.columns and "volume" in df.columns:
            # OFI-based force (buy/sell pressure)
            df["force_ofi"] = df["ofi"] * df["volume"]

            # Normalize force
            df["force_ofi"] = df.groupby("symbol")["force_ofi"].transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-9)
            )
        else:
            df["force_ofi"] = 0.0

        # Compute force from F = ma
        df["force_newton"] = df["mass"] * df["acceleration"]

        # Net force (combining OFI and Newton)
        df["force_net"] = df["force_ofi"] * 0.5 + df["force_newton"] * 0.5

        # Force impulse (accumulated force over short window)
        df["force_impulse"] = df.groupby("symbol")["force_net"].transform(
            lambda x: x.rolling(5, min_periods=1).sum()
        )

        return df

    def _compute_kinetic_energy(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute kinetic energy.

        KE = (1/2) * m * v²

        High kinetic energy = strong momentum
        Low kinetic energy = price near rest
        """
        df["kinetic_energy"] = 0.5 * df["mass"] * (df["velocity"] ** 2)

        # Kinetic energy change
        df["kinetic_energy_change"] = df.groupby("symbol")["kinetic_energy"].pct_change()

        # Kinetic energy momentum (is KE building or dissipating?)
        df["kinetic_energy_momentum"] = df.groupby("symbol")["kinetic_energy"].transform(
            lambda x: x.ewm(span=self.energy_lookback, adjust=False).mean()
        ) - df["kinetic_energy"]

        return df

    def _compute_potential_energy(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute potential energy (distance from equilibrium).

        PE = k * (x - x_eq)²

        Uses multiple equilibrium references:
        - VWAP (volume-weighted average price)
        - SMA (simple moving average)
        - Recent range midpoint
        """
        # Equilibrium 1: SMA
        df["sma_eq"] = df.groupby("symbol")["close"].transform(
            lambda x: x.rolling(self.potential_lookback, min_periods=10).mean()
        )

        # Equilibrium 2: EMA (faster equilibrium)
        df["ema_eq"] = df.groupby("symbol")["close"].transform(
            lambda x: x.ewm(span=self.potential_lookback // 2, adjust=False).mean()
        )

        # Equilibrium 3: Range midpoint
        high_roll = df.groupby("symbol")["high"].transform(
            lambda x: x.rolling(self.potential_lookback, min_periods=10).max()
        )
        low_roll = df.groupby("symbol")["low"].transform(
            lambda x: x.rolling(self.potential_lookback, min_periods=10).min()
        )
        df["range_eq"] = (high_roll + low_roll) / 2

        # Displacement from each equilibrium (normalized by price)
        df["displacement_sma"] = (df["close"] - df["sma_eq"]) / df["close"]
        df["displacement_ema"] = (df["close"] - df["ema_eq"]) / df["close"]
        df["displacement_range"] = (df["close"] - df["range_eq"]) / df["close"]

        # Potential energy (spring model: PE = 0.5 * k * x²)
        k = 100  # Spring constant
        df["potential_energy_sma"] = 0.5 * k * (df["displacement_sma"] ** 2)
        df["potential_energy_ema"] = 0.5 * k * (df["displacement_ema"] ** 2)
        df["potential_energy_range"] = 0.5 * k * (df["displacement_range"] ** 2)

        # Combined potential energy
        df["potential_energy"] = (
            df["potential_energy_sma"] * 0.4 +
            df["potential_energy_ema"] * 0.3 +
            df["potential_energy_range"] * 0.3
        )

        return df

    def _compute_total_energy(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute total mechanical energy.

        E = KE + PE

        Conservation of energy: if E is constant, market is in equilibrium
        Energy injection/dissipation indicates external forces
        """
        df["total_energy"] = df["kinetic_energy"] + df["potential_energy"]

        # Energy conservation violation (indicates external forces)
        df["energy_change"] = df.groupby("symbol")["total_energy"].diff()

        # Energy injection rate
        df["energy_injection"] = df.groupby("symbol")["energy_change"].transform(
            lambda x: x.ewm(span=5, adjust=False).mean()
        )

        # Energy state: >0 means energy being added, <0 means dissipating
        df["energy_state"] = np.sign(df["energy_injection"])

        return df

    def _compute_potential_field(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute potential field from support/resistance levels.

        Creates a force field where:
        - Support levels create upward force (repel from below)
        - Resistance levels create downward force (repel from above)
        """
        # Support level (recent lows)
        df["support_level"] = df.groupby("symbol")["low"].transform(
            lambda x: x.rolling(self.potential_lookback, min_periods=10).min()
        )

        # Resistance level (recent highs)
        df["resistance_level"] = df.groupby("symbol")["high"].transform(
            lambda x: x.rolling(self.potential_lookback, min_periods=10).max()
        )

        # Distance to support/resistance (normalized)
        total_range = df["resistance_level"] - df["support_level"] + 1e-9
        df["distance_to_support"] = (df["close"] - df["support_level"]) / total_range
        df["distance_to_resistance"] = (df["resistance_level"] - df["close"]) / total_range

        # Potential field gradient (force direction)
        # Near support: positive (upward force), near resistance: negative
        df["field_gradient"] = df["distance_to_resistance"] - df["distance_to_support"]

        # Field strength (stronger near boundaries)
        df["field_strength"] = 1.0 / (
            df["distance_to_support"] * df["distance_to_resistance"] + 0.01
        )
        df["field_strength"] = df["field_strength"].clip(upper=100)  # Cap extreme values

        return df

    def _compute_friction(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute friction/damping (mean reversion tendency).

        Friction opposes motion and causes mean reversion.
        f = -μ * v (proportional to velocity, opposite direction)
        """
        # Friction force
        df["friction"] = -self.friction_coefficient * df["velocity"]

        # Damping ratio (how quickly momentum decays)
        velocity_decay = df.groupby("symbol")["velocity_abs"].transform(
            lambda x: x.ewm(span=10, adjust=False).mean() / (x.shift(5).ewm(span=10, adjust=False).mean() + 1e-9)
        )
        df["damping_ratio"] = 1.0 - velocity_decay.clip(lower=0, upper=1)

        # Mean reversion strength (based on recent behavior)
        df["mean_reversion_strength"] = df.groupby("symbol")["returns"].transform(
            lambda x: -x.rolling(20, min_periods=5).apply(
                lambda y: np.corrcoef(y[:-1], y[1:])[0, 1] if len(y) > 1 else 0,
                raw=True
            )
        ).fillna(0)

        return df

    def _compute_volume_profile(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute volume profile features.

        Volume profile shows where most trading occurred, creating:
        - High volume nodes (HVN): Price levels with high activity (support/resistance)
        - Low volume nodes (LVN): Price levels with low activity (fast moves through)
        """
        # Volume-weighted average price (VWAP)
        df["_price_vol"] = df["close"] * df["volume"]
        df["_price_vol_sum"] = df.groupby("symbol")["_price_vol"].transform(
            lambda x: x.rolling(self.potential_lookback, min_periods=10).sum()
        )
        df["_vol_sum"] = df.groupby("symbol")["volume"].transform(
            lambda x: x.rolling(self.potential_lookback, min_periods=10).sum()
        )
        df["vwap"] = df["_price_vol_sum"] / (df["_vol_sum"] + 1e-9)
        df.drop(columns=["_price_vol", "_price_vol_sum", "_vol_sum"], inplace=True)

        # Distance from VWAP (normalized)
        df["vwap_distance"] = (df["close"] - df["vwap"]) / df["close"]

        # VWAP deviation (how far price typically deviates)
        df["vwap_std"] = df.groupby("symbol")["vwap_distance"].transform(
            lambda x: x.rolling(self.potential_lookback, min_periods=10).std()
        )

        # Z-score relative to VWAP
        df["vwap_zscore"] = df["vwap_distance"] / (df["vwap_std"] + 1e-9)

        # Volume momentum (is volume building?)
        df["volume_momentum"] = df.groupby("symbol")["volume"].transform(
            lambda x: x.ewm(span=5, adjust=False).mean() /
                     x.ewm(span=20, adjust=False).mean()
        ) - 1.0

        return df

    def _compute_anchored_vwap(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute anchored VWAP wells from impulse/volume/weekly anchors."""
        if "close" not in df.columns or "volume" not in df.columns:
            df["avwap_impulse"] = np.nan
            df["avwap_volume"] = np.nan
            df["avwap_weekly"] = np.nan
            df["avwap_impulse_distance"] = np.nan
            df["avwap_volume_distance"] = np.nan
            df["avwap_weekly_distance"] = np.nan
            return df

        def anchored_vwap(group: pd.DataFrame, anchor_mask: pd.Series) -> pd.Series:
            price = group["close"]
            volume = group["volume"].fillna(0.0)
            pv = price * volume
            cum_pv = pv.cumsum()
            cum_vol = volume.cumsum()
            anchor_mask = anchor_mask.reindex(group.index).fillna(False)
            anchor_pv = cum_pv.where(anchor_mask).ffill().fillna(0.0)
            anchor_vol = cum_vol.where(anchor_mask).ffill().fillna(0.0)
            anchored = (cum_pv - anchor_pv) / (cum_vol - anchor_vol + 1e-9)
            anchored = anchored.where(anchor_mask.cumsum() > 0, cum_pv / (cum_vol + 1e-9))
            return anchored

        avwap_impulse = []
        avwap_volume = []
        avwap_weekly = []

        for _, group in df.groupby("symbol", sort=False):
            returns = group["returns"].fillna(0.0)
            vol = group["volume"].fillna(0.0)

            impulse_std = returns.rolling(self.avwap_lookback, min_periods=5).std()
            impulse_anchor = returns.abs() > (impulse_std * self.avwap_impulse_std)

            volume_mean = vol.rolling(self.avwap_lookback, min_periods=5).mean()
            volume_std = vol.rolling(self.avwap_lookback, min_periods=5).std()
            volume_anchor = vol > (volume_mean + volume_std * self.avwap_volume_std)

            if "timestamp" in group.columns:
                timestamps = pd.to_datetime(group["timestamp"], utc=True, errors="coerce")
                week = timestamps.dt.to_period("W")
                weekly_anchor = week.ne(week.shift())
            else:
                weekly_anchor = pd.Series(False, index=group.index)

            avwap_impulse.append(anchored_vwap(group, impulse_anchor))
            avwap_volume.append(anchored_vwap(group, volume_anchor))
            avwap_weekly.append(anchored_vwap(group, weekly_anchor))

        df["avwap_impulse"] = pd.concat(avwap_impulse).sort_index()
        df["avwap_volume"] = pd.concat(avwap_volume).sort_index()
        df["avwap_weekly"] = pd.concat(avwap_weekly).sort_index()

        df["avwap_impulse_distance"] = (df["close"] - df["avwap_impulse"]) / df["close"]
        df["avwap_volume_distance"] = (df["close"] - df["avwap_volume"]) / df["close"]
        df["avwap_weekly_distance"] = (df["close"] - df["avwap_weekly"]) / df["close"]

        return df

    def _compute_bollinger_geometry(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute Bollinger Band geometry and squeeze state."""
        if "close" not in df.columns:
            df["bb_mid"] = np.nan
            df["bb_upper"] = np.nan
            df["bb_lower"] = np.nan
            df["bb_width"] = np.nan
            df["bb_squeeze"] = np.nan
            return df

        mid = df.groupby("symbol")["close"].transform(
            lambda x: x.rolling(self.bb_lookback, min_periods=5).mean()
        )
        std = df.groupby("symbol")["close"].transform(
            lambda x: x.rolling(self.bb_lookback, min_periods=5).std()
        )
        df["bb_mid"] = mid
        df["bb_upper"] = mid + self.bb_std_mult * std
        df["bb_lower"] = mid - self.bb_std_mult * std
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / (mid + 1e-9)

        df["bb_squeeze"] = df.groupby("symbol")["bb_width"].transform(
            lambda x: x / (x.rolling(self.bb_lookback, min_periods=5).mean() + 1e-9)
        )

        return df

    def _compute_rsi_throttle(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute RSI and an impulse efficiency throttle."""
        if "returns" not in df.columns:
            df["rsi"] = np.nan
            df["rsi_throttle"] = np.nan
            return df

        gains = df["returns"].clip(lower=0)
        losses = (-df["returns"]).clip(lower=0)

        avg_gain = gains.ewm(span=self.rsi_period, adjust=False).mean()
        avg_loss = losses.ewm(span=self.rsi_period, adjust=False).mean()

        rs = avg_gain / (avg_loss + 1e-9)
        rsi_raw = 100 - (100 / (1 + rs))
        # Ensure Series (groupby can return DataFrame with single group)
        if isinstance(rsi_raw, pd.DataFrame):
            rsi_raw = rsi_raw.iloc[:, 0]
        df["rsi"] = rsi_raw.values if hasattr(rsi_raw, 'values') else rsi_raw

        rsi_distance = (df["rsi"] - 50).abs() / 50
        df["rsi_throttle"] = (1.0 - rsi_distance).clip(lower=0.0)

        return df

    def _compute_ichimoku_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute Ichimoku cloud regime without forward shifts."""
        if not {"high", "low", "close"}.issubset(df.columns):
            df["ichimoku_tenkan"] = np.nan
            df["ichimoku_kijun"] = np.nan
            df["ichimoku_cloud_top"] = np.nan
            df["ichimoku_cloud_bottom"] = np.nan
            df["ichimoku_regime"] = np.nan
            df["ichimoku_cloud_thickness"] = np.nan
            return df

        high = df.groupby("symbol")["high"]
        low = df.groupby("symbol")["low"]

        tenkan = (high.transform(
            lambda x: x.rolling(self.ichimoku_tenkan, min_periods=5).max()
        ) + low.transform(
            lambda x: x.rolling(self.ichimoku_tenkan, min_periods=5).min()
        )) / 2
        kijun = (high.transform(
            lambda x: x.rolling(self.ichimoku_kijun, min_periods=5).max()
        ) + low.transform(
            lambda x: x.rolling(self.ichimoku_kijun, min_periods=5).min()
        )) / 2
        senkou_b = (high.transform(
            lambda x: x.rolling(self.ichimoku_senkou, min_periods=5).max()
        ) + low.transform(
            lambda x: x.rolling(self.ichimoku_senkou, min_periods=5).min()
        )) / 2
        senkou_a = (tenkan + kijun) / 2

        cloud_top = np.maximum(senkou_a, senkou_b)
        cloud_bottom = np.minimum(senkou_a, senkou_b)

        df["ichimoku_tenkan"] = tenkan
        df["ichimoku_kijun"] = kijun
        df["ichimoku_cloud_top"] = cloud_top
        df["ichimoku_cloud_bottom"] = cloud_bottom
        df["ichimoku_cloud_thickness"] = (cloud_top - cloud_bottom) / (df["close"] + 1e-9)

        regime = np.where(
            df["close"] > cloud_top,
            1.0,
            np.where(df["close"] < cloud_bottom, -1.0, 0.0)
        )
        df["ichimoku_regime"] = regime

        return df

    def _compute_crypto_forces(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute crypto-specific force proxies (funding, OI, liquidations, CVD)."""
        if "funding_rate" in df.columns:
            df["funding_force"] = df["funding_rate"]
        else:
            df["funding_force"] = 0.0

        if "open_interest" in df.columns:
            df["oi_change"] = df.groupby("symbol")["open_interest"].pct_change()
        elif "open_interest_change" in df.columns:
            df["oi_change"] = df["open_interest_change"]
        else:
            df["oi_change"] = 0.0

        df["oi_pressure"] = df.groupby("symbol")["oi_change"].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-9)
        )

        liquidation_cols = [
            col for col in ["liquidation_long", "liquidation_short", "liquidation_volume"]
            if col in df.columns
        ]
        if liquidation_cols:
            df["liquidation_density"] = df[liquidation_cols].sum(axis=1)
        else:
            df["liquidation_density"] = 0.0

        if "cvd" in df.columns:
            df["cvd"] = df["cvd"]
        elif "ofi" in df.columns and "volume" in df.columns:
            df["cvd"] = (df["ofi"] * df["volume"]).groupby(df["symbol"]).cumsum()
        else:
            df["cvd"] = 0.0

        df["cvd_momentum"] = df.groupby("symbol")["cvd"].transform(
            lambda x: x.ewm(span=5, adjust=False).mean() /
                     (x.ewm(span=20, adjust=False).mean() + 1e-9) - 1.0
        )

        df["jump_hazard"] = (
            df["oi_pressure"].abs() * (df["liquidation_density"] + 1.0) *
            (df.get("bb_squeeze", 1.0))
        )

        return df

    def _compute_particle_state(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute composite particle state features.

        Combines all physics features into interpretable states.
        """
        # Momentum state: strong/weak, building/fading
        df["momentum_state"] = (
            df["velocity_direction"] * df["kinetic_energy"] *
            (1 + df["momentum_alignment"])
        )

        # Tension state: how far from equilibrium and likely to snap back
        df["tension_state"] = df["potential_energy"] * np.sign(df["displacement_sma"])

        # Energy state: total energy level
        df["energy_level"] = df["total_energy"]

        # Breakout potential: energy building near boundaries
        df["breakout_potential"] = (
            df["kinetic_energy"] * df["field_strength"] *
            (1 - df["damping_ratio"])
        )

        # Mean reversion potential: high potential energy with weak momentum
        df["reversion_potential"] = (
            df["potential_energy"] * (1 - df["kinetic_energy"].clip(upper=1)) *
            df["mean_reversion_strength"].clip(lower=0)
        )

        # Composite particle score (directional signal)
        # Positive = bullish physics, negative = bearish physics
        df["particle_physics_score"] = (
            df["force_net"] * 0.3 +
            df["momentum_state"] * 0.2 +
            df["field_gradient"] * 0.2 +
            df["energy_injection"] * 0.15 +
            df["vwap_zscore"] * -0.15 +  # Mean reversion component
            df.get("funding_force", 0.0) * -0.05 +
            df.get("oi_pressure", 0.0) * 0.05 +
            df.get("cvd_momentum", 0.0) * 0.05
        )

        # Normalize to [-1, 1] range
        df["particle_physics_score"] = df.groupby("symbol")["particle_physics_score"].transform(
            lambda x: 2 * (x.rank(pct=True) - 0.5)
        )

        return df


def get_particle_feature_names() -> List[str]:
    """Get list of all particle physics feature names."""
    return [
        # Kinematics
        "velocity", "velocity_abs", "velocity_direction",
        "acceleration", "acceleration_abs", "acceleration_direction",
        "momentum_alignment", "jerk", "jerk_abs",

        # Mass
        "mass", "mass_change",

        # Forces
        "force_ofi", "force_newton", "force_net", "force_impulse",

        # Energy
        "kinetic_energy", "kinetic_energy_change", "kinetic_energy_momentum",
        "potential_energy", "potential_energy_sma", "potential_energy_ema",
        "total_energy", "energy_change", "energy_injection", "energy_state",

        # Potential field
        "distance_to_support", "distance_to_resistance",
        "field_gradient", "field_strength",

        # Friction
        "friction", "damping_ratio", "mean_reversion_strength",

        # Volume profile
        "vwap", "vwap_distance", "vwap_zscore", "volume_momentum",

        # Anchored VWAP
        "avwap_impulse", "avwap_volume", "avwap_weekly",
        "avwap_impulse_distance", "avwap_volume_distance", "avwap_weekly_distance",

        # Bollinger geometry
        "bb_mid", "bb_upper", "bb_lower", "bb_width", "bb_squeeze",

        # RSI throttle
        "rsi", "rsi_throttle",

        # Ichimoku regime
        "ichimoku_tenkan", "ichimoku_kijun", "ichimoku_cloud_top",
        "ichimoku_cloud_bottom", "ichimoku_cloud_thickness", "ichimoku_regime",

        # Crypto forces
        "funding_force", "oi_change", "oi_pressure", "liquidation_density",
        "cvd", "cvd_momentum", "jump_hazard",

        # Composite states
        "momentum_state", "tension_state", "energy_level",
        "breakout_potential", "reversion_potential",
        "particle_physics_score",
    ]
