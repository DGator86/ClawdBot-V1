"""Quantile prediction models.

Supports multiple backends:
- ridge: Simple Ridge regression with weighted loss (fast, baseline)
- quantile: Proper sklearn QuantileRegressor (accurate)
- bregman_fw: Bregman-Frank-Wolfe optimization (advanced, constrained)
- gradient_boost: XGBoost/LightGBM with quantile loss (nonlinear)
"""
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# Constants for quantile prediction
# Z-score for 90% confidence interval (two-tailed)
Z_SCORE_90 = 1.645
# Divisor to convert 90% interval width to approximate standard deviation
# Based on normal distribution: (q95 - q05) / 3.29 ≈ sigma
# where 3.29 = 2 * 1.645 (z-scores for 5th and 95th percentiles)
IQR_TO_SIGMA_DIVISOR = 3.29

# Extended feature set for improved predictions
EXTENDED_FEATURES = [
    # Core features
    "returns", "realized_vol", "ofi", "range_pct",
    "flow_momentum", "regime_stability", "barrier_proximity",
    "particle_score",
    # Advanced features (if available)
    "ofi_momentum", "volume_weighted_returns", "price_position",
    "intrabar_vol", "return_accel", "regime_entropy",
]

# Particle physics features (highest correlation with future returns)
PHYSICS_FEATURES = [
    # Kinematics (velocity, acceleration)
    "velocity", "acceleration", "jerk", "momentum_alignment",
    # Mass and forces
    "mass", "force_net", "force_impulse",
    # Energy states
    "kinetic_energy", "potential_energy", "energy_injection",
    # Potential field (support/resistance)
    "field_gradient", "field_strength",
    # Mean reversion
    "mean_reversion_strength", "damping_ratio",
    # Volume profile
    "vwap_zscore", "volume_momentum",
    # Composite scores
    "momentum_state", "tension_state",
    "breakout_potential", "reversion_potential",
    "particle_physics_score",
]


class QuantilePredictor:
    """Quantile predictor with multiple backend options.

    Backends:
        - ridge: Ridge regression with sample weights (default, fast)
        - quantile: sklearn QuantileRegressor (proper quantile loss)
        - bregman_fw: Frank-Wolfe optimization (constrained)
        - gradient_boost: Gradient boosting (nonlinear)
    """

    def __init__(self, models_config: dict):
        self.config = models_config.get("predictor", {})
        self.quantiles = self.config.get("quantiles", [0.05, 0.50, 0.95])
        self.l2_reg = self.config.get("l2_reg", 1.0)
        self.backend = self.config.get("backend", "ridge")
        self.use_extended_features = self.config.get("extended_features", False)
        self.normalize_features = self.config.get("normalize", True)
        self.models = {}
        self.scaler = None
        self._feature_cols = None

    def _get_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract feature matrix with optional extended features."""
        if self.use_extended_features:
            # Include both extended and physics features
            feature_cols = EXTENDED_FEATURES + PHYSICS_FEATURES
        else:
            feature_cols = [
                "returns", "realized_vol", "ofi", "range_pct",
                "flow_momentum", "regime_stability", "barrier_proximity",
                "particle_score"
            ]

        # Only use columns that exist (deduplicate)
        available = list(dict.fromkeys(c for c in feature_cols if c in df.columns))
        self._feature_cols = available
        X = df[available].fillna(0).values

        # Normalize if enabled
        if self.normalize_features and self.scaler is not None:
            X = self.scaler.transform(X)

        return X

    def fit(self, train_df: pd.DataFrame, target_col: str = "future_return") -> None:
        """Fit quantile models using configured backend."""
        # Determine available features first
        if self.use_extended_features:
            # Include both extended and physics features
            feature_cols = EXTENDED_FEATURES + PHYSICS_FEATURES
        else:
            feature_cols = [
                "returns", "realized_vol", "ofi", "range_pct",
                "flow_momentum", "regime_stability", "barrier_proximity",
                "particle_score"
            ]
        # Only use columns that exist (deduplicate)
        available = list(dict.fromkeys(c for c in feature_cols if c in train_df.columns))
        self._feature_cols = available

        X = train_df[available].fillna(0).values
        y = train_df[target_col].fillna(0).values

        # Fit scaler if normalizing
        if self.normalize_features:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)

        # Dispatch to appropriate backend
        if self.backend == "quantile":
            self._fit_quantile_regressor(X, y)
        elif self.backend == "bregman_fw":
            self._fit_bregman_fw(X, y)
        elif self.backend == "gradient_boost":
            self._fit_gradient_boost(X, y)
        else:
            self._fit_ridge(X, y)

    def _fit_ridge(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit using Ridge regression with sample weights."""
        for q in self.quantiles:
            weights = np.where(y > 0, q, 1 - q)
            model = Ridge(alpha=self.l2_reg)
            model.fit(X, y, sample_weight=weights)
            self.models[q] = model

    def _fit_quantile_regressor(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit using proper sklearn QuantileRegressor."""
        try:
            from sklearn.linear_model import QuantileRegressor
        except ImportError:
            warnings.warn("QuantileRegressor not available, falling back to Ridge")
            return self._fit_ridge(X, y)

        for q in self.quantiles:
            model = QuantileRegressor(
                quantile=q,
                alpha=self.l2_reg,
                solver='highs'
            )
            model.fit(X, y)
            self.models[q] = model

    def _fit_bregman_fw(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit using Bregman-Frank-Wolfe optimization."""
        try:
            from .bregman_fw import BregmanFWPredictor, FWConfig
        except ImportError:
            warnings.warn("Bregman-FW not available, falling back to Ridge")
            return self._fit_ridge(X, y)

        fw_config = FWConfig(
            quantiles=self.quantiles,
            l2_reg=self.l2_reg,
            alpha=self.config.get('fw_alpha', 0.1),
            max_iterations=self.config.get('fw_max_iter', 100)
        )

        predictor = BregmanFWPredictor(fw_config)
        predictor.fit(X, y)

        # Store the fitted predictor
        self.models['bregman_fw'] = predictor

    def _fit_gradient_boost(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit using gradient boosting with quantile loss."""
        try:
            from sklearn.ensemble import GradientBoostingRegressor
        except ImportError:
            warnings.warn("GradientBoostingRegressor not available")
            return self._fit_ridge(X, y)

        for q in self.quantiles:
            model = GradientBoostingRegressor(
                loss='quantile',
                alpha=q,
                n_estimators=self.config.get('n_estimators', 100),
                max_depth=self.config.get('max_depth', 3),
                learning_rate=self.config.get('learning_rate', 0.1),
                random_state=42
            )
            model.fit(X, y)
            self.models[q] = model

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate quantile predictions using configured backend."""
        X = self._get_features(df)
        result = df[["symbol", "bar_idx", "timestamp_end", "close"]].copy()

        # Handle Bregman-FW backend separately
        if self.backend == "bregman_fw" and 'bregman_fw' in self.models:
            preds = self.models['bregman_fw'].predict(X)
            result["q05"] = preds['q05']
            result["q50"] = preds['q50']
            result["q95"] = preds['q95']
        else:
            # Standard per-quantile model prediction
            for q in self.quantiles:
                if q in self.models:
                    pred = self.models[q].predict(X)
                    result[f"q{int(q*100):02d}"] = pred
                else:
                    result[f"q{int(q*100):02d}"] = 0.0

        # Point estimate is median
        result["x_hat"] = result["q50"]

        # Uncertainty from IQR (before any scaling)
        result["sigma_hat"] = (result["q95"] - result["q05"]) / 3.29  # approx std

        # Apply sigma_scale to widen/narrow intervals (SINGLE application)
        # Priority: models.predictor.sigma_scale > models.sigma_scale > 1.0
        sigma_scale = 1.0
        if isinstance(self.config, dict):
            sigma_scale = float(self.config.get('sigma_scale', 1.0))
            _pcfg = self.config.get('predictor', {})
            if isinstance(_pcfg, dict) and 'sigma_scale' in _pcfg:
                sigma_scale = float(_pcfg['sigma_scale'])

        if sigma_scale <= 0:
            sigma_scale = 1.0

        if sigma_scale != 1.0:
            center = result['q50']
            half = (result['q95'] - result['q05']) / 2.0
            half = half * sigma_scale
            result['q05'] = center - half
            result['q95'] = center + half
            result['sigma_hat'] = (result['q95'] - result['q05']) / 3.29

        return result


class BaselinePredictor:
    """Random walk + realized vol cone baseline."""

    def __init__(self):
        pass

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate baseline predictions (random walk + vol cone)."""
        result = df[["symbol", "bar_idx", "timestamp_end", "close"]].copy()

        # Random walk: point estimate = 0 return
        result["x_hat"] = 0.0

        # Vol cone: use realized vol for uncertainty
        vol = df["realized_vol"].fillna(df["realized_vol"].median())
        result["sigma_hat"] = vol

        # Quantiles from normal distribution
        result["q05"] = -1.645 * vol
        result["q50"] = 0.0
        result["q95"] = 1.645 * vol

        return result
