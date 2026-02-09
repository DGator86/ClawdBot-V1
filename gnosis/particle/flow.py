"""Particle flow state representation."""
import numpy as np
import pandas as pd


class ParticleState:
    """Represents the particle state combining flow, regime, and barrier info."""

    def __init__(self, models_config: dict):
        self.config = models_config.get("particle", {})
        self.flow_weight = self.config.get("flow_weight", 1.0)
        self.regime_weight = self.config.get("regime_weight", 1.0)
        self.barrier_weight = self.config.get("barrier_weight", 1.0)
        self.flow_span = self.config.get("flow_span", 10)  # EWM span for flow momentum

    def compute_state(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Compute particle state features."""
        df = features_df.copy()

        # Flow momentum (weighted cumulative OFI)
        df["flow_momentum"] = df.groupby("symbol")["ofi"].transform(
            lambda x: x.ewm(span=self.flow_span).mean()
        ) * self.flow_weight

        # Regime stability (how consistent the K label has been)
        # Use K_label if available, fall back to K for backwards compatibility
        k_col = "K_label" if "K_label" in df.columns else "K"
        df["regime_encoded"] = df[k_col].map({
            "K_TRENDING": 1,
            "K_MEAN_REVERTING": -1,
            "K_BALANCED": 0
        }).fillna(0)
        df["regime_stability"] = df.groupby("symbol")["regime_encoded"].transform(
            lambda x: x.rolling(10, min_periods=1).std()
        ) * self.regime_weight

        # Barrier proximity (distance to recent high/low)
        df["barrier_high"] = df.groupby("symbol")["high"].transform(
            lambda x: x.rolling(50, min_periods=10).max()
        )
        df["barrier_low"] = df.groupby("symbol")["low"].transform(
            lambda x: x.rolling(50, min_periods=10).min()
        )
        df["barrier_proximity"] = (
            (df["close"] - df["barrier_low"]) /
            (df["barrier_high"] - df["barrier_low"] + 1e-9)
        ) * self.barrier_weight

        # Combined particle score
        df["particle_score"] = (
            df["flow_momentum"] +
            (1 - df["regime_stability"]) * 0.5 +
            (df["barrier_proximity"] - 0.5) * 0.3
        )

        return df
