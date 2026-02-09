"""Scoring and calibration metrics."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

from gnosis.utils import drop_future_return_cols, safe_merge_no_truth


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    """Compute pinball (quantile) loss."""
    errors = y_true - y_pred
    return np.mean(np.where(errors >= 0, quantile * errors, (quantile - 1) * errors))


def coverage(y_true: np.ndarray, q_low: np.ndarray, q_high: np.ndarray) -> float:
    """Compute interval coverage."""
    in_interval = (y_true >= q_low) & (y_true <= q_high)
    return np.mean(in_interval)


def sharpness(q_low: np.ndarray, q_high: np.ndarray) -> float:
    """Compute sharpness (average interval width)."""
    return np.mean(q_high - q_low)


def crps_empirical(y_true: np.ndarray, quantiles: dict[float, np.ndarray]) -> float:
    """Approximate CRPS using quantile predictions."""
    # Simple approximation using available quantiles
    total = 0.0
    sorted_qs = sorted(quantiles.keys())

    for i, q in enumerate(sorted_qs):
        pred = quantiles[q]
        total += pinball_loss(y_true, pred, q)

    return total / len(sorted_qs)


def evaluate_predictions(
    predictions_df: pd.DataFrame,
    actuals_df: pd.DataFrame,
    target_col: str = "future_return",
) -> dict:
    """Evaluate prediction quality."""
    # Merge predictions with actuals
    merged = predictions_df.merge(
        actuals_df[["symbol", "bar_idx", target_col]],
        on=["symbol", "bar_idx"],
        how="inner",
    )

    y_true = merged[target_col].values
    valid_mask = ~np.isnan(y_true)
    y_true = y_true[valid_mask]

    if len(y_true) == 0:
        return {
            "pinball_05": np.nan,
            "pinball_50": np.nan,
            "pinball_95": np.nan,
            "coverage_90": np.nan,
            "sharpness": np.nan,
            "crps_approx": np.nan,
            "mae": np.nan,
            "n_samples": 0,
        }

    q05 = merged["q05"].values[valid_mask]
    q50 = merged["q50"].values[valid_mask]
    q95 = merged["q95"].values[valid_mask]

    return {
        "pinball_05": pinball_loss(y_true, q05, 0.05),
        "pinball_50": pinball_loss(y_true, q50, 0.50),
        "pinball_95": pinball_loss(y_true, q95, 0.95),
        "coverage_90": coverage(y_true, q05, q95),
        "sharpness": sharpness(q05, q95),
        "crps_approx": crps_empirical(y_true, {0.05: q05, 0.50: q50, 0.95: q95}),
        "mae": np.mean(np.abs(y_true - q50)),
        "n_samples": len(y_true),
    }


class IsotonicCalibrator:
    """Isotonic regression calibrator for probability calibration.

    Fits isotonic regression on predicted probabilities vs actual outcomes
    to produce calibrated probabilities. Used for calibrating S_pmax.
    """

    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
        self.calibration_map: Optional[List[Tuple[float, float]]] = None
        self._fitted = False

    def fit(self, probs: np.ndarray, outcomes: np.ndarray) -> "IsotonicCalibrator":
        """Fit isotonic calibrator on training data.

        Args:
            probs: Predicted probabilities (N,)
            outcomes: Binary outcomes (N,) - 1 if prediction was correct, 0 otherwise

        Returns:
            self for chaining
        """
        # Sort by probability
        sorted_idx = np.argsort(probs)
        sorted_probs = probs[sorted_idx]
        sorted_outcomes = outcomes[sorted_idx]

        # Compute cumulative sums for pool adjacent violators algorithm (PAVA)
        n = len(sorted_probs)
        if n == 0:
            self.calibration_map = [(0.0, 0.5), (1.0, 0.5)]
            self._fitted = True
            return self

        # Isotonic regression via PAVA
        calibrated = self._pava(sorted_outcomes)

        # Build calibration map: (input_prob, output_prob) pairs
        self.calibration_map = []
        for i in range(n):
            self.calibration_map.append((float(sorted_probs[i]), float(calibrated[i])))

        # Ensure we have boundary points
        if len(self.calibration_map) > 0:
            if self.calibration_map[0][0] > 0.0:
                self.calibration_map.insert(0, (0.0, self.calibration_map[0][1]))
            if self.calibration_map[-1][0] < 1.0:
                self.calibration_map.append((1.0, self.calibration_map[-1][1]))

        self._fitted = True
        return self

    def _pava(self, y: np.ndarray) -> np.ndarray:
        """Pool Adjacent Violators Algorithm for isotonic regression."""
        n = len(y)
        if n == 0:
            return y

        result = y.astype(float).copy()
        weights = np.ones(n)

        i = 0
        while i < n - 1:
            # Find the next block that violates isotonicity
            if result[i] > result[i + 1]:
                # Pool adjacent violators
                j = i + 1
                while j < n and result[j - 1] > result[j]:
                    j += 1

                # Compute weighted average for the block
                block_sum = np.sum(result[i:j] * weights[i:j])
                block_weight = np.sum(weights[i:j])
                avg = block_sum / block_weight

                # Update the block
                result[i:j] = avg
                weights[i:j] = block_weight

                # Check previous blocks for violations
                if i > 0:
                    i -= 1
                    continue

            i += 1

        return result

    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        """Apply calibration to new probabilities.

        Args:
            probs: Predicted probabilities to calibrate (N,)

        Returns:
            Calibrated probabilities (N,)
        """
        if not self._fitted or self.calibration_map is None:
            return probs

        result = np.zeros_like(probs)
        map_probs = np.array([p[0] for p in self.calibration_map])
        map_calib = np.array([p[1] for p in self.calibration_map])

        # Linear interpolation
        result = np.interp(probs, map_probs, map_calib)

        # Clip to valid probability range
        return np.clip(result, 0.0, 1.0)


def compute_ece(probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 10) -> Dict:
    """Compute Expected Calibration Error and reliability diagram data.

    Args:
        probs: Predicted probabilities (N,)
        outcomes: Binary outcomes (N,) - 1 if correct, 0 otherwise
        n_bins: Number of bins for calibration

    Returns:
        Dictionary with ECE and bin-level diagnostics
    """
    if len(probs) == 0:
        return {"ece": 0.0, "n_bins": n_bins, "bins": []}

    bins = []
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    total_weight = 0.0
    weighted_error = 0.0

    for i in range(n_bins):
        mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
        if i == n_bins - 1:  # Include right edge for last bin
            mask = (probs >= bin_edges[i]) & (probs <= bin_edges[i + 1])

        n_in_bin = np.sum(mask)
        if n_in_bin > 0:
            avg_prob = np.mean(probs[mask])
            avg_outcome = np.mean(outcomes[mask])
            bin_error = abs(avg_prob - avg_outcome)

            weighted_error += n_in_bin * bin_error
            total_weight += n_in_bin

            bins.append(
                {
                    "bin_idx": i,
                    "bin_start": float(bin_edges[i]),
                    "bin_end": float(bin_edges[i + 1]),
                    "n_samples": int(n_in_bin),
                    "avg_predicted_prob": float(avg_prob),
                    "avg_actual_freq": float(avg_outcome),
                    "calibration_error": float(bin_error),
                }
            )
        else:
            bins.append(
                {
                    "bin_idx": i,
                    "bin_start": float(bin_edges[i]),
                    "bin_end": float(bin_edges[i + 1]),
                    "n_samples": 0,
                    "avg_predicted_prob": float((bin_edges[i] + bin_edges[i + 1]) / 2),
                    "avg_actual_freq": None,
                    "calibration_error": None,
                }
            )

    ece = weighted_error / total_weight if total_weight > 0 else 0.0

    return {
        "ece": float(ece),
        "n_bins": n_bins,
        "n_samples": int(total_weight),
        "bins": bins,
    }


def compute_stability_metrics(df: pd.DataFrame, levels: List[str] = None) -> Dict:
    """Compute stability metrics for KPCOFGS regime labels.

    Args:
        df: DataFrame with regime labels ({level}_label columns)
        levels: List of levels to compute metrics for (default: K,P,C,O,F,G,S)

    Returns:
        Dictionary with flip_rate and avg_entropy per level
    """
    if levels is None:
        levels = ["K", "P", "C", "O", "F", "G", "S"]

    stability = {}

    for level in levels:
        label_col = f"{level}_label" if f"{level}_label" in df.columns else level
        entropy_col = f"{level}_entropy"

        if label_col not in df.columns:
            continue

        # Flip rate: fraction of steps where label changes
        labels = df[label_col].values
        if len(labels) > 1:
            flips = np.sum(labels[1:] != labels[:-1])
            flip_rate = flips / (len(labels) - 1)
        else:
            flip_rate = 0.0

        # Average entropy
        if entropy_col in df.columns:
            avg_entropy = float(df[entropy_col].mean())
        else:
            avg_entropy = 0.0

        stability[f"{level}_flip_rate"] = float(flip_rate)
        stability[f"{level}_avg_entropy"] = avg_entropy

    # Overall stability score (lower is more stable)
    flip_rates = [stability.get(f"{l}_flip_rate", 0.0) for l in levels]
    stability["overall_flip_rate"] = float(np.mean(flip_rates))

    return stability
