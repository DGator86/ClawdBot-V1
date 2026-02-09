from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    precision_score,
    recall_score,
)

from .constants import PRIMARY_TARGET_TF, TF_SECONDS
from .feature_engine import FeatureFrames, assemble_feature_row, build_feature_frames
from .models import EnsembleModel


@dataclass
class BacktestConfig:
    target_tf: str = PRIMARY_TARGET_TF
    train_window: int = 1000
    refit_every: int = 10


@dataclass
class BacktestResults:
    metrics: Dict[str, float]
    predictions: pd.DataFrame
    per_regime: Dict[str, Dict[str, float]]

    def to_json(self) -> str:
        return json.dumps(
            {
                "metrics": self.metrics,
                "per_regime": self.per_regime,
            },
            indent=2,
        )


def _ece_score(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if not np.any(mask):
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += np.abs(acc - conf) * (mask.sum() / len(y_true))
    return float(ece)


def _compute_metrics(pred_df: pd.DataFrame) -> Dict[str, float]:
    return {
        "log_loss": float(log_loss(pred_df["target"], pred_df["prob_up"], labels=[0, 1])),
        "brier": float(brier_score_loss(pred_df["target"], pred_df["prob_up"])),
        "accuracy": float(accuracy_score(pred_df["target"], pred_df["prob_up"] >= 0.5)),
        "precision": float(precision_score(pred_df["target"], pred_df["prob_up"] >= 0.5)),
        "recall": float(recall_score(pred_df["target"], pred_df["prob_up"] >= 0.5)),
        "ece": float(_ece_score(pred_df["target"].values, pred_df["prob_up"].values)),
    }


def _compute_per_regime(
    pred_df: pd.DataFrame,
    feature_df: pd.DataFrame,
) -> Dict[str, Dict[str, float]]:
    vol_col = feature_df.get("1h__vol_20")
    per_regime: Dict[str, Dict[str, float]] = {}
    if vol_col is None:
        return per_regime

    vol_threshold = np.nanmedian(vol_col.values)
    aligned_vol = vol_col.loc[pred_df.index]
    low_mask = aligned_vol <= vol_threshold
    high_mask = aligned_vol > vol_threshold
    for name, mask in {
        "low_vol": low_mask,
        "high_vol": high_mask,
    }.items():
        if mask.sum() == 0:
            continue
        per_regime[name] = {
            "accuracy": float(accuracy_score(pred_df["target"][mask], pred_df["prob_up"][mask] >= 0.5)),
            "log_loss": float(log_loss(pred_df["target"][mask], pred_df["prob_up"][mask], labels=[0, 1])),
        }
    return per_regime


def compute_metrics(
    pred_df: pd.DataFrame,
    feature_df: pd.DataFrame,
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    return _compute_metrics(pred_df), _compute_per_regime(pred_df, feature_df)


def build_dataset(
    bars_by_tf: Dict[str, pd.DataFrame],
    target_tf: str = PRIMARY_TARGET_TF,
) -> Tuple[pd.DataFrame, pd.Series]:
    feature_frames = build_feature_frames(bars_by_tf)
    target_df = bars_by_tf[target_tf].copy()
    target_df = target_df.sort_values("timestamp")
    target_df["close_time"] = target_df["timestamp"] + pd.to_timedelta(TF_SECONDS[target_tf], unit="s")

    features: List[Dict[str, float]] = []
    labels: List[int] = []
    times: List[pd.Timestamp] = []

    for idx in range(len(target_df) - 1):
        close_time = pd.to_datetime(target_df.iloc[idx]["close_time"], utc=True)
        feature_row = assemble_feature_row(feature_frames, close_time)
        features.append(feature_row)
        close_now = target_df.iloc[idx]["close"]
        close_next = target_df.iloc[idx + 1]["close"]
        labels.append(int(close_next > close_now))
        times.append(close_time)

    feature_df = pd.DataFrame(features, index=pd.DatetimeIndex(times, name="timestamp"))
    label_series = pd.Series(labels, index=feature_df.index, name="target")
    return feature_df, label_series


def walk_forward_backtest(
    feature_df: pd.DataFrame,
    label_series: pd.Series,
    config: BacktestConfig,
) -> BacktestResults:
    X = feature_df.values
    y = label_series.values

    preds = []
    pred_times = []

    model = EnsembleModel()

    for idx in range(config.train_window, len(feature_df)):
        if (idx - config.train_window) % config.refit_every == 0:
            X_train = X[idx - config.train_window : idx]
            y_train = y[idx - config.train_window : idx]
            model.fit(X_train, y_train)

        prob = model.predict_proba(X[idx : idx + 1])[0]
        preds.append(prob)
        pred_times.append(feature_df.index[idx])

    pred_df = pd.DataFrame(
        {
            "timestamp": pred_times,
            "prob_up": preds,
            "target": y[config.train_window :],
        }
    ).set_index("timestamp")

    metrics, per_regime = compute_metrics(pred_df, feature_df)

    return BacktestResults(metrics=metrics, predictions=pred_df, per_regime=per_regime)
