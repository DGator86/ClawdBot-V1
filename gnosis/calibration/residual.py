from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd


def _ewma_last(x: np.ndarray, lam: float) -> float:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return float("nan")
    s = x[0]
    for v in x[1:]:
        s = lam * v + (1.0 - lam) * s
    return float(s)


@dataclass(frozen=True)
class ResidualCalibSpec:
    enabled: bool = True
    alpha: float = 0.10
    cal_frac: float = 0.20
    min_cal: int = 50
    ewma_lambda: float = 0.35
    method: str = "quantile"   # "quantile" | "ewma"
    clip_lo: float = 0.0
    clip_hi: float = 0.10      # max half-width


def split_fit_cal(train_df: pd.DataFrame, cal_frac: float, min_cal: int) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    n = len(train_df)
    if n < (min_cal + 10):
        return train_df, None
    cal_n = max(min_cal, int(n * cal_frac))
    cal_n = min(cal_n, n - 10)
    return train_df.iloc[:-cal_n].copy(), train_df.iloc[-cal_n:].copy()


def fit_delta_symmetric(
    y_true: np.ndarray,
    q50: np.ndarray,
    *,
    alpha: float,
    method: str = "quantile",
    ewma_lambda: float = 0.35,
    clip_lo: float = 0.0,
    clip_hi: float = 0.10,
) -> float:
    y_true = np.asarray(y_true, dtype=float)
    q50 = np.asarray(q50, dtype=float)
    m = np.isfinite(y_true) & np.isfinite(q50)
    if not np.any(m):
        return float(np.clip(0.0, clip_lo, clip_hi))

    abs_res = np.abs(y_true[m] - q50[m])
    if abs_res.size == 0:
        return float(np.clip(0.0, clip_lo, clip_hi))

    if method == "ewma":
        s = _ewma_last(abs_res, ewma_lambda)
        med = float(np.median(abs_res))
        q = float(np.quantile(abs_res, 1.0 - alpha))
        k = (q / med) if med > 1e-12 else 1.0
        delta = float(s * k)
    else:
        delta = float(np.quantile(abs_res, 1.0 - alpha))

    return float(np.clip(delta, clip_lo, clip_hi))


def apply_delta_symmetric(preds: pd.DataFrame, delta: float) -> pd.DataFrame:
    out = preds.copy()
    if "q50" not in out.columns:
        raise ValueError("preds must contain q50")
    d = float(delta)
    out["q05"] = out["q50"] - d
    out["q95"] = out["q50"] + d
    if "x_hat" not in out.columns:
        out["x_hat"] = out["q50"]
    out["sigma_hat"] = (out["q95"] - out["q05"]) / 3.29
    return out
