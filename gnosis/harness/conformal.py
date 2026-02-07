import numpy as np
import pandas as pd

def cqr_delta(
    y_true: np.ndarray,
    q_lo: np.ndarray,
    q_hi: np.ndarray,
    sigma: np.ndarray | None = None,
    alpha: float = 0.10,
    normalized: bool = True,
) -> float:
    """
    Conformalized Quantile Regression (CQR) delta.
    Score per point: s = max(q_lo - y, y - q_hi).
    If normalized: s = s / sigma (sigma clipped away from 0).
    Delta = (1-alpha) quantile of scores using finite-sample correction.
    """
    y_true = np.asarray(y_true, dtype=float)
    q_lo   = np.asarray(q_lo, dtype=float)
    q_hi   = np.asarray(q_hi, dtype=float)

    s = np.maximum(q_lo - y_true, y_true - q_hi)
    s = np.maximum(s, 0.0)

    if normalized:
        if sigma is None:
            raise ValueError("normalized=True requires sigma.")
        sig = np.asarray(sigma, dtype=float)
        sig = np.clip(sig, 1e-12, np.inf)
        s = s / sig

    n = s.size
    if n == 0:
        return 0.0

    # Finite-sample conformal quantile: ceil((n+1)*(1-alpha))/n
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    k = min(max(k, 1), n)
    return float(np.partition(s, k - 1)[k - 1])

def apply_cqr_delta(
    preds: pd.DataFrame,
    delta: float,
    normalized: bool = True,
) -> pd.DataFrame:
    """
    Apply delta to widen q05/q95 (and keep q50/x_hat unchanged).
    If normalized: widen by delta*sigma_hat per row.
    Else: widen by constant delta.
    """
    out = preds.copy()
    if normalized:
        sig = np.clip(out["sigma_hat"].astype(float).values, 1e-12, np.inf)
        widen = delta * sig
    else:
        widen = delta

    out["q05"] = out["q05"].astype(float) - widen
    out["q95"] = out["q95"].astype(float) + widen

    # keep sigma_hat consistent with interval width (approx std for 90% normal)
    out["sigma_hat"] = (out["q95"].astype(float) - out["q05"].astype(float)) / 3.29
    return out
