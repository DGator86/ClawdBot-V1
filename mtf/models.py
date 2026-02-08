from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import importlib.util

from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.ensemble import GradientBoostingClassifier

_HAS_LGBM = importlib.util.find_spec("lightgbm") is not None
if _HAS_LGBM:
    import lightgbm as lgb
else:
    lgb = None


@dataclass
class ModelOutputs:
    prob: np.ndarray


class LogisticModel:
    def __init__(self):
        self.model = LogisticRegression(max_iter=200)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]


class GBMModel:
    def __init__(self):
        if lgb is None:
            raise RuntimeError("LightGBM is not available")
        self.model: Optional[lgb.LGBMClassifier] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=-1,
            subsample=0.8,
            colsample_bytree=0.8,
        )
        self.model.fit(X, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]


class FallbackGBMModel:
    def __init__(self):
        self.model = GradientBoostingClassifier()

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]


class CalibrationModel:
    def __init__(self):
        self.model = IsotonicRegression(out_of_bounds="clip")
        self._is_fit = False

    def fit(self, prob: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(prob, y)
        self._is_fit = True

    def predict(self, prob: np.ndarray) -> np.ndarray:
        if not self._is_fit:
            return prob
        return self.model.predict(prob)


class EnsembleModel:
    def __init__(self):
        self.logistic = LogisticModel()
        self.gbm = GBMModel() if _HAS_LGBM else FallbackGBMModel()
        self.calibrator = CalibrationModel()

    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        self.logistic.fit(X, y)
        self.gbm.fit(X, y)

        prob = self._raw_predict_proba(X)
        self.calibrator.fit(prob, y)
        return {
            "log_loss": float(log_loss(y, self.calibrator.predict(prob))),
        }

    def _raw_predict_proba(self, X: np.ndarray) -> np.ndarray:
        p_lr = self.logistic.predict_proba(X)
        p_gbm = self.gbm.predict_proba(X)
        return 0.5 * p_lr + 0.5 * p_gbm

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        prob = self._raw_predict_proba(X)
        return self.calibrator.predict(prob)
