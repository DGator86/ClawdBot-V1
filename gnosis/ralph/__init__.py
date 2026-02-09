"""
Ralph Wiggum — Continuous Learning System
==========================================
Tracks predictions vs outcomes, computes performance metrics,
and adjusts hyperparameters to improve the ClawdBot+Yoshi+Kalshi
pipeline over time.

The learning loop:
  1. Record: store every prediction + market state
  2. Resolve: when outcome is known, compute accuracy
  3. Score: Brier score, calibration, value capture, Sharpe
  4. Optimize: adjust hyperparameters (explore/exploit)
  5. Persist: save state to JSON DB for continuity across restarts
"""
from gnosis.ralph.learner import RalphLearner, LearningConfig
from gnosis.ralph.tracker import PredictionTracker, PredictionRecord
from gnosis.ralph.hyperparams import HyperParams

__all__ = [
    "RalphLearner",
    "LearningConfig",
    "PredictionTracker",
    "PredictionRecord",
    "HyperParams",
]
