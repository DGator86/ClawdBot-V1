"""Validation harness module."""
from .walkforward import WalkForwardHarness, Fold
from .scoring import (
    pinball_loss,
    coverage,
    sharpness,
    crps_empirical,
    evaluate_predictions,
    IsotonicCalibrator,
    compute_ece,
    compute_stability_metrics,
)

__all__ = [
    "WalkForwardHarness",
    "Fold",
    "pinball_loss",
    "coverage",
    "sharpness",
    "crps_empirical",
    "evaluate_predictions",
    "IsotonicCalibrator",
    "compute_ece",
    "compute_stability_metrics",
]

# Optional imports (may fail if dependencies not present)
try:
    from .trade_walkforward import TradeWalkForwardHarness, TradeFold
    __all__.extend(["TradeWalkForwardHarness", "TradeFold"])
except ImportError:
    pass

try:
    from .bregman_optimizer import (
        ProjectFWOptimizer, ProjectFWConfig, OptimizationResult,
    )
    __all__.extend(["ProjectFWOptimizer", "ProjectFWConfig", "OptimizationResult"])
except ImportError:
    pass
