"""Physics Parameter Calibration Framework.

Empirically calibrates REGIME_PARAMS and other physics constants
using historical backtesting with time-series cross-validation.

Optimizes for:
1. Calibration accuracy (predicted probabilities match actual outcomes)
2. Sharpness (narrow confidence intervals when certain)
3. Regime-specific performance

All calibration parameters are exposed as hyperparameters for ML tuning.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from itertools import product
import copy


@dataclass
class CalibrationConfig:
    """Hyperparameters for the calibration process.

    All parameters can be tuned by the improvement loop.
    """
    # Cross-validation settings
    n_folds: int = 5
    train_pct: float = 0.7  # Within each fold
    purge_bars: int = 10  # Gap between train and validation
    embargo_bars: int = 5  # Gap after validation

    # Optimization settings
    max_iterations: int = 100
    early_stopping_rounds: int = 10
    convergence_threshold: float = 0.001

    # Loss function weights
    calibration_weight: float = 1.0  # Weight for calibration error
    sharpness_weight: float = 0.5  # Weight for interval sharpness
    coverage_weight: float = 0.8  # Weight for coverage accuracy
    directional_weight: float = 0.3  # Weight for directional accuracy

    # Target coverage levels
    target_coverage_50: float = 0.50
    target_coverage_90: float = 0.90

    # Regularization
    l2_regularization: float = 0.01  # Penalize extreme parameters
    smoothness_regularization: float = 0.001  # Penalize volatile parameters

    # Search bounds (multipliers on defaults)
    param_lower_bound_mult: float = 0.1
    param_upper_bound_mult: float = 5.0

    # Grid search resolution
    grid_resolution: int = 5  # Points per parameter in grid search


@dataclass
class CalibrationResult:
    """Result of parameter calibration."""
    optimal_params: Dict[str, Any]
    calibration_error: float
    coverage_90: float
    coverage_50: float
    directional_accuracy: float
    sharpness: float
    total_loss: float
    fold_results: List[Dict[str, float]]
    iterations: int


class PhysicsCalibrator:
    """Calibrates physics engine parameters using historical data.

    The calibration process:
    1. Split data into time-series folds (with purge/embargo)
    2. For each parameter combination:
       - Train model on train set
       - Generate predictions on validation set
       - Compute calibration metrics
    3. Select parameters that minimize combined loss
    4. Validate on held-out test set

    The goal is to find parameters where:
    - 90% prediction intervals contain 90% of actual outcomes
    - Confidence correlates with actual accuracy
    - Predictions are as sharp (narrow) as possible while maintaining coverage
    """

    def __init__(self, config: Optional[CalibrationConfig] = None):
        """Initialize calibrator with config.

        Args:
            config: CalibrationConfig with hyperparameters
        """
        self.config = config or CalibrationConfig()
        self._best_params: Dict[str, Any] = {}
        self._calibration_history: List[CalibrationResult] = []

    def compute_calibration_error(
        self,
        predictions: pd.DataFrame,
        actuals: pd.Series,
    ) -> float:
        """Compute Expected Calibration Error (ECE).

        For binned predictions, ECE = sum(|accuracy - confidence| * bin_weight)

        Args:
            predictions: DataFrame with q05, q50, q95 columns
            actuals: Series of actual outcomes

        Returns:
            ECE (lower is better)
        """
        if len(predictions) == 0:
            return 1.0

        # Bin by predicted interval width (proxy for confidence)
        predictions = predictions.copy()
        predictions['interval_width'] = predictions['q95'] - predictions['q05']
        predictions['actual'] = actuals.values
        predictions['in_interval'] = (
            (predictions['actual'] >= predictions['q05']) &
            (predictions['actual'] <= predictions['q95'])
        )

        # Create bins
        try:
            predictions['confidence_bin'] = pd.qcut(
                predictions['interval_width'],
                q=10,
                duplicates='drop'
            )
        except ValueError:
            return 0.5  # Can't bin, return neutral error

        # Compute ECE
        ece = 0.0
        for bin_name, group in predictions.groupby('confidence_bin', observed=True):
            expected_coverage = 0.90  # We target 90% intervals
            actual_coverage = group['in_interval'].mean()
            bin_weight = len(group) / len(predictions)
            ece += bin_weight * abs(actual_coverage - expected_coverage)

        return ece

    def compute_coverage(
        self,
        predictions: pd.DataFrame,
        actuals: pd.Series,
        level: float = 0.90,
    ) -> float:
        """Compute coverage at a specific confidence level.

        Args:
            predictions: DataFrame with quantile columns
            actuals: Series of actual outcomes
            level: Confidence level (0.90 for 90%)

        Returns:
            Coverage (fraction of actuals within interval)
        """
        if len(predictions) == 0:
            return 0.0

        if level == 0.90:
            lower, upper = predictions['q05'], predictions['q95']
        elif level == 0.50:
            lower = predictions['q50'] - (predictions['q95'] - predictions['q05']) * 0.25
            upper = predictions['q50'] + (predictions['q95'] - predictions['q05']) * 0.25
        else:
            # Interpolate
            width_factor = level / 0.90
            half_width = (predictions['q95'] - predictions['q05']) / 2 * width_factor
            lower = predictions['q50'] - half_width
            upper = predictions['q50'] + half_width

        in_interval = (actuals >= lower.values) & (actuals <= upper.values)
        return in_interval.mean()

    def compute_sharpness(
        self,
        predictions: pd.DataFrame,
    ) -> float:
        """Compute sharpness (average interval width).

        Lower is better (sharper predictions).

        Args:
            predictions: DataFrame with q05, q95 columns

        Returns:
            Average interval width (normalized by q50)
        """
        if len(predictions) == 0:
            return 1.0

        interval_width = predictions['q95'] - predictions['q05']
        # Normalize by median prediction to get relative width
        relative_width = interval_width / (predictions['q50'].abs() + 1e-9)

        return relative_width.mean()

    def compute_directional_accuracy(
        self,
        predictions: pd.DataFrame,
        actuals: pd.Series,
    ) -> float:
        """Compute directional accuracy.

        Args:
            predictions: DataFrame with q50 (point estimate)
            actuals: Series of actual outcomes

        Returns:
            Fraction of correct directional predictions
        """
        if len(predictions) == 0:
            return 0.5

        pred_direction = np.sign(predictions['q50'].values)
        actual_direction = np.sign(actuals.values)

        correct = (pred_direction == actual_direction) | (actual_direction == 0)
        return correct.mean()

    def compute_total_loss(
        self,
        calibration_error: float,
        coverage_90: float,
        coverage_50: float,
        sharpness: float,
        directional_accuracy: float,
        params: Dict[str, Any],
    ) -> float:
        """Compute total loss for optimization.

        Args:
            calibration_error: ECE
            coverage_90: Coverage at 90% level
            coverage_50: Coverage at 50% level
            sharpness: Average interval width
            directional_accuracy: Directional accuracy
            params: Current parameters (for regularization)

        Returns:
            Total loss (lower is better)
        """
        cfg = self.config

        # Coverage errors (deviation from target)
        coverage_90_error = abs(coverage_90 - cfg.target_coverage_90)
        coverage_50_error = abs(coverage_50 - cfg.target_coverage_50)

        # Directional error (1 - accuracy)
        directional_error = 1.0 - directional_accuracy

        # Combine losses
        loss = (
            cfg.calibration_weight * calibration_error +
            cfg.coverage_weight * (coverage_90_error + coverage_50_error) +
            cfg.sharpness_weight * sharpness +
            cfg.directional_weight * directional_error
        )

        # L2 regularization on parameters
        if cfg.l2_regularization > 0 and params:
            param_values = [v for v in params.values() if isinstance(v, (int, float))]
            if param_values:
                l2_penalty = np.sum(np.array(param_values) ** 2) * cfg.l2_regularization
                loss += l2_penalty

        return loss

    def create_time_series_folds(
        self,
        df: pd.DataFrame,
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Create time-series cross-validation folds.

        Args:
            df: DataFrame sorted by time

        Returns:
            List of (train_df, val_df) tuples
        """
        n = len(df)
        folds = []

        fold_size = n // self.config.n_folds

        for i in range(self.config.n_folds):
            # Expanding window: use all data up to fold i for training
            train_end = (i + 1) * fold_size
            train_end = int(train_end * self.config.train_pct)

            # Validation window
            val_start = train_end + self.config.purge_bars
            val_end = (i + 1) * fold_size

            if val_start >= val_end or val_end > n:
                continue

            train_df = df.iloc[:train_end].copy()
            val_df = df.iloc[val_start:val_end].copy()

            if len(train_df) > 100 and len(val_df) > 20:
                folds.append((train_df, val_df))

        return folds

    def evaluate_params(
        self,
        params: Dict[str, Any],
        folds: List[Tuple[pd.DataFrame, pd.DataFrame]],
        predict_fn: Callable[[pd.DataFrame, Dict], pd.DataFrame],
    ) -> CalibrationResult:
        """Evaluate a parameter set across all folds.

        Args:
            params: Parameter dictionary to evaluate
            folds: List of (train, val) DataFrames
            predict_fn: Function that takes (df, params) and returns predictions

        Returns:
            CalibrationResult with aggregated metrics
        """
        fold_results = []

        for train_df, val_df in folds:
            # Generate predictions on validation set
            predictions = predict_fn(val_df, params)

            if len(predictions) == 0:
                continue

            # Compute metrics
            actuals = val_df['future_return'] if 'future_return' in val_df.columns else val_df['returns']

            calibration_error = self.compute_calibration_error(predictions, actuals)
            coverage_90 = self.compute_coverage(predictions, actuals, 0.90)
            coverage_50 = self.compute_coverage(predictions, actuals, 0.50)
            sharpness = self.compute_sharpness(predictions)
            directional_accuracy = self.compute_directional_accuracy(predictions, actuals)

            fold_results.append({
                'calibration_error': calibration_error,
                'coverage_90': coverage_90,
                'coverage_50': coverage_50,
                'sharpness': sharpness,
                'directional_accuracy': directional_accuracy,
            })

        if not fold_results:
            return CalibrationResult(
                optimal_params=params,
                calibration_error=1.0,
                coverage_90=0.0,
                coverage_50=0.0,
                directional_accuracy=0.5,
                sharpness=1.0,
                total_loss=float('inf'),
                fold_results=[],
                iterations=0,
            )

        # Aggregate across folds
        avg_calibration = np.mean([r['calibration_error'] for r in fold_results])
        avg_coverage_90 = np.mean([r['coverage_90'] for r in fold_results])
        avg_coverage_50 = np.mean([r['coverage_50'] for r in fold_results])
        avg_sharpness = np.mean([r['sharpness'] for r in fold_results])
        avg_directional = np.mean([r['directional_accuracy'] for r in fold_results])

        total_loss = self.compute_total_loss(
            avg_calibration, avg_coverage_90, avg_coverage_50,
            avg_sharpness, avg_directional, params
        )

        return CalibrationResult(
            optimal_params=params,
            calibration_error=avg_calibration,
            coverage_90=avg_coverage_90,
            coverage_50=avg_coverage_50,
            directional_accuracy=avg_directional,
            sharpness=avg_sharpness,
            total_loss=total_loss,
            fold_results=fold_results,
            iterations=1,
        )

    def grid_search(
        self,
        param_grid: Dict[str, List[Any]],
        folds: List[Tuple[pd.DataFrame, pd.DataFrame]],
        predict_fn: Callable[[pd.DataFrame, Dict], pd.DataFrame],
    ) -> CalibrationResult:
        """Perform grid search over parameter space.

        Args:
            param_grid: Dict mapping param names to candidate values
            folds: Cross-validation folds
            predict_fn: Prediction function

        Returns:
            Best CalibrationResult
        """
        best_result = None
        best_loss = float('inf')

        # Generate all combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())

        total_combinations = 1
        for values in param_values:
            total_combinations *= len(values)

        print(f"Grid search over {total_combinations} combinations...")

        for i, combination in enumerate(product(*param_values)):
            params = dict(zip(param_names, combination))

            result = self.evaluate_params(params, folds, predict_fn)

            if result.total_loss < best_loss:
                best_loss = result.total_loss
                best_result = result
                print(f"  [{i+1}/{total_combinations}] New best: loss={best_loss:.4f}")

            # Early stopping if perfect calibration achieved
            if best_loss < self.config.convergence_threshold:
                print(f"  Converged at iteration {i+1}")
                break

        return best_result

    def calibrate(
        self,
        df: pd.DataFrame,
        param_grid: Dict[str, List[Any]],
        predict_fn: Callable[[pd.DataFrame, Dict], pd.DataFrame],
    ) -> CalibrationResult:
        """Run full calibration pipeline.

        Args:
            df: Historical data DataFrame (sorted by time)
            param_grid: Dict mapping param names to candidate values
            predict_fn: Function that takes (df, params) and returns predictions DataFrame

        Returns:
            CalibrationResult with optimal parameters
        """
        print("=" * 60)
        print("PHYSICS PARAMETER CALIBRATION")
        print("=" * 60)

        # Create folds
        folds = self.create_time_series_folds(df)
        print(f"Created {len(folds)} time-series folds")

        # Run grid search
        result = self.grid_search(param_grid, folds, predict_fn)

        # Store results
        self._best_params = result.optimal_params
        self._calibration_history.append(result)

        # Print summary
        print("\n" + "=" * 60)
        print("CALIBRATION COMPLETE")
        print("=" * 60)
        print(f"Optimal Parameters: {result.optimal_params}")
        print(f"Calibration Error: {result.calibration_error:.4f}")
        print(f"Coverage (90%): {result.coverage_90:.1%}")
        print(f"Coverage (50%): {result.coverage_50:.1%}")
        print(f"Directional Accuracy: {result.directional_accuracy:.1%}")
        print(f"Sharpness: {result.sharpness:.4f}")
        print(f"Total Loss: {result.total_loss:.4f}")

        return result

    def get_best_params(self) -> Dict[str, Any]:
        """Get best calibrated parameters."""
        return self._best_params.copy()


def get_default_regime_param_grid() -> Dict[str, List[Any]]:
    """Get default grid for regime parameter calibration."""
    return {
        'funding_strength': [8.0, 10.0, 12.0, 15.0, 18.0],
        'imbalance_strength': [0.04, 0.06, 0.08, 0.10],
        'momentum_decay': [0.3, 0.5, 0.7],
        'jump_intensity': [0.01, 0.02, 0.05, 0.08],
        'mean_reversion': [0.1, 0.2, 0.35],
    }


def get_calibration_hyperparameters() -> List[Dict]:
    """Get hyperparameter definitions for improvement loop."""
    return [
        {
            'name': 'calibration_n_folds',
            'path': 'particle.calibration.n_folds',
            'current_value': 5,
            'candidates': [3, 4, 5, 7, 10],
            'variable_type': 'discrete',
        },
        {
            'name': 'calibration_train_pct',
            'path': 'particle.calibration.train_pct',
            'current_value': 0.7,
            'candidates': [0.6, 0.7, 0.8],
            'variable_type': 'continuous',
        },
        {
            'name': 'calibration_weight',
            'path': 'particle.calibration.calibration_weight',
            'current_value': 1.0,
            'candidates': [0.5, 0.8, 1.0, 1.2, 1.5],
            'variable_type': 'continuous',
        },
        {
            'name': 'calibration_sharpness_weight',
            'path': 'particle.calibration.sharpness_weight',
            'current_value': 0.5,
            'candidates': [0.2, 0.3, 0.5, 0.7, 1.0],
            'variable_type': 'continuous',
        },
        {
            'name': 'calibration_coverage_weight',
            'path': 'particle.calibration.coverage_weight',
            'current_value': 0.8,
            'candidates': [0.5, 0.8, 1.0, 1.2],
            'variable_type': 'continuous',
        },
        {
            'name': 'calibration_l2_regularization',
            'path': 'particle.calibration.l2_regularization',
            'current_value': 0.01,
            'candidates': [0.001, 0.01, 0.05, 0.1],
            'variable_type': 'continuous',
        },
    ]
