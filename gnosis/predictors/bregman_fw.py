"""Bregman Projection via Adaptive Fully-Corrective Frank-Wolfe Predictor.

Implementation of Algorithm 2 (ProjectFW) for quantile regression with:
- Bregman divergence-based loss (proper quantile loss)
- Frank-Wolfe optimization over constrained polytope
- Adaptive contraction for convergence
- Integer programming for descent direction

Reference: Bregman Projection via Adaptive Fully-Corrective Frank-Wolfe
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


@dataclass
class FWConfig:
    """Configuration for Frank-Wolfe optimization."""

    # Convergence parameters
    alpha: float = 0.1  # Approximation ratio (0, 1)
    epsilon_0: float = 0.5  # Initial contraction
    epsilon_D: float = 1e-6  # Convergence threshold
    max_iterations: int = 100

    # Quantile regression parameters
    quantiles: List[float] = field(default_factory=lambda: [0.05, 0.50, 0.95])
    l2_reg: float = 1e-4

    # Constraint bounds
    prediction_bounds: Tuple[float, float] = (-0.10, 0.10)  # Max ±10% return

    # Adaptive learning
    learning_rate: float = 0.01
    momentum: float = 0.9


class BregmanFWPredictor:
    """Quantile predictor using Bregman-Frank-Wolfe optimization.

    Implements the ProjectFW algorithm for robust quantile regression:
    - F(μ) = R̂_σ(μ) - θ·μ + C_σ(θ)  (objective function)
    - Adaptive contraction ε_t for convergence
    - Polytope constraints for bounded predictions

    Attributes:
        config: FW optimization configuration
        theta: Current state vector (learned parameters)
        active_vertices: Set of active vertices in polytope
    """

    def __init__(self, config: Optional[FWConfig] = None):
        """Initialize the Bregman-FW predictor.

        Args:
            config: Optimization configuration
        """
        self.config = config or FWConfig()

        # Model state
        self.theta: Optional[NDArray] = None
        self.active_vertices: List[NDArray] = []
        self.interior_point: Optional[NDArray] = None

        # Per-quantile models
        self.models: Dict[float, Dict] = {}

        # Training history for diagnostics
        self.history: List[Dict] = []

    def _pinball_loss(
        self,
        y_true: NDArray,
        y_pred: NDArray,
        quantile: float
    ) -> float:
        """Compute pinball (quantile) loss.

        L_q(y, ŷ) = q(y - ŷ)⁺ + (1-q)(ŷ - y)⁺

        Args:
            y_true: True values
            y_pred: Predicted values
            quantile: Quantile level (0, 1)

        Returns:
            Mean pinball loss
        """
        residual = y_true - y_pred
        return np.mean(
            quantile * np.maximum(residual, 0) +
            (1 - quantile) * np.maximum(-residual, 0)
        )

    def _bregman_divergence(
        self,
        mu: NDArray,
        theta: NDArray,
        sigma: NDArray
    ) -> float:
        """Compute Bregman divergence D_σ(μ || θ).

        For quantile regression, we use the pinball-based divergence:
        D_σ(μ || θ) = Σ_i σ_i * |μ_i - θ_i|

        Args:
            mu: Current point
            theta: Reference point
            sigma: Partial outcome weights

        Returns:
            Bregman divergence value
        """
        return np.sum(sigma * np.abs(mu - theta))

    def _init_fw(
        self,
        X: NDArray,
        y: NDArray,
        quantile: float
    ) -> Tuple[NDArray, List[NDArray], NDArray]:
        """Initialize interior point, active set, and partial outcome.

        InitFW(σ, A, b) from Algorithm 2.

        Args:
            X: Feature matrix
            y: Target vector
            quantile: Quantile level

        Returns:
            Tuple of (interior_point u, initial_vertices Z_0, weights σ)
        """
        n_features = X.shape[1]

        # Interior point: center of constraint polytope
        lb, ub = self.config.prediction_bounds
        u = np.zeros(n_features) + (lb + ub) / 2

        # Initial vertex: simple least squares solution (clipped)
        try:
            # Ridge regression as initial guess
            XtX = X.T @ X + self.config.l2_reg * np.eye(n_features)
            Xty = X.T @ y
            theta_init = np.linalg.solve(XtX, Xty)
            theta_init = np.clip(theta_init, lb, ub)
        except np.linalg.LinAlgError:
            theta_init = u.copy()

        Z_0 = [theta_init]

        # Partial outcome weights (quantile-dependent)
        sigma = np.full(len(y), quantile)

        return u, Z_0, sigma

    def _objective_function(
        self,
        mu: NDArray,
        X: NDArray,
        y: NDArray,
        theta: NDArray,
        sigma: NDArray,
        quantile: float
    ) -> float:
        """Compute objective F(μ) = R̂_σ(μ) - θ·μ + C_σ(θ).

        Args:
            mu: Current point
            X: Feature matrix
            y: Target values
            theta: State vector
            sigma: Partial outcome
            quantile: Quantile level

        Returns:
            Objective value
        """
        # Empirical risk R̂_σ(μ)
        y_pred = X @ mu
        risk = self._pinball_loss(y, y_pred, quantile)

        # Linear term
        linear = -np.dot(theta, mu)

        # Cost function C_σ(θ) - regularization
        cost = self.config.l2_reg * np.sum(theta ** 2)

        return risk + linear + cost

    def _gradient(
        self,
        mu: NDArray,
        X: NDArray,
        y: NDArray,
        quantile: float
    ) -> NDArray:
        """Compute gradient ∇R̂_σ(μ).

        For pinball loss: ∇L = -X.T @ (q - 1{y < Xμ})

        Args:
            mu: Current point
            X: Feature matrix
            y: Target values
            quantile: Quantile level

        Returns:
            Gradient vector
        """
        y_pred = X @ mu
        residual = y - y_pred

        # Subgradient of pinball loss
        indicator = (residual < 0).astype(float)
        subgrad = quantile - indicator

        return -X.T @ subgrad / len(y)

    def _find_descent_vertex(
        self,
        gradient: NDArray,
        theta: NDArray
    ) -> NDArray:
        """Find descent vertex via IP solver.

        z_t = argmin_{z ∈ Z_σ} (θ_t - θ) · z

        For box constraints, this reduces to:
        z_i = lb if (θ_t - θ)_i > 0 else ub

        Args:
            gradient: Current gradient (θ_t - θ)
            theta: Reference state

        Returns:
            Descent vertex
        """
        lb, ub = self.config.prediction_bounds

        # For box polytope: pick extreme point in descent direction
        z = np.where(gradient > 0, lb, ub)

        return z

    def _fw_gap(
        self,
        mu: NDArray,
        z: NDArray,
        theta_t: NDArray,
        theta: NDArray
    ) -> float:
        """Compute Frank-Wolfe gap g(μ_t) = (θ_t - θ) · (μ_t - z_t).

        Args:
            mu: Current iterate
            z: Descent vertex
            theta_t: Current gradient
            theta: Reference state

        Returns:
            FW gap value
        """
        direction = theta_t - theta
        return np.dot(direction, mu - z)

    def _line_search(
        self,
        mu: NDArray,
        z: NDArray,
        X: NDArray,
        y: NDArray,
        theta: NDArray,
        sigma: NDArray,
        quantile: float
    ) -> float:
        """Line search for optimal step size.

        Find γ* = argmin_{γ ∈ [0,1]} F(μ + γ(z - μ))

        Args:
            mu: Current point
            z: Descent direction endpoint
            X, y: Data
            theta, sigma: Algorithm state
            quantile: Quantile level

        Returns:
            Optimal step size
        """
        best_gamma = 0.0
        best_obj = self._objective_function(mu, X, y, theta, sigma, quantile)

        # Grid search (could use golden section for efficiency)
        for gamma in np.linspace(0, 1, 20):
            mu_new = mu + gamma * (z - mu)
            obj = self._objective_function(mu_new, X, y, theta, sigma, quantile)
            if obj < best_obj:
                best_obj = obj
                best_gamma = gamma

        return best_gamma

    def _project_fw(
        self,
        X: NDArray,
        y: NDArray,
        quantile: float,
        theta_init: Optional[NDArray] = None
    ) -> Tuple[NDArray, NDArray]:
        """Main ProjectFW algorithm (Algorithm 2).

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,)
            quantile: Quantile level
            theta_init: Initial state (optional)

        Returns:
            Tuple of (extended_outcome σ̂, optimal_state θ̂)
        """
        n_features = X.shape[1]

        # Initialize
        u, Z_t, sigma = self._init_fw(X, y, quantile)
        theta = theta_init if theta_init is not None else np.zeros(n_features)

        # Initial contraction
        epsilon_t = self.config.epsilon_0

        # Start from convex combination of initial vertices
        mu_t = Z_t[0].copy()

        # Track best iterate
        best_t = 0
        best_score = float('inf')
        mu_history = [mu_t.copy()]

        for t in range(1, self.config.max_iterations + 1):
            # Contract active set toward interior
            Z_prime = [(1 - epsilon_t) * z + epsilon_t * u for z in Z_t]

            # Find μ_t in conv(Z')
            mu_t = self._optimize_over_convex_hull(Z_prime, X, y, theta, sigma, quantile)

            # Compute gradient θ_t = ∇R̂_σ(μ_t)
            theta_t = self._gradient(mu_t, X, y, quantile)

            # Find descent vertex (IP solver)
            z_t = self._find_descent_vertex(theta_t - theta, theta)

            # Update active set
            Z_t.append(z_t)

            # Compute FW gap
            gap = self._fw_gap(mu_t, z_t, theta_t, theta)
            obj = self._objective_function(mu_t, X, y, theta, sigma, quantile)

            # Update best iterate
            score = obj - gap
            if score < best_score:
                best_score = score
                best_t = t

            mu_history.append(mu_t.copy())

            # Record history
            self.history.append({
                'iteration': t,
                'objective': obj,
                'gap': gap,
                'epsilon': epsilon_t,
                'quantile': quantile
            })

            # Check stopping conditions
            if gap <= (1 - self.config.alpha) * obj:
                break
            if obj <= self.config.epsilon_D:
                break

            # Adapt contraction
            g_u = np.dot(theta_t - theta, mu_t - u)
            if g_u < 0 and gap / (-4 * g_u) < epsilon_t:
                epsilon_t = min(gap / (-4 * g_u), epsilon_t / 2)

        # Return result based on gap condition
        if gap <= obj:
            theta_hat = theta_t
        else:
            theta_hat = theta

        sigma_hat = sigma  # Extended partial outcome

        return sigma_hat, mu_history[best_t]

    def _optimize_over_convex_hull(
        self,
        vertices: List[NDArray],
        X: NDArray,
        y: NDArray,
        theta: NDArray,
        sigma: NDArray,
        quantile: float
    ) -> NDArray:
        """Find μ = argmin_{μ ∈ conv(Z')} F(μ).

        Uses gradient descent with projection onto simplex.

        Args:
            vertices: List of vertices defining convex hull
            X, y: Data
            theta, sigma: Algorithm state
            quantile: Quantile level

        Returns:
            Optimal point in convex hull
        """
        n_vertices = len(vertices)
        if n_vertices == 1:
            return vertices[0].copy()

        # Parameterize as μ = Σ λ_i v_i where λ is on simplex
        V = np.column_stack(vertices)  # (n_features, n_vertices)

        # Initialize uniform weights
        lam = np.ones(n_vertices) / n_vertices

        # Gradient descent on simplex
        lr = self.config.learning_rate
        for _ in range(50):
            mu = V @ lam
            grad_mu = self._gradient(mu, X, y, quantile)

            # Gradient w.r.t. lambda
            grad_lam = V.T @ grad_mu

            # Projected gradient step
            lam_new = lam - lr * grad_lam
            lam_new = self._project_simplex(lam_new)

            if np.linalg.norm(lam_new - lam) < 1e-8:
                break
            lam = lam_new

        return V @ lam

    def _project_simplex(self, v: NDArray) -> NDArray:
        """Project vector onto probability simplex.

        Args:
            v: Input vector

        Returns:
            Projected vector on simplex
        """
        n = len(v)
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u)
        rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1))[0][-1]
        theta = (cssv[rho] - 1) / (rho + 1.0)
        return np.maximum(v - theta, 0)

    def fit(self, X: NDArray, y: NDArray) -> "BregmanFWPredictor":
        """Fit the predictor using Frank-Wolfe optimization.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,)

        Returns:
            Self
        """
        self.history = []

        for q in self.config.quantiles:
            sigma_hat, theta_hat = self._project_fw(X, y, q)

            self.models[q] = {
                'theta': theta_hat,
                'sigma': sigma_hat,
                'quantile': q
            }

        return self

    def predict(self, X: NDArray) -> Dict[str, NDArray]:
        """Generate quantile predictions.

        Args:
            X: Feature matrix

        Returns:
            Dictionary with q05, q50, q95 predictions
        """
        predictions = {}

        for q in self.config.quantiles:
            if q not in self.models:
                raise ValueError(f"Model not fitted for quantile {q}")

            theta = self.models[q]['theta']
            y_pred = X @ theta

            # Clip to bounds
            lb, ub = self.config.prediction_bounds
            y_pred = np.clip(y_pred, lb, ub)

            # Map quantile to key
            key = f"q{int(q * 100):02d}"
            predictions[key] = y_pred

        return predictions

    def get_convergence_stats(self) -> Dict:
        """Get convergence statistics from training.

        Returns:
            Dictionary with convergence metrics
        """
        if not self.history:
            return {}

        return {
            'total_iterations': len(self.history),
            'final_objective': self.history[-1]['objective'],
            'final_gap': self.history[-1]['gap'],
            'convergence_rate': self._estimate_convergence_rate()
        }

    def _estimate_convergence_rate(self) -> float:
        """Estimate convergence rate from history."""
        if len(self.history) < 10:
            return 0.0

        objectives = [h['objective'] for h in self.history[-10:]]
        if objectives[0] == 0:
            return 0.0

        # Estimate linear convergence rate
        ratios = [objectives[i+1] / objectives[i]
                  for i in range(len(objectives)-1)
                  if objectives[i] > 0]

        return np.mean(ratios) if ratios else 0.0


class GnosisCompatibleFWPredictor:
    """Wrapper to make BregmanFWPredictor compatible with Gnosis interface."""

    def __init__(self, config: Optional[Dict] = None):
        """Initialize with Gnosis-style config.

        Args:
            config: Dictionary config (converted to FWConfig)
        """
        self.config = config or {}

        fw_config = FWConfig(
            alpha=self.config.get('alpha', 0.1),
            epsilon_0=self.config.get('epsilon_0', 0.5),
            epsilon_D=self.config.get('epsilon_D', 1e-6),
            max_iterations=self.config.get('max_iterations', 100),
            l2_reg=self.config.get('l2_reg', 1e-4),
            quantiles=[0.05, 0.50, 0.95]
        )

        self.predictor = BregmanFWPredictor(fw_config)
        self.sigma_scale = self.config.get('sigma_scale', 1.0)

    def fit(self, X: NDArray, y: NDArray) -> "GnosisCompatibleFWPredictor":
        """Fit the predictor.

        Args:
            X: Feature matrix
            y: Target vector

        Returns:
            Self
        """
        self.predictor.fit(X, y)
        return self

    def predict(self, X: NDArray) -> Tuple[NDArray, NDArray, NDArray]:
        """Generate predictions in Gnosis format.

        Args:
            X: Feature matrix

        Returns:
            Tuple of (q05, q50, q95) arrays
        """
        preds = self.predictor.predict(X)

        q05 = preds['q05']
        q50 = preds['q50']
        q95 = preds['q95']

        # Apply sigma_scale to widen intervals
        width = (q95 - q05) / 2
        q05_scaled = q50 - width * self.sigma_scale
        q95_scaled = q50 + width * self.sigma_scale

        return q05_scaled, q50, q95_scaled
