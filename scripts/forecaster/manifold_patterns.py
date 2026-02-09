"""
Manifold Patterns — Motif Discovery & Classical Pattern Mapping
================================================================
Detects chart patterns (triangles, wedges, flags, channels) via
unsupervised clustering on the event-bar feature manifold, then
maps discovered clusters to classical pattern families and estimates
conditional forward distributions.

Architecture:
  Stage A: Motif discovery (unsupervised)
    - GMM / HDBSCAN on per-bar feature vectors X_j
    - Sequence clustering via DTW or envelope statistics
    - Shapelet mining for discriminative subsequences

  Stage B: Classical pattern labeling
    - Map clusters to pattern families by envelope statistics
    - Track conditional forward distributions P(Δp | pattern, regime)

  Stage C: Forward distribution estimation
    - Per-cluster conditional mean shift, variance reduction, tail asymmetry
    - Valid pattern: shifts mean, reduces variance, or reveals tail asymmetry

Integration:
  - Produces features for the meta-learner
  - Acts as a regime-gating variable (pattern + field → state)

Usage:
    from scripts.forecaster.manifold_patterns import (
        ManifoldPatternDetector,
        ManifoldPatternModule,
    )
    detector = ManifoldPatternDetector()
    result = detector.detect(event_bar_sequence)
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .schemas import (
    Bar, MarketSnapshot, ModuleOutput, PredictionTargets, Regime,
)
from .particle_candles import (
    ParticleCandleBuilder,
    EventBar,
    EventBarSequence,
)


# ═══════════════════════════════════════════════════════════════
# PATTERN TAXONOMY
# ═══════════════════════════════════════════════════════════════

# Classical chart-pattern families with their typical envelope
# signatures and expected breakout directions.

PATTERN_FAMILIES = {
    "ascending_triangle": {
        "slope_high": (-0.001, 0.001),     # flat top
        "slope_low": (0.001, None),         # rising bottom
        "contraction": (None, -0.01),       # narrowing range
        "breakout_bias": "bullish",
        "reliability": 0.65,
    },
    "descending_triangle": {
        "slope_high": (None, -0.001),       # falling top
        "slope_low": (-0.001, 0.001),       # flat bottom
        "contraction": (None, -0.01),       # narrowing range
        "breakout_bias": "bearish",
        "reliability": 0.63,
    },
    "symmetrical_triangle": {
        "slope_high": (None, -0.0005),      # falling top
        "slope_low": (0.0005, None),        # rising bottom
        "contraction": (None, -0.02),       # converging
        "breakout_bias": "continuation",    # breaks in trend direction
        "reliability": 0.55,
    },
    "ascending_wedge": {
        "slope_high": (0.001, None),        # both rising
        "slope_low": (0.001, None),
        "contraction": (None, -0.005),      # narrowing
        "breakout_bias": "bearish",         # reversal pattern
        "reliability": 0.60,
    },
    "descending_wedge": {
        "slope_high": (None, -0.001),       # both falling
        "slope_low": (None, -0.001),
        "contraction": (None, -0.005),
        "breakout_bias": "bullish",         # reversal pattern
        "reliability": 0.60,
    },
    "bull_flag": {
        "slope_high": (None, -0.0005),      # slight downward drift
        "slope_low": (None, -0.0005),
        "contraction": (-0.005, 0.005),     # roughly parallel
        "breakout_bias": "bullish",
        "reliability": 0.67,
    },
    "bear_flag": {
        "slope_high": (0.0005, None),       # slight upward drift
        "slope_low": (0.0005, None),
        "contraction": (-0.005, 0.005),
        "breakout_bias": "bearish",
        "reliability": 0.65,
    },
    "channel_up": {
        "slope_high": (0.001, None),
        "slope_low": (0.001, None),
        "contraction": (-0.005, 0.005),     # parallel
        "breakout_bias": "bullish",
        "reliability": 0.55,
    },
    "channel_down": {
        "slope_high": (None, -0.001),
        "slope_low": (None, -0.001),
        "contraction": (-0.005, 0.005),
        "breakout_bias": "bearish",
        "reliability": 0.55,
    },
    "broadening_top": {
        "slope_high": (0.001, None),        # expanding range
        "slope_low": (None, -0.001),
        "contraction": (0.01, None),        # widening
        "breakout_bias": "bearish",
        "reliability": 0.50,
    },
    "no_pattern": {
        "slope_high": None,
        "slope_low": None,
        "contraction": None,
        "breakout_bias": "neutral",
        "reliability": 0.50,
    },
}


# ═══════════════════════════════════════════════════════════════
# PATTERN DETECTION RESULT
# ═══════════════════════════════════════════════════════════════

@dataclass
class PatternDetection:
    """Result of pattern detection on a bar sequence."""
    pattern_name: str = "no_pattern"
    confidence: float = 0.0
    breakout_bias: str = "neutral"     # "bullish", "bearish", "continuation", "neutral"
    reliability: float = 0.50
    match_score: float = 0.0           # how well envelope fits the pattern template

    # Forward distribution estimates
    fwd_mean_shift: float = 0.0        # expected return shift vs unconditional
    fwd_var_ratio: float = 1.0         # variance ratio (< 1 = reduced uncertainty)
    fwd_skew: float = 0.0             # tail asymmetry
    fwd_breakout_prob: float = 0.5     # P(breakout in bias direction)

    # Envelope statistics used for detection
    envelope: dict = field(default_factory=dict)

    # Cluster info (from Stage A)
    cluster_id: int = -1
    cluster_distance: float = 0.0      # distance to cluster centroid

    # Feature vector for meta-learner
    features: dict = field(default_factory=dict)


@dataclass
class ClusterModel:
    """Lightweight cluster model for motif discovery."""
    n_clusters: int = 0
    centroids: Optional[np.ndarray] = None    # (k, d)
    covariances: Optional[np.ndarray] = None  # (k, d, d) for GMM
    labels: Optional[np.ndarray] = None
    cluster_counts: dict = field(default_factory=dict)
    cluster_patterns: dict = field(default_factory=dict)  # cluster_id -> pattern_name
    cluster_forward_stats: dict = field(default_factory=dict)  # cluster_id -> {mean, std, skew}


# ═══════════════════════════════════════════════════════════════
# MANIFOLD PATTERN DETECTOR
# ═══════════════════════════════════════════════════════════════

class ManifoldPatternDetector:
    """
    Detects patterns via unsupervised clustering + classical mapping.

    Pipeline:
    1. Compute envelope metrics from event bar sequence
    2. Match envelope against classical pattern templates (Stage B)
    3. Cluster feature vectors with GMM (Stage A) if enough history
    4. Estimate forward distributions per cluster
    5. Combine template match + cluster assignment for final detection

    The detector maintains a history of detections and forward outcomes
    to improve cluster-to-pattern mapping over time.
    """

    def __init__(self,
                 n_clusters: int = 8,
                 min_history: int = 30,
                 min_bars_for_pattern: int = 10):
        self.n_clusters = n_clusters
        self.min_history = min_history
        self.min_bars = min_bars_for_pattern

        # Historical data for clustering
        self._feature_history: list[np.ndarray] = []
        self._forward_returns: list[float] = []
        self._pattern_history: list[str] = []
        self._cluster_model: Optional[ClusterModel] = None

    def detect(self,
               sequence: EventBarSequence,
               regime: str = "range",
               prior_trend: str = "flat") -> PatternDetection:
        """
        Detect pattern from an event bar sequence.

        Args:
            sequence: EventBarSequence with m bars
            regime: current market regime
            prior_trend: trend before this pattern window

        Returns:
            PatternDetection with pattern name, confidence, and features.
        """
        if sequence.n_bars < self.min_bars:
            return PatternDetection(
                pattern_name="no_pattern",
                confidence=0.0,
                features={"mp__insufficient_bars": 1.0},
            )

        # Compute envelope metrics
        env = sequence.envelope_metrics()
        if not env:
            return PatternDetection(
                pattern_name="no_pattern",
                confidence=0.0,
                features={"mp__no_envelope": 1.0},
            )

        # Stage B: Template matching (always available)
        template_match = self._match_templates(env, prior_trend)

        # Stage A: Cluster assignment (if model trained)
        cluster_result = self._assign_cluster(sequence, env)

        # Combine template + cluster
        detection = self._combine(template_match, cluster_result, env, regime)

        # Store for future learning
        feat_vec = self._extract_cluster_features(sequence, env)
        self._feature_history.append(feat_vec)
        self._pattern_history.append(detection.pattern_name)

        # Retrain clusters periodically
        if (len(self._feature_history) >= self.min_history
                and len(self._feature_history) % 10 == 0):
            self._fit_clusters()

        return detection

    def record_outcome(self, forward_return: float):
        """
        Record the actual forward return after a pattern detection.
        Used to update cluster forward distributions.
        """
        self._forward_returns.append(forward_return)

        # Update cluster forward stats
        if self._cluster_model and self._cluster_model.labels is not None:
            n = len(self._forward_returns)
            if n <= len(self._cluster_model.labels):
                self._update_forward_stats()

    def _match_templates(self,
                          env: dict,
                          prior_trend: str) -> PatternDetection:
        """
        Match envelope metrics against classical pattern templates.

        Returns the best-matching pattern with confidence score.
        """
        best_name = "no_pattern"
        best_score = 0.0
        best_family = PATTERN_FAMILIES["no_pattern"]

        slope_h = env.get("slope_high", 0.0)
        slope_l = env.get("slope_low", 0.0)
        contraction = env.get("contraction_rate", 0.0)
        avg_range = env.get("avg_range", 0.0)

        # Normalize slopes by average range for comparability
        norm = avg_range if avg_range > 0 else 1.0

        for name, family in PATTERN_FAMILIES.items():
            if name == "no_pattern":
                continue

            score = 0.0
            n_criteria = 0

            # Check slope_high
            if family["slope_high"] is not None:
                lo, hi = family["slope_high"]
                n_criteria += 1
                if _in_range(slope_h / norm, lo, hi):
                    score += 1.0
                else:
                    dist = _range_distance(slope_h / norm, lo, hi)
                    score += max(0, 1.0 - dist * 100)

            # Check slope_low
            if family["slope_low"] is not None:
                lo, hi = family["slope_low"]
                n_criteria += 1
                if _in_range(slope_l / norm, lo, hi):
                    score += 1.0
                else:
                    dist = _range_distance(slope_l / norm, lo, hi)
                    score += max(0, 1.0 - dist * 100)

            # Check contraction
            if family["contraction"] is not None:
                lo, hi = family["contraction"]
                n_criteria += 1
                if _in_range(contraction, lo, hi):
                    score += 1.0
                else:
                    dist = _range_distance(contraction, lo, hi)
                    score += max(0, 1.0 - dist * 50)

            # Normalize score
            if n_criteria > 0:
                score /= n_criteria

            # Continuation patterns get bonus if aligned with prior trend
            if family["breakout_bias"] == "continuation":
                if (prior_trend == "up" and slope_h > 0) or \
                   (prior_trend == "down" and slope_h < 0):
                    score *= 1.15

            if score > best_score:
                best_score = score
                best_name = name
                best_family = family

        # Require minimum score
        if best_score < 0.35:
            best_name = "no_pattern"
            best_family = PATTERN_FAMILIES["no_pattern"]
            best_score = 0.0

        # Determine breakout direction
        bias = best_family["breakout_bias"]
        if bias == "continuation":
            bias = "bullish" if prior_trend in ("up", "trend_up") else "bearish"

        return PatternDetection(
            pattern_name=best_name,
            confidence=min(0.85, best_score),
            breakout_bias=bias,
            reliability=best_family["reliability"],
            match_score=best_score,
            envelope=env,
        )

    def _assign_cluster(self,
                         sequence: EventBarSequence,
                         env: dict) -> Optional[tuple[int, float]]:
        """
        Assign the sequence to a cluster from the GMM model.

        Returns (cluster_id, distance_to_centroid) or None.
        """
        if self._cluster_model is None or self._cluster_model.centroids is None:
            return None

        feat_vec = self._extract_cluster_features(sequence, env)
        centroids = self._cluster_model.centroids

        # Compute distances to each centroid
        distances = np.linalg.norm(centroids - feat_vec, axis=1)
        cluster_id = int(np.argmin(distances))
        dist = float(distances[cluster_id])

        return (cluster_id, dist)

    def _combine(self,
                  template: PatternDetection,
                  cluster: Optional[tuple[int, float]],
                  env: dict,
                  regime: str) -> PatternDetection:
        """
        Combine template matching and cluster assignment.
        """
        detection = PatternDetection(
            pattern_name=template.pattern_name,
            confidence=template.confidence,
            breakout_bias=template.breakout_bias,
            reliability=template.reliability,
            match_score=template.match_score,
            envelope=env,
        )

        # Build feature dict for meta-learner
        features = {}
        features["mp__pattern_confidence"] = template.confidence
        features["mp__match_score"] = template.match_score
        features["mp__reliability"] = template.reliability

        # Encode breakout bias as numeric
        bias_map = {"bullish": 1.0, "bearish": -1.0, "neutral": 0.0}
        features["mp__breakout_bias"] = bias_map.get(
            template.breakout_bias, 0.0
        )

        # Envelope features
        for k, v in env.items():
            if isinstance(v, (int, float)):
                features[f"mp__env__{k}"] = float(v)

        # Pattern one-hot (top patterns only)
        for pname in ("ascending_triangle", "descending_triangle",
                       "symmetrical_triangle", "bull_flag", "bear_flag",
                       "ascending_wedge", "descending_wedge"):
            features[f"mp__is_{pname}"] = 1.0 if template.pattern_name == pname else 0.0

        if cluster is not None:
            cluster_id, dist = cluster
            detection.cluster_id = cluster_id
            detection.cluster_distance = dist
            features["mp__cluster_id"] = float(cluster_id)
            features["mp__cluster_distance"] = dist

            # Forward distribution from cluster history
            fwd = self._cluster_model.cluster_forward_stats.get(cluster_id, {})
            if fwd:
                detection.fwd_mean_shift = fwd.get("mean", 0.0)
                detection.fwd_var_ratio = fwd.get("var_ratio", 1.0)
                detection.fwd_skew = fwd.get("skew", 0.0)
                detection.fwd_breakout_prob = fwd.get("breakout_prob", 0.5)

                features["mp__fwd_mean_shift"] = detection.fwd_mean_shift
                features["mp__fwd_var_ratio"] = detection.fwd_var_ratio
                features["mp__fwd_skew"] = detection.fwd_skew
                features["mp__fwd_breakout_prob"] = detection.fwd_breakout_prob

        # Regime interaction
        features[f"mp__regime_{regime}"] = 1.0

        detection.features = features
        return detection

    def _extract_cluster_features(self,
                                   sequence: EventBarSequence,
                                   env: dict) -> np.ndarray:
        """
        Build a fixed-length feature vector for clustering.

        12 features:
          0: slope_high
          1: slope_low
          2: contraction_rate
          3: wick_asymmetry
          4: ofi_trend
          5: curvature
          6: body_convergence
          7: entropy_trend
          8: avg_range
          9: mean_body_ratio
         10: mean_realized_vol
         11: range_slope
        """
        bars = sequence.bars

        # From envelope
        f = np.zeros(12)
        f[0] = env.get("slope_high", 0.0)
        f[1] = env.get("slope_low", 0.0)
        f[2] = env.get("contraction_rate", 0.0)
        f[3] = env.get("wick_asymmetry", 0.0)
        f[4] = env.get("ofi_trend", 0.0)
        f[5] = env.get("curvature", 0.0)
        f[6] = env.get("body_convergence", 0.0)
        f[7] = env.get("entropy_trend", 0.0)
        f[8] = env.get("avg_range", 0.0)

        # From bar statistics
        if bars:
            f[9] = float(np.mean([b.body_ratio for b in bars]))
            rvols = [b.realized_vol for b in bars if b.realized_vol > 0]
            f[10] = float(np.mean(rvols)) if rvols else 0.0
            ranges = [b.range_pct for b in bars]
            if len(ranges) >= 3:
                t = np.arange(len(ranges), dtype=float)
                try:
                    f[11] = float(np.polyfit(t, ranges, 1)[0])
                except (np.linalg.LinAlgError, ValueError):
                    f[11] = 0.0

        return np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)

    def _fit_clusters(self):
        """
        Fit GMM clusters on accumulated feature history.

        Uses a simple GMM implementation (numpy-only) for VPS
        compatibility — no sklearn required for basic clustering.
        """
        if len(self._feature_history) < self.min_history:
            return

        X = np.array(self._feature_history)
        n, d = X.shape

        # Normalize features
        means = X.mean(axis=0)
        stds = X.std(axis=0) + 1e-8
        X_norm = (X - means) / stds

        k = min(self.n_clusters, n // 5)
        if k < 2:
            return

        # Simple k-means for cluster initialization
        centroids, labels = _kmeans(X_norm, k, max_iter=50)

        model = ClusterModel(
            n_clusters=k,
            centroids=centroids,
            labels=labels,
        )

        # Count samples per cluster
        for c in range(k):
            mask = labels == c
            model.cluster_counts[c] = int(np.sum(mask))

        # Map clusters to patterns based on centroid features
        for c in range(k):
            mask = labels == c
            if np.sum(mask) < 3:
                model.cluster_patterns[c] = "no_pattern"
                continue

            # Use centroid features to infer pattern
            centroid = centroids[c]
            # De-normalize
            real_centroid = centroid * stds + means

            env_approx = {
                "slope_high": float(real_centroid[0]),
                "slope_low": float(real_centroid[1]),
                "contraction_rate": float(real_centroid[2]),
                "wick_asymmetry": float(real_centroid[3]),
                "avg_range": float(real_centroid[8]) if d > 8 else 1.0,
            }
            det = self._match_templates(env_approx, "flat")
            model.cluster_patterns[c] = det.pattern_name

        # Forward distribution stats per cluster
        self._update_forward_stats_for_model(model)

        self._cluster_model = model

    def _update_forward_stats(self):
        """Update forward stats for the current cluster model."""
        if self._cluster_model is None:
            return
        self._update_forward_stats_for_model(self._cluster_model)

    def _update_forward_stats_for_model(self, model: ClusterModel):
        """Compute per-cluster forward distribution statistics."""
        labels = model.labels
        if labels is None:
            return

        n_labels = len(labels)
        n_returns = len(self._forward_returns)
        n = min(n_labels, n_returns)

        if n < 5:
            return

        all_rets = np.array(self._forward_returns[:n])
        unconditional_mean = float(np.mean(all_rets))
        unconditional_var = float(np.var(all_rets)) + 1e-12

        for c in range(model.n_clusters):
            mask = labels[:n] == c
            cluster_rets = all_rets[mask]

            if len(cluster_rets) < 3:
                continue

            c_mean = float(np.mean(cluster_rets))
            c_var = float(np.var(cluster_rets)) + 1e-12

            # Skewness
            m3 = float(np.mean((cluster_rets - c_mean) ** 3))
            c_std = math.sqrt(c_var)
            c_skew = m3 / (c_std ** 3) if c_std > 0 else 0.0

            # Breakout probability: fraction of positive returns
            breakout_prob = float(np.mean(cluster_rets > 0))

            model.cluster_forward_stats[c] = {
                "mean": c_mean - unconditional_mean,  # shift vs unconditional
                "var_ratio": c_var / unconditional_var,
                "skew": c_skew,
                "breakout_prob": breakout_prob,
                "n_samples": int(np.sum(mask)),
            }


# ═══════════════════════════════════════════════════════════════
# NUMPY-ONLY K-MEANS
# ═══════════════════════════════════════════════════════════════

def _kmeans(X: np.ndarray, k: int,
            max_iter: int = 50,
            seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """
    Simple k-means clustering (numpy only, no sklearn required).

    Args:
        X: (n, d) data matrix
        k: number of clusters
        max_iter: maximum iterations

    Returns:
        (centroids, labels) where centroids is (k, d) and labels is (n,)
    """
    rng = np.random.RandomState(seed)
    n, d = X.shape

    # Initialize centroids with k-means++
    centroids = np.zeros((k, d))
    idx = rng.randint(0, n)
    centroids[0] = X[idx]

    for i in range(1, k):
        dists = np.min([
            np.sum((X - centroids[j]) ** 2, axis=1)
            for j in range(i)
        ], axis=0)
        probs = dists / (dists.sum() + 1e-12)
        idx = rng.choice(n, p=probs)
        centroids[i] = X[idx]

    labels = np.zeros(n, dtype=int)

    for _ in range(max_iter):
        # Assign labels
        distances = np.array([
            np.sum((X - centroids[j]) ** 2, axis=1)
            for j in range(k)
        ]).T  # (n, k)
        new_labels = np.argmin(distances, axis=1)

        if np.array_equal(new_labels, labels):
            break
        labels = new_labels

        # Update centroids
        for j in range(k):
            mask = labels == j
            if np.any(mask):
                centroids[j] = X[mask].mean(axis=0)

    return centroids, labels


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def _in_range(value: float,
              lo: Optional[float],
              hi: Optional[float]) -> bool:
    """Check if value is in [lo, hi] with None = unbounded."""
    if lo is not None and value < lo:
        return False
    if hi is not None and value > hi:
        return False
    return True


def _range_distance(value: float,
                     lo: Optional[float],
                     hi: Optional[float]) -> float:
    """Distance from value to nearest edge of [lo, hi]."""
    if lo is not None and value < lo:
        return lo - value
    if hi is not None and value > hi:
        return value - hi
    return 0.0


# ═══════════════════════════════════════════════════════════════
# MANIFOLD PATTERN FORECASTER MODULE
# ═══════════════════════════════════════════════════════════════

class ManifoldPatternModule:
    """
    Forecaster module: manifold-based pattern detection.

    Converts bars to event-quantized bars, runs pattern detection,
    and produces direction/confidence signals from detected patterns.

    Implements the standard predict(snapshot, horizon_hours) interface.
    """

    name = "manifold_pattern"
    trusted_regimes = {
        Regime.TREND_UP, Regime.TREND_DOWN,
        Regime.RANGE, Regime.POST_JUMP,
        Regime.VOL_EXPANSION,
    }
    failure_modes = ["insufficient_bars", "no_pattern"]

    def __init__(self,
                 window: int = 30,
                 candle_rule: str = "adaptive"):
        self.window = window
        self.builder = ParticleCandleBuilder(rule=candle_rule)
        self.detector = ManifoldPatternDetector(min_bars_for_pattern=8)

    def predict(self, snap: MarketSnapshot,
                horizon_hours: float = 24.0) -> ModuleOutput:
        """
        Detect patterns and produce prediction signals.
        """
        t0 = time.time()
        features = {}
        targets = PredictionTargets()

        bars = snap.bars_1h
        if len(bars) < 20:
            return ModuleOutput(
                module_name=self.name,
                targets=targets,
                confidence=0.0,
                features=features,
                metadata={"reason": "insufficient_bars"},
                elapsed_ms=(time.time() - t0) * 1000,
            )

        # Build event bars
        try:
            event_bars = self.builder.from_ohlcv(bars)
        except Exception:
            event_bars = []

        if len(event_bars) < 10:
            return ModuleOutput(
                module_name=self.name,
                targets=targets,
                confidence=0.0,
                features=features,
                metadata={"reason": "too_few_event_bars"},
                elapsed_ms=(time.time() - t0) * 1000,
            )

        # Create sequence from recent window
        w = min(self.window, len(event_bars))
        sequence = EventBarSequence(
            bars=event_bars[-w:],
            symbol=snap.symbol,
        )

        # Determine prior trend from earlier bars
        prior_trend = self._infer_prior_trend(event_bars, w)

        # Detect pattern
        detection = self.detector.detect(
            sequence,
            regime="range",  # Will be updated by engine
            prior_trend=prior_trend,
        )

        # Copy detection features to module features
        features.update(detection.features)

        # Direction signal from pattern
        bias = detection.breakout_bias
        reliability = detection.reliability
        confidence = detection.confidence

        if bias == "bullish" and confidence > 0.3:
            dir_prob = 0.50 + reliability * confidence * 0.20
        elif bias == "bearish" and confidence > 0.3:
            dir_prob = 0.50 - reliability * confidence * 0.20
        else:
            dir_prob = 0.50

        # Incorporate forward distribution if available
        if detection.fwd_mean_shift != 0.0:
            # Shift direction prob based on cluster forward stats
            shift = detection.fwd_mean_shift * 10  # Scale up
            dir_prob += max(-0.10, min(0.10, shift))

        dir_prob = max(0.30, min(0.70, dir_prob))
        targets.direction_prob = dir_prob

        # Expected return from pattern + forward stats
        price = snap.current_price
        if price > 0 and detection.pattern_name != "no_pattern":
            env = detection.envelope
            slope_mid = env.get("slope_mid", 0.0)
            targets.expected_return = slope_mid / price * horizon_hours
            targets.expected_return = max(-0.03, min(0.03, targets.expected_return))

            # Adjust by forward mean shift
            if detection.fwd_mean_shift != 0.0:
                targets.expected_return += detection.fwd_mean_shift * 0.5

        # Volatility: reduced if pattern provides structure
        if detection.fwd_var_ratio < 1.0 and detection.fwd_var_ratio > 0:
            # Pattern reduces uncertainty
            targets.volatility_forecast *= detection.fwd_var_ratio

        # Module confidence
        if detection.pattern_name == "no_pattern":
            mod_confidence = 0.1
        else:
            mod_confidence = min(0.70, confidence * reliability)

        elapsed = (time.time() - t0) * 1000

        return ModuleOutput(
            module_name=self.name,
            targets=targets,
            confidence=mod_confidence,
            features=features,
            metadata={
                "pattern": detection.pattern_name,
                "breakout_bias": detection.breakout_bias,
                "match_score": round(detection.match_score, 4),
                "cluster_id": detection.cluster_id,
                "fwd_mean_shift": round(detection.fwd_mean_shift, 6),
                "fwd_var_ratio": round(detection.fwd_var_ratio, 4),
                "reliability": round(detection.reliability, 4),
            },
            elapsed_ms=elapsed,
        )

    def _infer_prior_trend(self,
                            event_bars: list[EventBar],
                            window: int) -> str:
        """Infer prior trend from bars before the pattern window."""
        pre_window = event_bars[:-window] if len(event_bars) > window else []
        if len(pre_window) < 3:
            return "flat"

        # Look at the last 5 pre-window bars
        recent = pre_window[-5:]
        closes = [b.close for b in recent]
        if len(closes) < 2:
            return "flat"

        ret = (closes[-1] - closes[0]) / closes[0] if closes[0] > 0 else 0.0
        if ret > 0.01:
            return "up"
        elif ret < -0.01:
            return "down"
        return "flat"
