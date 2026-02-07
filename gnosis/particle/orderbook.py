"""Multi-Level Order Book Depth Analysis.

Analyzes order book depth at multiple price levels to compute:
- Depth imbalance at 0.5%, 1%, 2% from mid price
- Weighted pressure combining multiple levels
- Large order detection (icebergs, walls)
- Order flow momentum

All parameters are exposed as hyperparameters for ML tuning.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import deque


@dataclass
class OrderBookConfig:
    """Hyperparameters for order book analysis.

    All parameters can be tuned by the improvement loop.
    """
    # Depth levels to analyze (as % from mid price)
    level_1_pct: float = 0.005  # 0.5%
    level_2_pct: float = 0.01   # 1%
    level_3_pct: float = 0.02   # 2%
    level_4_pct: float = 0.05   # 5%

    # Weights for combining levels (closer = more weight)
    weight_level_1: float = 0.40
    weight_level_2: float = 0.30
    weight_level_3: float = 0.20
    weight_level_4: float = 0.10

    # Force calculation parameters
    imbalance_strength: float = 0.08  # Base strength of imbalance force
    depth_strength: float = 0.05  # Strength of depth-based force

    # Regime multipliers
    strength_trending: float = 0.7
    strength_ranging: float = 1.3
    strength_volatile: float = 1.0

    # Large order detection
    large_order_std_mult: float = 3.0  # Orders > mean + N*std are "large"
    wall_detection_mult: float = 5.0  # Orders > mean + N*std are "walls"
    iceberg_detection_enabled: bool = True

    # Order flow momentum
    flow_momentum_window: int = 20  # Snapshots for momentum calculation
    flow_ema_alpha: float = 0.2

    # Gravity model parameters
    gravity_constant: float = 1e-8
    gravity_distance_power: float = 2.0  # F ~ 1/r^n

    # Smoothing
    snapshot_smoothing: int = 3  # Average over N snapshots

    # Force clamping
    max_imbalance_force: float = 0.015  # 1.5% max
    max_depth_force: float = 0.01


@dataclass
class OrderBookSnapshot:
    """A single order book snapshot."""
    timestamp: float
    mid_price: float
    bids: np.ndarray  # Shape (N, 2) with [price, volume]
    asks: np.ndarray  # Shape (N, 2) with [price, volume]

    @property
    def spread(self) -> float:
        if len(self.bids) > 0 and len(self.asks) > 0:
            return self.asks[0, 0] - self.bids[0, 0]
        return 0.0

    @property
    def spread_bps(self) -> float:
        if self.mid_price > 0:
            return self.spread / self.mid_price * 10000
        return 0.0


class MultiLevelOrderBookAnalyzer:
    """Analyzes order book depth at multiple price levels.

    The order book creates steering forces through:
    1. Depth imbalance: More bids than asks = upward pressure
    2. Large orders: Walls and icebergs act as support/resistance
    3. Gravity: Price attracted toward deep liquidity
    4. Flow momentum: Changes in depth signal intent

    Multi-level analysis captures both immediate and broader market structure.
    """

    def __init__(self, config: Optional[OrderBookConfig] = None):
        """Initialize analyzer with config.

        Args:
            config: OrderBookConfig with hyperparameters
        """
        self.config = config or OrderBookConfig()
        self._snapshot_history: deque = deque(maxlen=100)
        self._flow_history: deque = deque(maxlen=self.config.flow_momentum_window)
        self._ema_imbalance: Optional[float] = None

    def process_snapshot(
        self,
        bids: List[List[float]],  # [[price, volume], ...]
        asks: List[List[float]],
        timestamp: Optional[float] = None,
    ) -> OrderBookSnapshot:
        """Process a raw order book snapshot.

        Args:
            bids: List of [price, volume] for bids
            asks: List of [price, volume] for asks
            timestamp: Unix timestamp

        Returns:
            OrderBookSnapshot object
        """
        import time

        bids_arr = np.array(bids) if bids else np.array([]).reshape(0, 2)
        asks_arr = np.array(asks) if asks else np.array([]).reshape(0, 2)

        # Sort bids descending, asks ascending
        if len(bids_arr) > 0:
            bids_arr = bids_arr[bids_arr[:, 0].argsort()[::-1]]
        if len(asks_arr) > 0:
            asks_arr = asks_arr[asks_arr[:, 0].argsort()]

        # Calculate mid price
        if len(bids_arr) > 0 and len(asks_arr) > 0:
            mid_price = (bids_arr[0, 0] + asks_arr[0, 0]) / 2
        elif len(bids_arr) > 0:
            mid_price = bids_arr[0, 0]
        elif len(asks_arr) > 0:
            mid_price = asks_arr[0, 0]
        else:
            mid_price = 0.0

        snapshot = OrderBookSnapshot(
            timestamp=timestamp or time.time(),
            mid_price=mid_price,
            bids=bids_arr,
            asks=asks_arr,
        )

        self._snapshot_history.append(snapshot)

        return snapshot

    def compute_depth_at_level(
        self,
        snapshot: OrderBookSnapshot,
        level_pct: float,
    ) -> Tuple[float, float]:
        """Compute bid and ask depth at a specific level.

        Args:
            snapshot: Order book snapshot
            level_pct: Distance from mid as percentage (e.g., 0.01 = 1%)

        Returns:
            Tuple of (bid_depth, ask_depth)
        """
        mid = snapshot.mid_price
        if mid <= 0:
            return 0.0, 0.0

        bid_threshold = mid * (1 - level_pct)
        ask_threshold = mid * (1 + level_pct)

        # Sum volume within threshold
        bid_depth = 0.0
        if len(snapshot.bids) > 0:
            mask = snapshot.bids[:, 0] >= bid_threshold
            bid_depth = snapshot.bids[mask, 1].sum()

        ask_depth = 0.0
        if len(snapshot.asks) > 0:
            mask = snapshot.asks[:, 0] <= ask_threshold
            ask_depth = snapshot.asks[mask, 1].sum()

        return bid_depth, ask_depth

    def compute_level_imbalance(
        self,
        bid_depth: float,
        ask_depth: float,
    ) -> float:
        """Compute normalized imbalance at a level.

        Returns value in [-1, 1] where:
        - 1 = all bids, no asks
        - -1 = all asks, no bids
        - 0 = balanced
        """
        total = bid_depth + ask_depth
        if total == 0:
            return 0.0
        return (bid_depth - ask_depth) / total

    def compute_multi_level_imbalance(
        self,
        snapshot: OrderBookSnapshot,
    ) -> Dict[str, float]:
        """Compute imbalance at multiple levels.

        Args:
            snapshot: Order book snapshot

        Returns:
            Dict with imbalances and weighted average
        """
        levels = [
            (self.config.level_1_pct, self.config.weight_level_1),
            (self.config.level_2_pct, self.config.weight_level_2),
            (self.config.level_3_pct, self.config.weight_level_3),
            (self.config.level_4_pct, self.config.weight_level_4),
        ]

        imbalances = {}
        weighted_sum = 0.0
        total_weight = 0.0

        for i, (level_pct, weight) in enumerate(levels, 1):
            bid_depth, ask_depth = self.compute_depth_at_level(snapshot, level_pct)
            imbalance = self.compute_level_imbalance(bid_depth, ask_depth)

            imbalances[f'imbalance_level_{i}'] = imbalance
            imbalances[f'bid_depth_level_{i}'] = bid_depth
            imbalances[f'ask_depth_level_{i}'] = ask_depth

            weighted_sum += weight * imbalance
            total_weight += weight

        # Weighted average imbalance
        if total_weight > 0:
            weighted_imbalance = weighted_sum / total_weight
        else:
            weighted_imbalance = 0.0

        imbalances['weighted_imbalance'] = weighted_imbalance

        return imbalances

    def detect_large_orders(
        self,
        snapshot: OrderBookSnapshot,
    ) -> Dict[str, float]:
        """Detect large orders (walls, icebergs).

        Args:
            snapshot: Order book snapshot

        Returns:
            Dict with large order metrics
        """
        result = {
            'large_bid_volume': 0.0,
            'large_ask_volume': 0.0,
            'wall_bid_price': 0.0,
            'wall_ask_price': 0.0,
            'wall_bid_volume': 0.0,
            'wall_ask_volume': 0.0,
        }

        if len(snapshot.bids) < 5 or len(snapshot.asks) < 5:
            return result

        # Calculate thresholds
        all_volumes = np.concatenate([snapshot.bids[:, 1], snapshot.asks[:, 1]])
        mean_vol = np.mean(all_volumes)
        std_vol = np.std(all_volumes)

        large_threshold = mean_vol + self.config.large_order_std_mult * std_vol
        wall_threshold = mean_vol + self.config.wall_detection_mult * std_vol

        # Detect large bids
        large_bid_mask = snapshot.bids[:, 1] > large_threshold
        if np.any(large_bid_mask):
            result['large_bid_volume'] = snapshot.bids[large_bid_mask, 1].sum()

        # Detect large asks
        large_ask_mask = snapshot.asks[:, 1] > large_threshold
        if np.any(large_ask_mask):
            result['large_ask_volume'] = snapshot.asks[large_ask_mask, 1].sum()

        # Detect walls (very large orders)
        wall_bid_mask = snapshot.bids[:, 1] > wall_threshold
        if np.any(wall_bid_mask):
            wall_idx = np.argmax(snapshot.bids[wall_bid_mask, 1])
            result['wall_bid_price'] = snapshot.bids[wall_bid_mask][wall_idx, 0]
            result['wall_bid_volume'] = snapshot.bids[wall_bid_mask][wall_idx, 1]

        wall_ask_mask = snapshot.asks[:, 1] > wall_threshold
        if np.any(wall_ask_mask):
            wall_idx = np.argmax(snapshot.asks[wall_ask_mask, 1])
            result['wall_ask_price'] = snapshot.asks[wall_ask_mask][wall_idx, 0]
            result['wall_ask_volume'] = snapshot.asks[wall_ask_mask][wall_idx, 1]

        return result

    def compute_flow_momentum(self) -> Dict[str, float]:
        """Compute order flow momentum from snapshot history.

        Measures how depth imbalance is changing over time.

        Returns:
            Dict with flow momentum metrics
        """
        if len(self._snapshot_history) < 3:
            return {
                'flow_momentum': 0.0,
                'flow_acceleration': 0.0,
            }

        # Compute imbalances for recent snapshots
        recent_imbalances = []
        for snapshot in list(self._snapshot_history)[-self.config.flow_momentum_window:]:
            imb = self.compute_multi_level_imbalance(snapshot)
            recent_imbalances.append(imb['weighted_imbalance'])

        if len(recent_imbalances) < 3:
            return {
                'flow_momentum': 0.0,
                'flow_acceleration': 0.0,
            }

        imb_series = np.array(recent_imbalances)

        # Momentum: change in imbalance
        momentum = imb_series[-1] - imb_series[0]

        # Acceleration: rate of change of momentum
        first_half = imb_series[:len(imb_series)//2].mean()
        second_half = imb_series[len(imb_series)//2:].mean()
        acceleration = second_half - first_half

        return {
            'flow_momentum': momentum,
            'flow_acceleration': acceleration,
        }

    def compute_gravity_force(
        self,
        snapshot: OrderBookSnapshot,
    ) -> float:
        """Compute gravitational attraction to deep liquidity.

        Large orders act as masses that attract price.

        Args:
            snapshot: Order book snapshot

        Returns:
            Gravity force (positive = upward attraction)
        """
        if len(snapshot.bids) < 3 or len(snapshot.asks) < 3:
            return 0.0

        G = self.config.gravity_constant
        n = self.config.gravity_distance_power
        mid = snapshot.mid_price

        net_force = 0.0

        # Bid gravity (attracts price down toward support)
        for price, volume in snapshot.bids[:20]:  # Top 20 levels
            distance = abs(mid - price) / mid
            if distance > 0.0001:
                force = G * volume / (distance ** n)
                net_force -= force  # Negative = pull toward bids (down)

        # Ask gravity (attracts price up toward resistance)
        for price, volume in snapshot.asks[:20]:
            distance = abs(price - mid) / mid
            if distance > 0.0001:
                force = G * volume / (distance ** n)
                net_force += force  # Positive = pull toward asks (up)

        return np.tanh(net_force)  # Normalize

    def calculate_force(
        self,
        snapshot: OrderBookSnapshot,
        regime: str = 'ranging',
    ) -> Dict[str, float]:
        """Calculate steering force from order book.

        Args:
            snapshot: Order book snapshot
            regime: Current market regime

        Returns:
            Dict with force components
        """
        # Get regime multiplier
        regime_mult = {
            'trending': self.config.strength_trending,
            'ranging': self.config.strength_ranging,
            'volatile': self.config.strength_volatile,
        }.get(regime, 1.0)

        # Multi-level imbalance
        imbalances = self.compute_multi_level_imbalance(snapshot)
        weighted_imbalance = imbalances['weighted_imbalance']

        # Apply EMA smoothing
        if self._ema_imbalance is None:
            self._ema_imbalance = weighted_imbalance
        else:
            alpha = self.config.flow_ema_alpha
            self._ema_imbalance = alpha * weighted_imbalance + (1 - alpha) * self._ema_imbalance

        # Calculate imbalance force
        imbalance_force = self._ema_imbalance * self.config.imbalance_strength * regime_mult
        imbalance_force = np.clip(
            imbalance_force,
            -self.config.max_imbalance_force,
            self.config.max_imbalance_force
        )

        # Large order detection
        large_orders = self.detect_large_orders(snapshot)

        # Gravity force
        gravity_force = self.compute_gravity_force(snapshot) * self.config.depth_strength * regime_mult
        gravity_force = np.clip(
            gravity_force,
            -self.config.max_depth_force,
            self.config.max_depth_force
        )

        # Flow momentum
        flow = self.compute_flow_momentum()

        # Total force
        total_force = imbalance_force + gravity_force

        return {
            'orderbook_force': total_force,
            'imbalance_force': imbalance_force,
            'gravity_force': gravity_force,
            'weighted_imbalance': weighted_imbalance,
            'smoothed_imbalance': self._ema_imbalance,
            'flow_momentum': flow['flow_momentum'],
            'flow_acceleration': flow['flow_acceleration'],
            'spread_bps': snapshot.spread_bps,
            **large_orders,
            **{k: v for k, v in imbalances.items() if k.startswith('imbalance_level')},
        }

    def get_orderbook_features(
        self,
        snapshot: OrderBookSnapshot,
        regime: str = 'ranging',
    ) -> Dict[str, float]:
        """Get order book features for ML model."""
        force_result = self.calculate_force(snapshot, regime)

        features = {
            'ob_force': force_result['orderbook_force'],
            'ob_imbalance_force': force_result['imbalance_force'],
            'ob_gravity_force': force_result['gravity_force'],
            'ob_weighted_imbalance': force_result['weighted_imbalance'],
            'ob_smoothed_imbalance': force_result['smoothed_imbalance'],
            'ob_flow_momentum': force_result['flow_momentum'],
            'ob_flow_acceleration': force_result['flow_acceleration'],
            'ob_spread_bps': force_result['spread_bps'],
            'ob_large_bid_volume': force_result['large_bid_volume'],
            'ob_large_ask_volume': force_result['large_ask_volume'],
        }

        # Add level-specific imbalances
        for i in range(1, 5):
            key = f'imbalance_level_{i}'
            if key in force_result:
                features[f'ob_{key}'] = force_result[key]

        # Derived features
        total_large = force_result['large_bid_volume'] + force_result['large_ask_volume']
        if total_large > 0:
            features['ob_large_order_imbalance'] = (
                force_result['large_bid_volume'] - force_result['large_ask_volume']
            ) / total_large
        else:
            features['ob_large_order_imbalance'] = 0.0

        # Wall features
        if force_result['wall_bid_price'] > 0:
            features['ob_wall_bid_distance'] = (
                snapshot.mid_price - force_result['wall_bid_price']
            ) / snapshot.mid_price
        else:
            features['ob_wall_bid_distance'] = 0.0

        if force_result['wall_ask_price'] > 0:
            features['ob_wall_ask_distance'] = (
                force_result['wall_ask_price'] - snapshot.mid_price
            ) / snapshot.mid_price
        else:
            features['ob_wall_ask_distance'] = 0.0

        return features

    def reset(self):
        """Reset internal state."""
        self._snapshot_history.clear()
        self._flow_history.clear()
        self._ema_imbalance = None


def get_orderbook_hyperparameters() -> List[Dict]:
    """Get hyperparameter definitions for improvement loop."""
    return [
        {
            'name': 'ob_level_1_pct',
            'path': 'particle.orderbook.level_1_pct',
            'current_value': 0.005,
            'candidates': [0.003, 0.005, 0.007, 0.01],
            'variable_type': 'continuous',
        },
        {
            'name': 'ob_level_2_pct',
            'path': 'particle.orderbook.level_2_pct',
            'current_value': 0.01,
            'candidates': [0.007, 0.01, 0.015, 0.02],
            'variable_type': 'continuous',
        },
        {
            'name': 'ob_weight_level_1',
            'path': 'particle.orderbook.weight_level_1',
            'current_value': 0.40,
            'candidates': [0.3, 0.4, 0.5, 0.6],
            'variable_type': 'continuous',
        },
        {
            'name': 'ob_weight_level_2',
            'path': 'particle.orderbook.weight_level_2',
            'current_value': 0.30,
            'candidates': [0.2, 0.25, 0.3, 0.35],
            'variable_type': 'continuous',
        },
        {
            'name': 'ob_imbalance_strength',
            'path': 'particle.orderbook.imbalance_strength',
            'current_value': 0.08,
            'candidates': [0.04, 0.06, 0.08, 0.10, 0.12],
            'variable_type': 'continuous',
        },
        {
            'name': 'ob_depth_strength',
            'path': 'particle.orderbook.depth_strength',
            'current_value': 0.05,
            'candidates': [0.02, 0.03, 0.05, 0.07],
            'variable_type': 'continuous',
        },
        {
            'name': 'ob_flow_ema_alpha',
            'path': 'particle.orderbook.flow_ema_alpha',
            'current_value': 0.2,
            'candidates': [0.1, 0.2, 0.3, 0.5],
            'variable_type': 'continuous',
        },
        {
            'name': 'ob_gravity_constant',
            'path': 'particle.orderbook.gravity_constant',
            'current_value': 1e-8,
            'candidates': [1e-9, 5e-9, 1e-8, 5e-8],
            'variable_type': 'continuous',
        },
        {
            'name': 'ob_large_order_std_mult',
            'path': 'particle.orderbook.large_order_std_mult',
            'current_value': 3.0,
            'candidates': [2.0, 2.5, 3.0, 4.0],
            'variable_type': 'continuous',
        },
    ]
