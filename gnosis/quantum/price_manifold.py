"""Price-Time Manifold with Supply/Demand Wavefunction Collapse.

Models price as a quantum particle in the price-time manifold where:
- Supply wavefunction: probability distribution of sellers at each price level
- Demand wavefunction: probability distribution of buyers at each price level
- Trade (collapse): price collapse to a single point

This is timescale-agnostic - works from 1m bars to daily bars.
"""
from dataclasses import dataclass
from typing import List, Dict

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class WavefunctionState:
    """Quantum state of price at a given time."""

    timestamp: pd.Timestamp
    price_levels: np.ndarray  # Price grid
    supply_psi: np.ndarray    # Supply wavefunction (|psi_s|^2 = probability)
    demand_psi: np.ndarray    # Demand wavefunction (|psi_d|^2 = probability)
    collapsed_price: float    # Actual trade price (wavefunction collapse)
    collapse_probability: float  # Probability of collapse at this price
    asymmetry: float          # Supply/Demand asymmetry
    overlap_peak_price: float  # Price where wavefunctions intersect maximally


@dataclass
class ManifoldPoint:
    """A point in the price-time manifold."""

    t: float  # Normalized time coordinate
    p: float  # Normalized price coordinate
    delta_t: float  # Time delta from reference (t-minus or t-plus)
    delta_p: float  # Price delta from reference
    probability: float  # Collapse probability
    regime: str  # "supply_dominated", "demand_dominated", "equilibrium"


class PriceTimeManifold:
    """Price-Time manifold with quantum-inspired dynamics.

    The price at any moment is the result of wavefunction collapse
    at the intersection of supply and demand probability distributions.
    """

    def __init__(
        self,
        price_resolution: int = 173,  # Optimized
        decay_rate: float = 0.656,     # Optimized
        sigma_supply: float = 0.018,   # Optimized
        sigma_demand: float = 0.016,   # Optimized
    ):
        self.price_resolution = price_resolution
        self.decay_rate = decay_rate
        self.sigma_supply = sigma_supply
        self.sigma_demand = sigma_demand

        self._states: List[WavefunctionState] = []
        self._manifold_points: List[ManifoldPoint] = []

    def fit_from_1m_bars(self, ohlcv_df: pd.DataFrame) -> "PriceTimeManifold":
        """Construct manifold from 1-minute OHLCV bars.

        Each bar is treated as a superposition that collapses to trades.
        """
        if ohlcv_df.empty:
            return self

        # Normalize prices to [0, 1] for manifold coordinates
        price_min = ohlcv_df['low'].min()
        price_max = ohlcv_df['high'].max()
        price_range = price_max - price_min if price_max > price_min else 1.0

        # Create price grid
        price_grid = np.linspace(price_min, price_max, self.price_resolution)

        for idx, row in ohlcv_df.iterrows():
            state = self._compute_wavefunction_state(
                row, price_grid, price_min, price_range
            )
            self._states.append(state)

        # Build manifold points with t-minus/t-plus deltas
        self._build_manifold_points(price_min, price_range)

        return self

    def _compute_wavefunction_state(
        self,
        bar: pd.Series,
        price_grid: np.ndarray,
        price_min: float,
        price_range: float,
    ) -> WavefunctionState:
        """Compute supply/demand wavefunctions for a single bar."""
        high_p, low_p, close_p = bar['high'], bar['low'], bar['close']

        # Get order flow data
        buy_vol = bar.get('buy_volume', 0.5)
        sell_vol = bar.get('sell_volume', 0.5)
        total_vol = buy_vol + sell_vol + 1e-10

        # Supply wavefunction (Sellers)
        supply_center = high_p
        supply_amplitude = 0.5 + sell_vol / total_vol
        # Dynamic width based on bar volatility
        s_sigma = self.sigma_supply * (high_p - low_p) / (price_range + 1e-10)
        s_sigma = max(s_sigma, 1e-5)

        supply_psi = supply_amplitude * np.exp(
            -0.5 * ((price_grid - supply_center) /
                    (s_sigma * price_range + 1e-10)) ** 2
        )
        supply_psi /= supply_psi.sum() + 1e-10

        # Demand wavefunction (Buyers)
        demand_center = low_p
        demand_amplitude = 0.5 + buy_vol / total_vol
        d_sigma = self.sigma_demand * (high_p - low_p) / (price_range + 1e-10)
        d_sigma = max(d_sigma, 1e-5)

        demand_psi = demand_amplitude * np.exp(
            -0.5 * ((price_grid - demand_center) /
                    (d_sigma * price_range + 1e-10)) ** 2
        )
        demand_psi /= demand_psi.sum() + 1e-10

        # Overlap and Intersection
        overlap = supply_psi * demand_psi
        overlap_norm = overlap / (overlap.sum() + 1e-10)

        overlap_peak_idx = np.argmax(overlap_norm)
        overlap_peak_price = price_grid[overlap_peak_idx]

        collapse_idx = np.argmin(np.abs(price_grid - close_p))
        collapse_prob = overlap_norm[collapse_idx]

        asymmetry = (supply_amplitude - demand_amplitude) / \
                    (supply_amplitude + demand_amplitude)

        return WavefunctionState(
            timestamp=bar['timestamp'] if 'timestamp' in bar else bar.name,
            price_levels=price_grid,
            supply_psi=supply_psi,
            demand_psi=demand_psi,
            collapsed_price=close_p,
            collapse_probability=float(collapse_prob),
            asymmetry=asymmetry,
            overlap_peak_price=overlap_peak_price,
        )

    def _build_manifold_points(self,
                               p_min: float,
                               p_range: float) -> None:
        """Build manifold points with t-minus and t-plus deltas."""
        if not self._states:
            return

        for i, state in enumerate(self._states):
            t_norm = i / len(self._states)
            p_norm = (state.collapsed_price - p_min) / (p_range + 1e-10)

            if i > 0:
                prev_state = self._states[i - 1]
                delta_t = -1.0
                delta_p = (state.collapsed_price - prev_state.collapsed_price) \
                    / (prev_state.collapsed_price + 1e-10)
            else:
                delta_t, delta_p = 0.0, 0.0

            # Regime from asymmetry (Bearish focus if > 0)
            if state.asymmetry > 0.002:
                regime = "supply_dominated"
            elif state.asymmetry < -0.002:
                regime = "demand_dominated"
            else:
                regime = "equilibrium"

            self._manifold_points.append(ManifoldPoint(
                t=t_norm,
                p=p_norm,
                delta_t=delta_t,
                delta_p=delta_p,
                probability=state.collapse_probability,
                regime=regime,
            ))

    def predict_collapse(
        self,
        horizons: List[int] = [1, 3, 5, 10, 20],
        n_simulations: int = 1000,
    ) -> pd.DataFrame:
        """Predict points using the median of Monte Carlo simulations."""
        if len(self._states) < 2:
            return pd.DataFrame()

        predictions = []
        lookback = 10

        for i, state in enumerate(self._states):
            if i < lookback:
                continue

            # Run MC simulation for each state and horizon
            sim_stats_map = self.predict_probabilistic(
                i, horizons, n_simulations
            )

            for h, sim_stats in sim_stats_map.items():
                if i + h >= len(self._states):
                    continue

                act_p = self._states[i + h].collapsed_price
                cur_p = state.collapsed_price
                pred_p = sim_stats['median']

                act_ret = (act_p - cur_p) / (cur_p + 1e-10)
                pred_ret = (pred_p - cur_p) / (cur_p + 1e-10)

                is_correct = np.sign(pred_ret) == np.sign(act_ret + 1e-10)

                # Regime logic
                regime = "equilibrium"
                if state.asymmetry > 0.002:
                    regime = "supply_dominated"
                elif state.asymmetry < -0.002:
                    regime = "demand_dominated"

                predictions.append({
                    'bar_idx': i,
                    'timestamp': state.timestamp,
                    'horizon': h,
                    'current_price': cur_p,
                    'predicted_price': float(pred_p),
                    'actual_price': act_p,
                    'predicted_return': pred_ret,
                    'actual_return': act_ret,
                    'direction_correct': is_correct,
                    'price_error': abs(pred_p - act_p),
                    'price_error_pct': abs(pred_ret - act_ret),
                    'upper_90': sim_stats['upper_90'],
                    'lower_90': sim_stats['lower_90'],
                    'regime': regime,
                    'force': sim_stats['force'],
                    'energy': sim_stats['energy']
                })

        return pd.DataFrame(predictions)

    def predict_probabilistic(
        self,
        state_idx: int,
        horizons: List[int],
        n_sims: int = 1000,
    ) -> Dict[int, Dict[str, float]]:
        """Simulate future paths using Monte Carlo in the manifold."""
        state = self._states[state_idx]
        cur_p = state.collapsed_price

        # Path history
        lookback = 10
        h_states = self._states[max(0, state_idx-lookback):state_idx+1]
        hist = np.array([s.collapsed_price for s in h_states])
        rets = np.diff(np.log(hist + 1e-10))

        vol = np.std(rets) if len(rets) > 1 else 0.001
        moms = np.mean(rets) if len(rets) > 1 else 0

        # Physics Forces
        f_pressure = -state.asymmetry
        f_revert = (state.overlap_peak_price - cur_p) / (cur_p + 1e-10)
        energy = 0.5 * (moms / (vol + 1e-10))**2

        # Net Force
        total_f = (0.6 * f_pressure) + (0.3 * f_revert) + (0.1 * moms)

        results = {}
        for h in horizons:
            # Brownian Motion scaling (dt = horizon)
            drift = total_f * h * (1 + energy)
            diffusion = vol * np.sqrt(h)

            # Monte Carlo paths
            shocks = np.random.normal(0, 1, n_sims)
            sim_rets = drift + diffusion * shocks
            sim_prices = cur_p * np.exp(np.clip(sim_rets, -0.5, 0.5))

            results[h] = {
                'mean': np.mean(sim_prices),
                'median': np.median(sim_prices),
                'upper_90': np.percentile(sim_prices, 95),
                'lower_90': np.percentile(sim_prices, 5),
                'std': np.std(sim_prices),
                'prob_above': float(np.mean(sim_prices > cur_p)),
                'force': total_f,
                'energy': energy
            }

        return results

    def predict_binary_market(
        self,
        strike: float,
        horizon_bars: int,
        n_sims: int = 5000
    ) -> Dict[str, float]:
        """Calculate probability of price > strike for Kalshi-style markets."""
        if not self._states:
            return {"prob": 0.5, "median": 0.0}

        res = self.predict_probabilistic(
            len(self._states) - 1,
            [horizon_bars],
            n_sims
        )

        sim_data = res[horizon_bars]
        return {
            'prob': sim_data['prob_above'],
            'median': sim_data['median'],
            'upper_90': sim_data['upper_90'],
            'lower_90': sim_data['lower_90']
        }

    def compute_accuracy_metrics(
        self,
        predictions_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute accuracy metrics by horizon."""
        if predictions_df.empty:
            return pd.DataFrame()

        metrics = []
        for horizon in predictions_df['horizon'].unique():
            horizon_df = predictions_df[predictions_df['horizon'] == horizon]
            metrics.append({
                'horizon_bars': horizon,
                'samples': len(horizon_df),
                'direction_accuracy': horizon_df['direction_correct'].mean(),
                'mae': horizon_df['price_error'].mean(),
                'rmse': np.sqrt((horizon_df['price_error'] ** 2).mean()),
                'mape': horizon_df['price_error_pct'].mean(),
            })
        return pd.DataFrame(metrics)


def aggregate_to_quantum_bars(
    ohlcv_1m: pd.DataFrame,
    vol_threshold_perc: float = 0.00975,
) -> pd.DataFrame:
    """Assemble 1m bars into aggregate bars based on cumulative price action."""
    if ohlcv_1m.empty:
        return ohlcv_1m

    bars = []
    symbol = ohlcv_1m['symbol'].iloc[0]

    c_open = ohlcv_1m['open'].iloc[0]
    c_high = ohlcv_1m['high'].iloc[0]
    c_low = ohlcv_1m['low'].iloc[0]
    c_vol, c_buy, c_sell = 0, 0, 0
    c_start = ohlcv_1m['timestamp'].iloc[0]

    cum_v = 0
    for i in range(len(ohlcv_1m)):
        row = ohlcv_1m.iloc[i]
        c_high = max(c_high, row['high'])
        c_low = min(c_low, row['low'])
        c_vol += row['volume']
        c_buy += row.get('buy_volume', row['volume']/2)
        c_sell += row.get('sell_volume', row['volume']/2)

        ret = abs(np.log(row['close'] / (row['open'] + 1e-10) + 1e-10))
        cum_v += ret

        if cum_v >= vol_threshold_perc or i == len(ohlcv_1m) - 1:
            bars.append({
                'symbol': symbol, 'timestamp': c_start, 'open': c_open,
                'high': c_high, 'low': c_low, 'close': row['close'],
                'volume': c_vol, 'buy_volume': c_buy, 'sell_volume': c_sell,
            })
            if i < len(ohlcv_1m) - 1:
                nr = ohlcv_1m.iloc[i+1]
                c_open, c_high, c_low = nr['open'], nr['high'], nr['low']
                c_vol, c_buy, c_sell = 0, 0, 0
                c_start, cum_v = nr['timestamp'], 0

    return pd.DataFrame(bars)


def compute_wavefunction_features(
    manifold: PriceTimeManifold,
) -> pd.DataFrame:
    """Extract features from wavefunction states."""
    features = []
    for i, state in enumerate(manifold._states):
        overlap = (state.supply_psi * state.demand_psi).sum()
        features.append({
            'bar_idx': i,
            'timestamp': state.timestamp,
            'collapsed_price': state.collapsed_price,
            'collapse_probability': state.collapse_probability,
            'wavefunction_overlap': overlap,
            'supply_demand_asymmetry': state.asymmetry,
            'supply_entropy': stats.entropy(state.supply_psi + 1e-10),
            'demand_entropy': stats.entropy(state.demand_psi + 1e-10),
        })
    return pd.DataFrame(features)
