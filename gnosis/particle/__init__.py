"""Particle state module - Physics-inspired price prediction.

This module implements a comprehensive physics-based framework for
Bitcoin/crypto price prediction. Imports are lazy to avoid import
failures when optional dependencies (ccxt, etc.) are missing.
"""

# Lazy imports — access via gnosis.particle.ClassName
_SUBMODULES = {
    "ParticleState": ".flow",
    "PriceParticle": ".physics",
    "get_particle_feature_names": ".physics",
    "QuantumPriceEngine": ".quantum",
    "EnhancedQuantumEngine": ".quantum",
    "MarketRegime": ".quantum",
    "SteeringForces": ".quantum",
    "PredictionResult": ".quantum",
    "compute_quantum_features": ".quantum",
    "FundingRateAggregator": ".funding",
    "FundingConfig": ".funding",
    "LiquidationHeatmap": ".liquidations",
    "LiquidationConfig": ".liquidations",
    "GammaFieldCalculator": ".gamma",
    "GammaConfig": ".gamma",
    "CrossAssetCoupling": ".macro",
    "MacroCouplingConfig": ".macro",
    "TimeOfDayEffects": ".temporal",
    "TemporalConfig": ".temporal",
    "MultiLevelOrderBookAnalyzer": ".orderbook",
    "OrderBookConfig": ".orderbook",
    "PhysicsCalibrator": ".calibration",
    "CalibrationConfig": ".calibration",
    "CryptoParticlePotential": ".crypto_potential",
    "CryptoParticleConfig": ".crypto_potential",
}


def __getattr__(name):
    if name in _SUBMODULES:
        import importlib
        mod = importlib.import_module(_SUBMODULES[name], package=__name__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
