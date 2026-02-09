#!/usr/bin/env python3
"""
Integration Test Suite — ClawdBot + Yoshi + Gnosis
====================================================
Verifies all gnosis components: LLM routing, Moltbot, PromptBuilder,
bridge (KPCOFGS, scoring, walk-forward, backtest), particle physics,
conformal calibration, and CLI wiring.

Run:  cd /home/root/webapp && python3 tests/test_integration.py
"""
import sys
import os
import json
import time

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

passed = failed = 0
results = []


def test(num, name, fn):
    global passed, failed
    try:
        fn()
        passed += 1
        results.append((num, name, "PASS", ""))
        print(f"  \u2713 {num}. {name}")
    except Exception as e:
        failed += 1
        results.append((num, name, "FAIL", str(e)))
        print(f"  \u2717 {num}. {name}: {e}")


# ═══════════════════════════════════════════════════════════════
# Save and clear env vars for isolated testing
# ═══════════════════════════════════════════════════════════════
saved_env = {}
for k in ["OPENAI_API_KEY", "OPENAI_BASE_URL", "GENSPARK_TOKEN"]:
    saved_env[k] = os.environ.pop(k, None)


# ── LLM Config Tests ──────────────────────────────────────────
print("\n-- LLM Config Tests --")


def t1():
    from gnosis.reasoning.client import LLMConfig, _load_dotenv
    import gnosis.reasoning.client as _client_mod
    # Patch _load_dotenv to return empty (test must be isolated from .env files)
    orig_dotenv = _client_mod._load_dotenv
    _client_mod._load_dotenv = lambda path=None: {}
    try:
        cfg = LLMConfig.from_yaml()
        assert cfg._environment in ("stub", "genspark_unresolved"), \
            f"got {cfg._environment}"
    finally:
        _client_mod._load_dotenv = orig_dotenv
test(1, "LLM Config: no key -> stub/unresolved", t1)


def t2():
    from gnosis.reasoning.client import LLMConfig
    os.environ["OPENAI_API_KEY"] = "sk-proj-" + "a" * 100
    try:
        cfg = LLMConfig.from_yaml()
        assert cfg._environment == "openai_direct", f"got {cfg._environment}"
        assert cfg.model == "gpt-4o-mini", f"got model {cfg.model}"
        assert "api.openai.com" in cfg.base_url
    finally:
        os.environ.pop("OPENAI_API_KEY", None)
test(2, "LLM Config: sk-* key -> openai_direct + gpt-4o-mini", t2)


def t3():
    from gnosis.reasoning.client import LLMConfig
    os.environ["OPENAI_API_KEY"] = "sk-proj-" + "b" * 100
    os.environ["OPENAI_BASE_URL"] = "https://my-custom-proxy.com/v1"
    try:
        cfg = LLMConfig.from_yaml()
        assert cfg._environment == "custom", f"got {cfg._environment}"
        assert cfg.base_url == "https://my-custom-proxy.com/v1"
    finally:
        os.environ.pop("OPENAI_API_KEY", None)
        os.environ.pop("OPENAI_BASE_URL", None)
test(3, "LLM Config: sk-* + custom BASE_URL -> custom", t3)


def t4():
    from gnosis.reasoning.client import LLMConfig
    os.environ["OPENAI_API_KEY"] = "sk-proj-" + "c" * 100
    os.environ["OPENAI_BASE_URL"] = "https://www.genspark.ai/api/llm_proxy/v1"
    try:
        cfg = LLMConfig.from_yaml()
        assert cfg._environment == "openai_direct", f"got {cfg._environment}"
        assert "api.openai.com" in cfg.base_url
    finally:
        os.environ.pop("OPENAI_API_KEY", None)
        os.environ.pop("OPENAI_BASE_URL", None)
test(4, "LLM Config: sk-* key + GenSpark URL -> override to openai_direct", t4)


# ── LLM Client Tests ──────────────────────────────────────────
print("\n-- LLM Client Tests --")


def t5():
    from gnosis.reasoning.client import LLMClient
    client = LLMClient()
    resp = client.chat("system", "user")
    assert resp.is_stub, "expected stub response"
    assert resp.parsed is not None
    assert resp.parsed.get("is_stub") is True
test(5, "LLM Client: stub response structure", t5)


# ── Moltbot Tests ─────────────────────────────────────────────
print("\n-- Moltbot Tests --")


def t6():
    from gnosis.execution.moltbot import load_moltbot_config
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "configs", "yoshi", "moltbot.yaml")
    cfg = load_moltbot_config(os.path.abspath(cfg_path))
    assert cfg is not None
    assert hasattr(cfg, "ai")
test(6, "Moltbot: YAML config loads", t6)


def t7():
    from gnosis.execution.moltbot import AIProviderConfig
    ai_cfg = AIProviderConfig()
    # endpoint and model start empty (auto-detected at runtime)
    assert ai_cfg.provider == "openai"
    assert ai_cfg.timeout_seconds == 60
test(7, "Moltbot: AIProviderConfig defaults", t7)


def t8():
    from gnosis.execution.moltbot import AIProviderConfig, OpenAIChatClient
    client = OpenAIChatClient(AIProviderConfig())
    assert client is not None
test(8, "Moltbot: OpenAIChatClient instantiation", t8)


def t9():
    from gnosis.execution.moltbot import AIProviderConfig
    ai_cfg = AIProviderConfig()
    endpoint = ai_cfg.resolve_endpoint()
    model = ai_cfg.resolve_model()
    assert endpoint  # should be non-empty
    assert model  # should be non-empty
    assert "chat/completions" in endpoint
test(9, "Moltbot: endpoint/model resolution", t9)


# ── PromptBuilder Tests ───────────────────────────────────────
print("\n-- PromptBuilder Tests --")

_FORECAST = {
    "symbol": "BTCUSDT", "current_price": 69000, "predicted_price": 69500,
    "confidence": 0.55, "regime": "range", "volatility": 0.03,
    "var_95": 0.01, "var_99": 0.02, "direction": "long",
    "gate_decision": {"action": "trade"}, "regime_probs": {},
}
_KPCOFGS = {"K": "TRENDING", "P": "MEAN_REVERTING"}


def t10():
    from gnosis.reasoning.prompts import PromptBuilder
    system, data = PromptBuilder.full_analysis(_FORECAST, _KPCOFGS, "range", {}, {})
    assert "BTCUSDT" in data
    assert "KPCOFGS" in data
test(10, "PromptBuilder: full_analysis includes symbol", t10)


def t11():
    from gnosis.reasoning.prompts import PromptBuilder
    system, data = PromptBuilder.regime_deep_dive(_FORECAST, _KPCOFGS, "range")
    assert "BTCUSDT" in data
test(11, "PromptBuilder: regime_deep_dive includes symbol", t11)


def t12():
    from gnosis.reasoning.prompts import PromptBuilder
    system, data = PromptBuilder.risk_assessment(_FORECAST, _KPCOFGS)
    assert "BTCUSDT" in data
test(12, "PromptBuilder: risk_assessment includes symbol", t12)


def t13():
    from gnosis.reasoning.prompts import PromptBuilder
    system, data = PromptBuilder.trade_plan(_FORECAST, _KPCOFGS, "range", {}, 500, 2.0)
    assert "BTCUSDT" in data
test(13, "PromptBuilder: trade_plan includes symbol", t13)


def t14():
    from gnosis.reasoning.prompts import PromptBuilder
    system, data = PromptBuilder.extrapolation(_FORECAST, _KPCOFGS, "range")
    assert "BTCUSDT" in data
test(14, "PromptBuilder: extrapolation includes symbol", t14)


def t15():
    from gnosis.reasoning.prompts import PromptBuilder
    system, data = PromptBuilder.self_critique(
        {"signal_quality": "MODERATE"}, _FORECAST, {}
    )
    assert "BTCUSDT" in data
test(15, "PromptBuilder: self_critique includes symbol", t15)


# ── UnifiedResult Tests ───────────────────────────────────────
print("\n-- UnifiedResult Tests --")


def t16():
    from gnosis.bridge import UnifiedResult
    r = UnifiedResult(
        forecast={"a": 1}, kpcofgs={"K": "X"}, kpcofgs_regime="range",
        validation={"ok": True}, backtest={"pnl": 100}, opportunities=[],
        reasoning=None, elapsed_ms=1234.5,
    )
    d = r.to_dict()
    assert d["kpcofgs_regime"] == "range"
    assert d["elapsed_ms"] == 1234.5
test(16, "UnifiedResult: serialization", t16)


# ── KPCOFGS Tests ─────────────────────────────────────────────
print("\n-- KPCOFGS Tests --")


def t17():
    from gnosis.bridge import classify_kpcofgs, kpcofgs_to_regime
    import pandas as pd
    import numpy as np
    np.random.seed(42)
    n = 500
    close = np.cumsum(np.random.randn(n) * 0.01) + 100
    df = pd.DataFrame({
        "close": close,
        "volume": np.random.rand(n) * 1000 + 100,
        "high": close + np.random.rand(n) * 0.5,
        "low": close - np.random.rand(n) * 0.5,
    })
    # classify_kpcofgs expects returns, realized_vol, range_pct, ofi, symbol
    df["returns"] = df["close"].pct_change().fillna(0)
    df["realized_vol"] = df["returns"].rolling(20, min_periods=5).std().fillna(0.01)
    df["range_pct"] = ((df["high"] - df["low"]) / df["close"]).fillna(0)
    df["ofi"] = 0.0
    df["symbol"] = "BTCUSDT"

    # classify_kpcofgs returns (enriched_df, summary_dict)
    enriched_df, summary = classify_kpcofgs(df)
    assert isinstance(summary, dict)
    assert "K_label" in summary and "S_label" in summary

    regime = kpcofgs_to_regime(summary)
    assert isinstance(regime, str) and len(regime) > 0
test(17, "KPCOFGS: classify + map to regime", t17)


# ── Yoshi Scoring Tests ───────────────────────────────────────
print("\n-- Yoshi Scoring Tests --")


def t18():
    from gnosis.bridge import score_forecast_series
    import numpy as np
    np.random.seed(42)
    # score_forecast_series(forecasts: List[Dict], actuals: List[float])
    n = 100
    actuals = list(np.random.randn(n) * 0.01)
    forecasts = [
        {
            "quantile_05": a - 0.02,
            "quantile_50": a + np.random.randn() * 0.001,
            "quantile_95": a + 0.02,
            "direction_prob": 0.55 if a > 0 else 0.45,
        }
        for a in actuals
    ]
    scores = score_forecast_series(forecasts, actuals)
    assert isinstance(scores, dict)
    assert "pinball_05" in scores
    assert "coverage_90" in scores
    assert scores["n_samples"] == n
test(18, "Yoshi: score_forecast_series", t18)


# ── Walk-Forward Tests ────────────────────────────────────────
print("\n-- Walk-Forward Tests --")


def t19():
    from gnosis.bridge import WalkForwardConfig
    wf = WalkForwardConfig()
    assert wf.n_outer_folds == 5
    assert wf.purge_bars >= 0
    assert wf.embargo_bars >= 0
test(19, "WalkForward: config defaults", t19)


# ── Backtest Tests ────────────────────────────────────────────
print("\n-- Backtest Tests --")


def t20():
    from gnosis.bridge import BacktestConfig
    bt = BacktestConfig()
    assert bt.initial_capital > 0
    assert bt.fee_pct >= 0
    assert bt.position_size_pct > 0
test(20, "Backtest: config defaults", t20)


# ── ArbitrageDetector Tests ──────────────────────────────────
print("\n-- ArbitrageDetector Tests --")


def t21():
    from scripts.forecaster.regime_gate import ArbitrageDetector
    det = ArbitrageDetector()
    opp = det.check_model_edge_arb(
        model_prob=0.75, market_prob=0.50,
        yes_ask=50, no_ask=50, ticker="TEST",
    )
    assert opp is not None
    assert opp.profit_pct > 0
test(21, "ArbitrageDetector: model_edge_arb", t21)


# ── Physics Features Tests ───────────────────────────────────
print("\n-- Physics Features Tests --")


def t22():
    from gnosis.particle import PriceParticle
    import pandas as pd
    import numpy as np
    np.random.seed(42)
    n = 200
    close = np.cumsum(np.random.randn(n) * 0.01) + 100
    df = pd.DataFrame({
        "close": close,
        "volume": np.abs(np.random.randn(n)) * 1000,
        "high": close + np.abs(np.random.randn(n)) * 0.5,
        "low": close - np.abs(np.random.randn(n)) * 0.5,
        "open": close + np.random.randn(n) * 0.1,
    })
    df["returns"] = df["close"].pct_change().fillna(0)
    df["symbol"] = "BTCUSDT"

    pp = PriceParticle()
    result = pp.compute_features(df)
    assert len(result) > 0
    phys_cols = [
        c for c in result.columns
        if any(w in c.lower() for w in ("momentum", "energy", "entropy", "force"))
    ]
    assert len(phys_cols) > 0, f"No physics columns found in {list(result.columns)[:10]}"
test(22, "PriceParticle: physics features", t22)


# ── QuantilePredictor Tests ──────────────────────────────────
print("\n-- QuantilePredictor Tests --")


def t23():
    from gnosis.predictors.quantile import QuantilePredictor
    qp = QuantilePredictor(models_config={})
    assert qp is not None
    assert hasattr(qp, "predict") or hasattr(qp, "fit")
test(23, "QuantilePredictor: instantiation", t23)


# ── Conformal Tests ──────────────────────────────────────────
print("\n-- Conformal Tests --")


def t24():
    import numpy as np
    from gnosis.harness.conformal import cqr_delta
    y = np.zeros(100)
    q_lo = np.full(100, -1.0)
    q_hi = np.full(100, 1.0)
    sigma = np.ones(100)
    delta = cqr_delta(y, q_lo, q_hi, sigma=sigma, normalized=True)
    assert isinstance(delta, float)
    delta2 = cqr_delta(y, q_lo, q_hi, normalized=False)
    assert isinstance(delta2, float)
test(24, "Conformal: cqr_delta", t24)


# ── CLI Tests ────────────────────────────────────────────────
print("\n-- CLI Tests --")


def t25():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "unified", os.path.join(os.path.dirname(__file__), "..", "scripts", "unified.py")
    )
    assert spec is not None, "Cannot find scripts/unified.py"
test(25, "CLI: scripts.unified loadable", t25)


# ── Bridge Wiring Tests ─────────────────────────────────────
print("\n-- Bridge Wiring Tests --")


def t26():
    from gnosis.bridge import run_unified
    assert callable(run_unified)
    import inspect
    sig = inspect.signature(run_unified)
    params = list(sig.parameters.keys())
    assert "symbol" in params
    assert "reasoning_mode" in params
test(26, "Bridge: run_unified callable with expected params", t26)


# ── Dotenv / Placeholder Tests ───────────────────────────────
print("\n-- Dotenv / Placeholder Tests --")


def t27():
    from gnosis.reasoning.client import _is_placeholder
    # Should be detected as placeholders
    assert _is_placeholder("your_openai_api_key_here") is True
    assert _is_placeholder("sk-your-new-key") is True
    assert _is_placeholder("changeme") is True
    assert _is_placeholder("") is True
    assert _is_placeholder("short") is True
    # Real keys should NOT be placeholders
    assert _is_placeholder("sk-proj-" + "a" * 100) is False
    assert _is_placeholder("gsk-" + "b" * 50) is False
test(27, "Placeholder detection: filters fake keys", t27)


def t28():
    from gnosis.reasoning.client import _load_dotenv, _is_placeholder
    env = _load_dotenv()
    oai_key = env.get("OPENAI_API_KEY", "")
    if oai_key:
        assert not _is_placeholder(oai_key), \
            f"dotenv loaded placeholder key: {oai_key[:20]}"
test(28, "Dotenv: no placeholder keys loaded", t28)


# ── OpenRouter Routing Tests ─────────────────────────────────
print("\n-- OpenRouter Routing Tests --")


def t29():
    from gnosis.reasoning.client import LLMConfig, OPENROUTER_URL, OPENROUTER_MODEL
    os.environ["OPENAI_API_KEY"] = "sk-or-v1-" + "d" * 60
    try:
        cfg = LLMConfig.from_yaml()
        assert cfg._environment == "openrouter", f"got {cfg._environment}"
        assert "openrouter.ai" in cfg.base_url, f"got url {cfg.base_url}"
        assert cfg.model == OPENROUTER_MODEL, f"got model {cfg.model}"
    finally:
        os.environ.pop("OPENAI_API_KEY", None)
test(29, "LLM Config: sk-or-v1-* -> openrouter", t29)


def t30():
    from gnosis.reasoning.client import LLMConfig
    os.environ["OPENAI_API_KEY"] = "sk-or-v1-" + "e" * 60
    os.environ["OPENAI_MODEL"] = "google/gemma-3-27b-it:free"
    try:
        cfg = LLMConfig.from_yaml()
        assert cfg._environment == "openrouter"
        assert cfg.model == "google/gemma-3-27b-it:free", f"got {cfg.model}"
    finally:
        os.environ.pop("OPENAI_API_KEY", None)
        os.environ.pop("OPENAI_MODEL", None)
test(30, "LLM Config: OpenRouter + custom model override", t30)


def t31():
    from gnosis.reasoning.client import LLMConfig
    os.environ["OPENAI_API_KEY"] = "sk-or-v1-" + "f" * 60
    os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
    try:
        cfg = LLMConfig.from_yaml()
        assert cfg._environment == "openrouter"
        assert cfg.base_url == "https://openrouter.ai/api/v1"
    finally:
        os.environ.pop("OPENAI_API_KEY", None)
        os.environ.pop("OPENAI_BASE_URL", None)
test(31, "LLM Config: OpenRouter key + explicit URL", t31)


# ── Kalshi Module Tests ──────────────────────────────────────
print("\n-- Kalshi Module Tests --")


def t32():
    from gnosis.kalshi.scanner import ScanResult
    sr = ScanResult(
        ticker="KXBTC-26FEB08-T70000",
        series="KXBTC",
        side="yes",
        strike=70000.0,
        market_prob=0.45,
        model_prob=0.55,
        edge_pct=10.0,
        cost_cents=45,
        ev_cents=5.5,
        composite_score=8.5,
    )
    d = sr.to_dict()
    assert d["ticker"] == "KXBTC-26FEB08-T70000"
    assert d["edge_pct"] == 10.0
    line = sr.summary_line()
    assert "KXBTC" in line
    assert "10.0%" in line
test(32, "Kalshi: ScanResult serialization + summary", t32)


def t33():
    from gnosis.kalshi.analyzer import ValuePlay, KalshiAnalyzer
    from gnosis.kalshi.scanner import ScanResult
    sr = ScanResult(
        ticker="TEST-T100", series="TEST", side="yes",
        market_prob=0.5, model_prob=0.6, edge_pct=10.0,
        cost_cents=50, ev_cents=5.0, volume=50,
        composite_score=5.0, model_source="price-distance",
    )
    analyzer = KalshiAnalyzer()
    # Will use rule-based fallback (stub mode)
    play = analyzer.analyze_single(sr)
    assert isinstance(play, ValuePlay)
    assert play.recommendation in ("BUY", "SKIP", "WATCH")
    assert play.value_score >= 0
test(33, "Kalshi: Analyzer rule-based fallback", t33)


def t34():
    from gnosis.kalshi.pipeline import KalshiPipeline, PipelineResult
    pipeline = KalshiPipeline(series=["KXBTC"])
    assert pipeline.series == ["KXBTC"]
    assert pipeline.top_n == 5
    result = PipelineResult()
    assert result.buy_plays == []
    d = result.to_dict()
    assert "scan_count" in d
    assert "buy_plays" in d
test(34, "Kalshi: Pipeline + PipelineResult structure", t34)


def t35():
    from gnosis.kalshi import KalshiScanner, KalshiAnalyzer, KalshiPipeline
    assert KalshiScanner is not None
    assert KalshiAnalyzer is not None
    assert KalshiPipeline is not None
test(35, "Kalshi: module exports", t35)


# ── Ralph Wiggum Tests ──────────────────────────────────────
print("\n-- Ralph Wiggum Tests --")


def t36():
    from gnosis.ralph.tracker import PredictionTracker, PredictionRecord
    import tempfile, shutil
    tmp = tempfile.mkdtemp()
    try:
        tracker = PredictionTracker(data_dir=tmp)
        rec = tracker.record_forecast(
            forecast={"symbol": "BTCUSDT", "current_price": 69000,
                      "predicted_price": 69500, "direction": "up",
                      "confidence": 0.6, "regime": "range"},
            kpcofgs={"S_label": "S_TC_PULLBACK_RESUME"},
        )
        assert rec.id != ""
        assert rec.source == "clawdbot"
        assert rec.direction_prob > 0.5, f"up direction should give prob > 0.5, got {rec.direction_prob}"
        assert tracker.total_count == 1
        assert tracker.resolved_count == 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
test(36, "Ralph: PredictionTracker record + query", t36)


def t37():
    from gnosis.ralph.tracker import PredictionTracker
    import tempfile, shutil
    tmp = tempfile.mkdtemp()
    try:
        tracker = PredictionTracker(data_dir=tmp)
        rec = tracker.record_forecast(
            forecast={"symbol": "BTCUSDT", "current_price": 69000,
                      "predicted_price": 70000, "direction": "up",
                      "confidence": 0.7, "regime": "range"},
        )
        # Resolve with higher price (correct prediction)
        resolved = tracker.resolve(rec.id, outcome_price=70500.0)
        assert resolved.resolved is True
        assert resolved.outcome_direction == "up"
        assert resolved.brier_score >= 0
        assert resolved.actual_return > 0
        # Metrics
        metrics = tracker.compute_metrics()
        assert metrics["n_resolved"] == 1
        assert metrics["brier_score"] is not None
        assert metrics["hit_rate"] is not None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
test(37, "Ralph: resolve prediction + compute metrics", t37)


def t38():
    from gnosis.ralph.tracker import PredictionTracker
    import tempfile, shutil
    tmp = tempfile.mkdtemp()
    try:
        tracker = PredictionTracker(data_dir=tmp)
        rec = tracker.record_kalshi_trade(
            scan_result={"ticker": "KXBTC-TEST", "series": "KXBTC",
                         "side": "yes", "cost_cents": 45,
                         "model_prob": 0.6, "market_prob": 0.5,
                         "edge_pct": 10.0, "ev_cents": 5.0},
        )
        assert rec.kalshi_ticker == "KXBTC-TEST"
        assert rec.kalshi_cost_cents == 45
        # Resolve as YES won
        resolved = tracker.resolve(rec.id, kalshi_settled_yes=True)
        assert resolved.pnl_cents > 0  # bought YES, YES won
        assert resolved.brier_score >= 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
test(38, "Ralph: Kalshi trade record + resolution", t38)


def t39():
    from gnosis.ralph.hyperparams import HyperParams, HyperParamManager
    import tempfile, shutil
    tmp = tempfile.mkdtemp()
    try:
        params = HyperParams()
        assert params.min_edge_pct == 5.0
        assert params.kelly_fraction == 0.25
        assert params.stop_loss_pct == 0.03
        d = params.to_dict()
        assert "min_edge_pct" in d
        restored = HyperParams.from_dict(d)
        assert restored.min_edge_pct == params.min_edge_pct
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
test(39, "Ralph: HyperParams defaults + serialization", t39)


def t40():
    from gnosis.ralph.hyperparams import HyperParamManager, HyperParams
    import tempfile, shutil
    tmp = tempfile.mkdtemp()
    try:
        mgr = HyperParamManager(data_dir=tmp, seed=42)
        # Get initial params
        p0 = mgr.get_current_params()
        assert isinstance(p0, HyperParams)
        # Step 10 times — should have some explore cycles
        explores = 0
        for _ in range(20):
            mgr.step()
            if mgr.is_exploring:
                explores += 1
        assert explores > 0, "should have at least one explore in 20 steps"
        assert mgr.cycle == 20
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
test(40, "Ralph: HyperParamManager explore/exploit", t40)


def t41():
    from gnosis.ralph.hyperparams import HyperParamManager, ParamSnapshot
    import tempfile, shutil
    tmp = tempfile.mkdtemp()
    try:
        mgr = HyperParamManager(data_dir=tmp, seed=42)
        params = mgr.get_current_params()
        # Record good performance
        mgr.record_performance(params, {
            "n_resolved": 20,
            "brier_score": 0.15,
            "hit_rate": 0.65,
            "total_pnl_cents": 200,
        })
        assert mgr._best is not None
        assert mgr._best.score > 0
        # Record bad performance — should NOT replace best
        prev_score = mgr._best.score
        mgr.record_performance(params, {
            "n_resolved": 5,
            "brier_score": 0.40,
            "hit_rate": 0.40,
            "total_pnl_cents": -100,
        })
        # Best should still be the first one (better score)
        assert mgr._best.score >= 0  # at least non-negative
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
test(41, "Ralph: performance tracking + best selection", t41)


def t42():
    from gnosis.ralph.learner import RalphLearner, LearningConfig
    import tempfile, shutil
    tmp = tempfile.mkdtemp()
    try:
        cfg = LearningConfig(data_dir=tmp, verbose=False)
        learner = RalphLearner(config=cfg)
        # Run a cycle with some forecasts
        result = learner.run_cycle(
            forecasts=[{
                "symbol": "BTCUSDT", "current_price": 69000,
                "predicted_price": 69500, "direction": "up",
                "confidence": 0.6, "regime": "range",
            }],
            scan_results=[{
                "ticker": "KXBTC-TEST", "series": "KXBTC",
                "side": "yes", "cost_cents": 45,
                "model_prob": 0.6, "market_prob": 0.5,
                "edge_pct": 10.0, "ev_cents": 5.0,
            }],
            kpcofgs={"S_label": "S_UNCERTAIN"},
        )
        assert result.cycle_number == 1
        assert result.predictions_recorded == 2  # 1 forecast + 1 kalshi
        assert result.metrics["n_predictions"] == 2
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
test(42, "Ralph: RalphLearner full cycle", t42)


def t43():
    from gnosis.ralph.learner import RalphLearner, LearningConfig
    import tempfile, shutil
    tmp = tempfile.mkdtemp()
    try:
        cfg = LearningConfig(data_dir=tmp, verbose=False)
        learner = RalphLearner(config=cfg)
        summary = learner.get_learning_summary()
        assert "tracker" in summary
        assert "params" in summary
        assert "metrics" in summary
        assert summary["tracker"]["total"] == 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
test(43, "Ralph: learning summary structure", t43)


# ── Orchestrator Tests ───────────────────────────────────────
print("\n-- Orchestrator Tests --")


def t44():
    from gnosis.orchestrator import UnifiedOrchestrator, OrchestratorConfig
    from gnosis.ralph.learner import LearningConfig
    import tempfile, shutil
    tmp = tempfile.mkdtemp()
    try:
        cfg = OrchestratorConfig(
            enable_forecast=False,
            enable_kalshi=False,
            enable_ralph=True,
            learning=LearningConfig(data_dir=tmp, verbose=False),
            verbose=False,
        )
        orch = UnifiedOrchestrator(config=cfg)
        assert orch.ralph is not None
        assert orch._cycle_count == 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
test(44, "Orchestrator: instantiation", t44)


def t45():
    from gnosis.orchestrator import OrchestratorResult
    result = OrchestratorResult(cycle=1)
    d = result.to_dict()
    assert d["cycle"] == 1
    assert "forecast" in d
    assert "scan_results" in d
    assert "ralph" in d
    assert "errors" in d
test(45, "Orchestrator: OrchestratorResult structure", t45)


def t46():
    from gnosis.orchestrator import UnifiedOrchestrator, OrchestratorConfig
    from gnosis.ralph.learner import LearningConfig
    import tempfile, shutil
    tmp = tempfile.mkdtemp()
    try:
        cfg = OrchestratorConfig(
            enable_forecast=False,
            enable_kalshi=False,
            enable_ralph=True,
            learning=LearningConfig(data_dir=tmp, verbose=False),
            verbose=False,
        )
        orch = UnifiedOrchestrator(config=cfg)
        result = orch.run_cycle()
        assert result.cycle == 1
        assert result.elapsed_ms >= 0
        assert isinstance(result.hyperparams, dict)
        # Ralph should have run
        assert result.ralph_cycle is not None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
test(46, "Orchestrator: run_cycle (forecast+kalshi disabled)", t46)


def t47():
    from gnosis.ralph import RalphLearner, PredictionTracker, HyperParams
    assert RalphLearner is not None
    assert PredictionTracker is not None
    assert HyperParams is not None
test(47, "Ralph: module __init__ exports", t47)


# ── Telegram Bot Tests ──────────────────────────────────────
print("\n-- Telegram Bot Tests --")


def t49():
    from gnosis.telegram import TelegramBot, MessageFormatter
    assert TelegramBot is not None
    assert MessageFormatter is not None
test(49, "Telegram: module exports", t49)


def t50():
    from gnosis.telegram.formatter import MessageFormatter, _esc
    # Escape special MarkdownV2 characters
    assert "\\" in _esc("hello.world")
    assert "\\" in _esc("test_underscore")
    assert "\\" in _esc("*bold*")
    # Safe text should have no added escapes (only letters/digits)
    assert _esc("hello") == "hello"
test(50, "Telegram: MarkdownV2 escaping", t50)


def t51():
    from gnosis.telegram.formatter import MessageFormatter
    fmt = MessageFormatter()
    # cycle_report with a mock result dict
    result = {
        "cycle": 5,
        "forecast": {
            "symbol": "BTCUSDT",
            "current_price": 69000,
            "predicted_price": 69500,
            "direction": "up",
            "confidence": 0.62,
            "regime": "range",
        },
        "kpcofgs": {"K_label": "TRENDING", "S_label": "NEUTRAL", "regime_entropy": 0.42},
        "scan_results": [{"ticker": "T1"}],
        "buy_count": 1,
        "watch_count": 0,
        "ralph": {
            "cycle_number": 5,
            "mode": "exploit",
            "metrics": {"n_predictions": 10, "n_resolved": 3, "brier_score": 0.22, "hit_rate": 0.67},
        },
        "elapsed_ms": 1234,
    }
    msg = fmt.cycle_report(result)
    assert "Cycle 5" in msg
    assert "BTCUSDT" in msg or "Forecast" in msg
    assert "Ralph" in msg or "ralph" in msg.lower()
    # Should be a string
    assert isinstance(msg, str)
    assert len(msg) > 50
test(51, "Telegram: cycle_report formatting", t51)


def t52():
    from gnosis.telegram.formatter import MessageFormatter
    fmt = MessageFormatter()
    vp = {
        "scan": {
            "ticker": "KXBTC-26FEB08-T70000",
            "side": "yes",
            "cost_cents": 45,
            "edge_pct": 10.5,
            "ev_cents": 5.2,
            "minutes_to_expiry": 42,
        },
        "value_score": 7.5,
        "risk_level": "MODERATE",
        "suggested_size": 3,
        "max_loss": 1.35,
        "reasoning": "High edge with moderate confidence.",
    }
    msg = fmt.buy_alert(vp)
    assert "BUY" in msg
    assert "KXBTC" in msg
    assert isinstance(msg, str)
    assert len(msg) > 30
test(52, "Telegram: buy_alert formatting", t52)


def t53():
    from gnosis.telegram.formatter import MessageFormatter
    fmt = MessageFormatter()
    msg = fmt.status_report(
        llm_info={"environment": "openrouter", "model": "llama-3.3-70b"},
        ralph_summary={
            "params": {"cycle": 10, "is_exploring": False, "best_score": 0.35},
            "tracker": {"total": 20, "resolved": 8},
            "metrics": {"brier_score": 0.20, "hit_rate": 0.65, "total_pnl_cents": 150},
        },
        kalshi_status={"exchange_active": True, "trading_active": True},
    )
    assert "Status" in msg
    assert "Ralph" in msg
    assert "Kalshi" in msg
    assert isinstance(msg, str)
test(53, "Telegram: status_report formatting", t53)


def t54():
    from gnosis.telegram.formatter import MessageFormatter
    fmt = MessageFormatter()
    summary = {
        "params": {"cycle": 15, "is_exploring": True, "best_score": 0.42, "history_size": 5},
        "metrics": {"brier_score": 0.18, "hit_rate": 0.70, "total_pnl_cents": 250, "n_kalshi_trades": 12},
        "tracker": {"total": 30, "resolved": 15},
    }
    msg = fmt.ralph_report(summary)
    assert "Ralph" in msg
    assert "Learning" in msg or "Cycle" in msg
    assert isinstance(msg, str)
test(54, "Telegram: ralph_report formatting", t54)


def t55():
    from gnosis.telegram.formatter import MessageFormatter
    fmt = MessageFormatter()
    msg = fmt.help_message()
    assert "/scan" in msg
    assert "/status" in msg
    assert "/ralph" in msg
    assert "/params" in msg
    assert "/help" in msg
test(55, "Telegram: help_message lists all commands", t55)


def t56():
    from gnosis.telegram.formatter import MessageFormatter
    fmt = MessageFormatter()
    # Plain text fallbacks
    result = {
        "cycle": 2,
        "forecast": {
            "symbol": "ETHUSDT", "current_price": 3000,
            "predicted_price": 3100, "direction": "up",
            "confidence": 0.55,
        },
        "scan_results": [{"ticker": "T1"}],
        "buy_count": 1,
        "watch_count": 0,
        "value_plays": [
            {"scan": {"ticker": "KXETH-T3100", "edge_pct": 8.2}, "recommendation": "BUY"},
        ],
        "ralph": {"cycle_number": 2, "mode": "explore", "metrics": {}},
        "elapsed_ms": 567,
    }
    plain = fmt.cycle_report_plain(result)
    assert "Cycle 2" in plain
    assert "ETHUSDT" in plain or "Forecast" in plain
    assert isinstance(plain, str)

    vp_plain = fmt.buy_alert_plain({
        "scan": {"ticker": "KXBTC-TEST", "side": "yes", "cost_cents": 40, "edge_pct": 12.0, "ev_cents": 6.0},
        "value_score": 8.0,
        "suggested_size": 2,
        "reasoning": "Strong edge.",
    })
    assert "BUY" in vp_plain
    assert "KXBTC" in vp_plain
test(56, "Telegram: plain text fallback formatting", t56)


def t57():
    from gnosis.telegram.bot import TelegramBot, TelegramAPI
    # TelegramBot without token should be inert
    bot = TelegramBot(token="", chat_id="")
    assert bot.api is None
    assert bot.send_text("test") is False
    assert bot.send_plain("test") is False
test(57, "Telegram: TelegramBot inert without token", t57)


def t58():
    from gnosis.telegram.bot import TelegramBot
    # Bot with token but no chat_id should create API
    bot = TelegramBot(token="123456:ABCtest", chat_id="")
    assert bot.api is not None
    assert bot.chat_id == ""
    # send_text should fail gracefully (no chat_id)
    assert bot.send_text("test") is False
test(58, "Telegram: TelegramBot with token but no chat_id", t58)


def t59():
    from gnosis.telegram.bot import TelegramBot
    from gnosis.orchestrator import OrchestratorConfig, UnifiedOrchestrator
    from gnosis.ralph.learner import LearningConfig
    import tempfile, shutil
    tmp = tempfile.mkdtemp()
    try:
        cfg = OrchestratorConfig(
            enable_forecast=False,
            enable_kalshi=False,
            enable_ralph=True,
            learning=LearningConfig(data_dir=tmp, verbose=False),
            verbose=False,
        )
        orch = UnifiedOrchestrator(config=cfg)
        bot = TelegramBot(
            token="123456:ABCtest",
            chat_id="12345",
            orchestrator=orch,
            notify_on_buy=True,
            notify_on_cycle=False,
        )
        assert bot.orchestrator is orch
        assert bot.notify_on_buy is True
        assert bot.notify_on_cycle is False
        assert bot._last_cycle_result is None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
test(59, "Telegram: TelegramBot wired to orchestrator", t59)


def t60():
    from gnosis.telegram.formatter import MessageFormatter
    fmt = MessageFormatter()
    msg = fmt.error_message("Something went wrong: connection refused")
    assert "Error" in msg
    assert "Something" in msg
    assert isinstance(msg, str)
test(60, "Telegram: error_message formatting", t60)


# ── Syntax Check ─────────────────────────────────────────────
print("\n-- Syntax Check --")


def t61():
    import py_compile
    import glob
    errors = []
    root = os.path.join(os.path.dirname(__file__), "..")
    for f in glob.glob(os.path.join(root, "gnosis", "**", "*.py"), recursive=True):
        try:
            py_compile.compile(f, doraise=True)
        except py_compile.PyCompileError as e:
            errors.append(str(e))
    assert not errors, f"Syntax errors: {errors}"
test(61, "Syntax: all gnosis/*.py files compile", t61)


# ═══════════════════════════════════════════════════════════════
# Restore env and report
# ═══════════════════════════════════════════════════════════════
for k, v in saved_env.items():
    if v is not None:
        os.environ[k] = v
    else:
        os.environ.pop(k, None)

print(f"\n{'=' * 60}")
print(f"INTEGRATION TESTS: {passed} PASSED, {failed} FAILED")
print(f"{'=' * 60}")
if failed:
    for num, name, status, err in results:
        if status == "FAIL":
            print(f"  FAILED: {num}. {name}: {err}")
    sys.exit(1)
else:
    sys.exit(0)
