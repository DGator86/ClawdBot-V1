# moltbot-repo-scout

Production-grade static-analysis research tool for Moltbot/OpenClaw. It discovers, scores, safely clones (read-only via archive download), scans, and extracts learnable patterns from public GitHub trading-related repositories, then normalizes output into LLM-friendly artifacts.

This tool never executes foreign code. Static analysis only.

## CLI

- reposcout search
- reposcout analyze
- reposcout run
- reposcout report

Add `--json` for machine-parsable output.

## Python API

```
from moltbot_repo_scout import run_research
run_research(query="volatility sizing", focus=["risk","sizing"], limit=25)
```

## Artifacts (per repo)
- repo_report.json
- strategy_specs.json
- risk_patterns.json
- feature_recipes.md
- provenance.json

## Safety rules
- No code execution, installs, imports, or subprocesses of foreign repos
- Static AST/regex/heuristics only

## Config
See config.example.yaml and .env.example for optional settings.

