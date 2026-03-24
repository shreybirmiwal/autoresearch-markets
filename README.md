# autoresearch-markets

Give an AI agent a small but real prediction market trading setup and let it experiment autonomously overnight.

The idea is identical to [karpathy/autoresearch](https://github.com/karpathy/autoresearch) but the domain is binary prediction markets (Kalshi) instead of LLM training. The agent modifies one file (`strategies.py`), runs a walk-forward tournament, checks if the composite score improved, commits the improvement or discards the regression, and loops.

Each run scores all strategies in a walk-forward tournament over historical market data and emits a single `score` to compare against the previous best. Expect ~10–30 experiments per session depending on dataset size.

## Core files

| File | Role | Modified by agent? |
|------|------|--------------------|
| `prepare.py` | One-time data prep (ingest Kalshi history or generate demo data) | No |
| `train.py` | Run one full tournament, print IMPROVED/NO_IMPROVEMENT verdict | No |
| `markets_research/strategies.py` | Strategy plugin interface + all strategy implementations | **Yes — only this file** |
| `program.md` | Instructions for the autonomous research agent | Human-edited |

Everything else under `markets_research/` is fixed infrastructure (backtest engine, scoring, attribution, data ingest).

## Quick start

```bash
uv sync

# generate synthetic demo data (no API key needed)
uv run python prepare.py --mode demo --out-dir data_lake

# initialize results log
echo -e "commit\tscore\tfinal_pnl\tsharpe\tstatus\tdescription" > results.tsv

# run first experiment
uv run python train.py --data-root data_lake --output-dir results
```

## What the agent does

1. Read `results/report.json` for winning/losing trade contexts and suggested hypotheses
2. Add a new `Strategy` subclass in `markets_research/strategies.py`
3. Run `train.py`, read the `score` and `status` lines from stdout
4. If IMPROVED: `git commit` the change and log `keep` in `results.tsv`
5. If NO_IMPROVEMENT: `git checkout markets_research/strategies.py` and log `discard`
6. Repeat

See `program.md` for the full agent instructions.

## Evaluation metric

```
score = 0.45 * z_final_pnl + 0.45 * z_sharpe + 0.10 * z_max_drawdown
```

Normalized across strategies in the same tournament run. Higher is better. Strategies with fewer than 10 trades or worse than −50% drawdown are filtered before scoring.

## Results artifacts

| File | Contents |
|------|----------|
| `results/leaderboard.csv` | All strategies ranked by composite score |
| `results/trade_attribution.csv` | Per-trade win/loss breakdown |
| `results/report.json` | Best strategy, top win/loss contexts, next hypotheses |
| `results.tsv` | Longitudinal log of every experiment (keep/discard/crash) |

## Real data

To ingest actual Kalshi history instead of demo data:

```bash
uv run python prepare.py --mode ingest --out-dir data_lake
```

Requires a Kalshi API key set as `KALSHI_API_KEY` in the environment.

## Notes

- Runs on CPU; no GPU required.
- `prepare.py` is run once; `train.py` is run once per experiment.
- Community forks welcome for other prediction market venues (Polymarket, Manifold, etc.).
