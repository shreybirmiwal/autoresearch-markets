# autoresearch-markets agent program

You are doing autonomous strategy research for binary prediction markets (Kalshi).
The goal is to improve the composite score by discovering better trading strategies.

## What you are optimizing

**Primary metric: `score`** (printed at the end of every run)

```
score = 0.45 * z_final_pnl + 0.45 * z_sharpe + 0.10 * z_max_drawdown
```

Higher is better. Strategies with fewer than 10 trades or worse than -50% drawdown are filtered out.

## The one file you modify

**`markets_research/strategies.py`** — add new `Strategy` subclasses here and register them in `default_strategy_registry()`.

Do NOT modify: `prepare.py`, `train.py`, or any other file under `markets_research/` (backtest, scoring, attribution, experiment).

## Setup (do once)

```bash
uv sync
uv run python prepare.py --mode demo --out-dir data_lake   # or --mode ingest for real data
```

Initialize the results log:

```bash
echo -e "commit\tscore\tfinal_pnl\tsharpe\tstatus\tdescription" > results.tsv
```

## The loop

Repeat forever until interrupted:

**1. Propose a hypothesis.**
Read `results/report.json` (top win/loss contexts and suggested hypotheses).
Form one clear idea: e.g. "momentum at short windows might capture micro-trends", "tighter threshold on threshold_edge reduces noise", "add a volume-weighted signal".

**2. Implement it.**
Add a new `Strategy` subclass in `markets_research/strategies.py`.
Register it in `default_strategy_registry()`.
Keep changes minimal — one idea per experiment.

**3. Run.**
```bash
uv run python train.py --data-root data_lake --output-dir results > run.log 2>&1
```

**4. Read the result.**
The final lines of stdout (or `run.log`) will show:
```
status        : IMPROVED  (0.312 > prev best 0.201)
```
or
```
status        : NO_IMPROVEMENT  (0.188 <= prev best 0.201)
```

**5. Keep or discard.**

If IMPROVED or FIRST_RUN:
```bash
git add markets_research/strategies.py
git commit -m "keep: <one-line description>"
```

If NO_IMPROVEMENT or CRASH:
```bash
git checkout markets_research/strategies.py
```

**6. Log the result.**
Append one line to `results.tsv`:
```
<git commit hash>	<score>	<final_pnl>	<sharpe>	<keep|discard|crash>	<description>
```

**7. Go to step 1.**

## What counts as a good experiment

- A clear hypothesis, not a random tweak
- Minimal code change (one idea at a time)
- A `keep` run improves score — even a small improvement is progress
- A code deletion that maintains score is also a win (simpler is better)
- Do not tune hyperparameters without a mechanistic reason

## Notes

- Each run scores ALL strategies in the registry against each other. New strategies are compared on the same data as baselines — the leaderboard score reflects relative performance.
- Results artifacts: `results/leaderboard.csv`, `results/trade_attribution.csv`, `results/report.json`
- Do not stop unless manually interrupted.
