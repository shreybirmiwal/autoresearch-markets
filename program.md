# autoresearch-markets

This is an experiment to have the LLM do its own research on prediction market trading strategies.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar23`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: Read these files for full context:
   - `README.md` — repository context and project overview.
   - `prepare.py` — data ingestion and preparation. Do not modify.
   - `train.py` — runs the backtest and scores all strategies. Do not modify.
   - `markets_research/strategies.py` — the **only** file you edit.
4. **Verify data exists**: Check that `data_lake/` exists and contains market data. Data has already been ingested from a Polymarket snapshot — do not re-run prepare.py. If `data_lake/` is somehow missing, tell the human.
5. **Initialize results log**: Create `results.tsv` with just the header row:
   ```
   commit	score	final_pnl	sharpe	status	description
   ```
6. **Confirm and go**: Confirm setup looks good, then kick off the experimentation.

## What you are optimizing

**Primary metric: `score`** (printed at the end of every run)

```
score = 0.45 * z_final_pnl + 0.45 * z_sharpe + 0.10 * z_max_drawdown
```

Higher is better. Strategies with fewer than 10 trades or worse than -50% drawdown are filtered out.

## The one file you modify

**`markets_research/strategies.py`** — add new `Strategy` subclasses here and register them in `default_strategy_registry()`.

**What you CAN do:**
- Add new `Strategy` subclasses with novel signal logic, entry/exit rules, thresholds, or combinations.
- Remove strategies that consistently underperform.
- Simplify or refactor existing strategies if it produces equal or better results.

**What you CANNOT do:**
- Modify `prepare.py`, `train.py`, or anything under `markets_research/` except `strategies.py`.
- Install new packages or add dependencies beyond what's already in `pyproject.toml`.
- Modify the scoring or evaluation logic.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing a strategy and maintaining score is a great outcome — that's a simplification win. A 0.001 score improvement that adds 50 lines of hacky code? Probably not worth it. A 0.001 improvement from deleting code? Definitely keep.

## Running an experiment

```bash
PYTHONPATH=. .venv2/bin/python train.py --data-root data_lake --output-dir results --max-rows 100000 --top-n-markets 20 --skip-robustness > run.log 2>&1
```

**Speed flags:**
- `--max-rows 100000`: cap at 100k trade rows (~13s per run vs 10+ min). Use `--max-rows 1000000` for a slower but more representative run.
- `--top-n-markets 20`: evaluate on 20 most-active markets instead of 200
- `--skip-robustness`: skip the 3-slippage robustness pass

The final lines will show:

```
status        : IMPROVED  (0.312 > prev best 0.201)
```

or

```
status        : NO_IMPROVEMENT  (0.188 <= prev best 0.201)
```

Results artifacts are written to: `results/leaderboard.csv`, `results/trade_attribution.csv`, `results/report.json`.

## The experiment loop

LOOP FOREVER:

1. Look at the git state: confirm the current branch and last commit.
2. Read `results/report.json` — it contains top win/loss contexts and suggested hypotheses. Form one clear hypothesis: e.g. "momentum at short windows might capture micro-trends", "tighter threshold reduces noise", "add a volume-weighted signal".
3. Implement it: add a new `Strategy` subclass in `markets_research/strategies.py` and register it in `default_strategy_registry()`. Keep changes minimal — one idea per experiment.
4. git commit the change.
5. Run the experiment: `PYTHONPATH=. .venv2/bin/python train.py --data-root data_lake --output-dir results --max-rows 100000 --top-n-markets 20 --skip-robustness > run.log 2>&1`
6. Read the result: check the final lines of `run.log` for `status`. If the output is missing or the script crashed, run `tail -n 50 run.log` to read the stack trace and attempt a fix.
7. Log the result to `results.tsv` (tab-separated, NOT comma-separated):
   ```
   <7-char commit hash>	<score>	<final_pnl>	<sharpe>	<keep|discard|crash>	<description>
   ```
   Do not commit `results.tsv` — leave it untracked.
8. If IMPROVED: keep the git commit and advance the branch.
9. If NO_IMPROVEMENT or CRASH: `git checkout markets_research/strategies.py` to revert, then go back to step 1.

**Crashes**: If a run crashes, use your judgment. If it's something trivial (typo, missing import), fix it and re-run. If the idea is fundamentally broken, log it as `crash`, revert, and move on. Do not spin on the same broken idea for more than 2–3 attempts.

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep or away from their computer and expects you to continue working *indefinitely* until manually stopped. You are autonomous. If you run out of ideas, think harder — re-read `results/report.json` for new angles, try combining previous near-misses, try more radical signal logic. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes ~3 minutes, you can run ~20/hour, for a total of ~160 experiments over the course of an average sleep. The user then wakes up to a leaderboard of results, all completed while they slept.

## What counts as a good experiment

- A clear hypothesis, not a random tweak
- Minimal code change — one idea at a time
- A `keep` run improves score — even a small improvement is progress
- A strategy deletion that maintains score is also a win (simpler is better)
- Do not tune hyperparameters without a mechanistic reason
