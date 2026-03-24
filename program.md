# autoresearch-markets

This is an experiment to have the LLM do its own research on prediction market trading strategies.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date plus a random 4-character hex suffix to avoid collisions (e.g. `mar23-a3f1`). Generate the suffix with `python3 -c "import secrets; print(secrets.token_hex(2))"`. The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
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
score = 0.45 * (sharpe / 20) + 0.45 * (final_pnl / 5000) + 0.10 * (1 + max_drawdown)
```

Higher is better. Strategies with fewer than 10 trades or worse than -50% drawdown are filtered out.

This is an **absolute** score — each strategy is scored against fixed reference values, not relative to other strategies in the registry. This means you **cannot improve the score by adding deliberately bad "baseline widener" strategies**. The only way to improve is to add a strategy that genuinely performs better.

## The one file you modify

**`markets_research/strategies.py`** — add new `Strategy` subclasses here and register them in `default_strategy_registry()`.

**What you CAN do:**
- Add new `Strategy` subclasses with novel signal logic, entry/exit rules, thresholds, or combinations.
- Add **exit logic**: strategies may emit an `Order` with negative `contracts` to reduce an existing position. The backtest handles sells correctly — slippage is applied adversely and cash is credited. Use this to recycle capital and avoid hitting the position cap.
- Remove strategies that consistently underperform.
- Simplify or refactor existing strategies if it produces equal or better results.

**What you CANNOT do:**
- Modify `prepare.py`, `train.py`, or anything under `markets_research/` except `strategies.py`.
- Install new packages or add dependencies beyond what's already in `pyproject.toml`.
- Modify the scoring or evaluation logic.

**Anti-patterns to avoid:**
- **Baseline wideners**: adding strategies with deliberately poor performance to manipulate a relative score metric. The score is now absolute — this does nothing.
- **Convergent signals**: all your new strategies converging to the same "buy YES at low price" logic as existing strategies. Check `report.json` for which contexts are already saturated.
- **No exit logic**: strategies that only accumulate positions and never reduce them hit the position cap quickly and cannot recycle capital. Include exit conditions where your signal reverses.

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

## Inventing new strategies

The biggest failure mode in autoresearch is **convergence**: every new strategy becomes a minor threshold tweak because the agent is pattern-matching to what it already knows. The antidote is to reason from the *market structure* first, then invent a mechanism, rather than picking from a mental menu of known strategy types.

### Step 1: Ask first-principles questions about the data

Before writing any code, sit with these questions and actually think through them:

- **What is special about prediction markets vs stock markets?** They have binary resolution — price is literally a probability of a yes/no outcome. Near resolution, the price should converge to 0 or 1. What does that imply about when to trade and when to exit?
- **What does trade SIZE tell you?** A large trade in a thin market moves price significantly. Who makes large trades — noise traders (random) or informed traders (have information)? If informed traders are more likely to trade large, what should you do after seeing a big YES buy at price 0.3?
- **What does price VELOCITY tell you?** If price moves from 0.4 to 0.35 in 5 events, is that different from sitting at 0.35 all along? Momentum or mean-reversion — which fits this market?
- **What happens as a market approaches its resolution date?** Does the edge from buying cheap YES get better or worse near the end? Does volatility increase? Does the bid-ask spread widen?
- **What can you learn from the SEQUENCE of trades, not just the snapshot price?** e.g. three consecutive YES buys by different traders might signal growing consensus. Three consecutive NO buys might signal informed selling.
- **Are there patterns within price buckets that the current strategy misses?** The current strategy treats all YES<0.36 identically. But is a market at 0.05 the same as one at 0.34? Maybe the 0.05 market is nearly resolved NO, while the 0.34 market is genuinely uncertain.
- **What does the history of a SPECIFIC market tell you?** A market that started at 0.5 and drifted to 0.3 over 100 trades is different from one that opened at 0.3. Can you use per-market history?
- **What can you infer from simultaneous price action in related markets?** (Harder — but if two markets are correlated, does one lead the other?)

### Step 2: Generate a genuine mechanism

A good mechanism has the form:
> *"In situation X, the market price is systematically wrong because of dynamic Y, so doing Z should profit."*

Examples of real mechanisms (not just category labels):
- "Markets where price has been monotonically decreasing for the last 10 trades are in informed-seller flow; buying YES there bets against the trend and tends to lose. Skip them." (mechanism: informed flow)
- "After a market resolves YES, the price has been drifting up in the last 20 events. Buying YES when price is in 0.3–0.5 AND the last 5 prices are all strictly increasing captures markets in late-stage confirmation drift." (mechanism: confirmation drift)
- "A large single trade (size > 10x median) that moves price by >5pp signals an informed trader. Fade small trades in the opposite direction for the next 5 events." (mechanism: informed vs. noise trade distinction)
- "Markets at YES<0.15 are often either nearly-resolved-NO or genuine longshots. Those that have been stuck near the same low price for 50+ events without resolution are likely longshots that occasionally resolve YES — overbet by the market." (mechanism: resolution lag)

### Step 3: Sanity-check novelty

Before coding, ask: **is this genuinely different from existing strategies, or is it just a threshold tweak?** If the answer is "I'm changing the threshold from 0.36 to 0.34", stop. Revert to step 1.

Genuinely novel means: different *input signals*, different *timing logic*, different *per-market state*, or different *trade sequencing* — not just different numbers in the same formula.

### Step 4: Coverage check (use sparingly)

As a secondary check, make sure you've attempted each of these mechanism families at least once across your full run. This is a coverage floor, not a menu to pick from:

- Price-only signal (static threshold) — done
- Price history / rolling window signal — tried (mean reversion)
- Trade-size signal — tried (large trade follower, briefly)
- Per-market learned state — tried (online logistic)
- Price velocity / trend direction — tried (trend-aware, briefly)
- **Trade sequence / order flow patterns** — not tried well
- **Confidence-scaled position sizing** (not just 1x or 2x flat) — not tried
- **Exit logic tied to the entry signal reversing** — not tried well
- **Near-resolution behavior** (e.g. scale down as market approaches 0 or 1) — not tried

If you've been logging 5+ consecutive discards, you are stuck. Force yourself to pick something from the bottom half of that list and build a mechanism around it from step 1.

## Learning journal

After **every** run (keep or discard), append a line to `learnings.md` (create it if it doesn't exist):

```
[+0.000 / -0.000] [category] hypothesis → result | WHY: <mechanistic explanation>
```

The WHY line is mandatory. Do not just say "it didn't improve" — explain the mechanism: why did sharpe drop? why were there too few trades? what does that tell you about the data?

**Example entries:**
```
[+0.000] threshold-exit: stop-loss at -0.5 → sharpe 5.47 vs 10.88 | WHY: exits kill positions mid-drift; markets near 0.1 keep drifting further toward 0, not reverting — stop-losses are wrong-signed here
[+0.000] position-sizing: 2x contracts → pnl 2x but sharpe unchanged then score up then wait sharpe crashed to 7.3 | WHY: doubling size doubles both PnL and variance; for a fixed-signal strategy sharpe is invariant but the score composite weights sharpe heavily so variance hurt us
[+0.000] market-filter: std<0.05 filter → removes 90% of markets | WHY: prediction market prices are mostly stable near resolution; low-std is the norm, not a filter
[+0.001] threshold-static: YES<0.37 → score 0.6108 | WHY: slightly wider threshold captures a few more cheap YES that resolve correctly; marginal gain suggests we're near the optimum for this approach
```

Before forming your next hypothesis, **re-read `learnings.md`** and ask: what patterns am I seeing? What mechanisms keep failing? What territory is genuinely unexplored?

## The experiment loop

LOOP FOREVER:

1. Look at the git state: confirm the current branch and last commit.
2. **Reflect before acting** — re-read `learnings.md` (if it exists). In 2-3 sentences, state:
   - What category have I been exploring most? Am I stuck?
   - What does my learning journal tell me about WHY things have failed?
   - What is the most promising **unexplored or underexplored category**?
3. Read `results/report.json` for additional context on win/loss patterns.
4. **Form one clear hypothesis** with a mechanistic justification. Not "try tighter threshold" — but "hypothesis: markets in the 0.2–0.4 YES price range that saw a large trade in the previous event are more likely to drift toward 0 because large sellers know something; strategy: after a large NO trade, buy YES only if price is 0.25–0.40". One idea, one mechanism, one experiment.
5. Implement it: add a new `Strategy` subclass in `markets_research/strategies.py` and register it in `default_strategy_registry()`. Keep changes minimal — one idea per experiment.
6. git commit the change.
7. Run the experiment: `PYTHONPATH=. .venv2/bin/python train.py --data-root data_lake --output-dir results --max-rows 100000 --top-n-markets 20 --skip-robustness > run.log 2>&1`
8. Read the result: check the final lines of `run.log` for `status`. If the output is missing or the script crashed, run `tail -n 50 run.log` to read the stack trace and attempt a fix.
9. **Write your learning entry** in `learnings.md` — explain WHY you think this result happened.
10. Log the result to `results.tsv` (tab-separated, NOT comma-separated):
    ```
    <7-char commit hash>	<score>	<final_pnl>	<sharpe>	<keep|discard|crash>	<description>
    ```
    Do not commit `results.tsv` — leave it untracked.
11. If IMPROVED: keep the git commit and advance the branch.
12. If NO_IMPROVEMENT or CRASH: `git checkout markets_research/strategies.py` to revert, then go back to step 1.

**Crashes**: If a run crashes, use your judgment. If it's something trivial (typo, missing import), fix it and re-run. If the idea is fundamentally broken, log it as `crash`, revert, and move on. Do not spin on the same broken idea for more than 2–3 attempts.

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?" or "what should I try next?". The human is away from their computer and expects you to continue working *indefinitely* until manually interrupted. You are fully autonomous. The loop runs forever. Never stop.

As an example use case, a user might leave you running while they sleep. If each experiment takes ~3 minutes, you can run ~20/hour, for a total of ~160 experiments over the course of an average sleep. The user then wakes up to a leaderboard of results, all completed while they slept.

## What counts as a good experiment

- A clear hypothesis, not a random tweak
- Minimal code change — one idea at a time
- A `keep` run improves score — even a small improvement is progress
- A strategy deletion that maintains score is also a win (simpler is better)
- Do not tune hyperparameters without a mechanistic reason
