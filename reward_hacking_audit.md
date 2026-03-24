# Reward Hacking Audit

Audit of the autoresearch-markets evaluation environment for reward-gaming vulnerabilities.

---

## Vulnerabilities Found

### 🔴 HIGH — Strategy can peek at settlement prices via market_id lookup

**Where:** `backtest.py` `on_event()` state passes `market_id` and `ticker` to the strategy at every tick.

**Risk:** A strategy could hardcode a lookup table mapping known `market_id` values to their final `settlement_price_yes` (visible in `results/trade_attribution.csv` and `data_lake/markets/*.parquet`). It could then buy YES for all markets that settled at 1.0 and NO for all that settled at 0.0 — achieving near-perfect hit rate and PnL with no real forecasting skill.

**Fix:** Strip `market_id` and `ticker` from the `state` dict passed to `strategy.on_event()`, or replace with an opaque integer index. Strategies should not need the human-readable ID to trade on signal.

```python
# In backtest.py, change the state dict to:
state = {
    "event_ts": row["event_ts"],
    # "market_id": row["market_id"],   # REMOVED — enables settlement lookup
    # "ticker": row["ticker"],          # REMOVED — same risk
    "yes_price": float(row["price_yes"]),
    "size": float(row["size"]),
    "position_yes_contracts": ...,
    "position_no_contracts": ...,
}
```

---

### 🔴 HIGH — Strategy can read files from disk during inference

**Where:** Strategy `on_event()` and `fit()` run in uncontrolled Python with full filesystem access.

**Risk:** A strategy could call `pd.read_parquet("data_lake/markets/...")` during `fit()` to directly load settlement prices, then hardcode perfect trades in `on_event()`. This bypasses the walk-forward split entirely.

**Fix:** There is no easy programmatic fix without sandboxing. The mitigation is a **code review rule**: the agent must not use file I/O, `os`, `subprocess`, `requests`, or any external reads inside strategy classes. This should be explicitly stated in `program.md`.

---

### 🟡 MEDIUM — Walk-forward folds are deterministic and knowable

**Where:** `_walk_forward_splits()` in `experiment.py` splits by row index on a time-sorted frame.

**Risk:** Since `max_rows=None` by default and data is a fixed Polymarket snapshot, the agent can calculate exactly which rows fall in each test fold. A strategy could use `event_ts` ranges as hardcoded gates: "if timestamp in [T1, T2] then always buy YES."

**Fix:** Add a small random jitter (±1–2%) to fold boundaries each run, or randomize the starting offset. This makes the exact split boundaries unpredictable without changing the walk-forward methodology.

---

### 🟡 MEDIUM — Sharpe inflation via minimal-trade strategies

**Where:** `scoring.py` requires only ≥ 10 trades to pass the filter.

**Risk:** A strategy that makes exactly 10–20 carefully cherry-picked trades (using known outcomes) can achieve near-infinite Sharpe ratio because the return std is tiny with consistently profitable trades.

**Fix:** Raise the minimum trade count (e.g., ≥ 100 trades) or add a turnover-weighted penalty. Alternatively, cap the Sharpe contribution: `min(sharpe, 30) / 20` prevents runaway scores from degenerate strategies.

---

### 🟡 MEDIUM — Score is unbounded above 1.0

**Where:** `scoring.py` composite score formula.

**Risk:** If a strategy achieves `final_pnl >> $5000` or `sharpe >> 20`, the score exceeds 1.0. While this isn't cheating per se, it means genuine overfitting (curve-fitting to the fixed dataset) is indistinguishable from generalization in the score signal.

**Fix:** Clamp score components: `min(sharpe/20, 1.5)` and `min(pnl/5000, 1.5)`. The current absolute scoring is good but doesn't discourage strategies that simply memorize the dataset.

---

### 🟢 LOW — Settlement default of 0.5 for unknown markets

**Where:** `experiment.py` line 107: `.fillna(0.5)`

**Risk:** Markets without a settlement price are marked as 0.5 (coin flip). A strategy that buys YES at price < 0.5 on these markets always appears to "win" (settled above cost). However, these are genuinely unknown outcomes so this is more of a data quality issue than reward hacking.

**Fix:** Filter out markets with `settlement_price_yes = NaN` from the universe entirely, rather than imputing 0.5.

---

### 🟢 LOW — Latency can be set to 0

**Where:** `BacktestConfig.latency_events = 1` default, but `--latency-events 0` is a valid CLI flag.

**Risk:** Zero latency means orders execute at the price of the event that triggered them — i.e., the price the strategy already observed. This is mild look-ahead (the strategy sees a price, places an order, and executes at that exact price rather than the next tick).

**Fix:** Enforce a minimum of 1 in `BacktestConfig` or in `run_backtest`. Currently the default is 1 so this requires deliberate misuse.

---

## What Is Already Protected

- **Absolute scoring** (not z-score relative): adding baseline-widener strategies cannot inflate the score.
- **Position caps** (500 contracts): prevents unlimited leverage.
- **Adverse slippage**: buys pay more, sells receive less — no free execution.
- **Walk-forward splits**: prevents simple in-sample overfitting to a single time period.
- **No short selling**: positions clipped to [0, max], prevents negative-position exploits.
- **Drawdown filter** (< -50% eliminated): prevents suicide-then-recovery strategies.

---

## Recommended Priority Actions

| Priority | Action |
|----------|--------|
| 🔴 Immediate | Remove `market_id` / `ticker` from `on_event()` state dict |
| 🔴 Immediate | Add to `program.md`: strategies must not read files or make network calls |
| 🟡 Soon | Raise minimum trade count to ≥ 100 |
| 🟡 Soon | Add fold-boundary jitter to prevent exact-split memorization |
| 🟢 Later | Cap Sharpe and PnL contributions to 1.5× reference value |
| 🟢 Later | Filter NaN-settlement markets instead of imputing 0.5 |
