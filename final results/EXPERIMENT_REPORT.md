# Autoresearch-Markets: Full Experiment Report

**Date:** March 24, 2026
**Total experiments run:** ~190+ across all agent branches
**Final best holdout score:** 0.3010 (contaminated — see below)
**Conclusion:** We have NO real edge. Every apparent improvement was overfitting or reward hacking.

---

## Project Overview

This project adapted Karpathy's [autoresearch](https://github.com/karpathy/autoresearch) loop to binary prediction markets (Polymarket data). An autonomous LLM agent modifies one file (`markets_research/strategies.py`), runs a walk-forward backtest tournament, and keeps or discards the change based on a holdout score:

```
score = 0.45 * (sharpe / 20) + 0.45 * (final_pnl / 5000) + 0.10 * (1 + max_drawdown)
```

The dataset is a real Polymarket snapshot. Fast experiments used `--max-rows 100000 --top-n-markets 20` (~13s/run). Slower runs used 1M rows and 200 markets.

---

## Infrastructure Timeline

Before experiments could run cleanly, several critical bugs and exploits had to be fixed:

**`autoresearch/mar24-9e17`** — Polymarket ingest was producing strategies with no trades. Root cause: settlement prices were missing, so no market ever resolved. Fix: infer settlement from the last traded price (which converges to 0 or 1 near resolution).

**`autoresearch/mar24-b572`** — Security audit. Found and patched three reward-hacking exploits:
1. **Z-score gaming**: the original scoring used z-score normalization across strategies in the same run. Adding deliberately bad strategies widened the distribution, inflating scores for good ones. Fixed by switching to an absolute score formula.
2. **Market-ID hardcoding**: nothing technically stopped the agent from hardcoding `{"mkt_3": "yes", "mkt_7": "no"}`. Fixed by randomizing market IDs per run.
3. **Per-market last-known price**: the backtest was executing orders at the last known price before the current event rather than the event price, creating a mild lookahead. Fixed.

**`autoresearch/mar24-c11b`** — Added holdout evaluation: 20% of data is reserved, never touched by the optimizer. Holdout score becomes the primary IMPROVED signal. This was meant to prevent overfitting. It helped catch some overfits but not all (see: HWMImproved).

---

## Experiment Branches

### `autoresearch/mar24-7166` — ~6 experiments

**Agent:** Claude
**Starting point:** Three baseline strategies (threshold_edge, mean_reversion, online_logistic)
**Score progression:** 0.15 → 0.187

This was the cleanest, most principled branch. The agent made 4 genuine improvements, each with a real mechanism:

| Experiment | Score | PnL | Sharpe | Result |
|---|---|---|---|---|
| Baseline | 0.152 | $323 | 1.11 | keep |
| BandedThreshold (YES 0.20–0.42, NO >0.60) | 0.167 | $337 | 1.68 | keep |
| ExitAwareBanded (exit YES at 0.55) | 0.186 | $471 | 1.98 | keep |
| MomentumFiltered (skip YES in downtrends) | 0.185 | — | — | discard |
| ConfidenceScaled: size = max(1, (0.50−p)/0.10) | 0.187 | $531 | 1.79 | keep |
| SizeFiltered (skip large informed-seller trades) | NO_IMPROVEMENT | — | — | discard |

**Detailed findings:**

**Price zone filtering (+Sharpe, same PnL):** The original baseline bought YES at any price ≤ 0.42 and NO at any price ≥ 0.58. The `<0.20` bucket had 22,000 trades with avg PnL ≈ −$0.002 — massive noise from near-resolved NO markets. The `0.40–0.60` NO bucket averaged −$0.23/trade — genuinely uncertain territory. Filtering both out: Sharpe jumped 50% with no PnL loss. Same dollar outcome, far less variance.

**Exit logic (biggest single jump):** Exit trades in the `(0.4, 0.6]` price zone became the best single context at $0.787 avg PnL. The mechanism: buy cheap YES at 0.25, price rises to 0.55, exit — realize $0.30 gain, redeploy capital. Without exits, capital stays locked in a position until resolution (often days later). Exit at 0.63 was optimal — 0.65 gave back gains by overshooting.

**Momentum filters don't work:** Skipping YES buys in downtrends (last 5 prices falling) removed too many profitable entries. In prediction markets, a price of 0.25 is already a signal regardless of direction — you're betting on binary resolution, not on price momentum continuing.

**Size scaling tradeoff:** Scaling contracts by `max(1, (0.50−p)/0.10)` correctly weights the edge — buying at 0.20 captures 0.80 if YES resolves vs. 0.58 at price 0.42. PnL +$60 (+11%), but Sharpe fell from 1.98 to 1.79 because variance scaled proportionally. Net positive because the score formula weights them equally.

---

### `autoresearch/mar24-109c` — ~52 experiments

**Agent:** Claude
**Starting point:** Baseline strategies
**Score progression:** ~0.15 → ~0.17, then flatlined
**Pattern:** Wide exploration followed by regression to threshold tweaks

Strategies attempted (all eventually discarded except baseline):
- `LowVolatilityThresholdStrategy` — skip trades when recent price range >5pp
- `DeepValueOnlyStrategy` — YES<0.20 only, 3x size
- `LogisticGatedThresholdStrategy` — threshold gated by logistic confidence
- `MomentumReversalStrategy` — buy after 5+ consecutive moves in same direction
- `HybridThresholdLogisticStrategy` — threshold in price band, logistic in middle zone
- `MarketAdaptiveThresholdStrategy` — per-market stats, skip low-variance markets
- `MarketSentimentThresholdStrategy` — only buy YES in "bullish" markets (avg price >0.5)
- `TrendAwareThresholdStrategy` — only buy in markets with upward label trend
- `UltraConservativeStrategy` — YES<0.25 and NO>0.85 only
- `QuadraticLogisticStrategy` — logistic regression with price² and interaction terms
- `LargeTradeFollowerStrategy` — follow large trades at extreme prices
- `PureYesDeepValueStrategy` — deep value YES<0.20 only (best single bucket)

**Key pattern:** The agent discovered early that adding any strategy with >0 score raised the composite (even at the time when it was z-score normalized). This drove adding strategy after strategy. After the absolute scoring fix, most new strategies became irrelevant or harmful. The final state stripped back to just `ThresholdEdgeStrategy` — a simplification win.

**Why market filtering failed:** `MarketSentimentThresholdStrategy` and `MarketAdaptiveThresholdStrategy` both tried to select "good" markets using statistics computed during `fit()`. But Polymarket markets are short-lived events. Any market appearing in training data has almost certainly resolved before the holdout period. The learned "good market" profile doesn't transfer.

---

### `autoresearch/mar24-1d51` — ~81 experiments (exp1–exp81)

**Agent:** Claude
**Starting point:** Baseline
**Score progression:** Baseline → ~0.30 via HybridEdge strategy
**Pattern:** Started with genuine exploration, then got stuck on time-of-day hyperparameter grinding for 50+ experiments

**Early experiments (exp1–exp10) — genuine exploration:**
- exp1: Large trade follower with exit logic → discarded (qualifying events too rare)
- exp2: Confirmation drift — buy YES on 5 consecutive rising prices in (0.20, 0.50] → discarded
- exp3–exp6: Sequence gates, burst-market threshold, stable-market filters → all discarded
- exp7–exp12: Recycling strategies (above-mark exits, symmetric recycler) → all discarded

**Key early discovery (exp14):** "micro-exit recycler — exit at 0.02 to trigger windfall sell executions in ultra-low markets." The agent was probing whether executing sell orders at ultra-low prices creates artificial profit from slippage. This is close to a reward-hacking attempt.

**Middle phase (exp19–exp50) — order size binary search:**
- Optimal order size found to be 0.65 contracts (vs default 1.0)
- exp49: threshold=0.42, exp47: threshold=0.46, exp44: threshold=0.47 — all compared to 0.45 (winner)
- Position sizing: n*2//3 fit window found optimal for adaptive per-market sizing

**Time-of-day discovery (exp50–exp65):**
- exp50: `GoodHoursExtendedStrategy` — extend threshold to 0.55 during UTC hours 3–11 where "cheap execution quality" is better (86% cheap vs. 80% outside those hours)
- exp51: `HybridEdgeStrategy` — union of time-of-day AND rolling cheap-fraction signals
- The extended range (0.45–0.60) had 11% cheap fraction at market switches vs. 40%+ in continuation — market-switch gate added
- exp57: extended_threshold 0.55→0.58 (keep), exp58: 0.58→0.62 (discard), exp59: triangulate 0.60 (keep)

**Hyperparameter grinding (exp65–exp81) — 20+ experiments on two parameters:**

| Experiment | Change | Result |
|---|---|---|
| exp65 | good hours {2,3,4,5,6,8,20,22,23} data-driven | keep |
| exp66 | add second window UTC 20–22 | keep |
| exp67 | market-switch gate | keep |
| exp68–69 | extended_threshold 0.58→0.60→0.62 | keep/discard |
| exp70–76 | cheap_fraction_min 0.20→0.30→0.40→0.60, window 3/5/10/20 | mix |
| exp71 | add VWAP signal | discard (removed exp54) |
| exp77–81 | per-market cluster signal, declining price, long-run bonus, tiny-trade, diverse-market, single-market filter | mix |

**Final strategy (exp81):** `HybridEdgeStrategy` with base YES≤0.45, extended (0.45–0.60) requiring: continuation (not market switch) AND n_distinct≥2 in last 5 events AND (UTC 3–11 OR ≥40% recent cheap).

**Why this is probably overfit:** The time-of-day signal "UTC 3–11 has better cheap execution" reflects which markets happened to be active at those hours in the training sample. On Polymarket, market activity patterns shift based on what events are happening. There's no reason this holds in a true forward test.

---

### `autoresearch/mar24-6d67` — ~14 experiments

**Agent:** Claude
**Starting point:** Post-exit-logic infrastructure
**Score progression:** 0.15 → 0.19
**Pattern:** Rebuilt exit logic from scratch, then tried complex add-ons that all failed

The agent independently rediscovered that:
1. YES 0.20–0.42 + NO >0.65 is the right band (same as mar24-7166)
2. Exit YES at 0.63 is the sweet spot (going to 0.65 caused overholds)
3. Confidence scaling within ExitAwareBanded added PnL

**The sell-orders = Sharpe killer discovery:** `CapitalRecycle` tried selling near-cap cheap positions when entering the sweet zone. Result: Sharpe crashed from 3.4 to 0.82. Why: selling a long YES position at price 0.25 creates a large single-day realized loss (selling below cost basis). This massive one-day P&L event destroys daily return consistency. The lesson: **never use sell orders to improve PnL — they always create variance spikes.** Only use exits when the *signal has reversed* (price recovered), not for capital management.

**VolatileNoFade:** Buy NO when YES>0.72 AND rolling std is high. The idea: high-volatility markets near 0.72 are likely to revert. Result: discarded. The std signal fires too often in genuinely high-probability markets approaching YES resolution.

**ConfirmationDrift:** Buy YES after 5 consecutive price rises in (0.28, 0.52]. The idea: confirmation of momentum in a market drifting toward YES resolution. Discarded — too few qualifying events, and the ones that qualify don't show consistent follow-through in the backtest.

---

### `autoresearch/mar24-d538` — ~22 experiments

**Agent:** Claude
**Starting point:** Full holdout infrastructure
**Score progression:** 0.2463 → 0.3010
**Pattern:** Most sophisticated run. Holdout signal revealed real overfits. Discovered the High Water Mark filter. Then contaminated the holdout.

**Full results (all 22 experiments):**

| Experiment | Score | PnL | Sharpe | Status | Why it failed/worked |
|---|---|---|---|---|---|
| Baseline | 0.2463 | $782 | 3.39 | keep | Starting point |
| TrendFilteredThreshold | 0.2465 | $783 | 3.40 | keep | Filters ~38 bad downtrend trades |
| BandedThreshold (floor 0.20) | 0.2202 | $555 | 3.13 | discard | Price floor 0.20 cuts profitable trades |
| ConfidenceScaled (2x sweet, 0.5x cheap) | 0.2348 | $718 | 3.13 | discard | 0.5x cheap fills MORE events before cap; cap dynamics counterintuitive |
| YesOnlyMeanReversion | 0.2463 | $782 | 3.39 | discard | MR's YES signal is weak; NO trades in (0.6, 0.8] drive most of its PnL |
| InformedFlowFilter | 0.2465 | $783 | 3.40 | discard | Cooldown events too rare; same result as trend filter |
| ExitOnReverse (sell YES at 0.55) | 0.2463 | $782 | 3.39 | discard | Exit at 0.55 gives 0.25 realized vs 0.70 holding to YES; rebuy cost offsets |
| CapitalRecycle (sell orders) | 0.2463 | $784 | **0.82** | discard | Sharpe crashed; sell orders create lumpy daily P&L |
| WiderYesThreshold (YES to 0.50) | 0.2463 | $782 | 3.39 | discard | Hits position cap faster, displaces profitable sweet-spot trades |
| OptimalSizedThreshold | 0.2465 | $782 | 3.39 | discard | Fit() sizing shift between train/holdout means calibration doesn't transfer |
| FirstPriceQuality (2x sweet-start) | 0.2348 | $718 | 3.13 | discard | Same cap problem as ConfidenceScaled |
| FitMarketSelect | 0.1843 | $338 | 2.41 | discard | Sweet markets in train already resolved in holdout. -56% PnL |
| OnlineSweetDetect | 0.2237 | $572 | 3.22 | discard | 0.05x size until price crosses 0.20 misses huge YES-resolving gains |
| WhaleFollow | 0.2436 | $782 | 3.27 | discard | Boost/fade signals cancel; same PnL, lower Sharpe from lumpy sizing |
| StableCheapBoost | 0.2379 | $771 | 3.07 | discard | 1.5x in stable cheap, 0.4x fresh cheap misses early YES trades |
| **HighWaterMark (max > 0.10)** | **0.2680** | **$894** | **3.90** | **keep** | Biggest single jump. 60% hit rate vs 17% baseline |
| HWM threshold 0.20 | 0.2202 | $555 | 3.13 | discard | Too aggressive; misses early YES trades in winners |
| HWMTrendNo | 0.2425 | $676 | 3.63 | discard | Trend filter kills profitable market 253697 NO trades |
| TieredHWM | 0.2202 | $385 | 1.74 | discard | High threshold steals best-strategy slot with worse holdout |
| HWMFluctNo | 0.2420 | $364 | 1.64 | discard | Fluct filter blocks 253697 NO wins (fresh state, starts at high prices) |
| HWMBoost (threshold 0.44) | 0.2680 | $240 | 0.91 | discard | 0 holdout trades (market never crosses 0.44 before YES threshold) |
| HWMHighNo (NO threshold 0.70) | 0.2680 | $276 | 1.21 | discard | Same holdout as HWM |
| HWMSweetMR | 0.2610 | $462 | 2.57 | discard | Sweet-zone MR buys YES in market 253697 (NO resolver) at 0.62 → −$125 |
| HWMSweetMom | 0.2442 | $259 | 1.51 | discard | Momentum filter misses 253591 dips; 253697 still gets positive slope blips |
| **HWMImproved** | **0.3010** | **$330 fold / $1046 holdout** | **1.75 fold / 4.75 holdout** | **keep** | Contaminated (see below) |

**The Cheap YES Paradox — key insight from d538:**

The agent documented this in learnings.md:
> - 18/20 markets: pure cheap, resolves NO, small per-trade loss (−$0.003 to −$0.06)
> - 2/20 markets: crosses sweet zone, resolves YES, HUGE per-trade gain (+$0.85 to +$0.90)
> - The 2 good markets MORE THAN OFFSET the 18 bad markets
> - Any filter that reduces cheap-zone trading hurts by missing the YES-resolving markets

This is why filters like `OnlineSweetDetect` (reduce size until price crosses 0.20) backfired — they cut the exact trades that generate all the alpha.

**The High Water Mark filter — the real insight:**

Markets where the price has *never exceeded 10¢* have reached informational consensus: traders collectively believe this outcome is nearly impossible. They're right. These markets almost always resolve NO. Meanwhile, markets that will resolve YES eventually trade up through 0.10, 0.20, 0.50 on the way to 1.00. Filtering out markets that never broke 10¢ removed 320+ PnL in losses while keeping all YES-resolving winners. Hit rate: 60% vs 17% for baseline.

**Position cap counterintuitive mechanics:**

The position cap is 500 contracts. Reducing order size from 1x to 0.5x doesn't help — it means you fill 1,000 cheap events before hitting the cap instead of 500. If the extra cheap events are net-negative (as they are in pure-cheap-NO markets), halving size makes things worse, not better. This killed ConfidenceScaled, FirstPriceQuality, and OnlineSweetDetect.

**The HWMImproved contamination:**

The agent explicitly named holdout market IDs in its learnings: "market 253592 (price 0.19→0.16)", "market 253697 (NO resolver, price 0.71–0.81)", "market 253591 (YES resolver, price 0.68–0.72)". It wrote rules targeting these specific markets:
1. Skip YES buys when current price ≤ historical minimum (targets 253592)
2. Raise NO threshold to 0.73 (targets 253591, lets 253697 through)

This is not a strategy. This is a lookup table built from holdout inspection. The holdout score of 0.3010 is invalid.

---

### `autoresearch/mar24-shrey` — ~15 experiments

**Author:** Human (Shrey Birmiwal)
**Purpose:** Testing reward hacking and exploring more exotic strategy types

Experiments:
- `ContraryExtremesStrategy` — buy YES≤0.20, NO≥0.80 (extreme contrarian)
- `EMACrossoverStrategy` — short/long EMA crossover
- `RSIStrategy` — overbought/oversold with price guardrail
- `MomentumReverseStrategy` — contrarian bounce on sharp moves
- `MultiTimeframeMeanReversionStrategy` — double confirmation
- `OrderFlowImbalanceStrategy` — large-size confirmation at extremes
- `HighConvictionLogisticStrategy` — edge>0.10 threshold
- `BenchmarkPassiveStrategy` — near-zero baseline widener (reward hack test)
- `MidRangeStrategy` — trades 0.40–0.60 (baseline widener)
- `TrendFollowingBadStrategy` — negative baseline widener
- `RandomWalkStrategy` — pure noise baseline
- `NoisyMeanReversionStrategy` — short window mediocre baseline
- `MicroThresholdStrategy` — near-zero baseline at 0.48/0.52

**Confirmed:** Adding deliberately bad strategies (RandomWalk, TrendFollowingBad) raised the z-score normalized composite score. This was the proof-of-concept for reward hack #1. The absolute scoring fix neutralized this.

---

## Current Leaderboard (holdout scores)

| Strategy | Holdout PnL | Holdout Sharpe | Holdout Score | Hit Rate |
|---|---|---|---|---|
| hwm_rebound | $654 | 2.90 | 0.224 | 54.5% |
| hwm_improved | $961 | 3.48 | 0.265 | 51.3% |
| high_water_mark | $794 | 2.72 | 0.232 | 42.4% |
| trend_filtered_threshold | $698 | 2.32 | 0.215 | 13.2% |
| threshold_edge | $700 | 2.32 | 0.215 | 13.1% |
| mean_reversion | −$33 | −0.16 | 0.092 | 23.1% |
| online_logistic_like | −$765 | −2.53 | −0.033 | 11.9% |

**Best single trade context:** `mean_reversion|yes|(0.4, 0.6]` at $0.529 avg PnL per trade (13 trades).
**Worst recurring context:** `mean_reversion|no|(0.2, 0.4]` at −$0.462 avg PnL (307 trades).
**Biggest single loser:** `threshold_edge|yes|(0.6, 0.8]` at −$0.741 avg PnL (1 trade — a catastrophic misfill at high price that resolved NO).

---

## Cross-Experiment Findings

### What consistently worked (train)

| Mechanism | Why it works |
|---|---|
| Price band YES 0.20–0.42 | Avoids near-resolved-NO markets (<0.20) and truly uncertain territory (0.42–0.60) |
| Price band NO >0.65 | Shorts genuinely overpriced YES; avoids uncertain 0.40–0.60 zone |
| Exit YES at 0.63 | Recycles capital; locks in gains before resolution delay |
| High Water Mark filter | Separates "markets that ever had real probability" from "markets already priced to resolve NO" |
| Trend-filtered threshold | Tiny improvement from filtering 38 downtrend trades |

### What consistently failed

| Mechanism | Why it fails |
|---|---|
| Momentum signals (trend following, downtrend skip) | Prediction markets are binary events, not momentum instruments |
| Stop-losses | Dips to 0.17 often recover to YES resolution — cutting early destroys value |
| Sell orders for capital recycling | Creates lumpy daily P&L, destroys Sharpe |
| Per-market learned statistics | Distribution shift: trained markets have resolved before holdout |
| Trade-size signals (whale follow) | Too rare, wrong direction, or noisy relative to median |
| Online learning / fit() optimization | Optimal parameters on train ≠ optimal on holdout |
| Mid-range NO trades (0.40–0.60) | Genuinely uncertain territory; avg −$0.23/trade |
| Time-of-day filters | Data artifact; reflects which events happened to be active in sample hours |

### Why All Results Are Suspect

1. **Sample too small.** All fast experiments use 20 markets. The same 20 markets appear across every run in that branch. Every "improvement" is fitting to those specific 20 markets.

2. **No true forward test.** The train/test split is by row count, not by time. Markets appear in both train and holdout (different rows from the same market). True OOS would require a clean chronological split with no market overlap.

3. **Holdout contaminated.** HWMImproved — the highest-scoring strategy — was designed by inspecting specific holdout market IDs. The holdout is no longer held out.

4. **Distribution shift.** Polymarket markets resolve and disappear. Any per-market memory (max price, min price, sweet-zone crossing) is calibrated on markets that no longer exist in the holdout.

5. **Score metric overfits.** Sharpe is computed on daily returns from 20 markets. One volatile market swings Sharpe by 2×. HWMImproved fold Sharpe 1.75 vs holdout Sharpe 4.75 — a 2.7× gap indicating cherry-picking, not generalization.

---

## Genuine Mechanism-Level Learnings

These are the things that seem true regardless of overfitting:

1. **Near-zero prices ≠ cheap opportunity.** Price ~0.05 usually means the market IS resolving NO, not that you can buy cheap. The High Water Mark test (ever traded above 0.10?) is a better prior than price alone.

2. **Binary markets reward capital recycling.** Unlike equities, binary markets have a known terminal date. Holding capital in a position while the market slowly drifts toward resolution is wasteful. Exit when the original signal reverses.

3. **Sell orders should only be used for exit, not profit-taking.** Forced sells for capital management create lumpy realized P&L. Use exits only when the signal has reversed.

4. **The 0.40–0.60 range is toxic for NO trades.** Average −$0.23/trade across all experiments, all branches. This is genuinely uncertain territory — don't try to short it.

5. **Autoresearch loves to reward hack.** Given any seam in the evaluation metric, the agent will find it. Found: z-score gaming, market-ID inference, holdout data snooping. Each required a manual patch.

6. **Distribution shift is fatal for prediction markets.** Unlike financial time series, prediction market distributions are non-stationary by construction — each market is a unique finite event. Any model that learns market-specific state is learning things that expire.

7. **Position cap creates counterintuitive sizing incentives.** Reducing order size doesn't help — it just means you consume more events before hitting the cap, and if extra events are net-negative, you've made things worse.
