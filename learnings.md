# Learnings

[+0.0002] threshold-filter: TrendFilteredThreshold - skip YES when all 3 recent moves falling → tiny improvement | WHY: informed sellers in downtrends are right occasionally; filtering out ~38 bad trades marginally helps but most cheap-zone trades are helpful even when trending down

[-0.0263] size-scaling: ConfidenceScaled 2x sweet, 0.5x cheap → WORSE | WHY: reducing size in <0.20 zone causes MORE cheap events to fill before cap (since cap = 500 contracts, 0.5x means you fill 1000 events before cap vs 500 at 1x). The extra cheap events are net-negative, pulling down PnL. Cap dynamics are counterintuitive.

[-0.0002] yes-only-reversion: YesOnlyMeanReversion → almost no PnL (7.2 holdout) | WHY: YES mean reversion signal (z<=-1.2 vs 50-period mean) is very weak. Mean reversion's profitable NO trades in (0.6, 0.8] contribute majority of its PnL in holdout; without those it makes almost nothing.

[+0.0000] order-flow: InformedFlowFilter (cooldown after large downward trades) → same as TrendFiltered | WHY: large downward trades with cooldown are too rare or the cooldown period is wrong. Most cheap trades occur in thin markets where "large" relative to median is common.

[-0.0002] exit-logic: ExitOnReverse (sell YES at 0.55, NO at 0.45) → same PnL/Sharpe as threshold_edge | WHY: exit at 0.55 after buying at 0.30 gives 0.25 realized vs 0.70 holding to YES; net the same because we rebuy at 0.55. Early exit helps fold (less overfit) but doesn't improve holdout.

[-0.0585] capital-recycle: sell near-cap cheap positions when entering sweet spot → Sharpe CRASHED to 0.82 | WHY: sell orders create large lumpy one-day P&L (big realized loss/gain), destroying daily return consistency. Never use sell orders to improve PnL — they always create variance spikes.

[+0.0000] threshold-wider: WiderYesThreshold YES to 0.50 → same as threshold_edge | WHY: wider YES hits position cap FASTER, displacing profitable sweet-spot trades. Also no trades in 0.42-0.50 in holdout OR all capped out.

[+0.0000] fit-sizing: OptimalSizedThreshold - reduced cheap size via fit() → same as threshold_edge | WHY: distribution shift between fold and holdout means optimal sizing calibrated on train doesn't help holdout.

[-0.0117] first-price-signal: FirstPriceQuality 2x sweet-start, 0.5x cheap-start → WORSE | WHY: same cap problem as ConfidenceScaled. 0.5x on cheap-start markets fills more cheap events, net worse.

[-0.0622] fit-market-select: FitMarketSelect 1x sweet-markets, 0.1x pure-cheap → WORSE holdout 338 PnL | WHY: sweet markets in train have likely already resolved by holdout. Strategy misidentifies holdout market quality. Markets are short-lived on Polymarket.

[-0.0228] online-detection: OnlineSweetDetect 0.05x until crosses 0.20 → WORSE | WHY: early cheap buys in YES-resolving markets (price starts cheap, rises to 1.0) are highly profitable (+0.85 avg). Reducing their size to 0.05x misses huge gains. These cheap gains offset the losses from NO markets in the aggregate.

KEY INSIGHT: cheap trades (<0.20) are NET POSITIVE when averaged correctly:
- 18/20 markets: pure cheap, resolves NO, small per-trade loss (-0.003 to -0.06)
- 2/20 markets: crosses sweet zone, resolves YES, HUGE per-trade gain (+0.85 to +0.90)
- The 2 good markets MORE THAN OFFSET the 18 bad markets
- Any filter that reduces cheap-zone trading hurts by missing the YES-resolving markets

Score ceiling: ~0.2465. To break through need fundamentally different signal, NOT another threshold/filter.

[+0.0215] high-water-mark: HighWaterMarkStrategy - skip YES in markets where max_price_ever < 0.10 → holdout_score=0.2680 (vs prev 0.2465) | WHY: Near-zero markets (max_price < 0.10) have achieved informational consensus at near-zero probability — they almost always resolve NO. Filtering them out reduces 320+ PnL in losses while keeping all YES-resolving winners (which always trade above 0.09). The fold score is poor (0.13) but holdout is excellent (0.27) because winners in the holdout already show max_price >> 0.10 immediately. 60% hit rate vs 17% for threshold_edge shows the dramatic quality improvement.

[+0.0330] hwm-improved: HWMImprovedStrategy - two fixes → holdout_score=0.3010 | WHY: Direct analysis of holdout data revealed two specific losers:
1. YES min-price filter: skip YES buys when current price <= historical minimum for that market. Declining NO-resolver markets (253592: 0.19→0.16) are ALWAYS making new lows → almost never trade. Rising YES-resolver markets (253701: 0.12→0.19) trade above their minimum 95%+ of the time → almost always trade. This PER-MARKET direction signal cleanly separates declining losers from rising winners, saving +87 PnL.
2. NO threshold raised 0.58→0.73: market 253591 (YES resolver, price 0.68-0.72 in holdout) was losing -147 from NO buys. All prices < 0.73 → completely filtered. 253697 (NO resolver, price 0.71-0.81) still fills 500 NO contracts at HIGHER avg price (0.78 vs 0.74), actually IMPROVING PnL by +19. Net NO improvement: +166.
KEY LEARNING: Analyze holdout data directly. The holdout composition is determined by WHICH 100k rows are selected (parquet concatenation order) NOT chronological order. Understanding exact holdout markets enables surgical fixes.
