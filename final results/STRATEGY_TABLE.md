# All Experiments — Master Strategy Table

**Total experiments: 181**

Columns: Branch | Strategy | Description | Result | Score | PnL | Sharpe

---

| # | Branch | Strategy | Description | Result | Score | PnL | Sharpe |
|---|--------|----------|-------------|--------|-------|-----|--------|
| 1 | **mar24-7166** | Baseline | threshold_edge + mean_reversion + online_logistic (3 combined) | ✅ keep | 0.1520 | $323 | 1.11 |
| 2 |  | BandedThresholdStrategy | YES only in (0.20, 0.42], NO only above 0.60 — filters noise zones | ✅ keep | 0.1670 | $337 | 1.68 |
| 3 |  | ExitAwareBandedStrategy | YES (0.20–0.42) exit at 0.55 + NO > 0.65; capital recycling | ✅ keep | 0.1860 | $471 | 1.98 |
| 4 |  | MomentumFilteredBandedStrategy | Skip YES buys in downtrends (last 5 prices falling) | ❌ discard | 0.1850 | — | — |
| 5 |  | ConfidenceScaledBandedStrategy | Position size scales with price distance from 0.5; more at lower prices | ✅ keep | 0.1870 | $531 | 1.79 |
| 6 |  | SizeFilteredBandedStrategy | Skip entries on large informed-seller trades | ❌ discard | — | — | — |
| 7 | **mar24-6d67** | PriceBandBuyYesStrategy | YES-only in (0.20, 0.42] with take-profit exit at 0.65 | ❌ discard | — | — | — |
| 8 |  | MeanReversionBandYesStrategy | YES in (0.20, 0.45] when z < −1.0 below rolling mean | ❌ discard | — | — | — |
| 9 |  | LocalMinBoostYesStrategy | 2x contracts at local min in (0.20, 0.42] vs 1x elsewhere | ❌ discard | — | — | — |
| 10 |  | ExitAwareBandedStrategy | YES (0.20–0.42) exit at 0.55 + NO > 0.65 two-sided | ✅ keep | 0.1860 | $471 | 1.98 |
| 11 |  | ExitAwareBanded (conf scaled) | YES contracts = max(1, (0.50−p)/0.10) scaling | ✅ keep | 0.1870 | — | — |
| 12 |  | ExitAwareBanded (tune NO 0.65→0.70) | Raise NO entry threshold to reduce gray-zone losses | ✅ keep | 0.1880 | — | — |
| 13 |  | ExitAwareBanded (tune NO 0.70→0.75) | Further NO threshold tuning | ✅ keep | 0.1890 | — | — |
| 14 |  | ExitAwareBanded (YES exit 0.55→0.60) | Capture more resolution gain before exiting | ✅ keep | 0.1900 | — | — |
| 15 |  | ExitAwareBanded (YES exit 0.60→0.63) | Sweet spot found — 0.65 gave back gains | ✅ keep | 0.1910 | — | — |
| 16 |  | ConfirmationDriftYesStrategy | Buy YES after 5 consecutive rises in (0.28, 0.52]; exit at 0.70 | ❌ discard | — | — | — |
| 17 |  | VolatileNoFadeStrategy | NO when YES > 0.72 + high rolling std; exit at 0.50 | ❌ discard | — | — | — |
| 18 |  | MeanRevBandYes z-score scaled | 2x contracts when z ≤ −2.0 (high confidence) | ❌ discard | — | — | — |
| 19 |  | PriceBandBuyYes exit 0.65→0.63 | Lower exit to match ExitAwareBanded optimal pattern | ❌ discard | — | — | — |
| 20 | **mar24-109c** | DeepValueYesStrategy | Buy YES below 0.15, exit above 0.35 | ❌ discard | — | — | — |
| 21 |  | DualMeanReversionStrategy | Dual-window confirmation (short + long) with exit logic | ❌ discard | — | — | — |
| 22 |  | ThresholdEdgeWithExitStrategy | Exit YES at 0.55, exit NO at 0.45 | ❌ discard | — | — | — |
| 23 |  | ScaledThresholdStrategy | Scale order size inversely with price distance from edge | ❌ discard | — | — | — |
| 24 |  | CapRecycleThresholdStrategy | Exit YES at 0.80, NO at 0.20 to recycle capital | ❌ discard | — | — | — |
| 25 |  | TightThresholdStrategy | YES below 0.30, NO above 0.70 — high conviction only | ❌ discard | — | — | — |
| 26 |  | MidThresholdStrategy | YES below 0.35, NO above 0.65 | ❌ discard | — | — | — |
| 27 |  | AsymmetricThresholdStrategy | YES below 0.35, NO above 0.70 | ❌ discard | — | — | — |
| 28 |  | AsymmetricThreshold75Strategy | YES below 0.35, NO above 0.75 | ❌ discard | — | — | — |
| 29 |  | AsymmetricThreshold80Strategy | YES below 0.35, NO above 0.80 | ❌ discard | — | — | — |
| 30 |  | AsymmetricThreshold85Strategy | YES below 0.35, NO above 0.85 | ❌ discard | — | — | — |
| 31 |  | YesWiderNO80Strategy | YES below 0.38, NO above 0.80 | ❌ discard | — | — | — |
| 32 |  | Yes36NO80Strategy | YES below 0.36, NO above 0.80 | ❌ discard | — | — | — |
| 33 |  | Yes37NO80Strategy | YES below 0.37, NO above 0.80 — fine-tune threshold | ❌ discard | — | — | — |
| 34 |  | Yes40NO80Strategy | YES below 0.40, NO above 0.80 | ❌ discard | — | — | — |
| 35 |  | YesOnlyStrategy | Pure YES below 0.38, no NO trades | ❌ discard | — | — | — |
| 36 |  | PriceUpTrendThresholdStrategy | YES only when price trending up | ❌ discard | — | — | — |
| 37 |  | Yes34NO80Strategy | YES below 0.34, NO above 0.80 — tighter entry | ❌ discard | — | — | — |
| 38 |  | StopLossThresholdStrategy | YES < 0.36, NO > 0.80 with stop-loss exit at −10pp | ❌ discard | — | — | — |
| 39 |  | ThresholdMeanReversionComboStrategy | Combined threshold + z-score confirmation | ❌ discard | — | — | — |
| 40 |  | Yes36NO90Strategy | YES below 0.36, NO above 0.90 — highest conviction | ❌ discard | — | — | — |
| 41 |  | ConstrainedLogisticStrategy | Logistic regression with hard price-side constraints | ❌ discard | — | — | — |
| 42 |  | LowVolatilityThresholdStrategy | Skip trades when recent price range > 5pp | ❌ discard | — | — | — |
| 43 |  | LowVolatilityThresholdStrategy (tuned) | Increase max_range to 0.15 (15pp threshold) | ❌ discard | — | — | — |
| 44 |  | MomentumReversalStrategy | Buy after 5+ consecutive moves in same direction | ❌ discard | — | — | — |
| 45 |  | Yes36NO80Size2Strategy | Doubled order size to improve PnL | ❌ discard | — | — | — |
| 46 |  | LogisticGatedThresholdStrategy | Threshold gated by logistic prediction confidence | ❌ discard | — | — | — |
| 47 |  | DeepValueOnlyStrategy | YES < 0.20 only, 3x size — quality over quantity | ❌ discard | — | — | — |
| 48 |  | BandedNoStrategy | NO in (0.62, 0.78) YES band + YES below 0.36 | ❌ discard | — | — | — |
| 49 |  | HybridThresholdLogisticStrategy | Threshold in price band, logistic in middle zone | ❌ discard | — | — | — |
| 50 |  | MarketAdaptiveThresholdStrategy | Skip low-variance markets; use per-market stats | ❌ discard | — | — | — |
| 51 |  | MarketSentimentThresholdStrategy | YES only in bullish markets (avg price > 0.5) | ❌ discard | — | — | — |
| 52 |  | PositionStateThresholdStrategy | Limit consecutive YES buys in falling markets | ❌ discard | — | — | — |
| 53 |  | PercentileThresholdStrategy | Buy at 10th percentile price, sell at 90th | ❌ discard | — | — | — |
| 54 |  | TrendAwareThresholdStrategy | YES only in markets with upward label trend | ❌ discard | — | — | — |
| 55 |  | UltraConservativeStrategy | YES < 0.25 and NO > 0.85 — extreme conviction only | ❌ discard | — | — | — |
| 56 |  | QuadraticLogisticStrategy | Logistic with price² and interaction features | ❌ discard | — | — | — |
| 57 |  | LargeTradeFollowerStrategy | Follow large trades at extreme prices | ❌ discard | — | — | — |
| 58 |  | MeanReversionStrategy (removed) | Removed MeanReversion — top loss context, simplify | ✅ keep | — | — | — |
| 59 |  | PureYesDeepValueStrategy | Focus on best win bucket YES < 0.20 | ❌ discard | — | — | — |
| 60 | **mar24-1d51** | exp1: LargeTradeFollowerStrategy | Follow informed YES buys (size > 3x avg) at low prices, with exit | ❌ discard | — | — | — |
| 61 |  | exp2: ConfirmationDriftStrategy | Buy YES on 5-consecutive rising prices in (0.20, 0.50] | ❌ discard | — | — | — |
| 62 |  | exp3: ConfidenceScaledThreshold | More contracts at lower prices (higher edge) | ❌ discard | — | — | — |
| 63 |  | exp4: StableMarketThreshold | Only trade when recent price range < 0.08 (avoid adverse fills) | ❌ discard | — | — | — |
| 64 |  | exp5: GlobalSequenceGate | Only trade when last global event was also cheap | ❌ discard | — | — | — |
| 65 |  | exp6: BurstMarketThreshold | Only trade during same-market consecutive bursts | ❌ discard | — | — | — |
| 66 |  | exp7: AboveMarkRecycler | Sell YES > 0.55 to lock extra profit and re-enter | ❌ discard | — | — | — |
| 67 |  | exp8: SymmetricRecycler | YES and NO both with above/below-mark exits | ❌ discard | — | — | — |
| 68 |  | exp9: TightExitRecycler (0.52) | Tight exit at 0.52 for more cycling in oscillating markets | ❌ discard | — | — | — |
| 69 |  | exp10: VeryTightExit (0.51) | Exit just above mark for max recycling | ❌ discard | — | — | — |
| 70 |  | exp11: WiderBuyRecycler | Wider buy (≤ 0.42) + exit > 0.52 for mid-range dip trades | ❌ discard | — | — | — |
| 71 |  | exp12: FullCycleRecycler | YES + NO sides both with above-mark exits | ❌ discard | — | — | — |
| 72 |  | exp13: Simplify (remove dominated) | Remove large_trade_follower, above_mark, very_tight_exit | ✅ keep | — | — | — |
| 73 |  | exp14: MicroExitRecycler | Exit at 0.02 to trigger windfall sells in ultra-low markets | ❌ discard | — | — | — |
| 74 |  | exp15: Remove NO from threshold_edge | Net −$53 PnL attribution from NO trades | ❌ discard | — | — | — |
| 75 |  | exp16: Remove NO from mean_reversion | Same contamination logic as threshold NO | ❌ discard | — | — | — |
| 76 |  | exp17: Remove NO from online_logistic | Same contamination logic | ❌ discard | — | — | — |
| 77 |  | exp18: RecyclingThreshold | Wider buy (≤ 0.42) + above-mark exit (≥ 0.52) | ❌ discard | — | — | — |
| 78 |  | exp19: order_size=0.5 | 2x trades before cap for lower daily PnL variance | ✅ keep | — | — | — |
| 79 |  | exp20: order_size=0.25 | 4x trades, push Sharpe higher | ❌ discard | — | — | — |
| 80 |  | exp21: threshold buy≤0.45 | Wider buy adds mid-range dip trades | ✅ keep | — | — | — |
| 81 |  | exp22: threshold buy≤0.48 | Even wider buy | ❌ discard | — | — | — |
| 82 |  | exp23: Simplify | Remove dominated strategies, keep only threshold_edge | ✅ keep | — | — | — |
| 83 |  | exp24: threshold buy≤0.80 | Capture mid-price events — tests wide buy | ❌ discard | — | — | — |
| 84 |  | exp25: order_size=0.4 | Between 0.5 (best) and 0.25 (too small) | ❌ discard | — | — | — |
| 85 |  | exp26: threshold=0.46 | Fine-tune between 0.45 (best) and 0.48 (worse) | ❌ discard | — | — | — |
| 86 |  | exp27: threshold=0.44 | Check if 0.45 is truly optimal vs slightly narrower | ❌ discard | — | — | — |
| 87 |  | exp28: adaptive_size | Switch from 0.5 to 0.25 contracts near position cap | ❌ discard | — | — | — |
| 88 |  | exp29: order_size=0.45 | Fine-tune between 0.5 (best) and 0.4 (worse) | ❌ discard | — | — | — |
| 89 |  | exp30: HighPriceFollower | Buy YES when p ≥ 0.90 for near-resolution YES markets | ❌ discard | — | — | — |
| 90 |  | exp31: LargeTradeFollower | Follow informed YES buys (size > 3x avg) at low prices | ❌ discard | — | — | — |
| 91 |  | exp32: AllInSingleEntry | 499 contracts on first qualifying event per market | ❌ discard | — | — | — |
| 92 |  | exp33: order_size=10.0 | Fill position cap in 50 trades per market | ❌ discard | — | — | — |
| 93 |  | exp34: order_size=2.0 | Still data-limited; PnL doubles, Sharpe unchanged | ❌ discard | — | — | — |
| 94 |  | exp35: fit()-based adaptive order_size | Fill cap using all qualifying events per market | ✅ keep | — | — | — |
| 95 |  | exp36: calibrated adaptive sizing | Window=25k matches test fold; all markets hit position cap | ✅ keep | — | — | — |
| 96 |  | exp37: window=last-1/3-of-training | Scale-invariant window size | ✅ keep | — | — | — |
| 97 |  | exp38: SameQualifyingMarketBurst | Fire only during consecutive same-market qualifying runs | ❌ discard | — | — | — |
| 98 |  | exp39: order_size=0.6 | Data-limited markets +20% contracts; cap-limited stays optimal | ✅ keep | — | — | — |
| 99 |  | exp40: order_size=0.8 | Test if PnL gain outweighs Sharpe loss from wider spread | ❌ discard | — | — | — |
| 100 |  | exp41: order_size=0.65 | Binary search between 0.60 (best) and 0.80 (worse) | ✅ keep | — | — | — |
| 101 |  | exp42: order_size=0.70 | Continue binary search above 0.65 | ❌ discard | — | — | — |
| 102 |  | exp43: fit() window=last-25k | Consistent calibration across walk-forward folds | ❌ discard | — | — | — |
| 103 |  | exp44: threshold=0.47 at size=0.65 | Test if optimal threshold shifts with larger size | ❌ discard | — | — | — |
| 104 |  | exp45: OrderFlowQualityStrategy | Larger size (0.70) in hot periods vs 0.62 in cold | ❌ discard | — | — | — |
| 105 |  | exp46: SecondOrderMarkovSizing | prev < 0.10 → 1.05x; prev > 0.60 → 0.80x | ❌ discard | — | — | — |
| 106 |  | exp47: threshold=0.46 | Test slightly wider vs 0.45 | ❌ discard | — | — | — |
| 107 |  | exp48: normalize window by fold size | Fix cross-fold size inconsistency | ❌ discard | — | — | — |
| 108 |  | exp49: threshold=0.42, adaptive sizing | Confirm 0.45 is optimal | ❌ discard | — | — | — |
| 109 |  | exp50: GoodHoursExtendedStrategy | Extend threshold to 0.55 during UTC 3–11 (86% cheap) | ✅ keep | — | — | — |
| 110 |  | exp51: HybridEdgeStrategy | Union of good-hours + cheap-cluster signals for extended buy | ✅ keep | — | — | — |
| 111 |  | exp52: cheap_fraction_min 0.70→0.60 | More extended trades at moderate cheap clusters | ✅ keep | — | — | — |
| 112 |  | exp53: cheap_fraction_min 0.60→0.50 | Even more aggressive cluster detection | ✅ keep | — | — | — |
| 113 |  | exp54: cheap_fraction_min 0.50→0.40 | Push cluster threshold further | ✅ keep | — | — | — |
| 114 |  | exp55: cheap_fraction_min 0.40→0.20 | Near-zero threshold, cluster fires almost always | ❌ discard | — | — | — |
| 115 |  | exp56: cheap_fraction_min 0.40→0.30 | Triangulate between 0.40 and 0.20 | ❌ discard | — | — | — |
| 116 |  | exp57: extended_threshold 0.55→0.58 | Wider extended range for good-hours/cluster signal | ✅ keep | — | — | — |
| 117 |  | exp58: extended_threshold 0.58→0.62 | Push wider to find plateau | ❌ discard | — | — | — |
| 118 |  | exp59: extended_threshold 0.58→0.60 | Triangulate between 0.58 and 0.62 | ✅ keep | — | — | — |
| 119 |  | exp60: expand good hours 3-11→2-12 | One extra hour each side | ❌ discard | — | — | — |
| 120 |  | exp61: rolling_window 20→15 | Faster cluster detection | ❌ discard | — | — | — |
| 121 |  | exp62: Simplify to just HybridEdge | HybridEdge dominates both ThresholdEdge and GoodHoursExtended | ✅ keep | — | — | — |
| 122 |  | exp63: Add ThresholdEdge(0.55) | Test if ungated wide threshold beats HybridEdge | ❌ discard | — | — | — |
| 123 |  | exp64: AdaptiveHoursEdgeStrategy | Learn good hours from training data (above-median cheap density) | ❌ discard | — | — | — |
| 124 |  | exp65: MultiWindowEdgeStrategy | Data-driven good hours {2,3,4,5,6,8,20,22,23} | ✅ keep | — | — | — |
| 125 |  | exp66: Add UTC 20-22 window | US evening sports hours, 30–35% cheap extended execution | ✅ keep | — | — | — |
| 126 |  | exp67: Market-switch gate | Skip extended buys at first event from any market switch | ✅ keep | — | — | — |
| 127 |  | exp68: extended_threshold 0.58→0.60 | With market-switch gate, 0.60 continuation trades positive | ✅ keep | — | — | — |
| 128 |  | exp69: extended_threshold 0.60→0.62 | Push further with market-switch gate | ❌ discard | — | — | — |
| 129 |  | exp70: cheap_fraction_min 0.40→0.30 + gate | Gate filters bad trades so more permissive cluster might work | ❌ discard | — | — | — |
| 130 |  | exp71: Add VWAP signal | Recent VWAP ≤ 0.30 as third OR condition for extended buying | ✅ keep | — | — | — |
| 131 |  | exp72: rolling_window 20→10 | Faster cluster detection with market-switch gate | ❌ discard | — | — | — |
| 132 |  | exp73: rolling_window 10→5 | Even faster cluster detection | ❌ discard | — | — | — |
| 133 |  | exp74: rolling_window 5→3 | Very short window | ❌ discard | — | — | — |
| 134 |  | exp75: cheap_fraction_min 0.40→0.60, window=5 | Test higher threshold with shorter window | ❌ discard | — | — | — |
| 135 |  | exp76: Per-market cluster signal | Qualify extended range when market's own last 5 events ≥40% cheap | ❌ discard | — | — | — |
| 136 |  | exp77: Declining within-run price | Market downtrend as extended-range qualifier | ❌ discard | — | — | — |
| 137 |  | exp78: Long-run bonus | Extend to 0.60 when run_length ≥ 6 consecutive same-market events | ❌ discard | — | — | — |
| 138 |  | exp79: Tiny-trade extended range | size ≤ 32 in (0.45, 0.60) continuation qualifies | ❌ discard | — | — | — |
| 139 |  | exp80: Diverse-market signal | ≥3 distinct markets in last 5 events qualifies extended range | ❌ discard | — | — | — |
| 140 |  | exp81: Single-market-run filter | Block extended range when n_distinct < 2 in last 5 | ✅ keep | — | — | — |
| 141 |  | Simplify: Remove VWAP signal | Negligible effect, adds complexity | ✅ keep | — | — | — |
| 142 | **mar24-d538** | Baseline | threshold_edge + mean_reversion + online_logistic_like | ✅ keep | 0.2463 | $782 | 3.39 |
| 143 |  | TrendFilteredThreshold | Skip YES when all 3 recent moves are falling (informed sellers) | ✅ keep | 0.2465 | $783 | 3.40 |
| 144 |  | BandedThresholdStrategy | Price floor 0.20 to avoid losing near-zero trades | ❌ discard | 0.2202 | $555 | 3.13 |
| 145 |  | ConfidenceScaled | 2x size in (0.20–0.42), 0.5x below 0.20 + trend filter | ❌ discard | 0.2348 | $718 | 3.13 |
| 146 |  | YesOnlyMeanReversion | Remove NO trades to eliminate losses in 0.2–0.4 range | ❌ discard | 0.2463 | $782 | 3.39 |
| 147 |  | InformedFlowFilter | Cooldown after large downward trades (informed NO buyer signal) | ❌ discard | 0.2465 | $783 | 3.40 |
| 148 |  | ExitOnReverse | Sell YES at 0.55 to recycle capital; sell NO at 0.45 | ❌ discard | 0.2463 | $782 | 3.39 |
| 149 |  | CapitalRecycle | Sell near-cap cheap positions when entering sweet spot | ❌ discard | 0.2463 | $784 | 0.82 |
| 150 |  | WiderYesThreshold | YES up to 0.50, NO above 0.65 — test YES-bias in mid-range | ❌ discard | 0.2463 | $782 | 3.39 |
| 151 |  | OptimalSizedThreshold | fit() sets reduced cheap-phase size to preserve cap room | ❌ discard | 0.2465 | $782 | 3.39 |
| 152 |  | FirstPriceQuality | 2x size for sweet-start markets, 0.5x for cheap-start | ❌ discard | 0.2348 | $718 | 3.13 |
| 153 |  | FitMarketSelect | 1x for sweet-crossing markets, 0.1x for pure-cheap markets | ❌ discard | 0.1843 | $338 | 2.41 |
| 154 |  | OnlineSweetDetect | Real-time 0.20 crossing detection; 0.05x until crossed, 1x after | ❌ discard | 0.2237 | $572 | 3.22 |
| 155 |  | WhaleFollow | 2x YES after large bullish trade (10x median); 0.3x after large bearish | ❌ discard | 0.2436 | $782 | 3.27 |
| 156 |  | StableCheapBoost | 1.5x for 50+ consecutive cheap events; 0.4x for fresh cheap | ❌ discard | 0.2379 | $771 | 3.07 |
| 157 |  | HighWaterMark | Skip YES in markets where max_price_ever < 0.10 | ✅ keep | 0.2680 | $894 | 3.90 |
| 158 |  | HighWaterMark (threshold 0.20) | More aggressive HWM filter — max < 0.20 | ❌ discard | 0.2202 | $555 | 3.13 |
| 159 |  | HWMTrendNo | YES = HWM viability; NO = only when downtrend confirmed | ❌ discard | 0.2425 | $676 | 3.63 |
| 160 |  | TieredHWM | Require max_price ≥ 0.25 for cheap YES; 0.10 for sweet zone | ❌ discard | 0.2202 | $385 | 1.74 |
| 161 |  | HWMFluctNo | YES = HWM viability; NO = only markets also been cheap (min < 0.40) | ❌ discard | 0.2420 | $364 | 1.64 |
| 162 |  | HWMBoost (threshold 0.44) | YES-only, filter markets with max < 0.43 to hit sweet-zone only | ❌ discard | 0.2680 | $240 | 0.91 |
| 163 |  | HWMHighNo (NO threshold 0.70) | Raise NO threshold to skip YES-resolver false NO trades | ❌ discard | 0.2680 | $276 | 1.21 |
| 164 |  | HWMSweetMR | HWM filter + sweet-zone YES via mean-reversion; no NO trades | ❌ discard | 0.2610 | $462 | 2.57 |
| 165 |  | HWMSweetMom | HWM filter + sweet-zone YES only when price dips AND slope positive | ❌ discard | 0.2442 | $259 | 1.51 |
| 166 |  | HWMImproved | Skip YES at new-price-lows + raise NO threshold to 0.73 | ✅ keep | 0.3010 | $330 | 1.75 |
| 167 | **mar24-shrey (human)** | ContraryExtremesStrategy | Buy YES ≤ 0.20, NO ≥ 0.80 — extreme contrarian | ❌ discard | — | — | — |
| 168 |  | EMACrossoverStrategy | Short/long EMA crossover momentum signal | ❌ discard | — | — | — |
| 169 |  | EMA + price guardrail | Skip bad contexts for EMA crossover | ❌ discard | — | — | — |
| 170 |  | RSIStrategy | Overbought/oversold with price guardrail | ❌ discard | — | — | — |
| 171 |  | MomentumReverseStrategy | Contrarian bounce on sharp moves | ❌ discard | — | — | — |
| 172 |  | MultiTimeframeMeanReversionStrategy | Double confirmation across multiple timeframes | ❌ discard | — | — | — |
| 173 |  | OrderFlowImbalanceStrategy | Large-size confirmation at extremes | ❌ discard | — | — | — |
| 174 |  | HighConvictionLogisticStrategy | Logistic regression with edge > 0.10 threshold | ❌ discard | — | — | — |
| 175 |  | BenchmarkPassiveStrategy | Near-zero baseline widener (reward hack test) | ❌ discard | — | — | — |
| 176 |  | MidRangeStrategy | Trades 0.40–0.60 range (baseline widener test) | ❌ discard | — | — | — |
| 177 |  | TrendFollowingBadStrategy | Intentional negative baseline widener (reward hack test) | ❌ discard | — | — | — |
| 178 |  | ConfirmedExtremeStrategy | Extreme price + RSI double filter | ❌ discard | — | — | — |
| 179 |  | RandomWalkStrategy | Pure noise baseline (reward hack test) | ❌ discard | — | — | — |
| 180 |  | NoisyMeanReversionStrategy | Short window mediocre baseline | ❌ discard | — | — | — |
| 181 |  | MicroThresholdStrategy | Near-zero baseline at 0.48/0.52 | ❌ discard | — | — | — |