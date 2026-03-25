"""Generate a master table of all ~190 strategy experiments across all branches."""

rows = []

# ─── mar24-7166 ──────────────────────────────────────────────────────────────
branch = "mar24-7166"
rows += [
    (branch, "Baseline", "threshold_edge + mean_reversion + online_logistic (3 combined)", "keep", 0.152, 323, 1.11),
    (branch, "BandedThresholdStrategy", "YES only in (0.20, 0.42], NO only above 0.60 — filters noise zones", "keep", 0.167, 337, 1.68),
    (branch, "ExitAwareBandedStrategy", "YES (0.20–0.42) exit at 0.55 + NO > 0.65; capital recycling", "keep", 0.186, 471, 1.98),
    (branch, "MomentumFilteredBandedStrategy", "Skip YES buys in downtrends (last 5 prices falling)", "discard", 0.185, None, None),
    (branch, "ConfidenceScaledBandedStrategy", "Position size scales with price distance from 0.5; more at lower prices", "keep", 0.187, 531, 1.79),
    (branch, "SizeFilteredBandedStrategy", "Skip entries on large informed-seller trades", "discard", None, None, None),
]

# ─── mar24-6d67 ──────────────────────────────────────────────────────────────
branch = "mar24-6d67"
rows += [
    (branch, "PriceBandBuyYesStrategy", "YES-only in (0.20, 0.42] with take-profit exit at 0.65", "discard", None, None, None),
    (branch, "MeanReversionBandYesStrategy", "YES in (0.20, 0.45] when z < −1.0 below rolling mean", "discard", None, None, None),
    (branch, "LocalMinBoostYesStrategy", "2x contracts at local min in (0.20, 0.42] vs 1x elsewhere", "discard", None, None, None),
    (branch, "ExitAwareBandedStrategy", "YES (0.20–0.42) exit at 0.55 + NO > 0.65 two-sided", "keep", 0.186, 471, 1.98),
    (branch, "ExitAwareBanded (conf scaled)", "YES contracts = max(1, (0.50−p)/0.10) scaling", "keep", 0.187, None, None),
    (branch, "ExitAwareBanded (tune NO 0.65→0.70)", "Raise NO entry threshold to reduce gray-zone losses", "keep", 0.188, None, None),
    (branch, "ExitAwareBanded (tune NO 0.70→0.75)", "Further NO threshold tuning", "keep", 0.189, None, None),
    (branch, "ExitAwareBanded (YES exit 0.55→0.60)", "Capture more resolution gain before exiting", "keep", 0.190, None, None),
    (branch, "ExitAwareBanded (YES exit 0.60→0.63)", "Sweet spot found — 0.65 gave back gains", "keep", 0.191, None, None),
    (branch, "ConfirmationDriftYesStrategy", "Buy YES after 5 consecutive rises in (0.28, 0.52]; exit at 0.70", "discard", None, None, None),
    (branch, "VolatileNoFadeStrategy", "NO when YES > 0.72 + high rolling std; exit at 0.50", "discard", None, None, None),
    (branch, "MeanRevBandYes z-score scaled", "2x contracts when z ≤ −2.0 (high confidence)", "discard", None, None, None),
    (branch, "PriceBandBuyYes exit 0.65→0.63", "Lower exit to match ExitAwareBanded optimal pattern", "discard", None, None, None),
]

# ─── mar24-109c ──────────────────────────────────────────────────────────────
branch = "mar24-109c"
rows += [
    (branch, "DeepValueYesStrategy", "Buy YES below 0.15, exit above 0.35", "discard", None, None, None),
    (branch, "DualMeanReversionStrategy", "Dual-window confirmation (short + long) with exit logic", "discard", None, None, None),
    (branch, "ThresholdEdgeWithExitStrategy", "Exit YES at 0.55, exit NO at 0.45", "discard", None, None, None),
    (branch, "ScaledThresholdStrategy", "Scale order size inversely with price distance from edge", "discard", None, None, None),
    (branch, "CapRecycleThresholdStrategy", "Exit YES at 0.80, NO at 0.20 to recycle capital", "discard", None, None, None),
    (branch, "TightThresholdStrategy", "YES below 0.30, NO above 0.70 — high conviction only", "discard", None, None, None),
    (branch, "MidThresholdStrategy", "YES below 0.35, NO above 0.65", "discard", None, None, None),
    (branch, "AsymmetricThresholdStrategy", "YES below 0.35, NO above 0.70", "discard", None, None, None),
    (branch, "AsymmetricThreshold75Strategy", "YES below 0.35, NO above 0.75", "discard", None, None, None),
    (branch, "AsymmetricThreshold80Strategy", "YES below 0.35, NO above 0.80", "discard", None, None, None),
    (branch, "AsymmetricThreshold85Strategy", "YES below 0.35, NO above 0.85", "discard", None, None, None),
    (branch, "YesWiderNO80Strategy", "YES below 0.38, NO above 0.80", "discard", None, None, None),
    (branch, "Yes36NO80Strategy", "YES below 0.36, NO above 0.80", "discard", None, None, None),
    (branch, "Yes37NO80Strategy", "YES below 0.37, NO above 0.80 — fine-tune threshold", "discard", None, None, None),
    (branch, "Yes40NO80Strategy", "YES below 0.40, NO above 0.80", "discard", None, None, None),
    (branch, "YesOnlyStrategy", "Pure YES below 0.38, no NO trades", "discard", None, None, None),
    (branch, "PriceUpTrendThresholdStrategy", "YES only when price trending up", "discard", None, None, None),
    (branch, "Yes34NO80Strategy", "YES below 0.34, NO above 0.80 — tighter entry", "discard", None, None, None),
    (branch, "StopLossThresholdStrategy", "YES < 0.36, NO > 0.80 with stop-loss exit at −10pp", "discard", None, None, None),
    (branch, "ThresholdMeanReversionComboStrategy", "Combined threshold + z-score confirmation", "discard", None, None, None),
    (branch, "Yes36NO90Strategy", "YES below 0.36, NO above 0.90 — highest conviction", "discard", None, None, None),
    (branch, "ConstrainedLogisticStrategy", "Logistic regression with hard price-side constraints", "discard", None, None, None),
    (branch, "LowVolatilityThresholdStrategy", "Skip trades when recent price range > 5pp", "discard", None, None, None),
    (branch, "LowVolatilityThresholdStrategy (tuned)", "Increase max_range to 0.15 (15pp threshold)", "discard", None, None, None),
    (branch, "MomentumReversalStrategy", "Buy after 5+ consecutive moves in same direction", "discard", None, None, None),
    (branch, "Yes36NO80Size2Strategy", "Doubled order size to improve PnL", "discard", None, None, None),
    (branch, "LogisticGatedThresholdStrategy", "Threshold gated by logistic prediction confidence", "discard", None, None, None),
    (branch, "DeepValueOnlyStrategy", "YES < 0.20 only, 3x size — quality over quantity", "discard", None, None, None),
    (branch, "BandedNoStrategy", "NO in (0.62, 0.78) YES band + YES below 0.36", "discard", None, None, None),
    (branch, "HybridThresholdLogisticStrategy", "Threshold in price band, logistic in middle zone", "discard", None, None, None),
    (branch, "MarketAdaptiveThresholdStrategy", "Skip low-variance markets; use per-market stats", "discard", None, None, None),
    (branch, "MarketSentimentThresholdStrategy", "YES only in bullish markets (avg price > 0.5)", "discard", None, None, None),
    (branch, "PositionStateThresholdStrategy", "Limit consecutive YES buys in falling markets", "discard", None, None, None),
    (branch, "PercentileThresholdStrategy", "Buy at 10th percentile price, sell at 90th", "discard", None, None, None),
    (branch, "TrendAwareThresholdStrategy", "YES only in markets with upward label trend", "discard", None, None, None),
    (branch, "UltraConservativeStrategy", "YES < 0.25 and NO > 0.85 — extreme conviction only", "discard", None, None, None),
    (branch, "QuadraticLogisticStrategy", "Logistic with price² and interaction features", "discard", None, None, None),
    (branch, "LargeTradeFollowerStrategy", "Follow large trades at extreme prices", "discard", None, None, None),
    (branch, "MeanReversionStrategy (removed)", "Removed MeanReversion — top loss context, simplify", "keep", None, None, None),
    (branch, "PureYesDeepValueStrategy", "Focus on best win bucket YES < 0.20", "discard", None, None, None),
]

# ─── mar24-1d51 (81 experiments) ─────────────────────────────────────────────
branch = "mar24-1d51"
rows += [
    (branch, "exp1: LargeTradeFollowerStrategy", "Follow informed YES buys (size > 3x avg) at low prices, with exit", "discard", None, None, None),
    (branch, "exp2: ConfirmationDriftStrategy", "Buy YES on 5-consecutive rising prices in (0.20, 0.50]", "discard", None, None, None),
    (branch, "exp3: ConfidenceScaledThreshold", "More contracts at lower prices (higher edge)", "discard", None, None, None),
    (branch, "exp4: StableMarketThreshold", "Only trade when recent price range < 0.08 (avoid adverse fills)", "discard", None, None, None),
    (branch, "exp5: GlobalSequenceGate", "Only trade when last global event was also cheap", "discard", None, None, None),
    (branch, "exp6: BurstMarketThreshold", "Only trade during same-market consecutive bursts", "discard", None, None, None),
    (branch, "exp7: AboveMarkRecycler", "Sell YES > 0.55 to lock extra profit and re-enter", "discard", None, None, None),
    (branch, "exp8: SymmetricRecycler", "YES and NO both with above/below-mark exits", "discard", None, None, None),
    (branch, "exp9: TightExitRecycler (0.52)", "Tight exit at 0.52 for more cycling in oscillating markets", "discard", None, None, None),
    (branch, "exp10: VeryTightExit (0.51)", "Exit just above mark for max recycling", "discard", None, None, None),
    (branch, "exp11: WiderBuyRecycler", "Wider buy (≤ 0.42) + exit > 0.52 for mid-range dip trades", "discard", None, None, None),
    (branch, "exp12: FullCycleRecycler", "YES + NO sides both with above-mark exits", "discard", None, None, None),
    (branch, "exp13: Simplify (remove dominated)", "Remove large_trade_follower, above_mark, very_tight_exit", "keep", None, None, None),
    (branch, "exp14: MicroExitRecycler", "Exit at 0.02 to trigger windfall sells in ultra-low markets", "discard", None, None, None),
    (branch, "exp15: Remove NO from threshold_edge", "Net −$53 PnL attribution from NO trades", "discard", None, None, None),
    (branch, "exp16: Remove NO from mean_reversion", "Same contamination logic as threshold NO", "discard", None, None, None),
    (branch, "exp17: Remove NO from online_logistic", "Same contamination logic", "discard", None, None, None),
    (branch, "exp18: RecyclingThreshold", "Wider buy (≤ 0.42) + above-mark exit (≥ 0.52)", "discard", None, None, None),
    (branch, "exp19: order_size=0.5", "2x trades before cap for lower daily PnL variance", "keep", None, None, None),
    (branch, "exp20: order_size=0.25", "4x trades, push Sharpe higher", "discard", None, None, None),
    (branch, "exp21: threshold buy≤0.45", "Wider buy adds mid-range dip trades", "keep", None, None, None),
    (branch, "exp22: threshold buy≤0.48", "Even wider buy", "discard", None, None, None),
    (branch, "exp23: Simplify", "Remove dominated strategies, keep only threshold_edge", "keep", None, None, None),
    (branch, "exp24: threshold buy≤0.80", "Capture mid-price events — tests wide buy", "discard", None, None, None),
    (branch, "exp25: order_size=0.4", "Between 0.5 (best) and 0.25 (too small)", "discard", None, None, None),
    (branch, "exp26: threshold=0.46", "Fine-tune between 0.45 (best) and 0.48 (worse)", "discard", None, None, None),
    (branch, "exp27: threshold=0.44", "Check if 0.45 is truly optimal vs slightly narrower", "discard", None, None, None),
    (branch, "exp28: adaptive_size", "Switch from 0.5 to 0.25 contracts near position cap", "discard", None, None, None),
    (branch, "exp29: order_size=0.45", "Fine-tune between 0.5 (best) and 0.4 (worse)", "discard", None, None, None),
    (branch, "exp30: HighPriceFollower", "Buy YES when p ≥ 0.90 for near-resolution YES markets", "discard", None, None, None),
    (branch, "exp31: LargeTradeFollower", "Follow informed YES buys (size > 3x avg) at low prices", "discard", None, None, None),
    (branch, "exp32: AllInSingleEntry", "499 contracts on first qualifying event per market", "discard", None, None, None),
    (branch, "exp33: order_size=10.0", "Fill position cap in 50 trades per market", "discard", None, None, None),
    (branch, "exp34: order_size=2.0", "Still data-limited; PnL doubles, Sharpe unchanged", "discard", None, None, None),
    (branch, "exp35: fit()-based adaptive order_size", "Fill cap using all qualifying events per market", "keep", None, None, None),
    (branch, "exp36: calibrated adaptive sizing", "Window=25k matches test fold; all markets hit position cap", "keep", None, None, None),
    (branch, "exp37: window=last-1/3-of-training", "Scale-invariant window size", "keep", None, None, None),
    (branch, "exp38: SameQualifyingMarketBurst", "Fire only during consecutive same-market qualifying runs", "discard", None, None, None),
    (branch, "exp39: order_size=0.6", "Data-limited markets +20% contracts; cap-limited stays optimal", "keep", None, None, None),
    (branch, "exp40: order_size=0.8", "Test if PnL gain outweighs Sharpe loss from wider spread", "discard", None, None, None),
    (branch, "exp41: order_size=0.65", "Binary search between 0.60 (best) and 0.80 (worse)", "keep", None, None, None),
    (branch, "exp42: order_size=0.70", "Continue binary search above 0.65", "discard", None, None, None),
    (branch, "exp43: fit() window=last-25k", "Consistent calibration across walk-forward folds", "discard", None, None, None),
    (branch, "exp44: threshold=0.47 at size=0.65", "Test if optimal threshold shifts with larger size", "discard", None, None, None),
    (branch, "exp45: OrderFlowQualityStrategy", "Larger size (0.70) in hot periods vs 0.62 in cold", "discard", None, None, None),
    (branch, "exp46: SecondOrderMarkovSizing", "prev < 0.10 → 1.05x; prev > 0.60 → 0.80x", "discard", None, None, None),
    (branch, "exp47: threshold=0.46", "Test slightly wider vs 0.45", "discard", None, None, None),
    (branch, "exp48: normalize window by fold size", "Fix cross-fold size inconsistency", "discard", None, None, None),
    (branch, "exp49: threshold=0.42, adaptive sizing", "Confirm 0.45 is optimal", "discard", None, None, None),
    (branch, "exp50: GoodHoursExtendedStrategy", "Extend threshold to 0.55 during UTC 3–11 (86% cheap)", "keep", None, None, None),
    (branch, "exp51: HybridEdgeStrategy", "Union of good-hours + cheap-cluster signals for extended buy", "keep", None, None, None),
    (branch, "exp52: cheap_fraction_min 0.70→0.60", "More extended trades at moderate cheap clusters", "keep", None, None, None),
    (branch, "exp53: cheap_fraction_min 0.60→0.50", "Even more aggressive cluster detection", "keep", None, None, None),
    (branch, "exp54: cheap_fraction_min 0.50→0.40", "Push cluster threshold further", "keep", None, None, None),
    (branch, "exp55: cheap_fraction_min 0.40→0.20", "Near-zero threshold, cluster fires almost always", "discard", None, None, None),
    (branch, "exp56: cheap_fraction_min 0.40→0.30", "Triangulate between 0.40 and 0.20", "discard", None, None, None),
    (branch, "exp57: extended_threshold 0.55→0.58", "Wider extended range for good-hours/cluster signal", "keep", None, None, None),
    (branch, "exp58: extended_threshold 0.58→0.62", "Push wider to find plateau", "discard", None, None, None),
    (branch, "exp59: extended_threshold 0.58→0.60", "Triangulate between 0.58 and 0.62", "keep", None, None, None),
    (branch, "exp60: expand good hours 3-11→2-12", "One extra hour each side", "discard", None, None, None),
    (branch, "exp61: rolling_window 20→15", "Faster cluster detection", "discard", None, None, None),
    (branch, "exp62: Simplify to just HybridEdge", "HybridEdge dominates both ThresholdEdge and GoodHoursExtended", "keep", None, None, None),
    (branch, "exp63: Add ThresholdEdge(0.55)", "Test if ungated wide threshold beats HybridEdge", "discard", None, None, None),
    (branch, "exp64: AdaptiveHoursEdgeStrategy", "Learn good hours from training data (above-median cheap density)", "discard", None, None, None),
    (branch, "exp65: MultiWindowEdgeStrategy", "Data-driven good hours {2,3,4,5,6,8,20,22,23}", "keep", None, None, None),
    (branch, "exp66: Add UTC 20-22 window", "US evening sports hours, 30–35% cheap extended execution", "keep", None, None, None),
    (branch, "exp67: Market-switch gate", "Skip extended buys at first event from any market switch", "keep", None, None, None),
    (branch, "exp68: extended_threshold 0.58→0.60", "With market-switch gate, 0.60 continuation trades positive", "keep", None, None, None),
    (branch, "exp69: extended_threshold 0.60→0.62", "Push further with market-switch gate", "discard", None, None, None),
    (branch, "exp70: cheap_fraction_min 0.40→0.30 + gate", "Gate filters bad trades so more permissive cluster might work", "discard", None, None, None),
    (branch, "exp71: Add VWAP signal", "Recent VWAP ≤ 0.30 as third OR condition for extended buying", "keep", None, None, None),
    (branch, "exp72: rolling_window 20→10", "Faster cluster detection with market-switch gate", "discard", None, None, None),
    (branch, "exp73: rolling_window 10→5", "Even faster cluster detection", "discard", None, None, None),
    (branch, "exp74: rolling_window 5→3", "Very short window", "discard", None, None, None),
    (branch, "exp75: cheap_fraction_min 0.40→0.60, window=5", "Test higher threshold with shorter window", "discard", None, None, None),
    (branch, "exp76: Per-market cluster signal", "Qualify extended range when market's own last 5 events ≥40% cheap", "discard", None, None, None),
    (branch, "exp77: Declining within-run price", "Market downtrend as extended-range qualifier", "discard", None, None, None),
    (branch, "exp78: Long-run bonus", "Extend to 0.60 when run_length ≥ 6 consecutive same-market events", "discard", None, None, None),
    (branch, "exp79: Tiny-trade extended range", "size ≤ 32 in (0.45, 0.60) continuation qualifies", "discard", None, None, None),
    (branch, "exp80: Diverse-market signal", "≥3 distinct markets in last 5 events qualifies extended range", "discard", None, None, None),
    (branch, "exp81: Single-market-run filter", "Block extended range when n_distinct < 2 in last 5", "keep", None, None, None),
    (branch, "Simplify: Remove VWAP signal", "Negligible effect, adds complexity", "keep", None, None, None),
]

# ─── mar24-d538 ──────────────────────────────────────────────────────────────
branch = "mar24-d538"
rows += [
    (branch, "Baseline", "threshold_edge + mean_reversion + online_logistic_like", "keep", 0.2463, 782, 3.39),
    (branch, "TrendFilteredThreshold", "Skip YES when all 3 recent moves are falling (informed sellers)", "keep", 0.2465, 783, 3.40),
    (branch, "BandedThresholdStrategy", "Price floor 0.20 to avoid losing near-zero trades", "discard", 0.2202, 555, 3.13),
    (branch, "ConfidenceScaled", "2x size in (0.20–0.42), 0.5x below 0.20 + trend filter", "discard", 0.2348, 718, 3.13),
    (branch, "YesOnlyMeanReversion", "Remove NO trades to eliminate losses in 0.2–0.4 range", "discard", 0.2463, 782, 3.39),
    (branch, "InformedFlowFilter", "Cooldown after large downward trades (informed NO buyer signal)", "discard", 0.2465, 783, 3.40),
    (branch, "ExitOnReverse", "Sell YES at 0.55 to recycle capital; sell NO at 0.45", "discard", 0.2463, 782, 3.39),
    (branch, "CapitalRecycle", "Sell near-cap cheap positions when entering sweet spot", "discard", 0.2463, 784, 0.82),
    (branch, "WiderYesThreshold", "YES up to 0.50, NO above 0.65 — test YES-bias in mid-range", "discard", 0.2463, 782, 3.39),
    (branch, "OptimalSizedThreshold", "fit() sets reduced cheap-phase size to preserve cap room", "discard", 0.2465, 782, 3.39),
    (branch, "FirstPriceQuality", "2x size for sweet-start markets, 0.5x for cheap-start", "discard", 0.2348, 718, 3.13),
    (branch, "FitMarketSelect", "1x for sweet-crossing markets, 0.1x for pure-cheap markets", "discard", 0.1843, 338, 2.41),
    (branch, "OnlineSweetDetect", "Real-time 0.20 crossing detection; 0.05x until crossed, 1x after", "discard", 0.2237, 572, 3.22),
    (branch, "WhaleFollow", "2x YES after large bullish trade (10x median); 0.3x after large bearish", "discard", 0.2436, 782, 3.27),
    (branch, "StableCheapBoost", "1.5x for 50+ consecutive cheap events; 0.4x for fresh cheap", "discard", 0.2379, 771, 3.07),
    (branch, "HighWaterMark", "Skip YES in markets where max_price_ever < 0.10", "keep", 0.2680, 894, 3.90),
    (branch, "HighWaterMark (threshold 0.20)", "More aggressive HWM filter — max < 0.20", "discard", 0.2202, 555, 3.13),
    (branch, "HWMTrendNo", "YES = HWM viability; NO = only when downtrend confirmed", "discard", 0.2425, 676, 3.63),
    (branch, "TieredHWM", "Require max_price ≥ 0.25 for cheap YES; 0.10 for sweet zone", "discard", 0.2202, 385, 1.74),
    (branch, "HWMFluctNo", "YES = HWM viability; NO = only markets also been cheap (min < 0.40)", "discard", 0.2420, 364, 1.64),
    (branch, "HWMBoost (threshold 0.44)", "YES-only, filter markets with max < 0.43 to hit sweet-zone only", "discard", 0.2680, 240, 0.91),
    (branch, "HWMHighNo (NO threshold 0.70)", "Raise NO threshold to skip YES-resolver false NO trades", "discard", 0.2680, 276, 1.21),
    (branch, "HWMSweetMR", "HWM filter + sweet-zone YES via mean-reversion; no NO trades", "discard", 0.2610, 462, 2.57),
    (branch, "HWMSweetMom", "HWM filter + sweet-zone YES only when price dips AND slope positive", "discard", 0.2442, 259, 1.51),
    (branch, "HWMImproved", "Skip YES at new-price-lows + raise NO threshold to 0.73", "keep", 0.3010, 330, 1.75),
]

# ─── mar24-shrey (human experiments) ─────────────────────────────────────────
branch = "mar24-shrey (human)"
rows += [
    (branch, "ContraryExtremesStrategy", "Buy YES ≤ 0.20, NO ≥ 0.80 — extreme contrarian", "discard", None, None, None),
    (branch, "EMACrossoverStrategy", "Short/long EMA crossover momentum signal", "discard", None, None, None),
    (branch, "EMA + price guardrail", "Skip bad contexts for EMA crossover", "discard", None, None, None),
    (branch, "RSIStrategy", "Overbought/oversold with price guardrail", "discard", None, None, None),
    (branch, "MomentumReverseStrategy", "Contrarian bounce on sharp moves", "discard", None, None, None),
    (branch, "MultiTimeframeMeanReversionStrategy", "Double confirmation across multiple timeframes", "discard", None, None, None),
    (branch, "OrderFlowImbalanceStrategy", "Large-size confirmation at extremes", "discard", None, None, None),
    (branch, "HighConvictionLogisticStrategy", "Logistic regression with edge > 0.10 threshold", "discard", None, None, None),
    (branch, "BenchmarkPassiveStrategy", "Near-zero baseline widener (reward hack test)", "discard", None, None, None),
    (branch, "MidRangeStrategy", "Trades 0.40–0.60 range (baseline widener test)", "discard", None, None, None),
    (branch, "TrendFollowingBadStrategy", "Intentional negative baseline widener (reward hack test)", "discard", None, None, None),
    (branch, "ConfirmedExtremeStrategy", "Extreme price + RSI double filter", "discard", None, None, None),
    (branch, "RandomWalkStrategy", "Pure noise baseline (reward hack test)", "discard", None, None, None),
    (branch, "NoisyMeanReversionStrategy", "Short window mediocre baseline", "discard", None, None, None),
    (branch, "MicroThresholdStrategy", "Near-zero baseline at 0.48/0.52", "discard", None, None, None),
]

# ─── Write markdown table ──────────────────────────────────────────────────────
lines = []
lines.append("# All Experiments — Master Strategy Table\n")
lines.append(f"**Total experiments: {len(rows)}**\n")
lines.append("Columns: Branch | Strategy | Description | Result | Score | PnL | Sharpe\n")
lines.append("---\n")

# Group by branch
from itertools import groupby
from operator import itemgetter

lines.append("| # | Branch | Strategy | Description | Result | Score | PnL | Sharpe |")
lines.append("|---|--------|----------|-------------|--------|-------|-----|--------|")

n = 0
prev_branch = None
for branch, strat, desc, result, score, pnl, sharpe in rows:
    n += 1
    score_str = f"{score:.4f}" if score else "—"
    pnl_str = f"${pnl:,.0f}" if pnl else "—"
    sharpe_str = f"{sharpe:.2f}" if sharpe else "—"
    result_icon = "✅ keep" if result == "keep" else "❌ discard"
    branch_disp = f"**{branch}**" if branch != prev_branch else ""
    prev_branch = branch
    lines.append(f"| {n} | {branch_disp} | {strat} | {desc} | {result_icon} | {score_str} | {pnl_str} | {sharpe_str} |")

content = "\n".join(lines)
with open("STRATEGY_TABLE.md", "w") as f:
    f.write(content)

print(f"Saved STRATEGY_TABLE.md — {len(rows)} strategies total")

# Count by result
keeps = sum(1 for r in rows if r[3] == "keep")
discards = sum(1 for r in rows if r[3] == "discard")
print(f"  Keep: {keeps}  |  Discard: {discards}")
