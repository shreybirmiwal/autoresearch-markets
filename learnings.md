# Learnings Journal

[+0.0151] baseline: threshold_edge + mean_reversion + online_logistic → score 0.1516, sharpe 1.11, pnl $323 | WHY: All three strategies pile into YES<0.42 without distinguishing quality. The <0.2 bucket has 22k trades at -0.002 avg pnl — massive noise. NO trades in 0.4-0.6 range are losing (-0.20 avg pnl) because that's genuinely uncertain territory.

[+0.0151 → +0.0167] banded_threshold: YES only in 0.20-0.42, NO only above 0.60 → score 0.1667, sharpe 1.68 (+0.57) | WHY: Filtering out YES<0.20 removes 22k near-zero-pnl trades that dilute signal quality. Filtering out NO in 0.4-0.6 removes -0.20 avg pnl trades. Result: same PnL ($337 vs $323) with much lower variance → sharpe almost +50%. Mechanism validated: near-zero prices are near-resolved-NO, not genuine opportunities.

[+0.0192] exit_logic: ExitAwareBandedStrategy - exit YES above 0.55, NO floor 0.65 → score 0.1859, sharpe 1.98, pnl $471 | WHY: Exit trades (exit_aware_banded|yes|(0.4, 0.6]) became the BEST context at 0.787 avg pnl. Recycling capital means we can enter new positions rather than holding indefinitely. Raising NO floor to 0.65 eliminated the gray-zone losing trades. Confirmed: exit logic is extremely valuable for prediction markets.

[+0.0000] momentum_filter: skip YES buys in downtrend (last 5 falling) → score 0.1853, NO_IMPROVEMENT | WHY: In prediction markets, even falling prices in 0.20-0.42 can recover. The downtrend filter removes too many profitable entries — the very act of price being in 0.20-0.42 is already a signal. Falling within that range doesn't mean it won't recover. Momentum filtering hurt more than helped.

[+0.0013] confidence_scaled_sizing: contracts = max(1, (0.50-p)/0.10) → score 0.1872, sharpe 1.79 (-0.19), pnl $531 (+$60) | WHY: Cheaper YES contracts have higher expected gain per contract (buying at 0.20 = gain of 0.80 if YES resolves vs 0.58 at 0.42). Scaling up size at lower prices increased total PnL. Sharpe dropped because variance also increased proportionally. The score formula weights both equally so net improvement. The 0.45*pnl/5000 component benefited more than 0.45*sharpe/20 was hurt.
