"""
Experiment entrypoint for prediction-market autoresearch.

Karpathy-style role: run one experiment, score it, emit artifacts.
Agent modifies: markets_research/strategies.py
Agent does NOT modify: this file, prepare.py, or any other markets_research/ module.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd

from markets_research.backtest import BacktestConfig
from markets_research.experiment import run_tournament


def _read_best_score(results_tsv: Path) -> float | None:
    """Return the highest composite score among all KEEP runs logged so far."""
    if not results_tsv.exists():
        return None
    best: float | None = None
    with open(results_tsv, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row.get("status", "").strip().lower() == "keep":
                try:
                    s = float(row["score"])
                    if best is None or s > best:
                        best = s
                except (ValueError, KeyError):
                    pass
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one prediction-market experiment tournament.")
    parser.add_argument("--data-root", type=Path, default=Path("data_lake"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--results-tsv", type=Path, default=Path("results.tsv"))
    parser.add_argument("--market-category", type=str, default=None)
    parser.add_argument("--top-n-markets", type=int, default=200)
    parser.add_argument("--initial-cash", type=float, default=10_000.0)
    parser.add_argument("--fee-bps", type=float, default=10.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--latency-events", type=int, default=1)
    parser.add_argument("--max-position-contracts", type=float, default=500.0)
    parser.add_argument("--sample-stride", type=int, default=1,
                        help="Keep 1-in-N trade rows (e.g. 5 = 5x fewer rows, much faster). Default 1 = all rows.")
    parser.add_argument("--skip-robustness", action="store_true",
                        help="Skip robustness checks (saves ~3x strategies extra backtests).")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Hard cap on rows loaded (e.g. 5000 for smoke-testing).")
    args = parser.parse_args()

    artifacts = run_tournament(
        data_root=args.data_root,
        output_dir=args.output_dir,
        backtest_cfg=BacktestConfig(
            initial_cash=args.initial_cash,
            max_position_contracts=args.max_position_contracts,
            fee_bps=args.fee_bps,
            slippage_bps=args.slippage_bps,
            latency_events=args.latency_events,
        ),
        market_category=args.market_category,
        top_n_markets=args.top_n_markets,
        sample_stride=args.sample_stride,
        skip_robustness=args.skip_robustness,
        max_rows=args.max_rows,
    )

    # --- print artifact paths ---
    print("artifacts:")
    for name, path in artifacts.items():
        print(f"  {name}: {path}")

    # --- score comparison vs best known KEEP ---
    leaderboard = pd.read_csv(artifacts["leaderboard"])
    if leaderboard.empty or "score" not in leaderboard.columns:
        print("\nno ranked strategies (too few trades or all filtered out)")
        return

    best_row = leaderboard.iloc[0]
    score = float(best_row["score"])
    pnl = float(best_row.get("final_pnl", float("nan")))
    sharpe = float(best_row.get("sharpe", float("nan")))
    strategy_name = str(best_row["strategy"])

    prev_best = _read_best_score(args.results_tsv)

    print("\n--- result ---")
    print(f"best_strategy : {strategy_name}")
    print(f"score         : {score:.4f}   (0.45*sharpe/20 + 0.45*pnl/5000 + 0.10*(1+drawdown))")
    print(f"final_pnl     : {pnl:.2f}")
    print(f"sharpe        : {sharpe:.4f}")

    if prev_best is None:
        verdict = "FIRST_RUN"
        print(f"\nstatus        : {verdict}  (no previous KEEP in results.tsv)")
    elif score > prev_best:
        verdict = "IMPROVED"
        print(f"\nstatus        : {verdict}  ({score:.4f} > prev best {prev_best:.4f})")
    else:
        verdict = "NO_IMPROVEMENT"
        print(f"\nstatus        : {verdict}  ({score:.4f} <= prev best {prev_best:.4f})")

    print("\n--- suggested next action ---")
    if verdict in ("FIRST_RUN", "IMPROVED"):
        print("  git add markets_research/strategies.py")
        print("  git commit -m 'keep: <one-line description of what changed>'")
        print(f"  then append to results.tsv: <commit>\\t{score:.4f}\\t{pnl:.2f}\\t{sharpe:.4f}\\tkeep\\t<description>")
    else:
        print("  git checkout markets_research/strategies.py   # discard this attempt")
        print(f"  then append to results.tsv: <commit>\\t{score:.4f}\\t{pnl:.2f}\\t{sharpe:.4f}\\tdiscard\\t<description>")


if __name__ == "__main__":
    main()
