"""
Analyze the current best strategy (ExitAwareBandedStrategy with confidence scaling)
on the last 3 months of data.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import timezone

from markets_research.backtest import BacktestConfig, run_backtest
from markets_research.strategies import ExitAwareBandedStrategy
from markets_research.scoring import compute_metrics


DATA_ROOT = Path(__file__).parent.parent / "data_lake"
OUT_DIR = Path(__file__).parent


def load_data():
    trade_files = sorted((DATA_ROOT / "trades").glob("**/*.parquet"))
    market_files = sorted((DATA_ROOT / "markets").glob("**/*.parquet"))
    trades = pd.concat([pd.read_parquet(p, columns=["market_id", "ticker", "event_ts", "price_yes", "size"])
                        for p in trade_files], ignore_index=True)
    markets = pd.concat([pd.read_parquet(p) for p in market_files], ignore_index=True)
    trades["event_ts"] = pd.to_datetime(trades["event_ts"], utc=True)
    return trades, markets


def main():
    print("Loading data...")
    trades, markets = load_data()

    print(f"Full dataset: {len(trades):,} trades from {trades['event_ts'].min().date()} to {trades['event_ts'].max().date()}")

    # Last 3 months
    cutoff = trades["event_ts"].max() - pd.Timedelta(days=91)
    trades_3m = trades[trades["event_ts"] >= cutoff].copy()
    print(f"Last 3 months (from {cutoff.date()}): {len(trades_3m):,} trades across {trades_3m['market_id'].nunique()} markets")

    # Settlement prices (settled markets only)
    settlement_raw = (
        markets.sort_values("snapshot_id")
        .groupby("market_id", as_index=False)["settlement_price_yes"]
        .last()
        .set_index("market_id")["settlement_price_yes"]
    )
    settlement = settlement_raw.dropna()
    settled_ids = set(settlement.index.astype(str))

    trades_3m = trades_3m[trades_3m["market_id"].astype(str).isin(settled_ids)].copy()
    print(f"After settlement filter: {len(trades_3m):,} trades across {trades_3m['market_id'].nunique()} settled markets")

    # Top 200 markets by volume (same as default tournament)
    counts = trades_3m["market_id"].value_counts()
    top200 = set(counts.head(200).index)
    trades_3m = trades_3m[trades_3m["market_id"].isin(top200)].copy()
    print(f"After top-200 filter: {len(trades_3m):,} trades")

    # Run best strategy
    strategy = ExitAwareBandedStrategy()
    cfg = BacktestConfig(
        initial_cash=10_000.0,
        max_position_contracts=500.0,
        fee_bps=10.0,
        slippage_bps=5.0,
        latency_events=1,
    )

    print("\nRunning ExitAwareBandedStrategy (confidence scaling, exit@0.60, NO>0.75)...")
    equity, fills = run_backtest(trades_3m, settlement, strategy, cfg)

    # --- Stats ---
    metrics = compute_metrics(equity, fills)

    print("\n" + "="*55)
    print("  RESULTS: Last 3 Months")
    print("="*55)
    print(f"  Date range      : {trades_3m['event_ts'].min().date()} → {trades_3m['event_ts'].max().date()}")
    print(f"  Total trades    : {int(metrics['num_trades']):,}")
    print(f"  Total PnL       : ${metrics['final_pnl']:+.2f}")
    print(f"  Return          : {metrics['final_pnl'] / 10_000 * 100:.2f}%  (on $10k capital)")
    print(f"  Sharpe          : {metrics['sharpe']:.4f}")
    print(f"  Max drawdown    : {metrics['max_drawdown']*100:.1f}%")
    print(f"  Hit rate        : {metrics['hit_rate']*100:.1f}%")
    print(f"  Turnover        : {metrics['turnover_contracts']:.0f} contracts")
    print("="*55)

    if not fills.empty:
        print(f"\n  Trade breakdown:")
        side_counts = fills["side"].value_counts()
        for side, cnt in side_counts.items():
            print(f"    {side:5s}: {cnt:,} fills")

        reason_counts = fills["reason"].value_counts()
        print(f"\n  Fill reasons:")
        for reason, cnt in reason_counts.items():
            print(f"    {reason:40s}: {cnt:,}")

        avg_contracts = fills["contracts"].abs().mean()
        print(f"\n  Avg contracts/fill : {avg_contracts:.2f}")
        print(f"  Total fees paid    : ${fills['fee'].sum():.2f}")

        if "exec_yes_price" in fills.columns:
            buy_fills = fills[fills["contracts"] > 0]
            if not buy_fills.empty:
                print(f"  Avg entry yes price: {buy_fills['exec_yes_price'].mean():.4f}")

    # --- PnL Chart ---
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle("ExitAwareBandedStrategy — Last 3 Months\n"
                 f"(YES in (0.20,0.42] + conf scaling | exit@0.60 | NO>0.75)",
                 fontsize=13, fontweight="bold")

    # 1. Equity curve
    ax1 = axes[0]
    eq_ts = pd.to_datetime(equity["event_ts"])
    ax1.plot(eq_ts, equity["equity"], color="#2196F3", linewidth=1.2, label="Portfolio equity")
    ax1.axhline(10_000, color="gray", linestyle="--", linewidth=0.8, label="Starting capital $10k")
    ax1.fill_between(eq_ts, 10_000, equity["equity"],
                     where=equity["equity"] >= 10_000, alpha=0.15, color="green")
    ax1.fill_between(eq_ts, 10_000, equity["equity"],
                     where=equity["equity"] < 10_000, alpha=0.15, color="red")
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.set_title(f"Equity Curve  (Final PnL: ${metrics['final_pnl']:+.2f})")
    ax1.legend(fontsize=9)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax1.grid(alpha=0.3)

    # 2. Drawdown
    ax2 = axes[1]
    rolling_max = equity["equity"].cummax()
    drawdown = ((equity["equity"] - rolling_max) / rolling_max.replace(0, np.nan)).fillna(0) * 100
    ax2.fill_between(eq_ts, drawdown, 0, color="#F44336", alpha=0.6)
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_title(f"Drawdown  (Max: {metrics['max_drawdown']*100:.1f}%)")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax2.grid(alpha=0.3)

    # 3. Trade entry prices histogram
    ax3 = axes[2]
    if not fills.empty:
        buy_fills = fills[fills["contracts"] > 0]
        if not buy_fills.empty:
            yes_buys = buy_fills[buy_fills["side"] == "yes"]["exec_yes_price"]
            no_buys = buy_fills[buy_fills["side"] == "no"]["exec_yes_price"]
            if not yes_buys.empty:
                ax3.hist(yes_buys, bins=40, alpha=0.7, color="#4CAF50", label=f"YES buys (n={len(yes_buys):,})")
            if not no_buys.empty:
                ax3.hist(no_buys, bins=40, alpha=0.7, color="#FF9800", label=f"NO buys YES-price (n={len(no_buys):,})")
    ax3.set_xlabel("Execution Yes Price")
    ax3.set_ylabel("Count")
    ax3.set_title("Entry Price Distribution")
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.3)

    plt.tight_layout()
    chart_path = OUT_DIR / "pnl_chart_last3months.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    print(f"\nChart saved to: {chart_path}")

    # Save fills CSV
    fills_path = OUT_DIR / "fills_last3months.csv"
    fills.to_csv(fills_path, index=False)
    print(f"Fills saved to: {fills_path}")

    equity_path = OUT_DIR / "equity_last3months.csv"
    equity.to_csv(equity_path, index=False)
    print(f"Equity saved to: {equity_path}")


if __name__ == "__main__":
    main()
