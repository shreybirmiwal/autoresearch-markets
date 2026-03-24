from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from markets_research.attribution import build_trade_attribution, propose_next_hypotheses, summarize_win_loss_contexts
from markets_research.backtest import BacktestConfig, run_backtest
from markets_research.scoring import compute_metrics, rank_experiments
from markets_research.strategies import Strategy, default_strategy_registry


def _load_latest_trades(data_root: Path) -> pd.DataFrame:
    trade_files = sorted((data_root / "trades").glob("**/*.parquet"))
    if not trade_files:
        raise FileNotFoundError("No parquet trades found. Run Kalshi ingest first.")
    dfs = [pd.read_parquet(p) for p in trade_files]
    out = pd.concat(dfs, ignore_index=True)
    out["event_ts"] = pd.to_datetime(out["event_ts"], utc=True)
    return out


def _load_latest_markets(data_root: Path) -> pd.DataFrame:
    market_files = sorted((data_root / "markets").glob("**/*.parquet"))
    if not market_files:
        raise FileNotFoundError("No parquet markets found. Run Kalshi ingest first.")
    return pd.concat([pd.read_parquet(p) for p in market_files], ignore_index=True)


def _walk_forward_splits(df: pd.DataFrame, folds: int = 3) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    df = df.sort_values("event_ts").reset_index(drop=True)
    n = len(df)
    chunk = max(1, n // (folds + 1))
    splits: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    for i in range(1, folds + 1):
        train_end = i * chunk
        test_end = min(n, (i + 1) * chunk)
        if test_end <= train_end:
            continue
        splits.append((df.iloc[:train_end].copy(), df.iloc[train_end:test_end].copy()))
    return splits


def run_tournament(
    data_root: Path,
    output_dir: Path,
    backtest_cfg: BacktestConfig,
    strategies: list[Strategy] | None = None,
) -> dict[str, Path]:
    strategies = strategies or default_strategy_registry()
    trades = _load_latest_trades(data_root)
    markets = _load_latest_markets(data_root)
    settlement = markets.set_index("market_id")["settlement_price_yes"].fillna(0.5)

    rows: list[dict[str, float | str]] = []
    attribution_outputs: list[pd.DataFrame] = []
    for strategy in strategies:
        strategy.reset()
        fold_metrics: list[dict[str, float]] = []
        for train_df, test_df in _walk_forward_splits(trades, folds=3):
            labels = train_df["market_id"].map(settlement).fillna(train_df["price_yes"]).astype(float)
            train_events = train_df.assign(label=labels).to_dict(orient="records")
            strategy.fit(train_events)
            equity, fills = run_backtest(test_df, settlement, strategy, backtest_cfg)
            attrib = build_trade_attribution(fills, settlement)
            attribution_outputs.append(attrib.assign(strategy=strategy.name))
            fold_metrics.append(compute_metrics(equity, attrib))
        if fold_metrics:
            agg = pd.DataFrame(fold_metrics).mean(numeric_only=True).to_dict()
            rows.append({"strategy": strategy.name, **agg})

    metrics_df = pd.DataFrame(rows)
    leaderboard = rank_experiments(metrics_df)
    all_attr = pd.concat(attribution_outputs, ignore_index=True) if attribution_outputs else pd.DataFrame()
    context = summarize_win_loss_contexts(all_attr)
    hypotheses = propose_next_hypotheses(context)

    output_dir.mkdir(parents=True, exist_ok=True)
    leaderboard_path = output_dir / "leaderboard.csv"
    attribution_path = output_dir / "trade_attribution.csv"
    report_path = output_dir / "report.json"
    leaderboard.to_csv(leaderboard_path, index=False)
    all_attr.to_csv(attribution_path, index=False)
    report_path.write_text(
        json.dumps(
            {
                "best_strategy": leaderboard.iloc[0]["strategy"] if not leaderboard.empty else None,
                "top_wins": context["top_wins"].to_dict(orient="records"),
                "top_losses": context["top_losses"].to_dict(orient="records"),
                "next_hypotheses": hypotheses,
                "robustness_checks": run_robustness_checks(trades, settlement, strategies, backtest_cfg),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"leaderboard": leaderboard_path, "attribution": attribution_path, "report": report_path}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run prediction-market strategy tournament.")
    parser.add_argument("--data-root", type=Path, default=Path("data_lake"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--initial-cash", type=float, default=10_000.0)
    parser.add_argument("--fee-bps", type=float, default=10.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--latency-events", type=int, default=1)
    parser.add_argument("--max-position-contracts", type=float, default=500.0)
    args = parser.parse_args()

    paths = run_tournament(
        data_root=args.data_root,
        output_dir=args.output_dir,
        backtest_cfg=BacktestConfig(
            initial_cash=args.initial_cash,
            max_position_contracts=args.max_position_contracts,
            fee_bps=args.fee_bps,
            slippage_bps=args.slippage_bps,
            latency_events=args.latency_events,
        ),
    )
    print("wrote artifacts:")
    for name, path in paths.items():
        print(f"  {name}: {path}")


def run_robustness_checks(
    trades: pd.DataFrame,
    settlement: pd.Series,
    strategies: list[Strategy],
    base_cfg: BacktestConfig,
) -> list[dict[str, float | str]]:
    checks: list[dict[str, float | str]] = []
    for strategy in strategies:
        for slippage_mult in [0.5, 1.0, 2.0]:
            cfg = BacktestConfig(
                initial_cash=base_cfg.initial_cash,
                max_position_contracts=base_cfg.max_position_contracts,
                fee_bps=base_cfg.fee_bps,
                slippage_bps=base_cfg.slippage_bps * slippage_mult,
                latency_events=base_cfg.latency_events,
            )
            _, test_df = _walk_forward_splits(trades, folds=1)[0]
            strategy.reset()
            equity, fills = run_backtest(test_df, settlement, strategy, cfg)
            metrics = compute_metrics(equity, fills)
            checks.append(
                {
                    "strategy": strategy.name,
                    "scenario": f"slippage_x{slippage_mult}",
                    "final_pnl": metrics["final_pnl"],
                    "sharpe": metrics["sharpe"],
                }
            )
    return checks


if __name__ == "__main__":
    main()

