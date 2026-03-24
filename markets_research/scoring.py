from __future__ import annotations

import math

import numpy as np
import pandas as pd


def compute_metrics(equity: pd.DataFrame, fills: pd.DataFrame) -> dict[str, float]:
    if equity.empty:
        return {
            "final_pnl": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "turnover_contracts": 0.0,
            "num_trades": 0.0,
            "hit_rate": 0.0,
        }
    returns = equity["equity"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    ret_std = float(returns.std())
    sharpe = 0.0 if ret_std <= 1e-12 else float((returns.mean() / ret_std) * math.sqrt(252))
    rolling_max = equity["equity"].cummax()
    drawdown = ((equity["equity"] - rolling_max) / rolling_max.replace(0.0, np.nan)).fillna(0.0)
    max_drawdown = float(drawdown.min())
    final_pnl = float(equity["equity"].iloc[-1] - equity["equity"].iloc[0])
    turnover = float(fills["contracts"].abs().sum()) if not fills.empty else 0.0
    num_trades = float(len(fills))
    hit_rate = 0.0
    if not fills.empty and "settle_yes" in fills.columns:
        wins = np.where(
            fills["side"] == "yes",
            fills["settle_yes"] > fills["exec_yes_price"],
            (1.0 - fills["settle_yes"]) > (1.0 - fills["exec_yes_price"]),
        )
        hit_rate = float(np.mean(wins)) if len(wins) else 0.0
    return {
        "final_pnl": final_pnl,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "turnover_contracts": turnover,
        "num_trades": num_trades,
        "hit_rate": hit_rate,
    }


def rank_experiments(metrics_df: pd.DataFrame) -> pd.DataFrame:
    if metrics_df.empty:
        return metrics_df
    df = metrics_df.copy()
    if "num_trades" in df.columns:
        df = df[df["num_trades"] >= 10].copy()
    if "max_drawdown" in df.columns:
        df = df[df["max_drawdown"] >= -0.5].copy()
    if df.empty:
        return df
    # Absolute score: fixed reference values so adding "baseline widener" strategies
    # cannot inflate the scores of good ones (unlike z-score normalization which is
    # relative to the current registry).
    #   sharpe / 20      — sharpe of 20 is excellent; scales to [0, 1] for typical range
    #   final_pnl / 5000 — $5k profit on $10k starting capital is a strong result
    #   1 + max_drawdown — drawdown is ≤ 0, so this maps it to [0, 1]
    _SHARPE_REF = 20.0
    _PNL_REF = 5_000.0
    df["score"] = (
        0.45 * df["sharpe"] / _SHARPE_REF
        + 0.45 * df["final_pnl"] / _PNL_REF
        + 0.10 * (1.0 + df["max_drawdown"])
    )
    return df.sort_values("score", ascending=False).reset_index(drop=True)

