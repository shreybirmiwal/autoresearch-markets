from __future__ import annotations

import pandas as pd


def build_trade_attribution(fills: pd.DataFrame, settlement: pd.Series) -> pd.DataFrame:
    if fills.empty:
        return fills
    out = fills.copy()
    out["settle_yes"] = out["market_id"].map(settlement).fillna(0.5)
    out["edge"] = out["settle_yes"] - out["exec_yes_price"]
    out["pnl_per_contract"] = out["edge"].where(out["side"] == "yes", -out["edge"])
    out["trade_pnl"] = out["pnl_per_contract"] * out["contracts"] - out["fee"]
    out["confidence_bucket"] = pd.cut(out["exec_yes_price"], bins=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], include_lowest=True)
    return out


def summarize_win_loss_contexts(attribution: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if attribution.empty:
        empty = pd.DataFrame(columns=["context", "count", "avg_trade_pnl"])
        return {"top_wins": empty, "top_losses": empty}
    grouped = (
        attribution.groupby(["reason", "side", "confidence_bucket"], observed=True)["trade_pnl"]
        .agg(["count", "mean"])
        .reset_index()
        .rename(columns={"mean": "avg_trade_pnl"})
    )
    grouped["context"] = (
        grouped["reason"].astype(str)
        + "|"
        + grouped["side"].astype(str)
        + "|"
        + grouped["confidence_bucket"].astype(str)
    )
    wins = grouped.sort_values("avg_trade_pnl", ascending=False).head(10)[["context", "count", "avg_trade_pnl"]]
    losses = grouped.sort_values("avg_trade_pnl", ascending=True).head(10)[["context", "count", "avg_trade_pnl"]]
    return {"top_wins": wins, "top_losses": losses}


def propose_next_hypotheses(context_summary: dict[str, pd.DataFrame]) -> list[str]:
    hypotheses: list[str] = []
    wins = context_summary["top_wins"]
    losses = context_summary["top_losses"]
    if not wins.empty:
        hypotheses.append(
            f"Increase size where contexts similar to best bucket: {wins.iloc[0]['context']}."
        )
    if not losses.empty:
        hypotheses.append(
            f"Add guardrail or avoid contexts similar to worst bucket: {losses.iloc[0]['context']}."
        )
    hypotheses.append("Sweep slippage and latency assumptions to keep only robust strategies.")
    hypotheses.append("Perturb thresholds +/-10% around top strategy params and re-run walk-forward.")
    return hypotheses

