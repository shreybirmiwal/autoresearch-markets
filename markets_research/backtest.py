from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestConfig:
    initial_cash: float = 10_000.0
    max_position_contracts: float = 500.0
    fee_bps: float = 10.0
    slippage_bps: float = 5.0
    latency_events: int = 1


@dataclass(frozen=True)
class Order:
    market_id: str
    side: str
    contracts: float
    reason: str


def _as_yes_price(side: str, yes_price: float) -> float:
    return yes_price if side == "yes" else 1.0 - yes_price


def run_backtest(
    trades_df: pd.DataFrame,
    market_settlement: pd.Series,
    strategy: Any,
    cfg: BacktestConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if trades_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    df = trades_df.sort_values("event_ts").reset_index(drop=True).copy()
    holdings_yes: dict[str, float] = {}
    holdings_no: dict[str, float] = {}
    cash = cfg.initial_cash
    pending_orders: list[tuple[int, Order]] = []
    equity_rows: list[dict[str, float]] = []
    fills: list[dict[str, Any]] = []

    for i, row in df.iterrows():
        to_execute = [entry for entry in pending_orders if entry[0] <= i]
        pending_orders = [entry for entry in pending_orders if entry[0] > i]
        for _, order in to_execute:
            mid = float(row["price_yes"])
            if order.side == "yes":
                current = holdings_yes.get(order.market_id, 0.0)
            else:
                current = holdings_no.get(order.market_id, 0.0)
            new_pos = float(np.clip(current + order.contracts, 0.0, cfg.max_position_contracts))
            executed_contracts = float(new_pos - current)
            if executed_contracts == 0:
                continue
            # Slippage is always adverse: buys pay more, sells receive less.
            # A sell is indicated by negative order.contracts (position reduction).
            is_buy = order.contracts >= 0
            slip = cfg.slippage_bps / 10_000.0
            if order.side == "yes":
                exec_yes = mid + (slip if is_buy else -slip)
            else:
                exec_yes = mid + (-slip if is_buy else slip)
            exec_yes = float(np.clip(exec_yes, 0.001, 0.999))
            price_per_contract = _as_yes_price(order.side, exec_yes)
            notional = abs(executed_contracts) * price_per_contract
            fee = notional * (cfg.fee_bps / 10_000.0)
            if is_buy:
                cash -= notional + fee
            else:
                cash += notional - fee
            if order.side == "yes":
                holdings_yes[order.market_id] = new_pos
            else:
                holdings_no[order.market_id] = new_pos
            fills.append(
                {
                    "event_ts": row["event_ts"],
                    "market_id": order.market_id,
                    "ticker": row["ticker"],
                    "side": order.side,
                    # Positive = open/buy, negative = close/sell
                    "contracts": float(executed_contracts),
                    "exec_yes_price": exec_yes,
                    "fee": fee,
                    "reason": order.reason,
                }
            )

        state = {
            "event_ts": row["event_ts"],
            "market_id": row["market_id"],
            "ticker": row["ticker"],
            "yes_price": float(row["price_yes"]),
            "size": float(row["size"]),
            "position_yes_contracts": holdings_yes.get(str(row["market_id"]), 0.0),
            "position_no_contracts": holdings_no.get(str(row["market_id"]), 0.0),
        }
        order = strategy.on_event(state)
        if order is not None:
            execute_idx = i + max(0, int(cfg.latency_events))
            pending_orders.append((execute_idx, order))

        mtm = cash
        for market_id, contracts in holdings_yes.items():
            settle_yes = float(market_settlement.get(market_id, 0.5))
            mtm += contracts * settle_yes
        for market_id, contracts in holdings_no.items():
            settle_yes = float(market_settlement.get(market_id, 0.5))
            mtm += contracts * (1.0 - settle_yes)
        equity_rows.append({"event_ts": row["event_ts"], "equity": mtm, "cash": cash})

    equity = pd.DataFrame(equity_rows)
    fills_df = pd.DataFrame(fills)
    return equity, fills_df

