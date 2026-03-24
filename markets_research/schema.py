from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Literal

import pandas as pd


Venue = Literal["kalshi", "polymarket"]
Side = Literal["yes", "no"]


@dataclass(frozen=True)
class Market:
    venue: Venue
    market_id: str
    ticker: str
    title: str
    open_ts: datetime | None
    close_ts: datetime | None
    settled_ts: datetime | None
    strike_info: str | None
    category: str | None
    is_resolved: bool
    settlement_price_yes: float | None
    fee_bps: float
    snapshot_id: str


@dataclass(frozen=True)
class TradeEvent:
    venue: Venue
    market_id: str
    ticker: str
    event_ts: datetime
    side: Side
    price_yes: float
    size: float
    trade_id: str
    snapshot_id: str


def _utc(ts: datetime | None) -> datetime | None:
    if ts is None:
        return None
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def validate_markets(df: pd.DataFrame) -> pd.DataFrame:
    required = {
        "venue",
        "market_id",
        "ticker",
        "title",
        "is_resolved",
        "settlement_price_yes",
        "fee_bps",
        "snapshot_id",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"markets schema missing columns: {missing}")
    if not df["venue"].isin(["kalshi", "polymarket"]).all():
        raise ValueError("unsupported venue in markets table")
    return df


def validate_trades(df: pd.DataFrame) -> pd.DataFrame:
    required = {
        "venue",
        "market_id",
        "ticker",
        "event_ts",
        "side",
        "price_yes",
        "size",
        "trade_id",
        "snapshot_id",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"trades schema missing columns: {missing}")
    if not df["venue"].isin(["kalshi", "polymarket"]).all():
        raise ValueError("unsupported venue in trades table")
    if not df["side"].isin(["yes", "no"]).all():
        raise ValueError("unsupported side in trades table")
    return df


def markets_to_frame(markets: list[Market]) -> pd.DataFrame:
    rows = []
    for market in markets:
        row = asdict(market)
        row["open_ts"] = _utc(row["open_ts"])
        row["close_ts"] = _utc(row["close_ts"])
        row["settled_ts"] = _utc(row["settled_ts"])
        rows.append(row)
    df = pd.DataFrame(rows)
    return validate_markets(df) if not df.empty else df


def trades_to_frame(trades: list[TradeEvent]) -> pd.DataFrame:
    rows = []
    for trade in trades:
        row = asdict(trade)
        row["event_ts"] = _utc(row["event_ts"])
        rows.append(row)
    df = pd.DataFrame(rows)
    return validate_trades(df) if not df.empty else df


def utc_from_unix(ts: int | float | None) -> datetime | None:
    if ts is None:
        return None
    return datetime.fromtimestamp(float(ts), tz=timezone.utc)

