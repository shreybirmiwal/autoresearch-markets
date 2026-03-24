from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from markets_research.schema import Market, TradeEvent, markets_to_frame, trades_to_frame, utc_from_unix
from markets_research.storage import SnapshotManifest, new_snapshot_id, write_manifest, write_partitioned_parquet

BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
HISTORICAL_BASE_URL = "https://api.elections.kalshi.com/trade-api/v2/historical"


@dataclass(frozen=True)
class IngestConfig:
    out_dir: Path
    min_ts: int | None
    max_ts: int | None
    limit: int
    retries: int
    backoff_sec: float


def _request_json(url: str, params: dict[str, Any], retries: int, backoff_sec: float) -> dict[str, Any]:
    for attempt in range(retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=60)
            resp.raise_for_status()
            return resp.json()
        except Exception:
            if attempt == retries:
                raise
            time.sleep(backoff_sec * (2**attempt))
    raise RuntimeError("unreachable retry state")


def fetch_cutoff(retries: int, backoff_sec: float) -> dict[str, Any]:
    return _request_json(f"{HISTORICAL_BASE_URL}/cutoff", {}, retries, backoff_sec)


def fetch_trades(cfg: IngestConfig) -> list[TradeEvent]:
    trades: list[TradeEvent] = []
    cursor: str | None = None
    while True:
        params: dict[str, Any] = {"limit": cfg.limit}
        if cfg.min_ts is not None:
            params["min_ts"] = cfg.min_ts
        if cfg.max_ts is not None:
            params["max_ts"] = cfg.max_ts
        if cursor:
            params["cursor"] = cursor
        payload = _request_json(f"{HISTORICAL_BASE_URL}/trades", params, cfg.retries, cfg.backoff_sec)
        for row in payload.get("trades", []):
            price = float(row.get("yes_price") or row.get("price") or 0.0)
            side = "yes" if str(row.get("taker_side", "yes")).lower() in {"yes", "buy"} else "no"
            trades.append(
                TradeEvent(
                    venue="kalshi",
                    market_id=str(row.get("market_ticker") or row.get("ticker") or ""),
                    ticker=str(row.get("market_ticker") or row.get("ticker") or ""),
                    event_ts=utc_from_unix(row.get("created_time") or row.get("ts") or 0) or utc_from_unix(0),
                    side=side,  # type: ignore[arg-type]
                    price_yes=max(0.0, min(1.0, price / 100.0 if price > 1.0 else price)),
                    size=float(row.get("count") or row.get("size") or 0.0),
                    trade_id=str(row.get("trade_id") or row.get("id") or f"trade-{len(trades)}"),
                    snapshot_id="",
                )
            )
        cursor = payload.get("cursor")
        if not cursor:
            break
    return trades


def fetch_markets(cfg: IngestConfig) -> list[Market]:
    markets: list[Market] = []
    cursor: str | None = None
    seen: set[str] = set()
    while True:
        params: dict[str, Any] = {"limit": cfg.limit, "status": "closed"}
        if cursor:
            params["cursor"] = cursor
        payload = _request_json(f"{BASE_URL}/markets", params, cfg.retries, cfg.backoff_sec)
        for row in payload.get("markets", []):
            ticker = str(row.get("ticker", ""))
            if not ticker or ticker in seen:
                continue
            seen.add(ticker)
            markets.append(
                Market(
                    venue="kalshi",
                    market_id=ticker,
                    ticker=ticker,
                    title=str(row.get("title") or row.get("subtitle") or ticker),
                    open_ts=utc_from_unix(row.get("open_time")),
                    close_ts=utc_from_unix(row.get("close_time")),
                    settled_ts=utc_from_unix(row.get("settlement_time")),
                    strike_info=str(row.get("strike_type")) if row.get("strike_type") is not None else None,
                    category=str(row.get("category")) if row.get("category") is not None else None,
                    is_resolved=bool(row.get("status") in {"resolved", "settled", "closed"}),
                    settlement_price_yes=(
                        max(0.0, min(1.0, float(row["settlement_value"]) / 100.0))
                        if row.get("settlement_value") is not None
                        else None
                    ),
                    fee_bps=0.0,
                    snapshot_id="",
                )
            )
        cursor = payload.get("cursor")
        if not cursor:
            break
    return markets


def run(cfg: IngestConfig) -> Path:
    snapshot_id = new_snapshot_id("kalshi")
    cutoff_payload = fetch_cutoff(cfg.retries, cfg.backoff_sec)
    trades = [
        TradeEvent(**{**t.__dict__, "snapshot_id": snapshot_id})  # type: ignore[arg-type]
        for t in fetch_trades(cfg)
    ]
    markets = [
        Market(**{**m.__dict__, "snapshot_id": snapshot_id})  # type: ignore[arg-type]
        for m in fetch_markets(cfg)
    ]
    trades_df = trades_to_frame(trades)
    if not trades_df.empty:
        trades_df["date"] = pd.to_datetime(trades_df["event_ts"], utc=True).dt.strftime("%Y-%m-%d")
    markets_df = markets_to_frame(markets)
    if not markets_df.empty:
        markets_df["date"] = pd.to_datetime(
            markets_df["settled_ts"].fillna(markets_df["close_ts"]).fillna(markets_df["open_ts"]),
            utc=True,
        ).dt.strftime("%Y-%m-%d")

    write_partitioned_parquet(trades_df, cfg.out_dir / "trades" / f"{snapshot_id}.parquet", ["venue", "date"])
    write_partitioned_parquet(markets_df, cfg.out_dir / "markets" / f"{snapshot_id}.parquet", ["venue", "date"])

    manifest = SnapshotManifest(
        snapshot_id=snapshot_id,
        venue="kalshi",
        created_at_utc=pd.Timestamp.utcnow().isoformat(),
        start_ts=cfg.min_ts,
        end_ts=cfg.max_ts,
        records={"trades": int(len(trades_df)), "markets": int(len(markets_df))},
        source=f"historical.cutoff={cutoff_payload}",
    )
    return write_manifest(cfg.out_dir, manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest Kalshi historical data to parquet.")
    parser.add_argument("--out-dir", type=Path, default=Path("data_lake"))
    parser.add_argument("--min-ts", type=int, default=None)
    parser.add_argument("--max-ts", type=int, default=None)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--backoff-sec", type=float, default=1.0)
    args = parser.parse_args()
    manifest_path = run(
        IngestConfig(
            out_dir=args.out_dir,
            min_ts=args.min_ts,
            max_ts=args.max_ts,
            limit=args.limit,
            retries=args.retries,
            backoff_sec=args.backoff_sec,
        )
    )
    print(f"wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()

