"""
Ingest Polymarket data from poly_data snapshot CSVs into the data_lake parquet format.

Data source: https://github.com/warproxxx/poly_data
Expected files:
  - markets.csv   (market metadata)
  - processed/trades.csv  (structured trade events)
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from markets_research.schema import Market, TradeEvent, markets_to_frame, trades_to_frame
from markets_research.storage import SnapshotManifest, new_snapshot_id, write_manifest, write_partitioned_parquet

# Chunk size for streaming the large trades CSV (72M+ rows)
_CHUNK = 500_000


def _parse_ts(val: str | None) -> datetime | None:
    if not val or pd.isna(val):
        return None
    try:
        return pd.Timestamp(val, tz="UTC").to_pydatetime()
    except Exception:
        return None


def _load_markets(markets_csv: Path, top_n: int) -> pd.DataFrame:
    """Load markets.csv, filter to top_n by volume, build lookup columns."""
    df = pd.read_csv(markets_csv, low_memory=False)
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)
    # Keep only top_n markets by volume
    df = df.nlargest(top_n, "volume").copy()
    # Normalise answer columns for YES/NO mapping
    df["answer1_lower"] = df["answer1"].fillna("").str.strip().str.lower()
    df["answer2_lower"] = df["answer2"].fillna("").str.strip().str.lower()
    # Use string index so joins on string market_id work cleanly
    df["id"] = df["id"].astype(str)
    return df.set_index("id")


def _token_side_to_yes_no(nonusdc_side: pd.Series, taker_direction: pd.Series,
                           answer1_lower: pd.Series, answer2_lower: pd.Series) -> pd.Series:
    """
    Map (nonusdc_side, taker_direction, answer1/2) → "yes" or "no".

    nonusdc_side: "token1" or "token2" — which outcome token was traded.
    taker_direction: "BUY" or "SELL" — did taker buy or sell that token?

    The 'side' in our schema represents what outcome the taker is expressing:
      - taker BUYs YES token  → side=yes
      - taker SELLs YES token → side=no  (going short YES = long NO)
      - taker BUYs NO token   → side=no
      - taker SELLs NO token  → side=yes
    """
    is_token1 = nonusdc_side == "token1"
    token_is_yes = (is_token1 & (answer1_lower == "yes")) | (~is_token1 & (answer2_lower == "yes"))
    taker_buys = taker_direction == "BUY"
    # taker expresses YES if: buys YES token, or sells NO token
    expresses_yes = (taker_buys & token_is_yes) | (~taker_buys & ~token_is_yes)
    return expresses_yes.map({True: "yes", False: "no"})


def _price_yes(price: pd.Series, nonusdc_side: pd.Series,
               answer1_lower: pd.Series) -> pd.Series:
    """
    Normalise price to always represent the YES token price.
    If the traded token is a NO token, price_yes = 1 - price.
    """
    token1_is_yes = answer1_lower == "yes"
    token1_traded = nonusdc_side == "token1"
    traded_token_is_yes = (token1_traded & token1_is_yes) | (~token1_traded & ~token1_is_yes)
    return price.where(traded_token_is_yes, 1.0 - price).clip(0.0, 1.0)


def build_market_objects(markets_df: pd.DataFrame, snapshot_id: str) -> list[Market]:
    out = []
    for market_id, row in markets_df.iterrows():
        ticker = str(row.get("ticker") or row.get("market_slug") or market_id)
        out.append(
            Market(
                venue="polymarket",
                market_id=str(market_id),
                ticker=ticker,
                title=str(row.get("question") or ticker),
                open_ts=_parse_ts(row.get("createdAt")),
                close_ts=_parse_ts(row.get("closedTime")),
                settled_ts=_parse_ts(row.get("closedTime")),
                strike_info=None,
                category=None,
                is_resolved=bool(row.get("closedTime")),
                settlement_price_yes=None,  # not available in this dataset
                fee_bps=0.0,
                snapshot_id=snapshot_id,
            )
        )
    return out


def run(
    markets_csv: Path,
    trades_csv: Path,
    out_dir: Path,
    top_n: int = 500,
) -> Path:
    snapshot_id = new_snapshot_id("polymarket")
    print(f"Snapshot ID: {snapshot_id}")

    print(f"Loading markets (top {top_n} by volume)...")
    markets_df = _load_markets(markets_csv, top_n)
    kept_ids = set(markets_df.index.astype(str))
    print(f"  Kept {len(kept_ids)} markets")

    market_objects = build_market_objects(markets_df, snapshot_id)
    mdf = markets_to_frame(market_objects)
    mdf["date"] = pd.to_datetime(
        mdf["settled_ts"].fillna(mdf["close_ts"]).fillna(mdf["open_ts"]), utc=True
    ).dt.strftime("%Y-%m-%d")
    # Flat parquet (same reason as trades: avoid partitioned-directory glob collision)
    markets_path = out_dir / "markets" / f"{snapshot_id}.parquet"
    markets_path.parent.mkdir(parents=True, exist_ok=True)
    mdf.to_parquet(markets_path, engine="pyarrow", index=False)
    print(f"  Markets written: {len(mdf)}")

    # Build per-market lookup series for vectorised YES/NO mapping
    a1 = markets_df["answer1_lower"]
    a2 = markets_df["answer2_lower"]

    print(f"Streaming trades CSV in chunks of {_CHUNK:,}...")
    total_trades = 0
    all_trade_chunks: list[pd.DataFrame] = []

    for chunk_idx, chunk in enumerate(pd.read_csv(trades_csv, chunksize=_CHUNK, low_memory=False)):
        chunk["market_id"] = chunk["market_id"].astype(str)
        chunk = chunk[chunk["market_id"].isin(kept_ids)].copy()
        if chunk.empty:
            if chunk_idx % 10 == 0:
                print(f"  chunk {chunk_idx}: 0 rows kept")
            continue

        # Merge market metadata for YES/NO mapping (both indexed as strings now)
        chunk = chunk.merge(
            markets_df[["answer1_lower", "answer2_lower"]],
            left_on="market_id", right_index=True, how="left",
        )

        chunk["price"] = pd.to_numeric(chunk["price"], errors="coerce").fillna(0.5)
        chunk["token_amount"] = pd.to_numeric(chunk["token_amount"], errors="coerce").fillna(0.0)

        chunk["side"] = _token_side_to_yes_no(
            chunk["nonusdc_side"], chunk["taker_direction"],
            chunk["answer1_lower"], chunk["answer2_lower"],
        )
        chunk["price_yes"] = _price_yes(chunk["price"], chunk["nonusdc_side"], chunk["answer1_lower"])

        chunk["event_ts"] = pd.to_datetime(chunk["timestamp"], utc=True, errors="coerce")
        chunk = chunk.dropna(subset=["event_ts"])

        tdf = pd.DataFrame({
            "venue": "polymarket",
            "market_id": chunk["market_id"],
            "ticker": chunk["market_id"],
            "event_ts": chunk["event_ts"],
            "side": chunk["side"],
            "price_yes": chunk["price_yes"],
            "size": chunk["token_amount"],
            "trade_id": chunk["transactionHash"].astype(str) + "-" + chunk.index.astype(str),
            "snapshot_id": snapshot_id,
        })

        all_trade_chunks.append(tdf)
        total_trades += len(tdf)
        if chunk_idx % 10 == 0:
            print(f"  chunk {chunk_idx}: {len(tdf):,} rows kept (total so far: {total_trades:,})")

    if all_trade_chunks:
        full_tdf = pd.concat(all_trade_chunks, ignore_index=True)
        full_tdf["date"] = full_tdf["event_ts"].dt.strftime("%Y-%m-%d")
        # Write as a flat parquet file (not partitioned directory) so the glob in
        # experiment._load_latest_trades("**/*.parquet") finds exactly one file per snapshot.
        trades_path = out_dir / "trades" / f"{snapshot_id}.parquet"
        trades_path.parent.mkdir(parents=True, exist_ok=True)
        full_tdf.to_parquet(trades_path, engine="pyarrow", index=False)
        print(f"  Trades written: {total_trades:,}")
    else:
        print("  No trades matched the selected markets.")

    manifest = SnapshotManifest(
        snapshot_id=snapshot_id,
        venue="polymarket",
        created_at_utc=pd.Timestamp.utcnow().isoformat(),
        start_ts=None,
        end_ts=None,
        records={"trades": total_trades, "markets": len(mdf)},
        source=f"poly_data snapshot | markets={markets_csv} trades={trades_csv} top_n={top_n}",
    )
    manifest_path = write_manifest(out_dir, manifest)
    print(f"Manifest written: {manifest_path}")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest poly_data snapshot into parquet data_lake.")
    parser.add_argument("--markets-csv", type=Path, required=True)
    parser.add_argument("--trades-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("data_lake"))
    parser.add_argument("--top-n", type=int, default=500, help="Keep top N markets by volume")
    args = parser.parse_args()
    run(args.markets_csv, args.trades_csv, args.out_dir, args.top_n)


if __name__ == "__main__":
    main()
