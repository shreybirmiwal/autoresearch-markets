from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from markets_research.storage import SnapshotManifest, new_snapshot_id, write_manifest, write_partitioned_parquet


def generate_demo_data(out_dir: Path, seed: int = 7) -> None:
    rng = np.random.default_rng(seed)
    snapshot_id = new_snapshot_id("kalshi-demo")
    n_markets = 20
    n_events_per_market = 120
    markets = []
    trades = []
    base_ts = pd.Timestamp("2024-01-01T00:00:00Z")
    for mi in range(n_markets):
        market_id = f"DEMO-{mi:03d}"
        settle = float(rng.choice([0.0, 1.0], p=[0.5, 0.5]))
        markets.append(
            {
                "venue": "kalshi",
                "market_id": market_id,
                "ticker": market_id,
                "title": f"Demo market {mi}",
                "open_ts": base_ts,
                "close_ts": base_ts + pd.Timedelta(days=1),
                "settled_ts": base_ts + pd.Timedelta(days=2),
                "strike_info": None,
                "category": "demo",
                "is_resolved": True,
                "settlement_price_yes": settle,
                "fee_bps": 10.0,
                "snapshot_id": snapshot_id,
                "date": "2024-01-03",
            }
        )
        p = 0.5 + rng.normal(0, 0.05)
        for ei in range(n_events_per_market):
            ts = base_ts + pd.Timedelta(minutes=mi * n_events_per_market + ei)
            p = np.clip(0.97 * p + 0.03 * settle + rng.normal(0, 0.01), 0.01, 0.99)
            side = "yes" if rng.random() > 0.5 else "no"
            trades.append(
                {
                    "venue": "kalshi",
                    "market_id": market_id,
                    "ticker": market_id,
                    "event_ts": ts,
                    "side": side,
                    "price_yes": float(p),
                    "size": float(rng.integers(1, 10)),
                    "trade_id": f"{market_id}-{ei}",
                    "snapshot_id": snapshot_id,
                    "date": ts.strftime("%Y-%m-%d"),
                }
            )
    trades_df = pd.DataFrame(trades)
    markets_df = pd.DataFrame(markets)
    write_partitioned_parquet(trades_df, out_dir / "trades" / f"{snapshot_id}.parquet", ["venue", "date"])
    write_partitioned_parquet(markets_df, out_dir / "markets" / f"{snapshot_id}.parquet", ["venue", "date"])
    write_manifest(
        out_dir,
        SnapshotManifest(
            snapshot_id=snapshot_id,
            venue="kalshi",
            created_at_utc=pd.Timestamp.utcnow().isoformat(),
            start_ts=None,
            end_ts=None,
            records={"trades": int(len(trades_df)), "markets": int(len(markets_df))},
            source="synthetic_demo",
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate demo prediction-market parquet data.")
    parser.add_argument("--out-dir", type=Path, default=Path("data_lake"))
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    generate_demo_data(args.out_dir, args.seed)
    print(f"demo data written to: {args.out_dir}")


if __name__ == "__main__":
    main()

