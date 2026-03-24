"""
Data preparation entrypoint for prediction-market autoresearch.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from markets_research.bootstrap_demo_data import generate_demo_data
from markets_research.data_ingest_kalshi import IngestConfig, run as run_ingest
from markets_research.data_ingest_polymarket import run as run_polymarket_ingest


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare data for prediction-market autoresearch.")
    parser.add_argument("--mode", choices=["ingest", "demo", "polymarket"], default="ingest")
    parser.add_argument("--out-dir", default="data_lake")
    # kalshi args
    parser.add_argument("--min-ts", type=int, default=None)
    parser.add_argument("--max-ts", type=int, default=None)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--backoff-sec", type=float, default=1.0)
    # demo args
    parser.add_argument("--seed", type=int, default=7)
    # polymarket args
    parser.add_argument("--markets-csv", type=Path, default=None, help="Path to poly_data markets.csv")
    parser.add_argument("--trades-csv", type=Path, default=None, help="Path to poly_data processed/trades.csv")
    parser.add_argument("--top-n", type=int, default=500, help="Top N markets by volume (polymarket mode)")
    args = parser.parse_args()

    if args.mode == "demo":
        generate_demo_data(Path(args.out_dir), seed=args.seed)
        print(f"Demo data written to: {args.out_dir}")
        return

    if args.mode == "polymarket":
        if not args.markets_csv or not args.trades_csv:
            parser.error("--mode polymarket requires --markets-csv and --trades-csv")
        run_polymarket_ingest(args.markets_csv, args.trades_csv, Path(args.out_dir), args.top_n)
        return

    manifest_path = run_ingest(
        IngestConfig(
            out_dir=Path(args.out_dir),
            min_ts=args.min_ts,
            max_ts=args.max_ts,
            limit=args.limit,
            retries=args.retries,
            backoff_sec=args.backoff_sec,
        )
    )
    print(f"Ingest complete. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
