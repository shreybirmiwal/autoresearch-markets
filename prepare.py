"""
Data preparation entrypoint for prediction-market autoresearch.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from markets_research.bootstrap_demo_data import generate_demo_data
from markets_research.data_ingest_kalshi import IngestConfig, run as run_ingest


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare data for Kalshi autoresearch.")
    parser.add_argument("--mode", choices=["ingest", "demo"], default="ingest")
    parser.add_argument("--out-dir", default="data_lake")
    parser.add_argument("--min-ts", type=int, default=None)
    parser.add_argument("--max-ts", type=int, default=None)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--backoff-sec", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    if args.mode == "demo":
        generate_demo_data(Path(args.out_dir), seed=args.seed)
        print(f"Demo data written to: {args.out_dir}")
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
