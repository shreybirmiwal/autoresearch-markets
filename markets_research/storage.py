from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class SnapshotManifest:
    snapshot_id: str
    venue: str
    created_at_utc: str
    start_ts: int | None
    end_ts: int | None
    records: dict[str, int]
    source: str


def ensure_layout(root: Path) -> None:
    (root / "markets").mkdir(parents=True, exist_ok=True)
    (root / "trades").mkdir(parents=True, exist_ok=True)
    (root / "manifests").mkdir(parents=True, exist_ok=True)


def write_partitioned_parquet(df: pd.DataFrame, path: Path, partitions: list[str]) -> None:
    if df.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow", partition_cols=partitions, index=False)


def write_manifest(root: Path, manifest: SnapshotManifest) -> Path:
    ensure_layout(root)
    out = root / "manifests" / f"{manifest.snapshot_id}.json"
    out.write_text(json.dumps(manifest.__dict__, indent=2), encoding="utf-8")
    return out


def new_snapshot_id(venue: str) -> str:
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{venue}-{now}"

