"""Manifest schema + loading.  See README sec. 3.1 (data brief for the data team).

Required columns: image_path, caption, video_id, bucket.
Optional: keypoints (17x3 list), bbox (xyxy-norm list), image_id, pair_image_id, split.
All rows are synthetic (is_real = False); OOPS!/real data is FORBIDDEN for training.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ["image_path", "caption", "video_id", "bucket"]
BUCKET_MAP = {"goal": 0, "wentwrong": 1, "full": 2, "normal": 0, "anomaly": 1}


def load_manifest(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix in (".parquet", ".pq"):
        df = pd.read_parquet(path)
    elif path.suffix in (".jsonl", ".json"):
        df = pd.read_json(path, lines=path.suffix == ".jsonl")
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"unsupported manifest format: {path.suffix}")
    validate_manifest(df)
    return df


def validate_manifest(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"manifest missing required columns: {missing}")
    if len(df) == 0:
        raise ValueError("manifest is empty")


def bucket_to_int(b) -> int:
    if isinstance(b, (int,)) or (isinstance(b, float) and float(b).is_integer()):
        return int(b)
    return BUCKET_MAP.get(str(b).lower().strip(), 0)
