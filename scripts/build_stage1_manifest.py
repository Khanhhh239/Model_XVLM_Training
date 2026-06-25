"""Build STAR Stage-1 training manifest from raw train_30k_hard.jsonl.

Converts data-team JSONL (hard_i / hard_i_id) into the parquet manifest expected by
PABDataset + PairBatchSampler: image_path, pair_image_id, keypoints, bbox, split.

Usage:
    python scripts/build_stage1_manifest.py \\
        --jsonl data/train_30k_hard.jsonl \\
        --vitpose data/train_30k_hard_vitpose.json \\
        --image-root data/train_webp \\
        --out data/manifest_30k_hard.parquet
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def bucket_of(path: str) -> str:
    p = path.replace("\\", "/")
    if "/wentwrong/" in p:
        return "wentwrong"
    if "/full/" in p:
        return "full"
    return "goal"


def normalize_webp_path(raw: str, image_root: Path) -> str:
    p = str(raw).replace("\\", "/")
    if p.startswith("train/"):
        p = p[len("train/"):]
    if "train_webp/" in p:
        p = p.split("train_webp/", 1)[-1]
    if p.startswith("data/"):
        p = p[len("data/"):]
        if p.startswith("train_webp/"):
            p = p[len("train_webp/"):]
    return p


def load_vitpose(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["items"] if isinstance(data, dict) and "items" in data else data


def kpts_of(pose: dict, image_id: str) -> list[float] | None:
    it = pose.get(str(image_id))
    if not it or it.get("status") != "ok" or not it.get("instances"):
        return None
    W = float(it.get("width", 384) or 384)
    H = float(it.get("height", 384) or 384)
    flat = []
    for x, y, c in it["instances"][0]["keypoints_xyc"]:
        flat.extend([float(x) / W, float(y) / H, float(c)])
    return flat if len(flat) == 51 else None


def bbox_of(pose: dict, image_id: str) -> list[float] | None:
    it = pose.get(str(image_id))
    b = it.get("primary_bbox_norm_xyxy") if it else None
    if not b or len(b) != 4:
        return None
    x1, y1, x2, y2 = [float(v) for v in b]
    w, h = max(x2 - x1, 1e-4), max(y2 - y1, 1e-4)
    return [(x1 + x2) / 2, (y1 + y2) / 2, w, h]


def build_manifest(
    jsonl: str,
    vitpose_json: str | None,
    image_root: str,
    val_query_videos: int = 400,
    val_distractor_videos: int = 600,
    seed: int = 42,
) -> pd.DataFrame:
    pose = load_vitpose(vitpose_json) if vitpose_json and Path(vitpose_json).exists() else {}
    img_root = Path(image_root)

    anchors, hard_rows, anchor_ids = [], {}, set()
    for line in open(jsonl, encoding="utf-8"):
        r = json.loads(line)
        anchor_ids.add(r["image_id"])
        bucket = r.get("bucket") or bucket_of(r.get("image_webp") or r.get("image", ""))
        anchors.append(dict(
            image_path=normalize_webp_path(r.get("image_webp") or r.get("image", ""), img_root),
            caption=r["caption"],
            sequence_id=f'v{r["video_id"]}_{bucket}',
            scene=f'v{r["video_id"]}',
            action=str(r.get("label", r.get("normal", "unk"))),
            video_id=r["video_id"],
            image_id=r["image_id"],
            pair_image_id=r.get("hard_i_id"),
            bucket=bucket,
            label_type=r.get("label_type", bucket),
        ))
        hid = r.get("hard_i_id")
        if hid and hid not in hard_rows:
            hard_path = normalize_webp_path(r.get("hard_image_webp") or r.get("hard_i", ""), img_root)
            hard_rows[hid] = dict(
                image_path=hard_path,
                caption=r.get("hard_c", ""),
                sequence_id=f'v{r["video_id"]}_{bucket_of(hard_path)}',
                scene=f'v{r["video_id"]}',
                action="hard_pair",
                video_id=r["video_id"],
                image_id=hid,
                pair_image_id=None,
                bucket=bucket_of(hard_path),
                label_type="wentwrong" if "/wentwrong/" in hard_path else "full",
            )

    df = pd.DataFrame(anchors + [v for k, v in hard_rows.items() if k not in anchor_ids])
    if pose:
        df["keypoints"] = df.image_id.map(lambda i: kpts_of(pose, i))
        df["bbox"] = df.image_id.map(lambda i: bbox_of(pose, i))
    else:
        df["keypoints"] = None
        df["bbox"] = None

    rng = np.random.default_rng(seed)
    vids = df.video_id.unique().copy()
    rng.shuffle(vids)
    counts = df.groupby("video_id").size()
    vq, vd, acc = set(), set(), 0
    it = iter(vids)
    for v in it:
        vq.add(v)
        acc += counts[v]
        if acc >= val_query_videos:
            break
    acc = 0
    for v in it:
        vd.add(v)
        acc += counts[v]
        if acc >= val_distractor_videos:
            break
    df["split"] = np.where(df.video_id.isin(vq | vd), "valb", "train")
    df.loc[df.video_id.isin(vd), "caption"] = ""
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--vitpose", default=None)
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--out", default="data/manifest_30k_hard.parquet")
    ap.add_argument("--val-query-videos", type=int, default=400)
    ap.add_argument("--val-distractor-videos", type=int, default=600)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = build_manifest(
        args.jsonl, args.vitpose, args.image_root,
        args.val_query_videos, args.val_distractor_videos, args.seed,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)

    train_vids = set(df[df.split == "train"].video_id)
    val_vids = set(df[df.split == "valb"].video_id)
    pairs = int(df[df.split == "train"].pair_image_id.notna().sum())
    kpt_cov = df.keypoints.notna().mean() if "keypoints" in df.columns else 0
    print(f"rows={len(df)} train={(df.split=='train').sum()} valb={(df.split=='valb').sum()}")
    print(f"pairs(train)={pairs} keypoints={kpt_cov:.1%} leakage={len(train_vids & val_vids)}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
