"""Phase 3 -- build phrase->box pseudo-labels for the bbox-grounding head.

Extract noun phrases from each caption, localise them with an open-vocabulary detector.
Default = DummyBoxDetector (centred box) so the pipeline runs without GroundingDINO; pass a
real detector for the actual ~10-20h pass.  Output: jsonl of {image_path, boxes:[{phrase,bbox,conf}]}.

    python -m startv4.scripts.build_box_pseudolabels --manifest data/manifest_1m.parquet \
        --out data/box_pseudolabels.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..data.grounding import DummyBoxDetector, label_boxes
from ..data.manifest import load_manifest


def _get_detector(name: str):
    if name == "dummy":
        return DummyBoxDetector()
    raise SystemExit(f"detector {name!r} not available; install GroundingDINO and wire it here")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--detector", default="dummy", choices=["dummy", "groundingdino"])
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--limit", type=int, default=0, help="0 = all rows")
    ap.add_argument("--out", default="data/box_pseudolabels.jsonl")
    a = ap.parse_args()

    df = load_manifest(a.manifest)
    if a.limit:
        df = df.head(a.limit)
    detector = _get_detector(a.detector)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(a.out, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            boxes = label_boxes(row["image_path"], row["caption"], detector, conf=a.conf)
            f.write(json.dumps({"image_path": row["image_path"], "boxes": boxes}) + "\n")
            n += 1
    print(f"[phase3] wrote {n} box-pseudolabel rows -> {a.out}")


if __name__ == "__main__":
    main()
