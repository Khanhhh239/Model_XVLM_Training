"""Phase 0 -- build the distractor-val index.

Gallery = old-test GT positives + external REAL person distractors that are PROVABLY NOT
OOPS! (Market-1501 / MSMT17 / COCO-person).  Distractors are perceptual-hash de-duped
against the test gallery so no test image can leak in.

    python -m startv4.scripts.build_distractor_val \
        --gt-dir old_test/gt --distractor-dir external/market1501 \
        --test-gallery-dir test/gallery --out distractor_val/index.json
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from ..eval.distractor_val import build_distractor_index

_EXT = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")


def _glob_images(d: str) -> list[str]:
    out: list[str] = []
    for e in _EXT:
        out += glob.glob(str(Path(d) / "**" / e), recursive=True)
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", required=True, help="old-test GT positive images")
    ap.add_argument("--distractor-dir", required=True, help="external NON-OOPS! distractor images")
    ap.add_argument(
        "--test-gallery-dir", required=True, help="test gallery images to de-dup AGAINST (never trained on)"
    )
    ap.add_argument("--threshold", type=int, default=5, help="max Hamming distance counted as duplicate")
    ap.add_argument("--out", default="distractor_val/index.json")
    a = ap.parse_args()

    idx = build_distractor_index(
        _glob_images(a.gt_dir), _glob_images(a.distractor_dir), _glob_images(a.test_gallery_dir), a.threshold
    )
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(idx, indent=2), encoding="utf-8")
    print(f"[phase0] gt={len(idx['gt'])} distractors_kept={len(idx['distractors'])} "
          f"removed={idx['removed']} -> {a.out}")


if __name__ == "__main__":
    main()
