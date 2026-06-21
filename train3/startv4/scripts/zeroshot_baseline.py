"""Phase 1 (pre-train) -- FROZEN zero-shot SigLIP-2-L on distractor-val.

THE measurement to run FIRST: how much of the target mAP is "free" from the pretrained
backbone vs how much training must add.  No training, no checkpoint -- just the off-the-shelf
SigLIP-2-L encoded against the distractor-val index.

Query captions (--query-json) must be in the SAME ORDER as the index 'gt' list (query i's GT
is gt[i]).

    python -m startv4.scripts.zeroshot_baseline --config configs/siglip_a100_80g_1m.yaml \
        --index distractor_val/index.json --query-json distractor_val/queries.json --scale 512
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ..config import load_config, parse_overrides
from ..data.tokenizer import build_tokenizer
from ..eval.distractor_val import evaluate_with_distractors
from ..infer.encode import encode_captions, encode_image_paths
from ..models.siglip_retrieval import build_siglip


def _load_captions(path: str) -> list[str]:
    p = Path(path)
    if p.suffix == ".jsonl":
        rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        rows = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(rows, dict):
        rows = rows.get("queries", [])
    return [r["caption"] if isinstance(r, dict) else str(r) for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--index", required=True, help="distractor_val/index.json (gt + distractors)")
    ap.add_argument("--query-json", required=True)
    ap.add_argument("--scale", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=64)
    a = ap.parse_args()

    cfg = load_config(a.config, parse_overrides(a.set))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_siglip(cfg).to(device).eval()  # FROZEN — no training, no checkpoint
    tok = build_tokenizer(cfg)

    idx = json.loads(Path(a.index).read_text(encoding="utf-8"))
    captions = _load_captions(a.query_json)
    q = encode_captions(model, captions, tok, device)
    gt = encode_image_paths(model, idx["gt"], a.scale, device, a.batch_size)
    dis = encode_image_paths(model, idx["distractors"], a.scale, device, a.batch_size)

    m = evaluate_with_distractors(q, gt, dis)
    print("[zeroshot SigLIP] " + "  ".join(f"{k}={v:.4f}" for k, v in m.items()))
    print("    ^ this is the 'free' baseline; training must beat it to be worth the cost.")
    return m


if __name__ == "__main__":
    main()
