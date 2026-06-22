"""Per-wrong-case-category proof: does the trained model handle each failure category?

Input `--wrong-json`: list of {caption, gt_path, category} (one per wrong query, category =
"#1".."#10" or any label).  `--index`: distractor-val index.json (its 'distractors' are the
gallery noise).  Encodes everything with the checkpoint and prints R@1/R@5/mAP PER CATEGORY +
overall.  Run it on the OLD checkpoint and the NEW one to show before->after per category --
that is the literal proof asked for ("kiến trúc xử lý được wrong case nào").

    python -m startv4.scripts.eval_wrong_cases --config configs/siglip_full1m_a100_40g.yaml \
        --ckpt checkpoints/siglip_v4.pth --use-ema --scale 512 \
        --wrong-json wrong_cases.json --index distractor_val/index.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ..config import load_config, parse_overrides
from ..data.tokenizer import build_tokenizer
from ..eval.distractor_val import evaluate_by_category
from ..infer.encode import encode_retrieval_images, encode_retrieval_text
from ..models.siglip_retrieval import build_siglip


def _load_ckpt(model, path, use_ema):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload.get("model", payload), strict=False)
    if use_ema and isinstance(payload, dict) and payload.get("ema"):
        sd = model.state_dict()
        for k, v in payload["ema"].items():
            if k in sd:
                sd[k].copy_(v.to(sd[k].dtype))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--ckpt", default=None, help="omit to evaluate the ZERO-SHOT backbone")
    ap.add_argument("--use-ema", action="store_true")
    ap.add_argument("--scale", type=int, default=512)
    ap.add_argument("--wrong-json", required=True, help="[{caption, gt_path, category}]")
    ap.add_argument("--index", required=True, help="distractor_val index.json (uses its 'distractors')")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    cfg = load_config(a.config, parse_overrides(a.set))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_siglip(cfg).to(device).eval()
    if a.ckpt:
        _load_ckpt(model, a.ckpt, a.use_ema)
    tok = build_tokenizer(cfg)

    rows = json.loads(Path(a.wrong_json).read_text(encoding="utf-8"))
    captions = [r["caption"] for r in rows]
    gt_paths = [r["gt_path"] for r in rows]
    cats = [r.get("category", "?") for r in rows]
    dis_paths = json.loads(Path(a.index).read_text(encoding="utf-8"))["distractors"]

    q = encode_retrieval_text(model, captions, tok, device)
    gt = encode_retrieval_images(model, gt_paths, a.scale, device)
    dis = encode_retrieval_images(model, dis_paths, a.scale, device)
    res = evaluate_by_category(q, gt, dis, cats)

    tag = "ZERO-SHOT" if not a.ckpt else Path(a.ckpt).name
    print(f"\n[wrong-case proof | {tag}]  (gallery = {len(gt)} GT + {len(dis)} distractors)")
    print(f"{'category':<14}{'n':>4}{'R@1':>8}{'R@5':>8}{'mAP':>8}")
    for cat in sorted(res, key=lambda c: (c == "__overall__", c)):
        m = res[cat]
        print(f"{cat:<14}{m['n']:>4}{m['R@1']:>8.3f}{m['R@5']:>8.3f}{m['mAP']:>8.3f}")
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res


if __name__ == "__main__":
    main()
