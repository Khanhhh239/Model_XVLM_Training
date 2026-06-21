"""Phase 1/2 -- encode images + queries with a SigLIP checkpoint, cache to .pt.

Run ONCE per checkpoint per scale (do 512 and 768 separately for TTA).  Two modes:
  --index    distractor-val: encodes {gt, distractor, query}
  --gallery-dir + --gallery-ids : submission: encodes {gallery, gallery_ids, query}

    python -m startv4.scripts.encode_cache --config configs/siglip_a100_80g_1m.yaml \
        --ckpt checkpoints/siglip_v4.pth --use-ema --scale 512 \
        --index distractor_val/index.json --query-json distractor_val/queries.json \
        --out cache/siglip_512.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ..config import load_config, parse_overrides
from ..data.tokenizer import build_tokenizer
from ..infer.encode import encode_captions, encode_image_paths, save_embeddings
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


def load_checkpoint_into(model, ckpt_path: str, use_ema: bool):
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = payload.get("model", payload)
    model.load_state_dict(sd, strict=False)
    if use_ema and isinstance(payload, dict) and payload.get("ema"):
        msd = model.state_dict()
        for k, v in payload["ema"].items():
            if k in msd:
                msd[k].copy_(v.to(msd[k].dtype))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--use-ema", action="store_true")
    ap.add_argument("--scale", type=int, default=512)
    ap.add_argument("--query-json", required=True)
    ap.add_argument("--index", default=None, help="distractor-val index.json (gt + distractors)")
    ap.add_argument("--gallery-dir", default=None)
    ap.add_argument("--gallery-ids", default=None, help="optional .txt of gallery ids (one per line)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    cfg = load_config(a.config, parse_overrides(a.set))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_siglip(cfg).to(device)
    load_checkpoint_into(model, a.ckpt, a.use_ema)
    tok = build_tokenizer(cfg)

    captions = _load_captions(a.query_json)
    q = encode_captions(model, captions, tok, device)

    if a.index:
        idx = json.loads(Path(a.index).read_text(encoding="utf-8"))
        gt = encode_image_paths(model, idx["gt"], a.scale, device, a.batch_size)
        dis = encode_image_paths(model, idx["distractors"], a.scale, device, a.batch_size)
        save_embeddings(a.out, query=q, gt=gt, distractor=dis)
        print(f"[encode] scale={a.scale} q={len(q)} gt={len(gt)} distractor={len(dis)} -> {a.out}")
    elif a.gallery_dir:
        import glob

        paths = sorted(glob.glob(str(Path(a.gallery_dir) / "**" / "*"), recursive=True))
        paths = [p for p in paths if Path(p).suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp")]
        g = encode_image_paths(model, paths, a.scale, device, a.batch_size)
        if a.gallery_ids:
            ids = Path(a.gallery_ids).read_text(encoding="utf-8").split()
        else:
            ids = [Path(p).stem for p in paths]
        save_embeddings(a.out, query=q, gallery=g, gallery_ids=ids)
        print(f"[encode] scale={a.scale} q={len(q)} gallery={len(g)} -> {a.out}")
    else:
        raise SystemExit("provide --index or --gallery-dir")


if __name__ == "__main__":
    main()
