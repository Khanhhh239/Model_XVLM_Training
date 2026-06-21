"""Entry point: train the SigLIP-2-L retrieval tower (Model A / Phase 1).

    python -m startv4.train.train_siglip --config configs/siglip_a100_80g_1m.yaml
    python -m startv4.train.train_siglip --config configs/_test_dummy.yaml --overfit-one-batch
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..config import load_config, parse_overrides, to_plain
from ..data import (
    BalancedBucketSampler,
    PABDatasetV4,
    PairBatchSampler,
    collate_fn,
)
from ..data.tokenizer import build_tokenizer
from ..infer.encode import make_distractor_eval_fn
from ..models.siglip_retrieval import build_siglip
from .trainer import SiglipTrainer


def _load_captions(path):
    p = Path(path)
    rows = ([json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
            if p.suffix == ".jsonl" else json.loads(p.read_text(encoding="utf-8")))
    if isinstance(rows, dict):
        rows = rows.get("queries", [])
    return [r["caption"] if isinstance(r, dict) else str(r) for r in rows]


def _build_tokenizer(cfg):
    # dummy -> None lets the dataset use its bundled DummyTokenizer
    return None if cfg.model.name == "dummy" else build_tokenizer(cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--overfit-one-batch", action="store_true")
    ap.add_argument("--val-index", default=None, help="distractor_val/index.json -> enables best-by-mAP")
    ap.add_argument("--val-query-json", default=None, help="captions aligned to val index 'gt'")
    ap.add_argument("--val-scale", type=int, default=None)
    ap.add_argument("--val-max-distractors", type=int, default=0, help="subsample distractors per eval (0=all)")
    ap.add_argument("--eval-every", type=int, default=0, help="run distractor-val every N steps")
    ap.add_argument("--out", default="checkpoints/siglip_v4.pth")
    args = ap.parse_args()

    cfg = load_config(args.config, parse_overrides(args.set))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[startv4] device={device}  model={cfg.model.name}")

    tokenizer = _build_tokenizer(cfg)
    ds = PABDatasetV4(
        cfg.data.manifest,
        cfg.data.image_root,
        tokenizer=tokenizer,
        split="train",
        image_size=int(cfg.data.image_size),
        train=True,
        max_token=int(cfg.data.get_path("max_token", 64)),
        cfg=cfg,
    )

    sampler_kind = cfg.data.get_path("sampler", "balanced")
    bs = int(cfg.train.batch_size)
    if sampler_kind == "pair" and ds.pairs():
        sampler = PairBatchSampler(ds.pairs(), bs, seed=int(cfg.train.seed))
    else:
        sampler = BalancedBucketSampler(ds.buckets(), bs, seed=int(cfg.train.seed))
    loader = DataLoader(
        ds, batch_sampler=sampler, num_workers=int(cfg.data.get_path("num_workers", 0)),
        collate_fn=collate_fn, pin_memory=(device == "cuda"),
    )

    model = build_siglip(cfg)
    trainer = SiglipTrainer(model, cfg, device)

    if args.overfit_one_batch:
        batch = next(iter(loader))
        losses = trainer.overfit_one_batch(batch, steps=50)
        print(f"[overfit] {losses[0]:.4f} -> {losses[-1]:.4f}")
        return

    def _log(ep, step, logs):
        print(f"[ep{ep} step{step}] " + " ".join(f"{k}={v:.4f}" for k, v in logs.items()))

    eval_fn, eval_every = None, 0
    if args.val_index and args.val_query_json:
        eval_tok = build_tokenizer(cfg)  # DummyTokenizer for dummy, HF processor otherwise
        eval_fn = make_distractor_eval_fn(
            json.loads(Path(args.val_index).read_text(encoding="utf-8")),
            _load_captions(args.val_query_json), eval_tok,
            args.val_scale or int(cfg.data.image_size), device,
            max_distractors=args.val_max_distractors,
        )
        eval_every = args.eval_every or len(loader)  # default: once per epoch
        print(f"[startv4] distractor-val every {eval_every} steps -> best-by-mAP")

    best = trainer.fit(loader, epochs=int(cfg.optim.epochs), on_log=_log,
                       eval_fn=eval_fn, eval_every=eval_every,
                       on_eval=lambda s, m, b: print(f"[eval step{s}] " +
                       " ".join(f"{k}={v:.4f}" for k, v in m.items()) + f"  (best mAP={b['mAP']:.4f})"))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    # save the BEST checkpoint if we evaluated, else the final model (review fix F3/P1d)
    model_sd = trainer.best_state if trainer.best_state is not None else model.state_dict()
    payload = {"model": model_sd, "cfg": to_plain(cfg), "best": best if eval_fn else None}
    if trainer.ema is not None:
        payload["ema"] = trainer.ema.state_dict()
    torch.save(payload, args.out)
    tag = f"BEST (mAP={best['mAP']:.4f} @step{best.get('step')})" if trainer.best_state is not None else "final"
    print(f"[startv4] saved {tag} -> {args.out}")


if __name__ == "__main__":
    main()
