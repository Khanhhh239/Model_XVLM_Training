"""ANCE Hard Negative Mining - STAGE 1 Tầng 2.

Mines cross-ID hard negatives by running the model on the training set and finding
top-K most similar but WRONG matches for each query. These mined negatives are used
as additional ITM negatives in subsequent epochs.

Usage:
    python scripts/mine_hard_negatives.py \\
        --config configs/stage1_30k_kaggle_t4.yaml \\
        --ckpt outputs/stage1_30k_t4/last.pth \\
        --output data/mined_hard_negatives.json \\
        --top_k 5

References:
    - ANCE (Xiong et al., ICLR 2021, arXiv:2007.00808)
    - ALBEF hard-neg sampling (Li et al., NeurIPS 2021)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# allow `python scripts/mine_hard_negatives.py` without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from star.config import load_config, parse_overrides  # noqa: E402
from star.data import PABDataset, collate_fn  # noqa: E402
from star.models import STARModel  # noqa: E402
from star.utils import get_logger, seed_everything  # noqa: E402
from star.utils.checkpoint import load_checkpoint  # noqa: E402

log = get_logger("star.mine_hard_neg")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Config YAML file")
    ap.add_argument("--ckpt", required=True, help="Checkpoint to load (last.pth or best.pth)")
    ap.add_argument("--output", default="data/mined_hard_negatives.json", help="Output JSON file")
    ap.add_argument("--top_k", type=int, default=5, help="Top-K hard negatives per query")
    ap.add_argument("--set", nargs="*", default=[], help="Config overrides")
    args = ap.parse_args()

    cfg = load_config(args.config, parse_overrides(args.set))
    seed_everything(cfg.train.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model = STARModel(cfg).to(device)
    load_checkpoint(args.ckpt, model, map_location=device)
    model.eval()
    tokenizer = model.backbone.tokenizer

    # Load train dataset (no augmentation for mining)
    train_ds = PABDataset(
        cfg.data.manifest, cfg.data.image_root, tokenizer, split="train",
        image_size=cfg.data.image_size, max_token=cfg.data.max_token, train=False,  # No aug!
        vitpose_json=getattr(cfg.data, "vitpose_json", None),
        boxes_json=getattr(cfg.data, "boxes_json", None),
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg.train.batch_size * 2, shuffle=False,
        num_workers=cfg.data.num_workers, collate_fn=collate_fn, pin_memory=True,
    )

    log.info(f"Mining hard negatives from {len(train_ds)} samples...")

    # Extract all embeddings
    img_feats, txt_feats, instance_ids, image_ids = [], [], [], []
    with torch.no_grad():
        for batch in tqdm(train_loader, desc="Extracting features"):
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            img_f, txt_f = model.encode_for_eval(
                batch["image"], batch["input_ids"], batch["attention_mask"],
                keypoints=batch.get("keypoints"),
            )
            img_feats.append(img_f.cpu())
            txt_feats.append(txt_f.cpu())
            instance_ids.append(batch["instance_id"].cpu())
            image_ids.extend(batch.get("image_id", [str(i) for i in range(len(batch["image"]))]))

    img_feats = torch.cat(img_feats, dim=0)  # [N, D]
    txt_feats = torch.cat(txt_feats, dim=0)  # [N, D]
    instance_ids = torch.cat(instance_ids, dim=0)  # [N]

    log.info(f"Extracted {img_feats.size(0)} features. Computing similarity matrix...")

    # Compute similarity matrix
    sim = (txt_feats @ img_feats.t()).cpu()  # [N, N]

    # Mine hard negatives
    mined_negatives = {}
    for i in tqdm(range(sim.size(0)), desc="Mining hard negatives"):
        # Mask same-instance (not a negative)
        mask = (instance_ids != instance_ids[i])
        sim_masked = sim[i].clone()
        sim_masked[~mask] = -float('inf')

        # Top-K most similar (but wrong) images
        top_k_scores, top_k_indices = torch.topk(sim_masked, k=min(args.top_k, mask.sum().item()))

        # Store results
        query_id = image_ids[i]
        mined_negatives[query_id] = {
            "hard_neg_ids": [image_ids[j] for j in top_k_indices.tolist()],
            "hard_neg_scores": top_k_scores.tolist(),
        }

    # Save to JSON
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(mined_negatives, f, indent=2)

    log.info(f"Mined hard negatives saved to {output_path}")
    log.info(f"  Total queries: {len(mined_negatives)}")
    log.info(f"  Avg hard-negs per query: {sum(len(v['hard_neg_ids']) for v in mined_negatives.values()) / len(mined_negatives):.1f}")


if __name__ == "__main__":
    main()
