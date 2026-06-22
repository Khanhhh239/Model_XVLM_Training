"""Distractor-val: measure RECALL in the presence of external distractors.

This is the proxy for the real leaderboard (recall is the competition bottleneck).
Gallery = old-test GT positives + ~20K REAL person distractors that are PROVABLY NOT
from OOPS!/fail-videos (e.g. Market-1501, MSMT17, COCO-person).  Distractors are used
ONLY to measure -- never for training -- and must be perceptual-hash de-duped against
the test gallery.
"""
from __future__ import annotations

import torch
from PIL import Image

from ..data.dedup import dedup_keep_mask, dhash
from .metrics import retrieval_metrics


def build_distractor_index(
    gt_paths: list[str],
    distractor_paths: list[str],
    test_gallery_paths: list[str],
    threshold: int = 5,
) -> dict:
    """Build a distractor-val index: keep only distractors that are NOT near-duplicates of
    any test-gallery image (perceptual-hash de-dup -> enforces the no-test-data rule).

    Returns {"gt": [...], "distractors": [kept...], "removed": n_removed}.
    """
    ref_hashes = [dhash(Image.open(p)) for p in test_gallery_paths]
    cand_hashes = [dhash(Image.open(p)) for p in distractor_paths]
    keep = dedup_keep_mask(cand_hashes, ref_hashes, threshold=threshold)
    kept = [p for p, k in zip(distractor_paths, keep) if k]
    return {
        "gt": list(gt_paths),
        "distractors": kept,
        "removed": len(distractor_paths) - len(kept),
    }


def evaluate_with_distractors(
    query_feat: torch.Tensor,
    gt_feat: torch.Tensor,
    distractor_feat: torch.Tensor,
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    """query_feat [Q,D], gt_feat [Q,D] (gt_feat[i] is the GT image for query i, L2-norm),
    distractor_feat [M,D] (L2-norm).  Builds a [Q, Q+M] gallery = [GTs ; distractors] and
    reports R@k + mAP with each query's GT at column i.
    """
    gallery = torch.cat([gt_feat, distractor_feat], dim=0)  # [Q+M, D]
    sim = query_feat @ gallery.t()                          # [Q, Q+M]
    gt_index = torch.arange(query_feat.size(0), device=query_feat.device)
    return retrieval_metrics(sim, gt_index, ks=ks)


def evaluate_by_category(
    query_feat: torch.Tensor,
    gt_feat: torch.Tensor,
    distractor_feat: torch.Tensor,
    categories: list,
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, dict]:
    """Per-wrong-case-category distractor-val.  Same gallery as evaluate_with_distractors, but
    metrics are reported per `categories[i]` group (the literal 'does it handle wrong case #k'
    proof) plus an "__overall__" row.
    """
    from collections import defaultdict

    gallery = torch.cat([gt_feat, distractor_feat], dim=0)
    sim = query_feat @ gallery.t()
    gt_index = torch.arange(query_feat.size(0), device=query_feat.device)

    groups: dict = defaultdict(list)
    for i, c in enumerate(categories):
        groups[str(c)].append(i)

    out: dict[str, dict] = {}
    for cat, idxs in groups.items():
        sel = torch.tensor(idxs, device=sim.device)
        out[cat] = retrieval_metrics(sim[sel], gt_index[sel], ks=ks)
        out[cat]["n"] = len(idxs)
    out["__overall__"] = retrieval_metrics(sim, gt_index, ks=ks)
    out["__overall__"]["n"] = query_feat.size(0)
    return out
