"""Retrieval metrics for single-ground-truth-per-query text->image search.

With exactly one relevant gallery image per query, Average Precision = 1/rank, so
mAP = mean reciprocal rank.  R@k = fraction of queries whose GT is in the top-k.
"""
from __future__ import annotations

import torch


def retrieval_metrics(
    sim: torch.Tensor, gt_index: torch.Tensor, ks: tuple[int, ...] = (1, 5, 10)
) -> dict[str, float]:
    """sim: [Q, G] similarity (higher = better).  gt_index: [Q] long, the GT column per
    query.  Ties are broken optimistically (rank = 1 + #strictly-greater).
    """
    q = sim.size(0)
    gt_index = gt_index.long()
    gt_scores = sim[torch.arange(q, device=sim.device), gt_index].unsqueeze(1)  # [Q,1]
    ranks = (sim > gt_scores).sum(dim=1) + 1  # [Q] 1-based
    out: dict[str, float] = {}
    for k in ks:
        out[f"R@{k}"] = (ranks <= k).float().mean().item()
    out["mAP"] = (1.0 / ranks.float()).mean().item()
    out["mean_rank"] = ranks.float().mean().item()
    return out
