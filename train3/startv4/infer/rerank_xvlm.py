"""X-VLM ITM cross-encoder re-ranking of the top-K candidates (Phase 3 precision stage).

The retrieval stage (SigLIP / ITC) gives a base score; we then cross-encode each query with
its top-K candidate images and blend the ITM match probability in.  Only top-K is cross-
encoded (cheap), the rest keep their base order below.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ..eval.rerank import minmax_per_row


@torch.no_grad()
def itm_scores_topk(model, query_tokens, query_mask, gallery_tokens, topk_idx, device="cpu") -> torch.Tensor:
    """query_tokens [Q,Lt,D], query_mask [Q,Lt], gallery_tokens [G,Ni,D], topk_idx [Q,K]
    -> ITM match probability [Q,K]."""
    model.eval()
    q, k = topk_idx.shape
    out = torch.zeros(q, k)
    for i in range(q):
        cand = gallery_tokens[topk_idx[i]].to(device)              # [K, Ni, D]
        qt = query_tokens[i : i + 1].to(device).expand(k, -1, -1)  # [K, Lt, D]
        qm = query_mask[i : i + 1].to(device).expand(k, -1)
        logits = model.itm_logits(cand, qt, qm)                    # [K, 2]
        out[i] = F.softmax(logits, dim=1)[:, 1].cpu()
    return out


@torch.no_grad()
def itm_rerank_ranking(
    model, base_score, query_tokens, query_mask, gallery_tokens, topk: int = 200, alpha: float = 0.5, device="cpu"
) -> torch.Tensor:
    """Return ranking [Q, G] of gallery indices: top-K re-ordered by blended
    ((1-alpha)*base + alpha*ITM), the remainder appended in base order."""
    q, g = base_score.shape
    k = min(topk, g)
    top = base_score.topk(k, dim=1).indices  # [Q, k]
    itm = itm_scores_topk(model, query_tokens, query_mask, gallery_tokens, top, device)
    base_top = base_score.gather(1, top)
    blended = (1 - alpha) * minmax_per_row(base_top) + alpha * minmax_per_row(itm)
    order = blended.argsort(dim=1, descending=True)
    top_sorted = top.gather(1, order)  # [Q, k] reranked gallery indices

    full = base_score.argsort(dim=1, descending=True)  # [Q, G] base order
    ranking = torch.empty_like(full)
    for i in range(q):
        topset = set(top[i].tolist())
        rest = [gi for gi in full[i].tolist() if gi not in topset]
        ranking[i] = torch.tensor(top_sorted[i].tolist() + rest, device=full.device)
    return ranking
