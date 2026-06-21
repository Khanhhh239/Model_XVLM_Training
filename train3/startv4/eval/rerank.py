"""Ensemble fusion + cross-modal re-ranking (training-free, run at inference).

- rrf_fuse           : Reciprocal Rank Fusion (rank-based, scale-free -> preferred).
- minmax_fuse        : per-query min-max normalise then weighted sum.
- query_expansion    : DBA -- pull a TEXT query into image-space via its top-k neighbours.
- k_reciprocal_rerank: Zhong et al. CVPR 2017.  For text queries, run query_expansion
                       FIRST, then this on the (now image-space) query (see README 8.4).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def minmax_per_row(sim: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mn = sim.min(dim=1, keepdim=True).values
    mx = sim.max(dim=1, keepdim=True).values
    return (sim - mn) / (mx - mn + eps)


def minmax_fuse(sims: list[torch.Tensor], weights: list[float]) -> torch.Tensor:
    """Per-query min-max each [Q,G] then weighted sum.  Higher = better."""
    if len(sims) != len(weights):
        raise ValueError("sims and weights length mismatch")
    out = torch.zeros_like(sims[0])
    for s, w in zip(sims, weights):
        out = out + float(w) * minmax_per_row(s)
    return out


def rrf_fuse(sims: list[torch.Tensor], k: int = 60) -> torch.Tensor:
    """Reciprocal Rank Fusion.  sims: list of [Q,G] (higher=better).  Returns [Q,G]."""
    fused = torch.zeros_like(sims[0])
    n = sims[0].size(1)
    for s in sims:
        order = torch.argsort(s, dim=1, descending=True)          # [Q,G] gallery idx by rank
        ranks = torch.empty_like(order)
        ar = torch.arange(n, device=s.device).expand_as(order)
        ranks.scatter_(1, order, ar)                              # ranks[q,g] = 0-based position
        fused = fused + 1.0 / (k + ranks.float() + 1.0)
    return fused


def query_expansion(
    query_feat: torch.Tensor, gallery_feat: torch.Tensor, topk: int = 5, alpha: float = 1.0
) -> torch.Tensor:
    """DBA.  query_feat [Q,D], gallery_feat [G,D] (L2-norm).  Returns expanded, re-normalised
    query [Q,D] living in image-space (so k-reciprocal can then run image-image)."""
    topk = min(topk, gallery_feat.size(0))
    sim = query_feat @ gallery_feat.t()                  # [Q,G]
    vals, idx = sim.topk(topk, dim=1)                     # [Q,topk]
    w = torch.softmax(vals, dim=1).unsqueeze(-1)          # [Q,topk,1]
    neigh = gallery_feat[idx]                             # [Q,topk,D]
    expanded = query_feat + alpha * (w * neigh).sum(dim=1)
    return F.normalize(expanded, dim=1)


def _euclidean(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise Euclidean distance (not squared)."""
    d2 = (a ** 2).sum(1)[:, None] + (b ** 2).sum(1)[None, :] - 2.0 * a @ b.T
    return np.sqrt(np.clip(d2, 0.0, None)).astype(np.float32)


def _re_ranking(q_g_dist, q_q_dist, g_g_dist, k1=20, k2=6, lambda_value=0.3):
    """Standard k-reciprocal re-ranking (Zhong et al., CVPR 2017).  Returns [Q,G] distance."""
    original_dist = np.concatenate(
        [
            np.concatenate([q_q_dist, q_g_dist], axis=1),
            np.concatenate([q_g_dist.T, g_g_dist], axis=1),
        ],
        axis=0,
    )
    original_dist = np.power(original_dist, 2).astype(np.float32)
    original_dist = np.transpose(original_dist / (np.max(original_dist, axis=0) + 1e-12))
    V = np.zeros_like(original_dist, dtype=np.float32)
    initial_rank = np.argsort(original_dist).astype(np.int32)

    query_num = q_g_dist.shape[0]
    all_num = original_dist.shape[0]
    k1 = min(k1, all_num - 1)
    half = int(np.around(k1 / 2)) + 1

    for i in range(all_num):
        forward = initial_rank[i, : k1 + 1]
        backward = initial_rank[forward, : k1 + 1]
        fi = np.where(backward == i)[0]
        k_recip = forward[fi]
        k_recip_exp = k_recip
        for cand in k_recip:
            cf = initial_rank[cand, :half]
            cb = initial_rank[cf, :half]
            fic = np.where(cb == cand)[0]
            cand_recip = cf[fic]
            if len(np.intersect1d(cand_recip, k_recip)) > 2.0 / 3.0 * len(cand_recip):
                k_recip_exp = np.append(k_recip_exp, cand_recip)
        k_recip_exp = np.unique(k_recip_exp)
        weight = np.exp(-original_dist[i, k_recip_exp])
        V[i, k_recip_exp] = (weight / np.sum(weight)).astype(np.float32)

    original_dist = original_dist[:query_num, ]
    if k2 != 1:
        k2 = min(k2, all_num)
        V_qe = np.zeros_like(V, dtype=np.float32)
        for i in range(all_num):
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe

    inv_index = [np.where(V[:, i] != 0)[0] for i in range(all_num)]
    jaccard = np.zeros_like(original_dist, dtype=np.float32)
    for i in range(query_num):
        temp_min = np.zeros((1, all_num), dtype=np.float32)
        ind_nz = np.where(V[i, :] != 0)[0]
        ind_imgs = [inv_index[ind] for ind in ind_nz]
        for j in range(len(ind_nz)):
            temp_min[0, ind_imgs[j]] += np.minimum(V[i, ind_nz[j]], V[ind_imgs[j], ind_nz[j]])
        jaccard[i] = 1.0 - temp_min / (2.0 - temp_min + 1e-12)

    final = jaccard * (1.0 - lambda_value) + original_dist * lambda_value
    return final[:query_num, query_num:]


def k_reciprocal_rerank(
    query_feat: torch.Tensor,
    gallery_feat: torch.Tensor,
    k1: int = 20,
    k2: int = 6,
    lam: float = 0.3,
) -> torch.Tensor:
    """query_feat [Q,D], gallery_feat [G,D] (image-space, L2-norm; run query_expansion
    first for text queries).  Returns a SCORE matrix [Q,G] (higher=better) = -final_dist.
    """
    q = query_feat.detach().cpu().numpy().astype(np.float32)
    g = gallery_feat.detach().cpu().numpy().astype(np.float32)
    q_g = _euclidean(q, g)
    q_q = _euclidean(q, q)
    g_g = _euclidean(g, g)
    final = _re_ranking(q_g, q_q, g_g, k1=k1, k2=k2, lambda_value=lam)
    return torch.from_numpy(-final).to(query_feat.device)
