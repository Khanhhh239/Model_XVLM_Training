"""Smooth-AP loss (Brown et al., ECCV 2020).

Directly optimises Average Precision by replacing the non-differentiable rank indicator
1[s_k > s_l] with a sigmoid.  AP ranking objective complements the contrastive losses.
"""
from __future__ import annotations

import torch


def smooth_ap_loss(
    sim: torch.Tensor, pos_mask: torch.Tensor, tau: float = 0.01, eps: float = 1e-6
) -> torch.Tensor:
    """sim: [B, N] query-vs-candidate scores.  pos_mask: [B, N] in {0, 1}, 1 = relevant.
    Returns 1 - mean(smooth-AP).  Queries with no positive contribute 0 AP.

    rank_all(l)  = 1 + sum_{k != l} sigmoid((s_k - s_l) / tau)
    rank_pos(l)  = 1 + sum_{k in P, k != l} sigmoid((s_k - s_l) / tau)
    AP_i = (1/|P_i|) sum_{l in P_i} rank_pos(l) / rank_all(l)
    """
    b, n = sim.shape
    # diff[i, a, b] = s[i, a] - s[i, b]
    diff = sim.unsqueeze(2) - sim.unsqueeze(1)          # [B, N, N]
    g = torch.sigmoid(-diff / tau)                       # g[i,a,b] = sigmoid((s_b - s_a)/tau)
    eye = torch.eye(n, device=sim.device, dtype=torch.bool)
    g = g.masked_fill(eye.unsqueeze(0), 0.0)

    pm = pos_mask.to(sim.dtype)
    rank_all = 1.0 + g.sum(dim=2)                        # [B, N]
    rank_pos = 1.0 + (g * pm.unsqueeze(1)).sum(dim=2)    # [B, N]
    ap_terms = (rank_pos / (rank_all + eps)) * pm        # only positives count
    n_pos = pm.sum(dim=1).clamp(min=1.0)
    ap = ap_terms.sum(dim=1) / n_pos                     # [B]
    has_pos = (pos_mask.sum(dim=1) > 0).to(sim.dtype)
    return (1.0 - ap * has_pos).mean()
