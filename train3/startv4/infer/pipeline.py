"""Fusion + re-ranking pipeline -> score / ranking / answer.txt  (Phase 2).

Stages (see ../STAR_v4_architecture.md sec 3.5):
  1. base sim = query . gallery   (after optional TTA multi-scale averaging upstream)
  2. ensemble with extra model sims (X-VLM) via RRF or per-query min-max
  3. QE/DBA (text query -> image-space) THEN k-reciprocal re-ranking
  4. argsort -> ranked gallery ids per query.

`fuse_and_rerank` returns the final SCORE matrix so metrics can be computed on it;
`build_ranking` argsorts it; `evaluate_with_pipeline` runs the whole thing under
distractor-val and reports R@k / mAP.
"""
from __future__ import annotations

from pathlib import Path

import torch

from ..eval.metrics import retrieval_metrics
from ..eval.rerank import k_reciprocal_rerank, minmax_fuse, query_expansion, rrf_fuse


def fuse_and_rerank(
    query_feat: torch.Tensor,
    gallery_feat: torch.Tensor,
    extra_sims: list[torch.Tensor] | None = None,
    fuse: str = "rrf",
    weights: list[float] | None = None,
    use_qe_kr: bool = False,
    qe_topk: int = 5,
    k1: int = 20,
    k2: int = 6,
    lam: float = 0.3,
) -> torch.Tensor:
    """Return final SCORE [Q, G] (higher = better)."""
    base = query_feat @ gallery_feat.t()  # [Q, G]
    sims = [base] + (extra_sims or [])

    if len(sims) == 1:
        score = base
    elif fuse == "rrf":
        score = rrf_fuse(sims)
    elif fuse == "minmax":
        w = weights or ([1.0 / len(sims)] * len(sims))
        score = minmax_fuse(sims, w)
    else:
        raise ValueError(f"unknown fuse {fuse!r}")

    if use_qe_kr:
        q_exp = query_expansion(query_feat, gallery_feat, topk=qe_topk)
        kr = k_reciprocal_rerank(q_exp, gallery_feat, k1=k1, k2=k2, lam=lam)  # [Q,G] score
        score = minmax_fuse([score, kr], [0.5, 0.5])
    return score


def build_ranking(query_feat, gallery_feat, **kw) -> torch.Tensor:
    """Return ranking [Q, G] of gallery indices (best first)."""
    score = fuse_and_rerank(query_feat, gallery_feat, **kw)
    return torch.argsort(score, dim=1, descending=True)


def evaluate_with_pipeline(
    query_feat: torch.Tensor,
    gt_feat: torch.Tensor,
    distractor_feat: torch.Tensor,
    ks: tuple[int, ...] = (1, 5, 10),
    extra_sims: list[torch.Tensor] | None = None,
    **rerank_kw,
) -> tuple[dict[str, float], torch.Tensor]:
    """Distractor-val under the full Phase-2 pipeline.  gallery = [GT ; distractors];
    each query's GT is column i.  Returns (metrics, final_score)."""
    gallery = torch.cat([gt_feat, distractor_feat], dim=0)  # [Q+M, D]
    score = fuse_and_rerank(query_feat, gallery, extra_sims=extra_sims, **rerank_kw)
    gt_index = torch.arange(query_feat.size(0), device=query_feat.device)
    return retrieval_metrics(score, gt_index, ks=ks), score


def write_answer(ranking: torch.Tensor, gallery_ids: list[str], out_path: str | Path, topk: int = 10) -> None:
    """ranking [Q, G] gallery indices -> one line per query of top-k gallery ids."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for row in ranking.tolist():
        ids = [gallery_ids[i] for i in row[:topk]]
        lines.append(" ".join(ids))
    Path(out_path).write_text("\n".join(lines), encoding="utf-8")
