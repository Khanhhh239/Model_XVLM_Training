"""Phase 2 -- ensemble (RRF) + TTA + QE + k-reciprocal, on CACHED embeddings (free on T4).

  --siglip a.pt[,b.pt]   SigLIP cache(s); multiple = TTA scales (averaged)
  --xvlm   a.pt[,b.pt]   optional X-VLM cache(s) for ensemble
Each cache is distractor-val ({gt,distractor,query}) or submission ({gallery,gallery_ids,query}).

    python -m startv4.scripts.run_phase2 --siglip cache/siglip_512.pt,cache/siglip_768.pt \
        --xvlm cache/xvlm.pt --fuse rrf --qe-kr --out answer.txt
"""
from __future__ import annotations

import argparse

import torch

from ..infer.encode import average_features, load_embeddings
from ..infer.pipeline import build_ranking, evaluate_with_pipeline, write_answer


def _tta(files: list[str], key: str) -> torch.Tensor:
    feats = [load_embeddings(f)[key] for f in files]
    return average_features(feats) if len(feats) > 1 else feats[0]


def _xvlm_sim(files: list[str], query_key: str, gallery_keys: list[str]) -> torch.Tensor:
    q = _tta(files, query_key)
    g = torch.cat([_tta(files, k) for k in gallery_keys], dim=0)
    return q @ g.t()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--siglip", required=True, help="comma-separated SigLIP cache .pt (TTA scales)")
    ap.add_argument("--xvlm", default=None, help="comma-separated X-VLM cache .pt (ensemble)")
    ap.add_argument("--fuse", default="rrf", choices=["rrf", "minmax"])
    ap.add_argument("--qe-kr", action="store_true", help="query-expansion + k-reciprocal re-rank")
    ap.add_argument("--k1", type=int, default=20)
    ap.add_argument("--k2", type=int, default=6)
    ap.add_argument("--lam", type=float, default=0.3)
    ap.add_argument("--weights", default=None,
                    help="comma weights for --fuse minmax when ensembling, e.g. 0.6,0.4 "
                         "(SigLIP first). Use this instead of bare RRF when one model is "
                         "much stronger — RRF is rank-only and gives both models equal say.")
    ap.add_argument("--out", default="answer.txt")
    ap.add_argument("--topk", type=int, default=10)
    a = ap.parse_args()

    sig = a.siglip.split(",")
    xv = a.xvlm.split(",") if a.xvlm else None
    keys = load_embeddings(sig[0])
    mode = "distractor" if "gt" in keys else "gallery"
    weights = [float(w) for w in a.weights.split(",")] if a.weights else None
    kw = dict(fuse=a.fuse, weights=weights, use_qe_kr=a.qe_kr, k1=a.k1, k2=a.k2, lam=a.lam)

    if mode == "distractor":
        q, gt, dis = _tta(sig, "query"), _tta(sig, "gt"), _tta(sig, "distractor")
        extra = [_xvlm_sim(xv, "query", ["gt", "distractor"])] if xv else None
        metrics, _ = evaluate_with_pipeline(q, gt, dis, extra_sims=extra, **kw)
        print("[phase2] distractor-val: " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
    else:
        q, g = _tta(sig, "query"), _tta(sig, "gallery")
        ids = keys["gallery_ids"]
        extra = [_xvlm_sim(xv, "query", ["gallery"])] if xv else None
        rank = build_ranking(q, g, extra_sims=extra, **kw)
        write_answer(rank, ids, a.out, topk=a.topk)
        print(f"[phase2] wrote {rank.size(0)} rows -> {a.out}")


if __name__ == "__main__":
    main()
