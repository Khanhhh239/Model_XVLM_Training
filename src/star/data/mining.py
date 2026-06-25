"""ANCE cross-ID hard-negative mining + hard_edges swap-pairs for the PairBatchSampler pool.

The data team's same-video pairs (`pair_image_id = hard_i_id`) give tier-1 hard negatives. This
module adds:
  - tier-2 ANCE: with the CURRENT model, each anchor's hardest CROSS-video image (a look-alike of a
    different identity — exactly what the 34k gallery distractors are). Re-mined each epoch.
  - hard_edges swap-pairs: same-video / different-action pairs from hard_edges_30k_hard.jsonl
    (wrong-case group B: identity/scene swap).

`build_pair_pool` blends the three sources into a fixed-size pool so the epoch length (and thus the
LR schedule) stays stable. All functions take plain tensors / lists -> unit-testable on CPU.
"""
from __future__ import annotations

import json
import random

import torch
import torch.nn.functional as F


@torch.no_grad()
def mine_cross_id_pairs(img_feats, txt_feats, video_ids, block: int = 1024):
    """For each anchor i return the image j with the highest text_i->image_j similarity AND a
    DIFFERENT video id (same-video entries, incl. self, are masked -> never chosen).

    img_feats / txt_feats: [N, D] (re-normalized here). video_ids: [N]. Returns LongTensor [N].
    Block-wise argmax so the [N,N] similarity is never materialized in full (30k-safe).
    """
    img = F.normalize(img_feats.float(), dim=-1)
    txt = F.normalize(txt_feats.float(), dim=-1)
    vid = torch.as_tensor(video_ids).to(img.device)          # match device (mining runs on GPU)
    n = img.size(0)
    picked = torch.empty(n, dtype=torch.long)                # always returned on CPU
    for s in range(0, n, block):
        e = min(n, s + block)
        sim = txt[s:e] @ img.t()                              # [b, N]
        same = vid[s:e, None] == vid[None, :]                 # same video (incl. self) -> not cross-ID
        sim = sim.masked_fill(same, float("-inf"))
        picked[s:e] = sim.argmax(dim=1).cpu()
    return picked


def load_hard_edges(path, id_to_index):
    """Parse hard_edges_*.jsonl -> [(anchor_idx, hard_idx)] for edges whose BOTH image ids exist in
    the dataset. `id_to_index` maps str(image_id) -> row index."""
    pairs = []
    if not path:
        return pairs
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            a = id_to_index.get(str(d.get("anchor_image_id")))
            h = id_to_index.get(str(d.get("hard_image_id")))
            if a is not None and h is not None and a != h:
                pairs.append((a, h))
    return pairs


def build_pair_pool(base_pairs, edge_pairs, mined_pairs, video_of_index,
                    mine_fraction: float = 0.5, max_edge_fraction: float = 0.25, rng=None):
    """Blend the three pair sources into a ~len(base_pairs)-sized pool for PairBatchSampler.

    Returns (pairs, groups) where group = anchor's video id (so a batch's anchors stay video-distinct
    -> cross-pair items are clean negatives). Fixed total size keeps steps/epoch (and the cosine LR
    schedule) stable across re-mining epochs.
    """
    rng = rng or random.Random(0)
    n = len(base_pairs)
    n_mine = min(len(mined_pairs), int(round(mine_fraction * n)))
    n_edge = min(len(edge_pairs), int(round(max_edge_fraction * n)))
    n_base = max(0, n - n_mine - n_edge)

    def take(src, k):
        return rng.sample(src, k) if 0 <= k < len(src) else list(src)

    chosen = take(base_pairs, n_base) + take(mined_pairs, n_mine) + take(edge_pairs, n_edge)
    rng.shuffle(chosen)
    groups = [video_of_index[a] for (a, _) in chosen]
    return chosen, groups
