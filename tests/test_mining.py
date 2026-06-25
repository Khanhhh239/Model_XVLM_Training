"""Unit tests for ANCE cross-ID mining + hard_edges + pair-pool blending (CPU, no model)."""
import json
import random

import torch

from star.data.mining import build_pair_pool, load_hard_edges, mine_cross_id_pairs


def test_mine_picks_hardest_cross_video():
    img = torch.eye(4)                       # img[j] = e_j
    txt = torch.zeros(4, 4)
    txt[0] = torch.tensor([0.0, 0.0, 1.0, 0.0])   # -> img2
    txt[1] = torch.tensor([0.9, 0.0, 0.0, 0.5])   # -> img0 (same video -> masked) then img3
    txt[2] = torch.tensor([1.0, 0.0, 0.0, 0.0])   # -> img0
    txt[3] = torch.tensor([0.0, 1.0, 0.0, 0.0])   # -> img1
    vids = [0, 0, 1, 1]
    picked = mine_cross_id_pairs(img, txt, vids, block=2)  # block=2 exercises the block loop
    assert picked.tolist() == [2, 3, 0, 1]


def test_mine_never_picks_same_video():
    torch.manual_seed(0)
    feats = torch.randn(20, 8)
    vids = [i // 5 for i in range(20)]       # 4 videos of 5 frames
    picked = mine_cross_id_pairs(feats, feats.clone(), vids)
    for i, j in enumerate(picked.tolist()):
        assert vids[j] != vids[i]            # cross-video guaranteed


def test_load_hard_edges(tmp_path):
    p = tmp_path / "edges.jsonl"
    p.write_text("\n".join([
        json.dumps({"anchor_image_id": "0_1", "hard_image_id": "0_9"}),
        json.dumps({"anchor_image_id": "0_2", "hard_image_id": "0_8"}),
        json.dumps({"anchor_image_id": "0_3", "hard_image_id": "9_9"}),  # 9_9 not in map -> skip
        "",                                                              # blank line ignored
    ]), encoding="utf-8")
    idx = {"0_1": 0, "0_9": 5, "0_2": 1, "0_8": 6, "0_3": 2}
    assert load_hard_edges(str(p), idx) == [(0, 5), (1, 6)]


def test_load_hard_edges_empty_path():
    assert load_hard_edges(None, {}) == []


def test_build_pair_pool_blend_counts_and_groups():
    base = [(i, i + 1000) for i in range(100)]
    edges = [(i, i + 2000) for i in range(50)]
    mined = [(i, i + 3000) for i in range(100)]
    vof = {i: i % 7 for i in range(100)}     # anchors are all < 100
    pairs, groups = build_pair_pool(base, edges, mined, vof,
                                    mine_fraction=0.5, max_edge_fraction=0.25,
                                    rng=random.Random(0))
    assert len(pairs) == len(base) == 100    # fixed pool size -> stable epoch length
    assert len(groups) == 100
    for (a, _), g in zip(pairs, groups):
        assert g == vof[a]                   # group = anchor's video id


def test_build_pair_pool_fraction_zero_is_all_base():
    base = [(i, i + 1000) for i in range(40)]
    edges = [(i, i + 2000) for i in range(40)]
    mined = [(i, i + 3000) for i in range(40)]
    vof = {i: 0 for i in range(40)}
    pairs, _ = build_pair_pool(base, edges, mined, vof,
                               mine_fraction=0.0, max_edge_fraction=0.0, rng=random.Random(1))
    assert sorted(pairs) == sorted(base)     # no mining / no edges -> pure base
