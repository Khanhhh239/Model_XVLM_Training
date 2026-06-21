from pathlib import Path

import torch
from torch.utils.data import DataLoader

from _helpers import make_dataset

from startv4.config import load_config
from startv4.data import PABDatasetV4, collate_fn
from startv4.infer.rerank_xvlm import itm_rerank_ranking, itm_scores_topk
from startv4.models.xvlm_v4 import build_xvlm
from startv4.train.mining import mine_hard_negatives

ROOT = Path(__file__).resolve().parents[1]


def _cfg():
    return load_config(ROOT / "configs" / "_test_xvlm_dummy.yaml")


def test_mining_returns_non_self_mapping(tmp_path):
    cfg = _cfg()
    mani, root = make_dataset(tmp_path, 8)
    ds = PABDatasetV4(mani, root, split="train", image_size=32, train=False, max_token=16)
    loader = DataLoader(ds, batch_size=4, collate_fn=collate_fn, num_workers=0)
    m = build_xvlm(cfg)
    mapping = mine_hard_negatives(m, loader, "cpu")
    assert len(mapping) == 8
    assert all(k != v for k, v in mapping.items())


def test_itm_scores_and_ranking():
    m = build_xvlm(_cfg())
    g, q = 8, 3
    _, gallery_tokens = m.encode_image(torch.randn(g, 3, 32, 32), torch.rand(g, 17, 3))
    _, qtok, qmask = m.encode_text(torch.randint(1, 1000, (q, 16)), torch.ones(q, 16, dtype=torch.long))

    top = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7], [0, 2, 4, 6]])
    scores = itm_scores_topk(m, qtok, qmask, gallery_tokens, top, "cpu")
    assert scores.shape == (3, 4)
    assert (scores >= 0).all() and (scores <= 1).all()

    base = torch.randn(q, g)
    rank = itm_rerank_ranking(m, base, qtok, qmask, gallery_tokens, topk=4, alpha=0.5)
    assert rank.shape == (q, g)
    for i in range(q):
        assert sorted(rank[i].tolist()) == list(range(g))  # valid permutation
