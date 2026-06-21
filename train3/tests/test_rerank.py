import torch
import torch.nn.functional as F

from startv4.eval.rerank import (
    k_reciprocal_rerank,
    minmax_fuse,
    minmax_per_row,
    query_expansion,
    rrf_fuse,
)


def test_minmax_per_row():
    s = torch.tensor([[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]])
    o = minmax_per_row(s)
    assert torch.allclose(o.min(1).values, torch.zeros(2), atol=1e-6)
    assert torch.allclose(o.max(1).values, torch.ones(2), atol=1e-6)


def test_minmax_fuse_weights():
    a = torch.tensor([[0.0, 1.0]])
    b = torch.tensor([[1.0, 0.0]])
    fused = minmax_fuse([a, b], [0.7, 0.3])
    assert fused.shape == (1, 2)


def test_rrf_agreement():
    s = torch.tensor([[3.0, 2.0, 1.0]])
    fused = rrf_fuse([s, s.clone()])
    assert fused.argmax(1).item() == 0


def test_query_expansion_normalised():
    q = F.normalize(torch.randn(2, 8), dim=1)
    g = F.normalize(torch.randn(10, 8), dim=1)
    e = query_expansion(q, g, topk=3)
    assert e.shape == (2, 8)
    assert torch.allclose(e.norm(dim=1), torch.ones(2), atol=1e-4)


def test_k_reciprocal_runs_finite_and_keeps_self():
    g = F.normalize(torch.randn(6, 8), dim=1)
    q = g[:3].clone()
    score = k_reciprocal_rerank(q, g, k1=4, k2=2, lam=0.3)
    assert score.shape == (3, 6)
    assert torch.isfinite(score).all()
    top2 = score.topk(2, dim=1).indices.tolist()
    for i in range(3):
        assert i in top2[i]
