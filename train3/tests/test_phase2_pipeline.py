import torch
import torch.nn.functional as F

from startv4.infer.pipeline import build_ranking, evaluate_with_pipeline, fuse_and_rerank


def test_score_and_ranking_consistent():
    g = F.normalize(torch.randn(10, 8), dim=1)
    q = g[[1, 4, 7]].clone()
    score = fuse_and_rerank(q, g)
    assert score.shape == (3, 10)
    assert torch.equal(build_ranking(q, g), torch.argsort(score, dim=1, descending=True))


def test_evaluate_perfect_and_noisy():
    q = F.normalize(torch.randn(4, 8), dim=1)
    m, score = evaluate_with_pipeline(q, q.clone(), F.normalize(torch.randn(30, 8), dim=1))
    assert m["R@1"] == 1.0 and score.shape == (4, 34)
    gt_noisy = F.normalize(q + 0.5 * torch.randn(4, 8), dim=1)
    m2, _ = evaluate_with_pipeline(q, gt_noisy, F.normalize(torch.randn(30, 8), dim=1), ks=(1, 10))
    assert 0.0 <= m2["R@1"] <= 1.0 and 0.0 <= m2["mAP"] <= 1.0


def test_ensemble_and_qekr_runs_perfect():
    q = F.normalize(torch.randn(3, 8), dim=1)
    gt = q.clone()
    dis = F.normalize(torch.randn(20, 8), dim=1)
    gallery = torch.cat([gt, dis], dim=0)
    extra = [q @ gallery.t()]
    m, _ = evaluate_with_pipeline(q, gt, dis, extra_sims=extra, fuse="rrf", use_qe_kr=True, k1=4, k2=2)
    assert m["R@1"] == 1.0


def test_minmax_fuse_mode_runs():
    q = F.normalize(torch.randn(3, 8), dim=1)
    g = F.normalize(torch.randn(12, 8), dim=1)
    extra = [q @ g.t()]
    score = fuse_and_rerank(q, g, extra_sims=extra, fuse="minmax", weights=[0.6, 0.4])
    assert score.shape == (3, 12) and torch.isfinite(score).all()
