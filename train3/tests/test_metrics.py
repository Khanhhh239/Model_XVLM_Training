import torch

from startv4.eval.metrics import retrieval_metrics


def test_known_ranks():
    # query0 GT=col0 ranked 1st; query1 GT=col3 ranked last (rank 4)
    sim = torch.tensor(
        [
            [0.9, 0.5, 0.4, 0.1],
            [0.9, 0.8, 0.7, 0.1],
        ]
    )
    gt = torch.tensor([0, 3])
    m = retrieval_metrics(sim, gt, ks=(1, 5))
    assert abs(m["R@1"] - 0.5) < 1e-6          # only query0 hits rank-1
    assert abs(m["mAP"] - (1.0 + 0.25) / 2) < 1e-6
    assert abs(m["mean_rank"] - 2.5) < 1e-6


def test_perfect_retrieval():
    sim = torch.eye(5)
    gt = torch.arange(5)
    m = retrieval_metrics(sim, gt)
    assert abs(m["R@1"] - 1.0) < 1e-6
    assert abs(m["mAP"] - 1.0) < 1e-6
