import torch
import torch.nn.functional as F

from startv4.eval.distractor_val import evaluate_with_distractors


def test_perfect_when_gt_equals_query():
    q = F.normalize(torch.randn(4, 8), dim=1)
    gt = q.clone()  # GT image embedding == query embedding
    distract = F.normalize(torch.randn(50, 8), dim=1)
    m = evaluate_with_distractors(q, gt, distract)
    assert m["R@1"] == 1.0
    assert abs(m["mAP"] - 1.0) < 1e-6


def test_recall_drops_with_hard_distractors():
    q = F.normalize(torch.randn(5, 8), dim=1)
    gt = F.normalize(q + 0.5 * torch.randn(5, 8), dim=1)  # noisy GT
    distract = F.normalize(torch.randn(200, 8), dim=1)
    m = evaluate_with_distractors(q, gt, distract, ks=(1, 10))
    assert 0.0 <= m["R@1"] <= 1.0 and 0.0 <= m["mAP"] <= 1.0
