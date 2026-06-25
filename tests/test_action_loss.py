"""Unit tests for the action-keyword alignment loss (math correctness, no GPU / no backbone)."""
import torch
import torch.nn.functional as F

from star.losses.action import action_alignment_loss


def test_reduces_to_symmetric_infonce_when_labels_unique():
    torch.manual_seed(0)
    n, d = 6, 16
    v = F.normalize(torch.randn(n, d), dim=-1)
    a = F.normalize(torch.randn(n, d), dim=-1)
    g = torch.arange(n)               # all unique -> ordinary InfoNCE
    temp = 0.07
    loss = action_alignment_loss(v, a, g, temp)
    sim = (v @ a.t()) / temp
    ref = 0.5 * (F.cross_entropy(sim, torch.arange(n)) + F.cross_entropy(sim.t(), torch.arange(n)))
    assert torch.allclose(loss, ref, atol=1e-5)


def test_matches_soft_target_formula_with_shared_labels():
    torch.manual_seed(1)
    v = F.normalize(torch.randn(5, 8), dim=-1)
    a = F.normalize(torch.randn(5, 8), dim=-1)
    g = torch.tensor([0, 0, 0, 1, 1])     # multi-positive groups
    temp = 0.1
    loss = action_alignment_loss(v, a, g, temp)
    sim = (v @ a.t()) / temp
    pos = (g[:, None] == g[None, :]).float()
    tgt = pos / pos.sum(1, keepdim=True)
    ref = 0.5 * (-(F.log_softmax(sim, 1) * tgt).sum(1).mean()
                 - (F.log_softmax(sim.t(), 1) * tgt).sum(1).mean())
    assert torch.allclose(loss, ref, atol=1e-6)


def test_perfect_alignment_is_lower_than_misaligned():
    torch.manual_seed(2)
    v = F.normalize(torch.randn(8, 32), dim=-1)
    g = torch.arange(8)
    aligned = action_alignment_loss(v, v.clone(), g, 0.05)
    misaligned = action_alignment_loss(v, F.normalize(torch.randn(8, 32), dim=-1), g, 0.05)
    assert aligned < misaligned


def test_group_ids_are_arbitrary_only_equality_matters():
    # the loss depends ONLY on the equality pattern of group_ids, not their literal values:
    # two same-action rows are positives regardless of which integer labels them.
    torch.manual_seed(4)
    v = F.normalize(torch.randn(6, 16), dim=-1)
    a = F.normalize(torch.randn(6, 16), dim=-1)
    g1 = torch.tensor([0, 0, 1, 1, 2, 2])
    g2 = torch.tensor([7, 7, 3, 3, 9, 9])          # same equality structure, different values
    assert torch.allclose(action_alignment_loss(v, a, g1, 0.07),
                          action_alignment_loss(v, a, g2, 0.07))


def test_multi_positive_differs_from_all_unique():
    # sanity: when several rows share an action, the loss is genuinely different from treating every
    # row as its own class (the soft target spreads probability mass over the positives).
    torch.manual_seed(5)
    v = F.normalize(torch.randn(6, 16), dim=-1)
    a = F.normalize(torch.randn(6, 16), dim=-1)
    shared = action_alignment_loss(v, a, torch.tensor([0, 0, 0, 1, 1, 1]), 0.07)
    unique = action_alignment_loss(v, a, torch.arange(6), 0.07)
    assert not torch.allclose(shared, unique)


def test_gradients_flow_to_image_features():
    v = F.normalize(torch.randn(4, 8), dim=-1).requires_grad_(True)
    a = F.normalize(torch.randn(4, 8), dim=-1)
    loss = action_alignment_loss(v, a, torch.arange(4), 0.07)
    loss.backward()
    assert v.grad is not None and torch.isfinite(v.grad).all()
