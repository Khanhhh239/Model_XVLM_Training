import torch
import torch.nn.functional as F

from startv4.losses import (
    box_giou,
    box_loss,
    filip_loss,
    filip_sim,
    info_nce,
    siglip_sigmoid_loss,
    smooth_ap_loss,
)

torch.manual_seed(0)


def test_siglip_aligned_beats_random():
    img = F.normalize(torch.randn(6, 16), dim=1)
    scale, bias = torch.tensor(10.0), torch.tensor(-10.0)
    aligned = siglip_sigmoid_loss(img, img.clone(), scale, bias)
    rnd = siglip_sigmoid_loss(img, F.normalize(torch.randn(6, 16), dim=1), scale, bias)
    assert torch.isfinite(aligned) and aligned >= 0
    assert aligned < rnd


def test_info_nce_with_queue_shapes_and_aligned():
    img = F.normalize(torch.randn(4, 8), dim=1)
    scale = torch.tensor(10.0)
    q = F.normalize(torch.randn(32, 8), dim=1)
    aligned = info_nce(img, img.clone(), scale, queue_text=q, queue_image=q)
    rnd = info_nce(img, F.normalize(torch.randn(4, 8), dim=1), scale)
    assert torch.isfinite(aligned) and aligned >= 0
    assert aligned < rnd


def test_filip_sim_and_loss():
    vi = F.normalize(torch.randn(3, 5, 8), dim=-1)
    vt = F.normalize(torch.randn(3, 7, 8), dim=-1)
    mask = torch.ones(3, 7)
    mask[:, 5:] = 0
    sim = filip_sim(vi, vt, mask)
    assert sim.shape == (3, 3)
    loss = filip_loss(vi, vt, torch.tensor(5.0), mask)
    assert torch.isfinite(loss)


def test_box_giou_and_loss():
    b = torch.tensor([[0.1, 0.1, 0.5, 0.5]])
    assert torch.allclose(box_giou(b, b), torch.ones(1), atol=1e-5)
    assert box_loss(b, b) < 1e-5
    far = torch.tensor([[0.6, 0.6, 0.9, 0.9]])
    assert box_loss(b, far) > box_loss(b, b)


def test_smooth_ap_perfect_vs_bad():
    # 1 query, 4 candidates, the single positive is index 0
    pos = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    good = torch.tensor([[5.0, 1.0, 0.5, 0.0]])    # positive ranked first
    bad = torch.tensor([[0.0, 1.0, 2.0, 5.0]])     # positive ranked last
    assert smooth_ap_loss(good, pos) < smooth_ap_loss(bad, pos)
    assert torch.isfinite(smooth_ap_loss(good, pos))


def test_smooth_ap_no_positive_is_zero_ap():
    sim = torch.randn(2, 5)
    nopos = torch.zeros(2, 5)
    # no positives -> AP term 0 -> loss == 1
    assert abs(smooth_ap_loss(sim, nopos).item() - 1.0) < 1e-5
