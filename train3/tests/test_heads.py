import torch

from startv4.models.heads import AnomalyHead, BBoxHead, FilipProjection, PoseRegionFuse


def test_bbox_head_range():
    h = BBoxHead(in_dim=32)
    out = h(torch.randn(5, 32))
    assert out.shape == (5, 4)
    assert (out >= 0).all() and (out <= 1).all()


def test_anomaly_head_shape():
    h = AnomalyHead(in_dim=32, num_classes=3)
    assert h(torch.randn(7, 32)).shape == (7, 3)


def test_pose_region_fuse_shape_and_gate():
    f = PoseRegionFuse(region_dim=16)
    region = torch.randn(2, 9, 16)
    kpts = torch.rand(2, 17, 3)
    out = f(region, kpts)
    assert out.shape == region.shape
    # with gate init 0.1 the change is bounded but nonzero
    assert not torch.allclose(out, region)


def test_filip_projection_normalised():
    p = FilipProjection(img_dim=16, txt_dim=24, proj_dim=8)
    vi, vt = p(torch.randn(2, 4, 16), torch.randn(2, 6, 24))
    assert vi.shape == (2, 4, 8) and vt.shape == (2, 6, 8)
    assert torch.allclose(vi.norm(dim=-1), torch.ones(2, 4), atol=1e-4)
