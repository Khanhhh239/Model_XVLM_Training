"""Unit tests for Box Grounding Head."""
import torch
import pytest
from star.models.heads import BoxGroundingHead, compute_giou_loss
from star.data.dataset import _bbox_from_kpts


def test_bbox_from_kpts_basic():
    """Keypoint extent -> COCO xywh-norm box. 4 visible joints spanning a 0.2..0.6 square."""
    kp = [0.0, 0.0, 0.0] * 17
    pts = {5: (0.2, 0.3), 6: (0.6, 0.3), 11: (0.2, 0.7), 12: (0.6, 0.7)}  # shoulders + hips
    for j, (x, y) in pts.items():
        kp[3 * j], kp[3 * j + 1], kp[3 * j + 2] = x, y, 0.9
    box = _bbox_from_kpts(kp, margin=0.0)
    assert box is not None
    cx, cy, w, h = box
    assert cx == pytest.approx(0.4, abs=1e-6) and cy == pytest.approx(0.5, abs=1e-6)
    assert w == pytest.approx(0.4, abs=1e-6) and h == pytest.approx(0.4, abs=1e-6)


def test_bbox_from_kpts_margin_and_clamp():
    """Margin expands the box but stays clamped to [0,1]."""
    kp = [0.0, 0.0, 0.0] * 17
    for j, (x, y) in {5: (0.02, 0.02), 12: (0.98, 0.98)}.items():
        kp[3 * j], kp[3 * j + 1], kp[3 * j + 2] = x, y, 0.9
    box = _bbox_from_kpts(kp, margin=0.05)
    cx, cy, w, h = box
    x1, y1 = cx - w / 2, cy - h / 2
    x2, y2 = cx + w / 2, cy + h / 2
    assert x1 >= 0.0 and y1 >= 0.0 and x2 <= 1.0 and y2 <= 1.0   # clamped


def test_bbox_from_kpts_low_conf_returns_none():
    """All joints below conf threshold (or fewer than 2 visible) -> None (masked out)."""
    kp = [0.5, 0.5, 0.05] * 17           # all conf 0.05 < 0.1
    assert _bbox_from_kpts(kp) is None
    one = [0.0, 0.0, 0.0] * 17           # only 1 visible joint
    one[0], one[1], one[2] = 0.4, 0.4, 0.9
    assert _bbox_from_kpts(one) is None


def test_bbox_from_kpts_bad_length():
    """Wrong-length input -> None (no crash)."""
    assert _bbox_from_kpts(None) is None
    assert _bbox_from_kpts([0.1, 0.2, 0.3]) is None


def test_box_head_forward():
    """Test box head forward pass."""
    head = BoxGroundingHead(input_dim=256, hidden_dim=128)
    img_feat = torch.randn(8, 256)
    
    bbox_pred = head(img_feat)
    
    assert bbox_pred.shape == (8, 4)
    # Should be in [0, 1] due to sigmoid
    assert (bbox_pred >= 0).all() and (bbox_pred <= 1).all()


def test_box_loss_with_mask():
    """Test box loss computation with valid/invalid mask."""
    bbox_pred = torch.tensor([
        [0.3, 0.2, 0.4, 0.5],
        [0.5, 0.5, 0.3, 0.3],
        [0.1, 0.1, 0.2, 0.2],
    ])
    bbox_gt = torch.tensor([
        [0.3, 0.2, 0.4, 0.5],  # Perfect match
        [0.6, 0.6, 0.2, 0.2],  # Mismatch
        [0.1, 0.1, 0.2, 0.2],  # Perfect match
    ])
    mask = torch.tensor([True, True, False])  # Skip last one
    
    loss = BoxGroundingHead.compute_loss(bbox_pred, bbox_gt, mask=mask)
    
    # Loss should be > 0 (not perfect due to second sample)
    assert loss.item() > 0
    
    # Test without mask (all samples)
    loss_all = BoxGroundingHead.compute_loss(bbox_pred, bbox_gt, mask=None)
    assert loss_all.item() > 0


def test_box_loss_empty_mask():
    """Test box loss with all invalid samples (should return 0)."""
    bbox_pred = torch.randn(5, 4).clamp(0, 1)
    bbox_gt = torch.randn(5, 4).clamp(0, 1)
    mask = torch.zeros(5, dtype=torch.bool)
    
    loss = BoxGroundingHead.compute_loss(bbox_pred, bbox_gt, mask=mask)
    
    assert loss.item() == 0.0


def test_giou_loss_perfect_match():
    """Test GIoU loss with perfect match (should be ~0)."""
    boxes1 = torch.tensor([[0.3, 0.3, 0.4, 0.4]])
    boxes2 = boxes1.clone()
    
    loss = compute_giou_loss(boxes1, boxes2)
    
    # GIoU = 1 for perfect match → loss = 1 - 1 = 0
    assert loss.item() < 1e-5


def test_giou_loss_no_overlap():
    """Test GIoU loss with no overlap."""
    boxes1 = torch.tensor([[0.1, 0.1, 0.2, 0.2]])  # Top-left
    boxes2 = torch.tensor([[0.7, 0.7, 0.2, 0.2]])  # Bottom-right
    
    loss = compute_giou_loss(boxes1, boxes2)
    
    # No overlap → IoU = 0 → GIoU < 0 → loss > 1
    assert loss.item() > 1.0


def test_giou_loss_partial_overlap():
    """Test GIoU loss with partial overlap."""
    boxes1 = torch.tensor([[0.3, 0.3, 0.4, 0.4]])
    boxes2 = torch.tensor([[0.4, 0.4, 0.4, 0.4]])  # Shifted
    
    loss = compute_giou_loss(boxes1, boxes2)
    
    # Some overlap → 0 < loss < 1
    assert 0 < loss.item() < 1


def test_giou_loss_batch():
    """Test GIoU loss with batch of boxes."""
    boxes1 = torch.tensor([
        [0.3, 0.3, 0.4, 0.4],
        [0.1, 0.1, 0.2, 0.2],
        [0.5, 0.5, 0.3, 0.3],
    ])
    boxes2 = torch.tensor([
        [0.3, 0.3, 0.4, 0.4],  # Perfect match
        [0.15, 0.15, 0.2, 0.2],  # Partial overlap
        [0.8, 0.8, 0.1, 0.1],  # No overlap
    ])
    
    loss = compute_giou_loss(boxes1, boxes2)
    
    # Should be mean of 3 samples
    assert loss.item() > 0


def test_box_format_conversion():
    """Test [x, y, w, h] center format conversion to corners."""
    # Box at center (0.5, 0.5) with size 0.4x0.4
    boxes = torch.tensor([[0.5, 0.5, 0.4, 0.4]])
    
    # Convert to corners manually
    x, y, w, h = 0.5, 0.5, 0.4, 0.4
    x1, y1 = x - w/2, y - h/2  # 0.3, 0.3
    x2, y2 = x + w/2, y + h/2  # 0.7, 0.7
    
    # Self-GIoU should be 0 (perfect match)
    loss = compute_giou_loss(boxes, boxes)
    assert loss.item() < 1e-5


def test_box_clamping():
    """Test that boxes are clamped to [0, 1] range."""
    # Create boxes that would go outside [0, 1]
    boxes1 = torch.tensor([[-0.1, -0.1, 0.5, 0.5]])  # Negative coords
    boxes2 = torch.tensor([[0.9, 0.9, 0.5, 0.5]])   # Exceeds 1.0
    
    # GIoU should handle clamping gracefully
    loss = compute_giou_loss(boxes1, boxes2)
    assert not torch.isnan(loss) and not torch.isinf(loss)


def test_box_loss_weight_ratio():
    """Test GIoU:L1 weight ratio of 2:5."""
    bbox_pred = torch.tensor([[0.3, 0.3, 0.4, 0.4]])
    bbox_gt = torch.tensor([[0.35, 0.35, 0.4, 0.4]])  # Slightly shifted
    
    # Test default weights (2:5)
    loss_default = BoxGroundingHead.compute_loss(bbox_pred, bbox_gt, w_giou=2.0, w_l1=5.0)
    
    # Test different weights
    loss_equal = BoxGroundingHead.compute_loss(bbox_pred, bbox_gt, w_giou=1.0, w_l1=1.0)
    
    # Both should be positive
    assert loss_default.item() > 0
    assert loss_equal.item() > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
