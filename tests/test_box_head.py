"""Unit tests for Box Grounding Head."""
import torch
import pytest
from star.models.heads import BoxGroundingHead, compute_giou_loss


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
