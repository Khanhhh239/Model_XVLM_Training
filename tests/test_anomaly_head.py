"""Unit tests for Anomaly Classification Head."""
import torch
import pytest
from star.models.heads import AnomalyClassificationHead


def test_anomaly_head_forward():
    """Test anomaly head forward pass."""
    head = AnomalyClassificationHead(input_dim=256, hidden_dim=128)
    img_feat = torch.randn(8, 256)
    
    logits = head(img_feat)
    
    assert logits.shape == (8, 2)  # Binary classification


def test_anomaly_loss_computation():
    """Test anomaly classification loss."""
    logits = torch.tensor([
        [2.0, -1.0],  # Predicts class 0 (normal)
        [-1.0, 2.0],  # Predicts class 1 (anomaly)
        [0.5, 0.5],   # Uncertain
    ])
    labels = torch.tensor([0, 1, 0])  # Ground truth: normal, anomaly, normal
    
    loss = AnomalyClassificationHead.compute_loss(logits, labels)
    
    # First two samples should have low loss (correct predictions)
    # Third sample should have higher loss (uncertain prediction for normal)
    assert loss.item() > 0


def test_anomaly_loss_perfect_predictions():
    """Test loss with perfect predictions (should be low)."""
    logits = torch.tensor([
        [10.0, -10.0],  # Very confident class 0
        [-10.0, 10.0],  # Very confident class 1
    ])
    labels = torch.tensor([0, 1])
    
    loss = AnomalyClassificationHead.compute_loss(logits, labels)
    
    # Perfect predictions → low loss
    assert loss.item() < 0.1


def test_anomaly_loss_with_mask():
    """Test loss computation with valid/invalid mask."""
    logits = torch.tensor([
        [2.0, -1.0],
        [-1.0, 2.0],
        [0.0, 0.0],   # Invalid sample (masked)
    ])
    labels = torch.tensor([0, 1, 0])
    mask = torch.tensor([True, True, False])
    
    loss_masked = AnomalyClassificationHead.compute_loss(logits, labels, mask=mask)
    
    # Should only compute loss on first 2 samples
    assert loss_masked.item() > 0
    
    # Compare with full loss
    loss_full = AnomalyClassificationHead.compute_loss(logits, labels, mask=None)
    
    # Masked loss should be different (excludes 3rd sample)
    assert abs(loss_masked.item() - loss_full.item()) > 1e-6


def test_anomaly_loss_empty_mask():
    """Test loss with all invalid samples (should return 0)."""
    logits = torch.randn(5, 2)
    labels = torch.randint(0, 2, (5,))
    mask = torch.zeros(5, dtype=torch.bool)
    
    loss = AnomalyClassificationHead.compute_loss(logits, labels, mask=mask)
    
    assert loss.item() == 0.0


def test_anomaly_label_range():
    """Test that labels should be 0 or 1."""
    head = AnomalyClassificationHead()
    img_feat = torch.randn(4, 256)
    logits = head(img_feat)
    
    # Valid labels: 0 (normal) and 1 (anomaly)
    valid_labels = torch.tensor([0, 1, 0, 1])
    loss = AnomalyClassificationHead.compute_loss(logits, valid_labels)
    assert not torch.isnan(loss)
    
    # Invalid labels should raise error in CrossEntropyLoss
    invalid_labels = torch.tensor([0, 1, 2, 3])  # Out of range
    with pytest.raises(Exception):
        AnomalyClassificationHead.compute_loss(logits, invalid_labels)


def test_anomaly_head_output_distribution():
    """Test that logits can represent both classes."""
    head = AnomalyClassificationHead(input_dim=256, hidden_dim=128)
    
    # Create features that might produce different predictions
    img_feat_normal = torch.randn(10, 256) * 0.5
    img_feat_anomaly = torch.randn(10, 256) * 1.5
    
    logits_normal = head(img_feat_normal)
    logits_anomaly = head(img_feat_anomaly)
    
    # Both should produce valid logits
    assert logits_normal.shape == (10, 2)
    assert logits_anomaly.shape == (10, 2)
    
    # Logits should not be all zeros (head is learning)
    assert logits_normal.abs().sum() > 0.1
    assert logits_anomaly.abs().sum() > 0.1


def test_anomaly_gradient_flow():
    """Test that gradients flow through anomaly head."""
    head = AnomalyClassificationHead(input_dim=256)
    img_feat = torch.randn(4, 256, requires_grad=True)
    logits = head(img_feat)
    labels = torch.tensor([0, 1, 0, 1])
    
    loss = AnomalyClassificationHead.compute_loss(logits, labels)
    loss.backward()
    
    # Check that gradients exist
    assert img_feat.grad is not None
    assert img_feat.grad.abs().sum() > 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
