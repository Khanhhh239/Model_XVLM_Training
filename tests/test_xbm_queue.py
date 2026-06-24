"""Unit tests for XBM Queue."""
import torch
import pytest
from star.losses.xbm_queue import XBMQueue


def test_xbm_queue_init():
    """Test XBM queue initialization."""
    queue = XBMQueue(size=100, dim=64)
    assert queue.size == 100
    assert queue.dim == 64
    assert queue.img_queue.shape == (100, 64)
    assert queue.txt_queue.shape == (100, 64)
    assert queue.ptr.item() == 0


def test_xbm_enqueue_single_batch():
    """Test enqueuing a single batch."""
    queue = XBMQueue(size=100, dim=64)
    img_feat = torch.randn(16, 64)
    txt_feat = torch.randn(16, 64)
    
    queue.enqueue(img_feat, txt_feat)
    
    assert queue.ptr.item() == 16
    # Check normalization
    img_q, txt_q = queue.get_queue()
    assert torch.allclose(img_q[:16].norm(dim=1), torch.ones(16), atol=1e-5)
    assert torch.allclose(txt_q[:16].norm(dim=1), torch.ones(16), atol=1e-5)


def test_xbm_enqueue_fifo_wrap():
    """Test FIFO wrap-around behavior."""
    queue = XBMQueue(size=100, dim=64)
    
    # Fill queue
    for _ in range(7):
        img_feat = torch.randn(16, 64)
        txt_feat = torch.randn(16, 64)
        queue.enqueue(img_feat, txt_feat)
    
    # ptr should be at 112 % 100 = 12
    assert queue.ptr.item() == 12
    
    # Oldest entries (0-11) should be overwritten by newest (96-99, 100-107)


def test_xbm_detach_no_grad():
    """Test that enqueued features are detached (no grad)."""
    queue = XBMQueue(size=100, dim=64)
    img_feat = torch.randn(16, 64, requires_grad=True)
    txt_feat = torch.randn(16, 64, requires_grad=True)
    
    queue.enqueue(img_feat, txt_feat)
    
    img_q, txt_q = queue.get_queue()
    assert not img_q.requires_grad
    assert not txt_q.requires_grad


def test_xbm_queue_different_batch_sizes():
    """Test enqueueing different batch sizes."""
    queue = XBMQueue(size=100, dim=64)
    
    queue.enqueue(torch.randn(10, 64), torch.randn(10, 64))
    assert queue.ptr.item() == 10
    
    queue.enqueue(torch.randn(5, 64), torch.randn(5, 64))
    assert queue.ptr.item() == 15
    
    queue.enqueue(torch.randn(20, 64), torch.randn(20, 64))
    assert queue.ptr.item() == 35


def test_xbm_queue_normalization():
    """Test L2 normalization is applied correctly."""
    queue = XBMQueue(size=100, dim=64)
    
    # Create unnormalized features
    img_feat = torch.randn(16, 64) * 5.0  # Large magnitude
    txt_feat = torch.randn(16, 64) * 0.1  # Small magnitude
    
    queue.enqueue(img_feat, txt_feat)
    
    img_q, txt_q = queue.get_queue()
    
    # All features should have norm ~1.0
    img_norms = img_q[:16].norm(dim=1)
    txt_norms = txt_q[:16].norm(dim=1)
    
    assert torch.allclose(img_norms, torch.ones(16), atol=1e-5)
    assert torch.allclose(txt_norms, torch.ones(16), atol=1e-5)


def test_xbm_large_batch_wrap():
    """Test wrapping when batch size > remaining space."""
    queue = XBMQueue(size=100, dim=64)
    
    # Fill to 90
    queue.enqueue(torch.randn(90, 64), torch.randn(90, 64))
    assert queue.ptr.item() == 90
    
    # Enqueue 20 (should wrap: 90-100 filled, then 0-9 overwritten)
    img_feat = torch.arange(20 * 64).reshape(20, 64).float()
    txt_feat = torch.arange(20 * 64).reshape(20, 64).float() + 1000
    queue.enqueue(img_feat, txt_feat)
    
    assert queue.ptr.item() == 10  # (90 + 20) % 100
    
    # Check split correctness
    img_q, txt_q = queue.get_queue()
    # Entries 90-99 should have data from first 10 of batch
    # Entries 0-9 should have data from last 10 of batch


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
