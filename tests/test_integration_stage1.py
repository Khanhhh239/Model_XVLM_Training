"""Integration tests for STAGE 1 full pipeline."""
import torch
import pytest
from star.models import STARModel
from star.config import Config, DataConfig, ModelConfig, LossConfig, OptimConfig, TrainConfig


def get_stage1_config():
    """Create STAGE 1 config for testing."""
    cfg = Config()
    cfg.model.embed_dim = 256
    cfg.model.bbox_enabled = True
    cfg.model.anomaly_enabled = True
    cfg.model.pose_enabled = True
    cfg.loss.xbm_enabled = True
    cfg.loss.xbm_size = 128  # Small for testing
    cfg.loss.lambda_box = 0.1
    cfg.loss.lambda_anomaly = 0.2
    cfg.loss.anomaly_rampup_steps = 50
    return cfg


def test_star_model_builds_with_stage1_heads():
    """Test that STARModel builds with all STAGE 1 heads."""
    cfg = get_stage1_config()
    model = STARModel(cfg)
    
    # Check heads exist
    assert model.bbox_head is not None
    assert model.anomaly_head is not None
    assert model.pose is not None
    assert model.xbm_queue is not None
    
    # Check XBM queue size
    assert model.xbm_queue.size == 128
    assert model.xbm_queue.dim == 256


def test_star_model_forward_with_all_fields():
    """Test forward pass with bbox, anomaly, keypoints."""
    cfg = get_stage1_config()
    model = STARModel(cfg)
    model.eval()
    
    batch = {
        "image": torch.randn(4, 3, 384, 384),
        "input_ids": torch.randint(0, 1000, (4, 100)),
        "attention_mask": torch.ones(4, 100),
        "instance_id": torch.tensor([0, 1, 2, 3]),
        "bbox": torch.rand(4, 4),  # Normalized [0, 1]
        "bbox_mask": torch.tensor([True, True, False, True]),
        "anomaly_label": torch.tensor([0, 1, 0, 1]),
        "anomaly_mask": torch.tensor([True, True, True, False]),
        "keypoints": torch.rand(4, 51),  # 17 joints × 3
    }
    
    with torch.no_grad():
        out = model(batch, step=0)
    
    # Check all losses are present
    assert "loss" in out
    assert "loss_itc" in out
    assert "loss_itm" in out
    assert "loss_smap" in out
    assert "loss_box" in out
    assert "loss_anomaly" in out
    
    # Check loss values are valid
    assert not torch.isnan(out["loss"])
    assert not torch.isinf(out["loss"])
    
    # Box loss should be 0 for masked samples only
    assert out["loss_box"].item() >= 0
    
    # Anomaly loss should be 0 at step 0 (ramp-up)
    assert out["loss_anomaly"].item() == 0.0


def test_star_model_forward_without_optional_fields():
    """Test forward pass without bbox, anomaly, keypoints (should not crash)."""
    cfg = get_stage1_config()
    model = STARModel(cfg)
    model.eval()
    
    batch = {
        "image": torch.randn(4, 3, 384, 384),
        "input_ids": torch.randint(0, 1000, (4, 100)),
        "attention_mask": torch.ones(4, 100),
        "instance_id": torch.tensor([0, 1, 2, 3]),
        # No bbox, anomaly, keypoints
    }
    
    with torch.no_grad():
        out = model(batch, step=0)
    
    # Should still work, with box/anomaly losses = 0
    assert out["loss_box"].item() == 0.0
    assert out["loss_anomaly"].item() == 0.0


def test_anomaly_rampup():
    """Test anomaly loss ramp-up over steps."""
    cfg = get_stage1_config()
    cfg.loss.anomaly_rampup_steps = 100
    model = STARModel(cfg)
    model.eval()
    
    batch = {
        "image": torch.randn(2, 3, 384, 384),
        "input_ids": torch.randint(0, 1000, (2, 100)),
        "attention_mask": torch.ones(2, 100),
        "instance_id": torch.tensor([0, 1]),
        "anomaly_label": torch.tensor([0, 1]),
        "anomaly_mask": torch.tensor([True, True]),
    }
    
    with torch.no_grad():
        # Step 0: rampup_factor = 0
        out_0 = model(batch, step=0)
        assert out_0["loss_anomaly"].item() == 0.0
        
        # Step 50: rampup_factor = 0.5
        out_50 = model(batch, step=50)
        loss_50 = out_50["loss_anomaly"].item()
        
        # Step 100: rampup_factor = 1.0
        out_100 = model(batch, step=100)
        loss_100 = out_100["loss_anomaly"].item()
        
        # Step 200: rampup_factor = 1.0 (capped)
        out_200 = model(batch, step=200)
        loss_200 = out_200["loss_anomaly"].item()
    
    # Losses should increase with ramp-up
    assert loss_50 > 0
    assert loss_100 > loss_50
    assert abs(loss_200 - loss_100) < 1e-5  # Capped at 1.0


def test_xbm_queue_updates():
    """Test that XBM queue is updated during forward."""
    cfg = get_stage1_config()
    model = STARModel(cfg)
    model.eval()
    
    batch = {
        "image": torch.randn(4, 3, 384, 384),
        "input_ids": torch.randint(0, 1000, (4, 100)),
        "attention_mask": torch.ones(4, 100),
        "instance_id": torch.tensor([0, 1, 2, 3]),
    }
    
    # Initial ptr
    ptr_before = model.xbm_queue.ptr.item()
    
    with torch.no_grad():
        model(batch, step=0)
    
    # Ptr should advance by batch size
    ptr_after = model.xbm_queue.ptr.item()
    assert ptr_after == ptr_before + 4


def test_loss_weighting():
    """Test that loss weights are applied correctly."""
    cfg = get_stage1_config()
    cfg.loss.w_itc = 1.0
    cfg.loss.lambda_itm = 1.5
    cfg.loss.lambda_smooth_ap = 0.2
    cfg.loss.lambda_box = 0.1
    cfg.loss.lambda_anomaly = 0.3
    
    model = STARModel(cfg)
    model.eval()
    
    batch = {
        "image": torch.randn(4, 3, 384, 384),
        "input_ids": torch.randint(0, 1000, (4, 100)),
        "attention_mask": torch.ones(4, 100),
        "instance_id": torch.tensor([0, 1, 2, 3]),
        "bbox": torch.rand(4, 4),
        "bbox_mask": torch.ones(4, dtype=torch.bool),
        "anomaly_label": torch.tensor([0, 1, 0, 1]),
        "anomaly_mask": torch.ones(4, dtype=torch.bool),
    }
    
    with torch.no_grad():
        out = model(batch, step=1000)  # After ramp-up
    
    # Total loss should be weighted sum (approximately)
    # Note: weighter might add small overhead, but should be close
    expected = (1.0 * out["loss_itc"] + 
                1.5 * out["loss_itm"] + 
                0.2 * out["loss_smap"] +
                0.1 * out["loss_box"] +
                0.3 * out["loss_anomaly"])
    
    # Allow small tolerance for weighter overhead
    assert abs(out["loss"].item() - expected.item()) < 0.1


def test_trainable_parameters():
    """Test that correct parameters are trainable in STAGE 1."""
    cfg = get_stage1_config()
    cfg.model.lora_enabled = False  # STAGE 1 doesn't use LoRA
    model = STARModel(cfg)
    
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    frozen = [name for name, p in model.named_parameters() if not p.requires_grad]
    
    # Check bbox_head is trainable
    assert any("bbox_head" in name for name in trainable)
    
    # Check anomaly_head is trainable
    assert any("anomaly_head" in name for name in trainable)
    
    # Check pose is trainable
    assert any("pose" in name for name in trainable)
    
    # Check backbone vision is frozen (dummy backbone doesn't have vision, skip this)
    # In real X-VLM: assert any("visual_encoder" in name for name in frozen)


def test_gradient_flow_to_new_heads():
    """Test that gradients flow to new heads."""
    cfg = get_stage1_config()
    model = STARModel(cfg)
    
    batch = {
        "image": torch.randn(2, 3, 384, 384),
        "input_ids": torch.randint(0, 1000, (2, 100)),
        "attention_mask": torch.ones(2, 100),
        "instance_id": torch.tensor([0, 1]),
        "bbox": torch.rand(2, 4),
        "bbox_mask": torch.ones(2, dtype=torch.bool),
        "anomaly_label": torch.tensor([0, 1]),
        "anomaly_mask": torch.ones(2, dtype=torch.bool),
    }
    
    # Zero grad
    model.zero_grad()
    
    # Forward + backward
    out = model(batch, step=100)
    out["loss"].backward()
    
    # Check gradients on new heads
    for name, param in model.named_parameters():
        if "bbox_head" in name or "anomaly_head" in name:
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert param.grad.abs().sum() > 1e-6, f"Zero gradient for {name}"


def test_safety_net_logic():
    """Test safety net: XBM queue and parameters are saved/restored correctly."""
    cfg = get_stage1_config()
    model = STARModel(cfg)
    
    # Fill XBM queue with some data
    batch = {
        "image": torch.randn(4, 3, 384, 384),
        "input_ids": torch.randint(0, 1000, (4, 100)),
        "attention_mask": torch.ones(4, 100),
        "instance_id": torch.tensor([0, 1, 2, 3]),
    }
    model.eval()
    with torch.no_grad():
        model(batch, step=0)
    
    # Save state - IMPORTANT: Deep copy to avoid shared memory issue in PyTorch 2.6+
    # In production, torch.save() handles this automatically
    import copy
    state_dict = copy.deepcopy(model.state_dict())
    
    # Clone original weights for comparison
    bbox_weights_orig = state_dict['bbox_head.mlp.0.weight'].clone()
    anomaly_weights_orig = state_dict['anomaly_head.mlp.0.weight'].clone()
    xbm_img_orig = state_dict['xbm_queue.img_queue'].clone()
    xbm_ptr_orig = state_dict['xbm_queue.ptr'].clone()
    
    # Simulate parameter drift during training (modify in-place)
    with torch.no_grad():
        model.bbox_head.mlp[0].weight.add_(torch.randn_like(model.bbox_head.mlp[0].weight) * 0.1)
        model.anomaly_head.mlp[0].weight.add_(torch.randn_like(model.anomaly_head.mlp[0].weight) * 0.1)
        model.xbm_queue.img_queue.add_(torch.randn_like(model.xbm_queue.img_queue) * 0.1)
        model.xbm_queue.ptr[0] = 50
    
    # Verify model has changed
    state_dict_modified = model.state_dict()
    assert not torch.allclose(state_dict_modified['bbox_head.mlp.0.weight'], bbox_weights_orig)
    
    # RESTORE from saved checkpoint (safety net scenario)
    model.load_state_dict(state_dict)
    
    # Verify restoration
    state_dict_restored = model.state_dict()
    assert torch.allclose(state_dict_restored['bbox_head.mlp.0.weight'], bbox_weights_orig, atol=1e-6)
    assert torch.allclose(state_dict_restored['anomaly_head.mlp.0.weight'], anomaly_weights_orig, atol=1e-6)
    assert torch.allclose(state_dict_restored['xbm_queue.img_queue'], xbm_img_orig, atol=1e-6)
    assert torch.equal(state_dict_restored['xbm_queue.ptr'], xbm_ptr_orig)


def test_batch_size_scaling():
    """Test model works with different batch sizes."""
    cfg = get_stage1_config()
    model = STARModel(cfg)
    model.eval()
    
    for batch_size in [1, 4, 8, 16]:
        batch = {
            "image": torch.randn(batch_size, 3, 384, 384),
            "input_ids": torch.randint(0, 1000, (batch_size, 100)),
            "attention_mask": torch.ones(batch_size, 100),
            "instance_id": torch.arange(batch_size),
        }
        
        with torch.no_grad():
            out = model(batch, step=0)
        
        assert not torch.isnan(out["loss"])
        assert out["loss"].item() > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
