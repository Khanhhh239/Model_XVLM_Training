"""End-to-end wiring test for STAGE 1.5: all heads ON (pose + box + anomaly + XBM + action) — the
forward must return all 7 finite loss terms and the full graph must be differentiable. Runs on the
dummy backbone (CPU, no downloads, no X-VLM)."""
import torch

from star.config import Config
from star.models import STARModel


def _cfg() -> Config:
    cfg = Config()
    cfg.model.backbone = "dummy"
    cfg.model.embed_dim = 64
    cfg.model.lora_enabled = True
    cfg.model.lora_freeze_text = True
    cfg.model.pose_enabled = True
    cfg.model.bbox_enabled = True
    cfg.model.anomaly_enabled = True
    cfg.loss.xbm_enabled = True
    cfg.loss.xbm_size = 32
    cfg.loss.lambda_box = 0.1
    cfg.loss.lambda_anomaly = 0.2
    cfg.loss.lambda_action = 0.2
    cfg.model.phrase_box_enabled = True
    cfg.loss.lambda_phrase_box = 0.1
    return cfg


def _batch(b=4, L=16):
    return {
        "image": torch.randn(b, 3, 384, 384),
        "input_ids": torch.randint(5, 900, (b, L)),
        "attention_mask": torch.ones(b, L, dtype=torch.long),
        "instance_id": torch.arange(b),
        "bbox": torch.rand(b, 4) * 0.5 + 0.2,                 # xywh in (0.2, 0.7)
        "bbox_mask": torch.ones(b, dtype=torch.bool),
        "anomaly_label": torch.randint(0, 2, (b,)),
        "anomaly_mask": torch.ones(b, dtype=torch.bool),
        "keypoints": torch.rand(b, 51),
        "action_input_ids": torch.randint(5, 900, (b, 8)),
        "action_attention_mask": torch.ones(b, 8, dtype=torch.long),
        "action_group": torch.tensor([0, 0, 1, 1]),
        "action_valid": torch.ones(b, dtype=torch.bool),
        "phrase_input_ids": torch.randint(5, 900, (b, 2, 8)),         # [B, P=2, L=8]
        "phrase_attention_mask": torch.ones(b, 2, 8, dtype=torch.long),
        "phrase_box": torch.rand(b, 2, 4) * 0.5 + 0.2,                # xywh in (0.2, 0.7)
        "phrase_mask": torch.ones(b, 2),
    }


def test_all_heads_forward_and_backward():
    torch.manual_seed(0)
    model = STARModel(_cfg())
    out = model(_batch(), step=10)
    keys = {"loss", "loss_itc", "loss_itm", "loss_smap", "loss_box", "loss_anomaly",
            "loss_action", "loss_pbox"}
    assert set(out) == keys
    for k in keys:
        assert torch.isfinite(torch.as_tensor(out[k])), f"{k} not finite"
    assert out["loss_action"].item() > 0      # action loss actually computed (valid rows present)
    assert out["loss_pbox"].item() > 0        # phrase-box loss computed (valid phrase boxes present)
    out["loss"].backward()                    # whole multi-head graph is differentiable


def test_action_skipped_when_lambda_zero():
    cfg = _cfg()
    cfg.loss.lambda_action = 0.0
    out = STARModel(cfg)(_batch(), step=1)
    assert float(out["loss_action"]) == 0.0   # not computed -> exact zero


def test_action_skipped_when_fewer_than_two_valid():
    cfg = _cfg()
    model = STARModel(cfg)
    b = _batch()
    b["action_valid"] = torch.tensor([True, False, False, False])   # only 1 valid -> skip
    out = model(b, step=1)
    assert float(out["loss_action"]) == 0.0


def test_config_loads_stage15_yaml():
    # the Stage-1.5 yaml must only use keys the typed config knows (load_config raises on unknown)
    from pathlib import Path

    from star.config import load_config
    cfg_path = Path(__file__).resolve().parents[1] / "configs" / "stage1_run3_bbox.yaml"
    cfg = load_config(str(cfg_path))
    assert cfg.model.bbox_enabled and cfg.model.anomaly_enabled
    assert cfg.loss.lambda_box > 0
    assert cfg.model.pose_enabled is True
