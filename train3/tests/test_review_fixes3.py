"""Third-pass (pragmatic / no-waste) tests: X-VLM rerank-only mode + distractor subsampling."""
from pathlib import Path

import torch
from PIL import Image

from startv4.config import load_config
from startv4.data.dataset import DummyTokenizer
from startv4.infer.encode import make_distractor_eval_fn
from startv4.models.siglip_retrieval import build_siglip
from startv4.models.xvlm_v4 import build_xvlm
from startv4.train.trainer_xvlm import XVLMTrainer

ROOT = Path(__file__).resolve().parents[1]


def _xbatch(b=8):
    return {
        "pixel_values": torch.randn(b, 3, 32, 32),
        "input_ids": torch.randint(1, 1000, (b, 16)),
        "attention_mask": torch.ones(b, 16, dtype=torch.long),
        "keypoints": torch.rand(b, 17, 3),
        "bbox": torch.tensor([[0.1, 0.1, 0.5, 0.6]] * b),
        "has_bbox": torch.ones(b, dtype=torch.bool),
        "bucket": torch.randint(0, 2, (b,)),
        "index": torch.arange(b),
    }


def _imgs(tmp_path, n):
    out = []
    for i in range(n):
        p = tmp_path / f"r{i}.png"
        Image.new("RGB", (40, 40), (i * 30 % 255, 90, 120)).save(p)
        out.append(str(p))
    return out


def test_rerank_only_skips_itc_queue_filip_smoothap():
    cfg = load_config(ROOT / "configs" / "_test_xvlm_dummy.yaml")
    cfg.optim["w_itc"] = 0.0
    cfg.optim["w_filip"] = 0.0
    cfg.optim["w_smoothap"] = 0.0
    cfg.optim["queue_size"] = 0
    cfg.model["use_filip"] = False
    cfg.model["use_anomaly"] = False
    cfg.model["use_pose"] = False
    trainer = XVLMTrainer(build_xvlm(cfg), cfg, "cpu")
    assert trainer.queue_img is None and trainer.queue_txt is None  # no ITC -> no queue
    logs = trainer.train_step(_xbatch())
    assert "itm" in logs and logs["total"] == logs["total"]  # finite, ITM-only
    for absent in ("itc", "filip", "smoothap", "anom"):
        assert absent not in logs


def test_rerank_only_overfits():
    cfg = load_config(ROOT / "configs" / "_test_xvlm_dummy.yaml")
    for k, v in dict(w_itc=0.0, w_filip=0.0, w_smoothap=0.0, queue_size=0).items():
        cfg.optim[k] = v
    cfg.model["use_filip"] = False
    cfg.model["use_anomaly"] = False
    cfg.model["use_pose"] = False
    trainer = XVLMTrainer(build_xvlm(cfg), cfg, "cpu")
    losses = trainer.overfit_one_batch(_xbatch(), steps=60)
    assert min(losses[-5:]) < losses[0]  # ITM-only still learns


def test_max_distractors_runs(tmp_path):
    paths = _imgs(tmp_path, 8)
    index = {"gt": paths[:2], "distractors": paths[2:]}  # 6 distractors
    fn = make_distractor_eval_fn(index, ["a b", "c d"], DummyTokenizer(16), 32, "cpu", max_distractors=3)
    m = fn(build_siglip(load_config(ROOT / "configs" / "_test_dummy.yaml")))
    assert "mAP" in m and 0.0 <= m["R@1"] <= 1.0
