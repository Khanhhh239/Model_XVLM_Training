"""Tests for the review remediation (P0/P1): scheduler, queues, keypoints threading,
eval hook + grad-norm, zeroshot baseline import."""
import importlib
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from _helpers import make_dataset

from startv4.config import load_config
from startv4.data import PABDatasetV4, collate_fn
from startv4.data.sampler import BalancedBucketSampler
from startv4.infer.encode import encode_image_loader
from startv4.models.siglip_retrieval import build_siglip
from startv4.models.xvlm_v4 import build_xvlm
from startv4.train.sched import cosine_warmup
from startv4.train.trainer import SiglipTrainer
from startv4.train.trainer_xvlm import XVLMTrainer

ROOT = Path(__file__).resolve().parents[1]


def _siglip_cfg():
    return load_config(ROOT / "configs" / "_test_dummy.yaml")


def _xvlm_cfg():
    return load_config(ROOT / "configs" / "_test_xvlm_dummy.yaml")


def _sbatch(b=8):
    return {
        "pixel_values": torch.randn(b, 3, 32, 32),
        "input_ids": torch.randint(1, 1000, (b, 16)),
        "attention_mask": torch.ones(b, 16, dtype=torch.long),
    }


# ---- A3: cosine warmup scheduler ----
def test_cosine_warmup_rises_then_decays():
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([p], lr=1.0)
    sch = cosine_warmup(opt, warmup_steps=5, total_steps=20)
    lrs = []
    for _ in range(20):
        opt.step()
        sch.step()
        lrs.append(opt.param_groups[0]["lr"])
    assert max(lrs[:6]) > lrs[0]      # warming up
    assert lrs[-1] < max(lrs)         # decaying
    assert lrs[-1] < 0.1              # near the cosine floor


# ---- A4: SigLIP accepts (ignores) keypoints; loader threads them ----
def test_siglip_encode_accepts_keypoints():
    m = build_siglip(_siglip_cfg())
    f, _ = m.encode_image(torch.randn(2, 3, 32, 32), torch.rand(2, 17, 3))  # must not raise
    assert f.shape == (2, 64)


def test_encode_loader_threads_keypoints_to_pose_model(tmp_path):
    cfg = _xvlm_cfg()
    mani, root = make_dataset(tmp_path, 6)
    ds = PABDatasetV4(mani, root, split="train", image_size=32, train=False, max_token=16)
    loader = DataLoader(ds, batch_size=3, collate_fn=collate_fn, num_workers=0)
    feat = encode_image_loader(build_xvlm(cfg), loader, "cpu")  # pose-on model + keypoints in batch
    assert feat.shape == (6, 64)


# ---- A2: X-VLM has two queues and enqueues both modalities ----
def test_xvlm_two_queues_enqueue():
    cfg = _xvlm_cfg()
    trainer = XVLMTrainer(build_xvlm(cfg), cfg, "cpu")
    assert trainer.queue_img is not None and trainer.queue_txt is not None
    trainer.train_step({
        "pixel_values": torch.randn(8, 3, 32, 32),
        "input_ids": torch.randint(1, 1000, (8, 16)),
        "attention_mask": torch.ones(8, 16, dtype=torch.long),
        "keypoints": torch.rand(8, 17, 3),
        "bbox": torch.tensor([[0.1, 0.1, 0.5, 0.6]] * 8),
        "has_bbox": torch.ones(8, dtype=torch.bool),
        "bucket": torch.randint(0, 2, (8,)),
        "index": torch.arange(8),
    })
    assert int(trainer.queue_img.filled.item()) == 8
    assert int(trainer.queue_txt.filled.item()) == 8


# ---- B: eval hook + grad-norm logging + scheduler wired in fit ----
def test_grad_norm_logged():
    trainer = SiglipTrainer(build_siglip(_siglip_cfg()), _siglip_cfg(), "cpu")
    logs = trainer.train_step(_sbatch())
    assert "grad_norm" in logs and logs["grad_norm"] >= 0


def test_eval_hook_tracks_best_and_builds_scheduler(tmp_path):
    cfg = _siglip_cfg()
    mani, root = make_dataset(tmp_path, 8)
    ds = PABDatasetV4(mani, root, split="train", image_size=32, train=True, max_token=16, cfg=cfg)
    loader = DataLoader(ds, batch_sampler=BalancedBucketSampler(ds.buckets(), 4, seed=1),
                        collate_fn=collate_fn, num_workers=0)
    trainer = SiglipTrainer(build_siglip(cfg), cfg, "cpu")
    # 8 items, batch 4, drop_last -> 2 steps/epoch * 2 epochs = 4 evals
    scores = iter([0.3, 0.7, 0.5, 0.6])
    seen = []
    best = trainer.fit(loader, epochs=2, log_every=1, eval_fn=lambda model: {"mAP": next(scores)},
                       eval_every=1, on_eval=lambda s, m, b: seen.append(m["mAP"]))
    assert best["mAP"] == 0.7 and "step" in best
    assert seen == [0.3, 0.7, 0.5, 0.6]
    assert trainer.sched is not None


# ---- P1.2: zeroshot baseline script importable ----
def test_zeroshot_script_has_main():
    assert hasattr(importlib.import_module("startv4.scripts.zeroshot_baseline"), "main")
