from pathlib import Path

import torch
from torch.utils.data import DataLoader

from _helpers import make_dataset

from startv4.config import load_config
from startv4.data import BalancedBucketSampler, PABDatasetV4, collate_fn
from startv4.models.xvlm_v4 import build_xvlm
from startv4.train.trainer_xvlm import XVLMTrainer

ROOT = Path(__file__).resolve().parents[1]


def _cfg():
    return load_config(ROOT / "configs" / "_test_xvlm_dummy.yaml")


def _batch(b=8):
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


def test_compute_loss_has_all_terms():
    trainer = XVLMTrainer(build_xvlm(_cfg()), _cfg(), "cpu")
    _, logs = trainer.compute_loss(_batch())
    for k in ("itc", "itm", "filip", "box", "anom", "smoothap", "total"):
        assert k in logs and logs[k] == logs[k]  # not NaN


def test_overfit_decreases():
    trainer = XVLMTrainer(build_xvlm(_cfg()), _cfg(), "cpu")
    losses = trainer.overfit_one_batch(_batch(), steps=100)
    first = sum(losses[:3]) / 3
    best_late = min(losses[-10:])
    assert best_late < first


def test_full_loop_and_ema(tmp_path):
    cfg = _cfg()
    mani, root = make_dataset(tmp_path, 8)
    ds = PABDatasetV4(mani, root, split="train", image_size=32, train=True, max_token=16, cfg=cfg)
    loader = DataLoader(ds, batch_sampler=BalancedBucketSampler(ds.buckets(), 4, seed=1),
                        collate_fn=collate_fn, num_workers=0)
    trainer = XVLMTrainer(build_xvlm(cfg), cfg, "cpu")
    logs = []
    trainer.fit(loader, epochs=2, log_every=1, on_log=lambda e, s, l: logs.append(l))
    assert logs and all(v == v for v in logs[-1].values())
    assert trainer.ema is not None and trainer.queue_img is not None and trainer.queue_txt is not None
