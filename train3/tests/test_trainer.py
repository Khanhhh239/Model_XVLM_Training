from pathlib import Path

import torch
from torch.utils.data import DataLoader

from _helpers import make_dataset

from startv4.config import load_config
from startv4.data import BalancedBucketSampler, PABDatasetV4, collate_fn
from startv4.models.siglip_retrieval import build_siglip
from startv4.train.trainer import SiglipTrainer

ROOT = Path(__file__).resolve().parents[1]


def _cfg():
    return load_config(ROOT / "configs" / "_test_dummy.yaml")


def test_overfit_one_batch_decreases():
    cfg = _cfg()
    trainer = SiglipTrainer(build_siglip(cfg), cfg, "cpu")
    b = 8
    batch = {
        "pixel_values": torch.randn(b, 3, 32, 32),
        "input_ids": torch.randint(1, 1000, (b, 16)),
        "attention_mask": torch.ones(b, 16, dtype=torch.long),
    }
    losses = trainer.overfit_one_batch(batch, steps=120)
    assert losses[-1] < losses[0]
    assert losses[-1] < 0.7 * losses[0]


def test_full_train_loop_runs(tmp_path):
    cfg = _cfg()
    mani, root = make_dataset(tmp_path, 8)
    ds = PABDatasetV4(mani, root, split="train", image_size=32, train=True, max_token=16, cfg=cfg)
    sampler = BalancedBucketSampler(ds.buckets(), 4, seed=42)
    loader = DataLoader(ds, batch_sampler=sampler, collate_fn=collate_fn, num_workers=0)
    trainer = SiglipTrainer(build_siglip(cfg), cfg, "cpu")
    logs = []
    trainer.fit(loader, epochs=2, log_every=1, on_log=lambda e, s, l: logs.append(l))
    assert logs and all(torch.isfinite(torch.tensor(x["total"])) for x in logs)


def test_ema_present_and_updates():
    cfg = _cfg()
    trainer = SiglipTrainer(build_siglip(cfg), cfg, "cpu")
    assert trainer.ema is not None
    before = {k: v.clone() for k, v in trainer.ema.state_dict().items()}
    batch = {
        "pixel_values": torch.randn(8, 3, 32, 32),
        "input_ids": torch.randint(1, 1000, (8, 16)),
        "attention_mask": torch.ones(8, 16, dtype=torch.long),
    }
    trainer.train_step(batch)
    after = trainer.ema.state_dict()
    assert any(not torch.allclose(before[k], after[k]) for k in before)
