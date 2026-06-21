"""Second-pass remediation tests: FILIP chunking, retrieval-space encoders, distractor
eval-fn builder, best-state snapshot, and a real-model smoke (skips if HF is unreachable)."""
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from _helpers import make_dataset

from startv4.config import Cfg, load_config
from startv4.data import BalancedBucketSampler, PABDatasetV4, collate_fn
from startv4.data.dataset import DummyTokenizer
from startv4.infer.encode import (
    encode_retrieval_images,
    encode_retrieval_text,
    make_distractor_eval_fn,
)
from startv4.losses.filip import filip_sim
from startv4.models.siglip_retrieval import build_siglip
from startv4.models.xvlm_v4 import build_xvlm
from startv4.train.trainer import SiglipTrainer

ROOT = Path(__file__).resolve().parents[1]


def _siglip_cfg():
    return load_config(ROOT / "configs" / "_test_dummy.yaml")


def _xvlm_cfg():
    return load_config(ROOT / "configs" / "_test_xvlm_dummy.yaml")


def _imgs(tmp_path, n):
    paths = []
    for i in range(n):
        p = tmp_path / f"r{i}.png"
        Image.new("RGB", (40, 40), (i * 30 % 255, 90, 120)).save(p)
        paths.append(str(p))
    return paths


def test_filip_chunk_matches_full():
    vi = F.normalize(torch.randn(6, 5, 8), dim=-1)
    vt = F.normalize(torch.randn(6, 7, 8), dim=-1)
    mask = torch.ones(6, 7)
    mask[:, 5:] = 0
    full = filip_sim(vi, vt, mask, chunk=0)
    chunked = filip_sim(vi, vt, mask, chunk=2)
    assert full.shape == (6, 6)
    assert torch.allclose(full, chunked, atol=1e-5)


def test_retrieval_encoders_normalised(tmp_path):
    paths = _imgs(tmp_path, 4)
    tok = DummyTokenizer(max_len=16)
    # SigLIP: encode_image is already the retrieval feature (no .itc)
    ms = build_siglip(_siglip_cfg())
    fi = encode_retrieval_images(ms, paths, 32, "cpu", 2)
    ft = encode_retrieval_text(ms, ["a b", "c d e"], tok, "cpu")
    assert fi.shape == (4, 64) and torch.allclose(fi.norm(dim=1), torch.ones(4), atol=1e-4)
    assert ft.shape == (2, 64) and torch.allclose(ft.norm(dim=1), torch.ones(2), atol=1e-4)
    # X-VLM: must apply ITC projection (has .itc)
    mx = build_xvlm(_xvlm_cfg())
    assert hasattr(mx, "itc")
    fix = encode_retrieval_images(mx, paths, 32, "cpu", 2)
    assert fix.shape == (4, 64) and torch.allclose(fix.norm(dim=1), torch.ones(4), atol=1e-4)


def test_make_distractor_eval_fn(tmp_path):
    paths = _imgs(tmp_path, 5)
    index = {"gt": paths[:2], "distractors": paths[2:]}
    fn = make_distractor_eval_fn(index, ["a b", "c d"], DummyTokenizer(16), 32, "cpu")
    m = fn(build_siglip(_siglip_cfg()))
    assert "mAP" in m and 0.0 <= m["R@1"] <= 1.0


def test_best_state_snapshot_on_eval(tmp_path):
    cfg = _siglip_cfg()
    mani, root = make_dataset(tmp_path, 8)
    ds = PABDatasetV4(mani, root, split="train", image_size=32, train=True, max_token=16, cfg=cfg)
    loader = DataLoader(ds, batch_sampler=BalancedBucketSampler(ds.buckets(), 4, seed=1),
                        collate_fn=collate_fn, num_workers=0)
    trainer = SiglipTrainer(build_siglip(cfg), cfg, "cpu")
    scores = iter([0.2, 0.9, 0.5, 0.5])  # 4 evals; best at the 2nd
    best = trainer.fit(loader, epochs=2, eval_fn=lambda model: {"mAP": next(scores)}, eval_every=1)
    assert best["mAP"] == 0.9
    assert trainer.best_state is not None and all(torch.is_tensor(v) for v in trainer.best_state.values())


def test_real_siglip_smoke():
    """Loads a tiny REAL SigLIP (proves the HF path + aligned space); skips if offline."""
    try:
        cfg = Cfg({"model": {
            "name": "hf-internal-testing/tiny-random-SiglipModel", "embed_dim": 32,
            "lora": {"enabled": True, "r": 4, "alpha": 8,
                     "target_modules": ["q_proj", "k_proj", "v_proj", "out_proj"]},
        }})
        m = build_siglip(cfg)
    except Exception as e:  # offline / hub error
        pytest.skip(f"HF model unavailable: {e}")
    out = m(torch.randn(2, 3, 30, 30), torch.randint(0, 64, (2, 8)), torch.ones(2, 8, dtype=torch.long))
    assert out["image_feat"].shape[0] == 2
    assert torch.allclose(out["image_feat"].norm(dim=1), torch.ones(2), atol=1e-3)
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    total = sum(p.numel() for p in m.parameters())
    assert 0 < trainable < total  # LoRA actually attached
