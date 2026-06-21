from pathlib import Path

import torch

from startv4.config import load_config
from startv4.models.siglip_retrieval import build_siglip

ROOT = Path(__file__).resolve().parents[1]


def test_forward_shapes_normalised():
    cfg = load_config(ROOT / "configs" / "_test_dummy.yaml")
    m = build_siglip(cfg)
    px = torch.randn(3, 3, 32, 32)
    ids = torch.randint(1, 1000, (3, 16))
    am = torch.ones(3, 16, dtype=torch.long)
    out = m(px, ids, am)
    assert out["image_feat"].shape == (3, 64)
    assert out["text_feat"].shape == (3, 64)
    assert torch.allclose(out["image_feat"].norm(dim=1), torch.ones(3), atol=1e-4)
    assert out["image_tokens"].shape == (3, 4, 64)  # 32/16=2 -> 2x2=4 patches
    assert out["logit_scale"].item() > 0
