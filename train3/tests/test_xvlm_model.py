from pathlib import Path

import torch

from startv4.config import load_config
from startv4.models.xvlm_v4 import build_xvlm

ROOT = Path(__file__).resolve().parents[1]


def _cfg():
    return load_config(ROOT / "configs" / "_test_xvlm_dummy.yaml")


def _batch(b=6):
    return (
        torch.randn(b, 3, 32, 32),
        torch.randint(1, 1000, (b, 16)),
        torch.ones(b, 16, dtype=torch.long),
        torch.rand(b, 17, 3),
    )


def test_encoders_and_itc_normalised():
    m = build_xvlm(_cfg())
    px, ids, am, kp = _batch()
    img_pooled, img_tok = m.encode_image(px, kp)
    txt_pooled, txt_tok, txt_mask = m.encode_text(ids, am)
    assert img_pooled.shape == (6, 64)
    assert txt_tok.shape[0] == 6 and txt_tok.shape[2] == 64
    ii, tt = m.itc(img_pooled, txt_pooled)
    assert torch.allclose(ii.norm(dim=1), torch.ones(6), atol=1e-4)


def test_cross_and_itm_logits():
    m = build_xvlm(_cfg())
    px, ids, am, kp = _batch()
    _, img_tok = m.encode_image(px, kp)
    _, txt_tok, txt_mask = m.encode_text(ids, am)
    fused = m.cross_cls(img_tok, txt_tok, txt_mask)
    assert fused.shape == (6, 64)
    logits = m.itm_logits(img_tok, txt_tok, txt_mask)
    assert logits.shape == (6, 2) and torch.isfinite(logits).all()


def test_aux_heads_present():
    m = build_xvlm(_cfg())
    assert m.use_pose and m.use_anom and m.use_box and m.use_filip
    fused = torch.randn(6, 64)
    assert m.box(fused).shape == (6, 4)
    assert m.anom(fused).shape == (6, 2)
