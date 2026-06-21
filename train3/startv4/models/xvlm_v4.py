"""X-VLM-v4 cross-encoder (Model B / Phase 3).

A self-contained, testable cross-encoder: image encoder + text encoder + a cross-attention
fusion stack + ITM head, plus the v4 aux heads (bbox / anomaly-bucket / pose-region-fuse /
FILIP).  Dummy backbone for CPU tests; HF Swin+BERT hook for the real run.

NOTE: for the actual submission you may instead port these heads/losses onto the pretrained
X-VLM in .. (warm-start best.pth) to reuse its pretrained ITM cross-encoder.  This
module is the reusable, tested Phase-3 machinery (heads, cross-attention, ITM, rerank).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .heads import AnomalyHead, BBoxHead, FilipProjection, PoseRegionFuse


class _DummyVision(nn.Module):
    def __init__(self, dim: int, patch: int = 16):
        super().__init__()
        self.proj = nn.Conv2d(3, dim, kernel_size=patch, stride=patch)

    def forward(self, px: torch.Tensor):
        tok = self.proj(px).flatten(2).transpose(1, 2)  # [B, N, D]
        return tok.mean(1), tok


class _DummyText(nn.Module):
    def __init__(self, dim: int, vocab: int = 1000):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim, padding_idx=0)
        self.cls = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

    def forward(self, ids: torch.Tensor, am: torch.Tensor):
        b = ids.size(0)
        tok = torch.cat([self.cls.expand(b, -1, -1), self.emb(ids)], dim=1)  # prepend CLS
        am = torch.cat([torch.ones(b, 1, device=am.device, dtype=am.dtype), am], dim=1)
        m = am.unsqueeze(-1).float()
        pooled = (tok * m).sum(1) / m.sum(1).clamp(min=1e-6)
        return pooled, tok, am


class CrossEncoder(nn.Module):
    """Text tokens attend to image tokens (multimodal fusion).  Position 0 = fused [CLS]."""

    def __init__(self, dim: int, layers: int = 2, heads: int = 4, dropout: float = 0.0):
        super().__init__()
        layer = nn.TransformerDecoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim * 4, dropout=dropout, batch_first=True
        )
        self.dec = nn.TransformerDecoder(layer, num_layers=layers)

    def forward(self, text_tokens, image_tokens, text_mask=None):
        tpad = (text_mask == 0) if text_mask is not None else None
        return self.dec(tgt=text_tokens, memory=image_tokens, tgt_key_padding_mask=tpad)


class XVLMv4(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        m = cfg.model
        gp = m.get_path
        self.embed_dim = int(gp("embed_dim", 256))
        self.name = gp("name", "dummy")
        self.is_dummy = self.name == "dummy"
        d = self.embed_dim

        if self.is_dummy:
            self.vision = _DummyVision(d, int(gp("patch", 16)))
            self.text = _DummyText(d, int(gp("vocab_size", 1000)))
            self.img_in = nn.Identity()
            self.txt_in = nn.Identity()
        else:
            self._build_hf(cfg, d)

        self.itc_img = nn.Linear(d, d)
        self.itc_txt = nn.Linear(d, d)
        self.cross = CrossEncoder(d, int(gp("cross_layers", 2)), int(gp("cross_heads", 4)))
        self.itm_head = nn.Linear(d, 2)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))

        self.use_pose = bool(gp("use_pose", False))
        self.use_anom = bool(gp("use_anomaly", False))
        self.use_box = bool(gp("use_box", False))
        self.use_filip = bool(gp("use_filip", False))
        if self.use_pose:
            self.pose = PoseRegionFuse(d)
        if self.use_anom:
            self.anom = AnomalyHead(d, int(gp("num_anomaly_classes", 3)))
        if self.use_box:
            self.box = BBoxHead(d)
        if self.use_filip:
            self.filip = FilipProjection(d, d, int(gp("filip_dim", 256)))

    def _build_hf(self, cfg, d):
        from transformers import AutoModel

        self.hf_vision = AutoModel.from_pretrained(cfg.model.image_model)
        self.hf_text = AutoModel.from_pretrained(cfg.model.text_model)
        vis_dim = getattr(self.hf_vision.config, "hidden_size", d)
        txt_dim = getattr(self.hf_text.config, "hidden_size", d)
        self.img_in = nn.Linear(vis_dim, d)
        self.txt_in = nn.Linear(txt_dim, d)

    # ---- encoders ----
    def encode_image(self, pixel_values, keypoints=None):
        if self.is_dummy:
            pooled, tok = self.vision(pixel_values)
        else:
            tok = self.img_in(self.hf_vision(pixel_values=pixel_values).last_hidden_state)
            pooled = tok.mean(1)
        if self.use_pose and keypoints is not None:
            tok = self.pose(tok, keypoints)
            pooled = tok.mean(1)
        return pooled, tok

    def encode_text(self, input_ids, attention_mask):
        if self.is_dummy:
            return self.text(input_ids, attention_mask)
        tok = self.txt_in(self.hf_text(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state)
        m = attention_mask.unsqueeze(-1).float()
        pooled = (tok * m).sum(1) / m.sum(1).clamp(min=1e-6)
        return pooled, tok, attention_mask

    # ---- heads ----
    def itc(self, img_pooled, txt_pooled):
        return (
            F.normalize(self.itc_img(img_pooled), dim=-1),
            F.normalize(self.itc_txt(txt_pooled), dim=-1),
        )

    def cross_cls(self, image_tokens, text_tokens, text_mask=None):
        return self.cross(text_tokens, image_tokens, text_mask)[:, 0]  # fused [CLS]

    def itm_logits(self, image_tokens, text_tokens, text_mask=None):
        return self.itm_head(self.cross_cls(image_tokens, text_tokens, text_mask))


def build_xvlm(cfg) -> XVLMv4:
    return XVLMv4(cfg)
