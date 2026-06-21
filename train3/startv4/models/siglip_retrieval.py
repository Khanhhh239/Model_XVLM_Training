"""SigLIP-2-L retrieval tower (Model A).

Real path: HF `Siglip2Model` + PEFT-LoRA on attention, low LR -> keep the billions-of-real-
images pretrain (the core sim2real lever).  Dummy path: a tiny conv/embedding model so the
whole pipeline (dataset -> loss -> trainer -> infer) is unit-testable on CPU with no download.

Real-path notes (review fixes A1):
  * pooled features come from `vision_model().pooler_output` / `text_model().pooler_output`,
    which ARE SigLIP's aligned contrastive embeddings (SigLIP has no separate projection head);
  * the model's PRETRAINED `logit_scale`/`logit_bias` are reused (PEFT freezes them, so we
    re-enable grad) instead of fresh, mis-calibrated parameters.
forward() returns image_feat/text_feat (L2-normalised), token features (for optional FILIP),
and logit_scale (>0) / logit_bias used by the sigmoid loss.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _DummyImageEncoder(nn.Module):
    def __init__(self, dim: int, patch: int = 16):
        super().__init__()
        self.proj = nn.Conv2d(3, dim, kernel_size=patch, stride=patch)

    def forward(self, pixel_values: torch.Tensor):
        x = self.proj(pixel_values)                      # [B, D, H', W']
        tokens = x.flatten(2).transpose(1, 2)            # [B, N, D]
        return tokens.mean(dim=1), tokens                # pooled, tokens


class _DummyTextEncoder(nn.Module):
    def __init__(self, dim: int, vocab: int = 1000):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim, padding_idx=0)

    def forward(self, input_ids: torch.Tensor, attn: torch.Tensor):
        tokens = self.emb(input_ids)                     # [B, L, D]
        m = attn.unsqueeze(-1).float()
        pooled = (tokens * m).sum(1) / m.sum(1).clamp(min=1e-6)
        return pooled, tokens


class SiglipRetrieval(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        m = cfg.model
        self.embed_dim = int(m.get_path("embed_dim", 1024) if hasattr(m, "get_path") else m["embed_dim"])
        self.name = m.get_path("name", "dummy") if hasattr(m, "get_path") else m.get("name", "dummy")
        self.is_dummy = self.name == "dummy"

        if self.is_dummy:
            patch = int(m.get_path("patch", 16))
            vocab = int(m.get_path("vocab_size", 1000))
            self.image_encoder = _DummyImageEncoder(self.embed_dim, patch)
            self.text_encoder = _DummyTextEncoder(self.embed_dim, vocab)
            self.img_proj = nn.Linear(self.embed_dim, self.embed_dim)
            self.txt_proj = nn.Linear(self.embed_dim, self.embed_dim)
            # dummy-only learnable temperature/bias (init like the SigLIP paper)
            self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))
            self.logit_bias = nn.Parameter(torch.tensor(-10.0))
        else:
            self._build_hf(cfg)

    # ---- real HF + LoRA path (not exercised by CPU tests) ----
    def _build_hf(self, cfg):
        from transformers import AutoModel

        self.hf = AutoModel.from_pretrained(cfg.model.name)
        lora = cfg.model.get_path("lora", None) if hasattr(cfg.model, "get_path") else None
        if lora and bool(lora.get("enabled", True)):
            from peft import LoraConfig, get_peft_model

            targets = list(lora.get("target_modules", ["q_proj", "k_proj", "v_proj", "out_proj"]))
            self.hf = get_peft_model(
                self.hf,
                LoraConfig(
                    r=int(lora.get("r", 32)),
                    lora_alpha=int(lora.get("alpha", 64)),
                    lora_dropout=float(lora.get("dropout", 0.05)),
                    target_modules=targets,
                    bias="none",
                ),
            )
        # reuse the PRETRAINED temperature/bias; PEFT froze them, so re-enable grad
        base = self._base()
        for n in ("logit_scale", "logit_bias"):
            if hasattr(base, n) and isinstance(getattr(base, n), torch.nn.Parameter):
                getattr(base, n).requires_grad_(True)

    def _base(self):
        return self.hf.get_base_model() if hasattr(self.hf, "get_base_model") else self.hf

    def _scale_bias(self):
        if self.is_dummy:
            return self.logit_scale.exp(), self.logit_bias
        base = self._base()
        return base.logit_scale.exp(), base.logit_bias

    def encode_image(self, pixel_values: torch.Tensor, keypoints=None):
        # keypoints accepted for a uniform encoder signature; SigLIP ignores it (no pose branch).
        if self.is_dummy:
            pooled, tokens = self.image_encoder(pixel_values)
            return F.normalize(self.img_proj(pooled), dim=-1), F.normalize(tokens, dim=-1)
        out = self.hf.vision_model(pixel_values=pixel_values)
        return F.normalize(out.pooler_output, dim=-1), F.normalize(out.last_hidden_state, dim=-1)

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        if self.is_dummy:
            pooled, tokens = self.text_encoder(input_ids, attention_mask)
            return F.normalize(self.txt_proj(pooled), dim=-1), F.normalize(tokens, dim=-1)
        out = self.hf.text_model(input_ids=input_ids, attention_mask=attention_mask)
        return F.normalize(out.pooler_output, dim=-1), F.normalize(out.last_hidden_state, dim=-1)

    def forward(self, pixel_values, input_ids, attention_mask):
        img_feat, img_tok = self.encode_image(pixel_values)
        txt_feat, txt_tok = self.encode_text(input_ids, attention_mask)
        scale, bias = self._scale_bias()
        return {
            "image_feat": img_feat,
            "text_feat": txt_feat,
            "image_tokens": img_tok,
            "text_tokens": txt_tok,
            "logit_scale": scale,
            "logit_bias": bias,
        }


def build_siglip(cfg) -> SiglipRetrieval:
    return SiglipRetrieval(cfg)
