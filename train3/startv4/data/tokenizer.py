"""Tokenizer builder shared by training and encoding.

dummy backbone -> bundled DummyTokenizer (no download); real -> HF SigLIP processor.
"""
from __future__ import annotations

import torch


def build_tokenizer(cfg):
    max_len = int(cfg.data.get_path("max_token", 64))
    if cfg.model.name == "dummy":
        from .dataset import DummyTokenizer

        return DummyTokenizer(max_len=max_len)

    from transformers import AutoProcessor

    proc = AutoProcessor.from_pretrained(cfg.model.name)

    def tok(text: str):
        enc = proc(
            text=[str(text)], padding="max_length", max_length=max_len, truncation=True, return_tensors="pt"
        )
        ids = enc["input_ids"][0]
        am = enc.get("attention_mask", torch.ones_like(ids))[0]
        return {"input_ids": ids, "attention_mask": am}

    return tok
