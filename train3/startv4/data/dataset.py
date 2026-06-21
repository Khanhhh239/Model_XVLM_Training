"""PABDatasetV4 + collate.  Works with a HF tokenizer or the bundled DummyTokenizer
(so unit tests run with no downloads).
"""
from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from .augment import build_eval_transform, build_train_transform
from .manifest import bucket_to_int, load_manifest


class DummyTokenizer:
    """Deterministic whitespace/hash tokenizer for tests (no vocab download)."""

    def __init__(self, vocab_size: int = 1000, max_len: int = 32):
        self.vocab_size, self.max_len, self.pad_id = vocab_size, max_len, 0

    def __call__(self, text: str):
        toks = [abs(hash(w)) % (self.vocab_size - 1) + 1 for w in str(text).split()][: self.max_len]
        attn = [1] * len(toks)
        pad = self.max_len - len(toks)
        return {
            "input_ids": torch.tensor(toks + [self.pad_id] * pad, dtype=torch.long),
            "attention_mask": torch.tensor(attn + [0] * pad, dtype=torch.long),
        }


def _parse_kpts(v) -> torch.Tensor:
    if v is None or (isinstance(v, float)):
        return torch.zeros(17, 3)
    t = torch.tensor(v, dtype=torch.float32).reshape(-1, 3)
    if t.size(0) < 17:
        t = torch.cat([t, torch.zeros(17 - t.size(0), 3)], 0)
    return t[:17]


def _parse_bbox(v):
    if v is None or isinstance(v, float):
        return torch.zeros(4), False
    return torch.tensor(v, dtype=torch.float32)[:4], True


class PABDatasetV4(Dataset):
    def __init__(
        self,
        manifest,
        image_root: str,
        tokenizer=None,
        split: str | None = None,
        image_size: int = 512,
        train: bool = True,
        max_token: int = 32,
        cfg=None,
    ):
        df = load_manifest(manifest) if isinstance(manifest, (str, Path)) else manifest.copy()
        if split is not None and "split" in df.columns:
            df = df[df["split"] == split].reset_index(drop=True)
        self.df = df.reset_index(drop=True)
        self.image_root = Path(image_root)
        self.tokenizer = tokenizer or DummyTokenizer(max_len=max_token)
        self.transform = (
            build_train_transform(image_size, cfg) if train else build_eval_transform(image_size)
        )
        self.train = train

    def __len__(self) -> int:
        return len(self.df)

    def pairs(self):
        """Return [(anchor_idx, partner_idx)] from image_id/pair_image_id columns."""
        if "image_id" not in self.df.columns or "pair_image_id" not in self.df.columns:
            return []
        id2idx = {str(v): i for i, v in enumerate(self.df["image_id"])}
        out = []
        for i, p in enumerate(self.df["pair_image_id"]):
            j = id2idx.get(str(p))
            if j is not None and j != i:
                out.append((i, j))
        return out

    def buckets(self) -> list[int]:
        return [bucket_to_int(b) for b in self.df["bucket"]]

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        img = Image.open(self.image_root / row["image_path"]).convert("RGB")
        pixel = self.transform(img)
        tok = self.tokenizer(row["caption"])
        bbox, has_bbox = _parse_bbox(row.get("bbox") if hasattr(row, "get") else None)
        return {
            "pixel_values": pixel,
            "input_ids": tok["input_ids"],
            "attention_mask": tok["attention_mask"],
            "bucket": torch.tensor(bucket_to_int(row["bucket"]), dtype=torch.long),
            "bbox": bbox,
            "has_bbox": torch.tensor(has_bbox),
            "keypoints": _parse_kpts(row["keypoints"] if "keypoints" in row else None),
            "image_id": str(row.get("image_id", idx)) if hasattr(row, "get") else str(idx),
            "index": idx,
        }


def collate_fn(batch: list[dict]) -> dict:
    out = {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "bucket": torch.stack([b["bucket"] for b in batch]),
        "bbox": torch.stack([b["bbox"] for b in batch]),
        "has_bbox": torch.stack([b["has_bbox"] for b in batch]),
        "keypoints": torch.stack([b["keypoints"] for b in batch]),
        "image_id": [b["image_id"] for b in batch],
        "index": torch.tensor([b["index"] for b in batch], dtype=torch.long),
    }
    return out
