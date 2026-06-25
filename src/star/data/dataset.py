"""PAB dataset + collate.

Reads the parquet manifest delivered by the DATA TEAM (see README.md "Data contract").
Each item yields: image (tensor), tokenized caption, instance id (for hard-neg/dup masking),
optional bbox (LHP), keypoints (pose), anomaly_label, and bbox for new heads.

STAGE 1 UPDATE: Added support for:
  - bbox: [x, y, w, h] normalized to [0, 1] for box grounding head (center COCO format)
  - label_type / bucket: normal vs anomaly for anomaly classification head
  - Reads from built manifest + train_30k_hard_vitpose.json + boxes_30k.jsonl
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from .transforms import LHPTransform, build_eval_transform


def _parse_list(v, expected_len):
    """Parse a manifest cell into a float list of `expected_len`, else None."""
    if v is None or isinstance(v, float):   # NaN / null
        return None
    if isinstance(v, str):
        try:
            v = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            return None
    try:
        if v is None or len(v) != expected_len:
            return None
        return [float(x) for x in v]
    except TypeError:
        return None


def _parse_bbox(v):
    return _parse_list(v, 4)


def _parse_kpts(v):
    return _parse_list(v, 17 * 3)


def _resolve_image_path(row) -> str:
    """Return path relative to image_root from common manifest columns."""
    for key in ("image_path", "image_webp", "image"):
        raw = row.get(key)
        if raw is None or (isinstance(raw, float) and pd.isna(raw)):
            continue
        p = str(raw).replace("\\", "/")
        if p.startswith("train/"):
            p = p[len("train/"):]
        if "train_webp/" in p:
            p = p.split("train_webp/", 1)[-1]
        if p.startswith("data/"):
            p = p[len("data/"):]
            if p.startswith("train_webp/"):
                p = p[len("train_webp/"):]
        return p
    raise KeyError("manifest row missing image_path / image_webp / image")


def _load_vitpose_dict(path: str) -> dict:
    """Load vitpose JSON; supports flat {path: kpts} or nested {items: {id: {...}}}."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "items" in data:
        return data["items"]
    return data if isinstance(data, dict) else {}


def _kpts_from_vitpose(vitpose_dict: dict, image_id: str, image_path: str) -> list[float] | None:
    """Extract flattened 51-float keypoints from vitpose item dict."""
    item = vitpose_dict.get(str(image_id))
    if item is None:
        item = vitpose_dict.get(image_path)
    if not isinstance(item, dict) or item.get("status") != "ok":
        return None
    insts = item.get("instances") or []
    if not insts:
        return None
    W = float(item.get("width", 384) or 384)
    H = float(item.get("height", 384) or 384)
    flat: list[float] = []
    for x, y, c in insts[0].get("keypoints_xyc", []):
        flat.extend([float(x) / W, float(y) / H, float(c)])
    return flat if len(flat) == 51 else None


def _bbox_from_vitpose(vitpose_dict: dict, image_id: str, image_path: str) -> list[float] | None:
    """Return COCO xywh normalized [x_center, y_center, w, h] from vitpose primary bbox."""
    item = vitpose_dict.get(str(image_id))
    if item is None:
        item = vitpose_dict.get(image_path)
    if not isinstance(item, dict):
        return None
    b = item.get("primary_bbox_norm_xyxy")
    if not b or len(b) != 4:
        return None
    x1, y1, x2, y2 = [float(v) for v in b]
    w, h = max(x2 - x1, 1e-4), max(y2 - y1, 1e-4)
    return [(x1 + x2) / 2, (y1 + y2) / 2, w, h]


def _xyxy_to_coco_xywh_norm(x1, y1, x2, y2, w_ref: float, h_ref: float) -> list[float]:
    """Convert pixel VOC xyxy to normalized center COCO xywh."""
    bw = max(x2 - x1, 1e-4) / w_ref
    bh = max(y2 - y1, 1e-4) / h_ref
    cx = (x1 + x2) / 2 / w_ref
    cy = (y1 + y2) / 2 / h_ref
    return [cx, cy, bw, bh]


def _parse_anomaly_label(row) -> tuple[int, bool]:
    """Return (label, mask). Matches train_30k_hard.jsonl bucket/label_type conventions."""
    bucket = str(row.get("bucket", "")).lower()
    label_type = str(row.get("label_type", "")).lower()
    if bucket in ("wentwrong", "full") or label_type in ("wentwrong", "ca", "anomaly"):
        return 1, True
    if bucket == "goal" or label_type in ("normal", "goal", "cn"):
        return 0, True
    return 0, False


class PABDataset(Dataset):
    def __init__(
        self,
        manifest: str,
        image_root: str,
        tokenizer,
        split: str = "train",
        image_size: int = 384,
        max_token: int = 100,
        train: bool = True,
        lhp_kwargs: dict | None = None,
        vitpose_json: str | None = None,
        boxes_json: str | None = None,
    ):
        manifest_path = Path(manifest)
        if manifest_path.suffix == ".parquet":
            df = pd.read_parquet(manifest)
        elif manifest_path.suffix == ".jsonl":
            df = pd.read_json(manifest, lines=True)
        else:
            raise ValueError(f"Unsupported manifest format: {manifest_path.suffix} (expected .parquet or .jsonl)")

        self.df = df[df["split"] == split].reset_index(drop=True) if "split" in df.columns else df
        self.image_root = Path(image_root)
        self.tokenizer = tokenizer
        self.max_token = max_token
        self.image_size = image_size
        self.train = train

        if train:
            self.transform = LHPTransform(size=image_size, **(lhp_kwargs or {}))
        else:
            self.transform = build_eval_transform(image_size)

        self.inst_ids = self.df.get("sequence_id", pd.Series(range(len(self.df)))).astype("category").cat.codes.values

        self.vitpose_dict = _load_vitpose_dict(vitpose_json) if vitpose_json and Path(vitpose_json).exists() else {}

        self.boxes_dict = {}
        if boxes_json and Path(boxes_json).exists():
            with open(boxes_json, "r", encoding="utf-8") as f:
                for line in f:
                    item = json.loads(line.strip())
                    key = item.get("image_id") or item.get("image_path")
                    if key and item.get("bbox"):
                        self.boxes_dict[str(key)] = item["bbox"]

    def __len__(self) -> int:
        return len(self.df)

    def group_ids(self, key: str = "scene") -> list:
        if key in ("none", None) or key not in self.df.columns:
            return [None] * len(self.df)
        return self.df[key].tolist()

    def pairs(self) -> tuple[list, list]:
        if "pair_image_id" not in self.df.columns or "image_id" not in self.df.columns:
            return [], []
        pos = {str(iid): i for i, iid in enumerate(self.df["image_id"])}
        group_col = ("video_id" if "video_id" in self.df.columns
                     else "scene" if "scene" in self.df.columns else None)
        pairs, groups = [], []
        for i, pid in enumerate(self.df["pair_image_id"]):
            if pid is None or (isinstance(pid, float) and pd.isna(pid)):
                continue
            j = pos.get(str(pid))
            if j is None or j == i:
                continue
            pairs.append((i, j))
            groups.append(self.df[group_col].iat[i] if group_col else i)
        return pairs, groups

    def _load_image(self, rel_path: str) -> Image.Image:
        p = Path(rel_path)
        if not p.is_absolute():
            p = self.image_root / rel_path
        return Image.open(p).convert("RGB")

    def __getitem__(self, i: int) -> dict:
        row = self.df.iloc[i]
        image_path = _resolve_image_path(row)
        image_id = str(row.get("image_id", image_path))
        img = self._load_image(image_path)
        img_w, img_h = img.size

        # Bbox: manifest COCO xywh, vitpose, or external boxes_jsonl
        bbox_coco = _parse_bbox(row.get("bbox"))
        if bbox_coco is None:
            bbox_coco = _bbox_from_vitpose(self.vitpose_dict, image_id, image_path)
        if bbox_coco is None and image_id in self.boxes_dict:
            bbox_coco = self.boxes_dict[image_id]
        if bbox_coco is None and image_path in self.boxes_dict:
            bbox_coco = self.boxes_dict[image_path]

        bbox_voc = None
        if bbox_coco is not None:
            # Manifest stores COCO xywh norm; LHPTransform expects pixel VOC xyxy
            if all(0 <= v <= 1 for v in bbox_coco):
                x, y, bw, bh = bbox_coco
                bbox_voc = [x * img_w - (bw * img_w) / 2, y * img_h - (bh * img_h) / 2,
                            x * img_w + (bw * img_w) / 2, y * img_h + (bh * img_h) / 2]
            else:
                x, y, bw, bh = bbox_coco
                bbox_voc = [x, y, x + bw, y + bh]

        kpts_raw = _parse_kpts(row.get("keypoints")) if "keypoints" in self.df.columns else None
        if kpts_raw is None:
            kpts_raw = _kpts_from_vitpose(self.vitpose_dict, image_id, image_path)

        if self.train and isinstance(self.transform, LHPTransform):
            image, bbox_aug, kpts_aug = self.transform(img, bbox_voc, kpts_raw)
            bbox_norm = None
            if bbox_aug is not None and bbox_aug.numel() == 4:
                x1, y1, x2, y2 = bbox_aug.tolist()
                bbox_norm = _xyxy_to_coco_xywh_norm(x1, y1, x2, y2, self.image_size, self.image_size)
            kpts = kpts_aug.tolist() if kpts_aug is not None else kpts_raw
        else:
            image = self.transform(img)
            bbox_norm = None
            if bbox_voc is not None:
                bbox_norm = _xyxy_to_coco_xywh_norm(*bbox_voc, img_w, img_h)
            kpts = kpts_raw

        caption = str(row.get("caption", ""))
        tok = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.max_token,
            return_tensors="pt",
        )
        anom_label, anom_mask = _parse_anomaly_label(row)

        # action phrase (manifest `action` column, e.g. "playing ping pong") for the action-keyword
        # loss. Hard-partner rows carry the placeholder "hard_pair" -> marked invalid (masked out).
        action_str = str(row.get("action", "") or "").strip()
        action_valid = action_str.lower() not in ("", "unk", "none", "nan", "hard_pair")
        act_tok = self.tokenizer(
            action_str if action_valid else "unknown",
            padding="max_length", truncation=True,
            max_length=min(self.max_token, 32), return_tensors="pt",
        )

        item = {
            "image": image,
            "input_ids": tok["input_ids"].squeeze(0),
            "attention_mask": tok["attention_mask"].squeeze(0),
            "instance_id": int(self.inst_ids[i]),
            "image_id": image_id,
            "is_query": bool(caption.strip()),
            "bbox": torch.tensor(bbox_norm if bbox_norm else [0.0, 0.0, 0.0, 0.0], dtype=torch.float),
            "bbox_mask": bbox_norm is not None,
            "anomaly_label": anom_label,
            "anomaly_mask": anom_mask,
            "action_input_ids": act_tok["input_ids"].squeeze(0),
            "action_attention_mask": act_tok["attention_mask"].squeeze(0),
            "action_str": action_str if action_valid else "",
            "action_valid": action_valid,
        }

        if kpts is not None and len(kpts) == 51:
            item["keypoints"] = torch.tensor(kpts, dtype=torch.float)

        return item


def _action_groups(batch: list[dict]) -> list[int]:
    """Per-row group id for the action loss: rows with the SAME valid action phrase share an id;
    invalid/empty actions get a unique NEGATIVE id (so they never group, and are masked out anyway)."""
    gmap: dict[str, int] = {}
    groups: list[int] = []
    nxt, neg = 0, -1
    for b in batch:
        s, valid = b.get("action_str", ""), b.get("action_valid", False)
        if valid and s:
            if s not in gmap:
                gmap[s] = nxt
                nxt += 1
            groups.append(gmap[s])
        else:
            groups.append(neg)
            neg -= 1
    return groups


def collate_fn(batch: list[dict]) -> dict:
    out = {
        "image": torch.stack([b["image"] for b in batch]),
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "instance_id": torch.tensor([b["instance_id"] for b in batch], dtype=torch.long),
        "bbox": torch.stack([b["bbox"] for b in batch]),
        "bbox_mask": torch.tensor([b["bbox_mask"] for b in batch], dtype=torch.bool),
        "anomaly_label": torch.tensor([b["anomaly_label"] for b in batch], dtype=torch.long),
        "anomaly_mask": torch.tensor([b["anomaly_mask"] for b in batch], dtype=torch.bool),
    }

    if all("keypoints" in b for b in batch):
        out["keypoints"] = torch.stack([b["keypoints"] for b in batch])

    if all("action_input_ids" in b for b in batch):
        out["action_input_ids"] = torch.stack([b["action_input_ids"] for b in batch])
        out["action_attention_mask"] = torch.stack([b["action_attention_mask"] for b in batch])
        out["action_group"] = torch.tensor(_action_groups(batch), dtype=torch.long)
        out["action_valid"] = torch.tensor([b["action_valid"] for b in batch], dtype=torch.bool)

    return out
