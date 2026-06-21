"""Encode gallery/query once, cache to disk.

The expensive step (encoding 36K gallery) is done ONCE per checkpoint and cached; then
fusion + re-ranking iterate for free on the cached embeddings (Kaggle T4 friendly).
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from ..data.augment import build_eval_transform


def average_features(feats: list[torch.Tensor]) -> torch.Tensor:
    """TTA fusion: L2-normalise each [N,D] feature set, average, re-normalise.
    Used to combine multi-scale (512 + 768) encodings of the same images/queries.
    """
    if not feats:
        raise ValueError("average_features needs at least one tensor")
    acc = torch.zeros_like(feats[0])
    for f in feats:
        acc = acc + F.normalize(f, dim=1)
    return F.normalize(acc, dim=1)


class _PathImageDataset(Dataset):
    def __init__(self, paths, image_size: int):
        self.paths = list(paths)
        self.t = build_eval_transform(image_size)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> dict:
        return {"pixel_values": self.t(Image.open(self.paths[i]).convert("RGB"))}


def _stack_pixels(batch):
    return {"pixel_values": torch.stack([b["pixel_values"] for b in batch])}


@torch.no_grad()
def encode_image_loader(model, loader, device: str = "cpu") -> torch.Tensor:
    """Run model.encode_image over a DataLoader -> [N, D] L2-normalised features (CPU).

    Passes keypoints through when the batch carries them so the pose branch is active at
    inference exactly as in training (review fix A4); models without a pose branch ignore it.
    """
    model.eval()
    feats = []
    for batch in loader:
        kp = batch.get("keypoints") if isinstance(batch, dict) else None
        kp = kp.to(device) if kp is not None else None
        feat, _ = model.encode_image(batch["pixel_values"].to(device), kp)
        feats.append(feat.float().cpu())
    return torch.cat(feats, dim=0)


@torch.no_grad()
def encode_image_paths(
    model, paths, image_size: int = 512, device: str = "cpu", batch_size: int = 64, num_workers: int = 0
) -> torch.Tensor:
    """Encode a list of image paths -> [N, D] (CPU)."""
    ds = _PathImageDataset(paths, image_size)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, collate_fn=_stack_pixels)
    return encode_image_loader(model, loader, device)


@torch.no_grad()
def encode_text_list(model, input_ids, attention_mask, device: str = "cpu", chunk: int = 256) -> torch.Tensor:
    """input_ids/attention_mask: [N, L] -> [N, D] L2-normalised (CPU)."""
    model.eval()
    feats = []
    for s in range(0, input_ids.size(0), chunk):
        feat, _ = model.encode_text(
            input_ids[s : s + chunk].to(device), attention_mask[s : s + chunk].to(device)
        )
        feats.append(feat.float().cpu())
    return torch.cat(feats, dim=0)


@torch.no_grad()
def encode_captions(model, captions, tokenizer, device: str = "cpu", chunk: int = 256) -> torch.Tensor:
    """Tokenise + encode a list of caption strings -> [N, D] (CPU)."""
    ids, masks = [], []
    for c in captions:
        t = tokenizer(c)
        ids.append(t["input_ids"])
        masks.append(t["attention_mask"])
    return encode_text_list(model, torch.stack(ids), torch.stack(masks), device, chunk)


@torch.no_grad()
def encode_retrieval_images(model, paths, image_size: int, device: str = "cpu", batch_size: int = 64) -> torch.Tensor:
    """Retrieval-space image features: SigLIP `encode_image` IS the aligned feature; X-VLM needs
    the extra ITC projection (`model.itc`).  Returns [N, D] L2-normalised (CPU)."""
    feat = encode_image_paths(model, paths, image_size, device, batch_size)
    if hasattr(model, "itc"):
        ii, _ = model.itc(feat.to(device), feat.to(device))
        feat = F.normalize(ii, dim=1).cpu()
    return feat


@torch.no_grad()
def encode_retrieval_text(model, captions, tokenizer, device: str = "cpu") -> torch.Tensor:
    """Retrieval-space text features (ITC projection for X-VLM).  Returns [N, D] (CPU)."""
    feat = encode_captions(model, captions, tokenizer, device)
    if hasattr(model, "itc"):
        _, tt = model.itc(feat.to(device), feat.to(device))
        feat = F.normalize(tt, dim=1).cpu()
    return feat


def make_distractor_eval_fn(index, captions, tokenizer, image_size, device="cpu", batch_size=64, max_distractors=0):
    """Build eval_fn(model) -> distractor-val metrics, for the trainer's eval hook (best-by-mAP).
    `index` = {gt:[paths], distractors:[paths]}; `captions` aligned to index['gt'].  `max_distractors`
    > 0 subsamples the distractors (deterministic) so per-eval re-encode stays cheap during training;
    run the FULL set once at the end."""
    import random

    from ..eval.distractor_val import evaluate_with_distractors

    gt_paths, dis_paths = list(index["gt"]), list(index["distractors"])
    if max_distractors and len(dis_paths) > max_distractors:
        dis_paths = random.Random(42).sample(dis_paths, max_distractors)

    def fn(model):
        model.eval()
        gtf = encode_retrieval_images(model, gt_paths, image_size, device, batch_size)
        disf = encode_retrieval_images(model, dis_paths, image_size, device, batch_size)
        qf = encode_retrieval_text(model, captions, tokenizer, device)
        return evaluate_with_distractors(qf, gtf, disf)

    return fn


def save_embeddings(path: str | Path, **tensors) -> None:
    """Save tensors (and plain metadata like id lists) to a .pt cache."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: (v.cpu() if torch.is_tensor(v) else v) for k, v in tensors.items()}, path)


def load_embeddings(path: str | Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)
