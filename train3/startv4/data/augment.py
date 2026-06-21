"""Sim2real augmentation.

Train images are clean synthetic; the test images are real video frames (blur, JPEG,
low-res, occlusion).  These augmentations force features that survive real-frame
degradations -- the main way STAR-v4 closes sim2real WITHOUT any real data (OOPS! banned).

CAUTION (encoded as defaults, see README): no large rotation (orientation defines
"falling"); flip only when the caption has no left/right; keep colour jitter moderate
(don't turn "red shirt" into orange).  Augmentation applies to TRAIN only.
"""
from __future__ import annotations

import io
import random

import torch
import torchvision.transforms as T
from PIL import Image

IMAGENET_MEAN = (0.5, 0.5, 0.5)  # SigLIP-style
IMAGENET_STD = (0.5, 0.5, 0.5)


class JpegCompression:
    def __init__(self, p: float = 0.4, q_range: tuple[int, int] = (30, 90)):
        self.p, self.q_range = p, q_range

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        q = random.randint(*self.q_range)
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=q)
        buf.seek(0)
        return Image.open(buf).convert("RGB")


class DownscaleUpscale:
    def __init__(self, p: float = 0.3, scale_range: tuple[float, float] = (0.5, 1.0)):
        self.p, self.scale_range = p, scale_range

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        w, h = img.size
        s = random.uniform(*self.scale_range)
        small = img.resize((max(1, int(w * s)), max(1, int(h * s))), Image.BILINEAR)
        return small.resize((w, h), Image.BILINEAR)


class MotionBlurTensor:
    """Horizontal or vertical box blur via depthwise conv (approximate motion blur)."""

    def __init__(self, p: float = 0.3, ksizes: tuple[int, ...] = (3, 5, 7, 9)):
        self.p, self.ksizes = p, ksizes

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return img
        import torch.nn.functional as F

        k = random.choice(self.ksizes)
        c = img.size(0)
        if random.random() < 0.5:  # horizontal
            kernel = torch.ones(c, 1, 1, k, device=img.device) / k
            pad = (k // 2, k // 2, 0, 0)
        else:  # vertical
            kernel = torch.ones(c, 1, k, 1, device=img.device) / k
            pad = (0, 0, k // 2, k // 2)
        x = F.pad(img.unsqueeze(0), pad, mode="reflect")
        return F.conv2d(x, kernel, groups=c).squeeze(0)


class GaussianNoiseTensor:
    def __init__(self, p: float = 0.2, std: float = 0.03):
        self.p, self.std = p, std

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return img
        return (img + torch.randn_like(img) * self.std).clamp(0.0, 1.0)


def build_train_transform(image_size: int = 512, cfg=None) -> T.Compose:
    """PIL -> normalised tensor [3, S, S] with sim2real + robustness augmentation."""
    g = (lambda k, d: cfg.get_path(k, d)) if cfg is not None and hasattr(cfg, "get_path") else (lambda k, d: d)
    return T.Compose(
        [
            T.RandomResizedCrop(image_size, scale=(g("aug.crop_min", 0.8), 1.0)),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
            JpegCompression(p=g("aug.jpeg_p", 0.4)),
            DownscaleUpscale(p=g("aug.downscale_p", 0.3)),
            T.ToTensor(),
            MotionBlurTensor(p=g("aug.blur_p", 0.3)),
            GaussianNoiseTensor(p=g("aug.noise_p", 0.2)),
            T.RandomErasing(p=g("aug.erase_p", 0.25), scale=(0.02, 0.2)),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_eval_transform(image_size: int = 512) -> T.Compose:
    """Deterministic resize for eval/inference (no augmentation)."""
    return T.Compose(
        [
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
