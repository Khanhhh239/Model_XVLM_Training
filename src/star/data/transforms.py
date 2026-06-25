"""LHP — Local-global Hybrid augmentation (train only) + plain eval transform.

analyze.md §10. Per image: p ~ N(0.5, 1/6).
  p > 0.5 -> LOCAL  : person-bbox RandomResizedCrop (scale >= 0.5), resize to S
  else    -> GLOBAL : full image resize to S
Inference uses GLOBAL only (build_eval_transform) so no detail is lost at scoring time.

Safety (answers the "won't local crop lose detail?" concern):
  - stochastic: GLOBAL is also seen across epochs
  - scale >= 0.5 (never an aggressive crop)
  - crop centered on the person bbox (keeps the subject), fallback to center crop
"""
from __future__ import annotations

import random
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF

import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# Albumentations renamed several kwargs across versions. A version check (1.4.x warns-and-ignores
# unknown kwargs instead of raising, so try/except is unreliable) keeps the same code running on
# whatever Kaggle ships. Stage-1 pins albumentations<2 (matches what trained best (3).pth).
_ALBU_MAJOR = int(A.__version__.split(".")[0])


def _aug_image_compression(p):
    if _ALBU_MAJOR >= 2:
        return A.ImageCompression(quality_range=(25, 85), p=p)
    return A.ImageCompression(quality_lower=25, quality_upper=85, p=p)


def _aug_downscale(p):
    if _ALBU_MAJOR >= 2:
        return A.Downscale(scale_range=(0.45, 0.85), p=p)
    return A.Downscale(scale_min=0.45, scale_max=0.85, p=p)


def _aug_gauss_noise(p):
    if _ALBU_MAJOR >= 2:
        return A.GaussNoise(std_range=(0.04, 0.2), p=p)            # 2.x: normalized std
    return A.GaussNoise(var_limit=(10.0, 50.0), p=p)              # 1.x: variance


def _aug_coarse_dropout(p):
    if _ALBU_MAJOR >= 2:
        return A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(16, 48),
                               hole_width_range=(16, 48), p=p)
    return A.CoarseDropout(max_holes=4, max_height=48, max_width=48,
                           min_holes=1, min_height=16, min_width=16, p=p)


def _aug_rrc(size, min_scale):                            # RandomResizedCrop: 2.x uses size=(h,w)
    if _ALBU_MAJOR >= 2:
        return A.RandomResizedCrop(size=(size, size), scale=(min_scale, 1.0))
    return A.RandomResizedCrop(height=size, width=size, scale=(min_scale, 1.0))


def _normal_p(mean: float = 0.5, var: float = 1.0 / 6.0) -> float:
    return random.gauss(mean, var ** 0.5)


class LHPTransform:
    """Callable transform using Albumentations. Pass the absolute bbox (xyxy) and keypoints."""

    def __init__(self, size: int = 384, min_scale: float = 0.5, use_bbox: bool = True, enabled: bool = True):
        self.size = size
        self.min_scale = min_scale
        self.use_bbox = use_bbox
        self.enabled = enabled

    def _get_local_crop(self, W, H, bbox, min_scale):
        # bbox is pascal_voc: [xmin, ymin, xmax, ymax]
        xmin, ymin, xmax, ymax = bbox
        w = xmax - xmin
        h = ymax - ymin
        cx, cy = xmin + w / 2, ymin + h / 2
        
        scale = random.uniform(min_scale, 1.0)
        side = max(w, h, scale * min(W, H))
        half = side / 2
        left = int(max(0, min(cx - half, W - side)))
        top = int(max(0, min(cy - half, H - side)))
        side = int(min(side, W - left, H - top))
        return [left, top, left + side, top + side]

    def __call__(self, img: Image.Image, bbox=None, keypoints=None, extra_boxes=None):
        img_np = np.array(img.convert("RGB"))
        H, W, _ = img_np.shape
        
        if bbox is None:
            bbox = [0.0, 0.0, float(W), float(H)]
        else:
            # Ensure bbox is valid
            bbox = [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
            bbox = [max(0.0, bbox[0]), max(0.0, bbox[1]), min(float(W), bbox[2]), min(float(H), bbox[3])]
            if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
                bbox = [0.0, 0.0, float(W), float(H)]

        kpts_xy = []
        kpts_c = []
        if keypoints is not None:
            if isinstance(keypoints, torch.Tensor):
                keypoints = keypoints.tolist()
            if len(keypoints) == 51:
                for i in range(0, 51, 3):
                    x, y, c = keypoints[i], keypoints[i + 1], keypoints[i + 2]
                    if x <= 1.0 and y <= 1.0:
                        kpts_xy.append((x * W, y * H))
                    else:
                        kpts_xy.append((x, y))
                    kpts_c.append(c)
            else:
                for kp in keypoints:
                    kpts_xy.append((kp[0], kp[1]))
                    kpts_c.append(kp[2])
        else:
            kpts_xy = [(W / 2, H / 2)] * 17
            kpts_c = [0.0] * 17

        is_local = self.enabled and _normal_p() > 0.5
        
        augs = []
        if is_local and self.use_bbox and bbox != [0.0, 0.0, float(W), float(H)]:
            crop_coords = self._get_local_crop(W, H, bbox, self.min_scale)
            augs.append(A.Crop(x_min=int(crop_coords[0]), y_min=int(crop_coords[1]), x_max=int(crop_coords[2]), y_max=int(crop_coords[3])))
        elif is_local:
            augs.append(_aug_rrc(self.size, self.min_scale))
            
        augs.extend([
            # Sim2Real: match blurry/noisy real gallery frames (group E #5/#4 sharpness, F low-light)
            A.MotionBlur(blur_limit=11, p=0.45),
            A.GaussianBlur(blur_limit=(3, 7), p=0.25),
            _aug_image_compression(0.4),
            _aug_downscale(0.35),
            A.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.3, hue=0.05, p=0.6),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.35),
            _aug_gauss_noise(0.35),
            _aug_coarse_dropout(0.3),
            A.Resize(self.size, self.size),
            A.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ToTensorV2(),
        ])
        
        # box list: person (label 0) + optional phrase boxes (labels 1..P), all clamped to the image.
        # Phrase boxes go through the SAME crop/resize as the image so their targets stay aligned;
        # boxes cropped fully out are dropped by albumentations -> marked invalid via the label map.
        def _clamp(b):
            return [max(0.0, float(b[0])), max(0.0, float(b[1])),
                    min(float(W), float(b[2])), min(float(H), float(b[3]))]
        box_list, box_labels = [bbox], [0]
        if extra_boxes:
            for k, eb in enumerate(extra_boxes):
                cb = _clamp(eb)
                if cb[0] < cb[2] and cb[1] < cb[3]:
                    box_list.append(cb); box_labels.append(k + 1)

        transform = A.Compose(augs,
                              bbox_params=A.BboxParams(format='pascal_voc', label_fields=['bbox_classes'],
                                                       min_visibility=0.0),
                              keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
        try:
            transformed = transform(image=img_np, bboxes=box_list, bbox_classes=box_labels, keypoints=kpts_xy)
        except Exception:
            transformed = transform(image=img_np, bboxes=[[0.0, 0.0, float(W), float(H)]],
                                    bbox_classes=[0], keypoints=kpts_xy)

        out_img = transformed['image']
        by_label = {int(l): b for b, l in zip(transformed['bboxes'], transformed['bbox_classes'])}
        out_bbox = by_label.get(0, [0, 0, 0, 0])                       # person box (pixel xyxy @ size)
        out_kpts_xy = transformed['keypoints']

        out_kpts = []
        for (x, y), c in zip(out_kpts_xy, kpts_c):
            if x < 0 or x >= self.size or y < 0 or y >= self.size:
                c = 0.0
            out_kpts.extend([x / self.size, y / self.size, c])

        person_t = torch.tensor(out_bbox, dtype=torch.float)
        kpts_t = torch.tensor(out_kpts, dtype=torch.float)
        if extra_boxes is None:
            return out_img, person_t, kpts_t
        # phrase boxes -> normalized center xywh + validity (dropped-out boxes => mask 0)
        P = len(extra_boxes)
        ph_box, ph_valid = torch.zeros(P, 4), torch.zeros(P)
        for k in range(P):
            b = by_label.get(k + 1)
            if b is not None and b[2] > b[0] and b[3] > b[1]:
                x1, y1, x2, y2 = b
                ph_box[k] = torch.tensor([((x1 + x2) / 2) / self.size, ((y1 + y2) / 2) / self.size,
                                          (x2 - x1) / self.size, (y2 - y1) / self.size])
                ph_valid[k] = 1.0
        return out_img, person_t, kpts_t, ph_box, ph_valid


def build_eval_transform(size: int = 384):
    """Deterministic transform used at validation / inference (GLOBAL full image)."""
    return T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
