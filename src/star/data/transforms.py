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

    def __call__(self, img: Image.Image, bbox=None, keypoints=None):
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
                    kpts_xy.append((keypoints[i], keypoints[i+1]))
                    kpts_c.append(keypoints[i+2])
            else:
                for kp in keypoints:
                    kpts_xy.append((kp[0], kp[1]))
                    kpts_c.append(kp[2])
        else:
            kpts_xy = [(0.0, 0.0)] * 17
            kpts_c = [0.0] * 17

        is_local = self.enabled and _normal_p() > 0.5
        
        augs = []
        if is_local and self.use_bbox and bbox != [0.0, 0.0, float(W), float(H)]:
            crop_coords = self._get_local_crop(W, H, bbox, self.min_scale)
            augs.append(A.Crop(x_min=int(crop_coords[0]), y_min=int(crop_coords[1]), x_max=int(crop_coords[2]), y_max=int(crop_coords[3])))
        elif is_local:
            augs.append(A.RandomResizedCrop(height=self.size, width=self.size, scale=(self.min_scale, 1.0)))
            
        augs.extend([
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.8),
            A.GaussNoise(var_limit=(10.0, 30.0), p=0.5),
            A.MotionBlur(blur_limit=5, p=0.5),
            # Safe erasing: not too large to avoid completely removing the person
            A.CoarseDropout(max_holes=4, max_height=32, max_width=32, min_holes=1, min_height=16, min_width=16, p=0.5),
            A.Resize(self.size, self.size),
            A.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ToTensorV2(),
        ])
        
        transform = A.Compose(augs, 
                              bbox_params=A.BboxParams(format='pascal_voc', label_fields=['bbox_classes']),
                              keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
        
        try:
            transformed = transform(image=img_np, bboxes=[bbox], bbox_classes=[1], keypoints=kpts_xy)
        except Exception:
            # Fallback
            transformed = transform(image=img_np, bboxes=[[0.0, 0.0, float(W), float(H)]], bbox_classes=[1], keypoints=kpts_xy)
            
        out_img = transformed['image']
        out_bbox = transformed['bboxes'][0] if transformed['bboxes'] else [0,0,0,0]
        out_kpts_xy = transformed['keypoints']
        
        out_kpts = []
        for (x, y), c in zip(out_kpts_xy, kpts_c):
            if x < 0 or x >= self.size or y < 0 or y >= self.size:
                c = 0.0
            out_kpts.extend([x, y, c])
            
        return out_img, torch.tensor(out_bbox, dtype=torch.float), torch.tensor(out_kpts, dtype=torch.float)


def build_eval_transform(size: int = 384):
    """Deterministic transform used at validation / inference (GLOBAL full image)."""
    return T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
