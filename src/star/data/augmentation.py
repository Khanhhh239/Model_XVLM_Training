"""Strong augmentation pipeline for Sim2Real gap.

Applies degradations that mimic real video frames (blur, compression, noise) to synthetic images,
helping the model generalize from clean synthetic training data to noisy real test frames.

Uses Albumentations for automatic bbox/keypoint coordinate adjustment.
"""
from __future__ import annotations

import cv2
import numpy as np

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    A = None
    ToTensorV2 = None


def build_strong_augmentation(image_size: int = 384, train: bool = True):
    """Build strong augmentation pipeline for STAGE 1 training.
    
    Args:
        image_size: Target image size (384 for Kaggle T4)
        train: If True, apply augmentations. If False, only resize+normalize.
        
    Returns:
        Albumentations Compose object (if available), otherwise None.
    """
    if not ALBUMENTATIONS_AVAILABLE:
        return None
    
    if not train:
        # Eval: simple resize + normalize
        return A.Compose([
            A.Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    
    # Train: STRONG augmentation for Sim2Real
    return A.Compose([
        # === Sim2Real (close synthetic→real gap) ===
        # Motion blur: Frame "đang ngã" bị nhòe
        A.MotionBlur(blur_limit=7, p=0.3),
        
        # JPEG compression: Video frame bị nén
        A.ImageCompression(quality_lower=30, quality_upper=90, compression_type=A.ImageCompression.ImageCompressionType.JPEG, p=0.4),
        
        # Downscale→Upscale: Độ phân giải video thấp
        A.Downscale(scale_min=0.5, scale_max=0.9, interpolation=cv2.INTER_LINEAR, p=0.3),
        
        # Gaussian noise: Nhiễu cảm biến
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
        
        # Color jitter: Ánh sáng real đa dạng
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05, p=0.5),
        
        # === Robustness ===
        # Random Erasing (CoarseDropout): Bền occlusion
        A.CoarseDropout(
            max_holes=3,
            max_height=int(0.15 * image_size),  # ~15% of image
            max_width=int(0.15 * image_size),
            min_holes=1,
            min_height=int(0.05 * image_size),
            min_width=int(0.05 * image_size),
            fill_value=0,
            p=0.25,
        ),
        
        # Random Resized Crop (Albumentations auto-adjusts bbox/keypoints!)
        A.RandomResizedCrop(image_size, image_size, scale=(0.8, 1.0), ratio=(0.9, 1.1), p=1.0),
        
        # ⚠️ NO large rotation (người đứng → trông như đang ngã → hỏng label)
        # ⚠️ NO flip nếu caption có "left/right" (đảo hướng sai nghĩa)
        
        # Normalize + ToTensor
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(
        format='coco',  # [x, y, w, h]
        min_area=0,
        min_visibility=0.3,  # Keep bbox if at least 30% visible after crop
        label_fields=['bbox_labels'],
    ), keypoint_params=A.KeypointParams(
        format='xy',  # Keypoints as (x, y) pairs
        label_fields=['kp_labels'],
        remove_invisible=False,
    ))


def apply_augmentation_with_bbox_kpts(
    image: np.ndarray,
    bbox: list[float] | None = None,
    keypoints: list[float] | None = None,
    transform=None,
):
    """Apply Albumentations transform with bbox/keypoints auto-adjustment.
    
    Args:
        image: [H, W, 3] numpy array (RGB, uint8)
        bbox: [x, y, w, h] normalized to [0, 1], or None
        keypoints: [17*3] flattened COCO keypoints (x, y, conf), or None
        transform: Albumentations Compose object
        
    Returns:
        Transformed image (tensor), bbox (adjusted), keypoints (adjusted)
    """
    if transform is None:
        # Fallback: simple resize if albumentations not available
        import torch
        from PIL import Image
        from torchvision import transforms as T
        img_pil = Image.fromarray(image)
        transform_fallback = T.Compose([
            T.Resize((384, 384)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        img_tensor = transform_fallback(img_pil)
        return img_tensor, bbox, keypoints
    
    # Prepare inputs for albumentations
    h, w = image.shape[:2]
    
    # Convert normalized bbox [x, y, w, h] to pixel coords for albumentations
    bboxes = []
    bbox_labels = []
    if bbox is not None:
        x, y, width, height = bbox
        bboxes = [[x * w, y * h, width * w, height * h]]  # COCO format expects pixel coords
        bbox_labels = [0]  # Dummy label
    
    # Convert keypoints [x1, y1, c1, x2, y2, c2, ...] to [(x1, y1), (x2, y2), ...]
    kpts = []
    kpt_labels = []
    if keypoints is not None and len(keypoints) == 51:  # 17 joints * 3
        for i in range(0, 51, 3):
            x_kpt, y_kpt, conf = keypoints[i], keypoints[i+1], keypoints[i+2]
            if conf > 0:  # Only include visible keypoints
                kpts.append((x_kpt * w, y_kpt * h))  # Pixel coords
                kpt_labels.append(i // 3)  # Joint index
    
    # Apply transform
    transformed = transform(
        image=image,
        bboxes=bboxes,
        bbox_labels=bbox_labels,
        keypoints=kpts,
        kp_labels=kpt_labels,
    )
    
    img_out = transformed['image']  # Tensor [3, H, W]
    
    # Convert bbox back to normalized [0, 1]
    bbox_out = None
    if len(transformed['bboxes']) > 0:
        x_px, y_px, w_px, h_px = transformed['bboxes'][0]
        bbox_out = [x_px / 384, y_px / 384, w_px / 384, h_px / 384]
    
    # Convert keypoints back to flattened [51] format
    kpts_out = None
    if len(transformed['keypoints']) > 0 and keypoints is not None:
        kpts_out = [0.0] * 51
        for (x_kpt, y_kpt), label in zip(transformed['keypoints'], transformed['kp_labels']):
            idx = label * 3
            kpts_out[idx] = x_kpt / 384
            kpts_out[idx + 1] = y_kpt / 384
            kpts_out[idx + 2] = 1.0  # Confidence (assume visible after transform)
    
    return img_out, bbox_out, kpts_out
