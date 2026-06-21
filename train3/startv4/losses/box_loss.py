"""Box-grounding loss: L1 + GIoU (Rezatofighi et al., CVPR 2019).

Auxiliary head on X-VLM: predicting WHERE the described region is forces the encoder to
encode spatial location -> helps failure groups #3/#7 (spatial relations).  Boxes are
xyxy normalised to [0, 1].
"""
from __future__ import annotations

import torch


def box_giou(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Element-wise GIoU for matched boxes.  pred/target: [N, 4] xyxy in [0, 1].
    Returns [N] in [-1, 1].
    """
    x1 = torch.max(pred[:, 0], target[:, 0])
    y1 = torch.max(pred[:, 1], target[:, 1])
    x2 = torch.min(pred[:, 2], target[:, 2])
    y2 = torch.min(pred[:, 3], target[:, 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)

    area_p = (pred[:, 2] - pred[:, 0]).clamp(min=0) * (pred[:, 3] - pred[:, 1]).clamp(min=0)
    area_t = (target[:, 2] - target[:, 0]).clamp(min=0) * (target[:, 3] - target[:, 1]).clamp(min=0)
    union = area_p + area_t - inter + eps
    iou = inter / union

    xc1 = torch.min(pred[:, 0], target[:, 0])
    yc1 = torch.min(pred[:, 1], target[:, 1])
    xc2 = torch.max(pred[:, 2], target[:, 2])
    yc2 = torch.max(pred[:, 3], target[:, 3])
    area_c = (xc2 - xc1).clamp(min=0) * (yc2 - yc1).clamp(min=0) + eps
    return iou - (area_c - union) / area_c


def box_loss(
    pred: torch.Tensor, target: torch.Tensor, l1_w: float = 1.0, giou_w: float = 1.0
) -> torch.Tensor:
    """pred/target: [N, 4] xyxy in [0, 1].  Returns scalar (>= 0)."""
    l1 = (pred - target).abs().sum(-1).mean()
    giou = box_giou(pred, target).mean()
    return l1_w * l1 + giou_w * (1.0 - giou)
