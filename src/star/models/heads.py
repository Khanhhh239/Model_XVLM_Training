"""Additional task heads for STAGE 1: Box Grounding + Anomaly Classification.

These heads are trained on top of the frozen vision encoder to add spatial understanding
(box grounding) and semantic understanding (normal vs anomaly behavior).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class BoxGroundingHead(nn.Module):
    """Predict bounding box [x, y, w, h] normalized to [0, 1] from image features.
    
    Loss: GIoU + L1 with weight ratio 2:5 (as per plan).
    """
    
    def __init__(self, input_dim: int = 256, hidden_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 4),
            nn.Sigmoid(),  # Output in [0, 1]
        )
    
    def forward(self, img_feat: Tensor) -> Tensor:
        """
        Args:
            img_feat: [B, D] image features (from backbone [CLS] token)
            
        Returns:
            bbox_pred: [B, 4] predicted box [x, y, w, h] in [0, 1]
        """
        return self.mlp(img_feat)
    
    @staticmethod
    def compute_loss(
        bbox_pred: Tensor,
        bbox_gt: Tensor,
        mask: Tensor | None = None,
        w_giou: float = 2.0,
        w_l1: float = 5.0,
    ) -> Tensor:
        """Combined GIoU + L1 loss.
        
        Args:
            bbox_pred: [B, 4] predicted boxes [x, y, w, h]
            bbox_gt: [B, 4] ground truth boxes [x, y, w, h]
            mask: [B] binary mask (1 = valid box, 0 = skip). If None, use all samples.
            w_giou: Weight for GIoU loss (default 2.0)
            w_l1: Weight for L1 loss (default 5.0)
            
        Returns:
            loss: Scalar weighted loss (0 if no valid boxes)
        """
        if mask is None:
            mask = torch.ones(bbox_pred.size(0), device=bbox_pred.device, dtype=torch.bool)
        
        if mask.sum() == 0:
            return torch.tensor(0.0, device=bbox_pred.device)
        
        # Apply mask
        pred = bbox_pred[mask]
        gt = bbox_gt[mask]
        
        # L1 loss
        loss_l1 = F.l1_loss(pred, gt, reduction='mean')
        
        # GIoU loss
        loss_giou = compute_giou_loss(pred, gt)
        
        return w_giou * loss_giou + w_l1 * loss_l1


class AnomalyClassificationHead(nn.Module):
    """Binary classifier: Normal (0) vs Anomaly (1) behavior.
    
    Uses label_type field from dataset:
      - "goal" → Normal (0)
      - "wentwrong" → Anomaly (1)
    """
    
    def __init__(self, input_dim: int = 256, hidden_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 2),  # Binary: [normal_logit, anomaly_logit]
        )
    
    def forward(self, img_feat: Tensor) -> Tensor:
        """
        Args:
            img_feat: [B, D] image features
            
        Returns:
            logits: [B, 2] classification logits
        """
        return self.mlp(img_feat)
    
    @staticmethod
    def compute_loss(
        logits: Tensor,
        labels: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        """Cross-entropy loss for anomaly classification.
        
        Args:
            logits: [B, 2] predicted logits
            labels: [B] ground truth labels (0=normal, 1=anomaly)
            mask: [B] binary mask (1 = valid label, 0 = skip). If None, use all.
            
        Returns:
            loss: Scalar CE loss (0 if no valid labels)
        """
        if mask is None:
            mask = torch.ones(logits.size(0), device=logits.device, dtype=torch.bool)
        
        if mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device)
        
        return F.cross_entropy(logits[mask], labels[mask])


class PhraseBoxHead(nn.Module):
    """Phrase-grounded box head (wrong-case group D: multi-person / #3 spatial).

    Predicts ONE box [x, y, w, h] in [0,1] from the CROSS-encoded (image, noun-phrase) [CLS] vector
    (backbone.cross_feature). Run once per caption noun-phrase -> grounds each phrase to its region,
    teaching the model WHICH person/object the caption refers to. Reuses BoxGroundingHead.compute_loss
    (masked GIoU+L1) on the up-to-P phrase boxes per image.
    """

    def __init__(self, cross_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cross_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 4),
            nn.Sigmoid(),
        )

    def forward(self, cross_cls: Tensor) -> Tensor:
        """cross_cls: [N, cross_dim] fused [CLS]. Returns [N, 4] box [x, y, w, h] in [0, 1]."""
        return self.mlp(cross_cls)

    @staticmethod
    def compute_loss(pred, gt, mask=None, w_giou: float = 2.0, w_l1: float = 5.0):
        return BoxGroundingHead.compute_loss(pred, gt, mask=mask, w_giou=w_giou, w_l1=w_l1)


def compute_giou_loss(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """Generalized IoU loss (1 - GIoU).
    
    Args:
        boxes1, boxes2: [N, 4] tensors in format [x, y, w, h], normalized to [0, 1]
        
    Returns:
        loss: Scalar (mean 1 - GIoU over batch)
        
    Reference: Rezatofighi et al., "Generalized Intersection over Union", CVPR 2019
    """
    # Convert [x, y, w, h] to [x1, y1, x2, y2]
    b1_x1, b1_y1 = boxes1[:, 0] - boxes1[:, 2] / 2, boxes1[:, 1] - boxes1[:, 3] / 2
    b1_x2, b1_y2 = boxes1[:, 0] + boxes1[:, 2] / 2, boxes1[:, 1] + boxes1[:, 3] / 2
    b2_x1, b2_y1 = boxes2[:, 0] - boxes2[:, 2] / 2, boxes2[:, 1] - boxes2[:, 3] / 2
    b2_x2, b2_y2 = boxes2[:, 0] + boxes2[:, 2] / 2, boxes2[:, 1] + boxes2[:, 3] / 2
    
    # Clamp to [0, 1]
    b1_x1, b1_y1, b1_x2, b1_y2 = b1_x1.clamp(0, 1), b1_y1.clamp(0, 1), b1_x2.clamp(0, 1), b1_y2.clamp(0, 1)
    b2_x1, b2_y1, b2_x2, b2_y2 = b2_x1.clamp(0, 1), b2_y1.clamp(0, 1), b2_x2.clamp(0, 1), b2_y2.clamp(0, 1)
    
    # Intersection area
    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)
    inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    
    # Union area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union_area = b1_area + b2_area - inter_area + 1e-7
    
    # IoU
    iou = inter_area / union_area
    
    # Smallest enclosing box
    enclose_x1 = torch.min(b1_x1, b2_x1)
    enclose_y1 = torch.min(b1_y1, b2_y1)
    enclose_x2 = torch.max(b1_x2, b2_x2)
    enclose_y2 = torch.max(b1_y2, b2_y2)
    enclose_area = (enclose_x2 - enclose_x1) * (enclose_y2 - enclose_y1) + 1e-7
    
    # GIoU
    giou = iou - (enclose_area - union_area) / enclose_area
    
    # Loss: 1 - GIoU (range [0, 2], lower is better)
    return (1 - giou).mean()
