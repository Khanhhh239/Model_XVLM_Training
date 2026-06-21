"""Auxiliary / interaction heads for X-VLM-v4.

- BBoxHead        : regress primary-region box (xyxy in [0,1]) -> spatial grounding (#3/#7).
- AnomalyHead     : classify normal/anomaly bucket (free synthetic label) -> anomaly salience.
- PoseRegionFuse  : fuse 17x3 keypoints into region tokens (off-balance cue for #1/#2).
- FilipProjection : project image/text tokens for token-wise late interaction (#2/#4).

BBoxHead and AnomalyHead are AUXILIARY: their gradients shape the shared encoder but they
are not used to produce retrieval scores at inference (see README sec. "auxiliary heads").
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BBoxHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, 4)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [N, in_dim] fused (region, phrase) feature -> [N, 4] xyxy in [0,1]."""
        return torch.sigmoid(self.mlp(x))


class AnomalyHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int = 3, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, cls_feat: torch.Tensor) -> torch.Tensor:
        """cls_feat: [N, in_dim] image [CLS] -> [N, num_classes] logits."""
        return self.mlp(cls_feat)


class PoseRegionFuse(nn.Module):
    """Encode 17 keypoints (x, y, conf) into one pose vector and add it (gated, learnable
    scalar init small) to EVERY token -- a GLOBAL pose fusion (broadcast over tokens), not a
    spatially-aligned per-patch fusion.  (A true spatial version would map each keypoint's
    (x, y) to its patch index; not implemented -- the name is kept for back-compat.)
    """

    def __init__(self, region_dim: int, num_kpts: int = 17, hidden: int = 128, gate_init: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(num_kpts * 3, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, region_dim)
        )
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(self, region_tokens: torch.Tensor, keypoints: torch.Tensor) -> torch.Tensor:
        """region_tokens: [B, T, D].  keypoints: [B, 17, 3] (x,y in [0,1], conf).
        Returns region_tokens + gate * pose_embedding (broadcast over tokens).
        """
        b = region_tokens.size(0)
        pose = self.mlp(keypoints.reshape(b, -1))           # [B, D]
        return region_tokens + self.gate * pose.unsqueeze(1)  # broadcast over T


class FilipProjection(nn.Module):
    def __init__(self, img_dim: int, txt_dim: int, proj_dim: int = 256):
        super().__init__()
        self.img_proj = nn.Linear(img_dim, proj_dim)
        self.txt_proj = nn.Linear(txt_dim, proj_dim)

    def forward(self, img_tokens: torch.Tensor, txt_tokens: torch.Tensor):
        """-> (img [B,Ni,P] L2-norm, txt [B,Nt,P] L2-norm)."""
        vi = F.normalize(self.img_proj(img_tokens), dim=-1)
        vt = F.normalize(self.txt_proj(txt_tokens), dim=-1)
        return vi, vt
