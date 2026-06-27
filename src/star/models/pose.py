"""Pose fusion (SSDC Eq1) — heatmap -> Pose Conv -> cross-attention into the IMAGE TOKEN stream.

Rebuilt from the old weak global gated-sum, which fused pose into the POOLED feature `img_feat`
[B,d] and therefore (a) only nudged one vector and (b) NEVER reached the ITM cross-encoder rerank
(which reads the raw `img_embeds` token sequence). Wrong-case analysis showed the real failure is
same-scene / different-pose frames (e.g. dumbbell "falling" vs "held") scored ~0.99 vs ~0.99 — a
pose-semantic gap the pooled-sum could not break.

New design (SSDC, arXiv:2604.23282, Eq1):
    fCA = softmax( (Wq fP)(Wk fI)^T / sqrt(d) )(Wv fI),   fV = fI + fCA
17 keypoints are rendered to per-joint Gaussian heatmaps, a small conv ("Pose Conv Module") encodes
them to a pose-token grid fP, and these cross-attend with the image patch tokens fI. The fused
residual is added back to the FULL image token sequence (`img_embeds`), so the enhancement reaches
BOTH `get_features` (the bi-encoder CLS / mean pool) AND `get_cross_embeds` (the ITM rerank).

Implementation note: the IMAGE tokens are the cross-attn QUERY (pose tokens are key/value) so the
residual lands on every image token — incl. the CLS that real X-VLM pools and the mean the dummy
pools. A learnable `gate` starts ~0 (sigmoid(-4)) so a freshly-built module does NOT disturb a
warm-init backbone; it opens as pose proves useful (the F1 floor still protects best.pth).
"""
from __future__ import annotations

import torch
from torch import Tensor, nn


class PoseHeatmapCrossAttn(nn.Module):
    def __init__(self, vision_width: int, n_keypoints: int = 17, heatmap_size: int = 48,
                 conv_ch: int = 128, pose_grid: int = 12, n_heads: int = 8,
                 sigma: float = 0.08, dropout: float = 0.0):
        super().__init__()
        self.n_kpts = n_keypoints
        self.hm = heatmap_size
        self.sigma = sigma
        # Pose Conv Module: [B,17,hm,hm] -> [B, vision_width, pose_grid, pose_grid]
        self.conv = nn.Sequential(
            nn.Conv2d(n_keypoints, conv_ch, 3, padding=1),
            nn.GroupNorm(8, conv_ch), nn.GELU(),
            nn.Conv2d(conv_ch, conv_ch, 3, stride=2, padding=1),
            nn.GroupNorm(8, conv_ch), nn.GELU(),
            nn.Conv2d(conv_ch, vision_width, 1),
            nn.AdaptiveAvgPool2d((pose_grid, pose_grid)),
        )
        # cross-attention: image tokens (query) attend to pose tokens (key/value) — SSDC Eq1
        self.ln_q = nn.LayerNorm(vision_width)
        self.ln_kv = nn.LayerNorm(vision_width)
        self.cross_attn = nn.MultiheadAttention(vision_width, n_heads, dropout=dropout,
                                                batch_first=True)
        # Zero-init the attention output projection so attn_out == 0 at init -> the module is an EXACT
        # identity on a warm-init backbone (the gate/biases can't leak a perturbation). out_proj still
        # receives gradient (via attention_result), so pose learns to switch on during training.
        nn.init.zeros_(self.cross_attn.out_proj.weight)
        nn.init.zeros_(self.cross_attn.out_proj.bias)
        self.gate = nn.Parameter(torch.zeros(1))           # sigmoid(0)=0.5; opens as out_proj learns

    def render_heatmap(self, keypoints: Tensor) -> Tensor:
        """keypoints [B, 17*3] normalized (x,y,conf) -> [B, 17, hm, hm] conf-weighted Gaussians."""
        B = keypoints.size(0)
        k = keypoints.view(B, self.n_kpts, 3)
        cx, cy, conf = k[..., 0], k[..., 1], k[..., 2]              # [B,17] in [0,1]
        coord = torch.linspace(0.0, 1.0, self.hm, device=keypoints.device, dtype=keypoints.dtype)
        gy, gx = torch.meshgrid(coord, coord, indexing="ij")        # [hm,hm]
        gx = gx.view(1, 1, self.hm, self.hm)
        gy = gy.view(1, 1, self.hm, self.hm)
        cx = cx.view(B, self.n_kpts, 1, 1)
        cy = cy.view(B, self.n_kpts, 1, 1)
        hm = torch.exp(-((gx - cx) ** 2 + (gy - cy) ** 2) / (2.0 * self.sigma ** 2))
        return hm * conf.view(B, self.n_kpts, 1, 1)                 # zero-conf joints contribute 0

    def forward(self, img_embeds: Tensor, keypoints: Tensor) -> Tensor:
        """img_embeds [B,N,vision_width], keypoints [B,51] -> pose-enhanced img_embeds [B,N,vision_width]."""
        hm = self.render_heatmap(keypoints.float())                 # [B,17,hm,hm]
        pf = self.conv(hm.to(img_embeds.dtype))                     # [B, vision_width, g, g]
        pose_tokens = pf.flatten(2).transpose(1, 2)                 # [B, g*g, vision_width]
        kv = self.ln_kv(pose_tokens)
        attn_out, _ = self.cross_attn(self.ln_q(img_embeds), kv, kv, need_weights=False)
        return img_embeds + torch.sigmoid(self.gate) * attn_out     # gated residual on ALL tokens
