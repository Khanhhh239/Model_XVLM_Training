"""SigLIP sigmoid pairwise loss (Zhai et al., ICCV 2023).

Each (image i, text j) pair is an independent binary problem: match (z=+1) on the
diagonal, non-match (z=-1) off-diagonal.  Unlike InfoNCE it needs no softmax over the
batch, so it is stable WITHOUT a huge batch or a negative queue -- that is exactly why
the SigLIP retrieval tower in STAR-v4 does NOT use the MoCo queue.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def siglip_sigmoid_loss(
    image_feat: torch.Tensor,
    text_feat: torch.Tensor,
    logit_scale: torch.Tensor,
    logit_bias: torch.Tensor,
) -> torch.Tensor:
    """image_feat / text_feat: [B, D] L2-normalised.  logit_scale (>0) and logit_bias
    are learnable scalars.  Returns a scalar loss.

    L = -(1/B) * sum_{i,j} log sigmoid( z_ij * (scale * <i,j> + bias) ),  z_ij = +1 if i==j else -1
    """
    if image_feat.shape != text_feat.shape:
        raise ValueError(f"shape mismatch {image_feat.shape} vs {text_feat.shape}")
    b = image_feat.size(0)
    logits = logit_scale * (image_feat @ text_feat.t()) + logit_bias  # [B, B]
    labels = 2.0 * torch.eye(b, device=logits.device, dtype=logits.dtype) - 1.0
    return -F.logsigmoid(labels * logits).sum() / b
