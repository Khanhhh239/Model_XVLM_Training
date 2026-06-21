"""FILIP token-wise late interaction (Yao et al., ICLR 2022).

Global pooling lets big nouns drown out verbs ("falling" is a small change in limb
angle).  FILIP forces EACH text token to find its best-matching image patch, giving
verbs their own matching path -> targets failure groups #2 (action) and #4 (foreground).

MEMORY WARNING: the full similarity tensor is [Bi, Bj, Ni, Nt].  At SigLIP @512
(Ni=1024 patches, Nt=64, B=128) that is ~5.5 GB -- FILIP is NOT cheap.  Use `chunk` to
bound peak memory by processing the image-batch dimension in slices (result is identical).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _filip_block(img_tokens, txt_tokens, txt_mask):
    sim = torch.einsum("ind,jmd->ijnm", img_tokens, txt_tokens)  # [bi, Bj, Ni, Nt]
    if txt_mask is not None:
        sim = sim.masked_fill(~txt_mask.bool()[None, :, None, :], float("-inf"))
    return sim.max(dim=-1).values.mean(dim=-1)  # [bi, Bj]


def filip_sim(
    img_tokens: torch.Tensor,
    txt_tokens: torch.Tensor,
    txt_mask: torch.Tensor | None = None,
    chunk: int = 0,
) -> torch.Tensor:
    """img_tokens [B, Ni, D] (L2-normalised), txt_tokens [B, Nt, D] (L2-normalised),
    txt_mask [B, Nt] (1 = real token, 0 = PAD).  Returns image->text sim [B, B] = mean over
    image patches of (max over valid text tokens of dot product).  `chunk` > 0 slices the
    image-batch dimension to cap peak memory (identical result)."""
    bi = img_tokens.size(0)
    if chunk and chunk < bi:
        return torch.cat(
            [_filip_block(img_tokens[s : s + chunk], txt_tokens, txt_mask) for s in range(0, bi, chunk)],
            dim=0,
        )
    return _filip_block(img_tokens, txt_tokens, txt_mask)


def filip_loss(
    img_tokens: torch.Tensor,
    txt_tokens: torch.Tensor,
    logit_scale: torch.Tensor,
    txt_mask: torch.Tensor | None = None,
    chunk: int = 0,
) -> torch.Tensor:
    """Symmetric InfoNCE over the token-wise similarity matrix."""
    sim = filip_sim(img_tokens, txt_tokens, txt_mask, chunk=chunk) * logit_scale  # [B, B]
    b = sim.size(0)
    labels = torch.arange(b, device=sim.device)
    return 0.5 * (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels))
