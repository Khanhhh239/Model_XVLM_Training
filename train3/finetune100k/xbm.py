"""Cross-Batch Memory (XBM, Wang CVPR20): a FIFO of DETACHED image+text features used as extra
negatives for the ITC contrastive loss — gives "huge-batch" contrast on a small GPU batch (the
A100-batch substitute on Kaggle T4). No momentum encoder; staleness is negligible at low fine-tune LR.
"""
import torch
import torch.nn.functional as F


class XBM:
    def __init__(self, dim, size=8192, device="cuda"):
        self.size = size
        self.img = torch.zeros(size, dim, device=device)
        self.txt = torch.zeros(size, dim, device=device)
        self.ptr = 0
        self.full = False

    @torch.no_grad()
    def enqueue(self, img_feat, txt_feat):
        b = img_feat.size(0)
        idx = (torch.arange(b, device=img_feat.device) + self.ptr) % self.size
        self.img[idx] = F.normalize(img_feat.detach(), dim=-1)
        self.txt[idx] = F.normalize(txt_feat.detach(), dim=-1)
        self.ptr = (self.ptr + b) % self.size
        self.full = self.full or self.ptr < b

    def get(self):
        n = self.size if self.full else self.ptr
        return self.img[:n], self.txt[:n]


def itc_with_xbm(img_feat, txt_feat, temp, xbm=None, idx=None):
    """Symmetric InfoNCE over the in-batch pairs PLUS xbm negatives (queue keys are negatives only).
    img_feat/txt_feat: [B,D] L2-normalized. temp: scalar (CMP's learned temp)."""
    img_feat = F.normalize(img_feat, dim=-1)
    txt_feat = F.normalize(txt_feat, dim=-1)
    B = img_feat.size(0)
    labels = torch.arange(B, device=img_feat.device)
    # gallery keys = in-batch images (+ xbm image bank); query keys = in-batch texts (+ xbm text bank)
    img_keys, txt_keys = img_feat, txt_feat
    if xbm is not None:
        qi, qt = xbm.get()
        if qi.size(0) > 0:
            img_keys = torch.cat([img_feat, qi], 0)
            txt_keys = torch.cat([txt_feat, qt], 0)
    logit_t2i = (txt_feat @ img_keys.t()) / temp        # [B, B+Q] positive at col i==row
    logit_i2t = (img_feat @ txt_keys.t()) / temp
    loss = 0.5 * (F.cross_entropy(logit_t2i, labels) + F.cross_entropy(logit_i2t, labels))
    return loss
