"""SearchFT = CMP Search + anomaly head + MINED cross-ID hard-negative ITM loss, plus
LoRA-on-encoders / full-fine-tune-cross helpers. Drop into a cloned CMP repo (sys.path has it).

Design (SAFE, no-degrade):
  - LoRA(r) on Swin vision + BERT self-attn  -> preserves the 80-mAP encoders (anti-forget)
  - FULL fine-tune on cross-attention + itm_head + anomaly_head + pose_block (precision/anomaly)
  - anomaly head: 2-class (normal/anomaly) on the image CLS token, free label from the attribute
  - mined neg: ANCE-style cross-ID hardest confuser, reuses CMP get_matching_loss_hard
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_search import Search


class SearchFT(Search):
    def __init__(self, config):
        super().__init__(config)
        self.w_anom = config.get("w_anom", 0.2)
        self.use_mined = config.get("use_mined_neg", True)
        vw = config.get("vision_width", 1024)            # swin-base final dim (verify for your ckpt)
        self.anomaly_head = nn.Linear(vw, 2)
        self.init_params.extend(["anomaly_head." + n for n, _ in self.anomaly_head.named_parameters()])

    def _img_with_pose(self, image, pose):
        emb, atts = self.get_vision_embeds(image)
        if self.be_pose_img and pose is not None:
            p = self.pose_conv(pose) if self.be_pose_conv else pose
            pe, _ = self.get_vision_embeds(p)
            emb = self.pose_block(emb, pe)
        return emb, atts

    def forward(self, image, text_ids, text_atts, text_ids_masked=None, masked_pos=None, masked_ids=None,
                idx=None, text_ids_eda=None, text_atts_eda=None, pose=None,
                hard_i=None, hard_i_pose=None, hard_text_ids=None, hard_text_atts=None,
                anomaly_label=None,
                mined_i=None, mined_i_pose=None, mined_text_ids=None, mined_text_atts=None):
        image_embeds, image_atts = self._img_with_pose(image, pose)
        text_embeds = self.get_text_embeds(text_ids, text_atts)
        image_feat, text_feat = self.get_image_feat(image_embeds), self.get_text_feat(text_embeds)

        loss_itc = self.get_contrastive_loss(image_feat, text_feat, idx=idx)
        loss_itm = self.get_matching_loss(image_embeds, image_atts, image_feat,
                                          text_embeds, text_atts, text_feat, idx=idx)
        # EDA (kept from CMP)
        if text_ids_eda is not None:
            te = self.get_text_embeds(text_ids_eda, text_atts_eda); tfe = self.get_text_feat(te)
            loss_itc = loss_itc + 0.8 * self.get_contrastive_loss(image_feat, tfe, idx=idx)
            loss_itm = loss_itm + 0.8 * self.get_matching_loss(image_embeds, image_atts, image_feat,
                                                               te, text_atts_eda, tfe, idx=idx)
        loss_mlm = self.get_mlm_loss(text_ids_masked, text_atts, image_embeds, image_atts, masked_pos, masked_ids)

        # CMP ID-based hard neg
        if self.be_hard and hard_i is not None:
            he, ha = self._img_with_pose(hard_i, hard_i_pose)
            hte = self.get_text_embeds(hard_text_ids, hard_text_atts)
            loss_itm = loss_itm + self.get_matching_loss_hard(image_embeds, image_atts, he, ha,
                                                              text_embeds, text_atts, hte, hard_text_atts)
        # MINED cross-ID hard neg (ANCE) — the "thật khó" lever
        if self.use_mined and mined_i is not None:
            me, ma = self._img_with_pose(mined_i, mined_i_pose)
            mte = self.get_text_embeds(mined_text_ids, mined_text_atts)
            loss_itm = loss_itm + self.get_matching_loss_hard(image_embeds, image_atts, me, ma,
                                                              text_embeds, text_atts, mte, mined_text_atts)
        # anomaly head (auxiliary, free label)
        loss_anom = image_embeds.new_zeros(())
        if anomaly_label is not None:
            logits = self.anomaly_head(image_embeds[:, 0, :])
            loss_anom = self.w_anom * F.cross_entropy(logits, anomaly_label)
        return loss_itc, loss_itm, loss_mlm, loss_anom


# ----------------------------- parameter-efficiency helpers -----------------------------
_FULL_FT_KEYS = ("crossattention", "itm_head", "anomaly_head", "pose_block", "pose_conv",
                 "vision_proj", "text_proj", "itc")           # full-FT precision/anomaly path


def apply_lora_keep_cross(model, r=16, alpha=32, dropout=0.05):
    """LoRA on encoder attn (Swin vision + BERT self-attn); FULL-FT the cross/heads/pose."""
    from peft import LoraConfig, get_peft_model
    targets = []
    for n, m in model.named_modules():
        if not isinstance(m, nn.Linear):
            continue
        low = n.lower()
        if any(k in low for k in _FULL_FT_KEYS):
            continue                                          # keep these full-FT (not LoRA)
        if ("vision_encoder" in low or "text_encoder" in low) and \
           any(t in low for t in ("query", "key", "value", "qkv", "attn", "attention")):
            targets.append(n)
    if not targets:
        print("WARN: no LoRA targets matched — verify module names; falling back to freeze_encoders.")
        return freeze_encoders_keep_cross(model)
    model = get_peft_model(model, LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout,
                                             target_modules=targets, bias="none"))
    for n, p in model.named_parameters():                     # re-enable FULL-FT cross/heads/pose
        if any(k in n.lower() for k in _FULL_FT_KEYS):
            p.requires_grad = True
    _report_trainable(model)
    return model


def freeze_encoders_keep_cross(model):
    """Ultra-safe fallback: FREEZE encoders entirely, FULL-FT only cross/heads/pose."""
    for n, p in model.named_parameters():
        p.requires_grad = any(k in n.lower() for k in _FULL_FT_KEYS)
    _report_trainable(model)
    return model


def _report_trainable(model):
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tot = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {tr/1e6:.1f}M / {tot/1e6:.1f}M ({100*tr/tot:.1f}%)")
