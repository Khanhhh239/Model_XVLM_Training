"""X-VLM-v4 trainer (Model B / Phase 3).

Loss = w_itc*ITC(+queue) + w_itm*ITM(hard-neg) + w_filip*FILIP + w_box*box(L1+GIoU)
       + w_anom*CE(bucket) + w_smoothap*Smooth-AP.

ITM hard negatives are sampled in-batch from the ITC similarity (ALBEF/X-VLM style):
for each pair we add (image_i, hardest text) and (hardest image, text_i) as label-0 pairs.
Golden rule: watch R@5/R@10 on VAL-B (eval hook) -- if an aux head drags it down, lower its
weight (grad_norm is logged to help spot a head swamping ITC).
"""
from __future__ import annotations

import contextlib

import torch
import torch.nn.functional as F
from torch.optim import AdamW

from ..losses import box_loss, filip_loss, info_nce, smooth_ap_loss
from ..models.ema import ModelEMA
from ..models.queue import NegativeQueue
from .sched import cosine_warmup


class XVLMTrainer:
    def __init__(self, model, cfg, device: str = "cpu"):
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        o = cfg.optim

        if o.get_path("warm_start", None):
            self._warm_start(o.get_path("warm_start"))

        params = [p for p in self.model.parameters() if p.requires_grad]
        self.opt = AdamW(params, lr=float(o.get_path("lr", 2e-4)),
                         weight_decay=float(o.get_path("wd", 0.05)), betas=(0.9, 0.98))
        self.grad_clip = float(o.get_path("grad_clip", 1.0))
        self.warmup_epochs = float(o.get_path("warmup_epochs", 1.0))
        self.sched = None

        self.w = {k: float(o.get_path(f"w_{k}", d)) for k, d in
                  dict(itc=1.0, itm=1.5, filip=0.5, box=0.5, anom=0.3, smoothap=0.2).items()}

        qsize = int(o.get_path("queue_size", 65536))
        # SEPARATE image + text queues (review fix A2): t2i negatives must be IMAGE features.
        self.queue_img = NegativeQueue(self.model.embed_dim, qsize).to(device) if qsize > 0 else None
        self.queue_txt = NegativeQueue(self.model.embed_dim, qsize).to(device) if qsize > 0 else None
        self.ema = ModelEMA(self.model, float(o.get_path("ema_decay", 0.999))) if bool(
            o.get_path("ema", True)) else None
        self.use_amp = device == "cuda" and bool(o.get_path("bf16", True))
        self.best_state = None
        # gradient checkpointing on the HF backbones (review fix F2: was never enabled for X-VLM)
        if device == "cuda" and bool(o.get_path("grad_checkpoint", True)):
            for sub in ("hf_vision", "hf_text"):
                mod = getattr(self.model, sub, None)
                if mod is not None and hasattr(mod, "gradient_checkpointing_enable"):
                    with contextlib.suppress(Exception):
                        mod.gradient_checkpointing_enable()

    def _warm_start(self, path):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        sd = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        print(f"[warm-start] {path}  missing={len(missing)} unexpected={len(unexpected)}")

    def _amp(self):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16) if self.use_amp else contextlib.nullcontext()

    def _itm_loss(self, image_tokens, text_tokens, text_mask, sim):
        """sim[i,j] = image_i . text_j (detached).  CE over [pos; neg-txt; neg-img]."""
        b = image_tokens.size(0)
        if b < 2:
            return image_tokens.new_zeros(())
        with torch.no_grad():
            w_i2t = F.softmax(sim, dim=1).clone()
            w_t2i = F.softmax(sim, dim=0).clone()
            idx = torch.arange(b, device=sim.device)
            w_i2t[idx, idx] = 0.0
            w_t2i[idx, idx] = 0.0
            neg_txt = torch.multinomial(w_i2t + 1e-9, 1).squeeze(1)      # image i -> hard text
            neg_img = torch.multinomial(w_t2i.t() + 1e-9, 1).squeeze(1)  # text i -> hard image

        img_all = torch.cat([image_tokens, image_tokens, image_tokens[neg_img]], dim=0)
        txt_all = torch.cat([text_tokens, text_tokens[neg_txt], text_tokens], dim=0)
        mask_all = torch.cat([text_mask, text_mask[neg_txt], text_mask], dim=0)
        logits = self.model.itm_logits(img_all, txt_all, mask_all)
        labels = torch.cat([torch.ones(b), torch.zeros(2 * b)]).long().to(self.device)
        return F.cross_entropy(logits, labels)

    def compute_loss(self, batch):
        b = batch
        px = b["pixel_values"].to(self.device)
        ids = b["input_ids"].to(self.device)
        am = b["attention_mask"].to(self.device)
        kpts = b["keypoints"].to(self.device) if "keypoints" in b else None

        img_pooled, img_tok = self.model.encode_image(px, kpts)
        txt_pooled, txt_tok, txt_mask = self.model.encode_text(ids, am)
        itc_i, itc_t = self.model.itc(img_pooled, txt_pooled)
        scale = self.model.logit_scale.exp()
        sim = (itc_i @ itc_t.t()).detach()  # for ITM hard-neg sampling (always needed)

        loss = torch.zeros((), device=self.device)
        logs = {}

        # ITC contrastive (+queue) -- SKIPPED in rerank-only mode (w_itc=0): saves the second
        # contrastive model's compute and the MoCo queue (review pass-3 / Waste #2).
        if self.w["itc"] > 0:
            qz_txt = self.queue_txt.get() if self.queue_txt is not None else None
            qz_img = self.queue_img.get() if self.queue_img is not None else None
            l_itc = info_nce(itc_i, itc_t, scale, queue_text=qz_txt, queue_image=qz_img)
            loss = loss + self.w["itc"] * l_itc
            logs["itc"] = float(l_itc.detach())

        l_itm = self._itm_loss(img_tok, txt_tok, txt_mask, sim)  # the rerank objective
        loss = loss + self.w["itm"] * l_itm
        logs["itm"] = float(l_itm.detach())

        if self.model.use_filip and self.w["filip"] > 0:
            vi, vt = self.model.filip(img_tok, txt_tok)
            l_f = filip_loss(vi, vt, scale, txt_mask)
            loss = loss + self.w["filip"] * l_f
            logs["filip"] = float(l_f.detach())

        if self.model.use_box and self.w["box"] > 0 and "has_bbox" in b and bool(b["has_bbox"].any()):
            keep = b["has_bbox"].to(self.device).bool()
            fused = self.model.cross_cls(img_tok, txt_tok, txt_mask)
            l_b = box_loss(self.model.box(fused)[keep], b["bbox"].to(self.device)[keep])
            loss = loss + self.w["box"] * l_b
            logs["box"] = float(l_b.detach())

        if self.model.use_anom and self.w["anom"] > 0 and "bucket" in b:
            l_a = F.cross_entropy(self.model.anom(img_pooled), b["bucket"].to(self.device))
            loss = loss + self.w["anom"] * l_a
            logs["anom"] = float(l_a.detach())

        if self.w["smoothap"] > 0:
            l_s = smooth_ap_loss(itc_i @ itc_t.t(), torch.eye(itc_i.size(0), device=self.device))
            loss = loss + self.w["smoothap"] * l_s
            logs["smoothap"] = float(l_s.detach())

        logs["total"] = float(loss.detach())
        if self.queue_txt is not None and self.w["itc"] > 0:  # queue only matters for ITC
            self.queue_txt.enqueue(itc_t.detach())
            self.queue_img.enqueue(itc_i.detach())
        return loss, logs

    def train_step(self, batch):
        self.model.train()
        self.opt.zero_grad(set_to_none=True)
        with self._amp():
            loss, logs = self.compute_loss(batch)
        loss.backward()
        if self.grad_clip > 0:
            logs["grad_norm"] = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip))
        self.opt.step()
        if self.sched is not None:
            self.sched.step()
        if self.ema is not None:
            self.ema.update(self.model)
        return logs

    def overfit_one_batch(self, batch, steps: int = 60):
        return [self.train_step(batch)["total"] for _ in range(steps)]

    def fit(self, loader, epochs: int = 1, log_every: int = 50, on_log=None,
            eval_fn=None, eval_every: int = 0, on_eval=None):
        spe = len(loader)
        self.sched = cosine_warmup(self.opt, int(self.warmup_epochs * spe), epochs * spe)
        best = {"mAP": -1.0}
        step = 0
        for ep in range(epochs):
            for batch in loader:
                logs = self.train_step(batch)
                step += 1
                if on_log and step % log_every == 0:
                    on_log(ep, step, logs)
                if eval_fn and eval_every > 0 and step % eval_every == 0:
                    m = eval_fn(self.model)
                    if m.get("mAP", -1) > best["mAP"]:
                        best = dict(m, step=step)
                        self.best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                    if on_eval:
                        on_eval(step, m, best)
        return best
