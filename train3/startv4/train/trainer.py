"""SigLIP retrieval trainer (Model A).

Core loss = SigLIP sigmoid (no queue).  Optional FILIP token loss.  bf16 autocast on CUDA,
EMA weights, grad clipping + grad-norm logging, cosine-warmup LR, optional eval hook with
best-by-mAP tracking, optional gradient checkpointing.  `overfit_one_batch` is the mandatory
sanity check before any long run.
"""
from __future__ import annotations

import contextlib

import torch
from torch.optim import AdamW

from ..losses import filip_loss, siglip_sigmoid_loss
from ..models.ema import ModelEMA
from .sched import cosine_warmup


class SiglipTrainer:
    def __init__(self, model, cfg, device: str = "cpu"):
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        o = cfg.optim

        params = [p for p in self.model.parameters() if p.requires_grad]
        self.opt = AdamW(
            params, lr=float(o.get_path("lr", 5e-5)),
            weight_decay=float(o.get_path("wd", 0.05)), betas=(0.9, 0.98),
        )
        self.grad_clip = float(o.get_path("grad_clip", 1.0))
        self.use_filip = bool(o.get_path("use_filip", False))
        self.w_filip = float(o.get_path("w_filip", 0.5))
        self.filip_chunk = int(o.get_path("filip_chunk", 0))  # >0 caps FILIP [B,B,Ni,Nt] memory
        self.warmup_epochs = float(o.get_path("warmup_epochs", 1.0))
        self.sched = None
        self.best_state = None

        self.ema = ModelEMA(self.model, float(o.get_path("ema_decay", 0.999))) if bool(
            o.get_path("ema", True)) else None
        self.use_amp = device == "cuda" and bool(o.get_path("bf16", True))
        if device == "cuda" and bool(o.get_path("grad_checkpoint", True)) and hasattr(self.model, "hf"):
            with contextlib.suppress(Exception):
                self.model.hf.gradient_checkpointing_enable()

    def _amp(self):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16) if self.use_amp else contextlib.nullcontext()

    def compute_loss(self, batch):
        out = self.model(
            batch["pixel_values"].to(self.device),
            batch["input_ids"].to(self.device),
            batch["attention_mask"].to(self.device),
        )
        loss = siglip_sigmoid_loss(out["image_feat"], out["text_feat"], out["logit_scale"], out["logit_bias"])
        logs = {"sigmoid": float(loss.detach())}
        if self.use_filip:
            lf = filip_loss(out["image_tokens"], out["text_tokens"], out["logit_scale"],
                            batch["attention_mask"].to(self.device), chunk=self.filip_chunk)
            loss = loss + self.w_filip * lf
            logs["filip"] = float(lf.detach())
        logs["total"] = float(loss.detach())
        return loss, logs

    def train_step(self, batch):
        self.model.train()
        self.opt.zero_grad(set_to_none=True)
        with self._amp():
            loss, logs = self.compute_loss(batch)
        loss.backward()
        if self.grad_clip > 0:
            gn = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            logs["grad_norm"] = float(gn)
        self.opt.step()
        if self.sched is not None:
            self.sched.step()
        if self.ema is not None:
            self.ema.update(self.model)
        return logs

    def overfit_one_batch(self, batch, steps: int = 50):
        return [self.train_step(batch)["total"] for _ in range(steps)]

    def fit(self, loader, epochs: int = 1, log_every: int = 50, on_log=None,
            eval_fn=None, eval_every: int = 0, on_eval=None):
        """Train; if eval_fn is given, call it every `eval_every` steps and track best mAP.
        eval_fn(model) -> metrics dict (must contain 'mAP').  Returns best metrics."""
        steps_per_epoch = len(loader)
        self.sched = cosine_warmup(
            self.opt, int(self.warmup_epochs * steps_per_epoch), epochs * steps_per_epoch
        )
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
