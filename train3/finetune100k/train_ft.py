"""SAFE 100K fine-tune of the 80-mAP CMP checkpoint.
  LoRA encoders + FULL-FT cross/heads/pose · CMP-hard + MINED cross-ID hard negs · anomaly head ·
  sim2real aug · pose · distractor-val every N steps -> save BEST · NEVER ship below the 80 baseline.

Run inside a cloned CMP repo (so `models`, `dataset`, `train`, `utils` import):
  python finetune100k/train_ft.py --config finetune100k/config_ft.yaml \
     --checkpoint checkpoint/cmp.pth --out out/ft100k
"""
import argparse, copy, json, math, os, sys, time
import torch
from torch.utils.data import DataLoader
from ruamel.yaml import YAML
from transformers import BertTokenizer

from models.model_search import Search  # noqa: F401 (ensures CMP importable)
from dataset.search_dataset import TextMaskingGenerator
from train import mlm as cmp_mlm
sys.path.insert(0, os.path.dirname(__file__))
from model_ft import SearchFT, apply_lora_keep_cross, freeze_encoders_keep_cross
from dataset_ft import FTTrainDataset, collate
from augment_ft import build_ft_image_transform, build_pose_transform
import mine_negatives, eval_distractor

yaml = YAML(typ="safe")


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    def update(self, model):
        for n, p in model.named_parameters():
            if n in self.shadow: self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)
    def swap_in(self, model):
        self.backup = {n: p.detach().clone() for n, p in model.named_parameters() if n in self.shadow}
        for n, p in model.named_parameters():
            if n in self.shadow: p.data.copy_(self.shadow[n])
    def swap_out(self, model):
        for n, p in model.named_parameters():
            if n in self.backup: p.data.copy_(self.backup[n])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)        # the 80-mAP cmp.pth
    ap.add_argument("--out", default="out/ft100k")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    cfg = yaml.load(open(args.config))
    device = "cuda"
    tok = BertTokenizer.from_pretrained(cfg["text_encoder"])

    # ---- model: warm-start the 80-mAP checkpoint, then LoRA-enc + full-cross ----
    model = SearchFT(cfg)
    try: model.load_pretrained(args.checkpoint)
    except Exception as e:
        print("load_pretrained fallback:", e)
        sd = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(sd.get("model", sd), strict=False)
    model = (apply_lora_keep_cross(model, r=cfg.get("lora_r", 16))
             if cfg.get("use_lora", True) else freeze_encoders_keep_cross(model))
    model = model.to(device)

    # ---- data ----
    anns = json.load(open(cfg["train_file_subset"]))
    if cfg.get("be_hard", True):
        anns = [a for a in anns if a.get("hard_i")]      # keep shapes uniform for hard-neg loss
    print("train rows:", len(anns))
    img_tf = build_ft_image_transform(cfg["h"], cfg["w"], cfg.get("erasing_p", 0.5),
                                      cfg.get("sim2real_p", 1.0))
    pose_tf = build_pose_transform(cfg["h"], cfg["w"])
    mask_gen = TextMaskingGenerator(tok, cfg["mask_prob"], cfg["max_masks"],
                                    cfg["skipgram_prb"], cfg["skipgram_size"], cfg["mask_whole_word"])

    # ---- distractor-val (the safety gauge) ----
    import zipfile, glob  # gallery may be a folder or unzipped already
    gal_dir = cfg["val_gallery_dir"]; gal_names = sorted(os.listdir(gal_dir))[: cfg.get("val_n_distract", 5000)]
    qcaps, gpaths, qgid = eval_distractor.build_val(cfg["val_dir"], gal_dir, gal_names,
                                                    cfg.get("val_n_distract", 5000))
    def evaluate():
        return eval_distractor.evaluate(model, tok, pose_tf, device, cfg["max_tokens"], qcaps, gpaths, qgid)

    baseline = evaluate(); print("BASELINE distractor-val:", baseline)
    base_map = baseline["mAP"]

    # ---- optimizer: low LR everywhere (don't degrade) ----
    enc, head = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        (head if any(k in n.lower() for k in ("crossattention", "itm_head", "anomaly_head", "pose", "proj")) else enc).append(p)
    opt = torch.optim.AdamW([{"params": enc, "lr": cfg.get("lr_enc", 5e-5)},
                             {"params": head, "lr": cfg.get("lr_head", 2e-5)}], weight_decay=0.01)
    ema = EMA(model, cfg.get("ema", 0.999))

    # AMP: bf16 on Ampere+ (A100); fp16+GradScaler on Kaggle T4/P100 (no usable bf16)
    bf16 = torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if bf16 else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=not bf16)
    print("AMP:", "bf16" if bf16 else "fp16 + GradScaler (T4/P100)")

    epochs = cfg.get("epochs", 3); bs = cfg.get("batch_size", 32)
    steps_total = epochs * math.ceil(len(anns) / bs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps_total)
    eval_every = cfg.get("eval_every", 500)
    best = base_map; best_path = os.path.join(args.out, "checkpoint_best.pth")
    last_path = os.path.join(args.out, "last.pth"); step = 0; start_ep = 0
    # RESUME across Kaggle sessions (re-run the cell; needs out/ persisted, e.g. saved as notebook output)
    if os.path.exists(last_path):
        ck = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        step, best, start_ep = ck["step"], ck["best"], ck["epoch"]
        ema.shadow = {k: v.to(device) for k, v in ck["ema"].items()}
        print(f"RESUMED {last_path}: step{step} best{best:.4f} epoch{start_ep}")
    t0 = time.time()

    for ep in range(start_ep, epochs):
        print(f"=== epoch {ep}: mining cross-ID hard negatives ===")
        mine_negatives.mine(model, anns, cfg["image_root"], pose_tf, tok, device, cfg["max_tokens"],
                            topk=cfg.get("mine_topk", 10))
        loader = DataLoader(FTTrainDataset(anns, cfg["image_root"], img_tf, pose_tf, cfg),
                            batch_size=bs, shuffle=True, num_workers=4, collate_fn=collate, drop_last=True)
        model.train()
        for b in loader:
            def tk(texts):
                t = tok(texts, padding="max_length", truncation=True, max_length=cfg["max_tokens"], return_tensors="pt").to(device)
                return t.input_ids, t.attention_mask
            ti = tok(b["caption"], padding="max_length", truncation=True, max_length=cfg["max_tokens"], return_tensors="pt").to(device)
            tim, mp, mi_ = cmp_mlm(b["caption"], ti, tok, device, mask_gen, cfg)
            eda_ids, eda_atts = tk(b["caption_eda"])
            h_ids, h_atts = tk(b["hard_caption"]) if b["hard_i"] is not None else (None, None)
            m_ids, m_atts = tk(b["mined_caption"]) if b["mined_i"] is not None else (None, None)
            be_pose = cfg.get("be_pose_img", True)
            with torch.autocast("cuda", dtype=amp_dtype):
                litc, litm, lmlm, lanom = model(
                    b["image"].to(device), ti.input_ids, ti.attention_mask,
                    text_ids_masked=tim, masked_pos=mp, masked_ids=mi_, idx=b["idx"].to(device),
                    text_ids_eda=eda_ids, text_atts_eda=eda_atts,
                    pose=b["pose"].to(device) if be_pose and b["pose"] is not None else None,
                    hard_i=b["hard_i"].to(device) if b["hard_i"] is not None else None,
                    hard_i_pose=b["hard_pose"].to(device) if be_pose and b["hard_pose"] is not None else None,
                    hard_text_ids=h_ids, hard_text_atts=h_atts,
                    anomaly_label=b["anomaly"].to(device),
                    mined_i=b["mined_i"].to(device) if b["mined_i"] is not None else None,
                    mined_i_pose=b["mined_pose"].to(device) if be_pose and b["mined_pose"] is not None else None,
                    mined_text_ids=m_ids, mined_text_atts=m_atts)
                loss = litc + litm + lmlm + lanom
            scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update(); sched.step(); opt.zero_grad(); ema.update(model); step += 1

            if step % 50 == 0:
                print(f"ep{ep} step{step}/{steps_total} loss {loss.item():.3f} "
                      f"(itc {litc.item():.2f} itm {litm.item():.2f} mlm {lmlm.item():.2f} anom {lanom.item():.3f}) "
                      f"{(time.time()-t0)/60:.1f}min")
            if step % eval_every == 0:
                ema.swap_in(model); m = evaluate(); ema.swap_out(model); model.train()
                tag = "BEST" if m["mAP"] > best else "  "
                print(f"[val step{step}] mAP {m['mAP']:.4f} R@1 {m['R@1']:.4f}  best {best:.4f} {tag}")
                if m["mAP"] > best:
                    best = m["mAP"]; ema.swap_in(model)
                    torch.save({"model": model.state_dict(), "config": cfg, "val": m}, best_path)
                    ema.swap_out(model)
                # resume checkpoint (every eval) so a Kaggle session cutoff can continue
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                            "ema": ema.shadow, "step": step, "best": best, "epoch": ep}, last_path)

    print(f"\n=== DONE in {(time.time()-t0)/60:.1f} min ===")
    print(f"baseline mAP {base_map:.4f} -> best mAP {best:.4f}  (delta {best-base_map:+.4f})")
    if best <= base_map + 1e-4:
        print("ABORT: fine-tune did NOT beat the baseline -> SHIP THE ORIGINAL cmp.pth, not this run.")
    else:
        print(f"SHIP {best_path} (beats baseline). Encode the 36K gallery with it, then submit.")


if __name__ == "__main__":
    main()
