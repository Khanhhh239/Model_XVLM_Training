"""STAGE-1 Kaggle-T4 trainer: freeze Swin + BERT-lower, full-FT fusion+heads, ITC(+XBM)+ITM(+ID-hard
+mined)+anomaly+box, drop MLM. fp16+GradScaler, distractor-val every N -> save BEST -> revert if <80,
resume via out/last.pth. Run inside a cloned CMP repo. Mine-once (frozen vision => gallery feats fixed)."""
import argparse, json, math, os, sys, time
import torch
from torch.utils.data import DataLoader
from ruamel.yaml import YAML
from transformers import BertTokenizer

from models.model_search import Search  # noqa
sys.path.insert(0, os.path.dirname(__file__))
from model_stage1 import SearchStage1
from dataset_stage1 import Stage1Dataset, collate, load_boxes
from augment_ft import build_ft_image_transform, build_pose_transform
import xbm as _xbm, eval_distractor, mine_negatives
yaml = YAML(typ="safe")


class EMA:
    def __init__(self, m, d=0.999):
        self.d = d; self.s = {n: p.detach().clone() for n, p in m.named_parameters() if p.requires_grad}
    def update(self, m):
        for n, p in m.named_parameters():
            if n in self.s: self.s[n].mul_(self.d).add_(p.detach(), alpha=1 - self.d)
    def swap_in(self, m):
        self.b = {n: p.detach().clone() for n, p in m.named_parameters() if n in self.s}
        for n, p in m.named_parameters():
            if n in self.s: p.data.copy_(self.s[n])
    def swap_out(self, m):
        for n, p in m.named_parameters():
            if n in self.b: p.data.copy_(self.b[n])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default="out/stage1")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    cfg = yaml.load(open(a.config)); dev = "cuda"
    tok = BertTokenizer.from_pretrained(cfg["text_encoder"])
    MAXT = cfg["max_tokens"]

    model = SearchStage1(cfg)
    try: model.load_pretrained(a.checkpoint)
    except Exception as e:
        print("load fallback:", e); sd = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(sd.get("model", sd), strict=False)
    model = model.to(dev)
    model.freeze_stage1(n_text_frozen=cfg.get("n_text_frozen", 6), freeze_vision=cfg.get("freeze_vision", True))
    model.xbm = _xbm.XBM(cfg["embed_dim"], cfg.get("xbm_size", 8192), dev)
    if cfg.get("grad_ckpt", True):
        try: model.text_encoder.bert.encoder.gradient_checkpointing = True
        except Exception: pass

    anns = [json.loads(l) for l in open(cfg["train_file_subset"], encoding="utf-8")]
    if cfg.get("be_hard", True): anns = [x for x in anns if x.get("hard_i")]
    boxes = load_boxes(cfg.get("boxes_file"), topk=cfg.get("box_p", 2))
    print(f"rows {len(anns)} | boxes {len(boxes)}")
    img_tf = build_ft_image_transform(cfg["h"], cfg["w"], cfg.get("erasing_p", 0.5), cfg.get("sim2real_p", 1.0))
    pose_tf = build_pose_transform(cfg["h"], cfg["w"])

    gal_dir = cfg["val_gallery_dir"]; gnames = sorted(os.listdir(gal_dir))[: cfg.get("val_n_distract", 5000)]
    qc, gp, qg = eval_distractor.build_val(cfg["val_dir"], gal_dir, gnames, cfg.get("val_n_distract", 5000))
    evaluate = lambda: eval_distractor.evaluate(model, tok, pose_tf, dev, MAXT, qc, gp, qg)
    base = evaluate(); print("BASELINE distractor-val:", base); base_map = base["mAP"]

    # mine-once cross-ID hard negatives (vision frozen -> features stable)
    if cfg.get("use_mined_neg", True):
        mine_negatives.mine(model, anns, cfg["webp_root"], img_tf, tok, dev, MAXT, topk=cfg.get("mine_topk", 10))

    # 3 LR groups: pretrained-trainable (low) vs new heads (higher)
    new_keys = ("box_head", "anomaly_head")
    g_new = [p for n, p in model.named_parameters() if p.requires_grad and any(k in n for k in new_keys)]
    g_old = [p for n, p in model.named_parameters() if p.requires_grad and not any(k in n for k in new_keys)]
    opt = torch.optim.AdamW([{"params": g_old, "lr": cfg.get("lr_old", 2e-5)},
                             {"params": g_new, "lr": cfg.get("lr_new", 1e-4)}], weight_decay=0.01)
    bf16 = torch.cuda.is_bf16_supported(); amp = torch.bfloat16 if bf16 else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=not bf16)
    ep_n = cfg.get("epochs", 3); bs = cfg.get("batch_size", 16)
    steps = ep_n * math.ceil(len(anns) / bs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    ema = EMA(model, cfg.get("ema", 0.999)); ev = cfg.get("eval_every", 400)
    best = base_map; bestp = f"{a.out}/checkpoint_best.pth"; lastp = f"{a.out}/last.pth"; step = 0; sep = 0
    if os.path.exists(lastp):
        ck = torch.load(lastp, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        step, best, sep = ck["step"], ck["best"], ck["epoch"]; ema.s = {k: v.to(dev) for k, v in ck["ema"].items()}
        print(f"RESUMED step{step} best{best:.4f}")
    t0 = time.time()

    def tk(texts):
        t = tok(texts, padding="max_length", truncation=True, max_length=MAXT, return_tensors="pt").to(dev)
        return t.input_ids, t.attention_mask

    for ep in range(sep, ep_n):
        ld = DataLoader(Stage1Dataset(anns, cfg["webp_root"], cfg.get("pose_root"), boxes, img_tf, pose_tf, cfg,
                                      P=cfg.get("box_p", 2)), batch_size=bs, shuffle=True, num_workers=4,
                        collate_fn=collate, drop_last=True)
        model.train()
        for b in ld:
            ti, ta = tk(b["caption"])
            hid, hat = tk(b["hard_caption"]) if b["hard_i"] is not None else (None, None)
            mid, mat = tk(b["mined_caption"]) if b["mined_i"] is not None else (None, None)
            P = cfg.get("box_p", 2); bxi, bxa = tk(b["box_phrases"]); L = bxi.shape[1]
            bxi = bxi.reshape(bs, P, L); bxa = bxa.reshape(bs, P, L)
            with torch.autocast("cuda", dtype=amp):
                out = model(b["image"].to(dev), ti, ta, idx=b["idx"].to(dev),
                            pose=b["pose"].to(dev) if b["pose"] is not None else None,
                            hard_i=b["hard_i"].to(dev) if b["hard_i"] is not None else None,
                            hard_pose=b["hard_pose"].to(dev) if b["hard_pose"] is not None else None,
                            hard_text_ids=hid, hard_text_atts=hat,
                            mined_i=b["mined_i"].to(dev) if b["mined_i"] is not None else None,
                            mined_pose=b["mined_pose"].to(dev) if b["mined_pose"] is not None else None,
                            mined_text_ids=mid, mined_text_atts=mat,
                            anomaly_label=b["anomaly"].to(dev),
                            box_text_ids=bxi, box_text_atts=bxa,
                            box_tgt=b["box_tgt"].to(dev), box_mask=b["box_mask"].to(dev))
                # aux warmup: ramp anom/box weight 0->1 over the first epoch
                w = min(1.0, step / max(1, math.ceil(len(anns) / bs)))
                loss = out["itc"] + out["itm"] + w * (out["anom"] + out["box"])
            scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.0)
            scaler.step(opt); scaler.update(); sched.step(); opt.zero_grad(); ema.update(model); step += 1
            if step % 50 == 0:
                print(f"ep{ep} s{step}/{steps} loss{loss.item():.3f} (itc{out['itc'].item():.2f} itm{out['itm'].item():.2f} "
                      f"anom{out['anom'].item():.3f} box{out['box'].item():.3f}) {(time.time()-t0)/60:.0f}m")
            if step % ev == 0:
                ema.swap_in(model); m = evaluate(); ema.swap_out(model); model.train()
                print(f"[val s{step}] mAP {m['mAP']:.4f} R@1 {m['R@1']:.4f} best {best:.4f}")
                if m["mAP"] > best:
                    best = m["mAP"]; ema.swap_in(model)
                    torch.save({"model": model.state_dict(), "config": cfg, "val": m}, bestp); ema.swap_out(model)
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                            "ema": ema.s, "step": step, "best": best, "epoch": ep}, lastp)
    print(f"\nDONE {(time.time()-t0)/60:.0f}m | baseline {base_map:.4f} -> best {best:.4f} ({best-base_map:+.4f})")
    print("SHIP " + bestp if best > base_map + 1e-4 else "ABORT: did not beat 80 -> ship original cmp.pth")


if __name__ == "__main__":
    main()
