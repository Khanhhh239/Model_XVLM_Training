# SAFE 100K fine-tune of the 80-mAP CMP checkpoint

Goal: sharpen the existing **80-mAP-on-distractor** CMP checkpoint with **genuinely hard negatives +
sim2real augmentation + pose + an anomaly head**, using **LoRA on the encoders but FULL fine-tune on
the cross-encoder**, WITHOUT dropping below 80 mAP. Expect a **modest, SAFE +1–4 mAP** (the mined
hard-neg is the main lever). Bigger jumps (→ ~89 R@1) need the BEiT-3/HUI rebuild — that's a separate plan.

## What each file does
| File | Role |
|---|---|
| `sample_100k.py` | balanced 50/50 normal/anomaly 100K subset, keeps `hard_i`, tags `anomaly` |
| `augment_ft.py` | sim2real degradation (motion/gauss blur, JPEG, downscale) for #9 + erasing #8; pose kept clean |
| `dataset_ft.py` | loads image+pose (joint flip) + hard_i + mined_i + anomaly label |
| `mine_negatives.py` | ANCE: each epoch, hardest **cross-ID** confuser per sample (the distractor lever) |
| `model_ft.py` | `SearchFT` = CMP + anomaly head + mined-neg ITM loss; LoRA-enc / full-cross helpers |
| `eval_distractor.py` | distractor-val mAP/R@k (the safety gauge) |
| `train_ft.py` | warm-start → train → eval every N → **save BEST → never ship < baseline** |
| `config_ft.yaml` | all paths/hparams |

## Run (inside a cloned CMP repo, A100 recommended)
```bash
git clone https://github.com/Shuyu-XJTU/CMP && cd CMP
# 1) put PAB under data/PAB (OneDrive/Baidu links in CMP README); place cmp.pth in checkpoint/
# 2) copy this folder in:
cp -r /path/finetune100k .
pip install peft scipy                       # extras beyond CMP's deps
# 3) sample 100K
python finetune100k/sample_100k.py --ann-glob "data/PAB/annotation/train/attr_*.json" \
       --out data/PAB/annotation/train/attr_100k.json --n 100000
# 4) train (warm-start the 80-mAP ckpt)
python finetune100k/train_ft.py --config finetune100k/config_ft.yaml \
       --checkpoint checkpoint/cmp.pth --out out/ft100k
```

## The safety protocol (the "không để tụt" guarantee)
1. `train_ft` measures the **baseline** distractor-val at step 0 (should read ≈ your 80).
2. It evaluates every `eval_every` steps with EMA weights and **saves ONLY when mAP beats the running best**.
3. At the end it prints `baseline → best (delta)`. **If best did NOT beat baseline → it tells you to SHIP
   THE ORIGINAL `cmp.pth`** (the fine-tune is discarded). You can never end up worse than 80.

## Time (1×A100-40GB, 100K, batch 32)
~45–60 min/epoch + ~10–15 min/epoch mining → **~4–5 h for 3 epochs**. (T4 ≈ 2–3× → ~10–14 h; resumable
by re-running — best checkpoint is on disk.)

## Ablate (keep only what pays — measured on distractor-val)
Toggle in `config_ft.yaml` and compare best-mAP: `use_mined_neg`, `w_anom` (0 vs 0.2),
`sim2real_p` (0 vs 1), `be_pose_img`, `use_lora` (LoRA vs freeze). Keep a component only if it raises mAP.

## ⚠️ First-run notes (this code can't be unit-tested without PAB+GPU — expect 1–2 small fixes)
- `vision_width` (anomaly-head input) is set to 1024 (swin-base). If the CLS dim differs, fix it in config.
- LoRA target-module names are matched by substring (`query/key/value/attn`). If `apply_lora_keep_cross`
  finds 0 targets it auto-falls back to `freeze_encoders` (still trains cross/heads — safe). Verify the
  printed "trainable params %" looks sane (a few % for LoRA, larger for freeze-mode cross-FT).
- `sample_100k.is_anomaly()` parses the attribute; if your attr schema differs, adjust the key list.
- If OOM: lower `batch_size` to 16/24; pose + hard + mined = up to ~4 image forwards/step.
