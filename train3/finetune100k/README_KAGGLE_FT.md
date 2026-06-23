# Run the 100K CMP fine-tune on Kaggle (T4)

⚠️ **Reality check:** the hard part is DATA, not code. PAB train is 1M images on OneDrive/Baidu (no
HuggingFace). You must stage a ~100K subset **once** and upload it as a Kaggle dataset (~10–25 GB).
Kaggle GPU = T4/P100 (no bf16 → the trainer auto-uses fp16) with a ~12 h/session limit → the run is
**resumable**. Honest expectation unchanged: **+1–4 mAP, never below the 80 baseline** (the trainer
aborts→ships-original if it doesn't beat 80).

## 1. Stage the 100K data ONCE (on a machine that downloaded PAB)
```bash
git clone https://github.com/Shuyu-XJTU/CMP && cd CMP    # get PAB into data/PAB (OneDrive links in its README)
# sample a balanced 100K (keeps hard_i, tags anomaly)
python /path/finetune100k/sample_100k.py --ann-glob "data/PAB/annotation/train/attr_*.json" \
       --out attr_100k.json --n 100000
# copy ONLY the images+pose the subset references (image + hard_i), so the upload stays ~100-200K imgs
python - <<'PY'
import json, os, shutil
anns = json.load(open("attr_100k.json")); root="data/PAB"; out="pab100k"
need=set()
for a in anns:
    need.add(a["image"])
    if a.get("hard_i"): need.add("train/"+a["hard_i"])
for rel in need:
    for sub in (rel, "pose/"+rel):           # image + its pose map
        s=os.path.join(root,sub); d=os.path.join(out,sub)
        if os.path.exists(s): os.makedirs(os.path.dirname(d),exist_ok=True); shutil.copy(s,d)
shutil.copy("attr_100k.json", out+"/attr_100k.json")
print("staged ->", out)
PY
# zip pab100k/ and upload as Kaggle Dataset "pab-100k"
```
> The mined cross-ID negatives are picked from WITHIN the 100K each epoch, so no extra images are needed.

## 2. Kaggle datasets to add (Add Input)
| Dataset | Contents |
|---|---|
| `pab-100k` | `attr_100k.json` + `train/...` images + `pose/train/...` maps (from step 1) |
| `cmp-models` | `cmp.pth` (the 80-mAP ckpt) + `bert-base-uncased/` (or flat bert files) |
| `aicity-official-test` | `name-masked_test-set/gallery` (distractors) + old labeled test `attr.json`+images (for distractor-val) |

## 3. Run
- New Notebook → **GPU T4**, **Internet ON** → import `kaggle_finetune.ipynb` (this folder).
- **CELL 0**: clones CMP + this repo, patches bert, pins transformers 4.44.2, detects cmp.pth/bert.
- **CELL 1**: **CHECK the printed paths** (IMAGE_ROOT / TRAIN_ANN / VAL_DIR / VAL_GALLERY). Fix the
  variables if a path is wrong, re-run. `image_root` must make `image_root + ann['image']` a real file.
- **CELL 2**: makes `attr_100k.json` (or reuses your uploaded one).
- **CELL 3**: trains. Watch for `BASELINE distractor-val ≈ 0.80` (sanity), then `[val step…] mAP … best …`.

## 4. Resume (12h cutoff)
The trainer saves `out/last.pth` every eval. Re-running CELL 3 in the SAME session continues. To continue
in a NEW session: Save Version (keeps `out/`), then in the new notebook add this notebook's output as
input and `cp -r /kaggle/input/<prev-output>/out /kaggle/working/out` before CELL 3.

## 5. Time / memory (T4)
~1.5–2 h/epoch + ~20–30 min/epoch mining → **2 epochs ≈ 4–6 h** (fits one session). If OOM: lower
`batch_size` to 12 in CELL 1 (pose+hard+mined = up to ~4 image forwards/step on 16 GB).

## 6. After training
Best ckpt = `out/checkpoint_best.pth` (ONLY exists if it beat 80). Encode the 36K gallery with it
(swap it into the inference notebook's `CKPT`) → submit → compare vs the 80-mAP baseline. If CELL 3
prints `ABORT: … ship original`, keep using `cmp.pth` — the fine-tune didn't help on this subset.

## 7. First-run notes (can't be tested without PAB+GPU — expect 1–2 fixes)
- `vision_width=1024` (anomaly-head input). If a shape error mentions the anomaly head, set it to the real CLS dim.
- LoRA module-name match: CELL logs `trainable params %`. If ~0 targets, it auto-falls back to freeze-encoders (still safe).
- `sample_100k.is_anomaly()` parses the attribute schema; if anomaly came out all-0, fix the key list there.
- Path detection in CELL 1 is best-effort — verify the 4 printed paths before CELL 3.
