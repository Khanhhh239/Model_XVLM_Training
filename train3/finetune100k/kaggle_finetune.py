# =====================================================================================
# Kaggle notebook — SAFE 100K fine-tune of the 80-mAP CMP checkpoint (T4/P100)
# Clones CMP + this repo's finetune100k, patches bert for modern transformers, pins transformers 4.44.2,
# auto-detects cmp.pth/bert, writes a Kaggle config (fp16, small batch), runs sample + train (resumable).
# READ README_KAGGLE_FT.md FIRST — you must upload the PAB-100K data as a Kaggle dataset.
# Paste each CELL into a Kaggle notebook (GPU T4, Internet ON).
# =====================================================================================

# ============================== CELL 0 — setup (clone + patch + deps + detect) ==============================
import os, sys, glob, subprocess
WORK = "/kaggle/working"
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "transformers==4.44.2", "peft", "scipy", "ruamel.yaml", "timm"], check=False)
for url, dst in [("https://github.com/Shuyu-XJTU/CMP", f"{WORK}/CMP"),
                 ("https://github.com/Khanhhh239/Model_XVLM_Training", f"{WORK}/MXT")]:
    if not os.path.isdir(dst):
        subprocess.run(["git", "clone", "--depth", "1", url, dst], check=False)
subprocess.run(["cp", "-r", f"{WORK}/MXT/train3/finetune100k", f"{WORK}/CMP/finetune100k"], check=False)

# patch CMP models/bert.py for modern transformers (same fix as the inference notebook)
_bp = f"{WORK}/CMP/models/bert.py"
_s = open(_bp, encoding="utf-8").read()
_s = _s.replace("from transformers.file_utils import", "from transformers.utils import")
_s = _s.replace(
    "from transformers.modeling_utils import (\n    PreTrainedModel,\n    apply_chunking_to_forward,\n    find_pruneable_heads_and_indices,\n    prune_linear_layer,\n)",
    "from transformers.modeling_utils import PreTrainedModel\n"
    "from transformers.pytorch_utils import apply_chunking_to_forward\n"
    "try:\n    from transformers.pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer\n"
    "except ImportError:\n"
    "    def find_pruneable_heads_and_indices(*a, **k): raise NotImplementedError\n"
    "    def prune_linear_layer(*a, **k): raise NotImplementedError")
open(_bp, "w", encoding="utf-8").write(_s)
print("patched bert.py")

def _find(pat):
    h = sorted(glob.glob(f"/kaggle/input/**/{pat}", recursive=True)); return h[0] if h else None

CKPT = _find("cmp.pth")
VOCAB = _find("vocab.txt")
BERT_DIR = os.path.dirname(VOCAB) if VOCAB else None
print("CKPT:", CKPT, "| BERT_DIR:", BERT_DIR)


# ============================== CELL 1 — paths & config (EDIT to match your datasets) ==============================
# After uploading PAB-100K (see README_KAGGLE_FT.md), set these to where Kaggle mounted them.
# image_root must be the folder so that image_root + ann['image'] is a real file (ann['image'] like 'train/imgs_0/goal/0.jpg').
IMAGE_ROOT   = _find("train") and os.path.dirname(_find("attr_100k.json") or _find("attr_0.json") or "") or "/kaggle/input/pab-100k"
TRAIN_ANN    = _find("attr_100k.json") or _find("attr_0.json")            # pre-sampled 100K, or raw attr_*.json
VAL_DIR      = os.path.dirname(_find("attr.json") or "/kaggle/input/aicity-official-test/old_test-set/attr.json")  # old labeled test
VAL_GALLERY  = (_find("query_text.json") and os.path.join(os.path.dirname(_find("query_text.json")), "gallery")) or "/kaggle/input/aicity-official-test/name-masked_test-set/gallery"
print("IMAGE_ROOT:", IMAGE_ROOT, "\nTRAIN_ANN:", TRAIN_ANN, "\nVAL_DIR:", VAL_DIR, "\nVAL_GALLERY:", VAL_GALLERY)

CFG = f"""
vision_config: '{WORK}/CMP/configs/config_swinB.json'
text_config:   '{WORK}/CMP/configs/config_bert.json'
text_encoder:  '{BERT_DIR}'
load_pretrained: True
h: 224
w: 224
embed_dim: 2048
vision_width: 1024
temp: 0.07
label_smooth: 0.2
itc_dp: 0.5
max_words: 56
max_tokens: 56
mask_prob: 0.25
max_masks: 10
skipgram_prb: 0.2
skipgram_size: 3
mask_whole_word: True
eda_p: 1
be_hard: True
be_pose_img: True
pose_conv: True
use_lora: True
lora_r: 16
use_mined_neg: True
mine_topk: 10
w_anom: 0.2
sim2real_p: 1.0
erasing_p: 0.5
epochs: 2
batch_size: 16
lr_enc: 5.0e-5
lr_head: 2.0e-5
ema: 0.999
eval_every: 400
image_root: '{IMAGE_ROOT}/'
train_file_subset: '{WORK}/attr_100k.json'
val_dir: '{VAL_DIR}'
val_gallery_dir: '{VAL_GALLERY}'
val_n_distract: 5000
"""
open(f"{WORK}/config_kaggle.yaml", "w").write(CFG)
print(CFG)


# ============================== CELL 2 — sample 100K (skip if you uploaded attr_100k.json) ==============================
if not os.path.exists(f"{WORK}/attr_100k.json"):
    if TRAIN_ANN and TRAIN_ANN.endswith("attr_100k.json"):
        subprocess.run(["cp", TRAIN_ANN, f"{WORK}/attr_100k.json"], check=False)
    else:
        subprocess.run([sys.executable, "finetune100k/sample_100k.py",
                        "--ann-glob", os.path.join(os.path.dirname(TRAIN_ANN or ""), "attr_*.json"),
                        "--out", f"{WORK}/attr_100k.json", "--n", "100000"],
                       cwd=f"{WORK}/CMP", check=False)
print("subset rows:", __import__("json").load(open(f"{WORK}/attr_100k.json")).__len__() if os.path.exists(f"{WORK}/attr_100k.json") else "MISSING")


# ============================== CELL 3 — train (resumable; ~6-9h on T4 for 2 epochs) ==============================
# Re-running this cell resumes from /kaggle/working/out/last.pth (saved every eval).
# Save Version keeps out/ as output; to resume in a NEW session, add this notebook's output as input + copy to out/.
subprocess.run([sys.executable, "finetune100k/train_ft.py",
                "--config", f"{WORK}/config_kaggle.yaml",
                "--checkpoint", CKPT,
                "--out", f"{WORK}/out"],
               cwd=f"{WORK}/CMP", check=False)
print("DONE. Best checkpoint (if it beat baseline): /kaggle/working/out/checkpoint_best.pth")
