# =====================================================================================
# Kaggle T4 — STAGE 1 fine-tune from best.pth (80 mAP): freeze encoders + full-FT cross/heads.
# Extracts the data-team .tar.zst, renders pose from ViTPose keypoints, writes a config with detected
# paths, runs train_stage1 (resumable, revert-if-<80). Needs datasets: aicity-30k-hard-enhanced (the
# .tar.zst), cmp-models (cmp.pth + bert), and (optional) a boxes-30k dataset (boxes_30k.jsonl from
# kaggle_make_boxes). GPU T4, Internet ON. Paste each CELL.
# =====================================================================================

# ============================== CELL 0 — setup + extract + detect ==============================
import os, sys, glob, json, time, subprocess
import torch
assert torch.cuda.is_available(), "Enable GPU T4 (Settings -> Accelerator)."
WORK = "/kaggle/working"
subprocess.run([sys.executable,"-m","pip","install","-q","transformers==4.44.2","peft","scipy",
                "ruamel.yaml","timm","opencv-python-headless"], check=False)
for url,dst in [("https://github.com/Shuyu-XJTU/CMP",f"{WORK}/CMP"),
                ("https://github.com/Khanhhh239/Model_XVLM_Training",f"{WORK}/MXT")]:
    if not os.path.isdir(dst): subprocess.run(["git","clone","--depth","1",url,dst], check=False)
subprocess.run(["cp","-r",f"{WORK}/MXT/train3/finetune100k",f"{WORK}/CMP/finetune100k"], check=False)
# patch CMP bert.py for modern transformers
_bp=f"{WORK}/CMP/models/bert.py"; _s=open(_bp,encoding="utf-8").read()
_s=_s.replace("from transformers.file_utils import","from transformers.utils import")
_s=_s.replace("from transformers.modeling_utils import (\n    PreTrainedModel,\n    apply_chunking_to_forward,\n    find_pruneable_heads_and_indices,\n    prune_linear_layer,\n)",
              "from transformers.modeling_utils import PreTrainedModel\nfrom transformers.pytorch_utils import apply_chunking_to_forward\ntry:\n    from transformers.pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer\nexcept ImportError:\n    def find_pruneable_heads_and_indices(*a,**k): raise NotImplementedError\n    def prune_linear_layer(*a,**k): raise NotImplementedError")
open(_bp,"w",encoding="utf-8").write(_s); print("patched bert.py")

EXT=f"{WORK}/ext"
if not glob.glob(f"{EXT}/**/train_webp", recursive=True):
    zst=sorted(glob.glob("/kaggle/input/**/*.tar.zst",recursive=True))
    if zst:
        import zstandard, tarfile
        os.makedirs(EXT,exist_ok=True); keep=("/train_webp/","train_30k_hard.jsonl","/subsets/","vitpose")
        print("extracting", zst[0], "..."); t0=time.time()
        with open(zst[0],"rb") as fh, zstandard.ZstdDecompressor().stream_reader(fh) as r:
            with tarfile.open(fileobj=r, mode="r|") as tar:
                for m in tar:
                    if m.isfile() and any(k in m.name for k in keep): tar.extract(m, EXT)
        print(f"extracted {(time.time()-t0)/60:.1f}m")

def find1(pat, roots=(EXT,"/kaggle/input","/kaggle/working")):
    for rt in roots:
        h=sorted(glob.glob(f"{rt}/**/{pat}",recursive=True))
        if h: return h[0]
    return None
CKPT=find1("cmp.pth"); BERT=os.path.dirname(find1("vocab.txt") or "")
JSONL=find1("train_30k_hard.jsonl"); VITPOSE=find1("train_30k_hard_vitpose.json")
WEBP=None
for c in glob.glob(f"{EXT}/**/train_webp",recursive=True)+glob.glob("/kaggle/input/**/train_webp",recursive=True):
    if os.path.isdir(c): WEBP=c; break
BOXES=find1("boxes_30k.jsonl")     # optional (upload from kaggle_make_boxes); box loss auto-skips if None
print("CKPT",CKPT,"\nBERT",BERT,"\nJSONL",JSONL,"\nWEBP",WEBP,"\nVITPOSE",VITPOSE,"\nBOXES",BOXES)


# ============================== CELL 1 — render pose (ViTPose keypoints -> skeleton @224) ==============================
if VITPOSE and not os.path.isdir(f"{WORK}/pose"):
    subprocess.run([sys.executable, f"{WORK}/CMP/finetune100k/render_pose.py",
                    "--vitpose", VITPOSE, "--out", f"{WORK}/pose", "--size", "224"], check=False)
print("pose images:", len(glob.glob(f"{WORK}/pose/**/*.webp", recursive=True)))


# ============================== CELL 2 — write config with detected paths ==============================
VAL_DIR = os.path.dirname(find1("attr.json") or "/kaggle/input/aicity-official-test/old_test-set/attr.json")
GAL = find1("query_text.json"); VAL_GAL = os.path.join(os.path.dirname(GAL),"gallery") if GAL else \
      "/kaggle/input/aicity-official-test/name-masked_test-set/gallery"
cfg = f"""
vision_config: '{WORK}/CMP/configs/config_swinB.json'
text_config: '{WORK}/CMP/configs/config_bert.json'
text_encoder: '{BERT}'
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
be_hard: True
be_pose_img: {bool(VITPOSE)}
pose_conv: True
n_text_frozen: 6
freeze_vision: True
xbm_size: 8192
grad_ckpt: True
w_anom: 0.2
w_box: {0.1 if BOXES else 0.0}
box_p: 2
box_src_size: 384
use_mined_neg: True
mine_topk: 10
epochs: 2
batch_size: 16
lr_old: 2.0e-5
lr_new: 1.0e-4
ema: 0.999
eval_every: 400
erasing_p: 0.5
sim2real_p: 1.0
webp_root: '{WEBP}'
pose_root: '{WORK}/pose'
train_file_subset: '{JSONL}'
boxes_file: '{BOXES or ""}'
val_dir: '{VAL_DIR}'
val_gallery_dir: '{VAL_GAL}'
val_n_distract: 5000
"""
open(f"{WORK}/config_stage1.yaml","w").write(cfg); print(cfg)


# ============================== CELL 3 — train (resumable; ~2-3h on T4) ==============================
# Watch: 'BASELINE distractor-val' (should ~0.80) then '[val s..] mAP ..'. Saves out/checkpoint_best.pth.
subprocess.run([sys.executable, "finetune100k/train_stage1.py",
                "--config", f"{WORK}/config_stage1.yaml", "--checkpoint", CKPT, "--out", f"{WORK}/out_stage1"],
               cwd=f"{WORK}/CMP", check=False)
print("DONE -> /kaggle/working/out_stage1/checkpoint_best.pth (only if it beat 80)")
