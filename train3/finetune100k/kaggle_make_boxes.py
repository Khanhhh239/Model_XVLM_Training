# =====================================================================================
# Kaggle notebook — phrase-grounded BOUNDING BOXES for the 30K hard subset (one-time preprocessing)
# Reads data team's data/subsets/train_30k_hard.jsonl -> extracts noun phrases (with modifiers, e.g.
# "a white rugby ball", "a girl in a shirt") from each caption -> GroundingDINO grounds them on the
# 384x384 webp image -> saves boxes_30k.jsonl. STANDALONE (no CMP) -> modern transformers, no clash.
# Output feeds the box-grounding head later. Resumable. Kaggle GPU T4, Internet ON.
# =====================================================================================

# ============================== CELL 0 — setup + EXTRACT .tar.zst + detect data ==============================
import os, sys, json, glob, time, subprocess
import torch
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "transformers>=4.45", "spacy", "pillow", "zstandard"], check=False)
subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=False)
WORK = "/kaggle/working"
device = "cuda" if torch.cuda.is_available() else "cpu"
EXT = f"{WORK}/ext"
OUT = f"{WORK}/boxes_30k.jsonl"

# ---- extract the dataset archive (input is train_30k_hard_data.tar.zst, NOT unpacked) ----
def _g1(pat):
    h = sorted(glob.glob(pat, recursive=True)); return h[0] if h else None

if not glob.glob(f"{EXT}/**/train_webp", recursive=True):
    zst = _g1("/kaggle/input/**/*.tar.zst")
    if zst:
        print("extracting", zst, "(train_webp + subsets + annotation only) ...")
        import zstandard, tarfile
        os.makedirs(EXT, exist_ok=True)
        keep = ("/train_webp/", "train_30k_hard.jsonl", "/subsets/", "/annotation/")
        t0 = time.time()
        with open(zst, "rb") as fh, zstandard.ZstdDecompressor().stream_reader(fh) as r:
            with tarfile.open(fileobj=r, mode="r|") as tar:
                for m in tar:
                    if m.isfile() and any(k in m.name for k in keep):
                        tar.extract(m, EXT)
        print(f"extracted in {(time.time()-t0)/60:.1f} min")
    else:
        print("WARN: no .tar.zst found under /kaggle/input — check the dataset is added")

def _find(pat):                                   # search the EXTRACTED tree (fallback raw input)
    for root in (EXT, "/kaggle/input"):
        h = sorted(glob.glob(f"{root}/**/{pat}", recursive=True))
        if h: return h[0]
    return None

JSONL = _find("train_30k_hard.jsonl")
WEBP_ROOT = None
for root in (EXT, "/kaggle/input"):
    for c in glob.glob(f"{root}/**/train_webp", recursive=True):
        if os.path.isdir(c): WEBP_ROOT = c; break
    if WEBP_ROOT: break
print("subset jsonl:", JSONL, "\ntrain_webp:", WEBP_ROOT, "\nout:", OUT)


# ============================== CELL 1 — load GroundingDINO + spaCy noun-phrase extractor ==============================
from PIL import Image
import spacy
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

nlp = spacy.load("en_core_web_sm")
GD = "IDEA-Research/grounding-dino-tiny"            # fast; use -base for accuracy (slower)
proc = AutoProcessor.from_pretrained(GD)
gdino = AutoModelForZeroShotObjectDetection.from_pretrained(GD).to(device).eval()

_DROP = {"image", "images", "background", "foreground", "scene", "view", "picture", "photo",
         "right", "left", "side", "distance", "front", "color", "colors", "appearance"}

def caption_phrases(cap, max_phrases=10):
    """Concrete noun phrases WITH modifiers (adj/color + noun), drop abstract/scene words."""
    doc = nlp(cap)
    out = []
    for ch in doc.noun_chunks:
        if ch.root.pos_ not in ("NOUN", "PROPN"):     # skip pronoun chunks ('they','it')
            continue
        head = ch.root.lemma_.lower()
        if head in _DROP:
            continue
        # strip leading determiner for a clean prompt phrase
        toks = [t.text for t in ch if t.pos_ != "DET"]
        phrase = " ".join(toks).strip().lower()
        if len(phrase) >= 3 and phrase not in out:
            out.append(phrase)
    return out[:max_phrases]

def img_path(rel):
    rel = rel[len("train/"):] if rel.startswith("train/") else rel    # 'train/imgs_0/..' -> 'imgs_0/..'
    return os.path.join(WEBP_ROOT, rel)

@torch.no_grad()
def ground(image, phrases, box_thr=0.30, txt_thr=0.22):
    prompt = " . ".join(phrases) + " ."
    inp = proc(images=image, text=prompt, return_tensors="pt").to(device)
    out = gdino(**inp)
    res = proc.post_process_grounded_object_detection(
        out, inp.input_ids, box_threshold=box_thr, text_threshold=txt_thr,
        target_sizes=[image.size[::-1]])[0]
    boxes = []
    for b, s, lab in zip(res["boxes"].tolist(), res["scores"].tolist(), res["labels"]):
        boxes.append({"phrase": lab, "box": [round(x, 1) for x in b], "score": round(float(s), 3)})
    return boxes

print("loaded GroundingDINO + spaCy. quick test:")
_rows = [json.loads(l) for l in open(JSONL, encoding="utf-8")]
print("rows:", len(_rows), "| phrases ex:", caption_phrases(_rows[0]["caption"]))


# ============================== CELL 2 — run on 30K (resumable) + time ==============================
done = set()
if os.path.exists(OUT):
    done = {json.loads(l)["image_id"] for l in open(OUT, encoding="utf-8")}
    print("resume: already done", len(done))

t0 = time.time(); n = 0
with open(OUT, "a", encoding="utf-8") as f:
    for r in _rows:
        iid = r.get("image_id")
        if iid in done:
            continue
        p = img_path(r["image"])
        if not os.path.exists(p):
            continue
        try:
            img = Image.open(p).convert("RGB")
            ph = caption_phrases(r["caption"])
            bxs = ground(img, ph) if ph else []
        except Exception as ex:
            bxs = []; print("err", iid, ex)
        f.write(json.dumps({"image_id": iid, "image": r["image"], "boxes": bxs}, ensure_ascii=False) + "\n")
        n += 1
        if n % 200 == 0:
            f.flush()
            rate = n / (time.time() - t0)
            print(f"{n} done | {rate:.1f} img/s | ETA {(len(_rows)-len(done)-n)/max(rate,1e-6)/3600:.1f}h")
print(f"DONE {n} images in {(time.time()-t0)/60:.1f} min -> {OUT}")
# boxes_30k.jsonl: {image_id, image, boxes:[{phrase, box[x1,y1,x2,y2], score}]}  -> upload as a dataset for the box head


# ============================== CELL 3 — VISUALIZE ~50 images with boxes + caption nouns ==============================
import math
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

K = 50
results = []
if os.path.exists(OUT):                                  # use generated boxes
    for l in open(OUT, encoding="utf-8"):
        results.append(json.loads(l))
        if len(results) >= K: break
else:                                                    # not generated yet -> compute K live (preview)
    for r in _rows[:K]:
        p = img_path(r["image"])
        if not os.path.exists(p): continue
        img = Image.open(p).convert("RGB")
        results.append({"image": r["image"], "boxes": ground(img, caption_phrases(r["caption"]))})

COLORS = ["#FF3838", "#39FF14", "#00E5FF", "#FFD400", "#FF00FF", "#FF8C00", "#FFFFFF", "#7CFC00"]
cols = 5
rows = math.ceil(len(results) / cols)
plt.figure(figsize=(cols * 3.4, rows * 3.6))
for i, res in enumerate(results):
    p = img_path(res["image"])
    if not os.path.exists(p): continue
    img = Image.open(p).convert("RGB"); dr = ImageDraw.Draw(img)
    for j, b in enumerate(res["boxes"]):
        x1, y1, x2, y2 = b["box"]; c = COLORS[j % len(COLORS)]
        dr.rectangle([x1, y1, x2, y2], outline=c, width=2)
        dr.text((x1 + 2, max(0, y1 + 1)), f'{b["phrase"]} {b["score"]:.2f}', fill=c)
    ax = plt.subplot(rows, cols, i + 1); ax.imshow(img); ax.axis("off")
    ax.set_title(" | ".join(b["phrase"] for b in res["boxes"][:4]) or "(no box)", fontsize=6)
plt.tight_layout(); plt.savefig(f"{WORK}/viz_boxes.png", dpi=120, bbox_inches="tight")
plt.show()
print(f"saved {WORK}/viz_boxes.png  ({len(results)} images;  "
      f"avg boxes/img = {sum(len(r['boxes']) for r in results)/max(len(results),1):.1f})")
