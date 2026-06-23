# =====================================================================================
# Kaggle notebook — phrase-grounded BOUNDING BOXES for the 30K hard subset (one-time preprocessing)
# Reads data team's data/subsets/train_30k_hard.jsonl -> extracts noun phrases (with modifiers, e.g.
# "a white rugby ball", "a girl in a shirt") from each caption -> GroundingDINO grounds them on the
# 384x384 webp image -> saves boxes_30k.jsonl. STANDALONE (no CMP) -> modern transformers, no clash.
# Output feeds the box-grounding head later. Resumable. Kaggle GPU T4, Internet ON.
# =====================================================================================

# ============================== CELL 0 — setup + detect data ==============================
import os, sys, json, glob, time, subprocess
import torch
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "transformers>=4.45", "spacy", "pillow"], check=False)
subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=False)
WORK = "/kaggle/working"
device = "cuda" if torch.cuda.is_available() else "cpu"

def _find(pat):
    h = sorted(glob.glob(f"/kaggle/input/**/{pat}", recursive=True)); return h[0] if h else None

JSONL = _find("train_30k_hard.jsonl")
WEBP_ROOT = None
for c in glob.glob("/kaggle/input/**/train_webp", recursive=True):
    if os.path.isdir(c): WEBP_ROOT = c; break
OUT = f"{WORK}/boxes_30k.jsonl"
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
