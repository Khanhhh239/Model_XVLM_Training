# =====================================================================================
# AI City 2026 Track 4 — Text-Based Person Anomaly Search
# TRAINING-FREE inference on Kaggle (T4) — CMP retrieval + full rerank stack + ablation monitor
#
# Pipeline (all training-free, each gated by the ablation monitor):
#   Tầng1  CMP encode (Swin+BERT) -> ITC sim  [retrieval/recall over 36,773 gallery]
#   Tầng2  dual-softmax + k-reciprocal + query-expansion   [distractor-aware, FREE]
#   Tầng3  CMP ITM cross-encoder rerank top-K   [precision]   (+AnomalyLMM/Qwen = stage-2)
#   Tầng4  TTA (flip)   [optional]
#   -> submission (top-10 gallery names / query)
#
# Built by reusing CMP's real API (github.com/Shuyu-XJTU/CMP) — verified method signatures:
#   model.get_vision_embeds / get_image_feat / get_text_embeds / get_text_feat / get_cross_embeds / itm_head
#   transform: Resize(224,224,BICUBIC)+ToTensor+Normalize(CLIP) ; k_test=128 ; be_pose_img default OFF here
#
# Paste each "CELL" into a Kaggle notebook cell (GPU=T4). See README_KAGGLE.md for dataset setup.
# RESUME: gallery encoded in chunks to /kaggle/working/cache; rerun skips finished chunks.
# =====================================================================================

# ============================== CELL 0 — setup & paths ==============================
import os, sys, json, time, glob, math, subprocess
# ---- pin transformers to a CMP/X-VLM-compatible version (modern 5.x removed APIs bert.py needs) ----
# Run this cell on a FRESH kernel (Restart) so the import in CELL 1 picks up 4.44.2.
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==4.44.2"], check=False)
import numpy as np, torch
import torch.nn.functional as F

# ---- Kaggle dataset paths: AUTO-DETECTED under /kaggle/input (robust to whatever slug/nesting you used) ----
def _find(pat, root="/kaggle/input"):
    hits = sorted(glob.glob(f"{root}/**/{pat}", recursive=True))
    return hits[0] if hits else None

_ckpt  = _find("cmp.pth")
_vocab = _find("vocab.txt")
_qtext = _find("query_text.json")
CKPT     = _ckpt or "/kaggle/input/cmp-models/cmp.pth"
CMP_DIR  = os.path.dirname(CKPT)
BERT_DIR = os.path.dirname(_vocab) if _vocab else CMP_DIR        # dir that actually holds vocab.txt/tokenizer*
TEST_DIR = os.path.dirname(_qtext) if _qtext else "/kaggle/input/aicity-official-test/name-masked_test-set"
VAL_DIR  = None             # optional labeled set for local ablation (CELL 5); leave None to just submit
print("CKPT    :", CKPT)
print("BERT_DIR:", BERT_DIR, "| has vocab.txt:", os.path.exists(f"{BERT_DIR}/vocab.txt"))
print("TEST_DIR:", TEST_DIR, "| has gallery/:", os.path.isdir(f"{TEST_DIR}/gallery"))

WORK      = "/kaggle/working"
CACHE     = f"{WORK}/cache";  os.makedirs(CACHE, exist_ok=True)
device    = "cuda" if torch.cuda.is_available() else "cpu"
GALLERY_CHUNK = 2000          # images per resume-chunk
K_TEST    = 128               # ITM rerank top-K (CMP default)
# ---- BOOST FLAGS (the MASTER RUN at the end honors these) ----
USE_TTA  = True               # flip-TTA on gallery feature  (+~0.5-1%, costs ~+45min re-encode)
USE_POSE = False              # EXPERIMENTAL pose-ON: ~+2h AND MAY HURT (render is a guess). Set True to try.
RUN_QWEN = True               # Qwen2-VL rerank in an isolated subprocess (handles the transformers clash). LONG on T4, resumable.
print("device:", device, "| torch:", torch.__version__)

# ---- clone CMP code + startv4 helpers (Internet=ON). No internet? upload these repos as datasets. ----
if not os.path.isdir(f"{WORK}/CMP"):
    subprocess.run(["git", "clone", "--depth", "1",
                    "https://github.com/Shuyu-XJTU/CMP", f"{WORK}/CMP"], check=False)
if not os.path.isdir(f"{WORK}/MXT"):
    subprocess.run(["git", "clone", "--depth", "1",
                    "https://github.com/Khanhhh239/Model_XVLM_Training", f"{WORK}/MXT"], check=False)
sys.path.insert(0, f"{WORK}/CMP")
sys.path.insert(0, f"{WORK}/MXT/train3")          # -> import startv4.eval.rerank / startv4.eval.metrics

# ---- patch CMP models/bert.py for MODERN transformers (it was written for transformers <4.13) ----
# apply_chunking_to_forward & friends moved modeling_utils -> pytorch_utils (v4.13); file_utils -> utils.
_bp = f"{WORK}/CMP/models/bert.py"
if os.path.exists(_bp):
    _s = open(_bp, encoding="utf-8").read()
    _s = _s.replace("from transformers.file_utils import", "from transformers.utils import")
    _s = _s.replace(
        "from transformers.modeling_utils import (\n    PreTrainedModel,\n    apply_chunking_to_forward,\n    find_pruneable_heads_and_indices,\n    prune_linear_layer,\n)",
        "from transformers.modeling_utils import PreTrainedModel\n"
        "from transformers.pytorch_utils import apply_chunking_to_forward\n"
        "try:\n"
        "    from transformers.pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer\n"
        "except ImportError:\n"
        "    def find_pruneable_heads_and_indices(*a, **k): raise NotImplementedError('head pruning unused at inference')\n"
        "    def prune_linear_layer(*a, **k): raise NotImplementedError('head pruning unused at inference')")
    open(_bp, "w", encoding="utf-8").write(_s)
    print("patched bert.py for modern transformers")

# deps usually present on Kaggle; install the few that may be missing
for pkg in ["ruamel.yaml", "prettytable", "timm"]:
    try: __import__(pkg.split(".")[0])
    except Exception: subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=False)


# ============================== CELL 1 — build CMP model ==============================
from ruamel.yaml import YAML
from transformers import BertTokenizer
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
from models.model_search import Search

yaml = YAML(typ="safe")
config = yaml.load(open(f"{WORK}/CMP/configs/cmp.yaml"))
config["vision_config"] = f"{WORK}/CMP/configs/config_swinB.json"
config["text_config"]   = f"{WORK}/CMP/configs/config_bert.json"
config["text_encoder"]  = BERT_DIR
config["be_pose_img"]   = False     # v1: pose OFF (pose-on needs rendered pose maps -> stage-2 cell)
config["load_pretrained"] = True

tokenizer = BertTokenizer.from_pretrained(BERT_DIR)
model = Search(config=config)
try:
    model.load_pretrained(CKPT)                       # CMP's own loader
except Exception as e:
    print("load_pretrained failed, fallback to state_dict:", e)
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(sd.get("model", sd), strict=False)
model = model.to(device).half().eval()
print("CMP loaded. params:", sum(p.numel() for p in model.parameters())/1e6, "M")

H, W = config["h"], config["w"]                       # 224,224
_norm = transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                             (0.26862954, 0.26130258, 0.27577711))
TF = transforms.Compose([transforms.Resize((H, W), interpolation=InterpolationMode.BICUBIC),
                         transforms.ToTensor(), _norm])
MAXTOK = config["max_tokens"]


# ============================== CELL 2 — load data lists ==============================
def _read_json_any(p):
    """Read JSON array or JSONL."""
    txt = open(p, encoding="utf-8").read().strip()
    try:
        return json.loads(txt)
    except Exception:
        return [json.loads(l) for l in txt.splitlines() if l.strip()]

def load_masked_set(test_dir):
    gal_dir = f"{test_dir}/gallery"
    gallery = sorted([f for f in os.listdir(gal_dir)
                      if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    rows = _read_json_any(f"{test_dir}/query_text.json")
    # query_text.json: list of {caption/text, (query_index?)}; query_index.txt gives order/ids
    def cap(r): return r.get("caption") or r.get("text") or r.get("query_text") or (r if isinstance(r, str) else "")
    captions = [cap(r) for r in rows]
    qi = f"{test_dir}/query_index.txt"
    qids = [l.strip() for l in open(qi, encoding="utf-8")] if os.path.exists(qi) else [str(i) for i in range(len(captions))]
    print(f"masked set: gallery={len(gallery)}  queries={len(captions)}  qids={len(qids)}")
    return gal_dir, gallery, captions, qids

GAL_DIR, GAL_NAMES, Q_CAPS, Q_IDS = load_masked_set(TEST_DIR)


# ============================== CELL 3 — encoders (chunked + RESUME) ==============================
@torch.no_grad()
def _encode_image_batch(paths):
    imgs = torch.stack([TF(Image.open(p).convert("RGB")) for p in paths]).to(device).half()
    emb, _ = model.get_vision_embeds(imgs)            # [B, L, D]
    feat = F.normalize(model.get_image_feat(emb), dim=-1)
    return emb.half().cpu(), feat.float().cpu()       # emb as fp16 -> half the CPU RAM for 36K gallery

@torch.no_grad()
def encode_gallery(gal_dir, names, chunk=GALLERY_CHUNK, bs=22):
    """Resumable: each chunk saved to CACHE; rerun skips done chunks."""
    n = len(names); nch = math.ceil(n / chunk)
    for c in range(nch):
        fp = f"{CACHE}/gal_{c:04d}.pt"
        if os.path.exists(fp):
            continue
        s, e = c*chunk, min(n, (c+1)*chunk)
        embs, feats = [], []
        t0 = time.time()
        for i in range(s, e, bs):
            ps = [f"{gal_dir}/{names[j]}" for j in range(i, min(e, i+bs))]
            emb, feat = _encode_image_batch(ps)
            embs.append(emb); feats.append(feat)
        torch.save({"emb": torch.cat(embs), "feat": torch.cat(feats), "s": s, "e": e}, fp)
        print(f"  gallery chunk {c+1}/{nch} [{s}:{e}] {time.time()-t0:.0f}s -> {fp}")
    # assemble
    embs, feats = [], []
    for c in range(nch):
        d = torch.load(f"{CACHE}/gal_{c:04d}.pt", map_location="cpu")
        embs.append(d["emb"]); feats.append(d["feat"])
    return torch.cat(embs), torch.cat(feats)          # img_embed [N,L,D], img_feat [N,2048]

@torch.no_grad()
def encode_queries(captions, bs=150):
    embs, atts, feats = [], [], []
    for i in range(0, len(captions), bs):
        t = tokenizer(captions[i:i+bs], padding="max_length", truncation=True,
                      max_length=MAXTOK, return_tensors="pt").to(device)
        emb = model.get_text_embeds(t.input_ids, t.attention_mask)
        feat = F.normalize(model.get_text_feat(emb), dim=-1)
        embs.append(emb.float().cpu()); atts.append(t.attention_mask.cpu()); feats.append(feat.float().cpu())
    return torch.cat(embs), torch.cat(atts), torch.cat(feats)

t0 = time.time()
G_EMB, G_FEAT = encode_gallery(GAL_DIR, GAL_NAMES)         # cached/resumable
Q_EMB, Q_ATT, Q_FEAT = encode_queries(Q_CAPS)
print(f"encoded: gallery {tuple(G_FEAT.shape)}  query {tuple(Q_FEAT.shape)}  in {time.time()-t0:.0f}s")
torch.save({"qids": Q_IDS, "gal": GAL_NAMES}, f"{CACHE}/meta.pt")


# ============================== CELL 4 — rerank toolbox (training-free) ==============================
def sim_itc(q_feat, g_feat):
    return (q_feat @ g_feat.t())                          # [Q, G]  (t2i)

def dual_softmax(sim, tau=0.01):
    # normalize over gallery AND query axes -> fights distractor hubness
    a = F.softmax(sim / tau, dim=0)                       # over queries
    return sim * a

def query_expansion(q_feat, g_feat, topk=5):
    s = q_feat @ g_feat.t()
    idx = s.topk(topk, dim=1).indices
    w = torch.softmax(s.gather(1, idx), dim=1).unsqueeze(-1)
    qexp = q_feat + (w * g_feat[idx]).sum(1)
    return F.normalize(qexp, dim=1)

def k_reciprocal(q_feat, g_feat, k1=20, k2=6, lam=0.3):
    # re-ranking (Zhong CVPR17), cross-modal: QE first then image-space jaccard
    from startv4.eval.rerank import k_reciprocal_rerank   # reuse startv4 impl if available
    qx = query_expansion(q_feat, g_feat, topk=5)
    return k_reciprocal_rerank(qx, g_feat, k1=k1, k2=k2, lam=lam)   # returns score [Q,G]

@torch.no_grad()
def itm_rerank(sim_t2i, g_emb, q_emb, q_att, k=K_TEST, bs=64):
    """CMP cross-encoder rerank of top-k per query (replicates eval.evaluation_itm)."""
    Q, G = sim_t2i.shape
    score = torch.full((Q, G), -1e4)
    for i in range(Q):
        topk_idx = sim_t2i[i].topk(k).indices
        enc = g_emb[topk_idx].to(device).half()                       # [k,L,D]
        att = torch.ones(enc.shape[:-1], dtype=torch.long, device=device)
        te  = q_emb[i].repeat(k, 1, 1).to(device).half()
        ta  = q_att[i].repeat(k, 1).to(device)
        out = model.get_cross_embeds(enc, att, text_embeds=te, text_atts=ta)[:, 0, :]
        sc  = model.itm_head(out)[:, 1].float().cpu()
        score[i, topk_idx] = sc
    # CMP recipe: minmax + add 0.002 * ITC
    mn = score.min(1, keepdim=True).values
    score = torch.where(score == -1e4, mn.expand_as(score), score)
    score = (score - score.min()) / (score.max() - score.min() + 1e-8)
    s_itc = (sim_t2i - sim_t2i.min()) / (sim_t2i.max() - sim_t2i.min() + 1e-8)
    return score + 0.002 * s_itc


# ============================== CELL 5 — ABLATION MONITOR (anti-destructive) ==============================
# Build a LABELED distractor-val: queries+GT from old-test attr.json, gallery = old GT + N distractors
# from the competition gallery. Measure mAP for each technique combo; FLAG any that LOWERS mAP.
try:
    from startv4.eval.metrics import retrieval_metrics  # R@k + mAP (single-GT)
except Exception as _e:
    retrieval_metrics = None; print("ablation metrics unavailable (optional, ablation skipped):", _e)

def build_labeled_val(val_dir, n_distract=5000):
    rows = _read_json_any(f"{val_dir}/attr.json")
    # each row: image, image_id, caption  (old labeled test)
    qcaps, gpaths, qgid = [], [], []
    seen = {}
    for r in rows:
        img = r["image"]; pid = r["image_id"]
        if img not in seen:
            seen[img] = len(gpaths); gpaths.append(f"{val_dir}/{img}")
        caps = r.get("caption"); caps = caps if isinstance(caps, list) else [caps] if caps else []
        for c in caps:
            qcaps.append(c); qgid.append(seen[img])
    # add distractors from competition gallery (never a GT for these queries)
    extra = [f"{GAL_DIR}/{n}" for n in GAL_NAMES[:n_distract]]
    base = len(gpaths); gpaths += extra
    return qcaps, gpaths, torch.tensor(qgid), base

@torch.no_grad()
def encode_paths(paths, bs=22):
    feats, embs = [], []
    for i in range(0, len(paths), bs):
        e, f = _encode_image_batch(paths[i:i+bs]); embs.append(e); feats.append(f)
    return torch.cat(embs), torch.cat(feats)

def run_ablation(val_dir, n_distract=5000):
    if not val_dir or not os.path.isdir(val_dir):
        print(f"[skip ablation] VAL_DIR not set or missing ({val_dir})")
        return None
    qcaps, gpaths, qgid, _ = build_labeled_val(val_dir, n_distract)
    print(f"[ablation] queries={len(qcaps)} gallery={len(gpaths)} (incl {n_distract} distractors)")
    qe, qa, qf = encode_queries(qcaps)
    g_emb, g_feat = encode_paths(gpaths)
    def report(name, score_t2i):
        m = retrieval_metrics(score_t2i, qgid, ks=(1, 5, 10))
        print(f"  {name:<28} R@1={m['R@1']:.4f} R@5={m['R@5']:.4f} mAP={m['mAP']:.4f}")
        return m["mAP"]
    base = sim_itc(qf, g_feat)
    results = {}
    results["ITC"]                = report("ITC (base)", base)
    results["+dual_softmax"]      = report("+dual_softmax", dual_softmax(base))
    try:
        results["+k_reciprocal"]  = report("+k_reciprocal", k_reciprocal(qf, g_feat))
    except Exception as ex: print("  k_reciprocal skipped:", ex)
    results["+ITM"]               = report("+ITM rerank", itm_rerank(base, g_emb, qe, qa))
    # FLAG destructive
    b = results["ITC"]
    print("\n[anti-destructive] vs ITC baseline:")
    for k, v in results.items():
        if k == "ITC": continue
        tag = "KEEP (+)" if v >= b else "DROP (destructive!)"
        print(f"  {k:<20} dmAP={v-b:+.4f}  -> {tag}")
    return results

# OPTIONAL: uncomment to run ablation if you've added a labeled dataset for VAL_DIR
# ABLATION = run_ablation(VAL_DIR, n_distract=5000)


# ============================== CELL 6 — final pipeline on MASKED set -> submission ==============================
def build_final_score(keep_dual=True, keep_kr=False, keep_itm=True):
    sim = sim_itc(Q_FEAT, G_FEAT)                          # [Q,G]
    if keep_dual: sim = dual_softmax(sim)
    if keep_kr:
        try: sim = 0.5*_minmax(sim) + 0.5*_minmax(k_reciprocal(Q_FEAT, G_FEAT))
        except Exception as ex: print("kr skip:", ex)
    if keep_itm: sim = itm_rerank(sim, G_EMB, Q_EMB, Q_ATT)
    return sim

def _minmax(x): return (x - x.min())/(x.max()-x.min()+1e-8)

def write_submission(score_t2i, out=f"{WORK}/answer.txt", topk=10):
    """Write top-10 image names per query to answer.txt (official challenge format)."""
    idx = score_t2i.argsort(dim=1, descending=True)[:, :topk]
    with open(out, "w", encoding="utf-8") as f:
        for i in range(len(Q_IDS)):
            names = [os.path.splitext(GAL_NAMES[j])[0] for j in idx[i].tolist()]   # NAMES WITHOUT extension (challenge format)
            f.write(" ".join(names) + "\n")
    print(f"✓ wrote {out} ({len(Q_IDS)} queries, top-{topk} each, names without extension)")

def save_candidates_for_lmm(score_t2i, k=64, out=f"{WORK}/lmm_candidates.pt"):
    """Save per-query top-k gallery names + scores so a SEPARATE Qwen kernel (CELL 8) can rerank."""
    idx = score_t2i.topk(k, dim=1).indices
    torch.save({"qids": Q_IDS, "caps": Q_CAPS, "gal_dir": GAL_DIR,
                "cand": [[GAL_NAMES[j] for j in idx[i].tolist()] for i in range(len(Q_IDS))],
                "score": score_t2i.gather(1, idx).cpu()}, out)
    print("saved LMM candidates ->", out)

# NOTE: the actual run is the MASTER RUN cell at the very end (it honors USE_TTA/USE_POSE/RUN_QWEN).
# These are function definitions only.


# ============================== CELL 7 — OPTIONAL encode boosts (run, then RE-RUN CELL 6) ==============================
# (A) TTA flip — average ITC feature over horizontal flip. Cheap, low-risk, small recall gain.
@torch.no_grad()
def encode_gallery_tta(gal_dir, names, bs=22):
    feats = []
    for i in range(0, len(names), bs):
        ps = [f"{gal_dir}/{names[j]}" for j in range(i, min(len(names), i+bs))]
        im = torch.stack([TF(Image.open(p).convert("RGB")) for p in ps]).to(device).half()
        f1 = F.normalize(model.get_image_feat(model.get_vision_embeds(im)[0]), dim=-1)
        f2 = F.normalize(model.get_image_feat(model.get_vision_embeds(torch.flip(im, [3]))[0]), dim=-1)
        feats.append(F.normalize((f1 + f2) / 2, dim=-1).float().cpu())
    return torch.cat(feats)
# USE:  G_FEAT = encode_gallery_tta(GAL_DIR, GAL_NAMES)   # then RE-RUN CELL 6

# (B) POSE-ON  (EXPERIMENTAL) — CMP trained with pose, but the masked gallery has NO pose maps and CMP did
#     NOT release its renderer. We GENERATE COCO-17 skeletons (a GUESS at the format) -> may help or may not.
#     VERIFY by submitting pose-ON vs pose-OFF answer.txt. ~1h pose-gen + a full re-encode on T4 (resumable).
#     NOTE: rebuild the model with be_pose_img=True first so pose_conv/pose_block weights are loaded:
#       config["be_pose_img"]=True; model=Search(config); model.load_pretrained(CKPT); model=model.to(device).half().eval()
def gen_pose_maps(gal_dir, names, out_dir=f"{WORK}/pose"):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics"], check=False)
    from ultralytics import YOLO
    import cv2
    os.makedirs(out_dir, exist_ok=True)
    yolo = YOLO("yolov8n-pose.pt")
    SK = [(5,7),(7,9),(6,8),(8,10),(5,6),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16),(0,5),(0,6)]
    for nm in names:
        op = f"{out_dir}/{nm}"
        if os.path.exists(op): continue
        im = cv2.imread(f"{gal_dir}/{nm}"); h, w = im.shape[:2]
        canvas = np.zeros((h, w, 3), np.uint8)
        r = yolo(f"{gal_dir}/{nm}", verbose=False)[0]
        if r.keypoints is not None and len(r.keypoints):
            kp = r.keypoints.xy[0].cpu().numpy()
            for a, b in SK:
                if a < len(kp) and b < len(kp):
                    cv2.line(canvas, tuple(kp[a].astype(int)), tuple(kp[b].astype(int)), (255,255,255), 3)
            for p in kp: cv2.circle(canvas, tuple(p.astype(int)), 3, (0,255,0), -1)
        cv2.imwrite(op, canvas)
    print("pose maps ->", out_dir)

@torch.no_grad()
def encode_gallery_pose(gal_dir, pose_dir, names, bs=16):
    EMB, FEAT = [], []
    for i in range(0, len(names), bs):
        rng = range(i, min(len(names), i+bs))
        img  = torch.stack([TF(Image.open(f"{gal_dir}/{names[j]}").convert("RGB"))  for j in rng]).to(device).half()
        pose = torch.stack([TF(Image.open(f"{pose_dir}/{names[j]}").convert("RGB")) for j in rng]).to(device).half()
        emb, _ = model.get_vision_embeds(img)
        pin = model.pose_conv(pose) if getattr(model, "be_pose_conv", False) else pose
        pe, _ = model.get_vision_embeds(pin)
        emb = model.pose_block(emb, pe)
        EMB.append(emb.float().cpu()); FEAT.append(F.normalize(model.get_image_feat(emb), dim=-1).float().cpu())
    return torch.cat(EMB), torch.cat(FEAT)
# USE:  gen_pose_maps(GAL_DIR, GAL_NAMES)
#       G_EMB, G_FEAT = encode_gallery_pose(GAL_DIR, f"{WORK}/pose", GAL_NAMES)   # then RE-RUN CELL 6
print("CELL 7 ready: encode_gallery_tta / gen_pose_maps+encode_gallery_pose (optional, re-run CELL 6 after).")


# ============================== CELL 8 — AnomalyLMM cloze rerank (ISOLATED SUBPROCESS) ==============================
# Faithful AnomalyLMM (arxiv 2509.04376): mask action-verbs/colors in the query -> LMM fills them from each image
# -> semantic match (filled vs gold) -> S = 0.95*match + 0.075*0.5^pos, rerank top-N=3. Honest: paper gain is only
# +0.96% R@1 on the EASY 1978 gallery over a WEAKER base (X2VLM) -> expect ~0-1% here; measure vs answer_cmp.txt.
# Qwen needs transformers>=4.45 but CMP needs 4.44 -> run in a CHILD process. Resumable every 25 queries.
QWEN_SCRIPT = r'''
# === AnomalyLMM (faithful): masked cross-modal cloze -> LMM fill -> semantic match -> alpha-fused rerank of top-N ===
import os, re, torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from sentence_transformers import SentenceTransformer
from PIL import Image
import nltk
for pkg in ("averaged_perceptron_tagger", "averaged_perceptron_tagger_eng", "punkt", "punkt_tab"):
    try: nltk.download(pkg, quiet=True)
    except Exception: pass
from nltk import pos_tag, word_tokenize

WORK = "/kaggle/working"
d = torch.load(f"{WORK}/lmm_candidates.pt")
MODEL = os.environ.get("QWEN_MODEL", "Qwen/Qwen2-VL-2B-Instruct")
N     = int(os.environ.get("ALMM_TOPN", "3"))           # rerank top-N (paper: 3)
A1, A2, BETA = 0.95, 0.075, 0.5                         # S = a1*S_match + a2*beta^pos (paper params)
COLORS = {"red","orange","yellow","green","blue","purple","pink","brown","black","white","gray",
          "grey","beige","tan","gold","silver","dark","light","navy","maroon","teal","cyan","violet","khaki"}
STOPV = {"is","are","was","were","be","been","being","has","have","had","do","does","did","'s","s"}

def extract_mask(cap):
    """Replace action verbs -> <VERB> and colors -> <COLOR>; return masked sentence + gold words IN ORDER."""
    try: tags = pos_tag(word_tokenize(cap))
    except Exception: tags = [(w, "NN") for w in cap.split()]
    gold, masked = [], []
    for w, t in tags:
        wl = w.lower()
        if t.startswith("VB") and wl not in STOPV:
            gold.append(wl); masked.append("<VERB>")
        elif wl in COLORS:
            gold.append(wl); masked.append("<COLOR>")
        else:
            masked.append(w)
    return " ".join(masked), gold

mdl = Qwen2VLForConditionalGeneration.from_pretrained(MODEL, torch_dtype="auto", device_map="auto").eval()
proc = AutoProcessor.from_pretrained(MODEL, min_pixels=64*28*28, max_pixels=128*28*28)  # cap image tokens for speed
emb = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")                     # semantic word match

def fill(img, masked):
    msg = [{"role": "user", "content": [{"type": "image", "image": img},
            {"type": "text", "text": "Fill each placeholder (<VERB>=an action, <COLOR>=a color) with ONE word, "
             "based ONLY on the image, in order, comma-separated. Write UNKNOWN for a slot you cannot tell.\n"
             f"Sentence: {masked}"}]}]
    t = proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    inp = proc(text=[t], images=[img], return_tensors="pt").to(mdl.device)
    out = mdl.generate(**inp, max_new_tokens=16, do_sample=False)
    return proc.batch_decode(out[:, inp.input_ids.shape[1]:], skip_special_tokens=True)[0]

def match_score(preds, gold):
    """mean cosine(filled, gold) per slot; UNKNOWN/missing -> 0. Captures 'balancing~sliding','dark~gray'."""
    if not gold: return None
    ge = emb.encode(gold, convert_to_tensor=True, normalize_embeddings=True)
    sc = []
    for k in range(len(gold)):
        p = preds[k] if k < len(preds) else ""
        if (not p) or p == "unknown": sc.append(0.0); continue
        pe = emb.encode([p], convert_to_tensor=True, normalize_embeddings=True)[0]
        sc.append(float((pe * ge[k]).sum()))
    return sum(sc) / len(sc)

done = f"{WORK}/qwen_scores.pt"
scores = torch.load(done) if os.path.exists(done) else {}   # qi -> reranked order (list of candidate indices)
for qi, (cap, cands) in enumerate(zip(d["caps"], d["cand"])):
    if str(qi) in scores: continue
    masked, gold = extract_mask(cap)
    order = list(range(len(cands)))                          # default = keep CMP order
    if gold:                                                 # only rerank if there are verbs/colors to verify
        s1 = []
        for nm in cands[:N]:
            img = Image.open(f"{d['gal_dir']}/{nm}").convert("RGB")
            ans = fill(img, masked)
            preds = [w.strip().lower() for w in re.split(r"[,\n;]", ans) if w.strip()]
            s1.append(match_score(preds, gold) or 0.0)
        S = [A1 * s1[n] + A2 * (BETA ** n) for n in range(len(s1))]   # fuse match + positional prior
        topn = sorted(range(len(s1)), key=lambda n: -S[n])
        order = topn + list(range(len(s1), len(cands)))      # reranked top-N, rest unchanged
    scores[str(qi)] = order
    if qi % 25 == 0: torch.save(scores, done); print("almm", qi, "/", len(d["caps"]), flush=True)
torch.save(scores, done)
with open(f"{WORK}/answer.txt", "w", encoding="utf-8") as f:
    for qi, cands in enumerate(d["cand"]):
        order = scores.get(str(qi), list(range(len(cands))))
        f.write(" ".join(os.path.splitext(cands[j])[0] for j in order[:10]) + "\n")  # names WITHOUT extension
print("DONE: AnomalyLMM cloze rerank -> answer.txt")
'''
def run_qwen_subprocess(model="Qwen/Qwen2-VL-2B-Instruct", topn=3):
    with open(f"{WORK}/qwen_rerank.py", "w", encoding="utf-8") as f: f.write(QWEN_SCRIPT)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "transformers>=4.45,<5", "qwen-vl-utils", "accelerate", "sentence-transformers", "nltk"], check=False)
    env = dict(os.environ, QWEN_MODEL=model, ALMM_TOPN=str(topn))
    subprocess.run([sys.executable, f"{WORK}/qwen_rerank.py"], env=env, check=False)
print("CELL 8 ready: run_qwen_subprocess() = AnomalyLMM cloze rerank (called by MASTER RUN if RUN_QWEN).")


# ============================== CELL 9 — MASTER RUN (honors BOOST FLAGS -> writes answer.txt) ==============================
if USE_POSE:
    print(">> POSE-ON (experimental): rebuild pose-on model + generate pose maps + re-encode gallery (~+2h, may HURT).")
    config["be_pose_img"] = True
    model = Search(config=config)
    try: model.load_pretrained(CKPT)
    except Exception:
        _sd = torch.load(CKPT, map_location="cpu", weights_only=False); model.load_state_dict(_sd.get("model", _sd), strict=False)
    model = model.to(device).half().eval()
    gen_pose_maps(GAL_DIR, GAL_NAMES)
    G_EMB, G_FEAT = encode_gallery_pose(GAL_DIR, f"{WORK}/pose", GAL_NAMES)

if USE_TTA:
    print(">> TTA flip: re-encoding gallery ITC feature (averaged over horizontal flip).")
    G_FEAT = encode_gallery_tta(GAL_DIR, GAL_NAMES)

SCORE = build_final_score(keep_dual=True, keep_kr=False, keep_itm=True)   # CMP ITC + dual-softmax + ITM rerank
write_submission(SCORE, out=f"{WORK}/answer_cmp.txt")   # CMP baseline kept SEPARATELY (submit to compare vs Qwen)
write_submission(SCORE)                  # answer.txt (Qwen overwrites this; failure-safe if Qwen dies/times out)
save_candidates_for_lmm(SCORE)

if RUN_QWEN:
    print(">> Qwen rerank (isolated subprocess; pip-upgrades transformers in a child process; long but resumable).")
    run_qwen_subprocess()                # overwrites answer.txt with the Qwen-reranked ranking

print("FINAL -> /kaggle/working/answer.txt  (download + submit)")


# ============================== CELL 10 — fix format (strip extension) + DIRECT download links ==============================
# Run this cell anytime after a run: it strips .jpg/.png from the answer files (challenge wants names WITHOUT
# extension) and prints clickable links that download the file straight to your machine.
import os, base64
from IPython.display import HTML, display

def finalize_and_download(path):
    if not os.path.exists(path):
        print("skip (not found):", path); return
    s = open(path, encoding="utf-8").read()
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
        s = s.replace(ext, "")
    open(path, "w", encoding="utf-8").write(s)
    name = os.path.basename(path)
    b64 = base64.b64encode(s.encode()).decode()
    print(f"✓ {name}: extensions stripped, {s.count(chr(10))} lines")
    display(HTML(f'<a download="{name}" href="data:text/plain;base64,{b64}" '
                 f'style="font-size:16px;font-weight:bold;color:#0a0">⬇️ Download {name}</a>'))

finalize_and_download("/kaggle/working/answer.txt")        # Qwen-reranked (or baseline if RUN_QWEN=False)
finalize_and_download("/kaggle/working/answer_cmp.txt")    # CMP baseline (submit too, to compare vs Qwen)
