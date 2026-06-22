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
            names = [GAL_NAMES[j] for j in idx[i].tolist()]
            f.write(" ".join(names) + "\n")
    print(f"✓ wrote {out} ({len(Q_IDS)} queries, top-{topk} each)")

def save_candidates_for_lmm(score_t2i, k=64, out=f"{WORK}/lmm_candidates.pt"):
    """Save per-query top-k gallery names + scores so a SEPARATE Qwen kernel (CELL 8) can rerank."""
    idx = score_t2i.topk(k, dim=1).indices
    torch.save({"qids": Q_IDS, "caps": Q_CAPS, "gal_dir": GAL_DIR,
                "cand": [[GAL_NAMES[j] for j in idx[i].tolist()] for i in range(len(Q_IDS))],
                "score": score_t2i.gather(1, idx).cpu()}, out)
    print("saved LMM candidates ->", out)

# CELL 6 — runs automatically on "Run All" -> writes /kaggle/working/answer.txt
SCORE = build_final_score(keep_dual=True, keep_kr=False, keep_itm=True)   # CMP ITC + dual-softmax + ITM rerank
write_submission(SCORE)                  # -> /kaggle/working/answer.txt  (download + submit this)
save_candidates_for_lmm(SCORE)           # for optional Qwen rerank later (CELL 8, separate kernel)


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


# ============================== CELL 8 — Qwen2-VL AnomalyLMM rerank (RUN IN A SEPARATE KERNEL) ==============================
# Qwen2-VL needs transformers>=4.45 but CMP needs 4.44 -> they CANNOT share a kernel. Workflow:
#   1) In the CMP kernel: run CELL 6 incl. save_candidates_for_lmm(SCORE) -> /kaggle/working/lmm_candidates.pt
#   2) Open a NEW notebook (same datasets), set RUN_QWEN=True, run ONLY this cell. It rewrites answer.txt.
# Honest: rerank touches PRECISION not recall; ~+1-2 mAP, slow (~hours), resumable. Keep only if it helps.
RUN_QWEN = False
if RUN_QWEN:
    os.system("pip install -q 'transformers>=4.45' qwen-vl-utils accelerate")
    import torch as _t
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    from PIL import Image as _Img
    WORK = "/kaggle/working"
    d = _t.load(f"{WORK}/lmm_candidates.pt")
    mdl = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct", torch_dtype="auto", device_map="auto")
    proc = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
    TOPK_RR = 10                                  # rerank the top-10 per query (raise if time allows)
    done = f"{WORK}/qwen_scores.pt"
    scores = _t.load(done) if os.path.exists(done) else {}
    for qi, (cap, cands) in enumerate(zip(d["caps"], d["cand"])):
        if str(qi) in scores: continue
        sc = []
        for nm in cands[:TOPK_RR]:
            img = _Img.open(f"{d['gal_dir']}/{nm}").convert("RGB")
            msg = [{"role": "user", "content": [{"type": "image", "image": img},
                    {"type": "text", "text": f"On a scale 0-100, how well does this image match: '{cap}'? Reply only the number."}]}]
            txt = proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            inp = proc(text=[txt], images=[img], return_tensors="pt").to(mdl.device)
            out = mdl.generate(**inp, max_new_tokens=8)
            ans = proc.batch_decode(out[:, inp.input_ids.shape[1]:], skip_special_tokens=True)[0]
            num = "".join(c for c in ans if c.isdigit())
            sc.append(float(num) if num else 0.0)
        scores[str(qi)] = sc
        if qi % 50 == 0: _t.save(scores, done); print("qwen", qi, "/", len(d["caps"]))
    _t.save(scores, done)
    with open(f"{WORK}/answer.txt", "w", encoding="utf-8") as f:    # Qwen score primary, original order tiebreak
        for qi, cands in enumerate(d["cand"]):
            s = scores.get(str(qi), [])
            order = sorted(range(len(cands)), key=lambda j: (-(s[j] if j < len(s) else 0), j))
            f.write(" ".join(cands[j] for j in order[:10]) + "\n")
    print("wrote answer.txt with Qwen rerank")
else:
    print("CELL 8 idle. Set RUN_QWEN=True in a SEPARATE kernel (transformers>=4.45) to rerank.")
