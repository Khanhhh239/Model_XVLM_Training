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
import numpy as np, torch
import torch.nn.functional as F

# ---- Kaggle dataset mount points (ADJUST to your dataset slugs) ----
TEST_DIR  = "/kaggle/input/aicity-official-test/name-masked_test-set"   # gallery/ + query_text.json + query_index.txt
CMP_DIR   = "/kaggle/input/cmp-models"                                   # cmp.pth + bert-base-uncased/ + (swin/xvlm init)
CKPT      = f"{CMP_DIR}/cmp.pth"
BERT_DIR  = f"{CMP_DIR}/bert-base-uncased"
# labeled validation (old test WITH GT) for the ablation monitor (optional but STRONGLY recommended)
VAL_DIR   = "/kaggle/input/aicity-official-test/old_test-set"            # test/ + attr.json   (adjust if separate dataset)

WORK      = "/kaggle/working"
CACHE     = f"{WORK}/cache";  os.makedirs(CACHE, exist_ok=True)
device    = "cuda" if torch.cuda.is_available() else "cpu"
GALLERY_CHUNK = 2000          # images per resume-chunk
K_TEST    = 128               # ITM rerank top-K (CMP default)
print("device:", device, "| torch:", torch.__version__)

# ---- clone CMP code (no internet? upload CMP repo as a dataset and add its path instead) ----
if not os.path.isdir(f"{WORK}/CMP"):
    subprocess.run(["git", "clone", "--depth", "1",
                    "https://github.com/Shuyu-XJTU/CMP", f"{WORK}/CMP"], check=False)
sys.path.insert(0, f"{WORK}/CMP")
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
    return emb.float().cpu(), feat.float().cpu()

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
from startv4.eval.metrics import retrieval_metrics  # R@k + mAP (single-GT)

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

# ABLATION = run_ablation(VAL_DIR, n_distract=5000)   # uncomment when VAL_DIR is set


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

def write_submission(score_t2i, out=f"{WORK}/submission.txt", topk=10):
    idx = score_t2i.argsort(dim=1, descending=True)[:, :topk]
    with open(out, "w", encoding="utf-8") as f:
        for i, qid in enumerate(Q_IDS):
            names = [GAL_NAMES[j] for j in idx[i].tolist()]
            f.write(" ".join(names) + "\n")              # NOTE: confirm exact submission format vs challenge spec
    print("wrote", out)

# After choosing the config that the ablation says is best:
# SCORE = build_final_score(keep_dual=True, keep_kr=False, keep_itm=True)
# write_submission(SCORE)


# ============================== CELL 7 — STAGE-2 add-ons (enable after core verified) ==============================
# (A) POSE-ON: CMP trained with be_pose_img=True. Needs rendered pose-map images for each gallery image
#     (CMP loads them from image_root/pose/<image>). Generate pose maps (ViTPose->render in CMP's format),
#     set config["be_pose_img"]=True, and fuse: emb,_=get_vision_embeds(img); pe,_=get_vision_embeds(pose);
#     emb = model.pose_block(emb, model.pose_conv(pe) if config["pose_conv"] else pe). Verify format vs models/pose.py.
# (B) AnomalyLMM (Qwen2-VL) cloze rerank: mask verbs/colors in query -> Qwen completes per top-K image -> compare.
#     Heavy on T4 (top-K x 1978 LMM forwards, ~hours) -> run with resume; GATE via ablation (keep only if +mAP).
#     Honest: our earlier naive Qwen rerank gave ~+1%; expect modest. Add LAST, keep only if monitor says +.
print("core notebook ready. Stage-2 (pose, Qwen) = enable after the core number is validated.")
