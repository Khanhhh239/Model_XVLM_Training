# STAR — Stage 1.5 Training Codebase
### AI City 2026 Track 4 — Text-Based Person Anomaly Retrieval (Sim2Real)

Fine-tuning pipeline for **STARModel** (X-VLM 16M + LoRA + pose), warm-started from a checkpoint
that scores **mAP 0.8323 / R@1 ~0.71 / R@10 0.99** on the 30k-hard VAL-B (synthetic, leakage-free
split by video). The goal of Stage 1.5 is to **sharpen rank-1** on top of that warm-init.

> Full algorithm + loss math + design rationale: **[`analyze.md`](analyze.md)**.

---

## 1. Architecture

```
 Image (.webp 384, LHP + sim2real aug)              Caption
        │                                              │
 X-VLM Image Encoder · Swin-B  [LoRA qkv]      X-VLM Text Encoder · BERT[0:6]  [FROZEN]
        │   └─ Pose branch (keypoint-MLP) → fuse       │
        │   → f_V                                      │ → f_T
        └──────────────────┬────────────────────────── ┘
                           │  ITC (cosine, identity soft targets) + Smooth-AP  ← PairBatchSampler
        X-VLM Cross-Encoder · BERT[6:12]  [LoRA query/value]
                           │  → ITM head (hard-neg)  [used by inference rerank]
        L = ITC + 2·ITM + 0.2·Smooth-AP   (+ optional box / anomaly heads)
```

| Component | State |
|---|---|
| Image encoder (Swin-B) | LoRA `qkv` |
| Cross-encoder (BERT 6–12) | LoRA `query/value` — **the module the ITM rerank uses at inference** |
| Text encoder (BERT 0–6) | **FROZEN** |
| Pose branch (keypoint-MLP) | ON (part of the warm-init), fused into f_V |
| ITM head, image proj, ITC temp | trained |
| bbox head (optional) | person-box from keypoint extent — `stage1_run3_bbox.yaml` only |

**Warm-init:** `scripts/train.py --init-from <best.pth>` loads `backbone.model.*` + `pose.*` + LoRA
(strict=False) on top of the vanilla X-VLM base (`xvlm_16m_base.th`). Fresh heads stay random.

**Safety floor:** the trainer evals the warm-init at step 0 and sets that as `best_metric`; `best.pth`
is overwritten **only** when a later eval beats it. Early-stop tracks the run's **own** best
(decoupled from the floor) so a slowly-recovering head isn't killed mid-learning.

---

## 2. Configs (only two are kept)

| Config | What |
|---|---|
| [`configs/stage1_safe_warmstart.yaml`](configs/stage1_safe_warmstart.yaml) | **BASELINE** — pair-batch + ITM + Smooth-AP + pose. The recipe that produced 0.8323. |
| [`configs/stage1_run3_bbox.yaml`](configs/stage1_run3_bbox.yaml) | **BBOX** — baseline + person-box head (BoxGroundingHead; GT derived from keypoint extent). |

> Other levers tried and **dropped** (none beat 0.8323 on VAL-B): anomaly head, phrase-box head,
> XBM + ANCE re-mining. See §5.

---

## 3. Run on Kaggle (T4)

Canonical notebook: **[`notebooks/kaggle_stage15_best3.ipynb`](notebooks/kaggle_stage15_best3.ipynb)**
(Accelerator = T4 GPU, Internet = ON). It git-clones this repo, builds the pinned X-VLM env via
`scripts/kaggle_setup.py` (transformers 4.12.5 + X-VLM source + 4 patches), locates the datasets,
(re)builds the manifest, and trains.

**Each run, change two things:**
1. Cell-5: `CONFIG = "configs/stage1_safe_warmstart.yaml"` (or `stage1_run3_bbox.yaml`)
2. Cell-7: `INIT_NAME = "best (4).pth"` (the warm-init checkpoint to load; `None` → use `best (3).pth`)

Required Kaggle datasets: `aicity-30k-hard-enhanced` (webp + jsonl + vitpose), `ckpt-30k-hard`
(`xvlm_16m_base.th` + manifest parquet), the warm-init `.pth`, and (for phrase-box only) `bbox-dataset`.

Watch at startup: `[baseline] warm-init VAL-B mAP=0.8323`, then per-eval `[VAL-B] ... R@1 ...`.

---

## 4. Inference (rerank)

```bash
python scripts/run_inference.py --ckpt best.pth --manifest <valb.parquet> \
    --image-root <webp> --base-ckpt xvlm_16m_base.th --topk 100
```
Pipeline: cosine Stage-1 → Top-100 → **ITM cross-encoder rerank** → Gale-Shapley → top-10. Reports
metrics at **every** stage, so `rerank` vs `stage1` shows the cross-encoder's R@1 lift directly.

---

## 5. Findings (honest status)

- **Bi-encoder VAL-B is saturated at ~0.8323.** Pair-batch + ITM + Smooth-AP (the warm-init recipe)
  is the ceiling. Across four fine-tune experiments, **no auxiliary training lever beat it**:
  anomaly head (null), phrase-box head (0.8264, hurts the bi-encoder slightly), XBM + ANCE
  (0.80, XBM drift confirmed — `itc` oscillates, doesn't converge).
- **The gap is rank-1 ordering, not recall** (R@10 0.99 vs R@1 0.71) → the highest-ROI lever is the
  **inference ITM rerank** (§4), not more bi-encoder aux losses.
- **VAL-B is synthetic and does not transfer 1:1 to the real test** (documented sim2real drop). For
  the real goal, the training-side levers worth trying are **text-LoRA (unfreeze the text tower)** and
  **caption augmentation / external real datasets** — not more heads.

---

## 6. Repo layout
```
├── README.md / analyze.md            # this file / full design + math reference
├── configs/                          # stage1_safe_warmstart.yaml (baseline), stage1_run3_bbox.yaml
├── notebooks/kaggle_stage15_best3.ipynb
├── src/star/
│   ├── config.py · metrics.py
│   ├── losses/   # itc, smooth_ap, itm, action, weighting
│   ├── models/   # backbone (X-VLM wrapper), lora, pose, heads, star_model
│   ├── data/     # dataset (manifest + keypoint→bbox), transforms (LHP), sampler, mining
│   ├── engine/   # optim, evaluator, trainer (safety floor + decoupled early-stop)
│   ├── inference/# pipeline (cosine → ITM rerank → Gale-Shapley)
│   └── utils/
├── scripts/      # train, run_inference, build_stage1_manifest, kaggle_setup, evaluate,
│                 # pose_rerank, qwen_rerank, extract_pose_yolo, vitpose_extract
└── tests/        # pytest unit + integration suite (run: PYTHONPATH=src pytest -q)
```

Local dev uses a built-in dummy backbone so `pytest` runs offline; the real X-VLM import is lazy
and only triggers when `model.backbone=xvlm`.
