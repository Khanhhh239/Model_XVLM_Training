# STAR-v4 — Text-Based Person Anomaly Retrieval (AI City 2026 Track 4)

> Full design rationale: `../STAR_v4_architecture.md`. This repo is the **implementation**.

## Final architecture (chosen)
**Retrieve → rerank; two decoupled models meeting only at the SCORE level.**
1. **RETRIEVAL = SigLIP-2** (`so400m@384` or `large@512`), LoRA + **sigmoid loss**, lr 5e-5 +
   cosine + EMA, sim2real **augmentation** (blur/JPEG/downscale/erase/text-mask), optional FILIP
   (chunked). The score driver — recall among the 35K distractors is the competition gap, and
   SigLIP-2's ~10B real-image pretrain (kept via light LoRA) is the only legal sim2real cure
   (OOPS! is banned). → `configs/siglip_full1m_a100_40g.yaml` (full 1M) / `siglip_so400m_384.yaml`.
2. **RERANK = X-VLM ITM cross-encoder** (add ONLY if distractor-val shows the gap is *precision*),
   warm-started from the pretrained X-VLM at the repo root; ITM-only by default
   (`configs/xvlm_rerank_only.yaml`), or all heads incl. box-grounding for #3
   (`configs/xvlm_full_wrongcase.yaml`, needs a GroundingDINO pass).
3. **Inference (training-free):** encode-once-cache → ensemble (RRF/min-max) → ITM rerank top-K →
   QE → k-reciprocal → TTA — keep each only if it beats its cost on **distractor-val**.

**Hardware:** 1× A100 **40GB is enough** (SigLIP LoRA @512 batch 48 ≈ 30GB) — right-size, don't
fill VRAM; no DDP; train Stage 1, measure, add Stage 2 only when proven.
**Honest expectation:** ~90-93% realistic, 94% the optimistic edge — UNPROVEN until run + measured.
**Workflow:** `zeroshot_baseline` → train SigLIP (early-stop on distractor-val) → measure
wrong-cases per category (`scripts/eval_wrong_cases.py`) → add rerank/box only if it helps.

---

## 🚫 COMPLIANCE — read first

The competition **test set = real frames from the OOPS! dataset**, and the provided
**synthetic training set was generated from OOPS! captions** (PAB paper,
[arxiv 2411.17776](https://arxiv.org/abs/2411.17776)).

- **NEVER train on OOPS!** or any YouTube fail-compilation data (FailArmy, Kinetics-fail, …).
  Using it = training on the test source = leakage + rule violation.
- Train uses **only** the provided **PAB synthetic** set.
- Sim2real is closed **without** real test-domain data: a **real-pretrained backbone
  (SigLIP)** + **augmentation** + **full 1M synthetic** (top teams reach 95% this way too).
- Distractor-val distractors must be **provably disjoint from OOPS!** (Market-1501 / MSMT17 /
  COCO-person) and perceptual-hash de-duped against the test gallery.

---

## Why this architecture (1-paragraph)

Recall among 35K distractors is the competition bottleneck (R@10 94% → mAP 80%). SigLIP-2-L,
pretrained on billions of **real** images, both lifts recall and survives the synthetic→real
gap — fine-tuned **lightly** (LoRA, lr 5e-5, EMA) so its real-image knowledge is preserved.
X-VLM (kept from v3) supplies a cross-encoder for precision rerank. Ensemble + k-reciprocal +
query-expansion + TTA add training-free mAP.

---

## Repo layout

```
startv4_full/                 # Phase 0-3 (copy of startv4 + Phase 3); package import name = startv4
  configs/
    siglip_a100_80g_1m.yaml   # Model A — the real A100 run (full 1M)
    xvlm_v4_a100.yaml         # Model B — X-VLM-v4 cross-encoder (Phase 3)
    _test_dummy.yaml          # tiny CPU config (SigLIP) for tests / smoke
    _test_xvlm_dummy.yaml     # tiny CPU config (X-VLM-v4) for tests / smoke
  startv4/
    config.py                 # YAML + dotted overrides
    losses/                   # sigmoid (SigLIP), infonce+queue, filip, box(L1+GIoU), smooth-AP
    models/
      siglip_retrieval.py     # Model A: HF SigLIP-2-L + LoRA  (+ dummy fallback)
      xvlm_v4.py              # Model B: cross-encoder (CrossEncoder) + ITC/ITM (+ dummy fallback)
      heads.py                # heads: bbox, anomaly-bucket, pose-fuse (global), FILIP
      ema.py  queue.py        # EMA weights, MoCo negative queue
    data/
      augment.py              # sim2real augmentation (blur/JPEG/downscale/noise/erase…)
      dataset.py manifest.py sampler.py
      dedup.py                # perceptual-hash de-dup (Phase 0 compliance)
      tokenizer.py            # HF SigLIP / dummy tokenizer builder
      grounding.py            # Phase 3: noun-phrase extraction + box pseudo-labels
    eval/
      metrics.py              # R@k, mAP (single-GT)
      rerank.py               # RRF, min-max fuse, query-expansion, k-reciprocal
      distractor_val.py       # recall under external distractors + build_distractor_index
    train/
      trainer.py              # SigLIP trainer (overfit/fit, EMA, cosine-warmup, eval hook, grad-norm)
      trainer_xvlm.py         # X-VLM-v4 trainer (ITC+ITM hard-neg+FILIP+anom+smoothAP, 2 queues)
      sched.py                # cosine LR schedule with linear warmup
      mining.py               # OFFLINE hard-negative mining utility (TRAIN; not in-loop)
      train_siglip.py  train_xvlm.py   # CLI entries
    infer/
      encode.py               # encode + cache embeddings (+ TTA average, keypoint-threaded) — T4-free
      pipeline.py             # fuse_and_rerank / build_ranking / evaluate_with_pipeline
      rerank_xvlm.py          # X-VLM ITM cross-encoder rerank of top-K
    scripts/
      build_distractor_val.py # Phase 0: build de-duped distractor-val index
      zeroshot_baseline.py    # Phase 1 (pre-train): FROZEN SigLIP zero-shot on distractor-val
      encode_cache.py         # Phase 1: encode images+queries -> .pt cache (per scale)
      run_phase2.py           # Phase 2: TTA + RRF ensemble + QE/k-reciprocal -> metrics/answer
      build_box_pseudolabels.py # Phase 3: phrase->box pseudo-labels (GroundingDINO/dummy)
  tests/                      # 81 tests (CPU; incl. a real tiny-SigLIP smoke, skips if offline)
```

---

## Install

```bash
pip install -r requirements.txt          # core + transformers/peft
# or: pip install -e .            (then  pip install -e ".[hf,dev]")
```

Everything runs CPU-only with a **dummy backbone** (for tests/smoke); the real run needs a
GPU + the HF SigLIP-2 weights.

---

## Tests

```bash
pytest -q          # 81 passed  (~48s; the real tiny-SigLIP smoke skips if HF hub unreachable)
```

Each module is covered: losses, augmentation, EMA, queue, metrics, rerank (incl. k-reciprocal),
dataset/sampler, both model forwards (SigLIP + X-VLM-v4), distractor-val, both trainers
(overfit-one-batch decreases), hard-neg mining, ITM rerank, grounding, and the full
Phase 0→1→2 + Phase 3 pipelines.

---

## Data prep (brief for the data team)

Deliver a **manifest** (`.parquet`/`.jsonl`) — one row per synthetic image:

| column | meaning | source |
|---|---|---|
| `image_path` | path under `image_root` | given |
| `caption` | text | given |
| `video_id` | source event id (split + pair-batch) | given |
| `bucket` | `normal`/`anomaly` (or goal/wentwrong/full) | from caption type |
| `keypoints` | 17×3 COCO `[[x,y,conf]…]` | **extract (ViTPose)** |
| `bbox` | `[x1,y1,x2,y2]` norm | **extract (detector)** |
| `image_id`, `pair_image_id` | same-video hard pair | **build from video_id** |

Required: `image_path, caption, video_id, bucket`. Others optional (heads degrade gracefully).
All rows are synthetic. **No OOPS!/real data.** Augmentation is applied **online in training**
— the team ships **clean images**, not pre-augmented ones.

Distractor-val (Phase 0): old-test GT positives + ~20K real **non-OOPS!** person distractors,
perceptual-hash de-duped vs the test gallery.

---

## Train — Model A (SigLIP, Phase 1)

```bash
# sanity check FIRST (loss must fall sharply):
python -m startv4.train.train_siglip --config configs/siglip_a100_80g_1m.yaml --overfit-one-batch

# full run, saving the BEST checkpoint by distractor-val mAP (recommended):
python -m startv4.train.train_siglip --config configs/siglip_a100_80g_1m.yaml \
    --val-index distractor_val/index.json --val-query-json distractor_val/queries.json \
    --eval-every 0 --out checkpoints/siglip_v4.pth     # eval-every 0 = once per epoch
```

Key knobs (in `configs/siglip_a100_80g_1m.yaml`): `optim.lr=5e-5` (low — preserves
real-pretrain), `optim.ema=true`, `optim.bf16=true`, `optim.grad_checkpoint=true`,
`train.batch_size=128`, `data.sampler=balanced|pair`. Without `--val-index` the **final**
(not best) model is saved. `train_xvlm` takes the same `--val-*` flags.

### Right-size the hardware (don't fill VRAM — that's the waste)

Filling the GPU is **not** the goal; mAP-per-GPU-hour is. This is a low-LR LoRA fine-tune that
preserves the pretrain, so gradients are small → **batch beyond ~64-128 gives diminishing returns
and fewer steps/epoch.** Pick batch for convergence, not for VRAM %.
- `batch_size: 64` @512 + bf16 + grad-ckpt ≈ **~20-30 GB → a 40 GB GPU suffices** for Model A.
  Profile one real step before renting; only step up to 80 GB for 768-FixRes or an X-VLM stage,
  and only if distractor-val proves it's needed.
- **Cheaper backbone option:** `configs/siglip_so400m_384.yaml` (so400m @384) often matches
  large@512 per-FLOP — measure both zero-shot on distractor-val and pick by mAP/GPU-hour.
- **Subset + early-stop:** train on a 300-500K balanced subset and pass
  `--val-index/--val-query-json --eval-every` to stop when distractor-val plateaus — don't pay for
  1M×8 if 400K×2 already plateaus.
- 768 FixRes is a later stage (`--set data.image_size=768 optim.lr=1e-5`), 512 first.
- One GPU → **no DDP**; train Model A, measure, then add Model B only if precision is the gap.

---

## Train — Model B (X-VLM-v4, Phase 3)

Self-contained cross-encoder (image + text + cross-attention fusion + ITM) with the v4 aux
heads.  Loss = `w_itc·ITC(+queue) + w_itm·ITM(hard-neg) + w_filip·FILIP + w_box·box + w_anom·CE
+ w_smoothap·SmoothAP` (weights in `configs/xvlm_v4_a100.yaml`; `w_itm=1.5` — the v3 insight).

**Add X-VLM only if distractor-val shows the gap is PRECISION (GT in top-K, mis-ranked), not
recall** — otherwise it's wasted compute. When you do, prefer the **rerank-only** config (trains
ITM alone; ITC/queue/SmoothAP/FILIP off):

```bash
# sanity check FIRST:
python -m startv4.train.train_xvlm --config configs/xvlm_rerank_only.yaml --overfit-one-batch
# rerank-only run, warm-started from the pretrained X-VLM in .. (set optim.warm_start):
python -m startv4.train.train_xvlm --config configs/xvlm_rerank_only.yaml --out checkpoints/xvlm_v4.pth
# (the full multi-head xvlm_v4_a100.yaml exists too, but is heavier and speculative)
```

> **Recommended primary path:** a from-scratch 2-6 layer cross-encoder cannot match a pretrained
> ITM, so for the last 2-3 mAP points **port these heads/losses onto the pretrained X-VLM in
> `..`** (set `optim.warm_start` to its `best.pth`).  The self-contained `xvlm_v4.py`
> (HF Swin+BERT / dummy) is the testable reference and the fallback if train2 is unavailable.
>
> **Defaults for the first 94% attempt** (`xvlm_v4_a100.yaml`): `use_box=false`, `use_pose=false`
> (box needs a ~10-20h GroundingDINO pass to be non-trivial; pose needs ViTPose on the 36K
> gallery at inference). Keep `use_anomaly` (free label) + `use_filip`. Enable box/pose later and
> watch `grad_norm` / VAL-B R@5/R@10 via the eval hook.

**ITM rerank at inference** (precision stage over the top-K):

```python
from startv4.infer.rerank_xvlm import itm_rerank_ranking
rank = itm_rerank_ranking(model, base_score, query_tokens, query_mask,
                          gallery_tokens, topk=200, alpha=0.5, device="cuda")
```

**Hard-negative mining** (every N epochs, TRAIN only — never test):

```python
from startv4.train.mining import mine_hard_negatives
hard = mine_hard_negatives(model, train_loader, device)   # {idx -> hardest-wrong idx}
```

**Box pseudo-labels** for the grounding head (Phase 3, optional):

```bash
python -m startv4.scripts.build_box_pseudolabels --manifest data/manifest_1m.parquet \
    --detector dummy --out data/box_pseudolabels.jsonl     # swap dummy -> GroundingDINO for real
```

---

## Inference — encode once, iterate free (Kaggle T4)

```python
from startv4.infer.encode import encode_image_loader, save_embeddings, load_embeddings
from startv4.infer.pipeline import build_ranking, write_answer

# (1) ENCODE ONCE per checkpoint (~2-3h on T4) and cache:
gal = encode_image_loader(model, gallery_loader, device)   # [36773, D]
qry = encode_text_list(model, ids, mask, device)           # [1978, D]
save_embeddings("cache/siglip.pt", gallery=gal, query=qry)

# (2) ITERATE FOR FREE on cached embeddings (RRF ensemble + QE + k-reciprocal):
e = load_embeddings("cache/siglip.pt")
rank = build_ranking(e["query"], e["gallery"], extra_sims=[xvlm_sim],
                     fuse="rrf", use_qe_kr=True, k1=20, k2=6, lam=0.3)
write_answer(rank, gallery_ids, "answer.txt", topk=10)
```

T4 16GB fits inference (models load sequentially). NOTE: inside `startv4_full` SigLIP and the
`xvlm_v4` cross-encoder BOTH use modern transformers — **no version conflict**. The
"encode in separate kernels" advice only applies if you use the OLD pretrained X-VLM in
`..` (transformers 4.12.5); then encode each in its own kernel and fuse the cached
embeddings offline.

> **Cache sizes:** pooled embeddings for retrieval/ensemble are tiny (36K×D ≈ a few hundred MB).
> The **ITM rerank** (`rerank_xvlm.itm_rerank_ranking`) instead needs **token-level** features for
> the top-K candidates, which are far larger (36K × Ntok × D = several GB if cached for the whole
> gallery) — so only cache/keep tokens for the top-K per query, not the full gallery.

---

## Run the phases (end-to-end CLI)

```bash
# PHASE 0 — build the distractor-val (de-duped against the test gallery):
python -m startv4.scripts.build_distractor_val \
    --gt-dir old_test/gt --distractor-dir external/market1501 \
    --test-gallery-dir test/gallery --out distractor_val/index.json

# PHASE 1 (pre-train) — measure the FROZEN zero-shot baseline FIRST (the 'free' mAP):
python -m startv4.scripts.zeroshot_baseline --config configs/siglip_a100_80g_1m.yaml \
    --index distractor_val/index.json --query-json distractor_val/queries.json --scale 512

# PHASE 1 — train SigLIP (above), then encode the distractor-val at each TTA scale:
python -m startv4.scripts.encode_cache --config configs/siglip_a100_80g_1m.yaml \
    --ckpt checkpoints/siglip_v4.pth --use-ema --scale 512 \
    --index distractor_val/index.json --query-json distractor_val/queries.json \
    --out cache/siglip_512.pt
python -m startv4.scripts.encode_cache ... --scale 768 --out cache/siglip_768.pt

# PHASE 2 — TTA + ensemble + QE/k-reciprocal -> distractor-val metrics (free, repeatable):
python -m startv4.scripts.run_phase2 --siglip cache/siglip_512.pt,cache/siglip_768.pt \
    --xvlm cache/xvlm.pt --fuse rrf --qe-kr
# submission: encode with --gallery-dir instead of --index, then run_phase2 writes answer.txt
```

The whole chain runs on the dummy backbone with no downloads (see `tests/test_scripts_and_e2e.py`).
**Measure distractor-val after each phase before building the next.**

---

## Auxiliary heads (Model B / X-VLM-v4) — why train what you don't infer

`bbox` and `anomaly-bucket` heads do **not** produce retrieval scores. Their loss flows into
the **shared encoder**, shaping its features; at inference you drop the head but keep the
better encoder (used by ITC/ITM). This is auxiliary-task / multi-task regularization.

- **anomaly-bucket** — label is **free** (the synthetic `bucket`), keep it.
- **bbox grounding** — needs labels we don't have. The cheap single-box pseudo-label is
  near-trivial; the **useful** version is **GroundingDINO** phrase→box pseudo-labels (targets
  failure group #3, the largest), but costs a ~10-20h preprocessing pass → **Phase 3 only**.
- **pose-fuse** — a GLOBAL pose vector broadcast to all tokens (gated), **not** spatially per-patch
  despite the class name; needs ViTPose on the 36K gallery at inference too (+~0.5h/run). Off by default.
- **FILIP** — no extra labels (reuses ITC pairs); helps verbs (#2/#4). **NOT cheap at scale**: the
  `[B,B,Ni,Nt]` sim is ~5.5 GB at SigLIP @512 — use `filip_loss(..., chunk=N)` to bound memory.

The heads are **wired into `models/xvlm_v4.py` and trained by `trainer_xvlm.py`** here (with a
dummy backbone for tests).  For the strongest submission you may port the same heads/losses onto
the pretrained X-VLM in `..` (warm-start `best.pth`) to reuse its pretrained ITM.

---

## Roadmap (measure on distractor-val after each phase)

| phase | what | expected mAP |
|---|---|---|
| 0 | distractor-val harness (recall proxy) | baseline |
| 1 | **SigLIP-2-L + full 1M synthetic + augmentation** | ~88-91% |
| 2 | ensemble (RRF) + QE + k-reciprocal + TTA | ~90-93% |
| 3 | X-VLM-v4 heads (FILIP, anomaly, pose; GroundingDINO bbox if #3 heavy) | ~92-95% |

## Implementation status (honest)

- ✅ **Phase 0-3 implemented + three review passes applied. 81 tests pass** (pyflakes clean). All
  CLI entries run end-to-end on the dummy backbone (build_distractor_val, **zeroshot_baseline**,
  encode_cache, run_phase2, train_siglip, train_xvlm, build_box_pseudolabels).
- ✅ **SigLIP real path now ACTUALLY exercised:** a tiny real `SiglipModel` is loaded in tests —
  LoRA attaches (targets q/k/v/out_proj), forward returns aligned (norm-1) features,
  loss back-props through LoRA. (Skips automatically if the HF hub is unreachable.)
- ✅ **Pass-1 fixes:** pooler embeddings + **pretrained logit_scale/bias** (re-enabled under PEFT);
  **separate image+text queues**; **cosine-warmup LR**; **keypoints threaded to inference**;
  eval hook + grad-norm; box/pose **off by default**.
- ✅ **Pass-2 fixes:** **X-VLM grad-checkpointing** enabled; **CLIs save the BEST checkpoint** via
  `--val-index/--val-query-json/--eval-every` (verified: "saved BEST mAP=… @stepN");
  **FILIP `chunk=`** to cap the memory bomb; **`run_phase2 --weights`** for weighted fusion;
  honest docstrings for **pose-fuse (global)** and **mining (offline utility)**.
- ✅ **Pass-3 (pragmatic / no-waste):** **right-size** configs (batch 64, 40 GB target, drop
  "fill 80 GB"); **`siglip_so400m_384.yaml`** cheaper backbone option; **`xvlm_rerank_only.yaml`**
  + trainer gating (w_itc=0 ⇒ skip ITC/queue; w_filip/w_smoothap=0 ⇒ skip) so the reranker trains
  ITM-only; **`--val-max-distractors`** to subsample distractors per eval.
- ⚠️ **Still not run here** (needs GPU + data): real **siglip2-large** download, **Swin/BERT**
  (and any `.bin`-only model fails — transformers 5.8 requires **safetensors**), 1M-scale train.
  **Run `zeroshot_baseline` first.**
- ⛔ **Recommended / heavier:** warm-start the **pretrained X-VLM in `..`** (the strongest
  Phase-3 cross-encoder path); real **GroundingDINO** for box labels; **ViTPose** extraction.

> **Retractions after validation (being honest about my own review):** (1) the claimed
> "bf16 × fp32 queue crash" is **WRONG** — torch 2.6 `cat` type-promotes, `info_nce` runs fine;
> (2) SigLIP retrieval was **not** "broken" — `pooler_output` is already the aligned space; the
> real wins were the pretrained logit_scale/bias and robustness. **94% remains unproven** — the
> #1 risk is still unvalidated sim2real, and the X-VLM from-scratch cross-encoder is the weak link.
