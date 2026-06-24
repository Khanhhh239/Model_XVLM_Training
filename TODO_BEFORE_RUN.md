# ✅ TODO Before Running STAGE 1

## Data Preparation

- [ ] **train_30k_hard.jsonl**: 
  - Format: JSONL với fields: `image_path`, `caption`, `label_type`, `video_id`, `sequence_id`
  - `label_type`: "goal" (normal) hoặc "wentwrong" (anomaly)
  - Path: `data/train_30k_hard/train_30k_hard.jsonl`

- [ ] **train_30k_hard_vitpose.json**:
  - Format: Dict `{image_path: [x1,y1,c1, ..., x17,y17,c17]}`  (51 floats)
  - Path: `data/train_30k_hard/train_30k_hard_vitpose.json`

- [ ] **boxes_30k.jsonl**:
  - Format: JSONL, each line: `{"image_path": "...", "bbox": [x, y, w, h]}`
  - Bbox normalized to [0, 1] (COCO format)
  - Path: `data/train_30k_hard/boxes_30k.jsonl`
  - **Status**: BẠN NÓI "đang chạy" - **CẦN XÁC NHẬN XONG!**

- [ ] **train_webp/** folder:
  - Chứa tất cả ảnh `.webp` (384×384 pixels)
  - Path: `data/train_30k_hard/train_webp/`

- [ ] **best.pth** checkpoint (mAP 80%):
  - Warm-start checkpoint
  - Path: `data/checkpoints/best.pth`
  - **CẦN UPLOAD lên Kaggle Dataset hoặc local!**

---

## Config Verification

- [ ] Update paths trong `configs/stage1_30k_kaggle_t4.yaml`:
  ```yaml
  data:
    manifest: data/train_30k_hard/train_30k_hard.jsonl  # ← CHECK!
    image_root: data/train_30k_hard/train_webp/         # ← CHECK!
    vitpose_json: data/train_30k_hard/train_30k_hard_vitpose.json  # ← CHECK!
    boxes_json: data/train_30k_hard/boxes_30k.jsonl     # ← CHECK! (nếu chưa xong, set null)
  
  model:
    checkpoint: data/checkpoints/best.pth                # ← CHECK!
  ```

- [ ] Verify keypoints format in vitpose JSON:
  ```python
  import json
  with open('data/train_30k_hard/train_30k_hard_vitpose.json') as f:
      data = json.load(f)
  # Should be: {"image/path.webp": [51 floats]}
  print(list(data.keys())[:3])
  print(len(list(data.values())[0]))  # Should be 51
  ```

- [ ] Verify bbox format in boxes JSONL:
  ```python
  import json
  with open('data/train_30k_hard/boxes_30k.jsonl') as f:
      line = f.readline()
      item = json.loads(line)
  # Should be: {"image_path": "...", "bbox": [x, y, w, h]}
  print(item)
  assert len(item['bbox']) == 4
  assert all(0 <= v <= 1 for v in item['bbox'])  # Normalized
  ```

---

## Dependencies

- [ ] Install requirements:
  ```bash
  pip install -r requirements.txt
  pip install albumentations  # Essential for strong augmentation!
  ```

- [ ] Verify albumentations installed:
  ```python
  import albumentations as A
  print(A.__version__)  # Should be >= 1.3.0
  ```

---

## Sanity Checks

- [ ] **Load config without errors**:
  ```bash
  python -c "from star.config import load_config; cfg = load_config('configs/stage1_30k_kaggle_t4.yaml'); print('✓ Config OK')"
  ```

- [ ] **Load model without errors**:
  ```bash
  python -c "from star.models import STARModel; from star.config import load_config; cfg = load_config('configs/stage1_30k_kaggle_t4.yaml'); model = STARModel(cfg); print('✓ Model OK')"
  ```

- [ ] **Load dataset without errors**:
  ```bash
  python -c "from star.data import PABDataset; from star.config import load_config; from transformers import BertTokenizer; cfg = load_config('configs/stage1_30k_kaggle_t4.yaml'); tokenizer = BertTokenizer.from_pretrained('bert-base-uncased'); ds = PABDataset(cfg.data.manifest, cfg.data.image_root, tokenizer, vitpose_json=cfg.data.vitpose_json, boxes_json=cfg.data.boxes_json); print(f'✓ Dataset OK ({len(ds)} samples)')"
  ```

- [ ] **Overfit one batch** (loss should drop):
  ```bash
  python scripts/train.py \
      --config configs/stage1_30k_kaggle_t4.yaml \
      --init-from data/checkpoints/best.pth \
      --overfit-one-batch
  # Expected: loss 2.96 → 0.23 in ~200 steps
  ```

---

## Kaggle-Specific

- [ ] **Upload best.pth to Kaggle Dataset**:
  - Create dataset "xvlm-checkpoints"
  - Upload `best.pth` (~500MB)
  - Set visibility: Private

- [ ] **Upload train_30k_hard to Kaggle Dataset**:
  - Create dataset "train-30k-hard"
  - Upload all 4 files + train_webp folder
  - Set visibility: Private

- [ ] **Update notebook paths**:
  - Edit `notebooks/kaggle_stage1_30k.ipynb`
  - Cell 2: Update `checkpoint_source` path
  - Cell 2: Update `data_source` path

- [ ] **Test notebook on Kaggle**:
  - Settings: GPU T4 × 1, Internet ON
  - Run first 4 cells (setup + data + sanity check)
  - Verify no errors

---

## Before Full Training

- [ ] **Check VRAM usage** (should be < 14GB):
  ```bash
  # Run overfit-one-batch and monitor:
  nvidia-smi
  # If VRAM > 14GB → reduce batch_size in config
  ```

- [ ] **Estimate training time**:
  - Run 1 epoch, measure time: `time python scripts/train.py --config ... --set optim.epochs=1`
  - Expected: ~25 phút/epoch × 3 epochs = ~1.5 giờ
  - If > 2 giờ/epoch → investigate (slow I/O? CPU bottleneck?)

- [ ] **Verify safety net**:
  ```python
  # Manually edit code to force mAP drop:
  # trainer.py line ~180: best_metric = 0.75  # Force < 80%
  # Should trigger: "mAP dropped! Reverting..."
  ```

---

## Optional (but recommended)

- [ ] **Write simple unit tests**:
  ```bash
  # Test XBM queue enqueue/dequeue
  # Test box head forward pass
  # Test anomaly head forward pass
  pytest tests/ -v
  ```

- [ ] **Set up Weights & Biases logging** (optional):
  ```yaml
  # configs/stage1_30k_kaggle_t4.yaml
  train:
    log_wandb: true
  ```
  Then: `wandb login` before training

- [ ] **Prepare resume strategy** (for Kaggle 12h limit):
  ```bash
  # If training interrupted, resume from last.pth:
  python scripts/train.py \
      --config configs/stage1_30k_kaggle_t4.yaml \
      --resume outputs/stage1_30k_t4/last.pth
  ```

---

## Final Checklist

Before running `python scripts/train.py --config configs/stage1_30k_kaggle_t4.yaml --init-from data/checkpoints/best.pth`:

- [ ] All data files present và verified
- [ ] Config paths updated
- [ ] Dependencies installed (especially albumentations!)
- [ ] Sanity checks passed (overfit-one-batch works)
- [ ] VRAM usage < 14GB
- [ ] Estimated time < 3 giờ (fits in Kaggle 12h window)

---

**If all checkboxes are ✅ → Ready to train! 🚀**

Nếu có bất kỳ checkbox nào là ❌ → Fix trước khi train, tránh lãng phí GPU time!
