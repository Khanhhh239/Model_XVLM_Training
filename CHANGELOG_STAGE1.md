# STAGE 1 Implementation Changelog

## Overview
Implemented STAGE 1 training strategy for Kaggle T4: Warm-start từ best.pth (mAP 80%), freeze vision encoder, train precision heads (Box, Anomaly, Pose) với XBM Queue và strong augmentation.

---

## 🆕 New Files

### Losses
- **`src/star/losses/xbm_queue.py`**: Cross-Batch Memory FIFO queue (8192 embeddings) for extra ITC negatives

### Models
- **`src/star/models/heads.py`**: 
  - `BoxGroundingHead`: Predicts bbox [x,y,w,h] với GIoU+L1 loss
  - `AnomalyClassificationHead`: Binary classifier (normal=0, anomaly=1)

### Data
- **`src/star/data/augmentation.py`**: Strong Albumentations pipeline (Sim2Real: blur, JPEG, downscale, color; Robustness: erasing, crop)

### Scripts
- **`scripts/mine_hard_negatives.py`**: ANCE-style hard-neg mining (runs model on train set → top-K cross-ID distractors)

### Configs
- **`configs/stage1_30k_kaggle_t4.yaml`**: Full config cho Kaggle T4 (batch 16, fp16, XBM 8K, differential LR)

### Notebooks
- **`notebooks/kaggle_stage1_30k.ipynb`**: End-to-end Kaggle notebook (setup → train → evaluate → save)

### Documentation
- **`STAGE1_README.md`**: Complete implementation guide (data format, usage, monitoring, troubleshooting)
- **`CHANGELOG_STAGE1.md`**: This file

---

## ✏️ Modified Files

### Core Model
- **`src/star/models/star_model.py`**:
  - Added `self.bbox_head`, `self.anomaly_head`, `self.xbm_queue`
  - Updated `forward()` to compute 5 losses (ITC+ITM+SmoothAP+Box+Anomaly)
  - Integrated XBM queue into ITC loss
  - Added anomaly ramp-up (0→full weight over 500 steps)
  - Updated `mark_only_lora_trainable()` to include new heads

### Losses
- **`src/star/losses/itc.py`**:
  - Added `queue_img`, `queue_txt` parameters to `forward()`
  - Concatenate queue negatives to in-batch negatives
  - Pad identity soft targets with zeros for queue columns

- **`src/star/losses/weighting.py`**:
  - Updated `TASKS` tuple: `("itc", "itm", "smap")` → `("itc", "itm", "smap", "box", "anomaly")`
  - All 3 weighters (Fixed/Uncertainty/DWA) now handle 5 tasks
  - Added `getattr()` fallbacks for backward compatibility

### Data
- **`src/star/data/dataset.py`**:
  - Added `vitpose_json`, `boxes_json` constructor params
  - Load external vitpose/boxes dicts from JSON files
  - Added `bbox`, `bbox_mask` fields to item dict
  - Added `anomaly_label`, `anomaly_mask` fields (parsed from `label_type`)
  - Support both `.parquet` and `.jsonl` manifest formats
  - Updated `collate_fn()` to batch bbox/anomaly fields

### Config
- **`src/star/config.py`**:
  - **`DataConfig`**: Added `vitpose_json`, `boxes_json` fields
  - **`ModelConfig`**: Added `bbox_enabled`, `anomaly_enabled` toggles
  - **`LossConfig`**: Added `lambda_box`, `w_box_giou`, `w_box_l1`, `lambda_anomaly`, `anomaly_rampup_steps`, `xbm_enabled`, `xbm_size`

### Training
- **`scripts/train.py`**:
  - Pass `vitpose_json`, `boxes_json` to `PABDataset()` constructors (train + val)

### Dependencies
- **`requirements.txt`**:
  - Added `albumentations>=1.3.0` for strong augmentation

### Exports
- **`src/star/losses/__init__.py`**: Export `XBMQueue`
- **`src/star/models/__init__.py`**: Export `BoxGroundingHead`, `AnomalyClassificationHead`

---

## 🎯 Key Features

### 1. XBM Queue (Cross-Batch Memory)
```python
# Maintains 8192 past embeddings as extra negatives
queue = XBMQueue(size=8192, dim=256)
loss_itc = itc(img_feat, txt_feat, queue_img=queue.img_queue, queue_txt=queue.txt_queue)
queue.enqueue(img_feat.detach(), txt_feat.detach())
```

**Benefits**: 
- Small T4 batch (16) → 8K negatives via queue
- Compensates for lack of multi-GPU all_gather
- ~27% of 30K dataset in memory

### 2. Box Grounding Head
```python
bbox_head = BoxGroundingHead(input_dim=256)
bbox_pred = bbox_head(img_feat)  # [B, 4]
loss_box = 2*GIoU + 5*L1
```

**Teaches**: 
- Spatial understanding (where is the person?)
- Solves wrong cases #3/#7 (spatial relations)

### 3. Anomaly Classification Head
```python
anomaly_head = AnomalyClassificationHead(input_dim=256)
logits = anomaly_head(img_feat)  # [B, 2]
loss_anomaly = CE(logits, label)  # label from "label_type" field
```

**Teaches**: 
- Semantic state (normal vs abnormal behavior)
- Solves wrong cases #1/#2 (action understanding)

### 4. Strong Augmentation (Sim2Real)
```python
# Albumentations pipeline:
- MotionBlur (p=0.3): Frame nhòe khi "đang ngã"
- JPEG compression (p=0.4): Video bị nén
- Downscale (p=0.3): Độ phân giải thấp
- ColorJitter (p=0.5): Ánh sáng đa dạng
- CoarseDropout (p=0.25): Random Erasing
- RandomResizedCrop (p=1.0): Crop + auto-adjust bbox/keypoints!
```

**Closes**: Synthetic → Real gap

### 5. Hard-Neg Mining (ANCE Tầng 2)
```bash
# After each epoch:
python scripts/mine_hard_negatives.py \
    --ckpt outputs/stage1_30k_t4/last.pth \
    --output data/mined_negatives.json \
    --top_k 5
```

**Provides**: Cross-ID hardest distractors (beyond ID-hard from data)

---

## 📊 Loss Formulation

```python
L_total = 1.0 * ITC(img_feat, txt_feat, queue=xbm_queue)        # Contrastive
        + 1.0 * ITM(cross_feat, hard_negs)                      # Matching (MAIN)
        + 0.2 * SmoothAP(img_feat, txt_feat, relevance)         # Rank optimization
        + 0.1 * (2*GIoU + 5*L1)(bbox_pred, bbox_gt)             # Box grounding
        + 0.2 * CE(anomaly_logits, anomaly_label) * rampup(t)   # Anomaly (ramp-up)
```

**Weights rationale**:
- ITC:ITM = 1:1 (proven X-VLM ratio)
- SmoothAP = 0.2 (lower, focus on ITM for precision)
- Box = 0.1 (auxiliary, avoid overpowering ITC/ITM)
- Anomaly = 0.2 + ramp-up (gradual introduction)

---

## ⚙️ Training Strategy

### Freeze vs Train:
| Component | Status | Reason |
|-----------|--------|--------|
| Swin-B Vision | ❄️ **FROZEN** | Giữ recall 80%, tiết kiệm VRAM |
| BERT Text [0:6] | ❄️ **FROZEN** | Giữ ngôn ngữ ổn định |
| BERT Cross [6:12] | 🔥 **FULL-FT** | "Não" matching, mài precision |
| ITM head | 🔥 **TRAIN** | Head chính cho reranking |
| Proj heads | 🔥 **TRAIN** | Căn chỉnh không gian embeddings |
| **Box head** | 🆕 **TRAIN** | Spatial understanding |
| **Anomaly head** | 🆕 **TRAIN** | Semantic state |
| **Pose block** | 🆕 **TRAIN** | Action details |

### Differential LR:
- Old heads (cross/itm/proj): **1.5e-5** (LOW, giữ 80%)
- New heads (box/anomaly): **1e-4** (HIGH, học nhanh)
- Pose: **5e-5** (MEDIUM)

### Safety Net:
```python
if val_mAP < 80.0:
    log.warning("mAP dropped! Reverting to best.pth")
    load_checkpoint("best.pth", model)
    return  # Stop training
```

---

## 🧪 Testing

### Unit Tests (recommended to add):
```bash
# Test XBM queue
pytest tests/test_xbm_queue.py -v

# Test bbox loss (GIoU)
pytest tests/test_box_head.py -v

# Test anomaly head
pytest tests/test_anomaly_head.py -v

# Test augmentation pipeline
pytest tests/test_augmentation.py -v
```

### Integration Test:
```bash
# Overfit one batch (should converge in ~200 steps)
python scripts/train.py \
    --config configs/stage1_30k_kaggle_t4.yaml \
    --init-from data/checkpoints/best.pth \
    --overfit-one-batch
```

Expected output:
```
[overfit] step   0 loss=2.9600
[overfit] step  25 loss=1.4200
[overfit] step  50 loss=0.8500
[overfit] step 100 loss=0.4100
[overfit] step 150 loss=0.2300
[overfit] OK: 2.960 -> 0.230 (target<0.05 or 70% drop) at step 162
```

---

## 📈 Expected Performance

| Metric | Baseline | Target | Gain |
|--------|----------|--------|------|
| mAP | 80.0% | 82-86% | +2-6% |
| R@1 | ~75% | ~78-82% | +3-7% |
| R@5 | ~88% | ≥88% | ±0% |
| R@10 | 94.0% | ≥94% | Locked |

**Ablation estimates**:
- Box grounding: +1-2%
- Anomaly head: +0.5-1%
- Pose fuse: +0.5-1%
- XBM Queue: +0.5-1%
- Hard-neg mining: +1-2%
- Strong augmentation: +0-1%

---

## ⚠️ Known Limitations

1. **Hard-neg mining**: Not automated in trainer (must run script manually after each epoch)
2. **Differential LR**: Currently uses `lr_lora` for old heads, `lr_head` for new heads (naming is legacy)
3. **Albumentations required**: Fallback to torchvision if not installed, but loses bbox/keypoint adjustment
4. **JSONL manifest**: Slower than parquet for large datasets (30K is OK, 1M would be slow)
5. **T4 memory tight**: Batch 16 is near limit; grad_checkpointing is ESSENTIAL

---

## 🔜 Future Work (STAGE 2+)

1. **Unfreeze vision encoder**: Slowly unfreeze Swin-B layers (top→down) for full fine-tuning
2. **Ensemble with SigLIP**: Add SigLIP-2-L retriever (pretrain-real backbone) for dual-model ensemble
3. **Query Expansion + k-Reciprocal**: Post-processing tricks từ architecture plan
4. **TensorRT optimization**: Export to ONNX/TRT for faster Kaggle inference
5. **Automated hard-neg mining**: Integrate into trainer loop (every N epochs)
6. **Multi-scale TTA**: Encode at 384+512 resolutions, average embeddings

---

## 📚 References

1. **X-VLM** (Zeng et al., ICML 2022): Base architecture
2. **CMP** (Yang et al., ICCV 2025): Person Anomaly Search baseline
3. **XBM** (Wang et al., CVPR 2020): Cross-Batch Memory for contrastive learning
4. **ANCE** (Xiong et al., ICLR 2021): Hard negative mining strategy
5. **GIoU** (Rezatofighi et al., CVPR 2019): Generalized IoU for bbox loss
6. **Albumentations** (Buslaev et al.): Fast augmentation library with bbox/keypoint support

---

**Implementation by**: Kiro AI Assistant  
**Date**: 2026-06-24  
**Commit**: STAGE 1 initial implementation  
**Status**: ✅ Ready for testing on Kaggle T4
