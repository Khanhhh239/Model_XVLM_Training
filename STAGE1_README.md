# STAGE 1: Kaggle T4 Training - Implementation Guide

## 🎯 Overview

STAGE 1 mài **precision** từ mAP 80% → 82-86% bằng:
- **Warm-start**: Load best.pth (mAP 80%)
- **Freeze**: Vision encoder (Swin-B) đóng băng 100%
- **Train**: Cross-Attention + ITM head + Box/Anomaly heads + Pose + XBM Queue
- **Safety net**: Revert nếu mAP < 80%

**Estimated time**: ~2 giờ trên Kaggle T4 (30K dataset, 3 epochs)

---

## 📦 Data Requirements

Cần 3 files trong folder `data/train_30k_hard/`:

1. **train_30k_hard.jsonl** (30,000 samples)
   ```json
   {
     "image_path": "action/video_123/frame_001.webp",
     "caption": "A person falling off a skateboard",
     "label_type": "wentwrong",  // ← "goal" (normal) or "wentwrong" (anomaly)
     "video_id": "video_123",
     "sequence_id": "video_123_001",
     "hard_image_path": "action/video_123/frame_050.webp",  // Cùng người, khác trạng thái
     "hard_caption": "A person standing with a skateboard"
   }
   ```

2. **train_30k_hard_vitpose.json** (Keypoints)
   ```json
   {
     "action/video_123/frame_001.webp": [
       x1, y1, conf1,  // Joint 0: nose
       x2, y2, conf2,  // Joint 1: left_eye
       ...             // Total: 17 joints × 3 = 51 floats
     ]
   }
   ```

3. **boxes_30k.jsonl** (Bounding boxes)
   ```jsonl
   {"image_path": "action/video_123/frame_001.webp", "bbox": [0.3, 0.2, 0.4, 0.6]}
   {"image_path": "action/video_123/frame_002.webp", "bbox": [0.25, 0.15, 0.45, 0.65]}
   ```
   Format: `[x, y, w, h]` normalized to [0, 1] (COCO format)

4. **train_webp/** folder chứa tất cả ảnh `.webp` (384×384 pixels)

---

## 🚀 Quick Start

### 1. Local Testing (có GPU)

```bash
# Clone repo
git clone https://github.com/Khanhhh239/Model_XVLM_Training.git
cd Model_XVLM_Training/trainv4

# Install dependencies
pip install -r requirements.txt
pip install albumentations

# Chuẩn bị data (adjust paths)
# - Copy best.pth → data/checkpoints/best.pth
# - Copy train_30k_hard files → data/train_30k_hard/

# Sanity check (overfit 1 batch)
python scripts/train.py \
    --config configs/stage1_30k_kaggle_t4.yaml \
    --init-from data/checkpoints/best.pth \
    --overfit-one-batch

# Full training
python scripts/train.py \
    --config configs/stage1_30k_kaggle_t4.yaml \
    --init-from data/checkpoints/best.pth

# Evaluate
python scripts/evaluate.py \
    --config configs/stage1_30k_kaggle_t4.yaml \
    --ckpt outputs/stage1_30k_t4/best.pth
```

### 2. Kaggle T4 Training

Upload notebook `notebooks/kaggle_stage1_30k.ipynb` lên Kaggle:
1. Settings: **GPU T4 × 1**, **Internet ON**
2. Add datasets:
   - `xvlm-checkpoints` (chứa best.pth)
   - `train-30k-hard` (chứa 4 files trên)
3. Run all cells → ~2 giờ

---

## 🏗️ Architecture Changes (so với trainv4 baseline)

### New Components:

1. **XBM Queue** (`src/star/losses/xbm_queue.py`)
   - FIFO buffer [8192, 256] lưu past embeddings
   - Provides ~8K extra negatives cho ITC loss
   - Bù đắp batch size nhỏ trên T4

2. **Box Grounding Head** (`src/star/models/heads.py`)
   - MLP: 256 → 256 → 4 (bbox [x, y, w, h])
   - Loss: 2×GIoU + 5×L1
   - Teaches spatial understanding (#3/#7 trong wrong cases)

3. **Anomaly Classification Head** (`src/star/models/heads.py`)
   - MLP: 256 → 128 → 2 (normal vs anomaly)
   - Loss: Cross-Entropy
   - Teaches semantic state (#1/#2 trong wrong cases)

4. **Strong Augmentation** (`src/star/data/augmentation.py`)
   - Sim2Real: MotionBlur, JPEG, Downscale, ColorJitter, Noise
   - Robustness: CoarseDropout (Random Erasing), RandomResizedCrop
   - Uses **Albumentations** → auto-adjust bbox/keypoints!

5. **Hard-neg Mining Script** (`scripts/mine_hard_negatives.py`)
   - Runs model on train set mỗi epoch
   - Extracts top-K cross-ID distractors
   - Feeds vào ITM loss (tầng 2 hard-neg)

### Modified Files:

- `src/star/models/star_model.py`: Tích hợp 5 losses (ITC+ITM+SmoothAP+Box+Anomaly)
- `src/star/losses/itc.py`: Support XBM queue negatives
- `src/star/losses/weighting.py`: Handle 5 tasks thay vì 3
- `src/star/data/dataset.py`: Đọc bbox, label_type, external vitpose/boxes JSON
- `src/star/config.py`: Thêm tham số mới (xbm_size, bbox_enabled, etc.)
- `scripts/train.py`: Pass vitpose_json, boxes_json vào dataset

---

## ⚙️ Key Hyperparameters

| Param | Value | Rationale |
|-------|-------|-----------|
| `batch_size` | 16 | T4 16GB, fallback 12/8 nếu OOM |
| `lr_lora` | 1.5e-5 | **LOW** cho old heads (giữ recall 80%) |
| `lr_head` | 1e-4 | **HIGH** cho new heads (học nhanh) |
| `lambda_itm` | 1.0 | ITM là loss CHÍNH cho precision |
| `lambda_box` | 0.1 | Nhẹ, tránh át ITC/ITM |
| `lambda_anomaly` | 0.2 | + ramp-up 500 steps |
| `xbm_size` | 8192 | ~27% của 30K, cân bằng freshness vs coverage |
| `epochs` | 8 | Early stop thường ~2-3 |
| `early_stop_patience` | 1 | STRICT: tụt 1 lần là dừng |
| `grad_clip` | 5.0 | Cao hơn baseline (nhiều heads mới) |

---

## 🔍 Monitoring & Debugging

### 1. Loss Balance
```bash
# Log grad norms every 200 steps
# Check if box/anomaly heads dominate (bad) or balanced (good)
python scripts/train.py --config ... --set train.grad_norm_every=200
```

Expected balance (grad norm %):
- ITC: ~30-40%
- ITM: ~30-40%
- Box: ~10-15%
- Anomaly: ~5-10%
- SmoothAP: ~5-10%

Nếu Box/Anomaly > 30% → giảm `lambda_box`, `lambda_anomaly`

### 2. Safety Net Trigger
```python
# Trong logs, tìm:
[VAL-B] mAP=0.7850 < 0.8000 → REVERT to best.pth!
```
→ Training stopped, reverted to mAP 80%

### 3. OOM Troubleshooting
```yaml
# Giảm batch size trong config:
train:
  batch_size: 12  # hoặc 8
  grad_accum: 2   # tăng lên để bù
```

### 4. XBM Queue Full Check
```python
# Trong logs step ~500:
XBM queue filled: 8192/8192 embeddings
```
→ Queue đầy, bắt đầu FIFO replace

---

## 📊 Expected Results

| Metric | Baseline (best.pth) | STAGE 1 Target | Gain |
|--------|---------------------|----------------|------|
| **mAP** | 80.0% | **82-86%** | +2-6% |
| **R@10** | 94.0% | **≥94%** (locked) | ±0% |
| **Precision** | Medium | **High** | ↑ |

### Ablation (dự kiến contribution):
- Box grounding: +1-2% (spatial understanding)
- Anomaly head: +0.5-1% (semantic state)
- Pose fuse: +0.5-1% (action details)
- XBM Queue: +0.5-1% (more negatives)
- Hard-neg mining: +1-2% (push distractors down)
- Augmentation: +0-1% (sim2real, dependent on backbone)

**Total**: +4-8% (optimistic), +2-4% (conservative)

---

## ⚠️ Known Issues & Limitations

1. **boxes_30k.jsonl chưa xong**: Dataset sẽ skip box loss (lambda_box=0 auto)
2. **Vitpose JSON format**: Phải là dict `{image_path: [51 floats]}`
3. **Albumentations coordinate transform**: Chỉ hoạt động nếu bbox trong [0, 1]
4. **T4 memory**: Nếu OOM, giảm batch_size hoặc tắt grad_checkpointing (trade memory for speed)
5. **Hard-neg mining**: Phải chạy manual sau mỗi epoch (chưa tự động trong trainer)

---

## 🔧 Customization

### Tắt một head nào đó:
```yaml
# configs/stage1_30k_kaggle_t4.yaml
model:
  bbox_enabled: false      # Tắt box head
  anomaly_enabled: false   # Tắt anomaly head
  pose_enabled: false      # Tắt pose
```

### Thay đổi augmentation strength:
```python
# src/star/data/augmentation.py
# Giảm prob:
A.MotionBlur(blur_limit=7, p=0.1),  # 0.3 → 0.1
A.ImageCompression(..., p=0.2),      # 0.4 → 0.2
```

### Differential LR cho từng head:
```python
# src/star/engine/optim.py (build_optimizer)
param_groups = [
    {"params": cross_attn_params, "lr": 1.5e-5},
    {"params": itm_head_params, "lr": 1.5e-5},
    {"params": box_head_params, "lr": 1e-4},
    {"params": anomaly_head_params, "lr": 1e-4},
    {"params": pose_params, "lr": 5e-5},
]
```

---

## 🎉 Next Steps (nếu thành công)

1. **Push checkpoint lên Git LFS**: `stage1_best.pth`
2. **STAGE 2 (optional)**: Unfreeze vision encoder, train full model (cần A100)
3. **Ensemble**: Kết hợp với SigLIP retriever (dual-model như architecture plan)
4. **Hard-neg mining automation**: Thêm vào trainer loop (mỗi epoch tự động mine)
5. **Inference optimization**: TensorRT, ONNX export cho Kaggle submission

---

## 📚 References

- [X-VLM Paper](https://arxiv.org/abs/2111.08276) - Base architecture
- [CMP Paper](https://arxiv.org/abs/2411.17776) - Person Anomaly Search baseline
- [XBM Paper](https://arxiv.org/abs/1912.06798) - Cross-Batch Memory
- [Albumentations Docs](https://albumentations.ai/docs/) - Augmentation library
- [ANCE Paper](https://arxiv.org/abs/2007.00808) - Hard negative mining

---

**Author**: Kiro AI Assistant  
**Date**: 2026-06-24  
**Version**: 1.0
