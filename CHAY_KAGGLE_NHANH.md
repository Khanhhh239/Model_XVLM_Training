# 🚀 HƯỚNG DẪN CHẠY KAGGLE NHANH (TIẾNG VIỆT)

## TL;DR - 3 Bước Chính

### Bước 1: Chuẩn Bị Data (Local)
```bash
# Đảm bảo có đủ files:
data/
├── checkpoints/best.pth                    # Checkpoint mAP 80%
├── train_30k_hard.jsonl                    # Manifest 30K samples
├── train_30k_hard_vitpose.json             # Pose keypoints
├── boxes_30k.jsonl                         # Bounding boxes (đang chạy)
└── train_webp/                             # Folder ảnh .webp (30K ảnh)
```

**Nén để upload nhanh**:
```bash
zip -r train_webp.zip data/train_webp/
```

### Bước 2: Upload Lên Kaggle Dataset

#### Cách 1: Web UI (Dễ nhất)
1. Vào https://www.kaggle.com/datasets
2. Click **"New Dataset"**
3. Upload files:
   - `best.pth` → Dataset tên "xvlm-checkpoints"
   - `train_30k_hard.jsonl`, `train_30k_hard_vitpose.json`, `boxes_30k.jsonl`, `train_webp.zip` → Dataset tên "train-30k-hard"

#### Cách 2: CLI (Nhanh hơn)
```bash
pip install kaggle
python scripts/kaggle_upload_dataset.py \
    --checkpoint data/checkpoints/best.pth \
    --data-dir data/train_30k_hard \
    --checkpoint-dataset your-username/xvlm-checkpoints \
    --data-dataset your-username/train-30k-hard
```

### Bước 3: Chạy Notebook Trên Kaggle

1. Vào https://www.kaggle.com/code
2. Click **"New Notebook"**
3. Chọn **GPU T4 x2** + Enable **Internet**
4. Click **"Add Data"** → Add 2 datasets vừa upload
5. Import notebook: `notebooks/kaggle_stage1_30k.ipynb`
6. Click **"Run All"**

⏱️ **Thời gian**: ~2-3 giờ training

---

## Chi Tiết Từng Cell

### Cell 1: Clone Repo
```python
!git clone https://github.com/Khanhhh239/Model_XVLM_Training.git
%cd Model_XVLM_Training/trainv4
!pip install -q -r requirements.txt albumentations
!pip install -q -e .
```

### Cell 2: Link Data
```python
# Unzip nếu upload dạng .zip
!unzip -q /kaggle/input/train-30k-hard/train_webp.zip -d data/

# Symlink datasets
!ln -sf /kaggle/input/train-30k-hard/* data/
!ln -sf /kaggle/input/xvlm-checkpoints/best.pth data/checkpoints/
```

### Cell 3: Sanity Check (QUAN TRỌNG!)
```python
# Test pipeline: Loss phải giảm từ 2.0 → 0.01
!python scripts/train.py \
    --config configs/stage1_30k_kaggle_t4.yaml \
    --init-from data/checkpoints/best.pth \
    --overfit-one-batch
```

**Expected**: Loss giảm nhanh sau 200 steps
- ✅ Pass → Tiếp tục
- ❌ Fail → Debug (check paths, data)

### Cell 4: TRAINING! 🔥
```python
!python scripts/train.py \
    --config configs/stage1_30k_kaggle_t4.yaml \
    --init-from data/checkpoints/best.pth \
    --max-hours 11.5
```

### Cell 5: Download Checkpoint
```python
import shutil
shutil.copy("outputs/stage1_30k_t4/best.pth", "/kaggle/working/stage1_best.pth")
# File sẽ ở Output tab → Click Download
```

---

## Expected Results

| Metric | Baseline (best.pth) | Target (STAGE 1) | Gain |
|--------|---------------------|------------------|------|
| mAP    | 80.0%              | **82-84%**       | +2-4% |
| R@1    | ~75%               | ~77-79%          | +2-4% |
| MRR    | ~80%               | ~82-84%          | +2-4% |

**Success criteria**: mAP ≥ 82%

---

## Troubleshooting Nhanh

### 1. "Checkpoint not found"
```python
# Check path
!ls -lh /kaggle/input/xvlm-checkpoints/
# Fix: Adjust symlink trong Cell 2
```

### 2. "CUDA OOM"
```yaml
# Edit configs/stage1_30k_kaggle_t4.yaml
train:
  batch_size: 12  # Giảm từ 16 → 12
```

### 3. "Data manifest not found"
```python
# Check files uploaded
!ls -lh /kaggle/input/train-30k-hard/
```

### 4. Training chậm (stuck)
```python
# Copy ảnh vào RAM disk (nhanh hơn)
!mkdir -p /tmp/images
!cp -r data/train_webp/* /tmp/images/
# Sửa config: image_root: /tmp/images/
```

### 5. Kaggle timeout (12h limit)
```python
# Resume từ checkpoint
!python scripts/train.py \
    --resume outputs/stage1_30k_t4/last.pth \
    --max-hours 11.5
```

---

## Monitoring

### Xem logs real-time:
```python
!tail -f outputs/stage1_30k_t4/train.log
```

### Check GPU:
```python
!nvidia-smi
```

### Expected loss trajectory:
- **Step 0-500**: Loss ~1.5 (warmup)
- **Step 500-2000**: Loss ~0.8 (learning)
- **Step 2000+**: Loss ~0.5-0.6 (converging)

---

## Next Steps Sau Training

### ✅ Nếu mAP ≥ 82% (Success!)
1. Download `stage1_best.pth`
2. Evaluate trên validation set
3. Tiếp tục **STAGE 2**: Unfreeze vision encoder cho +2-3% nữa

### ⚠️ Nếu mAP 80-82% (Tạm ổn)
1. Review loss curves
2. Điều chỉnh hyperparameters:
   - Giảm LR: `1.5e-5` → `1e-5`
   - Tăng loss weights: `lambda_box`, `lambda_anomaly`
3. Train thêm 2-3 epochs

### ❌ Nếu mAP < 80% (Revert)
1. Safety net đã tự động revert về best.pth
2. Check logs: `outputs/stage1_30k_t4/train.log`
3. Debug: Data quality, augmentation quá mạnh, LR quá cao

---

## Files Quan Trọng

- 📘 **KAGGLE_SETUP_GUIDE.md**: Hướng dẫn chi tiết
- 📓 **notebooks/kaggle_stage1_30k.ipynb**: Notebook chạy trên Kaggle
- ⚙️ **configs/stage1_30k_kaggle_t4.yaml**: Config file
- 🐚 **scripts/kaggle_quick_start.sh**: Test local trước khi upload
- 🔧 **scripts/train.py**: Main training script

---

## Tips & Tricks

1. ✅ **Always test local first**: Chạy `kaggle_quick_start.sh` trước
2. ✅ **Monitor early**: Nếu loss không giảm sau 500 steps → Stop & debug
3. ✅ **Save often**: Checkpoint auto-save mỗi 0.5 epoch
4. ✅ **Don't close browser**: Kaggle kill session nếu inactive
5. ✅ **Use wandb** (optional): Set `log_wandb: true` để track metrics

---

## Contact

Gặp lỗi? Copy **full error message** + **logs** và hỏi lại!

Good luck! 💪🚀
