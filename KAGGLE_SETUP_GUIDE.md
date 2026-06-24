# 🚀 Hướng Dẫn Chạy STAGE 1 Trên Kaggle

## 📋 Chuẩn Bị Trước Khi Chạy

### 1. Upload Data lên Kaggle Dataset

Tạo 2 Kaggle Datasets riêng biệt:

#### Dataset 1: `xvlm-checkpoints`
```
xvlm-checkpoints/
└── best.pth           # Checkpoint mAP 80% (warm-start)
```

**Cách upload**:
1. Vào https://www.kaggle.com/datasets
2. Click "New Dataset"
3. Upload file `best.pth` (từ local hoặc Google Drive)
4. Đặt tên: `xvlm-checkpoints`
5. Click "Create"

#### Dataset 2: `train-30k-hard`
```
train-30k-hard/
├── train_30k_hard.jsonl           # 30K hard samples manifest
├── train_30k_hard_vitpose.json    # Pose keypoints cho 30K samples
├── boxes_30k.jsonl                # Bounding boxes (đang chạy)
└── train_webp/                    # Folder chứa ảnh .webp
    ├── 000001.webp
    ├── 000002.webp
    └── ...
```

**Cách upload**:
1. Nén folder `train_webp/` thành `train_webp.zip` (để upload nhanh)
2. Vào https://www.kaggle.com/datasets
3. Click "New Dataset"
4. Upload tất cả files: `train_30k_hard.jsonl`, `train_30k_hard_vitpose.json`, `boxes_30k.jsonl`, `train_webp.zip`
5. Đặt tên: `train-30k-hard`
6. Click "Create"

**Lưu ý**: Nếu `boxes_30k.jsonl` chưa sẵn sàng, có thể upload sau và re-run notebook.

---

## 🚀 Chạy Training Trên Kaggle

### Bước 1: Tạo Kaggle Notebook

1. Vào https://www.kaggle.com/code
2. Click **"New Notebook"**
3. Chọn **"GPU T4 x2"** (free tier)
4. Enable **"Internet"** (để clone repo từ GitHub)

### Bước 2: Add Datasets vào Notebook

1. Click **"Add Data"** (bên phải notebook)
2. Search và add 2 datasets:
   - `your-username/xvlm-checkpoints`
   - `your-username/train-30k-hard`
3. Datasets sẽ available tại:
   - `/kaggle/input/xvlm-checkpoints/`
   - `/kaggle/input/train-30k-hard/`

### Bước 3: Copy Notebook Content

Option A: **Import notebook từ file** (khuyến nghị)
```bash
# Upload file notebooks/kaggle_stage1_30k.ipynb vào Kaggle
```

Option B: **Copy-paste từng cell** vào notebook mới

### Bước 4: Chạy Notebook

Chạy từng cell theo thứ tự:

#### Cell 1-2: Setup
```python
!git clone https://github.com/Khanhhh239/Model_XVLM_Training.git
%cd Model_XVLM_Training/trainv4
!pip install -q -r requirements.txt
!pip install -q albumentations
!pip install -q -e .
```
⏱️ **Thời gian**: ~3-5 phút

#### Cell 3: Link Data
```python
# Unzip train_webp nếu bạn upload dưới dạng .zip
!unzip -q /kaggle/input/train-30k-hard/train_webp.zip -d /kaggle/input/train-30k-hard/

# Symlink data
!ln -sf /kaggle/input/train-30k-hard data/train_30k_hard
!ln -sf /kaggle/input/xvlm-checkpoints data/checkpoints
```
⏱️ **Thời gian**: ~1 phút

#### Cell 4: Verify Config
```python
# Notebook sẽ tự động update paths trong config
```
⏱️ **Thời gian**: <1 phút

#### Cell 5: Sanity Check (QUAN TRỌNG!)
```python
!python scripts/train.py \
    --config configs/stage1_30k_kaggle_t4_updated.yaml \
    --init-from data/checkpoints/best.pth \
    --overfit-one-batch
```
⏱️ **Thời gian**: ~5-10 phút

**Expected output**: Loss giảm từ ~2.0 → ~0.01 sau 200 steps
- ✅ Nếu pass → Pipeline hoạt động, tiếp tục training
- ❌ Nếu fail → Debug (xem logs, check data paths)

#### Cell 6: FULL TRAINING 🔥
```python
!python scripts/train.py \
    --config configs/stage1_30k_kaggle_t4_updated.yaml \
    --init-from data/checkpoints/best.pth \
    --max-hours 11.5
```
⏱️ **Thời gian**: ~2 giờ (30K dataset, 3 epochs, T4)

**Safety net**:
- Training tự động save checkpoint mỗi epoch
- Nếu mAP < 80%, tự động revert về best.pth
- Nếu Kaggle timeout (12h), có thể resume từ `last.pth`

---

## 📊 Monitoring Training

### Xem Logs Real-time
```python
# Trong cell mới (khi training đang chạy)
!tail -f outputs/stage1_30k_t4/train.log
```

### Check GPU Usage
```python
!nvidia-smi
```

### Expected Metrics (Intermediate)
- **Epoch 0**: mAP ~80.0% (baseline từ best.pth)
- **Epoch 1**: mAP ~81-82% (bbox + anomaly heads học được patterns)
- **Epoch 2-3**: mAP ~82-84% (converge)
- **Target**: mAP ≥ 82% (improvement +2%)

---

## 💾 Lưu Checkpoint Sau Khi Training Xong

### Option 1: Download từ Kaggle Output
```python
# Cell cuối cùng trong notebook
import shutil
shutil.copy("outputs/stage1_30k_t4/best.pth", "/kaggle/working/stage1_best.pth")
```
- File sẽ xuất hiện trong **Output** tab (bên phải)
- Click **Download** để tải về

### Option 2: Save về Kaggle Dataset (cho lần chạy sau)
```python
# Create new dataset version với checkpoint mới
from kaggle import api
api.dataset_create_version(
    folder="/kaggle/working",
    version_notes="STAGE 1 trained model - mAP 82%",
    dataset="your-username/xvlm-checkpoints"
)
```

---

## 🐛 Troubleshooting

### Lỗi: "checkpoint not found"
**Fix**: Kiểm tra path trong cell 3
```python
print(os.listdir("/kaggle/input/xvlm-checkpoints/"))  # Should show best.pth
```

### Lỗi: "data manifest not found"
**Fix**: Kiểm tra files đã upload đầy đủ
```python
print(os.listdir("/kaggle/input/train-30k-hard/"))  # Should show .jsonl files
```

### Lỗi: "CUDA out of memory"
**Fix**: Giảm batch size trong config
```yaml
train:
  batch_size: 12  # Từ 16 → 12
```

### Lỗi: "Kaggle timeout before finish"
**Fix**: Resume từ last checkpoint
```python
!python scripts/train.py \
    --config configs/stage1_30k_kaggle_t4_updated.yaml \
    --resume outputs/stage1_30k_t4/last.pth \
    --max-hours 11.5
```

### Training bị stuck/slow
**Nguyên nhân**: I/O bottleneck (đọc ảnh .webp chậm)
**Fix**: Pre-extract ảnh sang tmpfs (RAM disk)
```python
!mkdir -p /tmp/train_images
!cp -r /kaggle/input/train-30k-hard/train_webp/* /tmp/train_images/
# Update config: image_root: /tmp/train_images/
```

---

## 📈 Expected Results

### Baseline (best.pth warm-start)
- mAP: **80.0%**
- R@1: ~75%
- MRR: ~80%

### Target (STAGE 1 sau training)
- mAP: **82-84%** (+2-4% gain)
- R@1: ~77-79%
- MRR: ~82-84%

### Success Criteria
- ✅ mAP ≥ 82% → Thành công, tiếp tục STAGE 2
- ⚠️ mAP 80-82% → Tạm ổn, xem xét tune hyperparameters
- ❌ mAP < 80% → Safety net revert, cần debug

---

## 🔄 Next Steps Sau Training

### Nếu mAP improved (≥82%)
1. Download `stage1_best.pth`
2. Evaluate trên test set
3. Nếu tốt → Tiếp tục **STAGE 2** (unfreeze vision encoder)

### Nếu mAP không improve
1. Review training logs (`outputs/stage1_30k_t4/train.log`)
2. Visualize loss curves (cell 8 trong notebook)
3. Điều chỉnh hyperparameters:
   - Learning rate: `3e-5` → `1e-5` (conservative hơn)
   - Loss weights: Tăng `lambda_box`, `lambda_anomaly`
   - Augmentation: Giảm `sim2real_prob`

### Hard-Negative Mining (Optional - nâng cao)
Nếu muốn push lên 84-86% mAP:
```python
# Run ANCE mining sau epoch 1
!python scripts/mine_hard_negatives.py \
    --ckpt outputs/stage1_30k_t4/epoch_1.pth \
    --manifest data/train_30k_hard/train_30k_hard.jsonl \
    --output data/train_30k_hard/hard_negs_ance.jsonl

# Re-train với hard negatives
```

---

## 💡 Tips

1. **Always run sanity check first** (cell 5) - không waste GPU time nếu setup sai
2. **Monitor logs** - nếu loss không giảm sau 500 steps, dừng lại debug
3. **Save intermediate checkpoints** - mỗi epoch tự động save
4. **Don't close browser** - Kaggle có thể kill session nếu inactive lâu
5. **Use Kaggle's auto-save** - Enable "Auto-save and run all" nếu muốn chạy overnight

---

## 📞 Contact

Nếu gặp lỗi không giải quyết được:
1. Check logs: `outputs/stage1_30k_t4/train.log`
2. Copy full error message và hỏi lại tôi
3. Hoặc tạo GitHub Issue với minimal reproducible example

Good luck! 🚀
