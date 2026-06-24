# ✅ CHECKLIST TRƯỚC KHI CHẠY KAGGLE

## 📦 Data Preparation

### Local Files (Cần có đầy đủ)
- [ ] `data/checkpoints/best.pth` - Checkpoint mAP 80% (~500MB)
- [ ] `data/train_30k_hard.jsonl` - Manifest 30K samples (~10MB)
- [ ] `data/train_30k_hard_vitpose.json` - Pose keypoints (~50MB)
- [ ] `data/boxes_30k.jsonl` - Bounding boxes (~5MB) ⚠️ **ĐANG CHẠY**
- [ ] `data/train_webp/` - Folder chứa 30K ảnh .webp (~15GB)

**Nếu `boxes_30k.jsonl` chưa có**: Có thể chạy training trước, sau đó re-run khi có file

### Verify Data Quality
```bash
# Count images
ls data/train_webp/*.webp | wc -l
# Should be ~30000

# Check manifest
wc -l data/train_30k_hard.jsonl
# Should be ~30000 lines

# Check vitpose
python -c "import json; print(len(json.load(open('data/train_30k_hard_vitpose.json'))))"
# Should be ~30000

# Check boxes (if available)
wc -l data/boxes_30k.jsonl
# Should be ~30000 lines
```

---

## 🧪 Local Testing (QUAN TRỌNG!)

### Bước 1: Install Dependencies
```bash
cd trainv4
pip install -r requirements.txt
pip install albumentations
pip install -e .
```

### Bước 2: Run Sanity Check
```bash
bash scripts/kaggle_quick_start.sh
```

**Expected output**:
```
✓ PyTorch 2.6.0+cu124
✓ Transformers installed
✓ Albumentations installed
✓ data/train_30k_hard.jsonl
✓ data/train_30k_hard_vitpose.json
✓ data/boxes_30k.jsonl
✓ data/checkpoints/best.pth
✓ Found 30000 .webp images
...
[Training for 200 steps]
...
✅ SUCCESS! Pipeline is working correctly.
```

**Nếu fail**: Fix lỗi local trước, đừng upload lên Kaggle!

---

## ☁️ Kaggle Upload

### Upload Checkpoint Dataset
- [ ] Created dataset "xvlm-checkpoints" on Kaggle
- [ ] Uploaded `best.pth` (verify ~500MB)
- [ ] Dataset is **Public** (để notebook access được)

### Upload Training Data Dataset
- [ ] Created dataset "train-30k-hard" on Kaggle
- [ ] Uploaded `train_30k_hard.jsonl`
- [ ] Uploaded `train_30k_hard_vitpose.json`
- [ ] Uploaded `boxes_30k.jsonl` (hoặc để empty nếu chưa có)
- [ ] Uploaded `train_webp.zip` (~15GB compressed)
- [ ] Dataset is **Public**

**Lưu ý**: Nén `train_webp/` thành `.zip` để upload nhanh hơn:
```bash
cd data
zip -r train_webp.zip train_webp/
# Upload train_webp.zip instead of raw folder
```

---

## 🎯 Kaggle Notebook Setup

### Notebook Configuration
- [ ] Created new notebook on Kaggle
- [ ] Selected **GPU T4 x2** (not P100 or other)
- [ ] Enabled **Internet** (để clone GitHub repo)
- [ ] Added dataset: `your-username/xvlm-checkpoints`
- [ ] Added dataset: `your-username/train-30k-hard`

### Notebook Content
Option A (Khuyến nghị):
- [ ] Imported `notebooks/kaggle_stage1_30k.ipynb` từ GitHub

Option B (Manual):
- [ ] Copy-paste từng cell từ notebook vào Kaggle

---

## 🔧 Config Verification

### Check Paths
Trong notebook, verify paths sau khi link data:
```python
# Cell để check
!ls -lh data/checkpoints/best.pth          # Should exist, ~500MB
!ls -lh data/train_30k_hard.jsonl          # Should exist, ~10MB
!ls -lh data/train_30k_hard_vitpose.json   # Should exist, ~50MB
!ls -lh data/boxes_30k.jsonl               # Should exist (hoặc empty)
!ls data/train_webp/ | head -10            # Should show .webp files
```

### Check Config Values
```python
import yaml
cfg = yaml.safe_load(open('configs/stage1_30k_kaggle_t4.yaml'))
print("Batch size:", cfg['train']['batch_size'])        # Should be 16
print("Epochs:", cfg['optim']['epochs'])                # Should be 8
print("XBM enabled:", cfg['loss']['xbm_enabled'])       # Should be True
print("Box head:", cfg['model']['bbox_enabled'])        # Should be True
print("Anomaly head:", cfg['model']['anomaly_enabled']) # Should be True
```

---

## 🚀 Training Execution

### Pre-flight Checks
- [ ] GPU is T4 (not CPU!)
- [ ] Internet enabled
- [ ] Both datasets loaded
- [ ] Config paths updated

### Execution Order
1. [ ] **Cell 1-2**: Setup (5 phút)
2. [ ] **Cell 3**: Sanity check (5-10 phút) ⚠️ **MUST PASS!**
3. [ ] **Cell 4**: Full training (2-3 giờ)
4. [ ] **Cell 5**: Evaluate results
5. [ ] **Cell 6**: Download checkpoint

### During Training
- [ ] Monitor logs: `!tail -f outputs/stage1_30k_t4/train.log`
- [ ] Check GPU usage: `!nvidia-smi` (should be ~90-100%)
- [ ] Loss should decrease: ~1.5 → ~0.5 over training

---

## 📊 Success Criteria

### Training Metrics (Expected)
- [ ] Loss giảm từ ~1.5 → ~0.5
- [ ] mAP ≥ 82% (baseline 80%)
- [ ] R@1 ≥ 77% (baseline ~75%)
- [ ] No NaN/Inf losses
- [ ] GPU utilization ~90%+

### Red Flags (Nếu thấy thì STOP!)
- ❌ Loss không giảm sau 500 steps
- ❌ Loss explode → NaN/Inf
- ❌ mAP tụt < 78%
- ❌ GPU util < 50% (I/O bottleneck)
- ❌ OOM errors

---

## 💾 Post-Training

### Checkpoint Verification
```python
# Check checkpoint size
!ls -lh outputs/stage1_30k_t4/best.pth
# Should be ~500MB (similar to input)

# Load and inspect
import torch
ckpt = torch.load("outputs/stage1_30k_t4/best.pth", map_location='cpu')
print("Epoch:", ckpt.get('epoch', 'N/A'))
print("mAP:", ckpt.get('report', {}).get('mAP', 'N/A'))
```

### Download Checklist
- [ ] Downloaded `stage1_best.pth` từ `/kaggle/working/`
- [ ] Downloaded training logs (optional)
- [ ] Downloaded loss curves (optional)
- [ ] Saved Kaggle notebook (click "Save Version")

---

## 🔄 Next Steps

### If mAP ≥ 82% (Success!)
- [ ] Evaluate on validation set
- [ ] Plan STAGE 2 (unfreeze vision encoder)
- [ ] Archive checkpoint safely

### If 80% ≤ mAP < 82% (Marginal)
- [ ] Review loss curves
- [ ] Tune hyperparameters
- [ ] Train longer (2-3 more epochs)

### If mAP < 80% (Failed)
- [ ] Review logs for anomalies
- [ ] Check data quality
- [ ] Reduce augmentation strength
- [ ] Lower learning rate
- [ ] Ask for help with full logs

---

## 🆘 Emergency Contacts

**Nếu stuck**:
1. Check logs: `outputs/stage1_30k_t4/train.log`
2. Copy full error message
3. Screenshot của cell output
4. Hỏi lại với context đầy đủ

**Common issues**: See `CHAY_KAGGLE_NHANH.md` → Troubleshooting section

---

## ✅ Final Check Before Running

```bash
# One-liner để check tất cả
python -c "
from pathlib import Path
checks = {
    'Checkpoint': Path('data/checkpoints/best.pth').exists(),
    'Manifest': Path('data/train_30k_hard.jsonl').exists(),
    'VitPose': Path('data/train_30k_hard_vitpose.json').exists(),
    'Boxes': Path('data/boxes_30k.jsonl').exists(),
    'Images': Path('data/train_webp').exists(),
}
for name, status in checks.items():
    print(f'{'✅' if status else '❌'} {name}')

if all(checks.values()):
    print('\n🚀 All checks passed! Ready to upload to Kaggle.')
else:
    print('\n⚠️  Some files missing. Fix before uploading.')
"
```

**Output should be all ✅**

---

## 🎯 Ready to Go?

Nếu tất cả checklist đều ✅:
1. Upload data lên Kaggle Dataset
2. Tạo notebook trên Kaggle
3. Run và monitor
4. Download checkpoint khi xong

**Estimated total time**: ~3-4 giờ (upload + training)

Good luck! 💪🚀
