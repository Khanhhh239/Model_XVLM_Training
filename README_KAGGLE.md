# 📚 TÀI LIỆU CHẠY KAGGLE - ĐỌC FILE NÀO?

## 🎯 Bạn Muốn Gì?

### "Tôi muốn chạy NGAY trên Kaggle!" → **CHAY_KAGGLE_NHANH.md**
- ✅ Hướng dẫn 3 bước ngắn gọn
- ✅ Copy-paste commands
- ✅ Troubleshooting nhanh
- ⏱️ **5 phút đọc**

### "Tôi muốn hiểu CHI TIẾT từng bước" → **KAGGLE_SETUP_GUIDE.md**
- ✅ Giải thích đầy đủ
- ✅ Expected output từng cell
- ✅ Troubleshooting chi tiết
- ✅ Tips & tricks
- ⏱️ **15 phút đọc**

### "Tôi muốn CHECK xem thiếu gì không" → **CHECKLIST_TRUOC_KHI_CHAY.md**
- ✅ Checklist từng mục
- ✅ Verify commands
- ✅ Success criteria
- ✅ Red flags
- ⏱️ **10 phút làm**

### "Tôi muốn TEST LOCAL trước" → **scripts/kaggle_quick_start.sh**
- ✅ Automated sanity check
- ✅ Verify data integrity
- ✅ Test training pipeline
- ⏱️ **10 phút chạy**

---

## 📁 Cấu Trúc Files

```
trainv4/
├── README_KAGGLE.md                    ← BẠN ĐANG ĐỌC FILE NÀY
├── CHAY_KAGGLE_NHANH.md               ← Quick start (Tiếng Việt)
├── KAGGLE_SETUP_GUIDE.md              ← Chi tiết đầy đủ (English)
├── CHECKLIST_TRUOC_KHI_CHAY.md        ← Checklist verify
│
├── notebooks/
│   └── kaggle_stage1_30k.ipynb        ← Notebook chạy trên Kaggle
│
├── configs/
│   └── stage1_30k_kaggle_t4.yaml      ← Config file cho training
│
├── scripts/
│   ├── kaggle_quick_start.sh          ← Test local script
│   ├── kaggle_upload_dataset.py       ← Upload data via CLI
│   └── train.py                       ← Main training script
│
└── data/                               ← Data folder (cần chuẩn bị)
    ├── checkpoints/best.pth
    ├── train_30k_hard.jsonl
    ├── train_30k_hard_vitpose.json
    ├── boxes_30k.jsonl
    └── train_webp/
```

---

## 🚀 Workflow Khuyến Nghị

### Lần Đầu Chạy (First-time setup)

1. **Đọc** → `CHAY_KAGGLE_NHANH.md` (5 phút)
2. **Check** → `CHECKLIST_TRUOC_KHI_CHAY.md` (10 phút)
3. **Test** → `bash scripts/kaggle_quick_start.sh` (10 phút)
4. **Upload** → Data lên Kaggle Dataset (30-60 phút)
5. **Run** → Notebook trên Kaggle (2-3 giờ)

**Total**: ~4 giờ

### Lần Sau (Đã quen)

1. **Upload** → Data mới lên Kaggle
2. **Run** → Notebook (2-3 giờ)

**Total**: ~3 giờ

---

## 📖 Chi Tiết Từng File

### 1. CHAY_KAGGLE_NHANH.md
**Mục đích**: Quick start guide tiếng Việt
**Nội dung**:
- 3 bước chính: Chuẩn bị → Upload → Chạy
- Commands copy-paste
- Troubleshooting table
- Expected results

**Đọc khi**: Bạn muốn chạy nhanh, không cần hiểu sâu

---

### 2. KAGGLE_SETUP_GUIDE.md
**Mục đích**: Chi tiết đầy đủ từng bước
**Nội dung**:
- Setup môi trường
- Giải thích từng cell notebook
- Expected output chi tiết
- Troubleshooting với solutions
- Next steps sau training

**Đọc khi**: Bạn muốn hiểu rõ pipeline, gặp lỗi phức tạp

---

### 3. CHECKLIST_TRUOC_KHI_CHAY.md
**Mục đích**: Verify setup trước khi chạy
**Nội dung**:
- Checklist data files
- Checklist config
- Checklist Kaggle setup
- Success criteria
- Red flags (warning signs)

**Dùng khi**: Trước khi upload lên Kaggle (tránh waste time)

---

### 4. notebooks/kaggle_stage1_30k.ipynb
**Mục đích**: Notebook để chạy trên Kaggle
**Nội dung**:
- 9 cells cover full pipeline
- Setup → Data → Training → Evaluation → Download
- Markdown explanations
- Inline comments

**Dùng khi**: Upload lên Kaggle và run

---

### 5. configs/stage1_30k_kaggle_t4.yaml
**Mục đích**: Config file cho training
**Nội dung**:
- Model architecture settings
- Training hyperparameters
- Loss weights
- Data paths

**Chỉnh sửa khi**: Tune hyperparameters, change paths

---

### 6. scripts/kaggle_quick_start.sh
**Mục đích**: Test pipeline local trước khi upload
**Nội dung**:
- Check dependencies
- Verify data files
- Run sanity check (overfit 1 batch)
- Report success/failure

**Chạy khi**: Trước khi upload lên Kaggle

---

### 7. scripts/kaggle_upload_dataset.py
**Mục đích**: Upload data lên Kaggle via CLI
**Nội dung**:
- Automated upload script
- Create dataset metadata
- Zip images for faster upload

**Dùng khi**: Bạn quen CLI, muốn automate upload

---

### 8. scripts/train.py
**Mục đích**: Main training script
**Nội dung**:
- Load config
- Setup model, optimizer, data
- Training loop
- Evaluation
- Checkpoint saving

**Không cần đọc**: Script tự động, notebook gọi nó

---

## 🎓 Learning Path

### Beginner (Lần đầu làm ML/Kaggle)
1. Đọc `CHAY_KAGGLE_NHANH.md` → Hiểu overview
2. Đọc `KAGGLE_SETUP_GUIDE.md` → Hiểu chi tiết
3. Làm `CHECKLIST_TRUOC_KHI_CHAY.md` → Verify setup
4. Follow notebook từng cell

**Time**: ~30 phút đọc + 4 giờ làm

### Intermediate (Đã quen Kaggle)
1. Skim `CHAY_KAGGLE_NHANH.md` → Nhớ lại flow
2. Check `CHECKLIST_TRUOC_KHI_CHAY.md` → Quick verify
3. Upload và run

**Time**: ~10 phút đọc + 3 giờ làm

### Advanced (Pro)
1. Chỉnh `configs/stage1_30k_kaggle_t4.yaml` theo ý
2. Run `scripts/kaggle_quick_start.sh` để test
3. Upload bằng `scripts/kaggle_upload_dataset.py`
4. Run notebook

**Time**: ~5 phút setup + 3 giờ training

---

## ❓ FAQ

### Q: Tôi nên bắt đầu từ file nào?
**A**: `CHAY_KAGGLE_NHANH.md` → Nhanh nhất

### Q: Tôi gặp lỗi, đọc file nào?
**A**: `KAGGLE_SETUP_GUIDE.md` → Section Troubleshooting

### Q: Làm sao biết data đã đủ chưa?
**A**: Chạy `scripts/kaggle_quick_start.sh` hoặc check `CHECKLIST_TRUOC_KHI_CHAY.md`

### Q: Notebook có sẵn rồi à?
**A**: Có! `notebooks/kaggle_stage1_30k.ipynb` → Import vào Kaggle

### Q: Config phải sửa gì không?
**A**: Thường không cần, notebook tự động update paths

### Q: Upload data mất bao lâu?
**A**: ~30-60 phút (tùy internet), nén .zip để nhanh hơn

### Q: Training mất bao lâu?
**A**: ~2-3 giờ trên T4 (30K dataset, 3 epochs)

### Q: Kết quả như nào là tốt?
**A**: mAP ≥ 82% (baseline 80%) → Success!

---

## 🆘 Cần Giúp?

### Nếu gặp lỗi:
1. Check **Troubleshooting** section trong `KAGGLE_SETUP_GUIDE.md`
2. Check **CHECKLIST** xem thiếu gì
3. Copy **full error message** + **logs** và hỏi lại

### Nếu stuck:
1. Đọc lại section tương ứng trong docs
2. Review notebook cell output
3. Check training logs: `outputs/stage1_30k_t4/train.log`

### Nếu kết quả không tốt:
1. Review **Expected Results** section
2. Check **Red Flags** section
3. Tune hyperparameters theo gợi ý

---

## 🎯 TL;DR - Đọc File Nào?

| Bạn Muốn Gì? | Đọc File Này | Thời Gian |
|--------------|--------------|-----------|
| Quick start | `CHAY_KAGGLE_NHANH.md` | 5 phút |
| Chi tiết đầy đủ | `KAGGLE_SETUP_GUIDE.md` | 15 phút |
| Verify setup | `CHECKLIST_TRUOC_KHI_CHAY.md` | 10 phút |
| Test local | Run `scripts/kaggle_quick_start.sh` | 10 phút |
| Upload CLI | Use `scripts/kaggle_upload_dataset.py` | N/A |
| Run Kaggle | Import `notebooks/kaggle_stage1_30k.ipynb` | 2-3 giờ |
| Tune params | Edit `configs/stage1_30k_kaggle_t4.yaml` | N/A |

---

**Bắt đầu từ**: `CHAY_KAGGLE_NHANH.md` 🚀

Good luck!
