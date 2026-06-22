# Chạy inference AI City 2026 Track 4 trên Kaggle (TRAINING-FREE)

Mục tiêu: dùng **CMP (open SOTA, đã train sẵn)** + các kỹ thuật rerank **training-free**, chạy trên
gallery **36.773 (có 34.795 distractor)**, đo + submit để biết **điểm tốt nhất KHI CHƯA train**.

## 1. Dataset cần tạo trên Kaggle (3 cái)
| Dataset (slug gợi ý) | Nội dung | Lấy từ đâu |
|---|---|---|
| `aicity-official-test` | `name-masked_test-set/` (gallery/ + query_text.json + query_index.txt) **và** `old_test-set/` (test/ + attr.json — để đo mAP local) | Bạn **đã có** ở máy: `Downloads/old_test-set/...` → nén & upload |
| `cmp-models` | `cmp.pth` (925MB, đã tải) + thư mục `bert-base-uncased/` | cmp.pth ở `Downloads/cmp_ckpt/`; bert: `huggingface.co/bert-base-uncased` (tải `config.json, vocab.txt, tokenizer*, pytorch_model.bin`) |
| `qwen2vl` *(tùy chọn, stage-2)* | Qwen2-VL-7B-Instruct | HF `Qwen/Qwen2-VL-7B-Instruct` (nặng ~16GB) |

> **Tải bert-base-uncased** (nhỏ): vào HF, tải cả thư mục, đưa vào `cmp-models/bert-base-uncased/`.
> cmp.pth đã chứa toàn bộ weight model → **KHÔNG cần** tải swin/X-VLM-init cho inference.

## 2. Bật GPU + Internet
- Notebook Settings → **Accelerator = GPU T4** (hoặc T4 ×2) · **Internet = ON** (để `git clone` CMP + startv4).
- Nếu không bật được Internet: upload repo **CMP** và **Model_XVLM_Training** (chứa `train3/startv4`) thành dataset, rồi sửa `sys.path`.

## 3. Sửa CELL 0 — thêm startv4 helpers (cho k-reciprocal + metrics)
Sau dòng clone CMP, thêm:
```python
subprocess.run(["git","clone","--depth","1","https://github.com/Khanhhh239/Model_XVLM_Training", f"{WORK}/MXT"], check=False)
sys.path.insert(0, f"{WORK}/MXT/train3")   # -> import startv4.eval.rerank / startv4.eval.metrics
```
Và chỉnh `TEST_DIR / CMP_DIR / VAL_DIR` cho khớp slug dataset bạn đặt.

## 4. Thứ tự chạy cell
1. **CELL 0–1**: setup + build CMP (in "CMP loaded ... 254M"). Nếu lỗi build → báo tôi log.
2. **CELL 2–3**: load data + **encode gallery (resume) + query**. Lần đầu ~15–40 phút; lưu `cache/gal_*.pt`.
   - Hết 9h/phiên? Chỉ cần **chạy lại CELL 0–3** ở phiên mới → tự **skip chunk đã xong** (resume).
3. **CELL 5**: bỏ comment `ABLATION = run_ablation(VAL_DIR, n_distract=5000)` → in bảng mAP từng kỹ thuật +
   **cờ KEEP/DROP** (cái nào kéo điểm xuống → DROP). Đây là cơ chế chống "trộn nhiều bị phá".
4. **CELL 6**: chọn cấu hình KEEP từ ablation → `SCORE = build_final_score(...)` → `write_submission(SCORE)`
   → `/kaggle/working/submission.txt` → nộp lên trang challenge để lấy mAP thật.

## 5. Resume & thời gian (T4)
- Nặng = encode 36.773 gallery (~15–40′) + ITM rerank top-128 × 1.978 (~20–40′). Tổng ~1h → 1 phiên đủ; có resume nếu đứt.
- `cache/` ở `/kaggle/working` → **Save Version** để giữ cache qua phiên (hoặc commit notebook).

## 6. Cơ chế chống phá hoại (đáp ứng yêu cầu)
`run_ablation` đo mAP trên **labeled val có distractor** (old-test GT + 5.000 nhiễu từ gallery thật) cho:
ITC → +dual_softmax → +k_reciprocal → +ITM. In `dmAP` mỗi cái + **KEEP/DROP**. Chỉ giữ kỹ thuật +điểm.

## 7. Stage-2 (sau khi core ra số) — bật để cố lên thêm
- **Pose-on**: CMP train pose-on; cần render pose-map cho 36.773 gallery đúng format CMP (xem `models/pose.py`).
  v1 để **pose-off** (chạy được, có số); pose-on là nâng cấp, verify format trước.
- **AnomalyLMM (Qwen2-VL) rerank**: cloze rerank top-K. **Nặng (vài giờ T4) + lời thường ~1%** → bật CUỐI,
  **gate bằng ablation**, giữ chỉ khi +mAP. (Đúng bài học Qwen +1.27% trước đây.)

## 8. Điểm dễ phải fix ở lần chạy đầu (workflow code→check→fix)
- `Search(config)` build: nếu đòi file init → cmp.pth đã đủ weight, đảm bảo `load_pretrained=True` + bert path đúng.
- `.half()` + `get_cross_embeds`: nếu lỗi dtype → bỏ `.half()`, chạy fp32 (chậm hơn, chắc hơn).
- `query_text.json` keys: hàm `load_masked_set` đã dò `caption/text/query_text`; nếu khác → sửa hàm `cap`.
- **Format submit**: hiện ghi mỗi dòng = 10 tên ảnh cách nhau bởi space, đúng thứ tự `query_index.txt`.
  **Kiểm lại spec submit của challenge** (có thể cần `query_id: name1,...` hay CSV) trước khi nộp.

## 9. Honest
- Số CMP công bố (mAP 91.66) là trên **1.978**; trên **36.773** chưa ai biết → **CELL 6 + submit cho con số thật**.
- Đây là **trần training-free**: CMP + rerank free. Muốn hơn nữa → train (CMP-UIT / SSDC) — việc khác.
