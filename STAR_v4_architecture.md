# STAR‑v4 — KIẾN TRÚC CHỐT & KẾ HOẠCH TRAIN (1× A100 80GB)

> **Bài:** Text‑Based Person Anomaly Retrieval — dataset **PAB** ("Beyond Walking", ICCV 2025, arxiv 2411.17776).
> **Train** = ~**1.013.605 ảnh SYNTHETIC** (diffusion sinh từ caption OOPS! bằng Realistic Vision V4.0, re‑caption bằng Qwen2‑VL).
> **Test** = **1.978 ảnh REAL trích từ video OOPS!** + 34.795 distractor (tổng gallery 36.773).
> **Phần cứng:** train 1× **A100 80GB**; **inference Kaggle T4 free** (xem Phần 9).
> **Mục tiêu (leaderboard):** mAP **~93‑95%** (top‑3). Hiện tại: mAP 80% / R@10 94% trên distractor.

---

## 🚫 CẢNH BÁO TUÂN THỦ (đọc đầu tiên — quyết định cả thiết kế)

**TEST SET = FRAME THẬT TỪ OOPS!. TRAIN SYNTHETIC = SINH TỪ CAPTION OOPS!.** (Xác nhận từ paper PAB.)
→ **TUYỆT ĐỐI CẤM dùng OOPS! (và mọi dataset YouTube‑fail‑compilation có thể trùng OOPS!) để train.** Dùng = **train trên nguồn của test = leakage + vi phạm.**
→ Mọi ý tưởng "tiêm real‑data từ OOPS!/FailArmy", "fail‑moment head từ nhãn OOPS!", "video‑teacher trên OOPS!" — **ĐÃ LOẠI BỎ khỏi bản này.**
→ **Sim2real phải đóng KHÔNG bằng real‑data của test‑domain**, mà bằng: **backbone pretrain‑real mạnh + FULL synthetic + augmentation** (top teams 95% cũng làm vậy, vì họ cũng bị cấm OOPS!).

---

## 0. CHẨN ĐOÁN — 2 nút thắt + gốc chung

**Nút 1 — RECALL trên distractor (QUYẾT ĐỊNH ĐIỂM THI).** R@10=94% → mAP=80%. **~119 GT (6%) không lọt top‑10** giữa 34.795 nhiễu → mAP **bị trần bởi recall**. Top teams **R@10 ~99%** → giải được, ta thua ~5%. **Đòn chính = đẩy recall.**

**Nút 2 — TOP‑1/2 trên old‑test.** GT vào top‑5 (~99%) nhưng kẹt rank‑2 (same‑scene). 10 nhóm lỗi.

**GỐC CHUNG = SIM2REAL.** Train = synthetic (diffusion) ↔ Test = frame real video. Gap: motion‑blur thật, ánh sáng/nén video, độ phức tạp cảnh real mà diffusion không bắt hết. X‑VLM (pretrain 16M) **over‑adapt synthetic → mất generalize real** (đã thấy 0.69→0.19). → **Đóng sim2real = backbone pretrain‑real + full‑data + augment, KHÔNG over‑fit synthetic.**

---

## 1. QUYẾT ĐỊNH KIẾN TRÚC (CHỐT) — Dual‑model + Ensemble

| Vai trò | Model | Lý do |
|---|---|---|
| **RETRIEVAL (recall + sim2real)** | **SigLIP‑2‑L** (ViT‑L/16, ~400M) | Pretrain **hàng tỉ ảnh THẬT** → feature "nói tiếng real" sẵn → fine‑tune nhẹ trên synthetic vẫn generalize sang test real → **đóng sim2real + đẩy R@10**. Sigmoid‑loss hợp distractor. |
| **RERANK (precision)** | **X‑VLM cross‑encoder ITM** — giữ từ v3 | SigLIP không có cross‑encoder; X‑VLM ITM đẩy R@1 (best.pth 0.7953/0.8454), region‑aware → fine‑grained. |
| **ENSEMBLE** | SigLIP ⊕ X‑VLM (RRF/min‑max) | 2 họ khác → lỗi không tương quan → **+2‑4% mAP** (chiêu chuẩn top‑team). |

**Loại:** BLIP‑2 (Q‑Former bottleneck → mất chi tiết mịn) · InternVL/Qwen‑VL (generative, fail rerank) · EVA‑CLIP‑E 4.7B (không FT nổi 1 GPU).
**Vì sao đổi từ X‑VLM thuần:** X‑VLM‑16M là trần của T4 cũ; A100 mở khóa SigLIP‑L (pretrain‑real → vừa recall vừa sim2real). Giữ X‑VLM làm reranker + ensemble.

---

## 2. PHÂN TÍCH CHI TIẾT TỪNG CẢI TIẾN (đã bỏ mọi thứ dựa OOPS!)

### I1 — DÙNG FULL ~1.000.000 SYNTHETIC (đòn data lớn nhất, hợp lệ) — Nút1 + generalize
**Vấn đề:** Bạn mới train trên **10K/50K** — chỉ ~5% official train set. Ít data → generalize kém → recall trên distractor yếu.
**Kỹ thuật:** Tải **FULL PAB train (~1.013.605 cặp synthetic)** từ nguồn chính thức cuộc thi (hợp lệ — đây là train set được cấp). Train trên toàn bộ (hoặc ≥300‑500K subset cân bằng anomaly/normal).
**Vì sao giải đúng:** Nhiều data đa dạng → embedding bao phủ không gian rộng → GT nổi hơn giữa nhiễu (recall) + ít overfit. Đây là lever **rẻ, an toàn, lớn** mà v3 bỏ qua.
**Thông số:** full 1M (hoặc balanced 500K), batch 96, A100. ~1 epoch 1M ≈ 10K step.
**Khác v3:** 10K/50K → **full 1M**.
**Kỳ vọng:** recall tăng đáng kể; nền cho mọi cải tiến khác.

### I2 — Backbone SigLIP‑2‑L (đòn SIM2REAL + recall CHÍNH) — Nút1
**Vấn đề:** Swin‑B (X‑VLM, pretrain 16M) over‑adapt synthetic → sập trên real (0.69→0.19). Embedding không đủ tách 34K nhiễu.
**Kỹ thuật:** SigLIP‑2‑L ViT‑L/16 (pretrain WebLI **tỉ ảnh real**). Fine‑tune **NHẸ** (LoRA r16‑32, **lr thấp 5e‑5**, freeze nhiều) để **GIỮ real‑pretrain** (không xoá bằng synthetic). Sigmoid pairwise loss.
**Vì sao giải đúng:** sim2real = khoảng cách feature synthetic↔real. Backbone đã học real (tỉ ảnh) → fine‑tune nhẹ synthetic → **vẫn đọc được real test** → recall không sập. Đây là cách đóng sim2real **không cần real‑data của test‑domain** (tuân thủ).
**Thông số (A100‑80GB):** ViT‑L/16 @512 train, LoRA r32 qkv, **lr 5e‑5 cosine**, wd 0.05, batch 96, bf16, 6‑8 ep. **Cảnh báo:** lr cao = xoá real‑pretrain → sập sim2real → giữ lr thấp + EMA.
**Khác v3:** Swin‑B 16M‑pretrain → ViT‑L tỉ‑ảnh‑pretrain.
**Paper:** SigLIP (Zhai ICCV23); SigLIP‑2 (2025).

### I3 — Resolution 512→768 (FixRes) — Nút1, #4/#5
**Vấn đề:** 384 + downsample xoá vật nhỏ (#5), foreground bị nền át (#4).
**Kỹ thuật:** train 512, **FixRes fine‑tune+eval 768**.
**Vì sao:** res cao giữ chi tiết nhỏ + foreground → recall (phân biệt nhiễu) + #4/#5.
**Thông số:** train 512 (Stage A/B), 768 (Stage C lr 1e‑5).
**Paper:** FixRes (Touvron NeurIPS19).

### I4 — bf16 + batch 96 + Negative Queue 65K (CHỈ X‑VLM InfoNCE) — Nút1, #10
**Vấn đề:** batch 20 (T4) → ít negative → margin yếu → caption chung (#10) khớp loạn.
**Kỹ thuật:** bf16 + batch 96 + **queue MoCo 65K cho nhánh ITC InfoNCE của X‑VLM** (key‑encoder EMA m=0.995, feature detach+norm). **SigLIP KHÔNG dùng queue** (sigmoid không cần — xem Phần 8.1).
**Vì sao:** nhiều negative → embedding ép xa phân bố nhiễu → recall‑among‑distractors.
**Paper:** MoCo (He CVPR20); ALBEF (Li NeurIPS21).

### I5 — Augmentation đóng SIM2REAL (blur/JPEG/downscale/color/erase/text‑mask) — Nút1, #6/#8/#9
**Vấn đề:** synthetic sạch‑nét; test real có **motion‑blur, nén JPEG, độ phân giải video kém, occlusion, chữ trên áo**. Model chưa thấy mấy thứ này → feature lệch → recall thấp + #9/#8/#6.
**Kỹ thuật:** train‑time: **motion‑blur, JPEG‑compression, downscale‑upscale (mô phỏng frame video), color‑jitter, Gaussian noise** (đóng gap synthetic→real) + **Random‑Erasing** (#8) + **text‑region mask** (#6).
**Vì sao giải đúng:** ép feature **bền với degradations của ảnh real** → đóng phần lớn sim2real **mà không cần real‑data**. Đây là lever sim2real lớn thứ 2 sau backbone.
**Thông số:** blur p0.3, JPEG q30‑90 p0.4, downscale 0.5‑1.0 p0.3, color‑jitter 0.4, erase p0.25, text‑mask p0.2.
**Paper:** Random Erasing (Zhong AAAI20); RandAugment; AugMix.

### I6 — Box grounding (bật bbox_head — ĐANG BỎ) trên X‑VLM — #3/#7
**Vấn đề:** feature global mất quan hệ không gian ("collision"=bbox→0; "trên/cạnh ghế"=trục Y) → #3 (~76, lớn nhất old‑test), #7.
**Kỹ thuật:** bật `bbox_head` (đang trong unexpected_keys = không nạp). Box loss L1+GIoU, target = **primary_bbox trong vitpose.json**. **Bản đơn giản:** regress box người‑chính từ caption (synthetic thiếu annotation cụm‑từ↔vùng — xem Phần 8.3).
**Vì sao:** ép encoder học **vật ở ĐÂU** → phân biệt quan hệ không gian.
**Thông số:** w_box=0.5; clamp box hợp lệ.
**Paper:** X‑VLM (Zeng ICML22); GIoU (Rezatofighi CVPR19).

### I7 — FILIP token‑wise late‑interaction (X‑VLM) — #2/#4
**Vấn đề:** global feature → danh từ át động từ ("falling" trọng số nhỏ) → #2 (~47).
**Kỹ thuật:** sim token‑wise `(1/n)Σᵢ maxⱼ(v̂ᵢ·t̂ⱼ)`, **mask PAD**. Ép từng từ khớp token ảnh → động từ có "đường khớp riêng".
**Thông số:** w_filip=0.5.
**Paper:** FILIP (Yao ICLR22).

### I8 — Pose region‑fuse + Anomaly‑bucket aux head (CHỈ synthetic, KHÔNG OOPS!) — #1/#2
**Vấn đề:** single‑frame mù temporal; pose là proxy "đang ngã". v3 pose MLP nhỏ fuse global → "nhầm người".
**Kỹ thuật:** (a) pose fuse vào **region feature**; (b) head phụ **"is‑anomaly (bucket=wentwrong)?"** dùng **nhãn bucket SẴN CÓ trong synthetic train** (goal/wentwrong/full) — **hợp lệ, không dùng nhãn OOPS!**. Dạy model nhận trạng‑thái‑bất‑thường.
**Vì sao:** pose region‑fuse sửa "nhầm người"; anomaly‑head dạy nhận "đang ngã/bất thường" từ data hợp lệ.
**Thông số:** w_anom=0.3 (bucket‑label), pose fuse tại region layer.
**Lưu ý:** đây là bản **hợp lệ** thay cho "fail‑moment head" (đã bỏ vì dùng nhãn OOPS!). Yếu hơn nhưng tuân thủ.
**Paper:** ViTPose (Xu NeurIPS22).

### I9 — Hard‑neg mining động từ lỗi model trên TRAIN (synthetic) — #1/#2
**Kỹ thuật:** mỗi 2 ep chạy model trên **TRAIN synthetic**, cache top‑1‑sai/GT → ITM hard‑neg pool. Curriculum dễ→khó. **CẤM mining trên test/old‑test.**
**Paper:** ANCE (Xiong ICLR21).

### I10 — Ensemble SigLIP ⊕ X‑VLM — mAP chung
**Kỹ thuật:** RRF (rank‑based, bền nhất) **hoặc** min‑max/z‑score per‑query rồi cộng α0.6/β0.4. **KHÔNG L2‑norm** (Phần 8.2). 2 model khác họ → lỗi không tương quan.
**Kỳ vọng:** +2‑4% mAP, 0 train.

### I11 — k‑reciprocal + Query‑Expansion (cross‑modal) — Nút1 distractor
**Kỹ thuật:** **(1) QE/DBA trước** (mở rộng query‑embedding bằng top‑k ảnh gần → đưa query vào image‑space) → **(2) k‑reciprocal trên gallery‑gallery** (image‑image). Đẩy nhiễu xuống. Code đúng thứ tự (Phần 8.4).
**Thông số:** k1=20,k2=6,λ=0.3; QE top‑5.
**Paper:** k‑reciprocal (Zhong CVPR17).

### I12 — TTA multi‑scale + EMA — mAP chung
**Kỹ thuật:** encode 512+768 (+flip nếu caption không có "trái/phải"), trung bình; EMA decay 0.999.
**Kỳ vọng:** +1‑2% mAP.

**ĐÃ BỎ (vì dựa OOPS! = vi phạm):** ~~real‑data injection từ OOPS!~~ · ~~fail‑moment head từ nhãn OOPS!~~ · ~~video‑teacher distill trên OOPS!~~.
**Bất khả (trần cứng):** #1 frame real y hệt (cue=0, **và giờ KHÔNG có temporal data hợp lệ** → #1 chỉ vớt được bằng pose, yếu) + #10 caption rác.

---

## 3. KIẾN TRÚC THỰC THI (chi tiết)

### 3.1 Data pipeline (CHỈ synthetic hợp lệ — KHÔNG real ngoài) — BRIEF CHO TEAM DATA

> ⚠️ **CẤM tải/đụng OOPS! hay mọi video "fail compilation" YouTube** — test cuộc thi LÀ frame OOPS! → train trên đó = vi phạm + leakage. Train **chỉ** dùng ảnh synthetic BTC cấp.

**A. Nguồn train:** FULL PAB train (~**1.013.605 ảnh SYNTHETIC**, diffusion sinh từ caption — không phải ảnh thật). v3 mới dùng 10K/50K = ~5% → lấy **toàn bộ** (tối thiểu 300‑500K cân bằng).

**A. Manifest cần giao** (1 file parquet/jsonl, mỗi dòng 1 ảnh):

| Cột | Nội dung | Nguồn |
|---|---|---|
| `image_path` | đường dẫn ảnh | có sẵn |
| `caption` | mô tả text | có sẵn |
| `video_id` | id sự kiện/video gốc của caption | có sẵn (để split + pair‑batch) |
| `bucket` | nhãn trạng thái **normal vs anomaly** (goal/wentwrong) | từ loại caption (Cn=normal / Ca=anomaly) |
| `keypoints` | 17 điểm pose COCO (x,y,conf) | **TEAM trích** (ViTPose/YOLO‑pose) |
| `bbox` | hộp người‑chính, chuẩn hóa xyxy [0,1] | **TEAM trích** (detector/pose) |
| `image_id` + `pair_image_id` | cặp hard cùng `video_id` cho PairBatchSampler | **TEAM build** từ video_id |
| `is_real` | luôn = **False** | cố định |

**A. Việc team tự làm:** (1) chạy **ViTPose** lấy keypoints+bbox cho ~1M ảnh; (2) build `pair_image_id` (group theo `video_id`, mỗi anchor ghép 1 ảnh khó cùng video).
**Sampler:** balanced normal/anomaly + pair‑batch (giữ v3). **VAL‑B:** split‑by‑video **seed 42** (KHÔNG để cùng video_id ở cả train+val → tránh leakage nội bộ).

**B. AUGMENTATION** — chạy **ONLINE trong code train** (random mỗi epoch); **team KHÔNG pre‑generate ảnh**, chỉ giao **ảnh sạch + manifest**. Liệt kê để team hiểu kế hoạch + giao đúng metadata (bbox để crop an toàn).

*B1 — Đóng sim2real (QUAN TRỌNG NHẤT: synthetic sạch‑nét → test là frame video mờ/nén):*
| Augment | Tham số | Vì sao |
|---|---|---|
| Motion blur | kernel 3‑9, p=0.3 | frame "đang ngã/chạy" nhòe |
| JPEG compression | q 30‑90, p=0.4 | frame video bị nén |
| Downscale→Upscale | scale 0.5‑1.0, p=0.3 | độ phân giải video thấp |
| Gaussian noise | nhẹ, p=0.2 | nhiễu cảm biến |
| Color jitter | bright/contrast/sat 0.3, hue 0.05, p=0.5 | ánh sáng real đa dạng |

*B2 — Robustness (đánh nhóm lỗi):*
| Augment | Tham số | Vì sao |
|---|---|---|
| Random Erasing/Cutout | area 2‑20%, p=0.25 | bền occlusion (#8) |
| Text‑region mask | p=0.2 | giảm bám chữ trên áo (#6) |
| RandomResizedCrop | scale **0.8‑1.0** (dùng bbox giữ người trong khung) | đa dạng khung |
| LHP (v3) | giữ nguyên | đã có |

*B3 — ⛔ CẤM / CẨN THẬN (dễ hỏng nhãn):*
- **KHÔNG xoay mạnh:** "đang ngã" phụ thuộc hướng → xoay người đứng → trông như ngã → **hỏng nhãn**. Tối đa ±5° hoặc bỏ.
- **KHÔNG flip nếu caption có "trái/phải/bên"** → đảo hướng sai nghĩa. Chỉ flip khi caption không nhắc hướng.
- **Color jitter vừa phải** → quá tay "áo đỏ"→"áo cam" → hỏng caption thuộc tính.
- Augment **CHỈ áp TRAIN**, **KHÔNG** áp VAL/test.

**C. Distractor‑val (Phase 0, BẮT BUỘC — để ĐO recall chế độ nhiễu, KHÔNG train):**
- **Positive (~1978):** old‑test GT (test public năm cũ đã có đáp án) — **chỉ ĐO** (validation trên test public hợp lệ).
- **Nhiễu (~20K):** ảnh người real **KHÔNG phải OOPS!/fail‑video**, chứng minh tách rời: ✅ Market‑1501 / MSMT17 (re‑ID giám sát) / COCO‑person / CUHK‑PEDES; ❌ KHÔNG Kinetics‑fail / FailArmy / mọi "fail compilation".
- **Bắt buộc de‑dup perceptual‑hash** ảnh nhiễu vs gallery test → loại trùng/gần‑trùng.

**D. Checklist team:** ☐ tải FULL ~1M synthetic (không OOPS!) · ☐ ViTPose keypoints+bbox mọi ảnh · ☐ manifest 8 cột + pair_image_id · ☐ VAL‑B by‑video seed 42 · ☐ giao **ảnh sạch** (augment ở code train) · ☐ ~20K nhiễu disjoint‑OOPS! + de‑dup cho distractor‑val.

### 3.2 Model A — SigLIP‑2‑L retrieval (lo RECALL + sim2real)

**1. Model + load:** SigLIP‑2‑L = ViT‑L/16 + text tower, pretrain WebLI **tỉ ảnh THẬT**. Load HF `google/siglip2-large-patch16-512` (`Siglip2Model`) — *verify id trên HF trước khi code*. Embedding chung **d=1024**; ảnh @512 patch16 → 1024 patch token.

**2. Train gì / FREEZE gì — CỐT LÕI SIM2REAL ⭐:** mục đích dùng SigLIP = nó **đã biết ảnh thật** → phải **GIỮ real‑pretrain**, chỉ chỉnh nhẹ. Fine‑tune mạnh = ghi đè → sập sim2real (cú 0.69→0.19 của X‑VLM).

| Phần | Cách | Vì sao |
|---|---|---|
| Backbone ViT‑L + text tower | **LoRA r32** (q,k,v,proj), **freeze weight gốc** | chỉnh nhẹ, không xoá real‑pretrain |
| Projection heads (img/txt→1024) | **FULL‑FT** (nhỏ) | nơi căn chỉnh task, rẻ |
| logit scale `t` + bias `b` | **train** | temperature/bias của sigmoid |
| Phần còn lại | **đóng băng** | giữ kiến thức real |

→ **lr THẤP 5e‑5 + EMA 0.999** là để bảo vệ real‑pretrain. **Quyết định quan trọng nhất của Model A** (lr cao = hỏng sim2real).

**3. Resolution — CHỐT: 512 TRƯỚC, 768 sau.** Phase 1 chạy **512 thuần** (ăn chắc, đo sạch hiệu quả backbone). FixRes 768 = nâng cấp về sau (cần nội suy bicubic positional‑embedding vì ViT‑L cố định res; chỉ +~1‑2% chi tiết nhỏ).

**4. Loss — CHỐT: Phase 1 chạy sigmoid THUẦN.**
- Lõi = **sigmoid pairwise (native SigLIP):** batch N cặp → ma trận N×N, chéo=positive(z+1), ngoài chéo=negative(z−1); `L=−Σ log σ(z·(t·s+b))`. Sigmoid **không cần batch khổng lồ/queue** → **KHÔNG dùng queue** cho SigLIP.
- **(Tùy chọn, Phase 3, KHÔNG lúc đầu):** FILIP token‑wise (động từ "falling"); pose‑fuse keypoint ViTPose (proxy off‑balance). Để sau cho gọn.

**5. Hyperparams (A100‑80GB):** res 512 · bf16 + grad‑checkpoint · batch 96 (dư VRAM, có thể 128) · LoRA r32 α64 (q,k,v,proj) · AdamW **lr 5e‑5** cosine, warmup 1 ep, wd 0.05, clip 1.0 · **6‑8 epoch** · EMA 0.999 (dùng EMA‑weights để eval/infer).

**6. Output & dùng:** encode gallery→`img_feat` (1024‑d, **L2‑norm**), query→`txt_feat` (L2‑norm). **`sim_S[q,g]=cosine(txt_feat,img_feat)`** → top‑K → **ensemble RRF với X‑VLM** → rerank (3.5). SigLIP lo **recall**, X‑VLM lo **precision**.

### 3.3 Model B — X‑VLM‑16M v4 (lo PRECISION/rerank)

> **Lưu ý kiến trúc:** cross‑encoder X‑VLM **ăn image‑feature của chính Swin‑B** → KHÔNG tách "cross‑only" được (không nhét được feature SigLIP vào). X‑VLM là **khối khép kín**: giữ cả encoder+cross. Vai trò: **rerank top‑K** (chính) + **ensemble sim_X** (tùy — quyết bằng distractor‑val: giữ nếu recall tăng, bỏ nếu kéo xuống).

**1. Warm‑start `--init-from best.pth(0.8454)`:** nạp weight, optimizer/step **mới** (flag có sẵn [train.py:48‑52](train2/scripts/train.py), `strict=False`). Đổi recipe → moment Adam cũ thành rác → reset.
**⚠️ Gotcha:** head mới (anomaly/FILIP) init mới; **bbox_head nạp từ `xvlm_16m_base.th`** (có sẵn grounding pretrain trong `unexpected_keys`) — KHÔNG random. Tức load 2 nguồn: weight chung từ best.pth + bbox_head từ xvlm base.

**2. Train/freeze (duyệt `named_parameters`, set theo pattern tên):**
| Phần | Cách | Vì sao |
|---|---|---|
| Swin q,k,v | **LoRA r32 α64** (v3 r16 → nâng) | chỉnh nhẹ |
| Cross 6‑11 (cross‑attn) | **FULL‑FT** (v3 chỉ LoRA do T4) | "não rerank" mạnh hơn |
| Text tower 0‑5 | **freeze** | giữ hiểu ngôn ngữ ổn định |

**3. Heads — build + nhãn + dùng lúc nào:**
| Head | Build | Nhãn | Inference? |
|---|---|---|---|
| **ITC** | proj img[CLS]+txt[CLS]→256‑d L2‑norm | cặp đúng | ✅ ra **sim_X** |
| **ITM** | cross‑attn fuse→[CLS]→MLP 2‑class | positive+1 hard‑neg | ✅ **rerank top‑K** |
| **bbox_head** | MLP(h→4)+sigmoid (xyxy norm) | `primary_bbox` | ❌ aux train (ép học định vị → region feature tốt → vớt #3/#7; KHÔNG dùng box test) |
| **pose region‑fuse** | MLP nhúng 17×3 kpt → fuse vào **region feature** | ngầm | ✅ **cần ViTPose cho 36K gallery lúc infer** (+~0.5h/lần) → off‑balance |
| **anomaly‑bucket** | MLP img[CLS]→3‑class | `bucket` (synthetic) | ❌ aux (dạy "trạng thái bất thường") |
| **FILIP** | proj patch‑token+text‑token → `(1/n)Σᵢ maxⱼ(v̂ᵢ·t̂ⱼ)`, **mask PAD** | contrastive | (tùy) động từ #2/#4 |
> Aux head (bbox/anomaly) = tín hiệu **phụ lúc train**, KHÔNG xuất điểm retrieval. ITC+ITM+pose mới dùng khi infer.

**4. Loss:** `1·ITC + 1.5·ITM(hard‑neg) + 0.5·FILIP + 0.5·(L1+GIoU) + 0.3·CE_anom + 0.2·SmoothAP`
- 1·ITC=contrastive(recall) · **1.5·ITM**=rerank (insight fine‑tune ăn +1.7%) · 0.5·FILIP=căn từ↔patch · 0.5·(L1+GIoU)=box (rows có bbox) · 0.3·CE_anom=bucket · 0.2·SmoothAP=tối ưu AP.
- **Cân bằng:** log grad‑norm; head phụ nuốt >40% grad → hạ w. **Luật vàng: R@5/R@10 VAL‑B KHÔNG tụt.**

**5. Negatives — 4 nguồn:**
| Nguồn | Cách | Tính chất |
|---|---|---|
| in‑batch 64 | 63 text còn lại trong batch | free |
| **queue 65K** | buffer FIFO[65536,d] + momentum key‑enc (EMA 0.995); dequeue 64 cũ/enqueue 64 mới; **detach+L2‑norm, không backprop** | nhiều neg cho ITC → recall |
| mined hard‑neg | mỗi 2 ep chạy model trên **TRAIN**, cache top‑1‑sai/query → pool ITM | động, đúng lỗi model |
| pair‑batch (v3) | PairBatchSampler ghép anchor+ảnh khó cùng `video_id` | hard‑neg same‑scene, **tĩnh** |
- Curriculum: pair‑batch (ep1‑3) → thêm mined (ep4+). **⚠️ CẤM mining trên test/old‑test.**

**6. Staging (gọn trước, head sau — 512 trước như SigLIP):**
- **Stage A (Phase 2):** warm‑start, 512, chỉ ITM(1.5)+ITC+cross‑full‑FT+hard‑neg → reranker mạnh (đủ cho ensemble).
- **Stage B (Phase 3):** bật bbox/FILIP/anomaly/pose, 512.
- **Stage C:** FixRes 768 (sau cùng).

### 3.4 Loss — công thức + kiểm soát
- **SigLIP sigmoid:** `−Σ_{ij} log σ(z_{ij}(t·s_{ij}+b))`, z=±1, t,b học.
- **ITC InfoNCE (X‑VLM):** `−½[i2t+t2i]`, neg = in‑batch ∪ queue, τ học.
- **ITM:** hard‑neg sample ∝ ITC‑sim → cross‑encoder BCE.
- **FILIP:** max‑sim token (mask PAD).
- **Box:** L1+GIoU, rows có bbox.
- **Anomaly‑bucket:** CE (goal/wentwrong/full) trên **synthetic** (có nhãn bucket).
- **Cân bằng:** bật grad‑norm log; **luật vàng: R@5/R@10 VAL‑B KHÔNG tụt** (tụt = ITC/sim2real bị bóp → hạ w phụ HOẶC hạ lr SigLIP).

### 3.5 Inference pipeline
```
1. SigLIP encode gallery+query, TTA{512,768} → sim_S [Q,G]
2. X‑VLM encode (pose‑ON) → sim_X [Q,G]
3. sim = RRF(sim_S, sim_X)  hoặc  0.6·minmax_row(sim_S)+0.4·minmax_row(sim_X)   # KHÔNG L2‑norm
4. DBSN/Sinkhorn → top‑K=200
5. X‑VLM cross‑encoder ITM rerank top‑K
6. QE/DBA (cross‑modal) → k‑reciprocal trên image‑space (k1=20,k2=6,λ=0.3)
7. no‑distractor → Gale‑Shapley ; distractor → bỏ GS
```

### 3.6 Bảng siêu tham số (A100‑80GB, khác v3 in đậm)
| Param | v3 | **v4** |
|---|---|---|
| retrieval backbone | X‑VLM Swin‑B | **SigLIP‑2‑L ViT‑L (pretrain‑real)** |
| rerank | X‑VLM ITM | X‑VLM ITM (giữ) |
| ensemble | — | **RRF / min‑max (0.6/0.4)** |
| **train data** | **10K/50K** | **FULL ~1M synthetic** |
| image_res | 384 | **512→768** |
| precision | fp16 | **bf16** |
| batch | 20 | **96** |
| queue (X‑VLM) | — | **65.536** |
| SigLIP queue | — | **KHÔNG (sigmoid)** |
| LoRA r | 16 | **32** |
| SigLIP lr | — | **5e‑5 (thấp, giữ real‑pretrain)** |
| X‑VLM lr | 2e‑4 | 2e‑4(A,B)/1e‑5(C) |
| losses | ITC+ITM+SmAP | **+FILIP+BOX+ANOM(bucket)** |
| bbox_head | OFF | **ON** |
| augment | LHP | **+blur+JPEG+downscale+color+erase+textmask (sim2real)** |
| inference | ITM+GS | **+RRF ensemble+QE+k‑reciprocal+TTA** |
| EMA | — | **0.999** |

---

## 4. SCALE — 1×A100 80GB (no DDP)
- KHÔNG DDP. Scale: grad‑checkpoint + batch 96 + bf16. Train 2 model **tuần tự**.
- 80GB dư → batch SigLIP 96‑128 @512. EMA + ckpt‑averaging (3 ckpt cuối).
- DataLoader num_workers≥12, ảnh trên local SSD (1M ảnh I/O nặng).

---

## 5. KIỂM TOÁN (math, VRAM, thời gian)

### 5.1 Math
- SigLIP sigmoid ✔ · InfoNCE+queue dims `[B,B+65K]`, detach+norm ✔ · FILIP mask‑PAD ✔ · GIoU clamp tránh NaN ✔ · anomaly‑bucket CE trên synthetic (có nhãn) ✔ · ensemble RRF/min‑max per‑query (KHÔNG L2) ✔ · k‑reciprocal sau QE (cross‑modal) ✔.

### 5.2 VRAM (A100‑80GB)
- SigLIP‑L 400M @512 batch96 grad‑ckpt ≈ **~45‑55GB** → vừa, dư. @768 batch48.
- X‑VLM @512 batch64 ≈ ~30GB.
- Queue 65K×1024×2B ≈ 0.13GB.
- **KHÔNG OOM.** (Số ước lượng — **profile 1 step thật**, OOM thì batch 96→64→48.)

### 5.3 Thời gian (1×A100‑80GB)
- **Tải full 1M synthetic + build manifest:** ~2‑4h (1 lần, ~? GB tùy nguồn).
- **SigLIP FT** (6 ep × 1M, batch96 ≈ 10.4K step/ep × ~0.8s) ≈ **~14h** *(1M lớn — nếu cần nhanh, train 500K balanced ≈ ~7h)*.
- **X‑VLM v4** (full/subset, 12 ep): ~5‑10h tùy data.
- **Inference** (T4 free, xem Phần 9): ~2‑3h/lần encode, tune rerank free.
- **TỔNG train ~1.5‑2 ngày A100** (full 1M); subset 500K ≈ ~1 ngày.

### 5.4 Papers
SigLIP (Zhai ICCV23) · SigLIP‑2 (2025) · EVA‑CLIP (Sun23) · X‑VLM (Zeng ICML22) · ALBEF (Li NeurIPS21) · MoCo (He CVPR20) · FILIP (Yao ICLR22) · FixRes (Touvron NeurIPS19) · ANCE (Xiong ICLR21) · Random‑Erasing (Zhong AAAI20) · GIoU (Rezatofighi CVPR19) · Smooth‑AP (Brown ECCV20) · k‑reciprocal (Zhong CVPR17) · ViTPose (Xu NeurIPS22) · **PAB/Beyond‑Walking (Yang ICCV25, arxiv 2411.17776)**.

---

## 6. LỘ TRÌNH CÀI ĐẶT (theo phase, đo mỗi bước)

**PHASE 0 — Distractor‑val (TRƯỚC TIÊN).**
Gallery = old‑test GT(1978) + ~20K nhiễu person **disjoint OOPS!** (Market‑1501/COCO‑person) + perceptual‑hash de‑dup chống trùng test. `eval_distractor.py` đo R@1/5/10+mAP. **Cổng:** baseline answer.txt v3 → số gốc.

**PHASE 1 — Full‑data + SigLIP (đòn recall+sim2real lớn nhất).**
(1a) Tải **full 1M PAB synthetic** → manifest. (1b) `siglip_retrieval.py` (HF Siglip2 + peft LoRA lr5e‑5 + sigmoid + augment sim2real). (1c) train 6‑8 ep. (1d) đo distractor‑val.
**Cổng:** R@10 phải **vượt rõ 94%**. **Kỳ vọng mAP ~88‑91%.**

**PHASE 2 — Ensemble + re‑ranking (rẻ, +mAP, ít train).**
RRF(SigLIP, X‑VLM‑v3) + QE + k‑reciprocal + TTA. **Kỳ vọng ~90‑93%.**

**PHASE 3 — X‑VLM v4 heads (fine‑grained, đánh 10 lỗi).**
Code vào `star`: bbox_head (I6) + FILIP (I7) + pose region‑fuse + anomaly‑bucket head (I8) + augment + hard‑neg mining. Train Stage A/B/C. Ensemble lại. **Kỳ vọng ~92‑95%.**

**Thứ tự code:** `eval_distractor.py` → `prep_full_synthetic.py` → `siglip_retrieval.py` → `rerank_ensemble_kr.py` → heads trong `star` + notebook X‑VLM‑v4.

---

## 7. RỦI RO & GIỚI HẠN THẬT
- **#1 temporal:** không còn data temporal hợp lệ (OOPS! cấm) → chỉ vớt bằng pose (off‑balance) → **yếu hơn nhiều so với bản OOPS! tôi vẽ sai trước**. Frame real y hệt = bất khả.
- **#10 caption rác:** bất khả.
- **Sim2real:** đóng bằng backbone‑real + augment + full‑data → **không chắc đủ mạnh**; cú 0.69→0.19 có thể vẫn còn một phần → **distractor‑val là bắt buộc** để biết thật/ảo.
- **Công sức code = nhiều ngày/tuần** (SigLIP trainer, 4 head, ensemble cross‑modal, full‑1M pipeline) → **bắt buộc phase + đo**, đừng code hết trước.
- **Kỳ vọng calibrated:** Phase1 ~88‑91 → Phase2 ~90‑93 → Phase3 ~92‑95. **95% là mép trên**, cần backbone+full‑data+ensemble+rerank đều ăn. Leaderboard chứng minh khả thi **không cần OOPS!**.

---

## 8. ĐÍNH CHÍNH SAU KIỂM TOÁN (giữ từ bản trước)
- **8.1** Queue 65K **chỉ X‑VLM InfoNCE**, SigLIP sigmoid không cần queue.
- **8.2** Ensemble **RRF / min‑max per‑query**, KHÔNG L2‑norm.
- **8.3** bbox bật lại = **regress primary‑box đơn giản** (synthetic thiếu phrase‑region), vớt #3 một phần.
- **8.4** k‑reciprocal **cross‑modal**: QE trước → đưa query vào image‑space → rồi k‑reciprocal gallery‑gallery.
- **8.5** ~~Auto‑caption OOPS!~~ → **BỎ (OOPS! cấm)**. Không tiêm caption real.
- **8.6** VRAM/time **ước lượng** → profile thật; verify SigLIP‑2 checkpoint id trên HF.
- **8.7 (MỚI, quan trọng nhất)** **OOPS! = nguồn test → CẤM.** Mọi cải tiến dựa real‑data/temporal/fail‑moment từ OOPS! đã loại. Sim2real đóng bằng backbone‑real + full‑data + augment.

---

## 9. INFERENCE TRÊN KAGGLE T4 FREE + CACHING (chống tốn tiền)
- **Bộ nhớ:** T4 16GB đủ (inference nhẹ, load model tuần tự). SigLIP‑L + X‑VLM forward đều vừa.
- **Tách encode (đắt, 1 lần) ↔ rerank/fusion (rẻ, lặp free):**
  - Commit 1: encode SigLIP 36K gallery+query → lưu `emb_siglip.pt` (transformers mới).
  - Commit 2: encode X‑VLM + region‑embeds → lưu `emb_xvlm.pt` (transformers cũ, kaggle_setup). *(2 env khác → tách commit.)*
  - Commit 3 (lặp nhiều, **free**): load emb → RRF ensemble + DBSN + QE + k‑reciprocal + ITM rerank → tune tham số thả ga.
- Embedding cache ~300MB. k‑reciprocal 36K² ≈ 2.7GB fp16 (chunk/faiss).
- **→ Train A100 (trả tiền, 1‑2 lần); inference + tune Kaggle T4 FREE; thử‑nhiều‑lần ~0 đồng.**

---

## TÓM TẮT 1 DÒNG
**SigLIP‑2‑L (pretrain‑real → recall+sim2real) ⊕ X‑VLM cross‑encoder (rerank) + FULL‑1M synthetic + augmentation‑sim2real + RRF‑ensemble + QE/k‑reciprocal/TTA.** **CẤM OOPS! (= nguồn test).** Train tuần tự ~1‑2 ngày 1×A100‑80GB; **infer Kaggle‑T4 free + caching**. Làm theo 4 phase, đo distractor‑val mỗi bước. Trần thực ~mAP 92‑95% (không có phép màu real‑data vì OOPS! cấm).
