# Evaluation

Harness end-to-end: `tests/eval_100.py` upload 6 dataset trong `data/samples/`, hỏi 100 câu
qua API thật, chấm câu trả lời với **ground truth tính bằng pandas tại chỗ** (không fit theo output model).

## Scoring

5 chiều. `overall` = tổng có trọng số; 2 bộ trọng số tuỳ câu có kiểm cứng được hay không:

| chiều | ý nghĩa | trọng số HARD | trọng số SOFT |
|---|---|---|---|
| **correctness** | số khớp GT: lệch ≤2%→1.0 · ≤10%→0.8 · ≤30%→0.3 · còn lại 0.0. keyword khớp đủ→1.0 | **0.40** | — (n/a) |
| insight | có "so what" / khuyến nghị, không chỉ đọc số | 0.22 | 0.42 |
| no_meta | không leak chain-of-thought / meta-commentary | 0.16 | 0.28 |
| vn_natural | tiếng Việt tự nhiên, không dump JSON/markdown/bảng thô | 0.12 | 0.17 |
| concise | 2–5 câu, không cụt, không lan man | 0.10 | 0.13 |

- **HARD**: câu có ground-truth đơn trị (54/100). **SOFT**: câu mở / off_topic / bot_info (46/100) — `correctness` không áp dụng.
- `correctness` là chiều **cứng**, đối chiếu số bằng pandas — bộ số parser xử lý cả định dạng
  Việt ("8.950", "0,212") lẫn viết tắt "28,63 triệu" / "1,5 tỷ".
- 3 chiều mềm (`insight`/`no_meta`/`concise`/`vn_natural`) là **heuristic regex** → đã calibrate bằng 2 cách độc lập, xem bên dưới.
- `overall ≥ 0.70` = pass.

## Kết quả baseline (run 2026-09-12, quota sạch)

Gemini `flash-lite → flash`, không bị rate-limit storm (khác run 2026-09-10 trước đó — xem lịch sử bên dưới).
Chi tiết: [`eval-baseline/summary.json`](eval-baseline/summary.json) · [`eval-baseline/results.csv`](eval-baseline/results.csv).

| | |
|---|---|
| Pass | **84 / 100** |
| Avg overall | **0.870** |
| Avg correctness | **0.770** (54 câu kiểm cứng) |
| Avg insight / no_meta / concise / vn_natural | 0.82 / 0.96 / 0.91 / 0.97 |
| Fallback rule-based | **0 / 100** |
| HTTP 5xx | 0 |
| Latency | median **4.8 s** · mean **6.1 s** (mean ≈ median → không còn nhiễu 429 như run trước) |

### Theo dataset

| dataset | câu | pass | avg overall |
|---|---|---|---|
| sales_data | 27 | 25/27 | 0.93 |
| expense_claims | 11 | 11/11 | 0.91 |
| viet_relational | 6 | 5/6 | 0.88 |
| sales_sample | 24 | 21/24 | 0.88 |
| general_ledger | 16 | 12/16 | 0.82 |
| **financial_sample** | 16 | **9/16** | **0.75** |

### Theo nhóm câu hỏi

| nhóm | câu | pass | avg |
|---|---|---|---|
| bot_info | 5 | 5/5 | 0.94 |
| trend | 4 | 4/4 | 0.91 |
| comparison | 4 | 3/4 | 0.89 |
| ranking | 26 | 22/26 | 0.87 |
| aggregation | 39 | 31/39 | 0.86 |
| edge | 12 | 11/12 | 0.90 |
| language (câu hỏi kiểu chat) | 5 | 4/5 | 0.89 |
| off_topic | 5 | 4/5 | 0.71 |

## Calibration — heuristic có đáng tin không?

Hai nguồn độc lập, cùng kết luận: **`insight` là chiều yếu nhất**, `concise`/`vn_natural` tin được.

**1. Nhãn tay** (`tests/eval_calibration.py`, 30 câu trả lời thật đóng băng ở
[`eval-baseline/calibration-run-20260910.csv`](eval-baseline/calibration-run-20260910.csv), 1 người review):

| chiều | MAE | agreement (Δ≤0.25) | Pearson r |
|---|---|---|---|
| vn_natural | 0.03 | 100% | **+0.93** |
| concise | 0.12 | 83% | **+0.81** |
| **insight** | 0.28 | 60% | **+0.14** |

**2. LLM-as-judge trên full 100 câu** (`--judge`, baseline hiện tại, dùng chính failover chain của backend):

| chiều | n | MAE | agreement (Δ≤0.25) | Pearson r |
|---|---|---|---|---|
| concise | 87 | 0.079 | 87% | **+0.88** |
| vn_natural | 87 | 0.056 | 92% | **+0.73** |
| **insight** | 89 | 0.173 | 58% | **+0.33** |

(11–13 câu mỗi lần bị 429/parse-fail — free tier, không phải bug judge.)

→ Cả nhãn tay (n=30, r=0.14) lẫn LLM-judge (n=89, r=0.33) đều xác nhận: **quy tắc regex đếm
từ khoá cho `insight` không bám sát chất lượng thật**. `concise`/`vn_natural` thì tin được sau
khi vá lỗi truncation + template-dump (commit trước). Kết luận thực dụng: dùng `--judge` cho
`insight` khi cần con số đáng tin, giữ heuristic cho 3 chiều còn lại (rẻ, đủ tốt).

⚠️ Bộ nhãn tay 30 câu do 1 người review — đủ sanity-check, không phải human-eval quy mô lớn.

## Bug agent do eval phát hiện (chưa sửa — ngoài scope harness)

Ổn định qua 2 lần chạy độc lập (2026-09-10 và 2026-09-12) — không phải nhiễu ngẫu nhiên:

| triệu chứng | dataset | ví dụ |
|---|---|---|
| Cột có dấu cách thừa (`" Profit "`, `"  Sales "`) → planner không resolve, trả 0 / "không tìm thấy" | financial_sample | #31 (GT 17.6M, trả 0), #33, #34, #36, #38, #43, #44 |
| Sổ cái đa tiền tệ → cộng gộp AUD+CAD+EUR+GBP+USD, số vô nghĩa | general_ledger | #46, #47, #48, #60 |
| Multi-sheet → đếm nhầm bảng | viet_relational | #71 (GT 12 khách hàng) |
| Sai logic nhóm (region/segment) dù không liên quan cột lỗi | sales_sample | #9 (model trả "North", GT thật là "East") |

## Ghi chú độ tin cậy

- Run 2026-09-12 quota sạch — mean latency ≈ median, không còn retry-storm 429 như run
  2026-09-10 trước đó (khi đó mean bị kéo lên 75s do vài request retry 15–17 phút).
- 1 câu (#88) client read-timeout đơn lẻ (60s) — hạ tầng, không phải lỗi logic.
- **Biến động giữa 2 lần chạy cùng bộ câu hỏi**: category `language` đổi từ 1/5 (run trước) lên
  4/5 (run này) — do `temperature=0.3` ở bước synthesize. Baseline 1 lần chạy không đại diện
  tuyệt đối; số ổn định qua nhiều lần chạy là cụm bug ở financial_sample/general_ledger.

## Chạy lại

```bash
# eval đầy đủ (cần server + GEMINI_API_KEY)
uvicorn backend.app.main:app --port 8000 &
python tests/eval_100.py --base-url http://localhost:8000 --delay 2 --judge \
    --baseline docs/eval-baseline/summary.json      # kèm bảng regression + judge agreement

# chấm lại run cũ bằng scoring hiện tại — KHÔNG cần server, KHÔNG tốn quota
python tests/eval_100.py --rescore docs/eval-baseline/results.csv

# calibration nhãn tay (trên bộ câu trả lời đóng băng 2026-09-10)
python tests/eval_calibration.py
```
