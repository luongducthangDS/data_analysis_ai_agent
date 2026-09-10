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

- **HARD**: câu có ground-truth đơn trị (53/100). **SOFT**: câu mở / off_topic / bot_info (47/100) — `correctness` không áp dụng.
- `correctness` là chiều **cứng** (đối chiếu số bằng pandas). 3 chiều `insight` / `no_meta` / `concise` là **heuristic regex** → đã calibrate, xem bên dưới.
- `overall ≥ 0.70` = pass.

## Kết quả baseline

Run 2026-09-10, Gemini `flash-lite → flash` (failover chain, free tier). Chi tiết:
[`eval-baseline/summary.json`](eval-baseline/summary.json) · [`eval-baseline/results.csv`](eval-baseline/results.csv).

| | |
|---|---|
| Pass | **81 / 100** |
| Avg overall | **0.826** |
| Avg correctness | **0.747** (53 câu kiểm cứng) |
| Avg insight / no_meta / concise / vn_natural | 0.72 / 0.96 / 0.88 / 0.92 |
| Fallback rule-based | **0 / 100** |
| HTTP 5xx | 0 |
| Latency median | **~4.7 s** (mean 75 s bị kéo lệch bởi retry storm 429 — xem ghi chú) |

> Phiên bản scoring đầu tiên cho 97/100 vì `factual_ok` chỉ ±30% và weight 0.10.
> Siết `correctness` xuống ±10% + weight 0.40 → 81/100. Con số thấp hơn nhưng phản ánh đúng
> chất lượng (xem "Bug agent" bên dưới).

### Theo dataset — lộ rõ điểm yếu

| dataset | câu | pass | avg overall |
|---|---|---|---|
| sales_data | 27 | 24/27 | 0.89 |
| expense_claims | 11 | 11/11 | 0.90 |
| sales_sample | 24 | 21/24 | 0.87 |
| viet_relational | 6 | 4/6 | 0.79 |
| general_ledger | 16 | 12/16 | 0.74 |
| **financial_sample** | 16 | **9/16** | **0.70** |

### Theo nhóm câu hỏi

| nhóm | câu | pass | avg |
|---|---|---|---|
| comparison | 4 | 4/4 | 1.00 |
| bot_info | 5 | 4/5 | 0.93 |
| edge | 12 | 12/12 | 0.90 |
| ranking | 26 | 21/26 | 0.86 |
| trend | 4 | 4/4 | 0.81 |
| aggregation | 39 | 31/39 | 0.80 |
| off_topic | 5 | 4/5 | 0.71 |
| **language** (câu hỏi kiểu chat) | 5 | **1/5** | **0.53** |

## Calibration — heuristic có đáng tin không?

`tests/eval_calibration.py`: 30 câu trả lời thật, gán nhãn tay 3 chiều mềm, so với heuristic.
Kết quả ([`eval-baseline/calibration.txt`](eval-baseline/calibration.txt)):

| chiều | MAE | agreement (Δ ≤ 0.25) | Pearson r | kết luận |
|---|---|---|---|---|
| vn_natural | 0.03 | 100% | **+0.93** | tin được |
| concise | 0.12 | 83% | **+0.81** | tin được (sau khi thêm bắt template dump) |
| **insight** | 0.28 | 60% | **+0.14** | **không tin được** — regex đếm từ khoá không bám sát chất lượng thật |

→ `insight` cần **LLM-as-judge**. Đã implement (`--judge`), nhưng full run 100 câu đang bị
free-tier quota chặn (22/30 call lỗi 429). Framework sẵn sàng, chạy lại khi có quota:

```bash
python tests/eval_100.py --rescore eval-baseline/results.csv --judge   # chấm lại + judge, không cần server
```

⚠️ 30 nhãn calibration do 1 người review — đủ để sanity-check heuristic, **không phải** human-eval quy mô lớn.

## Bug agent do eval phát hiện (chưa sửa — ngoài scope harness)

Cụm ~14 câu fail tập trung ở 3 dataset, đều là **lỗi agent thật**, không phải lỗi chấm điểm:

| triệu chứng | dataset | ví dụ |
|---|---|---|
| Cột có dấu cách thừa (`" Profit "`, `"  Sales "`) → planner không resolve, trả 0 / "không tìm thấy" | financial_sample | #31 (GT 17.6M, trả 0), #33, #34, #36, #38 |
| Sổ cái đa tiền tệ → cộng gộp AUD+CAD+EUR+GBP+USD, số vô nghĩa | general_ledger | #46, #47, #48, #60 |
| Multi-sheet → đếm nhầm bảng (đếm orders thay vì customers) | viet_relational | #71 (GT 12, trả 32) |
| Câu hỏi kiểu chat ("ai bán giỏi nhất?") → rơi về template `# Executive Data Brief` thay vì trả lời tự nhiên | mọi dataset | #97, #98, #99, #100 |

## Ghi chú độ tin cậy

- Run thực hiện với key Gemini free tier gần cạn quota (528× HTTP 429). **Latency mean vô nghĩa**;
  dùng median (~4.7 s). 2 câu (#37, #46) client read-timeout → điểm 0, không phải lỗi logic.
- Câu trả lời trong `results.csv` bị cắt ở 300 ký tự (run cũ; cap hiện tại 2000) → `concise`
  cho câu dài hơi lạc quan. Chạy lại sẽ chính xác hơn.
- Pass-rate và các chiều **ổn định** qua nhiều lần rescore.

## Chạy lại

```bash
# eval đầy đủ (cần server + GEMINI_API_KEY)
uvicorn backend.app.main:app --port 8000 &
python tests/eval_100.py --base-url http://localhost:8000 --delay 3 \
    --baseline docs/eval-baseline/summary.json      # in bảng regression vs baseline

# chấm lại run cũ bằng scoring hiện tại — KHÔNG cần server, KHÔNG tốn quota
python tests/eval_100.py --rescore docs/eval-baseline/results.csv

# calibrate heuristic vs nhãn tay
python tests/eval_calibration.py
```
