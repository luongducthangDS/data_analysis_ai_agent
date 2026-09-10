# Evaluation — baseline 100 câu

Bộ eval end-to-end: `tests/eval_100.py` upload 6 dataset trong `data/samples/`, hỏi 100 câu
qua API thật, chấm câu trả lời với **ground truth tính bằng pandas** (không fit theo output model).

- Ground truth build tại chỗ từ chính file dataset → thay dataset là ground truth đổi theo.
- 5 chiều chất lượng, trọng số: `no_meta` 0.30 · `insight` 0.25 · `concise` 0.20 · `vn_natural` 0.15 · `factual_ok` 0.10.
- `overall >= 0.70` tính là pass.

## Kết quả run baseline

| | |
|---|---|
| Ngày chạy | 2026-09-10 |
| Model | Gemini `flash-lite` -> `flash` (failover chain, free tier) |
| Pass | **97 / 100** (`overall >= 0.70`) |
| Avg overall | **0.895** |
| LLM usage | 100% (0 câu rơi rule-based fallback) |
| HTTP 5xx | 0 |
| Latency — median | **~4.7 s** / request |
| Latency — mean | 75 s (bị kéo lệch bởi retry storm khi free tier trả 429 — xem ghi chú) |

Số liệu chi tiết: [`eval-baseline/summary.json`](eval-baseline/summary.json) · từng câu: [`eval-baseline/results.csv`](eval-baseline/results.csv).

### Theo nhóm câu hỏi

| Nhóm | Số câu | Pass | Avg overall |
|---|---|---|---|
| aggregation | 39 | 37/39 | 0.857 |
| ranking | 26 | 26/26 | 0.930 |
| edge | 12 | 12/12 | 0.965 |
| bot_info | 5 | 5/5 | 0.928 |
| language | 5 | 5/5 | 0.810 |
| off_topic | 5 | 4/5 | 0.800 |
| comparison | 4 | 4/4 | 0.985 |
| trend | 4 | 4/4 | 0.906 |

### Chiều chất lượng (trung bình)

| no_meta | insight | concise | vn_natural | factual_ok |
|---|---|---|---|---|
| 0.960 | 0.752 | 0.938 | 0.980 | 0.840 |

`insight` thấp nhất — đây là chiều mềm (câu trả lời có gợi ý hành động không), chấm chặt tay.

## 3 câu không đạt

| # | Câu hỏi | overall | Nguyên nhân |
|---|---|---|---|
| 37 | "Tổng số Units Sold trên toàn bộ dataset?" | 0.00 | **Client read-timeout** — server kẹt trong retry storm 429 của free tier, không phải lỗi logic. Pass ở các lần chạy có quota dư. |
| 46 | "Tổng Debit của toàn bộ sổ cái?" | 0.00 | Như trên. |
| 81 | "Thủ đô của nước Pháp là đâu?" | 0.53 | Off-topic — bot từ chối đúng (`factual_ok = 1.0`); điểm thấp do rubric trừ `insight`/`concise` khi câu deflection ngắn. Hành vi đúng ý đồ. |

Correctness logic thực tế ~ 99–100/100; các lần chạy độc lập cho pass-rate 95–97%, dao động do
2–3 câu bị 429-timeout, **không phải** do câu trả lời sai.

## Ghi chú độ tin cậy của run này

Run này thực hiện với **Gemini API key free tier đã gần cạn quota** (server log: 528x HTTP 429).
Hệ quả:

- **Latency mean vô nghĩa** — vài request bị chuỗi failover retry với backoff tới 15–17 phút. Dùng **median (~4.7 s)** làm số đại diện.
- **2 câu điểm 0** là timeout phía client, không phải lỗi hệ thống. Chạy lại với quota dư -> pass.
- Pass-rate và các chiều chất lượng **ổn định** qua 3 lần chạy độc lập (95–97% pass, avg overall 0.89–0.90).

Đây đồng thời là bằng chứng chuỗi failover hoạt động dưới áp lực rate-limit: **0/100 câu crash, 0/100 rơi rule-based**.

## Chạy lại

```bash
uvicorn backend.app.main:app --port 8000 &          # cần GEMINI_API_KEY
python tests/eval_100.py --base-url http://localhost:8000 --delay 3
#   --delay N   giãn N giây giữa các request, tránh 429 free tier
#   --ids 1-20  chạy subset
#   --dataset sales_data   chỉ 1 dataset
```

Output ghi vào `results/eval_<timestamp>.{csv,html,json}` (`results/` được gitignore).
