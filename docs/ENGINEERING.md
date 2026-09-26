# SellerLens — chi tiết kỹ thuật

[← README](../README.vi.md)

| Chủ đề | Chi tiết |
|---|---|
| **Agent orchestration** | Máy trạng thái LangGraph: `classify → {bot_info \| off_topic \| data_summary \| planner → execute → synthesize}`. Streaming từng node qua SSE. |
| **Độ tin cậy LLM** | Chuỗi failover tới 9 mắt xích — 3 model Gemini × số key khai báo → 3 model OpenRouter. Quota free tier tính theo cặp (key, model), nên hết quota thì **đổi key trên cùng model trước**, hạ model sau — giữ model tốt nhất lâu nhất. Lỗi runtime (429 / 404 / timeout) tự chuyển model kế; cạn chuỗi mới xuống rule‑based. |
| **Safe tool‑calling** | Whitelist action & aggregation, plan validate với DataFrame, thực thi pandas tất định trong "sandbox" thao tác. |
| **Grounding** | Câu tổng hợp bị **từ chối** nếu chứa con số không khớp kết quả tính (`_numbers_grounded`) → tránh bịa số. |
| **Evaluation** | `tests/eval_100.py` — 100 câu / 6 dataset, ground truth tính bằng pandas. Mỗi chiều báo riêng, xem bảng [Đánh giá](#đánh-giá-đọc-số-thế-nào) bên dưới và [`docs/EVALUATION.md`](docs/EVALUATION.md). |
| **Observability** | `services/usage.py` đo token / chi phí / độ trễ từng lần gọi LLM và **quy về từng node** (`plan` chiếm 83% token, `synthesize` 17%), trả kèm trong response API. Eval báo cáo p95·p99 latency, `$/1.000 câu hỏi`, lỗi provider tách theo nguyên nhân. Chi phí chưa khai giá báo `n/a` thay vì `$0`. Không cần dịch vụ tracing ngoài. |
| **Routing eval** | `tests/test_routing.py` — đo bằng **macro-recall** chứ không phải accuracy, vì 85% câu là `data_query` nên accuracy trần che mất lỗi. Phát hiện `bot_info`/`off_topic` chỉ đúng 1/5; sửa bằng mẫu cấu trúc + tín hiệu dữ liệu phủ quyết → held-out macro-recall **33% → 100%**. |
| **Adversarial testing** | `tests/test_adversarial.py` — 18 đòn tấn công (SSRF / plan escape / prompt injection), chạy offline trong CI, chặn 18/18. Bộ này phát hiện và vá 4 lỗ hổng thật: SSRF ở import-from-URL, ReDoS ở filter `contains`, indirect prompt injection qua dữ liệu upload, `limit` không trần — chi tiết ở [`docs/EVALUATION.md`](docs/EVALUATION.md). |
| **Data hygiene** | `services/numeric_parse.py` đọc được cột tiền dạng chữ (`" $ (4,533.75) "` → −4533.75, định dạng châu Âu, `32.000.000 VNĐ`). Không có nó, `sum()` trên cột toàn `NaN` trả `0` và agent báo "Tổng lợi nhuận là 0". Bộ eval **cũng** dính lỗi này — ground truth cũ bỏ sót 63/700 dòng; nay cả hai dùng chung một bộ đọc số. |
| **Fuzzy column resolution** | LLM gọi sai tên cột (thiếu dấu, viết tắt) → resolver khớp mờ về tên thật trước khi validate. |
| **Multi‑sheet / multi‑file** | Tự phát hiện quan hệ giữa các sheet; planner sinh **cross‑sheet join**; cảnh báo fan‑out khi join 1‑nhiều làm phồng số dòng. |
| **Full‑stack** | FastAPI + React/Vite, dashboard tự sinh KPI theo domain, export CSV/Markdown, đóng gói Docker, health check. |
