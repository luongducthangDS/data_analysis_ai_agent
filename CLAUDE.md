## Ghi sai lầm & bài học (BẮT BUỘC)

- Sau mỗi lần sửa lỗi thật (bug, deploy hỏng, eval chấm sai, hiểu sai yêu cầu, hoặc chính Claude làm sai rồi phải sửa):
  thêm 1 mục vào đầu `docs/LESSONS.md` theo mẫu Sai gì / Gốc rễ / Sửa / Bài học, kèm hash commit.
- Ghi ngay lúc sửa, không để dồn cuối. `docs/LESSONS.md` và `docs/UX_FEEDBACK_*.md` là ghi chú nội bộ, đã gitignore — KHÔNG đưa lên GitHub.
  Docs được README link tới (EVALUATION, ENGINEERING, eval-baseline) thì vẫn commit bình thường.
- Bài học lặp lại hoặc dễ tái phạm → rút thêm thành 1 dòng quy tắc trong CLAUDE.md.

## Lessons — Deploy Errors (2026-05-26)

### Khi edit config deploy, phải trace flow thực thi end-to-end trước khi confirm xong
- Không chỉ check từng file riêng lẻ — phải đọc lại toàn bộ file đã edit và simulate
  "Railway/Docker sẽ thực thi cái gì, theo thứ tự nào"

### railway.toml: startCommand ghi đè Dockerfile CMD
- Nếu dùng `builder = "dockerfile"`, Railway vẫn chạy `startCommand` nếu có —
  `startCommand` không expand shell variable (`$PORT` → lỗi, phải dùng `sh -c`)
- Khi chuyển builder, phải kiểm tra xem `startCommand` có conflict không

### Git Bash đổi đối số `/path` thành đường dẫn Windows
- Gọi CLI (render, railway…) với đối số `/api/...` từ Git Bash → thêm `MSYS_NO_PATHCONV=1`, rồi đọc lại giá trị đã lưu

### Route mới: session qua `load_owned_session`/`get_session`, gọi LLM thì gắn `@limiter.limit(RATE_LIMIT)`
- Không gọi `session_store.get(id)` trần trong route (bỏ qua kiểm tra chủ). Code chặn (pandas, DB) → route `def`, không `async def`.
- File route có `@limiter.limit` không được dùng `from __future__ import annotations` (FastAPI không resolve được type body).

### Lưu trữ & config
- Lịch sử chat: `session_store.append_messages` (thêm dòng) + `recent_history` (đọc DB). Không ghi đè cả list — 2 worker sẽ đè nhau.
- Thêm cột vào model → `init_db` tự `ADD COLUMN` (chỉ cột nullable). Đổi tên/kiểu cột → cần Alembic.
- Config đọc qua `get_settings()`, không `os.getenv` rải rác; `.env` nạp ở MỘT chỗ (`core/config.py`).
- Lỗi ghi DB không được nuốt: log `exception` và báo cho client (upload → 503), trừ khi user đã có kết quả (chat).
- CI chạy `ruff check backend tests scripts`; dev cài `requirements-dev.txt`.

### FastAPI route ordering
- `/{full_path:path}` catch-all phải đăng ký CUỐI CÙNG, sau tất cả `/api/*` routes
- FastAPI match theo thứ tự đăng ký — đặt sai chỗ sẽ nuốt mất các route phía sau

## Nâng cấp TMĐT — shop Chị Lan (dữ liệu MÔ PHỎNG)

- Dữ liệu: `scripts/gen_shop_lan.py` → `data/samples/shop_lan/` (seed cố định, có `ground_truth.json` cho kịch bản S1–S4).
  Sửa generator thì chạy lại `python -m scripts.gen_shop_lan` và commit cả file sinh ra; test so file đã commit với generator.
  README/CV phải ghi rõ là mô phỏng, tham số lấy từ báo chí, không phải dữ liệu shop thật.
- Metric tiền (lãi, phí sàn) định nghĩa MỘT chỗ ở `backend/app/services/ecommerce_semantic.py`, tính sẵn lúc nạp file
  (`SessionStore._normalize_frame`). LLM chỉ sum các cột này, không tự ghép công thức. `loi_nhuan_truoc_qc` CHƯA trừ quảng cáo.
- File export sàn: thêm định dạng mới = thêm 1 dict vào `EXPORT_HEADERS`; giá vốn ghép từ bảng `sku, gia_von` qua `attach_cogs`.
- "Vì sao lãi đổi" = action `profit_bridge` (`ecommerce_semantic.profit_bridge`): 4 hạng mục cộng lại PHẢI bằng Δ lãi
  (giá vốn lấy phần dư). Gợi ý hành động sinh tất định ở `bridge_actions`, LLM không tự nghĩ; test bám đáp án S1/S4.
- Thêm action mới cho planner: sửa CẢ `ALLOWED_ACTIONS` (`services/planner/execute.py`) lẫn `_ALLOWED_ACTIONS` + `_ACTION_ALIASES` (`agents/nodes/plan.py`).
- So khớp từ khoá trên câu hỏi đã bỏ dấu phải khớp NGUYÊN TỪ (`\b`): chuỗi con "ai " từng khớp nhầm "lãi"/"loại"/"cái".
- Chạy test: `.venv/Scripts/python.exe -m pytest -q` (python hệ thống thiếu pytest-mock → 5 lỗi giả ở test_api).
- eval_100 cần server đang chạy: `tests/eval_100.py --base-url http://localhost:PORT --ids ...`; baseline ở `docs/eval-baseline/`.
- Lời hứa seller đo bằng `tests/eval_seller.py --base-url ...` (25 câu × 3 cách nạp A đủ / B không giá vốn / C thiếu 1/3);
  `wrong_looks_right` phải = 0 trước khi release. Phần tất định chạy offline ở `tests/test_seller_guard.py`.
- Thiếu đầu vào của metric → KHÔNG tạo cột đó (không fill 0, không để LLM thay cột khác); câu hỏi cần nó trả lời
  từ chối tất định (`missing_cogs_notice`). Cảnh báo dữ liệu nối vào câu trả lời ở `profit_notes`, không nhờ LLM nhớ.
- Kết quả tổng hợp 0 dòng phải là "không có dữ liệu" (bảng rỗng), không phải số 0. Cắt dữ liệu trước khi đưa LLM thì ghi rõ còn bao nhiêu dòng.
- Vá code có regex bằng script: ghi script ra file rồi chạy (heredoc từng biến `\b` thành backspace); sau đó `grep -P '\x08'`.
- Thêm field vào response API, giá trị `source` mới, hoặc câu trả lời nhắc tới một nút → sửa `frontend/src/main.tsx` trong cùng thay đổi (đã có lần backend bảo "bấm Tải mẫu giá vốn" mà UI không có nút).
