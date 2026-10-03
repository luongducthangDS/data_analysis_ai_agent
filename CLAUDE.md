## Ghi sai lầm & bài học (BẮT BUỘC)

- Sau mỗi lần sửa lỗi thật (bug, deploy hỏng, eval chấm sai, hiểu sai yêu cầu, hoặc chính Claude làm sai rồi phải sửa):
  thêm 1 mục vào đầu `docs/LESSONS.md` theo mẫu Sai gì / Gốc rễ / Sửa / Bài học, kèm hash commit.
- Ghi ngay lúc sửa, không để dồn cuối. `docs/LESSONS.md`, `docs/WORKLOG.md` và `docs/UX_FEEDBACK_*.md` là ghi chú nội bộ, đã gitignore — KHÔNG đưa lên GitHub.
  Docs được README link tới (EVALUATION, ENGINEERING, eval-baseline) thì vẫn commit bình thường.
- Bài học lặp lại hoặc dễ tái phạm → rút thêm thành 1 dòng quy tắc trong CLAUDE.md.
- Cuối mỗi ngày làm việc: thêm 1 mục vào đầu `docs/WORKLOG.md` (đã làm gì kèm commit, sự cố, tồn đọng,
  trạng thái cuối ngày).

## Nhiều session Claude cùng repo

- Trước mọi lệnh đổi trạng thái git (switch/checkout/reset/stash): `git status -sb` + `ListAgents`. Có session khác đang
  busy trong cùng thư mục → làm trong `git worktree add` riêng, không đổi nhánh của thư mục chung.
- Dừng server/tiến trình mình bật: chỉ kill process đang listen đúng port của mình. KHÔNG `taskkill /IM python.exe`
  (hay lọc theo tên) — giết luôn tiến trình của session khác.

## Lessons — Deploy Errors (2026-05-26)

### Khi edit config deploy, phải trace flow thực thi end-to-end trước khi confirm xong
- Không chỉ check từng file riêng lẻ — phải đọc lại toàn bộ file đã edit và simulate
  "Render/Docker sẽ thực thi cái gì, theo thứ tự nào"

### Git Bash đổi đối số `/path` thành đường dẫn Windows
- Gọi CLI (render…) với đối số `/api/...` từ Git Bash → thêm `MSYS_NO_PATHCONV=1`, rồi đọc lại giá trị đã lưu

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
- Giá trị LLM lọc/nhóm được phải đồng nhất từ lúc nạp (trạng thái đơn: `EXPORT_STATUS`), không chỉ trong công thức metric;
  số backend đưa vào prompt (ghi chú dữ liệu) phải nằm trong tập số được phép của `_numbers_grounded(notes=...)`.
- "Vì sao lãi đổi" = action `profit_bridge` (`ecommerce_semantic.profit_bridge`): 4 hạng mục cộng lại PHẢI bằng Δ lãi
  (giá vốn lấy phần dư). Gợi ý hành động sinh tất định ở `bridge_actions`, LLM không tự nghĩ; test bám đáp án S1/S4.
- Câu gợi ý (`seller_questions`) phải đúng cả khi KHÔNG có LLM: sửa/thêm câu → cập nhật `test_suggested_questions_are_right_without_llm`.
  Quy tắc metric mới viết vào prompt LLM thì thêm luôn vào `seller_plan` (`planner/seller_fallback.py`), không thì đường Luật làm sai.
  Đường Luật chỉ trả lời khi hiểu HẾT câu (metric, mọi giá trị lọc, chiều chia); còn lại từ chối (`_refuse`), không đoán.
- Thêm action mới cho planner: sửa CẢ `ALLOWED_ACTIONS` (`services/planner/execute.py`) lẫn `_ALLOWED_ACTIONS` + `_ACTION_ALIASES` (`agents/nodes/plan.py`).
- Mọi merge trên dữ liệu người dùng phải gọi `check_join_size` (`services/storage.py`) trước; lỗi guard (`JoinTooLarge`) không được bị `except Exception` nuốt.
- So khớp từ khoá trên câu hỏi đã bỏ dấu phải khớp NGUYÊN TỪ (`\b`): chuỗi con "ai " từng khớp nhầm "lãi"/"loại"/"cái".
- Chạy test: `.venv/Scripts/python.exe -m pytest -q` (python hệ thống thiếu pytest-mock → 5 lỗi giả ở test_api).
- eval_100 cần server đang chạy: `tests/eval_100.py --base-url http://localhost:PORT --ids ...`; baseline ở `docs/eval-baseline/`.
- Lời hứa seller đo bằng `tests/eval_seller.py --base-url ...` (đường Luật: `--offline`, không cần server; 167 câu, 3 cách nạp A đủ / B không giá vốn / C thiếu 1/3);
  thêm câu = thêm vào CUỐI (giữ id), câu "X nào lãi nhất" phải nhận cả người đứng đầu theo lãi trước lẫn sau QC,
  và `test_eval_seller_cases_pass_with_their_own_answer` phải qua (bắt số rơi vào vùng năm 1900–2100, NaN);
  Gemini free tier: chạy với `--delay 8`, câu `source=fallback` do quota thì chạy lại rồi `--merge` (merge chấm lại từ câu đã lưu);
  `--delay` chỉ chống quota THEO PHÚT: 1 lượt dev + held-out ≈ 400 lượt gọi, 3 lượt/ngày là cạn quota NGÀY (cả production
  nếu chung key — đã xảy ra 01/10) → eval dùng key ở PROJECT Google Cloud riêng (quota tính theo project, key mới
  cùng project vẫn chung quota), `.env` local không chứa key production; bộ held-out (`--heldout`) không được dùng để sửa app;
  `wrong_looks_right` phải = 0 trước khi release. Phần tất định chạy offline ở `tests/test_seller_guard.py`.
- Thiếu đầu vào của metric → KHÔNG tạo cột đó (không fill 0, không để LLM thay cột khác); câu hỏi cần nó trả lời
  từ chối tất định (`missing_cogs_notice`). Cảnh báo dữ liệu nối vào câu trả lời ở `profit_notes`, không nhờ LLM nhớ.
- Kết quả tổng hợp 0 dòng phải là "không có dữ liệu" (bảng rỗng), không phải số 0. Cắt dữ liệu trước khi đưa LLM thì ghi rõ còn bao nhiêu dòng.
- Vá code có regex bằng script: ghi script ra file rồi chạy (heredoc từng biến `\b` thành backspace); sau đó `grep -P '\x08'`.
- Thêm field vào response API, giá trị `source` mới, hoặc câu trả lời nhắc tới một nút → sửa `frontend/src/main.tsx` trong cùng thay đổi (đã có lần backend bảo "bấm Tải mẫu giá vốn" mà UI không có nút).
- Hai đường dựng cùng một trạng thái (create/restore) phải đi chung một hàm (`_build_sheets`), kèm test `restore(create(x)) == create(x)` với tên file trùng/khác dấu cách.
- Grounding/validator: sửa từ chối nhầm bằng cách so khớp chính xác hơn, KHÔNG bỏ bớt thông tin (`abs()`, xoá `%`); test cả hai chiều (câu đúng qua, câu sai bị chặn).
- Field từ LLM (plan, spec): kiểm miền giá trị và kiểu (`_coerce_filter_value`, `ALLOWED_GRAINS`), không dùng `.get(x, default)` im lặng. Định dạng số quyết định theo CỘT (`detect_decimal`).
