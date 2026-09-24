## Lessons — Deploy Errors (2026-05-26)

### Khi edit config deploy, phải trace flow thực thi end-to-end trước khi confirm xong
- Không chỉ check từng file riêng lẻ — phải đọc lại toàn bộ file đã edit và simulate
  "Railway/Docker sẽ thực thi cái gì, theo thứ tự nào"

### railway.toml: startCommand ghi đè Dockerfile CMD
- Nếu dùng `builder = "dockerfile"`, Railway vẫn chạy `startCommand` nếu có —
  `startCommand` không expand shell variable (`$PORT` → lỗi, phải dùng `sh -c`)
- Khi chuyển builder, phải kiểm tra xem `startCommand` có conflict không

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
- Thêm action mới cho planner: sửa CẢ `ALLOWED_ACTIONS` (`analysis_planner.py`) lẫn `_ALLOWED_ACTIONS` + `_ACTION_ALIASES` (`agents/nodes/plan.py`).
- So khớp từ khoá trên câu hỏi đã bỏ dấu phải khớp NGUYÊN TỪ (`\b`): chuỗi con "ai " từng khớp nhầm "lãi"/"loại"/"cái".
- Chạy test: `.venv/Scripts/python.exe -m pytest -q` (python hệ thống thiếu pytest-mock → 5 lỗi giả ở test_api).
- eval_100 cần server đang chạy: `tests/eval_100.py --base-url http://localhost:PORT --ids ...`; baseline ở `docs/eval-baseline/`.
