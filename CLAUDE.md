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