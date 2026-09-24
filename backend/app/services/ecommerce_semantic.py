"""
Semantic layer cho bảng đơn hàng TMĐT đã chuẩn hoá (schema của data/samples/shop_lan/shop_lan_orders.csv).

Metric được định nghĩa MỘT chỗ và tính tất định bằng pandas lúc nạp file. LLM không viết
công thức lợi nhuận, nó chỉ aggregate các cột đã tính sẵn (loi_nhuan_truoc_qc, phi_san...).
"""
from __future__ import annotations

import re

import pandas as pd

from backend.app.services.ecommerce_metrics import _is_cancelled, _is_returned, _normalize_status

FEE_COLS = ("phi_hoa_hong", "phi_thanh_toan", "phi_voucher_freeship", "phi_dich_vu", "thue_khau_tru")
REQUIRED_COLS = ("trang_thai", "doanh_thu", "giam_gia_shop", *FEE_COLS, "gia_von")

# ponytail: dict Python thay cho YAML, không thêm dependency; tách file khi có >1 schema.
METRICS: dict[str, str] = {
    "doanh_thu_thuan": "doanh thu đơn hoàn thành sau voucher shop (doanh_thu − giam_gia_shop); đơn huỷ/hoàn = 0",
    "phi_san": "tổng phí sàn của đơn: hoa hồng + thanh toán + voucher/freeship + dịch vụ + thuế khấu trừ",
    "loi_nhuan_truoc_qc": "lãi thật của đơn, CHƯA trừ quảng cáo: doanh_thu_thuan − phi_san − gia_von − chi_phi_hoan; đơn hoàn thường âm",
}

# Từ khoá trong câu hỏi (đã bỏ dấu) → metric. Thứ tự quan trọng: cụm dài trước.
_QUESTION_HINTS: tuple[tuple[str, str], ...] = (
    (r"doanh thu thuan|net revenue", "doanh_thu_thuan"),
    (r"phi san|tong phi|platform fee", "phi_san"),
    (r"\blai\b|\bloi nhuan\b|\blo\b|\bprofit\b|\bmargin\b", "loi_nhuan_truoc_qc"),
)


def add_metric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Thêm các cột trong METRICS nếu bảng có đủ REQUIRED_COLS. Không đụng tới cột người dùng đã có."""
    if not set(REQUIRED_COLS) <= set(df.columns) or set(METRICS) & set(df.columns):
        return df
    out = df.copy()

    def num(col: str) -> pd.Series:
        return pd.to_numeric(out[col], errors="coerce").fillna(0)

    status = _normalize_status(out["trang_thai"])
    done = ~(_is_cancelled(status) | _is_returned(status))
    out["doanh_thu_thuan"] = (num("doanh_thu") - num("giam_gia_shop")).where(done, 0)
    out["phi_san"] = sum(num(c) for c in FEE_COLS).where(done, 0)
    return_cost = num("chi_phi_hoan") if "chi_phi_hoan" in out.columns else 0
    out["loi_nhuan_truoc_qc"] = out["doanh_thu_thuan"] - out["phi_san"] - num("gia_von").where(done, 0) - return_cost
    return out


def pick_metric(normalized_question: str, df: pd.DataFrame) -> str | None:
    """Metric semantic ứng với câu hỏi (đã bỏ dấu, chữ thường), nếu bảng có cột đó."""
    for pattern, metric in _QUESTION_HINTS:
        if metric in df.columns and re.search(pattern, normalized_question):
            return metric
    return None


def describe_metrics(df: pd.DataFrame) -> str:
    """Khối mô tả metric cho planner prompt; rỗng nếu bảng không có semantic layer."""
    present = [m for m in METRICS if m in df.columns]
    if not present:
        return ""
    lines = ["", "METRIC ĐÃ TÍNH SẴN (semantic layer, ưu tiên dùng thay vì tự ghép công thức):"]
    lines += [f"  - {m}: {METRICS[m]}" for m in present]
    lines += [
        "  Hỏi lãi/lợi nhuận/lỗ → sum loi_nhuan_truoc_qc. Hỏi doanh thu (thực nhận) → sum doanh_thu_thuan.",
        "  Tỷ lệ phí sàn = sum(phi_san) / sum(doanh_thu_thuan) → compare_metrics với 2 metric sum đó.",
        "  Hỏi SKU/kênh nào lỗ → group_by rồi sum loi_nhuan_truoc_qc. KHÔNG filter loi_nhuan_truoc_qc < 0:",
        "  filter chạy trên TỪNG ĐƠN trước khi cộng, sẽ bỏ đơn lãi và cho tổng sai.",
        "  \"Top N doanh thu mà lỗ\" → sort theo doanh_thu_thuan desc, limit N, kèm metric loi_nhuan_truoc_qc.",
        "",
    ]
    return "\n".join(lines)
