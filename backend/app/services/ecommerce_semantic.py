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


# Header file export gốc của sàn → schema chuẩn.
# ponytail: đúng 2 định dạng mô phỏng ở data/samples/shop_lan; thêm sàn/phiên bản export = thêm 1 dict.
EXPORT_HEADERS: dict[str, dict[str, str]] = {
    "Shopee": {
        "Mã đơn hàng": "ma_don", "Ngày đặt hàng": "ngay_dat", "Trạng Thái Đơn Hàng": "trang_thai",
        "SKU phân loại hàng": "sku", "Tên sản phẩm": "ten_san_pham", "Số lượng": "so_luong", "Giá gốc": "gia_ban",
        "Tổng giá bán (sản phẩm)": "doanh_thu", "Mã giảm giá của Shop": "giam_gia_shop",
        "Phí cố định": "phi_hoa_hong", "Phí thanh toán": "phi_thanh_toan",
        "Phí Voucher Xtra & Freeship Xtra": "phi_voucher_freeship", "Phí Dịch Vụ": "phi_dich_vu",
        "Thuế khấu trừ": "thue_khau_tru", "Phí vận chuyển trả hàng": "chi_phi_hoan",
        "Tỉnh/Thành phố": "tinh", "Đơn Vị Vận Chuyển": "don_vi_van_chuyen",
    },
    "TikTok Shop": {
        "Order ID": "ma_don", "Created Time": "ngay_dat", "Order Status": "trang_thai",
        "Seller SKU": "sku", "Product Name": "ten_san_pham", "Quantity": "so_luong",
        "SKU Unit Original Price": "gia_ban", "SKU Subtotal Before Discount": "doanh_thu",
        "SKU Seller Discount": "giam_gia_shop", "Commission Fee": "phi_hoa_hong", "Transaction Fee": "phi_thanh_toan",
        "Campaign Service Fee": "phi_voucher_freeship", "Platform Service Fee": "phi_dich_vu",
        "Tax Withheld": "thue_khau_tru", "Return Shipping Fee": "chi_phi_hoan",
        "Province": "tinh", "Shipping Provider Name": "don_vi_van_chuyen",
    },
}
_DAYFIRST_CHANNELS = {"TikTok Shop"}  # "01/02/2026 ..." là 1/2, không phải 2/1
_EXPORT_CORE = {"ma_don", "ngay_dat", "trang_thai", "sku", "doanh_thu", "giam_gia_shop", *FEE_COLS}


def rename_export_columns(df: pd.DataFrame) -> pd.DataFrame:
    """File export Shopee/TikTok nhận ra được → đổi sang tên cột chuẩn và thêm cột kenh. Bảng khác giữ nguyên."""
    for channel, headers in EXPORT_HEADERS.items():
        present = {h: n for h, n in headers.items() if h in df.columns}
        if set(present.values()) >= _EXPORT_CORE:
            out = df.rename(columns=present)
            if channel in _DAYFIRST_CHANNELS:
                out["ngay_dat"] = pd.to_datetime(out["ngay_dat"], dayfirst=True, errors="coerce")
            if "kenh" not in out.columns:
                out["kenh"] = channel
            return out
    return df


def attach_cogs(sheets: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """File export sàn không có giá vốn: ghép từ bảng sản phẩm (cột sku, gia_von = giá vốn/đơn vị) rồi tính metric."""
    catalogs = [df for df in sheets.values() if {"sku", "gia_von"} <= set(df.columns) and "trang_thai" not in df.columns]
    if not catalogs:
        return sheets
    catalog = pd.concat(catalogs)
    unit_cost = pd.to_numeric(catalog["gia_von"], errors="coerce").groupby(catalog["sku"].astype(str).str.strip()).last()
    out = {}
    for key, df in sheets.items():
        if {"sku", "trang_thai"} <= set(df.columns) and "gia_von" not in df.columns:
            qty = pd.to_numeric(df["so_luong"], errors="coerce").fillna(1) if "so_luong" in df.columns else 1
            # SKU không có trong bảng sản phẩm → gia_von NaN, describe_metrics báo cho LLM.
            df = add_metric_columns(df.assign(gia_von=df["sku"].astype(str).str.strip().map(unit_cost) * qty))
        out[key] = df
    return out


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
        "  \"theo kênh/sàn\" → group_by kenh. Hỏi số của MỘT kỳ (vd tháng 5) → aggregate + filter ngày,",
        "  không dùng time_series (time_series bỏ qua group_by).",
    ]
    no_cost = int(df["gia_von"].isna().sum()) if "gia_von" in df.columns else 0
    if no_cost:
        lines.append(f"  CẢNH BÁO: {no_cost} đơn có SKU không có trong bảng sản phẩm nên thiếu giá vốn → lãi bị "
                     "tính cao hơn thực tế. Khi trả lời về lãi phải nói rõ điều này.")
    lines.append("")
    return "\n".join(lines)
