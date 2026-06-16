"""
E-commerce column alias registry and detection utilities.

Pure data + pure functions — no imports from the rest of the app.
This module maps canonical canonical field names to raw column header variants
found in real Shopee / Lazada / TikTok Shop export files.
"""
from __future__ import annotations

import re
import unicodedata

import pandas as pd

# ---------------------------------------------------------------------------
# Canonical field → list of raw aliases (Vietnamese + English variants)
# ---------------------------------------------------------------------------
PLATFORM_COLUMN_ALIASES: dict[str, list[str]] = {
    "revenue_col": [
        # Shopee
        "Thành tiền đơn hàng",
        "Tổng số tiền người mua thanh toán",
        "Số tiền người mua thanh toán",
        "Tổng giá trị đơn hàng",
        "Tổng số tiền thanh toán",
        # Lazada
        "Seller Revenue",
        "Total (Excl. Shipping)",
        "Seller Discount",
        # TikTok Shop
        "Tổng thực thu",
        "Tiền hàng",
        # Generic Vietnamese
        "Doanh thu",
        "Tổng tiền",
        "Thành tiền",
        "Giá trị đơn",
        "Tiền bán",
        # Generic English
        "GMV",
        "Revenue",
        "Amount",
        "Total Amount",
        "Sale Amount",
        "Net Revenue",
    ],
    "order_id_col": [
        "Mã đơn hàng",
        "Mã đơn",
        "Số đơn hàng",
        "Order ID",
        "Order Number",
        "orderId",
        "Mã vận đơn",
        "Tracking Number",
    ],
    "order_date_col": [
        "Ngày đặt hàng",
        "Thời gian đặt hàng",
        "Ngày tạo đơn",
        "Ngày đặt",
        "Ngày mua",
        "Ngày tạo",
        "Ngày giao dịch",
        "Ngày thanh toán",
        "Order Date",
        "Created Date",
        "Payment Date",
        "Transaction Date",
    ],
    "status_col": [
        "Trạng thái đơn hàng",
        "Trạng thái",
        "Tình trạng đơn",
        "Tình trạng",
        "Order Status",
        "Status",
        "Delivery Status",
    ],
    "product_name_col": [
        "Tên sản phẩm",
        "Tên SP",
        "Sản phẩm",
        "Tên hàng",
        "Tên mặt hàng",
        "Product Name",
        "Item Name",
        "SKU Name",
        "Variation Name",
    ],
    "sku_col": [
        "Mã SKU",
        "SKU",
        "Mã sản phẩm",
        "Product SKU",
        "Item SKU",
        "Seller SKU",
        "Barcode",
    ],
    "quantity_col": [
        "Số lượng",
        "SL",
        "Số lượng bán",
        "Qty",
        "Quantity",
        "Units Sold",
        "Order Quantity",
    ],
    "platform_col": [
        "Sàn",
        "Sàn bán hàng",
        "Nền tảng",
        "Kênh bán",
        "Kênh",
        "Platform",
        "Channel",
        "Source",
        "Marketplace",
    ],
    "shop_col": [
        "Tên shop",
        "Shop",
        "Tên cửa hàng",
        "Cửa hàng",
        "Store Name",
        "Shop Name",
        "Seller Name",
    ],
    "discount_col": [
        "Giảm giá",
        "Voucher",
        "Chiết khấu",
        "Mã giảm giá",
        "Discount",
        "Coupon",
        "Voucher Amount",
        "Seller Voucher",
        "Platform Voucher",
    ],
    "shipping_fee_col": [
        "Phí vận chuyển",
        "Phí ship",
        "Shipping Fee",
        "Delivery Fee",
        "Freight",
    ],
    "cancel_reason_col": [
        "Lý do huỷ",
        "Lý do hủy",
        "Cancellation Reason",
        "Cancel Reason",
    ],
    "buyer_col": [
        "Tên người mua",
        "Người mua",
        "Khách hàng",
        "Customer Name",
        "Buyer",
        "Buyer Username",
        "Buyer Name",
    ],
}

# Status values that indicate a cancelled order
CANCEL_STATUSES: tuple[str, ...] = (
    "đã huỷ", "da huy", "hủy", "huy", "cancelled", "canceled",
    "cancel", "void", "rejected", "từ chối",
)

# Status values that indicate a returned order
RETURN_STATUSES: tuple[str, ...] = (
    "đã trả hàng", "da tra hang", "trả hàng", "tra hang",
    "returned", "return", "refunded", "refund", "hoàn trả",
    "hoàn hàng", "khiếu nại",
)

# Minimum canonical columns matched to be considered an e-commerce session
_MIN_ECOMMERCE_COLS = 2


def _normalize(text: str) -> str:
    """Strip diacritics, lowercase, collapse whitespace."""
    t = unicodedata.normalize("NFKD", text.strip().lower())
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", t)


def detect_ecommerce_columns(df: pd.DataFrame) -> dict[str, str]:
    """
    Map canonical field names to actual column names in df.

    Returns a dict like:
        {"revenue_col": "Thành tiền đơn hàng", "status_col": "Trạng thái", ...}

    A session is considered e-commerce if len(result) >= _MIN_ECOMMERCE_COLS.
    Detection is case-insensitive and strips Vietnamese diacritics.
    """
    df_cols = {_normalize(col): col for col in df.columns}
    result: dict[str, str] = {}

    for canonical, aliases in PLATFORM_COLUMN_ALIASES.items():
        # 1. Exact normalized match
        for alias in aliases:
            norm_alias = _normalize(alias)
            if norm_alias in df_cols:
                result[canonical] = df_cols[norm_alias]
                break
        if canonical in result:
            continue

        # 2. Substring match (alias contained in column or vice-versa)
        for alias in aliases:
            norm_alias = _normalize(alias)
            for norm_col, orig_col in df_cols.items():
                if norm_alias in norm_col or norm_col in norm_alias:
                    result[canonical] = orig_col
                    break
            if canonical in result:
                break

    return result


def is_ecommerce_session(col_map: dict[str, str]) -> bool:
    return len(col_map) >= _MIN_ECOMMERCE_COLS


def detect_platform(
    df: pd.DataFrame,
    col_map: dict[str, str],
    filename: str = "",
) -> str | None:
    """
    Heuristically detect which platform the data comes from.
    Returns "shopee" | "lazada" | "tiktok" | None.
    """
    # 1. Filename heuristic
    fn_lower = filename.lower()
    if "shopee" in fn_lower:
        return "shopee"
    if "lazada" in fn_lower:
        return "lazada"
    if "tiktok" in fn_lower or "tik tok" in fn_lower:
        return "tiktok"

    # 2. Platform column values
    if "platform_col" in col_map:
        col = col_map["platform_col"]
        vals = df[col].dropna().astype(str).str.lower().unique()
        if any("shopee" in v for v in vals):
            return "shopee"
        if any("lazada" in v for v in vals):
            return "lazada"
        if any("tiktok" in v or "tik tok" in v for v in vals):
            return "tiktok"

    # 3. Shopee-specific column names
    shopee_indicators = {"tong so tien nguoi mua thanh toan", "thanh tien don hang"}
    for col in df.columns:
        if _normalize(col) in shopee_indicators:
            return "shopee"

    # 4. Lazada-specific column names
    lazada_indicators = {"seller revenue", "total (excl. shipping)"}
    for col in df.columns:
        if _normalize(col) in lazada_indicators:
            return "lazada"

    return None
