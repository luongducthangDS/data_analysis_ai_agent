"""Đọc số từ dữ liệu người thật nhập — tiền tệ, dấu phân cách, số âm kế toán.

`pd.to_numeric` bó tay trước những giá trị rất phổ biến trong file tài chính:

    " $ 16,185.00 "   → 16185.00
    " $ (4,533.75) "  → -4533.75      (ngoặc đơn = âm, quy ước kế toán)
    " $ -   "         → 0.0           (gạch ngang = không có giá trị)
    "1.234,56 €"      → 1234.56       (định dạng châu Âu)

Khi không đọc được, cả file bị tính sai im lặng: `sum()` trên cột toàn NaN
trả về 0, và agent tự tin báo "Tổng lợi nhuận là 0".

Tách riêng thành module để `storage` (lúc nạp file) và `analysis_planner`
(lúc thực thi plan) cùng dùng mà không tạo vòng import.
"""
from __future__ import annotations

import re

import pandas as pd

# Ký hiệu tiền tệ và đơn vị hay gặp, cắt bỏ trước khi đọc số.
_CURRENCY_CHARS = r"$€£¥₫₹#"
_UNIT_SUFFIXES = ("vnd", "vnđ", "usd", "eur", "gbp", "jpy", "aud", "cad", "đ")

# Giá trị mang nghĩa "trống" trong báo cáo tài chính.
_BLANK_TOKENS = {"", "-", "--", "—", "–", "n/a", "na", "nan", "none", "null", "."}


def parse_number(value: object) -> float | None:
    """Đọc một ô thành float. Trả None khi ô đó không phải số.

    Trả None chứ không phải 0.0 khi không đọc được — nhầm hai thứ này làm
    một cột chữ biến thành cột toàn 0 và mọi phép tính sau đó đều sai.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip().replace(" ", " ")
    if text.lower() in _BLANK_TOKENS:
        return None

    # Thứ tự quan trọng: bỏ ký hiệu tiền tệ và đơn vị TRƯỚC, vì " $ (4,533.75) "
    # bắt đầu bằng "$" nên nếu kiểm tra ngoặc đơn trước sẽ không nhận ra số âm.
    lowered = text.lower()
    for suffix in _UNIT_SUFFIXES:
        if lowered.endswith(suffix):
            text = text[: len(text) - len(suffix)].strip()
            break
    text = re.sub(f"[{re.escape(_CURRENCY_CHARS)}]", "", text).strip()

    # Ngoặc đơn = số âm (quy ước kế toán).
    negative = text.startswith("(") and text.endswith(")")
    if negative:
        text = text[1:-1].strip()

    text = text.replace(" ", "")
    if text.lower() in _BLANK_TOKENS or text in {"-", "+"}:
        # " $ -  " nghĩa là 0 trong báo cáo tài chính, không phải "không đọc được".
        return 0.0

    # Số sạch (kể cả "1e3", "-17") đọc thẳng, khỏi đoán dấu phân cách.
    try:
        number = float(text)
    except ValueError:
        pass
    else:
        return -number if negative else number

    # Còn sót chữ cái → đây là text, không phải số. Phải trả None, nếu không
    # một cột tên sản phẩm sẽ bị biến thành cột 0.
    if re.search(r"[A-Za-zÀ-ỹ]", text):
        return None

    text = _normalize_separators(text)
    if not re.fullmatch(r"[+-]?\d*\.?\d+", text):
        return None

    try:
        number = float(text)
    except ValueError:
        return None
    return -number if negative else number


def _normalize_separators(text: str) -> str:
    """Quy dấu phân cách nghìn/thập phân về dạng chuẩn.

    "1,234.56" (Anh-Mỹ) và "1.234,56" (châu Âu) đều hợp lệ, phải đoán bằng
    vị trí: dấu xuất hiện SAU cùng là dấu thập phân.
    """
    # Nhiều dấu chấm mà không có phẩy → tất cả là phân cách nghìn
    # ("32.000.000" kiểu Việt Nam / châu Âu).
    if text.count(".") > 1 and "," not in text:
        return text.replace(".", "")
    if text.count(",") > 1 and "." not in text:
        return text.replace(",", "")

    has_comma, has_dot = "," in text, "." in text
    if has_comma and has_dot:
        if text.rfind(",") > text.rfind("."):      # 1.234,56 → châu Âu
            return text.replace(".", "").replace(",", ".")
        return text.replace(",", "")               # 1,234.56 → Anh-Mỹ
    if has_comma:
        # Một dấu phẩy: "1,5" là thập phân, "1,234" là phân cách nghìn.
        whole, _, frac = text.rpartition(",")
        if len(frac) == 3 and whole.lstrip("+-").isdigit():
            return text.replace(",", "")
        return text.replace(",", ".")
    return text


def parse_numeric_series(series: pd.Series) -> pd.Series:
    """Ép một cột về số, hiểu cả định dạng tiền tệ."""
    if pd.api.types.is_numeric_dtype(series):
        return series
    converted = pd.to_numeric(series, errors="coerce")
    # Nếu cách nhanh đã đọc được hết thì khỏi đi đường chậm.
    if converted.notna().sum() == series.notna().sum():
        return converted
    return series.map(parse_number).astype("float64")


# Tỉ lệ ô đọc được tối thiểu để coi cả cột là cột số. Đặt cao để một cột chữ
# lẫn vài con số không bị ép kiểu nhầm.
MIN_PARSE_RATIO = 0.9


def coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Chuyển các cột tiền tệ dạng chữ thành cột số thật.

    Chỉ đổi khi gần như cả cột đọc được (`MIN_PARSE_RATIO`) VÀ `pd.to_numeric`
    thường đã bó tay — tức đúng loại cột mà hàm này sinh ra để cứu.
    """
    for column in df.columns:
        series = df[column]
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_datetime64_any_dtype(series):
            continue
        non_null = series.notna().sum()
        if not non_null:
            continue
        parsed = series.map(parse_number)
        if parsed.notna().sum() >= MIN_PARSE_RATIO * non_null:
            df[column] = parsed.astype("float64")
    return df


def strip_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Bỏ khoảng trắng thừa ở tên cột (" Profit " → "Profit").

    Giữ nguyên tên gốc nếu việc cắt làm hai cột trùng tên — thà để tên xấu
    còn hơn làm mất một cột dữ liệu.
    """
    stripped = [str(c).strip() for c in df.columns]
    if len(set(stripped)) != len(stripped):
        return df
    df.columns = stripped
    return df
