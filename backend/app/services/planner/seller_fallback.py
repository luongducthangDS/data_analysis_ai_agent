"""Plan luật cho bảng đơn TMĐT (có semantic layer) khi LLM không gọi được.

Chỉ trả lời khi hiểu HẾT câu hỏi: metric, mọi giá trị lọc (kỳ, kênh, tỉnh, đơn vị vận chuyển, danh mục, SKU) và
chiều chia. Gặp ý định chưa có luật (vì sao, so với, biên lãi, theo tuần…) thì trả plan `_generic` để synthesize
nói "chưa hiểu" — trả lãi cả shop cho câu hỏi về một tháng của một sàn là số sai trông như đúng.
"""
from __future__ import annotations

import calendar
import re
import unicodedata
from typing import Any

import pandas as pd

TIME_COL = "ngay_dat"
LABELS = {
    "loi_nhuan_truoc_qc": "Lãi trước QC", "loi_nhuan_rong": "Lãi ròng", "doanh_thu_thuan": "Doanh thu thuần",
    "phi_san": "Phí sàn", "chi_phi_qc": "Chi phí quảng cáo", "chi_phi_hoan": "Chi phí hoàn hàng",
}
RATE, ORDERS, FEE_RATE = "Tỷ lệ hoàn", "Số đơn", "Tỷ lệ phí sàn"
NO_METRIC = "không nhận ra chỉ số"

# Câu cần phép tính chưa có luật → từ chối, không đoán.
_UNSUPPORTED = re.compile(
    r"\b(?:vi sao|tai sao|nguyen nhan|bat thuong|so voi|chenh|tang|giam|bien|ty suat|trung binh|moi don|du bao"
    r"|neu|gia su|tuan|hom nay|hom qua|ngay nao|theo ngay|tung ngay|moi ngay|lai suat)\b")
# Thứ tự quan trọng: cụm cụ thể trước ("lãi trước quảng cáo" là lãi, không phải chi phí quảng cáo;
# "phí ship hoàn" không phải phí sàn).
_METRIC_HINTS: tuple[tuple[str, str], ...] = (
    (r"(?:lai|loi nhuan|loi) truoc (?:quang cao|qc|ads?)\b", "loi_nhuan_truoc_qc"),
    (r"\b(?:lai|loi nhuan|loi) rong\b|\bsau (?:khi )?(?:tru )?(?:quang cao|qc|ads?)\b|\btru (?:quang cao|qc|ads?)\b"
     r"|net profit", "loi_nhuan_rong"),
    (r"\b(?:phi|tien|chi phi) (?:ship |van chuyen )?(?:hoan|tra hang)\b", "chi_phi_hoan"),
    (r"quang cao|\bqc\b|\bads?\b", "chi_phi_qc"),
    (r"\bphi\b|platform fee", "phi_san"),
    (r"ty le hoan|hoan hang|tra hang|bi hoan|\bhoan\b(?! thanh)|return", "la_don_hoan"),
    (r"doanh thu|\bdt\b|revenue|ban chay|ban duoc", "doanh_thu_thuan"),
    (r"\b(?:lai|loi|loi nhuan|lo)\b|profit", "loi_nhuan_truoc_qc"),
)
_ORDERS = re.compile(r"\b(?:bao nhieu|may|tong so|so luong|so) don\b(?! vi)|\bdon hang\b")
_ORDER_STATUS = ((r"\bhoan thanh\b", "Hoàn thành"), (r"\b(?:bi |da )?huy\b", "Đã huỷ"),
                 (r"\b(?:bi hoan|tra hang|hoan)\b", "Đã trả hàng"))
_LOSS = re.compile(r"\b(?:lo|thua lo)\b")
_PERCENT = re.compile(r"ty le|%|phan tram|chiem|ty trong")
_SUPERLATIVE = re.compile(r"(?:cao|nhieu|lon|chay|tot|thap|it|nho|kem|nang) nhat|\btop\b|(?:nhieu|cao|it|thap) hon")
_ASCENDING = re.compile(r"(?:thap|it|nho|kem) (?:nhat|hon)")

# Chiều chia: từ khoá (đã bỏ dấu) → cột. Cần từ đi kèm ("tỉnh nào", "theo kênh", "top 3 SKU") vì "tinh" trùng
# "tính", "san" trùng "phí sàn".
_DIM_WORDS = {
    "sku": r"sku|ma hang|ma sp|ma san pham", "ten_san_pham": r"san pham|mat hang",
    "danh_muc": r"danh muc|nganh hang|nhom hang", "kenh": r"kenh|san",
    "tinh": r"tinh|tinh thanh|thanh pho",
    "don_vi_van_chuyen": r"don vi van chuyen|dvvc|nha van chuyen|ben van chuyen|hang van chuyen|van chuyen",
}
_PLAIN_DIM_WORDS = ("sku", "ten_san_pham", "danh_muc", "don_vi_van_chuyen")  # đủ rõ để đi với "cao nhất"
_VALUE_COLS = ("kenh", "tinh", "don_vi_van_chuyen", "danh_muc", "sku", "ten_san_pham")
_ALIASES = {"TP.HCM": ("hcm", "tphcm", "ho chi minh", "sai gon")}
# Tên ngắn của kênh/ĐVVC = bỏ hậu tố chung ("TikTok Shop" → "tiktok", "J&T Express" → "j&t"). Không lấy từ đầu:
# "Giao Hàng Nhanh" → "giao" khớp mọi câu "giao hàng".
_BRAND_SUFFIX = re.compile(r"\s+(?:express|shop|post|mall|official)$")
# ponytail: tên ngắn trùng từ thường ("best seller" ≠ Best Express); gặp thêm thì bổ sung.
_COMMON_WORDS = {"best", "top", "hot", "sale", "new", "fast"}


def plain(text: Any) -> str:
    """Chữ thường, bỏ dấu, đ→d: "Tỉnh Đà Nẵng" → "tinh da nang"."""
    t = unicodedata.normalize("NFKD", str(text).lower().replace("đ", "d"))
    return re.sub(r"\s+", " ", "".join(ch for ch in t if not unicodedata.combining(ch))).strip()


def _refuse(reason: str) -> dict[str, Any]:
    return {"action": "profile", "_generic": True, "_refuse_reason": reason}


def value_filters(df: pd.DataFrame, q: str) -> tuple[list[dict], list[str]]:
    """Giá trị nêu trong câu → filter. Một cột có ≥ 2 giá trị ("Shopee hay TikTok") → lọc `in` VÀ chia theo cột đó."""
    filters, split = [], []
    for col in _VALUE_COLS:
        if col not in df.columns:
            continue
        found = []
        for v in df[col].dropna().unique():
            keys = {plain(v), *_ALIASES.get(v, ())}
            if col in ("kenh", "don_vi_van_chuyen"):
                short = _BRAND_SUFFIX.sub("", plain(v))
                if short not in _COMMON_WORDS:
                    keys.add(short)
            for k in keys:
                pat = rf"(?<![\w&]){re.escape(k)}(?![\w&])"
                if col == "danh_muc" and len(k) <= 4:  # "áo", "quần" chỉ là danh mục khi có chữ dẫn
                    pat = rf"\b(?:danh muc|nhom|loai|mat hang) {re.escape(k)}(?![\w&])"
                if re.search(pat, q):
                    found.append(v)
                    break
        if len(found) == 1:
            filters.append({"column": col, "operator": "eq", "value": found[0]})
        elif found:
            filters.append({"column": col, "operator": "in", "value": found})
            split.append(col)
    return filters, split


def _period_filter(df: pd.DataFrame, q: str) -> tuple[dict | None, str | None]:
    """(filter ngày, lý do từ chối). Kỳ ngoài dữ liệu vẫn lọc → 0 dòng → synthesize nói dữ liệu có từ đâu đến đâu."""
    last = pd.to_datetime(df[TIME_COL], errors="coerce").max()
    if pd.isna(last):
        return None, None
    y = last.year
    if m := re.search(r"\bnam (20\d\d)\b", q):
        y = int(m.group(1))
    elif re.search(r"\bnam (?:ngoai|truoc)\b", q):
        y -= 1
    months = [int(x) for x in re.findall(r"\bthang (\d{1,2})\b", q)]
    if any(not 1 <= x <= 12 for x in months):
        return None, "tháng không hợp lệ"
    lo = hi = None
    if m := re.search(r"\btu thang (\d{1,2}) den (?:het )?thang (\d{1,2})\b", q):
        lo, hi = sorted((int(m.group(1)), int(m.group(2))))
    elif len(months) == 1:
        lo = hi = months[0]
    elif months:
        return None, "nhiều tháng"
    elif m := re.search(r"\bquy ([1-4])\b", q):
        lo, hi = 3 * int(m.group(1)) - 2, 3 * int(m.group(1))
    elif m := re.search(r"\bthang (nay|truoc)\b", q):
        p = last.to_period("M") - (m.group(1) == "truoc")
        y, lo, hi = p.year, p.month, p.month
    elif m := re.search(r"\b(\d{1,2}) thang dau(?: nam)?\b", q):
        lo, hi = 1, int(m.group(1))
    elif "nua dau nam" in q:
        lo, hi = 1, 6
    elif m := re.search(r"\b(\d{1,2}) thang (?:gan (?:nhat|day)|cuoi)\b", q):
        start = (last.to_period("M") - int(m.group(1)) + 1).start_time
        return {"column": TIME_COL, "operator": "between", "value": [f"{start:%Y-%m-%d}", f"{last:%Y-%m-%d}"]}, None
    elif re.search(r"\bnam (?:20\d\d|ngoai|truoc|nay)\b", q):
        lo, hi = 1, 12
    elif m := re.search(r"\b(\d{1,2}) thang\b", q):  # "6 tháng" = cả dữ liệu; số khác thì không rõ là tháng nào
        first = pd.to_datetime(df[TIME_COL], errors="coerce").min()
        span = (last.year - first.year) * 12 + last.month - first.month + 1
        return (None, None) if int(m.group(1)) >= span else (None, "khoảng thời gian không rõ")
    if lo is None:
        return None, None
    end = calendar.monthrange(y, hi)[1]
    return {"column": TIME_COL, "operator": "between", "value": [f"{y}-{lo:02d}-01", f"{y}-{hi:02d}-{end}"]}, None


def _metric(df: pd.DataFrame, q: str) -> str | None:
    if _ORDERS.search(q) and "ma_don" in df.columns:
        return ORDERS
    for pattern, col in _METRIC_HINTS:
        if re.search(pattern, q):
            return col
    return None


def _dims(df: pd.DataFrame, q: str, filtered: set[str]) -> tuple[list[str], str | None]:
    """(cột chia, grain thời gian "month"/"quarter" | None)."""
    dims = []
    for col, words in _DIM_WORDS.items():
        if col not in df.columns or col in filtered:
            continue
        cue = rf"\b(?:theo|tung|moi|cac|nhung|top \d+) (?:{words})\b|\b(?:{words}) nao\b"
        if col != "kenh":
            cue += rf"|\b(?:{words}) va\b"
        if re.search(cue, q) or (col in _PLAIN_DIM_WORDS and _SUPERLATIVE.search(q) and re.search(rf"\b(?:{words})\b", q)):
            dims.append(col)
    grain = None
    if re.search(r"\b(?:theo|tung|moi|hang|cac) thang\b|\bthang nao\b", q):
        grain = "month"
    elif re.search(r"\b(?:theo|tung|moi|cac) quy\b|\bquy nao\b", q):
        grain = "quarter"
    return dims, grain


def seller_plan(df: pd.DataFrame, question: str) -> dict[str, Any] | None:
    """Plan cho bảng đơn TMĐT, plan từ chối (`_generic`), hoặc None nếu không phải bảng đơn TMĐT."""
    if "doanh_thu_thuan" not in df.columns:
        return None
    q = plain(question)
    if _UNSUPPORTED.search(q):
        return _refuse("ý định chưa có luật")
    loss = bool(_LOSS.search(q)) and any(c in df.columns for c in ("loi_nhuan_truoc_qc", "loi_nhuan_rong"))
    metric = _metric(df, q)
    if loss and metric not in ("loi_nhuan_truoc_qc", "loi_nhuan_rong"):
        metric = "loi_nhuan_truoc_qc"
    if metric is None:
        return _refuse(NO_METRIC)
    if metric not in (ORDERS, "la_don_hoan") and metric not in df.columns:
        return _refuse(f"chưa có cột {metric}")  # vd lãi ròng khi chưa tải file quảng cáo
    fee_rate = metric == "phi_san" and bool(_PERCENT.search(q))
    if _PERCENT.search(q) and metric not in ("la_don_hoan",) and not fee_rate:
        return _refuse("tỷ lệ chưa có luật")

    filters, split = value_filters(df, q)
    period, why = _period_filter(df, q)
    if why:
        return _refuse(why)
    if period:
        filters.append(period)
    if metric == ORDERS:
        status = next((s for pat, s in _ORDER_STATUS if re.search(pat, q)), None)
        if status and "trang_thai" in df.columns:
            filters.append({"column": "trang_thai", "operator": "eq", "value": status})
    dims, grain = _dims(df, q, {f["column"] for f in filters if f["operator"] == "eq"})
    dims = split + [d for d in dims if d not in split]
    # "Danh mục Chân váy" mà không có giá trị đó (SKU ngoài bảng sản phẩm) → chia theo danh mục: bảng hiện nhóm
    # "(chưa có danh mục)" kèm category_notes, thay vì số cả shop.
    if "danh_muc" in df.columns and "danh_muc" not in dims and re.search(r"\bdanh muc \w", q) and not any(
            f["column"] == "danh_muc" for f in filters):
        dims.append("danh_muc")
    for col, words in _DIM_WORDS.items():  # hỏi theo một chiều mà bảng không có (vd danh mục khi chưa có bảng SP)
        if col not in df.columns and re.search(rf"\b(?:theo|tung|cac) (?:{words})\b|\b(?:{words}) nao\b", q):
            return _refuse(f"chưa có cột {col}")

    if metric == "la_don_hoan":
        metrics = [{"column": "la_don_hoan", "aggregation": "mean", "label": RATE}]
        if "ma_don" in df.columns:  # cỡ nhóm: 20% của 44 đơn khác 20% của 400 đơn
            metrics.append({"column": "ma_don", "aggregation": "count", "label": ORDERS})
        key = RATE
    elif metric == ORDERS:
        metrics, key = [{"column": "ma_don", "aggregation": "count", "label": ORDERS}], ORDERS
    elif fee_rate:
        metrics = [{"column": "phi_san", "aggregation": "sum", "label": LABELS["phi_san"]},
                   {"column": "doanh_thu_thuan", "aggregation": "sum", "label": LABELS["doanh_thu_thuan"]}]
        key = FEE_RATE
    else:
        metrics, key = [{"column": metric, "aggregation": "sum", "label": LABELS[metric]}], LABELS[metric]

    plan: dict[str, Any] = {"metrics": metrics, "filters": filters}
    if fee_rate:
        plan["ratios"] = [{"label": FEE_RATE, "numerator": LABELS["phi_san"], "denominator": LABELS["doanh_thu_thuan"]}]
    asc = bool(_ASCENDING.search(q))
    if loss and dims:  # "SKU nào đang lỗ" = nhóm có TỔNG lãi < 0 (filter từng đơn sẽ bỏ mất đơn lãi)
        plan["metrics"] = [{"column": "doanh_thu_thuan", "aggregation": "sum", "label": LABELS["doanh_thu_thuan"]},
                           metrics[0]]
        plan["having"] = [{"column": key, "operator": "lt", "value": 0}]
        if "doanh thu" in q or "ban nhieu" in q or "ban chay" in q:
            key, asc = LABELS["doanh_thu_thuan"], False
        else:
            asc = True
    top = re.search(r"\btop (\d+)\b", q)
    if grain and not dims:
        plan.update(action="time_series", time_column=TIME_COL, grain=grain)
        ranked = bool(_SUPERLATIVE.search(q)) and re.search(r"\b(?:thang|quy) nao\b", q)
        plan["sort"] = [{"column": key, "direction": "asc" if asc else "desc"}] if ranked else [
            {"column": grain, "direction": "asc"}]
        plan["limit"] = 1 if ranked else 24
        return plan
    if grain:  # chia theo kỳ VÀ theo chiều khác: kỳ thành cột dẫn xuất
        plan["derived_columns"] = [{"name": grain, "operation": grain, "source": TIME_COL}]
        dims = [grain, *dims]
    plan.update(action="aggregate", group_by=dims, sort=[{"column": key, "direction": "asc" if asc else "desc"}],
                limit=int(top.group(1)) if top else (5 if _SUPERLATIVE.search(q) else 20))
    return plan
