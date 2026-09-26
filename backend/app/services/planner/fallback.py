"""Rule-based plan when the LLM planner is unavailable or its plan is rejected."""
from __future__ import annotations

import re
from typing import Any

import pandas as pd

from backend.app.services.analysis_intent import infer_grouped_metric_intent
from backend.app.services.ecommerce_semantic import (
    BRIDGE_REQUIRED, month_periods, pick_metric,
)
from backend.app.services.planner.execute import _normalize


def build_fallback_plan(df: pd.DataFrame, question: str) -> dict[str, Any]:
    """Rule-based fallback when LLM planning fails. Covers common DA/accounting patterns."""
    normalized = _normalize(question)

    bridge = _profit_bridge_fallback(df, normalized)
    if bridge:
        return bridge

    # Insight / overview / profile questions → use profile action immediately
    _INSIGHT_KEYWORDS = (
        "insight", "tong quan", "mo ta", "phan tich", "overview", "describe",
        "du lieu co gi", "co gi dang", "nhan xet", "danh gia", "bao nhieu cot",
        "bao nhieu dong", "so luong cot", "thong ke", "kham pha", "kien thuc",
        "hieu biet", "ket luan", "du lieu nhu the nao",
    )
    if any(kw in normalized for kw in _INSIGHT_KEYWORDS):
        return {"action": "profile"}

    numeric_cols = _nonzero_numeric_cols(df)
    datetime_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
    cat_cols_all = df.select_dtypes(include=["object", "category"]).columns.tolist()

    metric = _pick_metric_from_question(normalized, df) or (numeric_cols[0] if numeric_cols else None)
    top_n = _extract_top_n(normalized) or 10
    asc = any(kw in normalized for kw in ("nho nhat", "thap nhat", "it nhat", "lowest", "smallest", "bottom"))
    sort_dir = "asc" if asc else "desc"

    # ── 0a. Distribution: "phân phối / phân bố / histogram" ───────────────────
    if metric and any(kw in normalized for kw in ("phan phoi", "phan bo", "distribution", "histogram")):
        return {"action": "distribution", "column": metric, "bins": 10}

    # ── 0b. Range / spread: "range / khoảng giá trị / min max / biên độ" ──────
    if metric and any(kw in normalized for kw in ("range", "khoang gia tri", "bien do", "spread", "min max", "min-max")):
        return {
            "action": "compare_metrics",
            "metrics": [
                {"column": metric, "aggregation": "min", "label": "Thấp nhất"},
                {"column": metric, "aggregation": "max", "label": "Cao nhất"},
            ],
        }

    # ── 1. Time series: "theo tháng / quý / năm" ──────────────────────────────
    if datetime_cols:
        if any(kw in normalized for kw in ("thang", "month")):
            return _time_series_plan(datetime_cols[0], "month", metric, normalized)
        if any(kw in normalized for kw in ("quy", "quarter", "q1", "q2", "q3", "q4")):
            return _time_series_plan(datetime_cols[0], "quarter", metric, normalized)
        if any(kw in normalized for kw in (" nam ", "yearly", "annual")):
            return _time_series_plan(datetime_cols[0], "year", metric, normalized)

    # ── 2. Discount revenue ────────────────────────────────────────────────────
    if "discount" in normalized and any("quantity" == c.lower() for c in df.columns):
        return {
            "action": "compare_metrics",
            "derived_columns": [
                {"name": "gross_revenue", "operation": "multiply", "columns": ["quantity", "unit_price"]},
                {"name": "net_revenue_after_discount", "operation": "net_revenue_from_discount_pct",
                 "quantity": "quantity", "unit_price": "unit_price", "discount_pct": "discount_pct"},
            ],
            "metrics": [
                {"column": "gross_revenue", "aggregation": "sum", "label": "Doanh thu gốc"},
                {"column": "net_revenue_after_discount", "aggregation": "sum", "label": "Doanh thu sau discount"},
            ],
        }

    # ── 3. Status filters: "chưa thanh toán", "rejected", "pending" ───────────
    status_col = _find_col_by_keywords(cat_cols_all, ("status", "trang_thai", "state", "stage"))
    status_filters = _detect_status_filters(normalized, df, status_col)

    # ── 4. "so sánh X vs Y" — filter + group ──────────────────────────────────
    vs_values = _detect_vs_comparison(normalized, df, cat_cols_all)
    if vs_values and metric:
        col, vals = vs_values
        agg = _detect_aggregation(normalized)
        return {
            "action": "aggregate",
            "filters": [{"column": col, "operator": "in", "value": vals}],
            "group_by": [col],
            "metrics": [{"column": metric, "aggregation": agg, "label": f"{agg.capitalize()} {metric}"},
                        {"column": _find_id_col(df) or metric, "aggregation": "count", "label": "Số records"}],
            "sort": [{"column": f"{agg.capitalize()} {metric}", "direction": "desc"}],
            "limit": 20,
        }

    # ── 5. Direct column name match for dimension ──────────────────────────────
    dim = _match_dimension_from_question(normalized, df, cat_cols_all)
    intent = infer_grouped_metric_intent(question, df)
    if not dim and intent:
        dim = intent.dimension

    # Use status_col as dimension if question mentions status keywords but no other dim found
    if not dim and status_filters and status_col:
        dim = status_col
    if not dim and status_col and any(kw in normalized for kw in ("trang thai", "status", "trang_thai")):
        dim = status_col

    if dim and metric:
        agg = _detect_aggregation(normalized)
        filters = status_filters or []
        return {
            "action": "aggregate",
            "filters": filters,
            "group_by": [dim],
            "metrics": [{"column": metric, "aggregation": agg, "label": f"{agg.capitalize()} {metric}"}],
            "sort": [{"column": f"{agg.capitalize()} {metric}", "direction": sort_dir}],
            "limit": top_n,
        }

    # ── 6. "top N [metric]" — group by entity + sort ──────────────────────────
    is_who = _is_who_question(normalized)
    if metric and any(kw in normalized for kw in ("top", "lon nhat", "nhieu nhat", "cao nhat", "nho nhat", "thap nhat")):
        filters = status_filters or []
        actual_agg = "max" if sort_dir == "desc" else "min"
        label = f"{'Max' if sort_dir == 'desc' else 'Min'} {metric}"
        entity_col = _find_name_col(df, cat_cols_all) or _find_id_col(df) or (cat_cols_all[0] if cat_cols_all else None)
        actual_limit = 1 if is_who else top_n
        plan: dict[str, Any] = {
            "action": "aggregate",
            "filters": filters,
            "metrics": [{"column": metric, "aggregation": actual_agg, "label": label}],
            "sort": [{"column": label, "direction": sort_dir}],
            "limit": actual_limit,
        }
        if entity_col:
            plan["group_by"] = [entity_col]
        return plan

    # ── 7. Value filter: "doanh thu theo Travel" ──────────────────────────────
    value_filter = _detect_value_filter(normalized, df)
    if value_filter and metric:
        col, val = value_filter
        good_cat = [c for c in cat_cols_all if c != col and 1 < df[c].nunique() <= 30
                    and not _normalize(c).endswith("id")]
        group_col = min(good_cat, key=lambda c: df[c].nunique()) if good_cat else col
        return {
            "action": "aggregate",
            "filters": [{"column": col, "operator": "eq", "value": val}],
            "group_by": [group_col],
            "metrics": [{"column": metric, "aggregation": "sum", "label": f"Tổng {metric}"}],
            "sort": [{"column": metric, "direction": "desc"}],
            "limit": 20,
        }

    # ── 8. Numeric range / threshold filter ────────────────────────────────────
    # e.g. "số lượng học sinh điểm dưới 5", "count records where X < threshold"
    num_threshold = _detect_numeric_threshold(normalized)
    if num_threshold and metric:
        op, val = num_threshold
        return {
            "action": "aggregate",
            "filters": [{"column": metric, "operator": op, "value": val}],
            "metrics": [{"column": metric, "aggregation": "count", "label": "Số lượng"}],
            "limit": 1,
        }

    # ── 9. Last resort ─────────────────────────────────────────────────────────
    # `_generic`: không khớp luật nào → kết quả không trả lời câu hỏi; synthesize sẽ nói "chưa hiểu"
    # thay vì để LLM dựng chuyện quanh 4 con số tổng (từng bịa "đơn trang sức" cho câu "giá vàng hôm nay").
    if numeric_cols:
        return {
            "action": "compare_metrics",
            "metrics": [{"column": c, "aggregation": "sum", "label": c} for c in numeric_cols[:4]],
            "_generic": True,
        }
    return {"action": "profile", "_generic": True}


# ── Fallback helpers ───────────────────────────────────────────────────────────

_PROFIT_WORD = re.compile(r"\b(?:lai|loi nhuan|profit)\b")
_CHANGE_WORD = re.compile(r"\b(?:giam|tang|sut|tut|vi sao|tai sao|nguyen nhan|so voi|why)\b")


def _profit_bridge_fallback(df: pd.DataFrame, normalized: str) -> dict[str, Any] | None:
    """"Vì sao lãi tháng 5 giảm" → profit_bridge tháng 5 so với tháng 4 (hoặc tháng thứ hai được nhắc tới)."""
    if not set(BRIDGE_REQUIRED) <= set(df.columns) or "ngay_dat" not in df.columns:
        return None
    if not (_PROFIT_WORD.search(normalized) and _CHANGE_WORD.search(normalized)):
        return None
    plan: dict[str, Any] = {"action": "profit_bridge", "time_column": "ngay_dat"}
    months = [int(m) for m in re.findall(r"\bthang (\d{1,2})\b", normalized) if 1 <= int(m) <= 12]
    last = pd.to_datetime(df["ngay_dat"], errors="coerce").max()
    if months:
        plan["periods"] = month_periods(last.year, months[0], months[1] if len(months) > 1 else None)
    elif re.search(r"\bthang truoc\b", normalized) and not re.search(r"\bthang nay\b", normalized):
        prev = last.to_period("M") - 1  # "tháng trước" = tháng liền trước tháng cuối trong dữ liệu
        plan["periods"] = month_periods(prev.year, prev.month)
    return plan


def _detect_numeric_threshold(normalized: str) -> tuple[str, float] | None:
    """
    Parse queries like "điểm dưới 5", "score >= 8", "salary trên 10 triệu".
    Returns (operator, value) or None.
    """
    _OP_PATTERNS: list[tuple[str, str]] = [
        (r"(?:duoi|nho hon|<|thap hon|below|less than|under)\s+([\d,.]+)", "lt"),
        (r"(?:tren|lon hon|>|cao hon|above|greater than|over)\s+([\d,.]+)", "gt"),
        (r"(?:bang|=|equal)\s+([\d,.]+)", "eq"),
        (r"(?:>=|lon hon hoac bang|at least)\s+([\d,.]+)", "gte"),
        (r"(?:<=|nho hon hoac bang|at most)\s+([\d,.]+)", "lte"),
    ]
    for pattern, op in _OP_PATTERNS:
        m = re.search(pattern, normalized)
        if m:
            raw = m.group(1).replace(",", "")
            try:
                return op, float(raw)
            except ValueError:
                pass
    return None

def _time_series_plan(dt_col: str, grain: str, metric: str | None, normalized: str) -> dict[str, Any]:
    if not metric:
        return {"action": "profile"}
    filters = []
    year_match = re.search(r"\b(20\d{2})\b", normalized)
    if year_match:
        year = year_match.group(1)
        filters.append({"column": dt_col, "operator": "between", "value": [f"{year}-01-01", f"{year}-12-31"]})
    derived = []
    time_col = grain
    if grain == "quarter":
        derived = [{"name": "quarter", "operation": "quarter", "source": dt_col}]
        time_col = "quarter"
    return {
        "action": "time_series",
        "derived_columns": derived,
        "filters": filters,
        "time_column": time_col if not derived else None,
        **({"time_column": dt_col} if not derived else {"time_column": "quarter"}),
        "grain": grain,
        "metrics": [{"column": metric, "aggregation": "sum", "label": f"Tổng {metric}"}],
        "sort": [{"column": grain, "direction": "asc"}],
        "limit": 24,
    }


def _pick_metric_from_question(normalized: str, df: pd.DataFrame) -> str | None:
    semantic = pick_metric(normalized, df)
    if semantic:
        return semantic
    numeric_cols = _nonzero_numeric_cols(df)
    # Direct column name in question
    for col in numeric_cols:
        if _normalize(col) in normalized:
            return col
    # Keyword match
    return _pick_metric(df, ("amount", "revenue", "doanh thu", "sales", "total", "cost", "price", "value", "tien"))


def _nonzero_numeric_cols(df: pd.DataFrame) -> list[str]:
    """Return numeric columns that have at least some non-zero values — skip all-zero derived cols."""
    cols = []
    for col in df.select_dtypes(include="number").columns:
        try:
            if df[col].abs().sum() > 0:
                cols.append(col)
        except Exception:
            cols.append(col)
    return cols or df.select_dtypes(include="number").columns.tolist()


# Synonym map: Vietnamese/English phrase -> keyword that appears in column name
_SYNONYMS: list[tuple[str, str]] = [
    ("trang thai", "status"),
    ("trang_thai", "status"),
    ("nhan vien", "employee"),
    ("employee", "employee"),
    ("nguoi duyet", "approved"),
    ("manager", "approved"),
    ("approvedby", "approved"),
    ("quan ly", "approved"),
    ("loai", "category"),
    ("phan loai", "category"),
    ("don vi tien", "currency"),
    ("tien te", "currency"),
    ("ngay", "date"),
]


def _match_dimension_from_question(normalized: str, df: pd.DataFrame, cat_cols: list[str]) -> str | None:
    """Find a categorical column whose name (or partial name) appears in the question."""
    for col in cat_cols:
        col_norm = _normalize(col)
        # Exact match
        if col_norm in normalized:
            return col
        # Partial: any word in col_norm appears in question (min 4 chars to avoid noise)
        for word in col_norm.split():
            if len(word) >= 4 and word in normalized:
                return col
        # Reverse: any word from question (min 4 chars) appears as substring of col_norm
        # Skip ID columns (high cardinality, not useful as dimensions)
        is_id_col = col_norm.endswith("id") or col_norm.endswith("_id") or col_norm == "id"
        if not is_id_col:
            for qword in normalized.split():
                if len(qword) >= 4 and qword in col_norm:
                    return col
    # Synonym fallback: map Vietnamese phrase -> column keyword
    for phrase, keyword in _SYNONYMS:
        if phrase in normalized:
            for col in cat_cols:
                if keyword in _normalize(col):
                    return col
    return None


def _detect_aggregation(normalized: str) -> str:
    if any(kw in normalized for kw in ("trung binh", "average", "mean", "avg")):
        return "mean"
    if any(kw in normalized for kw in ("so luong", "count", "dem", "bao nhieu")):
        return "count"
    if any(kw in normalized for kw in ("lon nhat", "max", "cao nhat")):
        return "max"
    if any(kw in normalized for kw in ("nho nhat", "min", "thap nhat")):
        return "min"
    return "sum"


def _detect_status_filters(normalized: str, df: pd.DataFrame, status_col: str | None) -> list[dict]:
    if not status_col:
        return []
    status_keywords = {
        "chua thanh toan": ("ne", "Paid"),
        "chua duyet": ("in", ["Submitted", "Pending"]),
        "rejected": ("eq", "Rejected"),
        "bi tu choi": ("eq", "Rejected"),
        "da duyet": ("eq", "Approved"),
        "da thanh toan": ("eq", "Paid"),
        "paid": ("eq", "Paid"),
        "approved": ("eq", "Approved"),
        "submitted": ("eq", "Submitted"),
    }
    for kw, (op, val) in status_keywords.items():
        if kw in normalized:
            return [{"column": status_col, "operator": op, "value": val}]
    return []


def _detect_vs_comparison(normalized: str, df: pd.DataFrame, cat_cols: list[str]) -> tuple[str, list[str]] | None:
    """Detect 'X vs Y' or 'so sánh X và Y' patterns."""
    # Pattern: "X vs Y" or "so sanh X va Y"
    vs_match = re.search(r"\bvs\b|\bversus\b", normalized)
    va_match = re.search(r"so sanh\s+([\w]+)\s+va\s+([\w]+)", normalized)
    pairs = []
    if vs_match:
        before = normalized[:vs_match.start()].strip().split()
        after = normalized[vs_match.end():].strip().split()
        if before and after:
            pairs.append((before[-1], after[0]))
    if va_match:
        pairs.append((va_match.group(1), va_match.group(2)))
    for v1, v2 in pairs:
        for col in cat_cols:
            vals_lower = {str(v).lower(): str(v) for v in df[col].dropna().unique()}
            if v1 in vals_lower and v2 in vals_lower:
                return col, [vals_lower[v1], vals_lower[v2]]
    return None


def _find_col_by_keywords(cols: list[str], keywords: tuple[str, ...]) -> str | None:
    for col in cols:
        if any(kw in _normalize(col) for kw in keywords):
            return col
    return None


def _find_id_col(df: pd.DataFrame) -> str | None:
    """Find a meaningful ID/name column to use as dimension."""
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    # Prefer: not high cardinality ID, not pure ID suffix
    for col in cat_cols:
        n = df[col].nunique()
        if 1 < n <= 50 and not _normalize(col).endswith("id"):
            return col
    return None


_NAME_HINTS = ("name", "ten", "ho ten", "hoten", "ho_ten", "hoc sinh", "hocsinh",
               "student", "khach hang", "khachhang", "nhan vien", "nhanvien",
               "employee", "customer", "nguoi", "person", "user")


def _find_name_col(df: pd.DataFrame, cat_cols: list[str]) -> str | None:
    """Find the entity/name column — prefer cols whose name hints at a person/entity."""
    for col in cat_cols:
        cn = _normalize(col)
        if any(h in cn for h in _NAME_HINTS):
            return col
    # fallback: low-cardinality categorical col that is NOT an ID
    for col in cat_cols:
        n = df[col].nunique()
        if 2 <= n <= 100 and not _normalize(col).endswith("id"):
            return col
    return None


def _detect_value_filter(normalized_question: str, df: pd.DataFrame) -> tuple[str, str] | None:
    """Check if any word/phrase in the question matches a categorical column VALUE."""
    match = re.search(r"\btheo\s+([\w\s]+?)(?:\s*$|\s+va\s|\s+hoac\s)", normalized_question)
    phrase = match.group(1).strip() if match else None
    if not phrase:
        return None
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in cat_cols:
        values_lower = {str(v).lower(): str(v) for v in df[col].dropna().unique()}
        if phrase in values_lower:
            return col, values_lower[phrase]
    return None


# Khớp NGUYÊN TỪ: kiểu chuỗi con "ai " từng khớp nhầm "lãi tháng 5" (bỏ dấu = "lai thang").
_WHO_PATTERN = re.compile(
    r"\b(?:ai|nguoi nao|hoc sinh nao|khach hang nao|nhan vien nao|ten nao|nao co|san pham nao|mat hang nao|cai nao)\b"
)


def _is_who_question(normalized_question: str) -> bool:
    return bool(_WHO_PATTERN.search(normalized_question))


def _pick_metric(df: pd.DataFrame, tokens: tuple[str, ...]) -> str | None:
    normalized_tokens = tuple(_normalize(token) for token in tokens)
    for col in df.select_dtypes(include="number").columns:
        normalized = _normalize(col)
        if any(token in normalized for token in normalized_tokens):
            return str(col)
    return None


def _extract_top_n(normalized: str) -> int | None:
    """Extract N from 'top N', 'top-5', '5 lớn nhất' etc."""
    m = re.search(r"\btop[\s\-]?(\d+)\b", normalized)
    if m:
        return int(m.group(1))
    m = re.search(r"\b(\d+)\s*(?:lon nhat|nhieu nhat|cao nhat|nho nhat|thap nhat|largest|biggest|smallest)\b", normalized)
    if m:
        return int(m.group(1))
    return None
