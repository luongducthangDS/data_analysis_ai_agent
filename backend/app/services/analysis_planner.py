from __future__ import annotations

import json
import math
import re
import uuid
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import plotly.express as px

from backend.app.services.analysis_intent import infer_grouped_metric_intent
from backend.app.services.llm_service import get_llm_client


@dataclass
class PlannerResult:
    answer: str
    charts: list[dict[str, Any]]
    executed_queries: list[str] = field(default_factory=list)
    plan: dict[str, Any] = field(default_factory=dict)


ALLOWED_ACTIONS = {"aggregate", "compare_metrics", "time_series", "profile"}
ALLOWED_AGGREGATIONS = {"sum", "mean", "median", "min", "max", "count", "nunique"}
ALLOWED_DERIVED_OPS = {
    "multiply",
    "net_revenue_from_discount_pct",
    "quarter",
    "month",
    "year",
    "date",
}
ALLOWED_FILTER_OPERATORS = {"eq", "ne", "gt", "gte", "lt", "lte", "between", "in", "contains"}


def run_planned_analysis(df: pd.DataFrame, question: str | None, profile: dict[str, Any]) -> PlannerResult:
    if not question:
        raise ValueError("Planner needs a user question.")

    plan = _build_plan_with_llm(df, question, profile)
    result = execute_plan(df, plan)
    answer = _synthesize_answer(question, result, plan, source_df=df)
    charts = _build_charts_from_result(result, plan)
    return PlannerResult(
        answer=answer,
        charts=charts,
        executed_queries=[_describe_plan(plan)],
        plan=plan,
    )


def execute_plan(df: pd.DataFrame, plan: dict[str, Any]) -> pd.DataFrame:
    _validate_plan_shape(plan)
    work = df.copy()
    work = _apply_derived_columns(work, plan.get("derived_columns", []))
    work = _apply_filters(work, plan.get("filters", []))

    action = plan["action"]
    if action == "profile":
        return _profile_frame(work)
    if action == "aggregate":
        return _execute_aggregate(work, plan)
    if action == "compare_metrics":
        return _execute_compare_metrics(work, plan)
    if action == "time_series":
        return _execute_time_series(work, plan)
    raise ValueError(f"Unsupported action: {action}")


def build_fallback_plan(df: pd.DataFrame, question: str) -> dict[str, Any]:
    """Rule-based fallback when LLM planning fails. Covers common DA/accounting patterns."""
    normalized = _normalize(question)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
    cat_cols_all = df.select_dtypes(include=["object", "category"]).columns.tolist()

    metric = _pick_metric_from_question(normalized, df) or (numeric_cols[0] if numeric_cols else None)
    top_n = _extract_top_n(normalized) or 10
    asc = any(kw in normalized for kw in ("nho nhat", "thap nhat", "it nhat", "lowest", "smallest", "bottom"))
    sort_dir = "asc" if asc else "desc"

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

    # ── 6. "top N [metric]" — sort only, no grouping ──────────────────────────
    if metric and any(kw in normalized for kw in ("top", "lon nhat", "nhieu nhat", "cao nhat", "nho nhat", "thap nhat")):
        filters = status_filters or []
        return {
            "action": "aggregate",
            "filters": filters,
            "group_by": [_find_id_col(df) or cat_cols_all[0]] if cat_cols_all else [],
            "metrics": [{"column": metric, "aggregation": "sum", "label": f"Tổng {metric}"}],
            "sort": [{"column": metric, "direction": sort_dir}],
            "limit": top_n,
        }

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

    # ── 8. Last resort ─────────────────────────────────────────────────────────
    if numeric_cols:
        return {
            "action": "compare_metrics",
            "metrics": [{"column": c, "aggregation": "sum", "label": c} for c in numeric_cols[:4]],
        }
    return {"action": "profile"}


# ── Fallback helpers ───────────────────────────────────────────────────────────

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
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    # Direct column name in question
    for col in numeric_cols:
        if _normalize(col) in normalized:
            return col
    # Keyword match
    return _pick_metric(df, ("amount", "revenue", "doanh thu", "sales", "total", "cost", "price", "value", "tien"))


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


def _build_plan_with_llm(df: pd.DataFrame, question: str, profile: dict[str, Any]) -> dict[str, Any]:
    try:
        client = get_llm_client()
        prompt = _build_planner_prompt(df, question, profile)
        raw = client.generate(prompt, max_tokens=900, temperature=0.0, top_p=0.9)
        plan = _extract_json_object(raw)
        plan = _repair_plan_for_question(plan, question)
        _validate_plan_against_dataframe(df, plan)
        return plan
    except Exception as exc:
        fallback = build_fallback_plan(df, question)
        fallback["_planner_fallback_reason"] = str(exc)
        _validate_plan_against_dataframe(df, fallback)
        return fallback


def _build_planner_prompt(df: pd.DataFrame, question: str, profile: dict[str, Any]) -> str:
    schema_lines = []
    for col, dtype in profile["column_types"].items():
        examples = df[col].dropna().astype(str).head(5).tolist()
        schema_lines.append(f"- {col} ({dtype}): {examples}")

    # Detect currency columns for multi-currency warning in prompt
    currency_cols = [c for c in df.columns if _normalize(c) in ("currency", "tien_te", "don_vi_tien")]
    currency_note = ""
    if currency_cols:
        currencies = df[currency_cols[0]].dropna().unique().tolist()
        if len(currencies) > 1:
            currency_note = f"\nLƯU Ý: Cột '{currency_cols[0]}' có {len(currencies)} loại tiền tệ {currencies}. Nếu câu hỏi hỏi về 1 loại cụ thể, thêm filter currency vào plan.\n"

    return f"""Bạn là senior data analyst / financial analyst AI. Nhiệm vụ: chuyển câu hỏi tự nhiên (tiếng Việt hoặc Anh) thành JSON plan chính xác để backend thực thi bằng Pandas.

QUY TẮC TUYỆT ĐỐI:
1. Chỉ trả về JSON object thuần túy — không markdown, không giải thích, không code block.
2. Chỉ dùng tên cột CHÍNH XÁC từ SCHEMA bên dưới.
3. aggregation hợp lệ: sum | mean | median | min | max | count | nunique
4. operator hợp lệ: eq | ne | gt | gte | lt | lte | between | in | contains
5. Nếu câu hỏi hỏi "bao nhiêu", "số lượng", "count" → aggregation = "count"
6. Nếu câu hỏi hỏi "trung bình", "average" → aggregation = "mean"
7. Nếu câu hỏi hỏi "tỷ lệ %", "phần trăm" → vẫn dùng "sum" hoặc "count" để group, backend tính % từ đó
8. Nếu có cột ngày giờ và câu hỏi hỏi theo tháng/quý/năm → dùng time_series với grain phù hợp
9. Từ "chưa thanh toán" / "pending" → filter Status != "Paid" hoặc Status = "Submitted"/"Approved"
10. Từ "lớn nhất", "top N", "cao nhất" → sort desc, limit = N (mặc định 10)
11. Từ "nhỏ nhất", "thấp nhất", "bottom N" → sort asc, limit = N
{currency_note}
SCHEMA DATASET:
{chr(10).join(schema_lines)}

VÍ DỤ MINH HỌA (kế toán / kiểm toán / DA):

Q: "tổng amount theo category"
{{"action":"aggregate","group_by":["Category"],"metrics":[{{"column":"Amount","aggregation":"sum","label":"Tổng Amount"}}],"sort":[{{"column":"Amount","direction":"desc"}}],"limit":20}}

Q: "top 5 employee chi tiêu nhiều nhất"
{{"action":"aggregate","group_by":["EmployeeID"],"metrics":[{{"column":"Amount","aggregation":"sum","label":"Tổng chi tiêu"}}],"sort":[{{"column":"Amount","direction":"desc"}}],"limit":5}}

Q: "số lượng claim theo trạng thái"
{{"action":"aggregate","group_by":["Status"],"metrics":[{{"column":"ClaimID","aggregation":"count","label":"Số claim"}}],"sort":[{{"column":"Số claim","direction":"desc"}}],"limit":10}}

Q: "claim nào chưa thanh toán trên 500"
{{"action":"aggregate","filters":[{{"column":"Status","operator":"ne","value":"Paid"}},{{"column":"Amount","operator":"gt","value":500}}],"group_by":["Status"],"metrics":[{{"column":"Amount","aggregation":"sum","label":"Tổng Amount chưa TT"}}],"sort":[{{"column":"Amount","direction":"desc"}}],"limit":20}}

Q: "trend expense theo tháng năm 2024"
{{"action":"time_series","filters":[{{"column":"SubmitDate","operator":"between","value":["2024-01-01","2024-12-31"]}}],"time_column":"SubmitDate","grain":"month","metrics":[{{"column":"Amount","aggregation":"sum","label":"Tổng Amount"}}],"sort":[{{"column":"month","direction":"asc"}}],"limit":12}}

Q: "amount trung bình theo manager"
{{"action":"aggregate","group_by":["ApprovedBy"],"metrics":[{{"column":"Amount","aggregation":"mean","label":"Amount trung bình"}}],"sort":[{{"column":"Amount trung bình","direction":"desc"}}],"limit":10}}

Q: "so sánh approved vs rejected"
{{"action":"aggregate","filters":[{{"column":"Status","operator":"in","value":["Approved","Rejected"]}}],"group_by":["Status"],"metrics":[{{"column":"Amount","aggregation":"sum","label":"Tổng Amount"}},{{"column":"ClaimID","aggregation":"count","label":"Số claim"}}],"sort":[{{"column":"Tổng Amount","direction":"desc"}}],"limit":5}}

Q: "expense theo category và status"
{{"action":"aggregate","group_by":["Category","Status"],"metrics":[{{"column":"Amount","aggregation":"sum","label":"Tổng Amount"}}],"sort":[{{"column":"Tổng Amount","direction":"desc"}}],"limit":30}}

CÂU HỎI CẦN PHÂN TÍCH:
{question}""".strip()


def _repair_plan_for_question(plan: dict[str, Any], question: str) -> dict[str, Any]:
    repaired = dict(plan)
    normalized = _normalize(question)
    quarters = _mentioned_quarters(normalized)
    year_match = re.search(r"\b(20\d{2})\b", normalized)
    if not quarters or not year_match:
        return repaired

    year = int(year_match.group(1))
    derived_columns = list(repaired.get("derived_columns", []) or [])
    if not any(item.get("name") == "quarter" for item in derived_columns):
        source = repaired.get("time_column") or "order_date"
        derived_columns.append({"name": "quarter", "operation": "quarter", "source": source})
    repaired["derived_columns"] = derived_columns

    filters = list(repaired.get("filters", []) or [])
    filters.append(
        {
            "column": "quarter",
            "operator": "in",
            "value": [f"{year}Q{quarter}" for quarter in sorted(quarters)],
        }
    )
    repaired["filters"] = filters

    if repaired.get("action") in {"aggregate", "compare_metrics", "time_series"}:
        repaired["group_by"] = ["quarter"]
        repaired["sort"] = [{"column": "quarter", "direction": "asc"}]
        repaired["limit"] = len(quarters)
    return repaired


def _validate_plan_shape(plan: dict[str, Any]) -> None:
    if not isinstance(plan, dict):
        raise ValueError("Plan must be a JSON object.")
    action = plan.get("action")
    if action not in ALLOWED_ACTIONS:
        raise ValueError(f"Unsupported or missing action: {action}")
    if int(plan.get("limit", 100) or 100) < 1:
        raise ValueError("limit must be positive.")


def _validate_plan_against_dataframe(df: pd.DataFrame, plan: dict[str, Any]) -> None:
    _validate_plan_shape(plan)
    known = set(df.columns)
    for derived in plan.get("derived_columns", []) or []:
        op = derived.get("operation")
        if op not in ALLOWED_DERIVED_OPS:
            raise ValueError(f"Unsupported derived operation: {op}")
        _validate_derived_sources(known, derived)
        known.add(derived["name"])

    for item in plan.get("filters", []) or []:
        _require_column(known, item.get("column"))
        if item.get("operator") not in ALLOWED_FILTER_OPERATORS:
            raise ValueError(f"Unsupported filter operator: {item.get('operator')}")
    for col in plan.get("group_by", []) or []:
        _require_column(known, col)
    if plan.get("time_column"):
        _require_column(known, plan["time_column"])
    for metric in plan.get("metrics", []) or []:
        _require_column(known, metric.get("column"))
        if metric.get("aggregation", "sum") not in ALLOWED_AGGREGATIONS:
            raise ValueError(f"Unsupported aggregation: {metric.get('aggregation')}")


def _validate_derived_sources(known: set[str], derived: dict[str, Any]) -> None:
    if not derived.get("name"):
        raise ValueError("Derived column needs a name.")
    op = derived.get("operation")
    if op == "multiply":
        for col in derived.get("columns", []):
            _require_column(known, col)
    elif op in {"quarter", "month", "year", "date"}:
        _require_column(known, derived.get("source"))
    elif op == "net_revenue_from_discount_pct":
        for key in ("quantity", "unit_price", "discount_pct"):
            _require_column(known, derived.get(key))


def _apply_derived_columns(df: pd.DataFrame, derived_columns: list[dict[str, Any]]) -> pd.DataFrame:
    work = df.copy()
    for derived in derived_columns or []:
        name = derived["name"]
        op = derived["operation"]
        if op == "multiply":
            cols = derived["columns"]
            work[name] = _numeric(work[cols[0]])
            for col in cols[1:]:
                work[name] = work[name] * _numeric(work[col])
        elif op == "net_revenue_from_discount_pct":
            discount = _numeric(work[derived["discount_pct"]]).fillna(0) / 100
            work[name] = _numeric(work[derived["quantity"]]) * _numeric(work[derived["unit_price"]]) * (1 - discount)
        elif op == "quarter":
            work[name] = pd.to_datetime(work[derived["source"]], errors="coerce").dt.to_period("Q").astype(str)
        elif op == "month":
            work[name] = pd.to_datetime(work[derived["source"]], errors="coerce").dt.to_period("M").astype(str)
        elif op == "year":
            work[name] = pd.to_datetime(work[derived["source"]], errors="coerce").dt.year
        elif op == "date":
            work[name] = pd.to_datetime(work[derived["source"]], errors="coerce").dt.date.astype(str)
    return work


def _apply_filters(df: pd.DataFrame, filters: list[dict[str, Any]]) -> pd.DataFrame:
    work = df.copy()
    for item in filters or []:
        col = item["column"]
        op = item["operator"]
        value = item.get("value")
        series = work[col]
        if pd.api.types.is_datetime64_any_dtype(series):
            series_cmp = pd.to_datetime(series, errors="coerce")
            if isinstance(value, list):
                value_cmp = [pd.to_datetime(v) for v in value]
            else:
                value_cmp = pd.to_datetime(value)
        else:
            series_cmp = series
            value_cmp = value

        if op == "eq":
            mask = series_cmp == value_cmp
        elif op == "ne":
            mask = series_cmp != value_cmp
        elif op == "gt":
            mask = series_cmp > value_cmp
        elif op == "gte":
            mask = series_cmp >= value_cmp
        elif op == "lt":
            mask = series_cmp < value_cmp
        elif op == "lte":
            mask = series_cmp <= value_cmp
        elif op == "between":
            lo, hi = value_cmp
            mask = (series_cmp >= lo) & (series_cmp <= hi)
        elif op == "in":
            mask = series_cmp.isin(value_cmp if isinstance(value_cmp, list) else [value_cmp])
        elif op == "contains":
            mask = series_cmp.astype(str).str.contains(str(value_cmp), case=False, na=False)
        else:
            raise ValueError(f"Unsupported filter operator: {op}")
        work = work.loc[mask].copy()
    return work


def _execute_aggregate(df: pd.DataFrame, plan: dict[str, Any]) -> pd.DataFrame:
    group_by = plan.get("group_by", []) or []
    metrics = plan.get("metrics", []) or []
    if not metrics:
        raise ValueError("aggregate needs metrics.")

    if group_by:
        named_aggs = {}
        rename_map = {}
        for metric in metrics:
            source = metric["column"]
            agg = metric.get("aggregation", "sum")
            out = metric.get("label") or f"{agg}_{source}"
            safe_out = _safe_column(out)
            named_aggs[safe_out] = (source, agg)
            rename_map[safe_out] = out
        result = df.groupby(group_by, dropna=False).agg(**named_aggs).reset_index()
        result = result.rename(columns=rename_map)
    else:
        rows = {}
        for metric in metrics:
            label = metric.get("label") or f"{metric.get('aggregation', 'sum')}_{metric['column']}"
            rows[label] = _aggregate_series(df[metric["column"]], metric.get("aggregation", "sum"))
        result = pd.DataFrame([rows])

    return _sort_and_limit(result, plan)


def _execute_compare_metrics(df: pd.DataFrame, plan: dict[str, Any]) -> pd.DataFrame:
    rows = []
    group_by = plan.get("group_by", []) or []
    if group_by:
        return _execute_aggregate(df, plan)
    for metric in plan.get("metrics", []) or []:
        label = metric.get("label") or metric["column"]
        rows.append(
            {
                "metric": label,
                "value": _aggregate_series(df[metric["column"]], metric.get("aggregation", "sum")),
            }
        )
    if not rows:
        raise ValueError("compare_metrics needs metrics.")
    return pd.DataFrame(rows)


def _execute_time_series(df: pd.DataFrame, plan: dict[str, Any]) -> pd.DataFrame:
    time_col = plan.get("time_column")
    grain = plan.get("grain", "month")
    if not time_col:
        raise ValueError("time_series needs time_column.")
    work = df.copy()
    if grain == "quarter" and not pd.api.types.is_datetime64_any_dtype(work[time_col]):
        work["_period"] = work[time_col].astype(str)
    else:
        dt = pd.to_datetime(work[time_col], errors="coerce")
        if grain == "quarter":
            work["_period"] = dt.dt.to_period("Q").astype(str)
        elif grain == "year":
            work["_period"] = dt.dt.year
        elif grain == "date":
            work["_period"] = dt.dt.date.astype(str)
        else:
            work["_period"] = dt.dt.to_period("M").astype(str)

    plan2 = dict(plan)
    plan2["group_by"] = ["_period"]
    result = _execute_aggregate(work.dropna(subset=["_period"]), plan2)
    return result.rename(columns={"_period": grain})


def _profile_frame(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "rows": len(df),
                "columns": len(df.columns),
                "numeric_columns": len(df.select_dtypes(include="number").columns),
                "categorical_columns": len(df.select_dtypes(include=["object", "category", "bool"]).columns),
                "datetime_columns": len(df.select_dtypes(include=["datetime", "datetimetz"]).columns),
                "missing_cells": int(df.isna().sum().sum()),
            }
        ]
    )


def _sort_and_limit(result: pd.DataFrame, plan: dict[str, Any]) -> pd.DataFrame:
    for sort in reversed(plan.get("sort", []) or []):
        col = sort.get("column")
        if col not in result.columns:
            col = _resolve_metric_output_column(plan, col) or col
        if col not in result.columns:
            matched = _find_matching_result_column(result, col)
            if not matched:
                continue
            col = matched
        result = result.sort_values(col, ascending=sort.get("direction", "desc") == "asc")
    limit = min(int(plan.get("limit", 100) or 100), 500)
    return result.head(limit).reset_index(drop=True)


def _build_charts_from_result(result: pd.DataFrame, plan: dict[str, Any]) -> list[dict[str, Any]]:
    if result.empty or len(result.columns) < 2:
        return []
    x = result.columns[0]
    numeric_cols = result.select_dtypes(include="number").columns.tolist()
    y = numeric_cols[0] if numeric_cols else result.columns[1]
    if plan["action"] == "time_series":
        fig = px.line(result, x=x, y=y, title=f"{y} theo {x}", markers=True)
        return [_chart("line", f"{y} theo {x}", fig, x=x, y=y)]
    if plan["action"] in {"aggregate", "compare_metrics"}:
        fig = px.bar(result, x=x, y=y, title=f"{y} theo {x}", text=y)
        fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
        return [_chart("bar", f"{y} theo {x}", fig, x=x, y=y)]
    return []


def _synthesize_answer(question: str, result: pd.DataFrame, plan: dict[str, Any], source_df: pd.DataFrame | None = None) -> str:
    return _deterministic_answer(question, result, plan, source_df=source_df)


def _deterministic_answer(question: str, result: pd.DataFrame, plan: dict[str, Any], source_df: pd.DataFrame | None = None) -> str:
    filters = plan.get("filters", []) or []
    filter_desc = ""
    if filters:
        parts = []
        for f in filters:
            parts.append(f"{f.get('column')} = {f.get('value')}")
        filter_desc = f" (lọc: {', '.join(parts)})"

    lines = ["# Executive Data Brief", "", f"## Kết quả phân tích{filter_desc}", f"Câu hỏi: {question}", ""]

    if result.empty:
        lines.append("Không có dữ liệu phù hợp với điều kiện phân tích.")
        return "\n".join(lines)

    currency_warning = _build_currency_warning(source_df, plan)
    if currency_warning:
        lines.append(f"⚠️  {currency_warning}")
        lines.append("")

    if plan.get("action") == "compare_metrics" and set(result.columns) >= {"metric", "value"}:
        lines.append("### So sánh chỉ số")
        for index, row in enumerate(result.to_dict(orient="records"), start=1):
            lines.append(f"{index}. {row['metric']}: {_format_cell(row['value'])}")
        if len(result) == 2:
            diff = float(result.iloc[0]["value"]) - float(result.iloc[1]["value"])
            direction = "cao hơn" if diff >= 0 else "thấp hơn"
            lines.append("")
            lines.append(
                f"Chênh lệch: {result.iloc[0]['metric']} {direction} {result.iloc[1]['metric']} "
                f"{_format_cell(abs(diff))}."
            )
        return "\n".join(lines)

    numeric_cols = result.select_dtypes(include="number").columns.tolist()
    if len(result.columns) >= 2 and numeric_cols:
        dim = result.columns[0]
        metric = numeric_cols[0]
        total = sum(float(r[metric]) for r in result.to_dict(orient="records") if r[metric] is not None)
        lines.append("### Xếp hạng / kết quả")
        for index, row in enumerate(result.to_dict(orient="records"), start=1):
            val = row[metric]
            pct = f" ({float(val)/total*100:.1f}%)" if total and val is not None else ""
            lines.append(f"{index}. {row[dim]}: {_format_cell(val)}{pct}")
        if len(result) > 1:
            lines.append(f"\nTổng: {_format_cell(total)}")
        return "\n".join(lines)

    lines.append(_frame_to_markdown(result))
    return "\n".join(lines)


def _build_currency_warning(df: pd.DataFrame | None, plan: dict[str, Any]) -> str | None:
    if df is None:
        return None
    currency_cols = [c for c in df.columns if _normalize(c) in ("currency", "tien_te", "don_vi_tien")]
    if not currency_cols:
        return None
    try:
        work = _apply_filters(df, plan.get("filters", []) or [])
    except Exception:
        work = df
    currencies = work[currency_cols[0]].dropna().unique()
    if len(currencies) > 1:
        return f"Dữ liệu có {len(currencies)} loại tiền tệ ({', '.join(sorted(str(c) for c in currencies))}). Tổng số được tính gộp nhiều currency — cần quy đổi để so sánh chính xác."
    return None


def _frame_to_markdown(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "(no rows)"
    preview = df.head(max_rows).copy()
    columns = [str(col) for col in preview.columns]
    rows = []
    rows.append("| " + " | ".join(columns) + " |")
    rows.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for record in preview.to_dict(orient="records"):
        values = [_format_cell(record.get(col)) for col in preview.columns]
        rows.append("| " + " | ".join(values) + " |")
    if len(df) > max_rows:
        rows.append(f"\n... còn {len(df) - max_rows} dòng")
    return "\n".join(rows)


def _format_cell(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        if value.is_integer():
            return f"{int(value):,}"
        return f"{value:,.4f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value).replace("|", "\\|")


def _describe_plan(plan: dict[str, Any]) -> str:
    compact = {k: v for k, v in plan.items() if not k.startswith("_")}
    return json.dumps(compact, ensure_ascii=False)


def _extract_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"LLM did not return JSON: {raw[:200]}")
    return json.loads(text[start : end + 1])


def _aggregate_series(series: pd.Series, aggregation: str) -> Any:
    if aggregation == "count":
        return int(series.count())
    if aggregation == "nunique":
        return int(series.nunique(dropna=True))
    numeric = _numeric(series)
    if aggregation == "sum":
        return _clean_number(numeric.sum(skipna=True))
    if aggregation == "mean":
        return _clean_number(numeric.mean(skipna=True))
    if aggregation == "median":
        return _clean_number(numeric.median(skipna=True))
    if aggregation == "min":
        return _clean_number(numeric.min(skipna=True))
    if aggregation == "max":
        return _clean_number(numeric.max(skipna=True))
    raise ValueError(f"Unsupported aggregation: {aggregation}")


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _clean_number(value: Any) -> Any:
    if pd.isna(value):
        return None
    value = float(value)
    if math.isfinite(value) and value.is_integer():
        return int(value)
    return round(value, 4)


def _safe_column(label: str) -> str:
    return re.sub(r"\W+", "_", label.lower()).strip("_") or "value"


def _require_column(known: set[str], col: Any) -> None:
    if not isinstance(col, str) or col not in known:
        raise ValueError(f"Unknown column: {col}")


def _find_matching_result_column(result: pd.DataFrame, requested: str | None) -> str | None:
    if not requested:
        return None
    normalized = _normalize(requested)
    for col in result.columns:
        if normalized in _normalize(col) or _normalize(col) in normalized:
            return str(col)
    return None


def _resolve_metric_output_column(plan: dict[str, Any], requested: str | None) -> str | None:
    if not requested:
        return None
    for metric in plan.get("metrics", []) or []:
        if metric.get("column") == requested and metric.get("label"):
            return metric["label"]
    return None


def _pick_metric(df: pd.DataFrame, tokens: tuple[str, ...]) -> str | None:
    normalized_tokens = tuple(_normalize(token) for token in tokens)
    for col in df.select_dtypes(include="number").columns:
        normalized = _normalize(col)
        if any(token in normalized for token in normalized_tokens):
            return str(col)
    return None


def _quarter_filters(normalized_question: str, date_column: str, default_year: int) -> list[dict[str, Any]]:
    year_match = re.search(r"\b(20\d{2})\b", normalized_question)
    year = int(year_match.group(1)) if year_match else default_year
    filters: list[dict[str, Any]] = [
        {"column": date_column, "operator": "between", "value": [f"{year}-01-01", f"{year}-12-31"]}
    ]
    quarters = _mentioned_quarters(normalized_question)
    if quarters:
        filters.append(
            {
                "column": "quarter",
                "operator": "in",
                "value": [f"{year}Q{quarter}" for quarter in sorted(quarters)],
            }
        )
    return filters


def _mentioned_quarters(normalized_question: str) -> set[int]:
    quarters = {int(match) for match in re.findall(r"\bq([1-4])\b", normalized_question)}
    quarters.update(int(match) for match in re.findall(r"\bquy\s*([1-4])\b", normalized_question))
    return quarters


def _normalize(text: Any) -> str:
    import unicodedata
    normalized = unicodedata.normalize("NFKD", str(text).strip().lower())
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", normalized)


def _chart(chart_type: str, title: str, fig: Any, x: str | None = None, y: str | None = None) -> dict[str, Any]:
    fig.update_layout(margin=dict(l=48, r=24, t=64, b=72), height=420)
    return {
        "chart_id": uuid.uuid4().hex,
        "title": title,
        "chart_type": chart_type,
        "x": x,
        "y": y,
        "plotly_json": json.loads(fig.to_json()),
    }


def _extract_top_n(normalized: str) -> int | None:
    """Extract N from 'top N', 'top-5', '5 lớn nhất' etc."""
    m = re.search(r"\btop[\s\-]?(\d+)\b", normalized)
    if m:
        return int(m.group(1))
    m = re.search(r"\b(\d+)\s*(?:lon nhat|nhieu nhat|cao nhat|nho nhat|thap nhat|largest|biggest|smallest)\b", normalized)
    if m:
        return int(m.group(1))
    return None
