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
    answer = _synthesize_answer(question, result, plan)
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
    """Rule-light fallback used only if LLM planning fails."""
    normalized = _normalize(question)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()

    if ("q1" in normalized or "q2" in normalized or "quy" in normalized) and datetime_cols:
        metric = _pick_metric(df, ("revenue", "doanh thu", "sales")) or (numeric_cols[0] if numeric_cols else "")
        return {
            "action": "time_series",
            "derived_columns": [{"name": "quarter", "operation": "quarter", "source": datetime_cols[0]}],
            "filters": _quarter_filters(normalized, datetime_cols[0], 2024),
            "time_column": "quarter",
            "grain": "quarter",
            "metrics": [{"column": metric, "aggregation": "sum", "label": f"Tổng {metric}"}],
            "sort": [{"column": "quarter", "direction": "asc"}],
            "limit": 20,
        }

    if "discount" in normalized and any("quantity" == col.lower() for col in df.columns):
        return {
            "action": "compare_metrics",
            "derived_columns": [
                {"name": "gross_revenue", "operation": "multiply", "columns": ["quantity", "unit_price"]},
                {
                    "name": "net_revenue_after_discount",
                    "operation": "net_revenue_from_discount_pct",
                    "quantity": "quantity",
                    "unit_price": "unit_price",
                    "discount_pct": "discount_pct",
                },
            ],
            "metrics": [
                {"column": "gross_revenue", "aggregation": "sum", "label": "Doanh thu gốc"},
                {"column": "net_revenue_after_discount", "aggregation": "sum", "label": "Doanh thu sau discount"},
            ],
        }

    intent = infer_grouped_metric_intent(question, df)
    if intent:
        return {
            "action": "aggregate",
            "group_by": [intent.dimension],
            "metrics": [{"column": intent.metric, "aggregation": "sum", "label": intent.metric_label}],
            "sort": [{"column": intent.metric, "direction": "desc"}],
            "limit": intent.top_n or 10,
        }

    if numeric_cols:
        return {
            "action": "compare_metrics",
            "metrics": [{"column": col, "aggregation": "sum", "label": col} for col in numeric_cols[:4]],
        }
    return {"action": "profile"}


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
        examples = df[col].dropna().astype(str).head(3).tolist()
        schema_lines.append(f"- {col}: {dtype}; examples={examples}")

    return f"""
Bạn là data analysis planner. Nhiệm vụ của bạn là chuyển câu hỏi tự nhiên thành JSON plan để backend thực thi bằng Pandas.

QUY TẮC BẮT BUỘC:
- Chỉ trả về một JSON object hợp lệ, không markdown, không giải thích.
- Không tự tính số liệu.
- Chỉ dùng cột có trong schema hoặc derived_columns do bạn tạo.
- Không sinh Python code.
- Nếu cần tính toán, dùng action/tool được cho phép.
- Nếu hỏi doanh thu gốc: quantity * unit_price.
- Nếu hỏi doanh thu thực tế/sau discount: ưu tiên cột revenue nếu có; nếu không có thì tạo net_revenue_from_discount_pct từ quantity, unit_price, discount_pct.
- Nếu hỏi Q1/Q2/năm/tháng, dùng cột datetime phù hợp và derived column quarter/month/year.

SCHEMA DATASET:
{chr(10).join(schema_lines)}

JSON PLAN SCHEMA:
{{
  "action": "aggregate | compare_metrics | time_series | profile",
  "derived_columns": [
    {{"name": "quarter", "operation": "quarter", "source": "order_date"}},
    {{"name": "gross_revenue", "operation": "multiply", "columns": ["quantity", "unit_price"]}},
    {{"name": "net_revenue_after_discount", "operation": "net_revenue_from_discount_pct", "quantity": "quantity", "unit_price": "unit_price", "discount_pct": "discount_pct"}}
  ],
  "filters": [
    {{"column": "order_date", "operator": "between", "value": ["2024-01-01", "2024-12-31"]}},
    {{"column": "customer_region", "operator": "eq", "value": "Hà Nội"}}
  ],
  "group_by": ["product_name"],
  "time_column": "order_date",
  "grain": "month | quarter | year | date",
  "metrics": [
    {{"column": "quantity", "aggregation": "sum", "label": "Số lượng"}}
  ],
  "sort": [{{"column": "quantity", "direction": "desc"}}],
  "limit": 10
}}

CÂU HỎI:
{question}
""".strip()


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


def _synthesize_answer(question: str, result: pd.DataFrame, plan: dict[str, Any]) -> str:
    return _deterministic_answer(question, result, plan)


def _deterministic_answer(question: str, result: pd.DataFrame, plan: dict[str, Any]) -> str:
    lines = ["# Executive Data Brief", "", f"## Kết quả phân tích", f"Câu hỏi: {question}", ""]
    if result.empty:
        lines.append("Không có dữ liệu phù hợp với điều kiện phân tích.")
        return "\n".join(lines)
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
        lines.append("### Xếp hạng / kết quả")
        for index, row in enumerate(result.to_dict(orient="records"), start=1):
            lines.append(f"{index}. {row[dim]}: {_format_cell(row[metric])}")
        return "\n".join(lines)

    lines.append(_frame_to_markdown(result))
    return "\n".join(lines)


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
