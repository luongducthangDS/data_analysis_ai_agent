from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from backend.app.services.analysis_planner import execute_plan, build_fallback_plan, _validate_plan_against_dataframe
from backend.app.services.charts import generate_question_charts
from backend.app.services.profiler import build_profile
from backend.app.services.query_engine import run_readonly_query


AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "analyze_data",
            "description": (
                "Chạy phân tích có cấu trúc trên dataset: aggregate (nhóm + tính tổng/trung bình...), "
                "time_series (xu hướng theo thời gian), compare_metrics (so sánh nhiều chỉ số), "
                "hoặc profile (tổng quan dataset). Dùng khi cần kết quả số liệu cụ thể."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["aggregate", "time_series", "compare_metrics", "profile"],
                        "description": "Loại phân tích cần thực hiện",
                    },
                    "group_by": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Danh sách cột để group by (dùng cho aggregate)",
                    },
                    "metrics": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "column": {"type": "string"},
                                "aggregation": {
                                    "type": "string",
                                    "enum": ["sum", "mean", "median", "min", "max", "count", "nunique"],
                                },
                                "label": {"type": "string"},
                            },
                            "required": ["column", "aggregation"],
                        },
                    },
                    "filters": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "column": {"type": "string"},
                                "operator": {
                                    "type": "string",
                                    "enum": ["eq", "ne", "gt", "gte", "lt", "lte", "between", "in", "contains"],
                                },
                                "value": {},
                            },
                            "required": ["column", "operator", "value"],
                        },
                    },
                    "time_column": {
                        "type": "string",
                        "description": "Cột thời gian (dùng cho time_series)",
                    },
                    "grain": {
                        "type": "string",
                        "enum": ["month", "quarter", "year", "date"],
                        "description": "Độ phân giải thời gian (dùng cho time_series)",
                    },
                    "sort": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "column": {"type": "string"},
                                "direction": {"type": "string", "enum": ["asc", "desc"]},
                            },
                        },
                    },
                    "limit": {"type": "integer", "description": "Số dòng kết quả tối đa (mặc định 20)"},
                },
                "required": ["action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_sql",
            "description": (
                "Chạy câu SQL SELECT trực tiếp trên dataset (tên bảng là 'dataset'). "
                "Dùng khi cần query linh hoạt hơn analyze_data không hỗ trợ."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "Câu SQL SELECT. Chỉ SELECT, không INSERT/UPDATE/DELETE.",
                    }
                },
                "required": ["sql"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_profile",
            "description": (
                "Lấy thống kê tổng quan dataset: số dòng/cột, kiểu dữ liệu từng cột, "
                "missing values, top values của categorical columns, min/max/mean của numeric columns. "
                "Dùng khi cần hiểu cấu trúc dữ liệu trước khi phân tích."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_chart",
            "description": "Tạo biểu đồ Plotly phù hợp với câu hỏi. Dùng sau khi đã có kết quả số liệu.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "Câu hỏi hoặc mô tả biểu đồ cần tạo",
                    }
                },
                "required": ["question"],
            },
        },
    },
]


@dataclass
class ToolResult:
    summary: str
    data_preview: str = ""
    charts: list[dict[str, Any]] = field(default_factory=list)


def execute_tool(df: pd.DataFrame, profile: dict[str, Any], tool_name: str, arguments: dict[str, Any]) -> ToolResult:
    if tool_name == "analyze_data":
        return _run_analyze(df, arguments)
    if tool_name == "query_sql":
        return _run_sql(df, arguments.get("sql", ""))
    if tool_name == "get_profile":
        return _run_profile(df, profile)
    if tool_name == "generate_chart":
        return _run_chart(df, arguments.get("question", ""))
    return ToolResult(summary=f"Unknown tool: {tool_name}")


def _run_analyze(df: pd.DataFrame, plan: dict[str, Any]) -> ToolResult:
    try:
        plan.setdefault("limit", 20)
        _validate_plan_against_dataframe(df, plan)
        result = execute_plan(df, plan)
    except Exception as exc:
        try:
            fallback = build_fallback_plan(df, str(plan.get("action", "")))
            result = execute_plan(df, fallback)
        except Exception:
            return ToolResult(summary=f"Lỗi phân tích: {exc}")

    if result.empty:
        return ToolResult(summary="Không có dữ liệu phù hợp.")

    summary_lines = [f"Kết quả analyze_data ({plan.get('action')}): {len(result)} dòng"]
    numeric_cols = result.select_dtypes(include="number").columns.tolist()
    if numeric_cols and len(result.columns) >= 2:
        dim = result.columns[0]
        metric = numeric_cols[0]
        total = result[metric].sum()
        top = result.iloc[0]
        summary_lines.append(f"Tổng {metric}: {total:,.2f} | Top: {top[dim]} = {top[metric]:,.2f}")

    preview = _df_to_text(result.head(10))
    return ToolResult(summary="\n".join(summary_lines), data_preview=preview)


def _run_sql(df: pd.DataFrame, sql: str) -> ToolResult:
    try:
        rows = run_readonly_query(df, sql, limit=50)
    except Exception as exc:
        return ToolResult(summary=f"SQL error: {exc}")

    if not rows:
        return ToolResult(summary="SQL trả về 0 dòng.")

    summary = f"SQL trả về {len(rows)} dòng."
    if len(rows) == 1:
        summary += " " + " | ".join(f"{k}={v}" for k, v in rows[0].items())

    preview = json.dumps(rows[:5], ensure_ascii=False, default=str)
    return ToolResult(summary=summary, data_preview=preview)


def _run_profile(df: pd.DataFrame, profile: dict[str, Any]) -> ToolResult:
    p = profile or build_profile(df)
    lines = [
        f"Dataset: {p.get('rows')} dòng × {p.get('columns')} cột",
        f"Numeric columns: {list(p.get('numeric_summary', {}).keys())}",
        f"Categorical columns: {list(p.get('categorical_summary', {}).keys())}",
        f"Missing values: { {k: v for k, v in p.get('missing_values', {}).items() if v > 0} }",
    ]
    for col, stats in list(p.get("numeric_summary", {}).items())[:3]:
        lines.append(f"  {col}: min={stats.get('min')}, max={stats.get('max')}, mean={stats.get('mean'):.2f}" if stats.get("mean") else f"  {col}: {stats}")
    for col, top_vals in list(p.get("categorical_summary", {}).items())[:3]:
        top = [f"{v['value']}({v['count']})" for v in top_vals[:3]]
        lines.append(f"  {col}: {', '.join(top)}")
    return ToolResult(summary="\n".join(lines))


def _run_chart(df: pd.DataFrame, question: str) -> ToolResult:
    try:
        charts = generate_question_charts(df, question)
    except Exception as exc:
        return ToolResult(summary=f"Lỗi tạo chart: {exc}")
    if not charts:
        return ToolResult(summary="Không tạo được biểu đồ phù hợp.")
    return ToolResult(
        summary=f"Đã tạo {len(charts)} biểu đồ: {', '.join(c.get('title', '') for c in charts)}",
        charts=charts,
    )


def _df_to_text(df: pd.DataFrame) -> str:
    if df.empty:
        return "(empty)"
    cols = list(df.columns)
    lines = [" | ".join(cols)]
    for _, row in df.iterrows():
        lines.append(" | ".join(str(row[c]) for c in cols))
    return "\n".join(lines)
