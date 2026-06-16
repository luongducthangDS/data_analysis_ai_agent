"""
AI-driven dashboard — LLM understands the dataset, decides what KPIs and charts
to show. Works for any domain: e-commerce, stock market, manufacturing, HR, etc.

Flow:
  GET /api/dashboard/{session_id}
  → LLM analyzes dataset profile → outputs JSON spec {domain, kpi_specs, chart_specs}
  → backend executes each spec item via execute_plan()
  → returns generic KPICard + ChartSpec list
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any

import pandas as pd
import plotly.express as px
from fastapi import APIRouter, Depends, HTTPException

from backend.app.api.deps import get_current_user, get_session
from backend.app.schemas import ChartSpec, DashboardResponse, KPICard
from backend.app.services.analysis_planner import execute_plan
from backend.app.services.storage import DatasetSession

_log = logging.getLogger(__name__)
router = APIRouter()

# ─────────────────────────────────────────────────────────────────────────────
# LLM prompt
# ─────────────────────────────────────────────────────────────────────────────

_SPEC_PROMPT = """\
Bạn là data analyst expert. Nhiệm vụ: phân tích dataset bên dưới và output JSON spec \
cho dashboard tự động. Output JSON thuần túy — không markdown, không giải thích.

DATASET: {filename}
SCHEMA ({rows:,} dòng × {cols} cột):
{schema}

THỐNG KÊ SỐ:
{num_stats}

THỐNG KÊ DANH MỤC:
{cat_stats}

{ecom_hint}\
Hãy output JSON theo format sau (không thêm key nào khác):
{{
  "domain": "<tên domain tiếng Việt ngắn gọn, ví dụ: E-Commerce, Chứng khoán, Sản xuất, HR, Tài chính>",
  "kpi_specs": [
    {{
      "label": "<tên KPI tiếng Việt>",
      "column": "<tên cột CHÍNH XÁC từ SCHEMA>",
      "aggregation": "<sum|mean|count|max|min|nunique>",
      "filters": [],
      "format": "<number|currency_vnd|percent|integer>",
      "formula": "<mô tả công thức ngắn>"
    }}
  ],
  "chart_specs": [
    {{
      "title": "<tiêu đề biểu đồ tiếng Việt>",
      "chart_type": "<line|bar>",
      "x_col": "<tên cột trục X — CHÍNH XÁC từ SCHEMA>",
      "y_col": "<tên cột trục Y số — CHÍNH XÁC>",
      "aggregation": "<sum|mean|count>",
      "is_time_series": <true nếu x_col là datetime, false nếu không>
    }}
  ],
  "top_dimension": "<tên cột categorical dùng làm dimension chính — hoặc null>",
  "suggested_queries": ["<câu hỏi phân tích tiếng Việt phù hợp với dataset>", ...]
}}

QUY TẮC:
- kpi_specs: 3-5 KPI, chọn những metric quan trọng nhất cho domain này
- chart_specs: 2-3 chart, ưu tiên 1 line chart theo thời gian (nếu có cột date) + 1 bar chart top dimension
- column phải là tên cột CHÍNH XÁC trong SCHEMA, không tự đặt tên mới
- format currency_vnd nếu cột có đơn vị tiền (VND, đồng, ₫, revenue, amount, price)
- suggested_queries: 3 câu hỏi cụ thể dựa trên TÊN CỘT THỰC TẾ
- Nếu không có cột datetime, không tạo line chart
- is_time_series: true chỉ khi x_col là cột datetime thực sự
"""


def _build_llm_prompt(
    df: pd.DataFrame,
    profile: dict[str, Any],
    filename: str,
    ecom_col_map: dict[str, str],
) -> str:
    col_types = profile.get("column_types", {})
    num_summary = profile.get("numeric_summary", {})
    cat_summary = profile.get("categorical_summary", {})

    schema_lines: list[str] = []
    for col, dtype in col_types.items():
        samples = df[col].dropna().astype(str).head(4).tolist()
        schema_lines.append(f"  - {col} ({dtype}): {samples}")

    num_lines: list[str] = []
    for col, stats in list(num_summary.items())[:8]:
        mn, mx, mean = stats.get("min"), stats.get("max"), stats.get("mean")
        if mn is not None:
            num_lines.append(f"  - {col}: min={mn:,.1f}, max={mx:,.1f}, mean={mean:,.1f}")

    cat_lines: list[str] = []
    for col, vals in list(cat_summary.items())[:6]:
        top = ", ".join(str(v["value"]) for v in vals[:4])
        cat_lines.append(f"  - {col}: [{top}]")

    ecom_hint = ""
    if ecom_col_map:
        mapping_str = ", ".join(f"{k}→'{v}'" for k, v in list(ecom_col_map.items())[:6])
        ecom_hint = f"GỢI Ý E-COMMERCE: các cột đặc thù đã nhận diện: {mapping_str}\n\n"

    return _SPEC_PROMPT.format(
        filename=filename,
        rows=profile.get("rows", len(df)),
        cols=profile.get("columns", len(df.columns)),
        schema="\n".join(schema_lines) or "  (không có thông tin)",
        num_stats="\n".join(num_lines) or "  (không có cột số)",
        cat_stats="\n".join(cat_lines) or "  (không có cột danh mục)",
        ecom_hint=ecom_hint,
    )


def _call_llm_for_spec(prompt: str) -> dict:
    from backend.app.services.llm_service import get_llm_client
    client = get_llm_client()
    raw = client.generate(prompt, max_tokens=700, temperature=0.1)
    # Strip markdown fences if present
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        text = text.rsplit("```", 1)[0]
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"LLM did not return JSON: {raw[:200]}")
    return json.loads(text[start: end + 1])


# ─────────────────────────────────────────────────────────────────────────────
# Execution helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_value(value: float | None, fmt: str) -> str:
    if value is None:
        return "—"
    if fmt == "currency_vnd":
        if abs(value) >= 1_000_000_000:
            return f"{value / 1_000_000_000:,.1f} tỷ ₫"
        if abs(value) >= 1_000_000:
            return f"{value / 1_000_000:,.1f}M ₫"
        return f"{value:,.0f} ₫"
    if fmt == "percent":
        return f"{value * 100:.1f}%"
    if fmt == "integer":
        return f"{int(value):,}"
    # number (default)
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:,.2f}M"
    if abs(value) >= 1_000:
        return f"{value:,.1f}"
    return f"{value:.2f}"


def _execute_kpi_spec(df: pd.DataFrame, kpi: dict) -> KPICard | None:
    col = kpi.get("column", "")
    agg = kpi.get("aggregation", "sum")
    fmt = kpi.get("format", "number")
    label = kpi.get("label", col)
    formula = kpi.get("formula", f"{agg}([{col}])")
    filters = kpi.get("filters", [])

    if col not in df.columns:
        _log.warning("dashboard kpi: column %r not in df", col)
        return None

    try:
        plan: dict[str, Any] = {
            "action": "compare_metrics",
            "filters": filters,
            "metrics": [{"column": col, "aggregation": agg, "label": label}],
        }
        result = execute_plan(df, plan)
        if result.empty:
            return None

        # compare_metrics returns a "metric"/"value" table
        val_col = "value" if "value" in result.columns else result.columns[-1]
        raw_val = result[val_col].iloc[0]
        value_str = _fmt_value(float(raw_val), fmt) if pd.notna(raw_val) else "—"
    except Exception as exc:
        _log.warning("dashboard kpi: exec failed col=%r agg=%r: %s", col, agg, exc)
        return None

    return KPICard(label=label, value=value_str, formula=formula)


def _execute_chart_spec(df: pd.DataFrame, chart: dict) -> dict | None:
    title = chart.get("title", "")
    chart_type = chart.get("chart_type", "bar")
    x_col = chart.get("x_col", "")
    y_col = chart.get("y_col", "")
    agg = chart.get("aggregation", "sum")
    is_time_series = chart.get("is_time_series", False)

    if x_col not in df.columns or y_col not in df.columns:
        _log.warning("dashboard chart: columns %r/%r not in df", x_col, y_col)
        return None

    try:
        if is_time_series:
            # Choose daily grain for ≤180 days of data, monthly for longer spans
            try:
                dt_range = pd.to_datetime(df[x_col], errors="coerce")
                span_days = (dt_range.max() - dt_range.min()).days
                grain = "date" if span_days <= 180 else "month"
            except Exception:
                grain = "month"
            plan: dict[str, Any] = {
                "action": "time_series",
                "time_column": x_col,
                "grain": grain,
                "metrics": [{"column": y_col, "aggregation": agg, "label": y_col}],
            }
        else:
            plan = {
                "action": "aggregate",
                "group_by": [x_col],
                "metrics": [{"column": y_col, "aggregation": agg, "label": y_col}],
                "sort": [{"column": y_col, "direction": "desc"}],
                "limit": 15,
            }

        result = execute_plan(df, plan)
        if result.empty:
            return None

        x_data = result.columns[0]
        y_data = result.columns[-1] if result.shape[1] > 1 else result.columns[0]

        if chart_type == "line" or is_time_series:
            fig = px.line(result, x=x_data, y=y_data, title=title, markers=True)
        else:
            top_n = result.head(12)
            fig = px.bar(
                top_n.sort_values(y_data, ascending=True),
                x=y_data, y=x_data, orientation="h",
                title=title, text=y_data,
            )
            fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside")

        fig.update_layout(margin=dict(l=48, r=24, t=64, b=64), height=380)
        return {
            "chart_id": uuid.uuid4().hex,
            "title": title,
            "chart_type": "line" if (chart_type == "line" or is_time_series) else "bar",
            "x": str(x_data),
            "y": str(y_data),
            "plotly_json": json.loads(fig.to_json()),
        }
    except Exception as exc:
        _log.warning("dashboard chart: exec failed %r: %s", title, exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic fallback — no LLM needed
# ─────────────────────────────────────────────────────────────────────────────

def _fallback_spec(df: pd.DataFrame, profile: dict[str, Any]) -> dict:
    """Build a minimal dashboard spec without LLM when it fails."""
    num_cols = list(profile.get("numeric_summary", {}).keys())[:4]
    cat_cols = list(profile.get("categorical_summary", {}).keys())[:1]
    col_types = profile.get("column_types", {})
    date_cols = [c for c, t in col_types.items() if "datetime" in str(t)]

    kpi_specs = []
    for col in num_cols[:4]:
        kpi_specs.append({
            "label": col,
            "column": col,
            "aggregation": "sum",
            "filters": [],
            "format": "number",
            "formula": f"Tổng [{col}]",
        })

    chart_specs = []
    if date_cols and num_cols:
        chart_specs.append({
            "title": f"{num_cols[0]} theo thời gian",
            "chart_type": "line",
            "x_col": date_cols[0],
            "y_col": num_cols[0],
            "aggregation": "sum",
            "is_time_series": True,
        })
    if cat_cols and num_cols:
        chart_specs.append({
            "title": f"Top {cat_cols[0]} theo {num_cols[0]}",
            "chart_type": "bar",
            "x_col": cat_cols[0],
            "y_col": num_cols[0],
            "aggregation": "sum",
            "is_time_series": False,
        })

    return {
        "domain": "Dữ liệu tổng hợp",
        "kpi_specs": kpi_specs,
        "chart_specs": chart_specs,
        "top_dimension": cat_cols[0] if cat_cols else None,
        "suggested_queries": [
            f"Top 10 giá trị {cat_cols[0]} theo {num_cols[0]}?" if cat_cols and num_cols else "Tổng quan dataset?",
            f"Xu hướng {num_cols[0]} theo thời gian?" if date_cols and num_cols else "So sánh các cột số?",
            "Phân phối dữ liệu như thế nào?",
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Endpoint
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/api/dashboard/{session_id}", response_model=DashboardResponse)
def get_dashboard(
    session_id: str,
    _session: DatasetSession = Depends(get_session),
    _user: dict = Depends(get_current_user),
) -> DashboardResponse:
    """
    AI-driven dashboard for any dataset domain.

    LLM analyzes dataset profile → decides what KPIs + charts to show.
    Result is cached on the session object for subsequent calls.
    """
    # ── Cache hit ─────────────────────────────────────────────────────────────
    cached = getattr(_session, "_dashboard_cache", None)
    if cached is not None:
        _log.info("dashboard: cache hit session=%s", session_id)
        return cached

    df = _session.dataframe
    profile = _session.profile or {}

    if df.empty or not profile:
        return DashboardResponse(session_id=session_id, is_ecommerce=False)

    filename = ", ".join(_session.file_names) if _session.file_names else _session.filename
    col_map = _session.ecommerce_col_map or {}

    # ── Generate spec via LLM ─────────────────────────────────────────────────
    spec: dict = {}
    t0 = time.perf_counter()
    try:
        prompt = _build_llm_prompt(df, profile, filename, col_map)
        spec = _call_llm_for_spec(prompt)
        _log.info(
            "dashboard: LLM spec done in %.2fs domain=%r session=%s",
            time.perf_counter() - t0, spec.get("domain"), session_id,
        )
    except Exception as exc:
        _log.warning("dashboard: LLM spec failed (%.2fs): %s — using fallback", time.perf_counter() - t0, exc)
        spec = _fallback_spec(df, profile)

    domain = spec.get("domain", "")

    # ── Execute KPI specs ─────────────────────────────────────────────────────
    kpi_cards: list[KPICard] = []
    for kpi in spec.get("kpi_specs", []):
        card = _execute_kpi_spec(df, kpi)
        if card:
            kpi_cards.append(card)

    # ── Execute chart specs ───────────────────────────────────────────────────
    charts_raw: list[dict] = []
    for chart in spec.get("chart_specs", []):
        c = _execute_chart_spec(df, chart)
        if c:
            charts_raw.append(c)

    charts = [ChartSpec(**c) for c in charts_raw]

    # ── Build top_products-equivalent table (generic top dimension breakdown) ─
    top_rows: list[dict] = []
    top_dim = spec.get("top_dimension")
    num_cols = list(profile.get("numeric_summary", {}).keys())
    if top_dim and top_dim in df.columns and num_cols:
        metric_col = num_cols[0]
        try:
            grp = (
                df.groupby(top_dim, dropna=False)[metric_col]
                .sum()
                .reset_index()
                .sort_values(metric_col, ascending=False)
                .head(10)
            )
            grp["rank"] = range(1, len(grp) + 1)
            top_rows = grp.rename(columns={top_dim: "name", metric_col: "value"}).to_dict("records")
        except Exception:
            pass

    suggested = spec.get("suggested_queries", [])

    # ── Cache result ──────────────────────────────────────────────────────────
    response = DashboardResponse(
        session_id=session_id,
        platform=_session.detected_platform or domain,
        kpi_cards=kpi_cards,
        charts=charts,
        top_products=top_rows,
        col_map=col_map,
        unmapped_cols=[],
        is_ecommerce=True,  # always show dashboard for any dataset
        suggested_queries=suggested,
    )
    _session._dashboard_cache = response        # type: ignore[attr-defined]
    _session._dashboard_charts_raw = charts_raw  # type: ignore[attr-defined]

    _log.info(
        "dashboard: built domain=%r kpis=%d charts=%d session=%s",
        domain, len(kpi_cards), len(charts), session_id,
    )
    return response
