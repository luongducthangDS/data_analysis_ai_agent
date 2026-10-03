"""Turn an executed plan's result into charts and a deterministic (no-LLM) answer."""
from __future__ import annotations

import json
import re
import uuid
from typing import Any

import pandas as pd
import plotly.express as px

from backend.app.services.ecommerce_semantic import (
    bridge_actions, fmt_num, fmt_pct,
)
from backend.app.services.planner.execute import _apply_filters, _describe_numeric, _metric_label, _normalize


def _build_charts_from_result(result: pd.DataFrame, plan: dict[str, Any]) -> list[dict[str, Any]]:
    action = plan.get("action")
    if result is None or result.empty:
        return []

    # Profile = metadata table, never charted.
    if action == "profile":
        return []

    # Distribution → histogram (bar over ordered bins).
    if action == "distribution":
        x = result.columns[0]
        y = result.columns[-1]
        col_name = plan.get("column", "")
        title = f"Phân phối {col_name}".strip()
        fig = px.bar(result, x=x, y=y, title=title)
        fig.update_layout(bargap=0.02)  # adjacent bars → histogram look
        return [_chart("histogram", title, fig, x=str(x), y=str(y))]

    if action == "profit_bridge":
        # 4 hạng mục + dòng Lãi; tỷ lệ và nhóm driver đã nằm trong bảng/câu trả lời.
        impact = result.head(5)
        fig = px.bar(impact, x="hang_muc", y="anh_huong_lai", title="Ảnh hưởng tới lãi", text="anh_huong_lai")
        fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
        return [_chart("bar", "Ảnh hưởng tới lãi", fig, x="hang_muc", y="anh_huong_lai")]

    # Meaningfulness gate: a single value / single row produces a 1-bar chart
    # that adds nothing. Only chart when there are ≥2 groups to compare.
    if len(result) < 2 or len(result.columns) < 2:
        return []

    x = result.columns[0]
    numeric_cols = result.select_dtypes(include="number").columns.tolist()
    y = numeric_cols[0] if numeric_cols else result.columns[1]
    if action == "time_series":
        fig = px.line(result, x=x, y=y, title=f"{y} theo {x}", markers=True)
        return [_chart("line", f"{y} theo {x}", fig, x=x, y=y)]
    if action in {"aggregate", "compare_metrics"}:
        fig = px.bar(result, x=x, y=y, title=f"{y} theo {x}", text=y)
        fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
        return [_chart("bar", f"{y} theo {x}", fig, x=x, y=y)]
    return []


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

    if plan.get("action") == "distribution":
        col = plan.get("column", "")
        stats = _describe_numeric(source_df, col) if source_df is not None else {}
        if stats:
            lines.append(f"### Phân phối **{col}**")
            lines.append(
                f"- Khoảng giá trị: **{_format_cell(stats['min'])} – {_format_cell(stats['max'])}** "
                f"(biên độ {_format_cell(stats['range'])})"
            )
            lines.append(
                f"- Trung bình: {_format_cell(stats['mean'])} · "
                f"Trung vị: {_format_cell(stats['median'])} · "
                f"Độ lệch chuẩn: {_format_cell(stats['std'])}"
            )
            lines.append(f"- Tứ phân vị: Q1 = {_format_cell(stats['q1'])}, Q3 = {_format_cell(stats['q3'])}")
            lines.append("")
        # Top bins by count
        if "so_luong" in result.columns:
            top = result.sort_values("so_luong", ascending=False).head(3)
            lines.append("### Khoảng tập trung nhiều nhất")
            for index, row in enumerate(top.to_dict(orient="records"), start=1):
                lines.append(f"{index}. {row['khoang_gia_tri']}: {_format_cell(row['so_luong'])} bản ghi")
        return "\n".join(lines)

    if plan.get("action") == "profit_bridge":
        return "\n".join(lines + _bridge_brief(result))

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

    rates = _rate_columns(plan)
    numeric_cols = result.select_dtypes(include="number").columns.tolist()
    if len(numeric_cols) > 1:
        # Nhiều chỉ số (vd doanh thu + lãi): xếp hạng theo cột đầu sẽ giấu mất cột lãi → in cả bảng.
        lines.append(_frame_to_markdown(result, rates=rates))
        return "\n".join(lines + _ratio_comparison(result, plan))
    if len(result.columns) >= 2 and numeric_cols:
        dim = result.columns[0]
        metric = numeric_cols[0]
        is_rate = metric in rates  # tổng/tỷ trọng của các tỷ lệ là số vô nghĩa
        # Tỷ trọng tháng/tổng in cạnh lãi ("34.002.675 (20,5%)") đọc như biên lãi.
        share = not is_rate and plan.get("action") != "time_series" and len(result) > 1  # 1 dòng: "(100,0%)"
        total = sum(float(r[metric]) for r in result.to_dict(orient="records") if r[metric] is not None)
        lines.append("### Xếp hạng / kết quả")
        for index, row in enumerate(result.to_dict(orient="records"), start=1):
            val = row[metric]
            pct = f" ({fmt_pct(float(val) / total)})" if total and val is not None and share else ""
            lines.append(f"{index}. {row[dim]}: {_format_value(val, is_rate)}{pct}")
        if len(result) > 1 and not is_rate:
            lines.append(f"\nTổng: {_format_cell(total)}")
            if avg := period_average(question, result, plan):
                lines.append(avg[0])
        return "\n".join(lines)

    lines.append(_frame_to_markdown(result, rates=rates))
    return "\n".join(lines)


def _ratio_comparison(result: pd.DataFrame, plan: dict[str, Any]) -> list[str]:
    """2 nhóm so một tỷ lệ ("sàn nào phí cao hơn"): nói thẳng bên nào cao hơn bao nhiêu điểm %.
    34,52% và 34,50% làm tròn ra "34,5% | 34,5%" trông như bằng nhau."""
    ratios = _percent_ratios(plan)
    dims = [d for d in plan.get("group_by") or [] if d in result.columns]
    if len(result) != 2 or not ratios or not dims or ratios[0] not in result.columns:
        return []
    label = ratios[0]
    hi, lo = result.sort_values(label, ascending=False).to_dict(orient="records")
    if pd.isna(hi[label]) or pd.isna(lo[label]):
        return []
    top, other = (" / ".join(str(row[d]) for d in dims) for row in (hi, lo))
    gap = (float(hi[label]) - float(lo[label])) * 100
    if round(gap, 2) == 0:
        return ["", f"{label} của {top} và {other} bằng nhau ({fmt_pct(float(hi[label]))})."]
    near = "gần như bằng nhau, " if gap < 0.1 else ""
    return ["", f"{label}: {near}{top} cao hơn {other} {fmt_num(gap, 2)} điểm %."]


def period_average(question: str, result: pd.DataFrame, plan: dict[str, Any]) -> tuple[str, float] | None:
    """("Trung bình mỗi tháng: X (N tháng)", X) cho câu hỏi trung bình trên chuỗi thời gian một chỉ số.

    Chỉ bảng các kỳ thì chưa trả lời: LLM bị cấm tự tính số mới, câu tất định từng chỉ in bảng. Tính ở đây để
    câu tất định in ra và synthesize đưa vào prompt + tập số được phép."""
    if plan.get("action") != "time_series" or not re.search(r"\b(?:trung binh|binh quan)\b", _normalize(question)):
        return None
    numeric = result.select_dtypes(include="number").columns.tolist()
    if len(numeric) != 1 or numeric[0] in _rate_columns(plan):  # trung bình các tỷ lệ là số vô nghĩa
        return None
    values = result[numeric[0]].dropna()
    # ponytail: đủ `limit` dòng thì có thể đã bị cắt → không tính (trung bình 12 tháng đầu ≠ trung bình cả kỳ).
    if len(values) < 2 or len(result) >= int(plan.get("limit") or 10**6):
        return None
    unit = {"month": "tháng", "quarter": "quý"}.get(plan.get("grain"), "kỳ")
    value = float(values.sum()) / len(values)
    return f"Trung bình mỗi {unit}: {fmt_num(value)} ({len(values)} {unit})", value


def _percent_ratios(plan: dict[str, Any]) -> list[str]:
    """Label các ratio là tỷ lệ; `"percent": false` là số tiền (doanh thu / số đơn)."""
    return [r.get("label") for r in plan.get("ratios") or [] if isinstance(r, dict) and r.get("percent", True)]


def _rate_columns(plan: dict[str, Any]) -> set[str]:
    """Cột kết quả là tỷ lệ 0–1: ratios của plan và mean(la_don_hoan). In dạng 33,5%, không phải 0,3350."""
    cols = set(_percent_ratios(plan))
    cols |= {_metric_label(m) for m in plan.get("metrics") or []
             if isinstance(m, dict) and m.get("column") == "la_don_hoan" and m.get("aggregation") == "mean"}
    return cols


def _format_value(value: Any, is_rate: bool) -> str:
    if is_rate and not pd.isna(value):
        return fmt_pct(float(value))
    return _format_cell(value)


def _bridge_brief(result: pd.DataFrame) -> list[str]:
    p0, p1 = result.columns[1], result.columns[2]
    by_name = result.set_index("hang_muc")
    before, after, delta = by_name.loc["Lãi trước QC", [p0, p1, "anh_huong_lai"]]
    lines = [f"### Lãi trước QC {'giảm' if delta < 0 else 'tăng'} {_format_cell(abs(delta))} "
             f"({p0}: {_format_cell(before)} → {p1}: {_format_cell(after)})", "", "Ảnh hưởng theo hạng mục:"]
    for name in ("Doanh thu thuần", "Phí sàn", "Giá vốn", "Chi phí hoàn hàng"):
        lines.append(f"- {name}: {_format_cell(by_name.loc[name, 'anh_huong_lai'])}")
    drivers = result[result["hang_muc"].str.contains(":", regex=False)]
    if not drivers.empty:
        lines += ["", "Nhóm kéo lãi đi nhiều nhất:"]
        lines += [f"- {r.hang_muc}: {_format_cell(r.anh_huong_lai)}" for r in drivers.itertuples()]
    lines += ["", "### Nên kiểm tra"] + [f"- {a}" for a in bridge_actions(result)]
    return lines


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


def _frame_to_markdown(df: pd.DataFrame, max_rows: int = 20, rates: set[str] = frozenset()) -> str:
    if df.empty:
        return "(no rows)"
    preview = df.head(max_rows).copy()
    columns = [str(col) for col in preview.columns]
    rows = []
    rows.append("| " + " | ".join(columns) + " |")
    rows.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for record in preview.to_dict(orient="records"):
        values = [_format_value(record.get(col), col in rates) for col in preview.columns]
        rows.append("| " + " | ".join(values) + " |")
    if len(df) > max_rows:
        rows.append(f"\n... còn {len(df) - max_rows} dòng")
    return "\n".join(rows)


def _format_cell(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        if value.is_integer():
            return fmt_num(value)
        return fmt_num(value, 4)
    if isinstance(value, int):
        return fmt_num(value)
    return str(value).replace("|", "\\|")


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
