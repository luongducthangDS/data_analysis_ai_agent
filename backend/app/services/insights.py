from __future__ import annotations

import os
from typing import Any

import pandas as pd

from backend.app.services.analysis_intent import (
    build_grouped_metric_frame,
    infer_grouped_metric_intent,
)


def generate_insights(
    df: pd.DataFrame,
    profile: dict[str, Any],
    question: str | None = None,
    sheets: dict[str, pd.DataFrame] | None = None,
    sheets_context: str | None = None,
) -> str:
    """Generate an executive-ready analysis, always attempting LLM synthesis."""
    import logging
    _log = logging.getLogger(__name__)

    # Compute deterministic facts first — used as LLM context
    focused = _generate_focused_grouped_metric_answer(df, question)
    deterministic = _generate_template_insights(df, profile, question, sheets=sheets)
    grounding = focused or deterministic

    try:
        from backend.app.services.llm_service import get_llm_client, get_active_provider
        client = get_llm_client()
        _log.info("generate_insights: calling LLM (%s) for question=%r", get_active_provider(), question)
        context = _build_context_for_llm(df, profile, question, sheets=sheets,
                                         sheets_context=sheets_context, grounding=grounding)
        if question:
            return client.answer_question(question, context, max_tokens=700)
        return client.generate_insights(context, max_tokens=900)
    except Exception as exc:
        _log.warning("LLM inference failed (%s), using deterministic answer.", exc)
        return grounding


def _generate_focused_grouped_metric_answer(df: pd.DataFrame, question: str | None) -> str | None:
    intent = infer_grouped_metric_intent(question, df)
    if not intent:
        return None

    grouped = build_grouped_metric_frame(df, intent)
    if grouped.empty:
        return f"Không có dữ liệu phù hợp để tính {intent.metric_label} theo {intent.dimension_label}."

    total = float(grouped[intent.metric].sum())
    top = grouped.iloc[0]
    top_value = float(top[intent.metric])
    top_share = top_value / total * 100 if total else 0
    top_n = intent.top_n or 10
    heading = (
        f"Top {top_n} {intent.dimension_label} theo {intent.metric_label}"
        if intent.top_n
        else f"{intent.metric_label.capitalize()} theo {intent.dimension_label}"
    )

    lines = [
        "# Executive Data Brief",
        "",
        f"## {heading}",
        f"Tổng {intent.metric_label}: {_format_number(total)}.",
        (
            f"{intent.dimension_label.capitalize()} dẫn đầu là {top[intent.dimension]} "
            f"với {_format_number(top_value)}, chiếm {top_share:.1f}% tổng {intent.metric_label}."
        ),
        "",
        "### Xếp hạng đóng góp",
    ]

    for index, row in enumerate(grouped.head(top_n).to_dict(orient="records"), start=1):
        dimension_value = row[intent.dimension]
        metric_value = float(row[intent.metric])
        share = metric_value / total * 100 if total else 0
        lines.append(f"{index}. {dimension_value}: {_format_number(metric_value)} ({share:.1f}%)")

    if len(grouped) > 1:
        top_3_share = float(grouped.head(3)[intent.metric].sum()) / total * 100 if total else 0
        lines.extend(
            [
                "",
                "### Nhận định",
                f"Top 3 nhóm đóng góp {top_3_share:.1f}% tổng {intent.metric_label}.",
                "Kết quả trên được tính trực tiếp từ dữ liệu đã upload, không suy diễn thêm sản phẩm hoặc số liệu ngoài file.",
            ]
        )

    return "\n".join(lines)


def _generate_template_insights(
    df: pd.DataFrame,
    profile: dict[str, Any],
    question: str | None = None,
    sheets: dict[str, pd.DataFrame] | None = None,
) -> str:
    rows, cols = profile["rows"], profile["columns"]
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()

    lines: list[str] = [
        "# Executive Data Brief",
        "",
        f"Phạm vi phân tích: {question}" if question else "Phạm vi phân tích: tự động rà soát dữ liệu, KPI, điểm nổi bật và rủi ro chất lượng.",
    ]
    if sheets and len(sheets) > 1:
        total_rows = sum(len(sheet_df) for sheet_df in sheets.values())
        lines.append(
            f"File có {len(sheets)} sheet/bảng, tổng {total_rows:,} dòng. "
            f"Phân tích định lượng đang chạy trên bảng chính có {rows:,} dòng và {cols:,} cột."
        )
    else:
        lines.append(f"Dataset có {rows:,} dòng và {cols:,} cột.")

    lines.extend(["", "## 1. Tóm tắt điều hành"])
    lines.append(f"- Có {len(numeric_cols)} cột số, {len(categorical_cols)} cột phân loại và {len(datetime_cols)} cột thời gian.")
    missing_total = sum(profile["missing_values"].values())
    if missing_total:
        missing_pct = missing_total / (rows * cols) * 100 if rows and cols else 0
        lines.append(f"- Có {missing_total:,} ô thiếu dữ liệu ({missing_pct:.1f}% toàn bảng).")
    else:
        lines.append("- Không phát hiện missing values ở bảng phân tích chính.")

    lines.extend(["", "## 2. KPI chính"])
    if numeric_cols:
        for col, summary in list(profile["numeric_summary"].items())[:8]:
            total = df[col].sum(skipna=True)
            lines.append(
                f"- `{col}`: tổng {_format_number(total)}, mean {summary['mean']}, "
                f"median {summary['median']}, min {summary['min']}, max {summary['max']}."
            )
    else:
        lines.append("- Không có cột số để tính KPI.")

    driver_lines = _driver_lines(df, numeric_cols, categorical_cols)
    if driver_lines:
        lines.extend(["", "## 3. Nhóm đóng góp lớn nhất"])
        lines.extend(driver_lines)

    if datetime_cols:
        lines.extend(["", "## 4. Cột thời gian"])
        for col in datetime_cols[:3]:
            valid = df[col].dropna()
            if valid.empty:
                continue
            lines.append(f"- `{col}`: từ {valid.min().date()} đến {valid.max().date()}.")

    lines.extend(["", "## 5. Rủi ro và gợi ý"])
    lines.append("- Kiểm tra định nghĩa cột, đơn vị đo, dữ liệu trùng lặp và missing values trước khi dùng cho báo cáo chính thức.")
    if sheets and len(sheets) > 1:
        lines.extend(["", "## 6. Cấu trúc file"])
        lines.extend(_sheet_summary_lines(sheets))

    return "\n".join(lines)


def _driver_lines(df: pd.DataFrame, numeric_cols: list[str], categorical_cols: list[str]) -> list[str]:
    lines: list[str] = []
    for metric in numeric_cols[:3]:
        best_dim = _best_dimension_for_metric(df, metric, categorical_cols)
        if not best_dim:
            continue
        dim, grouped = best_dim
        total = float(grouped[metric].sum())
        lines.append(f"- Theo `{dim}` và `{metric}`:")
        for row in grouped.head(5).to_dict(orient="records"):
            value = row[dim]
            metric_value = float(row[metric])
            share = metric_value / total * 100 if total else 0
            lines.append(f"  {value}: {_format_number(metric_value)} ({share:.1f}%)")
    return lines


def _best_dimension_for_metric(
    df: pd.DataFrame,
    metric: str,
    categorical_cols: list[str],
) -> tuple[str, pd.DataFrame] | None:
    candidates = []
    for dim in categorical_cols:
        unique_count = df[dim].nunique(dropna=False)
        if unique_count < 2 or unique_count > 30:
            continue
        grouped = (
            df.groupby(dim, dropna=False)[metric]
            .sum()
            .reset_index()
            .sort_values(metric, ascending=False)
        )
        total = float(grouped[metric].sum())
        if not total:
            continue
        concentration = float(grouped.head(3)[metric].sum()) / total
        candidates.append((concentration, dim, grouped))
    if not candidates:
        return None
    _, dim, grouped = max(candidates, key=lambda item: item[0])
    grouped[dim] = grouped[dim].astype("string").fillna("<missing>")
    return dim, grouped


def _sheet_summary_lines(sheets: dict[str, pd.DataFrame]) -> list[str]:
    lines = []
    for name, sheet_df in sheets.items():
        numeric_count = len(sheet_df.select_dtypes(include="number").columns)
        category_count = len(sheet_df.select_dtypes(include=["object", "category", "bool"]).columns)
        datetime_count = len(sheet_df.select_dtypes(include=["datetime", "datetimetz"]).columns)
        lines.append(
            f"- `{name}`: {len(sheet_df):,} dòng, {len(sheet_df.columns):,} cột, "
            f"{numeric_count} cột số, {category_count} cột phân loại, {datetime_count} cột thời gian."
        )
    return lines


def _build_context_for_llm(
    df: pd.DataFrame,
    profile: dict[str, Any],
    question: str | None,
    sheets: dict[str, pd.DataFrame] | None = None,
    sheets_context: str | None = None,
    grounding: str | None = None,
) -> str:
    facts = grounding or _generate_template_insights(df, profile, question, sheets=sheets)
    lines = [
        "Bạn là trợ lý phân tích dữ liệu cho CEO. Chỉ dùng các số liệu đã tính sẵn trong context, không bịa thêm số.",
        "Trả lời bằng tiếng Việt, có cấu trúc: Tóm tắt điều hành, Phát hiện chính, Rủi ro, Hành động đề xuất.",
        "",
        "Số liệu đã tính sẵn từ dữ liệu thực:",
        facts,
        "",
        "Mẫu dữ liệu (8 dòng đầu):",
        df.head(8).to_string(),
    ]
    if sheets_context:
        lines.extend(["", "Cấu trúc file nhiều sheet:", sheets_context])
    return "\n".join(lines)


def _format_number(value: float | int) -> str:
    if float(value).is_integer():
        return f"{int(value):,}"
    return f"{value:,.2f}"
