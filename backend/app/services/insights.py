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
    """Generate executive-ready data analysis with an optional LLM layer."""
    focused = _generate_focused_grouped_metric_answer(df, question)
    if focused:
        return focused

    deterministic_answer = _generate_template_insights(df, profile, question, sheets=sheets)

    hf_token = os.getenv("HF_TOKEN", "").strip()
    if not hf_token:
        return deterministic_answer

    try:
        from backend.app.services.llm_service import get_hf_client

        client = get_hf_client()
        context = _build_context_for_llm(df, profile, question, sheets=sheets, sheets_context=sheets_context)
        if question:
            return client.answer_question(question, context, max_tokens=700)
        return client.generate_insights(context, max_tokens=900)
    except Exception as exc:
        print(f"LLM inference failed, falling back to deterministic analysis: {exc}")
        return deterministic_answer


def _generate_template_insights(
    df: pd.DataFrame,
    profile: dict[str, Any],
    question: str | None = None,
    sheets: dict[str, pd.DataFrame] | None = None,
) -> str:
    rows, cols = profile["rows"], profile["columns"]
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    lines: list[str] = []
    lines.append("# Executive Data Brief")
    lines.append("")
    if question:
        lines.append(f"Phạm vi phân tích: {question}")
    else:
        lines.append("Phạm vi phân tích: tự động rà soát dữ liệu, tìm KPI, điểm nổi bật, rủi ro chất lượng và hướng hành động.")

    if sheets and len(sheets) > 1:
        total_rows = sum(len(sheet_df) for sheet_df in sheets.values())
        lines.append(f"File có {len(sheets)} sheet/bảng, tổng {total_rows:,} dòng. Phần phân tích định lượng dưới đây đang chạy trên bảng phân tích chính có {rows:,} dòng và {cols:,} cột.")
    else:
        lines.append(f"Dataset có {rows:,} dòng và {cols:,} cột.")

    lines.extend(["", "## 1. Tóm tắt điều hành"])
    lines.extend(_executive_summary(df, profile, numeric_cols, categorical_cols))

    lines.extend(["", "## 2. KPI và biến động chính"])
    lines.extend(_numeric_kpi_lines(df, profile))

    driver_lines = _driver_lines(df, numeric_cols, categorical_cols)
    if driver_lines:
        lines.extend(["", "## 3. Nhóm đóng góp lớn nhất"])
        lines.extend(driver_lines)

    quality_lines = _data_quality_lines(profile)
    lines.extend(["", "## 4. Rủi ro dữ liệu"])
    lines.extend(quality_lines)

    recommendation_lines = _recommendation_lines(df, numeric_cols, categorical_cols, quality_lines)
    lines.extend(["", "## 5. Gợi ý hành động"])
    lines.extend(recommendation_lines)

    if sheets and len(sheets) > 1:
        lines.extend(["", "## 6. Cấu trúc file"])
        lines.extend(_sheet_summary_lines(sheets))

    return "\n".join(lines)


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
    top_share = (top_value / total * 100) if total else 0

    lines = [
        "# Executive Data Brief",
        "",
        f"## {intent.metric_label.capitalize()} theo {intent.dimension_label}",
        f"Tổng {intent.metric_label}: {total:,.0f}.",
        (
            f"{intent.dimension_label.capitalize()} dẫn đầu là {top[intent.dimension]} "
            f"với {top_value:,.0f}, chiếm {top_share:.1f}% tổng {intent.metric_label}."
        ),
        "",
        "### Xếp hạng đóng góp",
    ]

    for index, row in enumerate(grouped.head(10).to_dict(orient="records"), start=1):
        dimension_value = row[intent.dimension]
        metric_value = float(row[intent.metric])
        share = (metric_value / total * 100) if total else 0
        lines.append(f"{index}. {dimension_value}: {metric_value:,.0f} ({share:.1f}%)")

    if len(grouped) > 1:
        top_3_share = float(grouped.head(3)[intent.metric].sum()) / total * 100 if total else 0
        lines.extend(
            [
                "",
                "### Nhận định",
                f"Top 3 nhóm đóng góp {top_3_share:.1f}% tổng {intent.metric_label}. Nếu tỷ trọng này quá cao, CEO nên xem đây là rủi ro phụ thuộc và cần kiểm tra khả năng mở rộng sang nhóm còn lại.",
                "Biểu đồ cột đi kèm giúp so sánh nhanh mức đóng góp giữa các nhóm.",
            ]
        )

    return "\n".join(lines)


def _executive_summary(
    df: pd.DataFrame,
    profile: dict[str, Any],
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> list[str]:
    lines: list[str] = []
    missing_total = sum(profile["missing_values"].values())
    missing_pct = missing_total / (profile["rows"] * profile["columns"]) * 100 if profile["rows"] and profile["columns"] else 0

    lines.append(f"- Dữ liệu gồm {len(numeric_cols)} chỉ số số học và {len(categorical_cols)} chiều phân loại có thể dùng để phân tích đóng góp.")
    if missing_total:
        lines.append(f"- Có {missing_total:,} ô thiếu dữ liệu ({missing_pct:.1f}% toàn bảng), cần xử lý trước khi dùng cho quyết định chính thức.")
    else:
        lines.append("- Không phát hiện missing values ở bảng phân tích chính.")

    strongest = _largest_numeric_range(profile["numeric_summary"])
    if strongest:
        col, summary = strongest
        lines.append(f"- Chỉ số biến động mạnh nhất là `{col}`: min {summary['min']}, max {summary['max']}, trung bình {summary['mean']}.")

    driver = _top_driver(df, numeric_cols, categorical_cols)
    if driver:
        dim, metric, value, metric_value, share = driver
        lines.append(f"- Nhóm `{value}` trong `{dim}` đang dẫn đầu theo `{metric}` với {metric_value:,.0f}, chiếm {share:.1f}% tổng.")

    return lines


def _numeric_kpi_lines(df: pd.DataFrame, profile: dict[str, Any]) -> list[str]:
    numeric_summary = profile["numeric_summary"]
    if not numeric_summary:
        return ["- Không có cột số để tính KPI."]

    lines = []
    for col, summary in list(numeric_summary.items())[:8]:
        total = df[col].sum(skipna=True)
        lines.append(
            f"- `{col}`: tổng {total:,.0f}, mean {summary['mean']}, median {summary['median']}, min {summary['min']}, max {summary['max']}."
        )
    return lines


def _driver_lines(df: pd.DataFrame, numeric_cols: list[str], categorical_cols: list[str]) -> list[str]:
    lines: list[str] = []
    if not numeric_cols or not categorical_cols:
        return lines

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
            share = (metric_value / total * 100) if total else 0
            lines.append(f"  {value}: {metric_value:,.0f} ({share:.1f}%)")
    return lines


def _data_quality_lines(profile: dict[str, Any]) -> list[str]:
    rows = profile["rows"]
    missing = [(col, count) for col, count in profile["missing_values"].items() if count > 0]
    if not missing:
        return ["- Chưa thấy lỗi thiếu dữ liệu rõ ràng. Vẫn nên kiểm tra định nghĩa cột, đơn vị đo và dữ liệu trùng lặp trước khi báo cáo chính thức."]

    lines = []
    for col, count in sorted(missing, key=lambda item: item[1], reverse=True)[:5]:
        pct = count / rows * 100 if rows else 0
        lines.append(f"- `{col}` thiếu {count:,} dòng ({pct:.1f}%).")
    return lines


def _recommendation_lines(
    df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    quality_lines: list[str],
) -> list[str]:
    lines = []
    driver = _top_driver(df, numeric_cols, categorical_cols)
    if driver:
        dim, metric, value, _, share = driver
        lines.append(f"- Ưu tiên kiểm tra nguyên nhân nhóm `{value}` trong `{dim}` chiếm {share:.1f}% `{metric}`: đây có thể là động lực tăng trưởng hoặc điểm phụ thuộc.")
    if numeric_cols and categorical_cols:
        lines.append("- So sánh top/bottom nhóm theo từng KPI để tìm cơ hội tối ưu, ví dụ tăng doanh thu, giảm chi phí hoặc phân bổ nguồn lực.")
    if any(line.startswith("- `") and "thiếu" in line.lower() for line in quality_lines):
        lines.append("- Chuẩn hóa missing values trước khi đưa số liệu vào dashboard điều hành.")
    lines.append("- Dùng các biểu đồ bên dưới để xác nhận nhanh: đóng góp theo nhóm, phân phối KPI và quan hệ giữa các KPI.")
    return lines


def _sheet_summary_lines(sheets: dict[str, pd.DataFrame]) -> list[str]:
    lines = []
    for name, sheet_df in sheets.items():
        numeric_count = len(sheet_df.select_dtypes(include="number").columns)
        category_count = len(sheet_df.select_dtypes(include=["object", "category", "bool"]).columns)
        lines.append(f"- `{name}`: {len(sheet_df):,} dòng, {len(sheet_df.columns):,} cột, {numeric_count} cột số, {category_count} cột phân loại.")
    return lines


def _top_driver(
    df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> tuple[str, str, Any, float, float] | None:
    for metric in numeric_cols:
        best_dim = _best_dimension_for_metric(df, metric, categorical_cols)
        if not best_dim:
            continue
        dim, grouped = best_dim
        total = float(grouped[metric].sum())
        if not total:
            continue
        top = grouped.iloc[0]
        value = top[dim]
        metric_value = float(top[metric])
        return dim, metric, value, metric_value, metric_value / total * 100
    return None


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


def _largest_numeric_range(numeric_summary: dict[str, dict[str, Any]]) -> tuple[str, dict[str, Any]] | None:
    candidates = []
    for col, summary in numeric_summary.items():
        if summary["min"] is None or summary["max"] is None:
            continue
        candidates.append((col, summary, float(summary["max"]) - float(summary["min"])))
    if not candidates:
        return None
    col, summary, _ = max(candidates, key=lambda item: item[2])
    return col, summary


def _build_context_for_llm(
    df: pd.DataFrame,
    profile: dict[str, Any],
    question: str | None,
    sheets: dict[str, pd.DataFrame] | None = None,
    sheets_context: str | None = None,
) -> str:
    deterministic = _generate_template_insights(df, profile, question, sheets=sheets)
    lines = [
        "Bạn là trợ lý phân tích dữ liệu cho CEO. Hãy dùng các số liệu đã tính sẵn, không bịa thêm số.",
        "Trả lời bằng tiếng Việt, có cấu trúc: Tóm tắt điều hành, Phát hiện chính, Rủi ro, Hành động đề xuất.",
        "",
        "Phân tích đã tính sẵn:",
        deterministic,
        "",
        "Mẫu dữ liệu:",
        df.head(8).to_string(),
    ]
    if sheets_context:
        lines.extend(["", "Cấu trúc file nhiều sheet:", sheets_context])
    return "\n".join(lines)
