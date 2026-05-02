from __future__ import annotations

from typing import Any

import pandas as pd


def generate_insights(df: pd.DataFrame, profile: dict[str, Any], question: str | None = None) -> str:
    lines: list[str] = []
    if question:
        lines.append(f"Phân tích theo câu hỏi: {question}")
    else:
        lines.append("Phân tích tự động từ dataset đã upload.")

    rows, cols = profile["rows"], profile["columns"]
    lines.append(f"Dataset có {rows:,} dòng và {cols:,} cột.")

    missing = profile["missing_values"]
    missing_cols = [(col, count) for col, count in missing.items() if count > 0]
    if missing_cols:
        top_missing = sorted(missing_cols, key=lambda item: item[1], reverse=True)[:3]
        formatted = ", ".join(f"{col}: {count:,}" for col, count in top_missing)
        lines.append(f"Các cột thiếu dữ liệu nhiều nhất: {formatted}.")
    else:
        lines.append("Không phát hiện missing values trong dataset.")

    numeric_summary = profile["numeric_summary"]
    if numeric_summary:
        strongest = _largest_numeric_range(numeric_summary)
        if strongest:
            col, summary = strongest
            lines.append(
                f"Cột số `{col}` có biên độ lớn nhất, từ {summary['min']} đến {summary['max']}, "
                f"mean khoảng {summary['mean']}."
            )

    categorical_summary = profile["categorical_summary"]
    if categorical_summary:
        first_col = next(iter(categorical_summary))
        top_values = categorical_summary[first_col][:3]
        formatted = ", ".join(f"{item['value']} ({item['count']})" for item in top_values)
        lines.append(f"Cột phân loại `{first_col}` có các giá trị phổ biến: {formatted}.")

    chart_count = _recommended_chart_count(df)
    lines.append(f"Hệ thống đề xuất {chart_count} biểu đồ đầu tiên để khám phá nhanh xu hướng và phân phối.")
    return "\n".join(lines)


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


def _recommended_chart_count(df: pd.DataFrame) -> int:
    numeric = len(df.select_dtypes(include="number").columns)
    categorical = len(df.select_dtypes(include=["object", "category", "bool"]).columns)
    if numeric >= 2 and categorical >= 1:
        return 3
    if numeric >= 1 and categorical >= 1:
        return 2
    if numeric >= 1:
        return 1
    return 0

