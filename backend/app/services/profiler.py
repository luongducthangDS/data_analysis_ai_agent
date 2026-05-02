from __future__ import annotations

from typing import Any

import pandas as pd


def build_profile(df: pd.DataFrame) -> dict[str, Any]:
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    numeric_summary: dict[str, dict[str, float | int | None]] = {}
    for col in numeric_cols:
        series = df[col].dropna()
        numeric_summary[col] = {
            "min": _safe_float(series.min()) if not series.empty else None,
            "max": _safe_float(series.max()) if not series.empty else None,
            "mean": _safe_float(series.mean()) if not series.empty else None,
            "median": _safe_float(series.median()) if not series.empty else None,
            "std": _safe_float(series.std()) if len(series) > 1 else None,
        }

    categorical_summary: dict[str, list[dict[str, Any]]] = {}
    for col in categorical_cols[:12]:
        counts = df[col].astype("string").fillna("<missing>").value_counts().head(5)
        categorical_summary[col] = [
            {"value": str(value), "count": int(count)} for value, count in counts.items()
        ]

    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_values": {col: int(count) for col, count in df.isna().sum().items()},
        "numeric_summary": numeric_summary,
        "categorical_summary": categorical_summary,
    }


def _safe_float(value: Any) -> float | None:
    if pd.isna(value):
        return None
    return round(float(value), 4)

