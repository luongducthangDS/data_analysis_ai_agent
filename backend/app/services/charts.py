from __future__ import annotations

import uuid
import json
from typing import Any

import pandas as pd
import plotly.express as px


def generate_recommended_charts(df: pd.DataFrame, max_charts: int = 3) -> list[dict[str, Any]]:
    charts: list[dict[str, Any]] = []
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()

    if categorical_cols and numeric_cols:
        cat, num = categorical_cols[0], numeric_cols[0]
        grouped = df.groupby(cat, dropna=False)[num].sum().reset_index().sort_values(num, ascending=False).head(10)
        fig = px.bar(grouped, x=cat, y=num, title=f"Top {cat} by total {num}")
        charts.append(_chart("bar", f"Top {cat} by total {num}", fig, x=cat, y=num))

    if numeric_cols:
        num = numeric_cols[0]
        fig = px.histogram(df, x=num, title=f"Distribution of {num}")
        charts.append(_chart("histogram", f"Distribution of {num}", fig, x=num))

    if len(numeric_cols) >= 2:
        x, y = numeric_cols[0], numeric_cols[1]
        fig = px.scatter(df, x=x, y=y, title=f"{y} vs {x}")
        charts.append(_chart("scatter", f"{y} vs {x}", fig, x=x, y=y))

    if datetime_cols and numeric_cols and len(charts) < max_charts:
        date_col, num = datetime_cols[0], numeric_cols[0]
        timeseries = df[[date_col, num]].dropna().sort_values(date_col)
        fig = px.line(timeseries, x=date_col, y=num, title=f"{num} over time")
        charts.append(_chart("line", f"{num} over time", fig, x=date_col, y=num))

    return charts[:max_charts]


def _chart(chart_type: str, title: str, fig: Any, x: str | None = None, y: str | None = None) -> dict[str, Any]:
    return {
        "chart_id": uuid.uuid4().hex,
        "title": title,
        "chart_type": chart_type,
        "x": x,
        "y": y,
        "plotly_json": json.loads(fig.to_json()),
    }
