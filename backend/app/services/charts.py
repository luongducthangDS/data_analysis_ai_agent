from __future__ import annotations

import json
import uuid
from typing import Any

import pandas as pd
import plotly.express as px

from backend.app.services.analysis_intent import (
    build_grouped_metric_frame,
    infer_grouped_metric_intent,
)


def generate_question_charts(df: pd.DataFrame, question: str | None) -> list[dict[str, Any]]:
    intent = infer_grouped_metric_intent(question, df)
    if not intent:
        return []

    grouped = build_grouped_metric_frame(df, intent).head(intent.top_n or 12)
    title = f"{intent.metric_label.capitalize()} theo {intent.dimension_label}"
    fig = px.bar(
        grouped,
        x=intent.dimension,
        y=intent.metric,
        title=title,
        text=intent.metric,
    )
    fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
    fig.update_layout(xaxis_title=intent.dimension_label.capitalize(), yaxis_title=intent.metric_label.capitalize())
    return [_chart("bar", title, fig, x=intent.dimension, y=intent.metric)]


def generate_recommended_charts(df: pd.DataFrame, max_charts: int = 4) -> list[dict[str, Any]]:
    charts: list[dict[str, Any]] = []
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()

    contribution = _best_contribution_chart(df, numeric_cols, categorical_cols)
    if contribution:
        charts.append(contribution)

    timeseries = _best_timeseries_chart(df, numeric_cols, datetime_cols)
    if timeseries:
        charts.append(timeseries)

    distribution = _best_distribution_chart(df, numeric_cols)
    if distribution:
        charts.append(distribution)

    relationship = _best_relationship_chart(df, numeric_cols)
    if relationship:
        charts.append(relationship)

    return charts[:max_charts]


def _best_contribution_chart(
    df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> dict[str, Any] | None:
    best: tuple[float, str, str, pd.DataFrame] | None = None
    for metric in numeric_cols:
        for dim in categorical_cols:
            unique_count = df[dim].nunique(dropna=False)
            if unique_count < 2 or unique_count > 30:
                continue
            grouped = (
                df.groupby(dim, dropna=False)[metric]
                .sum()
                .reset_index()
                .sort_values(metric, ascending=False)
                .head(12)
            )
            total = float(grouped[metric].sum())
            if not total:
                continue
            score = float(grouped.head(3)[metric].sum()) / total
            if best is None or score > best[0]:
                best = (score, dim, metric, grouped)

    if not best:
        return None

    _, dim, metric, grouped = best
    title = f"Đóng góp {metric} theo {dim}"
    fig = px.bar(grouped, x=dim, y=metric, title=title, text=metric)
    fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
    fig.update_layout(xaxis_title=dim, yaxis_title=metric)
    return _chart("bar", title, fig, x=dim, y=metric)


def _best_timeseries_chart(
    df: pd.DataFrame,
    numeric_cols: list[str],
    datetime_cols: list[str],
) -> dict[str, Any] | None:
    if not datetime_cols or not numeric_cols:
        return None
    date_col = datetime_cols[0]
    metric = _largest_variance_numeric(df, numeric_cols) or numeric_cols[0]
    data = df[[date_col, metric]].dropna().sort_values(date_col)
    if data.empty:
        return None
    data = data.groupby(date_col, as_index=False)[metric].sum()
    title = f"Xu hướng {metric} theo thời gian"
    fig = px.line(data, x=date_col, y=metric, title=title, markers=True)
    return _chart("line", title, fig, x=date_col, y=metric)


def _best_distribution_chart(df: pd.DataFrame, numeric_cols: list[str]) -> dict[str, Any] | None:
    metric = _largest_variance_numeric(df, numeric_cols)
    if not metric:
        return None
    title = f"Phân phối {metric}"
    fig = px.histogram(df, x=metric, title=title, nbins=min(30, max(5, int(len(df) ** 0.5))))
    return _chart("histogram", title, fig, x=metric)


def _best_relationship_chart(df: pd.DataFrame, numeric_cols: list[str]) -> dict[str, Any] | None:
    if len(numeric_cols) < 2:
        return None
    corr = df[numeric_cols].corr(numeric_only=True).abs()
    best_pair: tuple[float, str, str] | None = None
    for i, x in enumerate(numeric_cols):
        for y in numeric_cols[i + 1 :]:
            value = corr.loc[x, y]
            if pd.isna(value):
                continue
            if best_pair is None or float(value) > best_pair[0]:
                best_pair = (float(value), x, y)
    if not best_pair:
        return None
    _, x, y = best_pair
    title = f"Quan hệ {y} và {x}"
    fig = px.scatter(df, x=x, y=y, title=title, trendline=None)
    return _chart("scatter", title, fig, x=x, y=y)


def _largest_variance_numeric(df: pd.DataFrame, numeric_cols: list[str]) -> str | None:
    if not numeric_cols:
        return None
    variances = df[numeric_cols].var(numeric_only=True).dropna()
    if variances.empty:
        return numeric_cols[0]
    return str(variances.sort_values(ascending=False).index[0])


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
