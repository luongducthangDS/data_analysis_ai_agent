"""Plan grammar, validation and deterministic Pandas execution. No LLM here."""
from __future__ import annotations

import math
import re
from typing import Any

import pandas as pd

from backend.app.services.ecommerce_semantic import (
    BRIDGE_REQUIRED, profit_bridge,
)
from backend.app.services.numeric_parse import parse_numeric_series
from backend.app.services.security import LITERAL_CONTAINS


ALLOWED_ACTIONS = {"aggregate", "compare_metrics", "time_series", "profile", "distribution", "profit_bridge"}
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
# Trần số dòng trả về. Kết quả còn phải đi qua LLM synthesis nên limit khổng lồ
# vừa phình payload vừa phình token — chặn ngay ở tầng validate.
MAX_PLAN_LIMIT = 10_000


def execute_plan(df: pd.DataFrame, plan: dict[str, Any]) -> pd.DataFrame:
    _validate_plan_shape(plan)
    work = df.copy()
    work = _apply_derived_columns(work, plan.get("derived_columns", []))
    work = _apply_filters(work, plan.get("filters", []))

    action = plan["action"]
    if work.empty and action in {"aggregate", "compare_metrics", "time_series"}:
        # sum() trên 0 dòng = 0 → từng thành "lãi tháng 7 là 0 đ" khi dữ liệu chỉ tới tháng 6.
        return pd.DataFrame()
    if action == "profile":
        return _profile_frame(work)
    if action == "aggregate":
        return _execute_aggregate(work, plan)
    if action == "compare_metrics":
        return _execute_compare_metrics(work, plan)
    if action == "time_series":
        return _execute_time_series(work, plan)
    if action == "distribution":
        return _execute_distribution(work, plan)
    if action == "profit_bridge":
        return profit_bridge(work, plan)
    raise ValueError(f"Unsupported action: {action}")


def _validate_plan_shape(plan: dict[str, Any]) -> None:
    if not isinstance(plan, dict):
        raise ValueError("Plan must be a JSON object.")
    action = plan.get("action")
    if action not in ALLOWED_ACTIONS:
        raise ValueError(f"Unsupported or missing action: {action}")
    limit = int(plan.get("limit", 100) or 100)
    if limit < 1:
        raise ValueError("limit must be positive.")
    if limit > MAX_PLAN_LIMIT:
        raise ValueError(f"limit vượt trần cho phép ({MAX_PLAN_LIMIT}).")


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
    if plan.get("action") == "distribution":
        _require_column(known, plan.get("column"))
    if plan.get("action") == "profit_bridge":
        for col in (*BRIDGE_REQUIRED, plan.get("time_column") or "ngay_dat"):
            _require_column(known, col)
        periods = plan.get("periods")
        if periods is not None and not (
            isinstance(periods, list) and len(periods) == 2
            and all(isinstance(p, list) and len(p) == 2 and all(_is_date_only(d) for d in p) for p in periods)
        ):
            raise ValueError("profit_bridge.periods phải là [[từ, đến], [từ, đến]] dạng YYYY-MM-DD.")
    for metric in plan.get("metrics", []) or []:
        _require_column(known, metric.get("column"))
        if metric.get("aggregation", "sum") not in ALLOWED_AGGREGATIONS:
            raise ValueError(f"Unsupported aggregation: {metric.get('aggregation')}")
    labels = {_metric_label(m) for m in plan.get("metrics", []) or []}
    for ratio in plan.get("ratios", []) or []:
        if not isinstance(ratio, dict) or not ratio.get("label"):
            raise ValueError("ratio cần label, numerator, denominator.")
        for key in ("numerator", "denominator"):
            if ratio.get(key) not in labels:
                raise ValueError(f"ratio.{key} phải là label của một metric trong plan: {ratio.get(key)}")


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
        is_dt = pd.api.types.is_datetime64_any_dtype(series)
        if is_dt:
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
        elif op == "gt" and is_dt and _is_date_only(value):
            mask = series_cmp >= value_cmp + pd.Timedelta(days=1)
        elif op == "gt":
            mask = series_cmp > value_cmp
        elif op == "gte":
            mask = series_cmp >= value_cmp
        elif op == "lt":
            mask = series_cmp < value_cmp
        elif op == "lte" and is_dt and _is_date_only(value):
            mask = series_cmp < value_cmp + pd.Timedelta(days=1)
        elif op == "lte":
            mask = series_cmp <= value_cmp
        elif op == "between":
            lo, hi = value_cmp
            if is_dt and _is_date_only(value[1]):
                # "2026-05-31" là cả ngày 31/5, không phải 00:00 — cột có giờ sẽ mất ngày cuối kỳ.
                mask = (series_cmp >= lo) & (series_cmp < hi + pd.Timedelta(days=1))
            else:
                mask = (series_cmp >= lo) & (series_cmp <= hi)
        elif op == "in":
            mask = series_cmp.isin(value_cmp if isinstance(value_cmp, list) else [value_cmp])
        elif op == "contains":
            # regex=False: `contains` ở đây luôn có nghĩa "chứa chuỗi con".
            # Để mặc định (regex=True) thì một giá trị filter như "(a+)+$"
            # gây catastrophic backtracking — xem services/security.py.
            mask = series_cmp.astype(str).str.contains(
                str(value_cmp), case=False, na=False, regex=LITERAL_CONTAINS
            )
        else:
            raise ValueError(f"Unsupported filter operator: {op}")
        work = work.loc[mask].copy()
    return work


def _is_date_only(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"\s*\d{4}-\d{1,2}-\d{1,2}\s*", value) is not None


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
            rows[_metric_label(metric)] = _aggregate_series(df[metric["column"]], metric.get("aggregation", "sum"))
        result = pd.DataFrame([rows])

    for ratio in plan.get("ratios", []) or []:
        # Tỷ số của 2 TỔNG (vd phí sàn / doanh thu thuần), không phải trung bình tỷ lệ từng đơn.
        den = pd.to_numeric(result[ratio["denominator"]], errors="coerce")
        result[ratio["label"]] = (pd.to_numeric(result[ratio["numerator"]], errors="coerce") / den.where(den != 0)).round(4)
    return _sort_and_limit(result, plan)


def _metric_label(metric: dict[str, Any]) -> str:
    return metric.get("label") or f"{metric.get('aggregation', 'sum')}_{metric.get('column')}"



def _execute_compare_metrics(df: pd.DataFrame, plan: dict[str, Any]) -> pd.DataFrame:
    rows = []
    group_by = plan.get("group_by", []) or []
    if group_by or plan.get("ratios"):
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


def _execute_distribution(df: pd.DataFrame, plan: dict[str, Any]) -> pd.DataFrame:
    """
    Histogram of a numeric column → DataFrame [khoang_gia_tri (bin label), so_luong (count)].
    Bins are ordered ascending so the result reads like a histogram.
    """
    column = plan.get("column")
    if not column:
        raise ValueError("distribution needs a column.")
    series = pd.to_numeric(df[column], errors="coerce").dropna()
    if series.empty:
        raise ValueError(f"Column '{column}' has no numeric values for distribution.")

    n_bins = min(int(plan.get("bins", 10) or 10), 50)
    # If few distinct values, don't over-bin.
    n_unique = series.nunique()
    if n_unique <= 1:
        return pd.DataFrame([{"khoang_gia_tri": f"{series.iloc[0]:g}", "so_luong": int(len(series))}])
    n_bins = max(2, min(n_bins, n_unique))

    binned = pd.cut(series, bins=n_bins)
    counts = binned.value_counts(sort=False)
    rows = [
        {"khoang_gia_tri": f"{interval.left:.1f}–{interval.right:.1f}", "so_luong": int(count)}
        for interval, count in counts.items()
    ]
    return pd.DataFrame(rows)


def _describe_numeric(df: pd.DataFrame, column: str) -> dict[str, float | int | None]:
    """Descriptive stats for a numeric column — used to answer distribution/range questions."""
    if column not in df.columns:
        return {}
    series = pd.to_numeric(df[column], errors="coerce").dropna()
    if series.empty:
        return {}
    return {
        "count": int(series.count()),
        "min": _round(float(series.min())),
        "max": _round(float(series.max())),
        "range": _round(float(series.max() - series.min())),
        "mean": _round(float(series.mean())),
        "median": _round(float(series.median())),
        "std": _round(float(series.std())) if len(series) > 1 else None,
        "q1": _round(float(series.quantile(0.25))),
        "q3": _round(float(series.quantile(0.75))),
    }


def _round(value: float | None) -> float | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return round(value, 2)


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
    # Hiểu cả " $ (4,533.75) ". Dữ liệu nạp qua storage đã được chuẩn hoá,
    # nhưng đường này còn phục vụ DataFrame dựng trực tiếp (test, join nhiều
    # sheet), nên vẫn phải tự phòng.
    return parse_numeric_series(series)


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


def _normalize(text: Any) -> str:
    import unicodedata
    normalized = unicodedata.normalize("NFKD", str(text).strip().lower())
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", normalized)
