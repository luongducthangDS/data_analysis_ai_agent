"""
E-commerce KPI computation engine.

Computes GMV, AOV, cancel/return rate, top products, revenue trends
from a DataFrame + ecommerce_col_map already detected at upload time.
All columns are resolved strictly from col_map — no guessing.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import pandas as pd

from backend.app.services.ecommerce_columns import CANCEL_STATUSES, RETURN_STATUSES


@dataclass
class EcommerceKPIs:
    # Core KPIs
    gmv: float | None = None               # sum(revenue) excl. cancelled
    order_count: int | None = None         # valid (non-cancelled) orders
    aov: float | None = None               # GMV / order_count
    cancel_rate: float | None = None       # % cancelled
    return_rate: float | None = None       # % returned
    total_quantity: int | None = None      # sum(quantity_col)

    # Breakdowns
    top_products: list[dict] = field(default_factory=list)   # [{name, revenue, qty, rank}]
    revenue_by_platform: list[dict] = field(default_factory=list)
    revenue_by_date: list[dict] = field(default_factory=list)  # [{date, revenue}]

    # Meta
    period_label: str = "Toàn bộ dữ liệu"
    currency_warning: str | None = None
    mapped_cols: dict[str, str] = field(default_factory=dict)  # which cols were used


def _normalize_status(series: pd.Series) -> pd.Series:
    """Lowercase + strip diacritics for Vietnamese status matching."""
    import unicodedata
    def _norm(v: str) -> str:
        t = unicodedata.normalize("NFKD", str(v).strip().lower())
        return "".join(ch for ch in t if not unicodedata.combining(ch))
    return series.fillna("").map(_norm)


def _is_cancelled(status_norm: pd.Series) -> pd.Series:
    pattern = "|".join(re.escape(s) for s in CANCEL_STATUSES)
    return status_norm.str.contains(pattern, na=False)


def _is_returned(status_norm: pd.Series) -> pd.Series:
    pattern = "|".join(re.escape(s) for s in RETURN_STATUSES)
    return status_norm.str.contains(pattern, na=False)


def _format_vnd(value: float) -> str:
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:,.1f} tỷ ₫"
    if value >= 1_000_000:
        return f"{value / 1_000_000:,.1f}M ₫"
    return f"{value:,.0f} ₫"


def compute_ecommerce_kpis(
    df: pd.DataFrame,
    col_map: dict[str, str],
    date_filter_days: int = 30,
) -> EcommerceKPIs:
    """
    Compute e-commerce KPIs from df using resolved col_map.

    date_filter_days: if a date column is found, only consider rows from
    the last N days. Pass 0 to use all data.
    """
    result = EcommerceKPIs(mapped_cols=dict(col_map))
    if df.empty:
        return result

    work_df = df.copy()

    # ── Optional date filter ──────────────────────────────────────────────────
    date_col = col_map.get("order_date_col")
    period_label = "Toàn bộ dữ liệu"
    if date_col and date_col in work_df.columns and date_filter_days > 0:
        date_series = pd.to_datetime(work_df[date_col], errors="coerce")
        if date_series.notna().any():
            # Use max date in data as "today" to avoid timezone issues
            max_date = date_series.max()
            cutoff = max_date - timedelta(days=date_filter_days)
            work_df = work_df[date_series >= cutoff].copy()
            period_label = f"{date_filter_days} ngày gần nhất"
    result.period_label = period_label

    # ── Status masks ─────────────────────────────────────────────────────────
    status_col = col_map.get("status_col")
    if status_col and status_col in work_df.columns:
        status_norm = _normalize_status(work_df[status_col])
        cancelled_mask = _is_cancelled(status_norm)
        returned_mask = _is_returned(status_norm)
        valid_mask = ~(cancelled_mask | returned_mask)
        result.cancel_rate = round(float(cancelled_mask.mean()), 4) if len(work_df) > 0 else None
        result.return_rate = round(float(returned_mask.mean()), 4) if len(work_df) > 0 else None
    else:
        valid_mask = pd.Series([True] * len(work_df), index=work_df.index)

    valid_df = work_df[valid_mask]

    # ── Revenue (GMV) ─────────────────────────────────────────────────────────
    revenue_col = col_map.get("revenue_col")
    if revenue_col and revenue_col in valid_df.columns:
        rev_series = pd.to_numeric(valid_df[revenue_col], errors="coerce").fillna(0)
        result.gmv = float(rev_series.sum())
        result.order_count = int(valid_mask.sum())
        if result.order_count > 0:
            result.aov = result.gmv / result.order_count

        # Revenue trend by date
        if date_col and date_col in valid_df.columns:
            d_series = pd.to_datetime(valid_df[date_col], errors="coerce")
            trend_df = valid_df.copy()
            trend_df["_date"] = d_series.dt.date
            trend = (
                trend_df.dropna(subset=["_date"])
                .assign(_rev=rev_series)
                .groupby("_date")["_rev"]
                .sum()
                .reset_index()
                .sort_values("_date")
            )
            result.revenue_by_date = [
                {"date": str(row["_date"]), "revenue": float(row["_rev"])}
                for _, row in trend.iterrows()
            ]

        # Revenue by platform
        platform_col = col_map.get("platform_col")
        if platform_col and platform_col in valid_df.columns:
            plat = (
                valid_df.assign(_rev=rev_series)
                .groupby(platform_col)["_rev"]
                .sum()
                .reset_index()
                .sort_values("_rev", ascending=False)
            )
            result.revenue_by_platform = [
                {"platform": str(row[platform_col]), "revenue": float(row["_rev"])}
                for _, row in plat.iterrows()
            ]

    # ── Quantity ──────────────────────────────────────────────────────────────
    qty_col = col_map.get("quantity_col")
    if qty_col and qty_col in valid_df.columns:
        result.total_quantity = int(
            pd.to_numeric(valid_df[qty_col], errors="coerce").fillna(0).sum()
        )

    # ── Top products ──────────────────────────────────────────────────────────
    product_col = col_map.get("product_name_col") or col_map.get("sku_col")
    if product_col and product_col in valid_df.columns and revenue_col and revenue_col in valid_df.columns:
        rev_series_prod = pd.to_numeric(valid_df[revenue_col], errors="coerce").fillna(0)
        grp = valid_df.assign(_rev=rev_series_prod)
        if qty_col and qty_col in valid_df.columns:
            qty_s = pd.to_numeric(valid_df[qty_col], errors="coerce").fillna(0)
            grp = grp.assign(_qty=qty_s).groupby(product_col).agg(
                revenue=("_rev", "sum"), qty=("_qty", "sum")
            )
        else:
            grp = grp.groupby(product_col).agg(revenue=("_rev", "sum"))
            grp["qty"] = None
        grp = grp.reset_index().sort_values("revenue", ascending=False).head(10)
        grp["rank"] = range(1, len(grp) + 1)
        result.top_products = [
            {
                "rank": int(row["rank"]),
                "name": str(row[product_col]),
                "revenue": float(row["revenue"]),
                "qty": int(row["qty"]) if "qty" in row and row["qty"] is not None and pd.notna(row["qty"]) else None,
            }
            for _, row in grp.iterrows()
        ]

    # ── Currency warning ──────────────────────────────────────────────────────
    if revenue_col and revenue_col in valid_df.columns:
        rev_vals = pd.to_numeric(valid_df[revenue_col], errors="coerce").dropna()
        if len(rev_vals) > 1:
            cv = rev_vals.std() / rev_vals.mean() if rev_vals.mean() != 0 else 0
            if cv > 10:
                result.currency_warning = "Cột doanh thu có độ phân tán rất lớn — có thể trộn nhiều đơn vị tiền tệ."

    return result


def compute_period_delta(
    df: pd.DataFrame,
    col_map: dict[str, str],
    current_days: int = 7,
) -> dict[str, float | None]:
    """
    Compare current period (last current_days) vs prior period (same length before that).
    Returns {"gmv": delta_pct, "order_count": delta_pct, "aov": delta_pct}
    where delta_pct is a signed fraction (0.12 = +12%).
    """
    date_col = col_map.get("order_date_col")
    if not date_col or date_col not in df.columns:
        return {"gmv": None, "order_count": None, "aov": None}

    d_series = pd.to_datetime(df[date_col], errors="coerce")
    if d_series.isna().all():
        return {"gmv": None, "order_count": None, "aov": None}

    max_date = d_series.max()
    cutoff_curr = max_date - timedelta(days=current_days)
    cutoff_prev = cutoff_curr - timedelta(days=current_days)

    curr_df = df[d_series >= cutoff_curr]
    prev_df = df[(d_series >= cutoff_prev) & (d_series < cutoff_curr)]

    curr = compute_ecommerce_kpis(curr_df, col_map, date_filter_days=0)
    prev = compute_ecommerce_kpis(prev_df, col_map, date_filter_days=0)

    def _delta(a: float | None, b: float | None) -> float | None:
        if a is None or b is None or b == 0:
            return None
        return round((a - b) / abs(b), 4)

    return {
        "gmv": _delta(curr.gmv, prev.gmv),
        "order_count": _delta(
            float(curr.order_count) if curr.order_count else None,
            float(prev.order_count) if prev.order_count else None,
        ),
        "aov": _delta(curr.aov, prev.aov),
        "cancel_rate": _delta(curr.cancel_rate, prev.cancel_rate),
    }
