from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class GroupedMetricIntent:
    metric: str
    dimension: str
    metric_label: str
    dimension_label: str


METRIC_SYNONYMS = {
    "sales": ("sales", "revenue", "amount", "total", "doanh thu", "turnover"),
    "profit": ("profit", "margin", "lợi nhuận", "loi nhuan"),
    "quantity": ("quantity", "qty", "units", "số lượng", "so luong"),
}

DIMENSION_SYNONYMS = {
    "region": ("region", "area", "zone", "vùng", "vung", "khu vực", "khu vuc", "miền", "mien"),
    "category": ("category", "segment", "product category", "nhóm", "nhom", "danh mục", "danh muc"),
    "country": ("country", "quốc gia", "quoc gia"),
    "customer": ("customer", "client", "khách hàng", "khach hang"),
}


def infer_grouped_metric_intent(question: str | None, df: pd.DataFrame) -> GroupedMetricIntent | None:
    if not question:
        return None

    normalized = normalize_text(question)
    if " theo " not in f" {normalized} " and not any(
        token in normalized for token in ("by ", "theo từng", "theo tung")
    ):
        return None

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    dimension_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    if not numeric_cols or not dimension_cols:
        return None

    metric = _match_column(normalized, numeric_cols, METRIC_SYNONYMS) or numeric_cols[0]
    dimension = _match_column(normalized, dimension_cols, DIMENSION_SYNONYMS)
    if not dimension:
        dimension = _match_dimension_after_theo(normalized, dimension_cols)
    if not dimension:
        return None

    return GroupedMetricIntent(
        metric=metric,
        dimension=dimension,
        metric_label=_friendly_metric_label(metric),
        dimension_label=_friendly_dimension_label(dimension),
    )


def build_grouped_metric_frame(df: pd.DataFrame, intent: GroupedMetricIntent) -> pd.DataFrame:
    grouped = (
        df.groupby(intent.dimension, dropna=False)[intent.metric]
        .sum()
        .reset_index()
        .sort_values(intent.metric, ascending=False)
    )
    grouped[intent.dimension] = grouped[intent.dimension].astype("string").fillna("<missing>")
    return grouped


def normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", text)


def _match_column(
    normalized_question: str,
    columns: list[str],
    synonym_groups: dict[str, tuple[str, ...]],
) -> str | None:
    normalized_cols = {normalize_text(col): col for col in columns}

    for normalized_col, original_col in normalized_cols.items():
        if normalized_col in normalized_question:
            return original_col

    for canonical, synonyms in synonym_groups.items():
        if not any(normalize_text(synonym) in normalized_question for synonym in synonyms):
            continue
        for normalized_col, original_col in normalized_cols.items():
            if canonical in normalized_col or any(normalize_text(synonym) == normalized_col for synonym in synonyms):
                return original_col
    return None


def _match_dimension_after_theo(normalized_question: str, columns: list[str]) -> str | None:
    match = re.search(r"\btheo\s+([\w\s]+)$", normalized_question)
    if not match:
        return None
    phrase = match.group(1).strip()
    return _match_column(phrase, columns, DIMENSION_SYNONYMS)


def _friendly_metric_label(column: str) -> str:
    normalized = normalize_text(column)
    if "sales" in normalized or "revenue" in normalized or "doanh" in normalized:
        return "doanh thu"
    if "profit" in normalized:
        return "lợi nhuận"
    if "quantity" in normalized or "qty" in normalized:
        return "số lượng"
    return column


def _friendly_dimension_label(column: str) -> str:
    normalized = normalize_text(column)
    if "region" in normalized:
        return "vùng"
    if "category" in normalized:
        return "nhóm"
    if "country" in normalized:
        return "quốc gia"
    return column

