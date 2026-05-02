from __future__ import annotations

import re
from typing import Any

import duckdb
import pandas as pd


FORBIDDEN_SQL = re.compile(
    r"\b(insert|update|delete|drop|alter|truncate|create|replace|copy|attach|detach)\b|;.*\w",
    re.IGNORECASE | re.DOTALL,
)


def run_readonly_query(df: pd.DataFrame, sql: str, limit: int = 100) -> list[dict[str, Any]]:
    cleaned = sql.strip().rstrip(";")
    if not re.match(r"^\s*select\b", cleaned, re.IGNORECASE):
        raise ValueError("Only SELECT queries are allowed.")
    if FORBIDDEN_SQL.search(cleaned):
        raise ValueError("Query contains a forbidden SQL pattern.")
    if " limit " not in f" {cleaned.lower()} ":
        cleaned = f"{cleaned} LIMIT {limit}"

    with duckdb.connect(database=":memory:") as conn:
        conn.register("dataset", df)
        result = conn.execute(cleaned).fetchdf()
    return _frame_to_records(result)


def simple_question_to_sql(question: str, df: pd.DataFrame) -> str:
    normalized = question.lower()
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    if "bao nhiêu dòng" in normalized or "how many rows" in normalized or "số dòng" in normalized:
        return "SELECT COUNT(*) AS row_count FROM dataset"

    if numeric_cols and ("trung bình" in normalized or "average" in normalized or "mean" in normalized):
        col = _find_column(normalized, numeric_cols) or numeric_cols[0]
        return f'SELECT AVG("{col}") AS avg_{_safe_alias(col)} FROM dataset'

    if numeric_cols and ("tổng" in normalized or "sum" in normalized):
        col = _find_column(normalized, numeric_cols) or numeric_cols[0]
        return f'SELECT SUM("{col}") AS sum_{_safe_alias(col)} FROM dataset'

    if categorical_cols and ("top" in normalized or "phổ biến" in normalized or "nhiều nhất" in normalized):
        cat = _find_column(normalized, categorical_cols) or categorical_cols[0]
        return f'SELECT "{cat}", COUNT(*) AS count FROM dataset GROUP BY "{cat}" ORDER BY count DESC LIMIT 10'

    return "SELECT * FROM dataset LIMIT 10"


def _find_column(question: str, columns: list[str]) -> str | None:
    for col in columns:
        if col.lower() in question:
            return col
    return None


def _safe_alias(column: str) -> str:
    return re.sub(r"\W+", "_", column.lower()).strip("_")


def _frame_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    records = df.where(pd.notna(df), None).to_dict(orient="records")
    return [{str(key): value for key, value in row.items()} for row in records]

