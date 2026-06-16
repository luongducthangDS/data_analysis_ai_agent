"""
Tests for backend service layer — 16 test functions.

Coverage:
  profiler (6), analysis_intent (5), reports (2), analysis_planner (3)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from backend.app.services.profiler import build_profile
from backend.app.services.analysis_intent import (
    GroupedMetricIntent,
    build_grouped_metric_frame,
    infer_grouped_metric_intent,
    normalize_text,
)
from backend.app.services.reports import write_markdown_report
from backend.app.services.analysis_planner import build_fallback_plan, execute_plan


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_df() -> pd.DataFrame:
    return pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]),
        "category": ["A", "B", "A", "C", "B"],
        "amount": [100.0, 200.0, 150.0, 300.0, 250.0],
        "count": [1, 2, 1, 3, 2],
    })


def _make_profile(df=None) -> dict[str, Any]:
    return build_profile(df if df is not None else _make_df())


# ═════════════════════════════════════════════════════════════════════════════
# PROFILER (1–6)
# ═════════════════════════════════════════════════════════════════════════════

def test_build_profile_has_required_keys():
    profile = build_profile(_make_df())
    for key in ("rows", "columns", "column_types", "missing_values", "numeric_summary", "categorical_summary"):
        assert key in profile

def test_build_profile_row_count():
    assert build_profile(_make_df())["rows"] == 5

def test_build_profile_column_count():
    assert build_profile(_make_df())["columns"] == 4

def test_build_profile_missing_values_counted():
    df = pd.DataFrame({"x": [1.0, None, 3.0], "y": ["a", "b", "c"]})
    assert build_profile(df)["missing_values"]["x"] == 1

def test_build_profile_numeric_summary_has_stats():
    stats = build_profile(pd.DataFrame({"value": [10.0, 20.0, 30.0]}))["numeric_summary"]["value"]
    assert "mean" in stats and "min" in stats and "max" in stats

def test_build_profile_categorical_summary_top_values():
    top = build_profile(pd.DataFrame({"cat": ["a", "b", "a", "c", "a"]}))["categorical_summary"]["cat"]
    assert isinstance(top, list) and "value" in top[0] and "count" in top[0]


# ═════════════════════════════════════════════════════════════════════════════
# ANALYSIS INTENT (7–11)
# ═════════════════════════════════════════════════════════════════════════════

def test_infer_grouped_metric_none_when_no_question():
    assert infer_grouped_metric_intent(None, _make_df()) is None

def test_infer_grouped_metric_none_when_no_match():
    assert infer_grouped_metric_intent("hello world", _make_df()) is None

def test_infer_grouped_metric_detects_theo_pattern():
    df = pd.DataFrame({"category": ["A", "B", "A"], "amount": [100.0, 200.0, 150.0]})
    result = infer_grouped_metric_intent("tổng amount theo category", df)
    assert result is not None and result.dimension == "category"

def test_normalize_text_lowercase():
    assert normalize_text("Hello WORLD") == "hello world"

def test_build_grouped_metric_frame_aggregates():
    df = pd.DataFrame({"category": ["A", "B", "A", "C"], "amount": [100.0, 200.0, 150.0, 300.0]})
    intent = GroupedMetricIntent(metric="amount", dimension="category",
                                  metric_label="amount", dimension_label="category")
    result = build_grouped_metric_frame(df, intent)
    assert isinstance(result, pd.DataFrame) and len(result) == 3


# ═════════════════════════════════════════════════════════════════════════════
# ANALYSIS PLANNER (12–14)
# ═════════════════════════════════════════════════════════════════════════════

def test_build_fallback_plan_returns_valid_plan():
    plan = build_fallback_plan(_make_df(), "tổng amount theo category")
    assert isinstance(plan, dict) and "action" in plan

def test_execute_plan_aggregate_returns_dataframe():
    plan = {
        "action": "aggregate",
        "group_by": ["category"],
        "metrics": [{"column": "amount", "aggregation": "sum", "label": "total_amount"}],
    }
    result = execute_plan(_make_df(), plan)
    assert isinstance(result, pd.DataFrame) and not result.empty and "category" in result.columns

def test_execute_plan_profile_returns_dataframe():
    result = execute_plan(_make_df(), {"action": "profile"})
    assert isinstance(result, pd.DataFrame) and not result.empty


# ═════════════════════════════════════════════════════════════════════════════
# REPORTS (15–16)
# ═════════════════════════════════════════════════════════════════════════════

def test_write_markdown_report_creates_file():
    from backend.app.services.storage import REPORT_DIR
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    _, path = write_markdown_report("Test answer.", _make_profile(), [])
    assert isinstance(path, Path) and path.exists()

def test_write_markdown_report_returns_report_id():
    from backend.app.services.storage import REPORT_DIR
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_id, _ = write_markdown_report("Answer.", _make_profile(), [])
    assert isinstance(report_id, str) and len(report_id) > 0
