"""
Tests for backend service layer — 30 test functions.

Coverage:
  profiler (6), query_engine (6), analysis_intent (5),
  agent_tools (7), charts (3), reports (2), agent_runner (1)
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# conftest.py sets env vars before these imports run
from backend.app.services.profiler import build_profile
from backend.app.services.query_engine import run_readonly_query, simple_question_to_sql
from backend.app.services.analysis_intent import (
    GroupedMetricIntent,
    build_grouped_metric_frame,
    infer_grouped_metric_intent,
    normalize_text,
)
from backend.app.services.agent_tools import ToolResult, execute_tool
from backend.app.services.charts import generate_question_charts, generate_recommended_charts
from backend.app.services.reports import write_markdown_report
from backend.app.services.agent_runner import _schema_summary


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


def _make_profile(df: pd.DataFrame | None = None) -> dict[str, Any]:
    return build_profile(df if df is not None else _make_df())


# ═════════════════════════════════════════════════════════════════════════════
# PROFILER TESTS (1–6)
# ═════════════════════════════════════════════════════════════════════════════

def test_build_profile_has_required_keys():
    profile = build_profile(_make_df())
    for key in ("rows", "columns", "column_types", "missing_values", "numeric_summary", "categorical_summary"):
        assert key in profile, f"Missing key: {key}"


def test_build_profile_row_count():
    df = _make_df()
    profile = build_profile(df)
    assert profile["rows"] == 5


def test_build_profile_column_count():
    df = _make_df()
    profile = build_profile(df)
    assert profile["columns"] == 4


def test_build_profile_missing_values_counted():
    df = pd.DataFrame({"x": [1.0, None, 3.0], "y": ["a", "b", "c"]})
    profile = build_profile(df)
    assert profile["missing_values"]["x"] == 1


def test_build_profile_numeric_summary_has_stats():
    df = pd.DataFrame({"value": [10.0, 20.0, 30.0]})
    profile = build_profile(df)
    stats = profile["numeric_summary"]["value"]
    assert "mean" in stats
    assert "min" in stats
    assert "max" in stats


def test_build_profile_categorical_summary_top_values():
    df = pd.DataFrame({"cat": ["apple", "banana", "apple", "cherry", "apple"]})
    profile = build_profile(df)
    top = profile["categorical_summary"]["cat"]
    assert isinstance(top, list)
    assert len(top) >= 1
    assert "value" in top[0]
    assert "count" in top[0]


# ═════════════════════════════════════════════════════════════════════════════
# QUERY ENGINE TESTS (7–12)
# ═════════════════════════════════════════════════════════════════════════════

def test_run_readonly_query_basic_select():
    df = _make_df()
    rows = run_readonly_query(df, "SELECT * FROM dataset LIMIT 3")
    assert isinstance(rows, list)
    assert len(rows) == 3


def test_run_readonly_query_aggregation():
    df = _make_df()
    rows = run_readonly_query(df, "SELECT SUM(amount) as total FROM dataset")
    assert len(rows) == 1
    assert rows[0]["total"] > 0


def test_run_readonly_query_rejects_drop():
    df = _make_df()
    with pytest.raises(ValueError):
        run_readonly_query(df, "DROP TABLE dataset")


def test_run_readonly_query_rejects_delete():
    df = _make_df()
    with pytest.raises(ValueError):
        run_readonly_query(df, "DELETE FROM dataset")


def test_run_readonly_query_rejects_insert():
    df = _make_df()
    with pytest.raises(ValueError):
        run_readonly_query(df, "INSERT INTO dataset VALUES (1, 'x', 99, 1)")


def test_simple_question_to_sql_returns_string():
    df = _make_df()
    sql = simple_question_to_sql("how many rows", df)
    assert isinstance(sql, str)
    assert sql.strip().upper().startswith("SELECT")


# ═════════════════════════════════════════════════════════════════════════════
# ANALYSIS INTENT TESTS (13–17)
# ═════════════════════════════════════════════════════════════════════════════

def test_infer_grouped_metric_none_when_no_question():
    df = _make_df()
    result = infer_grouped_metric_intent(None, df)
    assert result is None


def test_infer_grouped_metric_none_when_no_match():
    df = _make_df()
    result = infer_grouped_metric_intent("hello world", df)
    assert result is None


def test_infer_grouped_metric_detects_theo_pattern():
    df = pd.DataFrame({
        "category": ["A", "B", "A"],
        "amount": [100.0, 200.0, 150.0],
    })
    result = infer_grouped_metric_intent("tổng amount theo category", df)
    assert result is not None
    assert isinstance(result, GroupedMetricIntent)
    assert result.dimension == "category"


def test_normalize_text_lowercase():
    result = normalize_text("Hello WORLD")
    assert result == "hello world"


def test_build_grouped_metric_frame_aggregates():
    df = pd.DataFrame({
        "category": ["A", "B", "A", "C"],
        "amount": [100.0, 200.0, 150.0, 300.0],
    })
    intent = GroupedMetricIntent(
        metric="amount",
        dimension="category",
        metric_label="amount",
        dimension_label="category",
    )
    result = build_grouped_metric_frame(df, intent)
    assert isinstance(result, pd.DataFrame)
    assert "category" in result.columns
    assert "amount" in result.columns
    # A: 100+150=250, B: 200, C: 300 → 3 rows
    assert len(result) == 3


# ═════════════════════════════════════════════════════════════════════════════
# AGENT TOOLS TESTS (18–24)
# ═════════════════════════════════════════════════════════════════════════════

def test_execute_tool_get_profile_returns_summary():
    df = _make_df()
    profile = _make_profile(df)
    result = execute_tool(df, profile, "get_profile", {})
    assert isinstance(result, ToolResult)
    assert isinstance(result.summary, str)
    assert len(result.summary) > 0


def test_execute_tool_query_sql_valid():
    df = _make_df()
    profile = _make_profile(df)
    result = execute_tool(df, profile, "query_sql", {"sql": "SELECT COUNT(*) as n FROM dataset"})
    assert isinstance(result, ToolResult)
    assert isinstance(result.summary, str)
    assert len(result.summary) > 0


def test_execute_tool_query_sql_invalid():
    df = _make_df()
    profile = _make_profile(df)
    # execute_tool NEVER raises — errors surface in summary
    result = execute_tool(df, profile, "query_sql", {"sql": "DROP TABLE dataset"})
    assert isinstance(result, ToolResult)
    assert len(result.summary) > 0  # non-empty error message


def test_execute_tool_analyze_data_returns_result():
    df = _make_df()
    profile = _make_profile(df)
    arguments = {
        "action": "aggregate",
        "group_by": ["category"],
        "metrics": [{"column": "amount", "aggregation": "sum"}],
    }
    result = execute_tool(df, profile, "analyze_data", arguments)
    assert isinstance(result, ToolResult)
    assert isinstance(result.summary, str)


def test_execute_tool_generate_chart_returns_charts_or_empty():
    df = _make_df()
    profile = _make_profile(df)
    result = execute_tool(df, profile, "generate_chart", {"question": "amount by category"})
    assert isinstance(result, ToolResult)
    assert isinstance(result.charts, list)


def test_execute_tool_unknown_tool_returns_error_summary():
    df = _make_df()
    profile = _make_profile(df)
    result = execute_tool(df, profile, "nonexistent", {})
    assert isinstance(result, ToolResult)
    assert len(result.summary) > 0
    assert "nonexistent" in result.summary or "Unknown" in result.summary


def test_tool_result_summary_is_string():
    df = _make_df()
    profile = _make_profile(df)
    tool_calls = [
        ("get_profile", {}),
        ("query_sql", {"sql": "SELECT * FROM dataset LIMIT 1"}),
        ("generate_chart", {"question": "amount"}),
        ("analyze_data", {"action": "profile"}),
    ]
    for tool_name, arguments in tool_calls:
        result = execute_tool(df, profile, tool_name, arguments)
        assert isinstance(result.summary, str), f"Non-string summary for tool: {tool_name}"


# ═════════════════════════════════════════════════════════════════════════════
# CHARTS TESTS (25–27)
# ═════════════════════════════════════════════════════════════════════════════

def test_generate_recommended_charts_returns_list():
    df = _make_df()
    charts = generate_recommended_charts(df)
    assert isinstance(charts, list)


def test_generate_question_charts_returns_list():
    df = _make_df()
    charts = generate_question_charts(df, "amount theo category")
    assert isinstance(charts, list)


def test_chart_spec_has_chart_id_and_title():
    df = pd.DataFrame({
        "category": ["A", "B", "C", "D"],
        "amount": [100.0, 200.0, 300.0, 400.0],
    })
    charts = generate_recommended_charts(df)
    if charts:
        first = charts[0]
        assert "chart_id" in first
        assert "title" in first
        assert isinstance(first["chart_id"], str)
        assert isinstance(first["title"], str)


# ═════════════════════════════════════════════════════════════════════════════
# REPORTS TESTS (28–29)
# ═════════════════════════════════════════════════════════════════════════════

def test_write_markdown_report_creates_file():
    # Ensure REPORT_DIR exists (storage.py creates it on SessionStore init,
    # but we can call it directly here too)
    from backend.app.services.storage import REPORT_DIR
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    profile = _make_profile()
    _, path = write_markdown_report("Test answer.", profile, [])
    assert isinstance(path, Path)
    assert path.exists()


def test_write_markdown_report_returns_report_id():
    from backend.app.services.storage import REPORT_DIR
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    profile = _make_profile()
    report_id, _ = write_markdown_report("Another answer.", profile, [])
    assert isinstance(report_id, str)
    assert len(report_id) > 0


# ═════════════════════════════════════════════════════════════════════════════
# AGENT RUNNER TESTS (30)
# ═════════════════════════════════════════════════════════════════════════════

def test_schema_summary_format():
    profile = {
        "rows": 100,
        "columns": 4,
        "column_types": {"amount": "float64", "category": "object"},
        "missing_values": {"amount": 0, "category": 2},
    }
    summary = _schema_summary(profile)
    assert isinstance(summary, str)
    assert summary.startswith("Rows:")
    assert "amount" in summary
    assert "category" in summary
