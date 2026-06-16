"""
Tests for guardrails — 2 test functions.

SQL guardrail tests removed: query_engine.py was deleted (SQL enforcement
now happens inside DuckDB via analysis_planner, not a standalone module).
"""
from __future__ import annotations

from backend.app.services.guardrails import assert_allowed_tool, describe_guardrails


def test_describe_guardrails_returns_list_of_strings():
    result = describe_guardrails()
    assert isinstance(result, list) and len(result) >= 1
    assert all(isinstance(item, str) for item in result)


def test_assert_allowed_tool_raises_for_unknown():
    import pytest
    with pytest.raises(ValueError):
        assert_allowed_tool("hack_system")
