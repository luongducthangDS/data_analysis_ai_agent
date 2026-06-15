"""
Tests for guardrails and the SELECT-only query enforcement — 10 test functions.
"""
from __future__ import annotations

import pandas as pd
import pytest

# conftest.py sets env vars before these imports run
from backend.app.services.query_engine import run_readonly_query
from backend.app.services.guardrails import assert_allowed_tool, describe_guardrails


# ─────────────────────────────────────────────────────────────────────────────
# Shared fixture (defined inline to avoid ordering issues)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def simple_df() -> pd.DataFrame:
    return pd.DataFrame({"id": [1, 2, 3], "val": [10, 20, 30]})


# ─────────────────────────────────────────────────────────────────────────────
# SQL guardrail tests (1–8)
# ─────────────────────────────────────────────────────────────────────────────

def test_select_is_allowed(simple_df):
    rows = run_readonly_query(simple_df, "SELECT * FROM dataset LIMIT 1")
    assert isinstance(rows, list)
    assert len(rows) == 1


def test_drop_table_raises(simple_df):
    with pytest.raises(ValueError):
        run_readonly_query(simple_df, "DROP TABLE dataset")


def test_delete_raises(simple_df):
    with pytest.raises(ValueError):
        run_readonly_query(simple_df, "DELETE FROM dataset WHERE 1=1")


def test_update_raises(simple_df):
    with pytest.raises(ValueError):
        run_readonly_query(simple_df, "UPDATE dataset SET val=0")


def test_insert_raises(simple_df):
    with pytest.raises(ValueError):
        run_readonly_query(simple_df, "INSERT INTO dataset VALUES (1,2,3,4)")


def test_create_table_raises(simple_df):
    with pytest.raises(ValueError):
        run_readonly_query(simple_df, "CREATE TABLE foo (id INT)")


def test_alter_table_raises(simple_df):
    with pytest.raises(ValueError):
        run_readonly_query(simple_df, "ALTER TABLE dataset ADD COLUMN x INT")


def test_truncate_raises(simple_df):
    with pytest.raises(ValueError):
        run_readonly_query(simple_df, "TRUNCATE TABLE dataset")


# ─────────────────────────────────────────────────────────────────────────────
# describe_guardrails / assert_allowed_tool tests (9–10)
# ─────────────────────────────────────────────────────────────────────────────

def test_describe_guardrails_returns_list_of_strings():
    result = describe_guardrails()
    assert isinstance(result, list)
    assert len(result) >= 1
    for item in result:
        assert isinstance(item, str), f"Non-string item in guardrails list: {item!r}"


def test_assert_allowed_tool_raises_for_unknown():
    with pytest.raises(ValueError):
        assert_allowed_tool("hack_system")
