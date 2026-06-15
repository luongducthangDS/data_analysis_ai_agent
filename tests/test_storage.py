"""
Tests for SessionStore and DatasetSession — 20 test functions.
"""
from __future__ import annotations

import io
import pandas as pd
import pytest

from tests.conftest import make_csv_bytes, make_xlsx_bytes
from backend.app.services.storage import SessionStore, DatasetSession, UPLOAD_DIR


def _fresh_store() -> SessionStore:
    """Create a new SessionStore backed by the same test DB (set by conftest env vars)."""
    return SessionStore()


# ─────────────────────────────────────────────────────────────────────────────
# Create
# ─────────────────────────────────────────────────────────────────────────────

def test_create_csv_returns_session():
    store = _fresh_store()
    session = store.create("sales.csv", make_csv_bytes())
    assert isinstance(session, DatasetSession)
    assert "sales.csv" in session.filename


def test_create_stores_in_cache():
    store = _fresh_store()
    session = store.create("data.csv", make_csv_bytes())
    assert session.session_id in store._sessions


def test_create_csv_profile_is_empty_dict():
    store = _fresh_store()
    session = store.create("data.csv", make_csv_bytes())
    assert session.profile == {}


def test_create_xlsx_returns_session():
    store = _fresh_store()
    session = store.create("workbook.xlsx", make_xlsx_bytes())
    assert isinstance(session, DatasetSession)
    assert session.dataframe is not None
    assert len(session.dataframe) > 0


def test_create_multiple_csv_same_columns_concatenates():
    store = _fresh_store()
    csv = make_csv_bytes()
    session = store.create_multiple([("a.csv", csv), ("b.csv", csv)])
    # Same columns → concat with _source_sheet column
    assert "_source_sheet" in session.dataframe.columns


def test_create_multiple_csv_different_columns_picks_best():
    store = _fresh_store()
    numeric_df = pd.DataFrame({"n1": range(10), "n2": range(10), "n3": range(10)})
    text_df = pd.DataFrame({"name": ["x"] * 5})
    buf1, buf2 = io.BytesIO(), io.BytesIO()
    numeric_df.to_csv(buf1, index=False)
    text_df.to_csv(buf2, index=False)
    session = store.create_multiple([("nums.csv", buf1.getvalue()), ("text.csv", buf2.getvalue())])
    # Should pick the numeric-heavy sheet
    assert "n1" in session.dataframe.columns


def test_create_invalid_format_raises_value_error():
    store = _fresh_store()
    with pytest.raises(ValueError):
        store.create("notes.pdf", b"not a real pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Get
# ─────────────────────────────────────────────────────────────────────────────

def test_get_from_cache():
    store = _fresh_store()
    session = store.create("test.csv", make_csv_bytes())
    retrieved = store.get(session.session_id)
    assert retrieved.session_id == session.session_id


def test_get_unknown_raises_key_error():
    store = _fresh_store()
    with pytest.raises(KeyError):
        store.get("totally-nonexistent-session-id")


def test_get_from_db_after_cache_cleared():
    store = _fresh_store()
    session = store.create("restore.csv", make_csv_bytes())
    sid = session.session_id
    # Clear in-memory cache to force DB restore
    store._sessions.clear()
    restored = store.get(sid)
    assert restored.session_id == sid
    assert restored.dataframe is not None


# ─────────────────────────────────────────────────────────────────────────────
# Save / persist
# ─────────────────────────────────────────────────────────────────────────────

def test_save_persists_profile_to_db():
    store = _fresh_store()
    session = store.create("p.csv", make_csv_bytes())
    session.profile = {"rows": 5, "columns": 4}
    store.save(session)
    store._sessions.clear()
    restored = store.get(session.session_id)
    assert restored.profile == {"rows": 5, "columns": 4}


def test_save_persists_history_to_db():
    store = _fresh_store()
    session = store.create("h.csv", make_csv_bytes())
    session.history.append({"role": "user", "content": "hello"})
    session.history.append({"role": "assistant", "content": "hi"})
    store.save(session)
    store._sessions.clear()
    restored = store.get(session.session_id)
    assert len(restored.history) == 2
    assert restored.history[0]["role"] == "user"
    assert restored.history[1]["content"] == "hi"


def test_save_persists_report_id_to_db():
    store = _fresh_store()
    session = store.create("r.csv", make_csv_bytes())
    session.report_id = "abc123report"
    store.save(session)
    store._sessions.clear()
    restored = store.get(session.session_id)
    assert restored.report_id == "abc123report"


def test_count_returns_db_count():
    store = _fresh_store()
    before = store.count()
    store.create("c1.csv", make_csv_bytes())
    store.create("c2.csv", make_csv_bytes())
    assert store.count() >= before + 2


# ─────────────────────────────────────────────────────────────────────────────
# _build_analysis_dataframe
# ─────────────────────────────────────────────────────────────────────────────

def test_build_analysis_df_single_sheet():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    result = SessionStore._build_analysis_dataframe({"only": df})
    assert list(result.columns) == ["a", "b"]
    assert len(result) == 2


def test_build_analysis_df_identical_columns_concat():
    df1 = pd.DataFrame({"x": [1], "y": [2]})
    df2 = pd.DataFrame({"x": [3], "y": [4]})
    result = SessionStore._build_analysis_dataframe({"s1": df1, "s2": df2})
    assert "_source_sheet" in result.columns
    assert len(result) == 2


def test_build_analysis_df_best_score_selected():
    numeric = pd.DataFrame({f"n{i}": range(5) for i in range(6)})
    textonly = pd.DataFrame({"name": ["a"] * 5})
    result = SessionStore._build_analysis_dataframe({"num": numeric, "txt": textonly})
    assert "n0" in result.columns


# ─────────────────────────────────────────────────────────────────────────────
# _analysis_score
# ─────────────────────────────────────────────────────────────────────────────

def test_analysis_score_numeric_cols_boost_score():
    many_numeric = pd.DataFrame({f"n{i}": range(10) for i in range(5)})
    few_numeric = pd.DataFrame({"n1": range(10), "cat": ["a"] * 10})
    assert SessionStore._analysis_score(many_numeric) > SessionStore._analysis_score(few_numeric)


# ─────────────────────────────────────────────────────────────────────────────
# _coerce_datetime_columns
# ─────────────────────────────────────────────────────────────────────────────

def test_coerce_datetime_date_column_converts():
    df = pd.DataFrame({"date": ["2024-01-01", "2024-02-15", "2024-03-20"]})
    result = SessionStore._coerce_datetime_columns(df)
    assert pd.api.types.is_datetime64_any_dtype(result["date"])


def test_coerce_datetime_text_column_stays():
    df = pd.DataFrame({"name": ["Alice", "Bob", "Charlie"]})
    result = SessionStore._coerce_datetime_columns(df)
    assert result["name"].dtype == object
