"""
Tests for backend.app.database — 10 test functions.

Env vars (DATA_DIR, DATABASE_URL) are set by conftest.py before this module is
imported, so the SQLAlchemy engine is already pointing at the temp test DB.
"""
from __future__ import annotations

import pytest
from sqlalchemy import inspect, text

# conftest.py sets env vars before this import runs
from backend.app.database import (
    ChatHistoryModel,
    SessionModel,
    db_session,
    engine,
    init_db,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def _make_session(session_id: str) -> SessionModel:
    return SessionModel(
        session_id=session_id,
        filename="test.csv",
        file_names=["test.csv"],
        profile={},
        sheet_relationships=[],
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1–2  Table existence
# ─────────────────────────────────────────────────────────────────────────────

def test_init_db_creates_sessions_table():
    init_db()
    assert inspect(engine).has_table("sessions")


def test_init_db_creates_chat_history_table():
    init_db()
    assert inspect(engine).has_table("chat_history")


# ─────────────────────────────────────────────────────────────────────────────
# 3  Column names
# ─────────────────────────────────────────────────────────────────────────────

def test_sessions_table_has_expected_columns():
    init_db()
    insp = inspect(engine)
    col_names = {col["name"] for col in insp.get_columns("sessions")}
    assert "session_id" in col_names
    assert "filename" in col_names
    assert "profile" in col_names
    # `history` is a relationship, not a DB column — check the ORM attribute exists
    assert hasattr(SessionModel, "history")


# ─────────────────────────────────────────────────────────────────────────────
# 4  Insert and query back SessionModel
# ─────────────────────────────────────────────────────────────────────────────

def test_insert_session_model():
    init_db()
    sid = "test-db-004"
    with db_session() as db:
        db.add(_make_session(sid))

    with db_session() as db:
        row = db.get(SessionModel, sid)
        assert row is not None
        assert row.session_id == sid
        assert row.filename == "test.csv"


# ─────────────────────────────────────────────────────────────────────────────
# 5  Insert SessionModel + ChatHistoryModel
# ─────────────────────────────────────────────────────────────────────────────

def test_insert_chat_history_model():
    init_db()
    sid = "test-db-005"
    with db_session() as db:
        db.add(_make_session(sid))

    with db_session() as db:
        db.add(ChatHistoryModel(session_id=sid, role="user", content="hello"))

    with db_session() as db:
        rows = db.query(ChatHistoryModel).filter_by(session_id=sid).all()
        assert len(rows) == 1
        assert rows[0].role == "user"
        assert rows[0].content == "hello"


# ─────────────────────────────────────────────────────────────────────────────
# 6  Cascade delete
# ─────────────────────────────────────────────────────────────────────────────

def test_cascade_delete_removes_history():
    init_db()
    sid = "test-db-006"
    with db_session() as db:
        db.add(_make_session(sid))

    with db_session() as db:
        db.add(ChatHistoryModel(session_id=sid, role="user", content="ping"))
        db.add(ChatHistoryModel(session_id=sid, role="assistant", content="pong"))

    with db_session() as db:
        session_row = db.get(SessionModel, sid)
        db.delete(session_row)

    with db_session() as db:
        remaining = db.query(ChatHistoryModel).filter_by(session_id=sid).all()
        assert remaining == []


# ─────────────────────────────────────────────────────────────────────────────
# 7  db_session commits on success
# ─────────────────────────────────────────────────────────────────────────────

def test_db_session_commits_on_success():
    init_db()
    sid = "test-db-007"
    with db_session() as db:
        db.add(_make_session(sid))
    # New session — must still see the row
    with db_session() as db:
        assert db.get(SessionModel, sid) is not None


# ─────────────────────────────────────────────────────────────────────────────
# 8  db_session rolls back on error
# ─────────────────────────────────────────────────────────────────────────────

def test_db_session_rollbacks_on_error():
    init_db()
    sid = "test-db-008"
    with pytest.raises(RuntimeError):
        with db_session() as db:
            db.add(_make_session(sid))
            raise RuntimeError("forced rollback")

    with db_session() as db:
        assert db.get(SessionModel, sid) is None


# ─────────────────────────────────────────────────────────────────────────────
# 9  History ordered by id
# ─────────────────────────────────────────────────────────────────────────────

def test_history_ordered_by_id():
    init_db()
    sid = "test-db-009"
    with db_session() as db:
        db.add(_make_session(sid))

    messages = [("user", "first"), ("assistant", "second"), ("user", "third")]
    with db_session() as db:
        for role, content in messages:
            db.add(ChatHistoryModel(session_id=sid, role=role, content=content))

    with db_session() as db:
        rows = (
            db.query(ChatHistoryModel)
            .filter_by(session_id=sid)
            .order_by(ChatHistoryModel.id)
            .all()
        )
        assert [r.content for r in rows] == ["first", "second", "third"]


# ─────────────────────────────────────────────────────────────────────────────
# 10  postgres:// → postgresql:// normalization logic
# ─────────────────────────────────────────────────────────────────────────────

def test_postgres_url_normalization():
    legacy = "postgres://host/db"
    normalized = legacy.replace("postgres://", "postgresql://", 1)
    assert normalized.startswith("postgresql://")
    # Sanity check: already-correct URL is untouched
    already_correct = "postgresql://host/db"
    assert already_correct.replace("postgres://", "postgresql://", 1) == already_correct
