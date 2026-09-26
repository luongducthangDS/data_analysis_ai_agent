"""Regression checks for docs/BACKEND_FEEDBACK_2026-09-26.md items 5–12."""
from __future__ import annotations

import pytest
from sqlalchemy import inspect, text

from backend.app.database import engine, init_db
from backend.app.services.storage import UPLOAD_DIR, SessionStore
from tests.conftest import make_csv_bytes


def test_two_workers_appending_to_one_session_lose_nothing():
    """Two SessionStore instances = two uvicorn workers with their own RAM cache."""
    worker_a, worker_b = SessionStore(), SessionStore()
    sid = worker_a.create("w.csv", make_csv_bytes()).session_id
    session_a, session_b = worker_a.get(sid), worker_b.get(sid)

    worker_a.append_messages(session_a, [{"role": "user", "content": "from A"}])
    worker_b.append_messages(session_b, [{"role": "user", "content": "from B"}])
    worker_a.save(session_a)   # save() used to rewrite the whole history blob

    assert [m["content"] for m in worker_a.recent_history(sid)] == ["from A", "from B"]


def test_init_db_adds_a_column_missing_from_an_existing_table():
    with engine.begin() as conn:
        conn.execute(text('ALTER TABLE chat_history DROP COLUMN "source"'))
    init_db()
    assert "source" in {c["name"] for c in inspect(engine).get_columns("chat_history")}


def test_failed_db_insert_does_not_leave_a_ram_only_session(mocker):
    store = SessionStore()
    mocker.patch.object(store, "_insert_session", side_effect=RuntimeError("db down"))
    before = set(UPLOAD_DIR.iterdir())
    with pytest.raises(RuntimeError):
        store.create("x.csv", make_csv_bytes())
    assert store._sessions == {}
    assert set(UPLOAD_DIR.iterdir()) == before


def test_upload_db_failure_is_503_without_internals(client, mocker):
    from sqlalchemy.exc import OperationalError
    mocker.patch("backend.app.services.storage.SessionStore._insert_session",
                 side_effect=OperationalError("INSERT secret-host:5432", {}, Exception()))
    resp = client.post("/api/upload", files=[("files", ("d.csv", make_csv_bytes(), "text/csv"))])
    assert resp.status_code == 503
    assert "secret-host" not in resp.text


def test_unknown_api_path_is_json_404_not_the_spa(client):
    resp = client.get("/api/no-such-endpoint")
    assert resp.status_code == 404
    assert resp.headers["content-type"].startswith("application/json")
