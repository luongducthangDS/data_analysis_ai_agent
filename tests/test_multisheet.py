"""
Tests for multi-file / multi-sheet handling: active-sheet switching, merge,
cross-sheet source resolution, and restore persistence.
"""
from __future__ import annotations

import io

import pandas as pd
import pytest

from backend.app.services.storage import (
    SessionStore,
    build_source_frame,
    resolve_sheet_key,
)


def _fresh_store() -> SessionStore:
    return SessionStore()


def _two_sheet_xlsx(diff_schema: bool = True) -> bytes:
    """Workbook with Orders + Items sharing order_id (joinable)."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        pd.DataFrame({"order_id": [1, 2, 3], "revenue": [100, 200, 300]}).to_excel(
            w, sheet_name="Orders", index=False
        )
        items = (
            {"order_id": [1, 2, 3], "product": ["A", "B", "A"]}
            if diff_schema
            else {"order_id": [4, 5, 6], "revenue": [10, 20, 30]}
        )
        pd.DataFrame(items).to_excel(w, sheet_name="Items", index=False)
    return buf.getvalue()


# ── Ingestion + active sheet ──────────────────────────────────────────────────

def test_multisheet_reads_all_sheets():
    store = _fresh_store()
    sess = store.create_multiple([("shop.xlsx", _two_sheet_xlsx())])
    assert set(sess.sheets.keys()) == {"shop::Orders", "shop::Items"}


def test_active_sheet_set_on_create():
    store = _fresh_store()
    sess = store.create_multiple([("shop.xlsx", _two_sheet_xlsx())])
    assert sess.active_sheet in {"shop::Orders", "shop::Items"}


def test_set_active_sheet_switches_dataframe():
    store = _fresh_store()
    sess = store.create_multiple([("shop.xlsx", _two_sheet_xlsx())])
    store.set_active_sheet(sess, "Items")
    assert sess.active_sheet == "shop::Items"
    assert "product" in sess.dataframe.columns
    assert sess.profile  # re-profiled


def test_set_active_sheet_unknown_raises():
    store = _fresh_store()
    sess = store.create_multiple([("shop.xlsx", _two_sheet_xlsx())])
    try:
        store.set_active_sheet(sess, "Nonexistent")
        assert False, "should have raised"
    except ValueError:
        pass


def test_resolve_sheet_key_by_short_name():
    sheets = {"shop::Orders": pd.DataFrame(), "shop::Items": pd.DataFrame()}
    assert resolve_sheet_key("Orders", sheets) == "shop::Orders"
    assert resolve_sheet_key("shop::Items", sheets) == "shop::Items"


def test_active_sheet_persists_across_restore():
    store = _fresh_store()
    sess = store.create_multiple([("shop.xlsx", _two_sheet_xlsx())])
    store.set_active_sheet(sess, "Items")
    sid = sess.session_id
    # Drop from cache → force restore from DB + disk
    store._sessions.pop(sid, None)
    store._last_accessed.pop(sid, None)
    restored = store.get(sid)
    assert restored.active_sheet == "shop::Items"
    assert "product" in restored.dataframe.columns


# ── Cross-sheet source resolver ───────────────────────────────────────────────

def test_build_source_frame_none_returns_active():
    store = _fresh_store()
    sess = store.create_multiple([("shop.xlsx", _two_sheet_xlsx())])
    frame, warn = build_source_frame(sess, None)
    assert frame is sess.dataframe
    assert warn is None


def test_build_source_frame_join():
    store = _fresh_store()
    sess = store.create_multiple([("shop.xlsx", _two_sheet_xlsx())])
    src = {"join": {"base": "Orders", "with": "Items", "on": "order_id", "how": "left"}}
    frame, _ = build_source_frame(sess, src)
    assert {"revenue", "product"} <= set(frame.columns)
    assert len(frame) == 3


def test_build_source_frame_single_sheet():
    store = _fresh_store()
    sess = store.create_multiple([("shop.xlsx", _two_sheet_xlsx())])
    frame, _ = build_source_frame(sess, {"sheet": "Items"})
    assert "product" in frame.columns


def test_build_source_frame_bad_join_falls_back():
    store = _fresh_store()
    sess = store.create_multiple([("shop.xlsx", _two_sheet_xlsx())])
    frame, _ = build_source_frame(sess, {"join": {"base": "Orders", "with": "Ghost", "on": "x"}})
    assert list(frame.columns) == list(sess.dataframe.columns)


# ── Sheet gộp sống sót qua restart; mất thì từ chối, không đoán (P0 integrity) ──

@pytest.fixture
def offline(monkeypatch):
    from backend.app.agents.nodes import respond

    def boom():
        raise RuntimeError("LLM disabled in test")

    monkeypatch.setattr("backend.app.services.llm_service.get_llm_client", boom)
    monkeypatch.setattr(respond, "get_llm_client", boom)


def _merged_session(client) -> tuple[str, pd.DataFrame]:
    from backend.app.services.storage import session_store
    sid = client.post("/api/upload", files=[("files", ("wb.xlsx", _two_sheet_xlsx(), "application/octet-stream"))]).json()["session_id"]
    resp = client.post("/api/merge-sheets", json={"session_id": sid, "sheet_names": ["Orders", "Items"]})
    assert resp.status_code == 200
    return sid, session_store.get(sid).dataframe.copy()


def _restart(sid: str):
    from backend.app.services.storage import session_store
    session_store._forget(sid)
    return session_store.get(sid)


def _set_persisted_active(sid: str, name: str | None, clear_recipes: bool = False) -> None:
    from backend.app.database import SessionModel, db_session
    with db_session() as db:
        row = db.get(SessionModel, sid)
        row.active_sheet = name
        if clear_recipes:
            row.merged_sheets = None


def test_merged_sheet_is_rebuilt_after_restart(client):
    sid, before = _merged_session(client)
    after = _restart(sid)
    assert after.active_sheet == "Orders + Items"
    assert not after.active_sheet_lost
    pd.testing.assert_frame_equal(after.dataframe.reset_index(drop=True), before.reset_index(drop=True))


def test_unrecoverable_merged_sheet_refuses_instead_of_answering_on_another_sheet(client, offline):
    from backend.app.agents.runner import run
    sid, _ = _merged_session(client)
    _set_persisted_active(sid, "Orders + Items", clear_recipes=True)   # phiên cũ, gộp trước khi có recipe
    session = _restart(sid)
    assert session.active_sheet_lost

    out = run(sid, "tổng revenue", [])
    assert out.source == "deterministic"
    assert out.executed_queries == ["[sheet_lost]"]
    assert "Orders + Items" in out.answer and "Gộp" in out.answer
    assert client.get(f"/api/dashboard/{sid}").status_code == 409


def test_lost_flag_survives_the_answer_being_saved(client, offline):
    """Lượt chat bị từ chối vẫn gọi save(); không được ghi đè tên sheet mất bằng sheet tự chọn."""
    from backend.app.api.routes.chat import _persist_and_report
    sid, _ = _merged_session(client)
    _set_persisted_active(sid, "Orders + Items", clear_recipes=True)
    session = _restart(sid)
    _persist_and_report(session, "q", "a", [], "deterministic")
    assert _restart(sid).active_sheet_lost


def test_choosing_a_sheet_clears_the_refusal(client, offline):
    from backend.app.agents.runner import run
    sid, _ = _merged_session(client)
    _set_persisted_active(sid, "Orders + Items", clear_recipes=True)
    _restart(sid)
    assert client.post(f"/api/session/{sid}/active-sheet", json={"sheet_name": "Orders"}).status_code == 200
    assert run(sid, "tổng revenue", []).executed_queries != ["[sheet_lost]"]


def test_concat_label_is_not_a_lost_sheet():
    store = _fresh_store()
    session = store.create("wb.xlsx", _two_sheet_xlsx(diff_schema=False))
    assert session.active_sheet == "__concat__"
    store._forget(session.session_id)
    assert not store.get(session.session_id).active_sheet_lost
