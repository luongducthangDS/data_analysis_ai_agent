"""
Tests for multi-file / multi-sheet handling: active-sheet switching, merge,
cross-sheet source resolution, and restore persistence.
"""
from __future__ import annotations

import io

import pandas as pd

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
