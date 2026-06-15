"""
Tests for backend.app.services.workspace_connectors — 10 test functions.

Tests cover URL normalisation, sheet ID extraction, credential guard,
and HTTP fetch behaviour using mocked requests.
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

# conftest.py sets env vars before these imports run
from backend.app.services.workspace_connectors import (
    _extract_sheet_id,
    _normalize_gsheet_url,
    fetch_from_gsheet,
    fetch_from_url,
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_SHEET_ID = "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgVE2upms"
_EDIT_URL = f"https://docs.google.com/spreadsheets/d/{_SHEET_ID}/edit#gid=0"
_PUB_URL = f"https://docs.google.com/spreadsheets/d/{_SHEET_ID}/pub?gid=12345&single=true"
_EXPORT_URL = f"https://docs.google.com/spreadsheets/d/{_SHEET_ID}/export?format=csv"


# ─────────────────────────────────────────────────────────────────────────────
# 1–4  _normalize_gsheet_url
# ─────────────────────────────────────────────────────────────────────────────

def test_normalize_edit_url_to_export():
    result = _normalize_gsheet_url(_EDIT_URL)
    assert "export?format=csv" in result
    assert _SHEET_ID in result


def test_normalize_pub_url_to_export():
    result = _normalize_gsheet_url(_PUB_URL)
    assert "export?format=csv" in result
    assert _SHEET_ID in result
    # gid should be preserved
    assert "gid=12345" in result


def test_normalize_already_export_url_unchanged():
    result = _normalize_gsheet_url(_EXPORT_URL)
    # Already has export?format=csv — the function still re-normalises to the
    # canonical export URL because it contains /spreadsheets/d/{id}
    assert "export?format=csv" in result
    assert _SHEET_ID in result


def test_normalize_returns_url_unchanged_for_non_gsheet():
    url = "https://example.com/file.csv"
    result = _normalize_gsheet_url(url)
    # Non-GSheets URL: no /spreadsheets/d/ pattern → returned as-is
    assert result == url


# ─────────────────────────────────────────────────────────────────────────────
# 5–7  _extract_sheet_id
# ─────────────────────────────────────────────────────────────────────────────

def test_extract_sheet_id_from_url():
    result = _extract_sheet_id(_EDIT_URL)
    assert result == _SHEET_ID


def test_extract_sheet_id_raw_id_passthrough():
    # Raw base64url ID (≥20 chars) → returned as-is
    raw_id = _SHEET_ID
    result = _extract_sheet_id(raw_id)
    assert result == raw_id


def test_extract_sheet_id_invalid_raises():
    # Too short (< 20 chars) and not a URL → ValueError
    with pytest.raises(ValueError):
        _extract_sheet_id("not-an-id")


# ─────────────────────────────────────────────────────────────────────────────
# 8  fetch_from_gsheet — no credentials
# ─────────────────────────────────────────────────────────────────────────────

def test_fetch_from_gsheet_no_credentials_raises(monkeypatch):
    monkeypatch.delenv("GOOGLE_CREDENTIALS_JSON", raising=False)
    with pytest.raises(ValueError, match="GOOGLE_CREDENTIALS_JSON"):
        fetch_from_gsheet(_EDIT_URL)


# ─────────────────────────────────────────────────────────────────────────────
# 9  fetch_from_url — successful CSV download via mocked requests.get
# ─────────────────────────────────────────────────────────────────────────────

def test_fetch_from_url_csv_success():
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.content = b"id,val\n1,100\n2,200\n"
    mock_resp.headers = {
        "content-type": "text/csv",
        "content-disposition": 'attachment; filename="data.csv"',
    }

    with patch("backend.app.services.workspace_connectors.requests.get", return_value=mock_resp):
        filename, content = fetch_from_url("https://example.com/data.csv")

    assert isinstance(filename, str)
    assert len(filename) > 0
    assert isinstance(content, bytes)
    assert content == b"id,val\n1,100\n2,200\n"


# ─────────────────────────────────────────────────────────────────────────────
# 10  fetch_from_url — unknown content-type falls back to import.csv filename
#     (the function does NOT raise for unknown formats — it infers a filename)
# ─────────────────────────────────────────────────────────────────────────────

def test_fetch_from_url_unknown_content_type_returns_fallback_filename():
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.content = b"\x00\x01\x02binary garbage"
    # No content-disposition, no recognisable content-type
    mock_resp.headers = {"content-type": "application/octet-stream"}

    with patch("backend.app.services.workspace_connectors.requests.get", return_value=mock_resp):
        # URL has no recognisable extension either
        filename, content = fetch_from_url("https://example.com/unknownfile")

    # Should fall back gracefully — filename is a non-empty string
    assert isinstance(filename, str)
    assert len(filename) > 0
    assert isinstance(content, bytes)
