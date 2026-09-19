"""
Tests for backend.app.services.workspace_connectors — URL import only.

Covers Google-Sheets URL normalisation and the HTTP fetch behaviour
(mocked requests). The service-account import path was removed.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from backend.app.services.workspace_connectors import (
    _normalize_gsheet_url,
    fetch_from_url,
)

_SHEET_ID = "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgVE2upms"
_EDIT_URL = f"https://docs.google.com/spreadsheets/d/{_SHEET_ID}/edit#gid=0"
_PUB_URL = f"https://docs.google.com/spreadsheets/d/{_SHEET_ID}/pub?gid=12345&single=true"
_EXPORT_URL = f"https://docs.google.com/spreadsheets/d/{_SHEET_ID}/export?format=csv"


# ── _normalize_gsheet_url ────────────────────────────────────────────────────

def test_normalize_edit_url_to_export():
    result = _normalize_gsheet_url(_EDIT_URL)
    assert "export?format=csv" in result
    assert _SHEET_ID in result


def test_normalize_pub_url_preserves_gid():
    result = _normalize_gsheet_url(_PUB_URL)
    assert "export?format=csv" in result
    assert "gid=12345" in result


def test_normalize_already_export_url():
    result = _normalize_gsheet_url(_EXPORT_URL)
    assert "export?format=csv" in result
    assert _SHEET_ID in result


def test_normalize_non_gsheet_url_unchanged():
    url = "https://example.com/file.csv"
    assert _normalize_gsheet_url(url) == url


# ── fetch_from_url ───────────────────────────────────────────────────────────

def test_fetch_from_url_csv_success():
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.content = b"id,val\n1,100\n2,200\n"
    mock_resp.headers = {
        "content-type": "text/csv",
        "content-disposition": 'attachment; filename="data.csv"',
    }
    with patch("backend.app.services.workspace_connectors.safe_fetch", return_value=mock_resp):
        filename, content = fetch_from_url("https://example.com/data.csv")
    assert filename == "data.csv"
    assert content == b"id,val\n1,100\n2,200\n"


def test_fetch_from_url_unknown_content_type_falls_back_filename():
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.content = b"\x00\x01\x02binary garbage"
    mock_resp.headers = {"content-type": "application/octet-stream"}
    with patch("backend.app.services.workspace_connectors.safe_fetch", return_value=mock_resp):
        filename, content = fetch_from_url("https://example.com/unknownfile")
    assert isinstance(filename, str) and filename
    assert isinstance(content, bytes)


def test_fetch_from_url_gsheet_edit_url_is_normalised():
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.content = b"a,b\n1,2\n"
    mock_resp.headers = {"content-type": "text/csv"}
    with patch("backend.app.services.workspace_connectors.safe_fetch", return_value=mock_resp) as g:
        fetch_from_url(_EDIT_URL)
    called_url = g.call_args[0][0]
    assert "export?format=csv" in called_url
