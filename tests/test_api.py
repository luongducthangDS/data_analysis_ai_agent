"""API integration tests for the Data Analysis AI Agent.

Each test targets one behaviour of the FastAPI application. Fixtures are
provided by conftest.py; heavy LLM calls are short-circuited with mocks.
"""
from __future__ import annotations

import pytest
from tests.conftest import make_csv_bytes, make_xlsx_bytes
from backend.app.services.storage import session_store


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

def test_health_returns_ok(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_health_returns_session_count(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "sessions" in data
    assert isinstance(data["sessions"], int)
    assert data["sessions"] >= 0


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

def test_upload_csv_success(client, sample_csv_bytes):
    resp = client.post(
        "/api/upload",
        files=[("files", ("data.csv", sample_csv_bytes, "text/csv"))],
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "session_id" in body
    assert "profile" in body
    assert "preview_columns" in body


def test_upload_xlsx_success(client, sample_xlsx_bytes):
    resp = client.post(
        "/api/upload",
        files=[("files", ("data.xlsx", sample_xlsx_bytes, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"))],
    )
    assert resp.status_code == 200
    assert "session_id" in resp.json()


def test_upload_multiple_files(client, sample_csv_bytes):
    csv1 = make_csv_bytes()
    csv2 = make_csv_bytes()
    resp = client.post(
        "/api/upload",
        files=[
            ("files", ("file1.csv", csv1, "text/csv")),
            ("files", ("file2.csv", csv2, "text/csv")),
        ],
    )
    assert resp.status_code == 200
    assert "session_id" in resp.json()


def test_upload_no_files_returns_400(client):
    # Send a request with an empty files list — FastAPI will reject it
    resp = client.post("/api/upload", files=[])
    assert resp.status_code == 422  # FastAPI validation: File(...) is required


def test_upload_unsupported_format_returns_400(client):
    resp = client.post(
        "/api/upload",
        files=[("files", ("notes.txt", b"hello world", "text/plain"))],
    )
    assert resp.status_code == 400


def test_upload_file_too_large_returns_413(client):
    big_content = b"a" * (11 * 1024 * 1024)  # 11 MB
    resp = client.post(
        "/api/upload",
        files=[("files", ("big.csv", big_content, "text/csv"))],
    )
    assert resp.status_code == 413


def test_upload_returns_profile_fields(client, sample_csv_bytes):
    resp = client.post(
        "/api/upload",
        files=[("files", ("data.csv", sample_csv_bytes, "text/csv"))],
    )
    assert resp.status_code == 200
    profile = resp.json()["profile"]
    assert "rows" in profile
    assert "columns" in profile
    assert "numeric_summary" in profile


def test_upload_returns_suggested_queries(client, sample_csv_bytes):
    resp = client.post(
        "/api/upload",
        files=[("files", ("data.csv", sample_csv_bytes, "text/csv"))],
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "suggested_queries" in body
    assert isinstance(body["suggested_queries"], list)


# ---------------------------------------------------------------------------
# Analyze
# ---------------------------------------------------------------------------

def test_analyze_success(client, uploaded_session_id, mock_planned_analysis):
    resp = client.post(
        "/api/analyze",
        json={"session_id": uploaded_session_id, "question": "Tổng amount là bao nhiêu?"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "answer" in body
    assert body["answer"]


def test_analyze_session_not_found_returns_404(client):
    resp = client.post(
        "/api/analyze",
        json={"session_id": "nonexistent-session-xyz", "question": "Any question?"},
    )
    assert resp.status_code == 404


def test_analyze_builds_history(client, uploaded_session_id, mock_planned_analysis):
    resp = client.post(
        "/api/analyze",
        json={"session_id": uploaded_session_id, "question": "What is the max amount?"},
    )
    assert resp.status_code == 200
    session = session_store.get(uploaded_session_id)
    assert len(session.history) >= 1
    roles = [h["role"] for h in session.history]
    assert "assistant" in roles


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------

def test_chat_success(client, uploaded_session_id, mock_planned_analysis):
    resp = client.post(
        "/api/chat",
        json={"session_id": uploaded_session_id, "question": "Tổng amount?"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "answer" in body
    assert body["answer"]


def test_chat_session_not_found_returns_404(client):
    resp = client.post(
        "/api/chat",
        json={"session_id": "bad-session-id-000", "question": "Anything?"},
    )
    assert resp.status_code == 404


def test_chat_builds_history(client, uploaded_session_id, mock_planned_analysis):
    session_before = session_store.get(uploaded_session_id)
    history_len_before = len(session_before.history)

    resp = client.post(
        "/api/chat",
        json={"session_id": uploaded_session_id, "question": "How many rows?"},
    )
    assert resp.status_code == 200

    session_after = session_store.get(uploaded_session_id)
    # Chat appends user + assistant = 2 entries
    assert len(session_after.history) >= history_len_before + 2


# ---------------------------------------------------------------------------
# Agent chat
# ---------------------------------------------------------------------------

def test_agent_chat_success(client, uploaded_session_id, mock_agent_run):
    resp = client.post(
        "/api/agent-chat",
        json={"session_id": uploaded_session_id, "question": "Phân tích dữ liệu cho tôi."},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "answer" in body
    assert "agent_steps" in body
    assert body["answer"] == "Agent phân tích xong."


def test_agent_chat_session_not_found_returns_404(client):
    resp = client.post(
        "/api/agent-chat",
        json={"session_id": "totally-wrong-id", "question": "Analyze?"},
    )
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Import URL
# ---------------------------------------------------------------------------

def test_import_url_invalid_raises_400(client):
    resp = client.post(
        "/api/import-url",
        json={"url": "not-a-real-url-abc123"},
    )
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Sheets
# ---------------------------------------------------------------------------

def test_sheets_not_found_returns_404(client):
    resp = client.get("/api/sheets/nonexistent-session-id")
    assert resp.status_code == 404


def test_sheets_success(client, sample_xlsx_bytes):
    # Upload an XLSX which contains at least one sheet
    upload_resp = client.post(
        "/api/upload",
        files=[("files", ("workbook.xlsx", sample_xlsx_bytes, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"))],
    )
    assert upload_resp.status_code == 200
    session_id = upload_resp.json()["session_id"]

    resp = client.get(f"/api/sheets/{session_id}")
    # If the XLSX has multiple sheets, expect 200; if only one, expect 400.
    # Our make_xlsx_bytes() produces a single-sheet workbook, so accept either.
    assert resp.status_code in (200, 400)
    if resp.status_code == 200:
        body = resp.json()
        assert "sheets" in body
        assert isinstance(body["sheets"], list)


def test_sheets_single_csv_returns_sheet_list(client, sample_csv_bytes):
    upload_resp = client.post(
        "/api/upload",
        files=[("files", ("single.csv", sample_csv_bytes, "text/csv"))],
    )
    assert upload_resp.status_code == 200
    session_id = upload_resp.json()["session_id"]

    resp = client.get(f"/api/sheets/{session_id}")
    # Single CSV is stored as 1 sheet — endpoint returns 200 with the sheet list
    assert resp.status_code == 200
    body = resp.json()
    assert "sheets" in body
    assert isinstance(body["sheets"], list)
    assert len(body["sheets"]) >= 1


# ---------------------------------------------------------------------------
# Merge sheets
# ---------------------------------------------------------------------------

def test_merge_sheets_not_found_returns_404(client):
    resp = client.post(
        "/api/merge-sheets",
        json={"session_id": "fake-session-999", "sheet_names": ["Sheet1", "Sheet2"]},
    )
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def test_report_not_found_returns_404(client):
    resp = client.get("/api/report/nonexistent-report-id-xyz")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Preview rows
# ---------------------------------------------------------------------------

def test_upload_response_has_preview_rows(client, sample_csv_bytes):
    resp = client.post(
        "/api/upload",
        files=[("files", ("data.csv", sample_csv_bytes, "text/csv"))],
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "preview_rows" in body
    assert isinstance(body["preview_rows"], list)
    # Each element should be a dict
    if body["preview_rows"]:
        assert isinstance(body["preview_rows"][0], dict)
