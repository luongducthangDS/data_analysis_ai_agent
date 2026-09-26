"""Auth, session ownership and rate limit at the HTTP boundary."""
from __future__ import annotations

import uuid

import pytest

from backend.app.api.deps import limiter
from backend.app.core.config import get_settings

KEY_A, KEY_B = "key-alice-0001", "key-alice-0002"   # same 8-char prefix on purpose


@pytest.fixture
def auth_on(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "allow_no_auth", False)
    monkeypatch.setattr(settings, "api_keys", [KEY_A, KEY_B])


def test_no_keys_configured_rejects_everything(client, monkeypatch, sample_csv_bytes):
    settings = get_settings()
    monkeypatch.setattr(settings, "allow_no_auth", False)
    monkeypatch.setattr(settings, "api_keys", [])
    resp = client.post("/api/upload", files=[("files", ("d.csv", sample_csv_bytes, "text/csv"))])
    assert resp.status_code == 401


def test_chat_on_someone_elses_session_is_forbidden(client, auth_on, sample_csv_bytes, mock_agent_run):
    up = client.post("/api/upload", headers={"X-API-Key": KEY_A},
                     files=[("files", ("d.csv", sample_csv_bytes, "text/csv"))])
    assert up.status_code == 200
    body = {"session_id": up.json()["session_id"], "question": "tổng amount"}

    for path in ("/api/chat", "/api/chat/stream", "/api/analyze", "/api/agent-chat"):
        assert client.post(path, json=body, headers={"X-API-Key": KEY_B}).status_code == 403, path
    assert client.post("/api/chat", json=body, headers={"X-API-Key": KEY_A}).status_code == 200


def test_rate_limit_not_dodged_by_fake_key_or_spoofed_forwarded_for(client):
    limiter.enabled = True
    limiter.reset()
    try:
        limit = int(get_settings().rate_limit.split("/")[0])
        codes = [
            client.post(
                "/api/chat",
                json={"session_id": "missing", "question": "x"},
                headers={
                    "X-API-Key": uuid.uuid4().hex,                           # invalid key → IP bucket
                    "X-Forwarded-For": f"{uuid.uuid4().hex}, 203.0.113.7",   # left part is spoofable
                },
            ).status_code
            for _ in range(limit + 1)
        ]
    finally:
        limiter.enabled = False
        limiter.reset()
    assert codes[:limit] == [404] * limit
    assert codes[-1] == 429


def test_owner_can_stream_and_turn_is_persisted(client, auth_on, sample_csv_bytes, mocker):
    async def fake_stream(session_id, question, history):
        yield {"type": "token", "content": "Xong."}
        yield {"type": "done", "charts": [], "source": "llm"}

    mocker.patch("backend.app.api.routes.chat.stream_answer", fake_stream)
    limiter.enabled = True   # decorator path must also work for the async SSE route
    limiter.reset()
    try:
        up = client.post("/api/upload", headers={"X-API-Key": KEY_A},
                         files=[("files", ("d.csv", sample_csv_bytes, "text/csv"))])
        sid = up.json()["session_id"]
        resp = client.post("/api/chat/stream", json={"session_id": sid, "question": "q"},
                           headers={"X-API-Key": KEY_A})
    finally:
        limiter.enabled = False
        limiter.reset()
    assert resp.status_code == 200 and '"type": "done"' in resp.text

    from backend.app.services.storage import session_store
    assert session_store.get(sid).history[-1]["content"] == "Xong."
