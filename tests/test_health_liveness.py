"""T09: /api/health là liveness probe — async, không I/O.

Trước đây health là route `def` có query DB: chạy chung threadpool (40 slot) với
/api/chat. LLM treo giữ hết slot → health đứng → Render restart container.
"""
import asyncio

from backend.app.api.routes import health as health_route


def test_health_is_async(client):
    assert asyncio.iscoroutinefunction(health_route.health)


def test_health_answers_when_db_is_down(client, mocker):
    mocker.patch("backend.app.services.storage.SessionStore.count", side_effect=RuntimeError("DB down"))
    mocker.patch("backend.app.services.storage.db_session", side_effect=RuntimeError("DB down"))
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
