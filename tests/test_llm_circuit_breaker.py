"""RF2/T07: provider 429/503 được nghỉ 60s, agent có hạn 45s, route tốn tài nguyên có rate limit."""
from __future__ import annotations

import pytest
import requests

from backend.app.agents import runner
from backend.app.api.deps import limiter
from backend.app.core.config import get_settings
from backend.app.services import llm_service
from backend.app.services.llm_service import FailoverLLMClient


class _Stub:
    def __init__(self, name, status=None):
        self.name, self.status, self.calls = name, status, 0

    def generate(self, prompt, **kw):
        self.calls += 1
        if self.status:
            resp = requests.Response()
            resp.status_code = self.status
            raise requests.HTTPError(f"{self.status}", response=resp)
        return f"{self.name}:{prompt}"


def _chain(first_status):
    a, b = _Stub("a", first_status), _Stub("b")
    return FailoverLLMClient([("a", lambda: a), ("b", lambda: b)]), a


@pytest.mark.parametrize("status", [429, 503])
def test_rate_limited_provider_is_skipped_on_next_request(status):
    client, a = _chain(status)
    assert client.generate("1") == "b:1"
    assert client.generate("2") == "b:2"
    assert a.calls == 1          # lần 2 không gửi request tới 'a'


def test_cooldown_expires(monkeypatch):
    client, a = _chain(429)
    client.generate("1")
    now = llm_service.time.monotonic()
    monkeypatch.setattr(llm_service.time, "monotonic", lambda: now + 61)
    client.generate("2")
    assert a.calls == 2


def test_other_errors_do_not_cool_down():
    client, a = _chain(500)
    client.generate("1")
    client.generate("2")
    assert a.calls == 2


def test_expired_deadline_skips_every_provider():
    client, a = _chain(None)
    with llm_service.llm_deadline(0), pytest.raises(RuntimeError):
        client.generate("x")
    assert a.calls == 0


def test_agent_past_deadline_answers_deterministically(monkeypatch, uploaded_session_id):
    monkeypatch.setattr(runner, "AGENT_DEADLINE_S", 0)
    out = runner.run(uploaded_session_id, "tổng amount theo category")
    assert out.answer and out.usage.get("llm_calls", 0) == 0


@pytest.fixture
def limiter_on():
    limiter.enabled = True
    limiter.reset()
    yield int(get_settings().rate_limit.split("/")[0])
    limiter.enabled = False


def test_dashboard_is_rate_limited(client, limiter_on):
    sid = client.post("/api/sample-shop").json()["session_id"]   # shop mẫu: dashboard không gọi LLM
    codes = [client.get(f"/api/dashboard/{sid}").status_code for _ in range(limiter_on)]
    assert codes[-1] == 200
    assert client.get(f"/api/dashboard/{sid}").status_code == 429


def test_merge_sheets_is_rate_limited(client, limiter_on):
    body = {"session_id": "missing", "sheet_names": ["a", "b"]}
    for _ in range(limiter_on):
        client.post("/api/merge-sheets", json=body)
    assert client.post("/api/merge-sheets", json=body).status_code == 429
