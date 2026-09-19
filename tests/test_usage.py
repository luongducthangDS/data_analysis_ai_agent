"""Kiểm tra lớp đo token / chi phí / độ trễ (services/usage.py)."""
from __future__ import annotations

import asyncio

import pytest

from backend.app.services import usage
from backend.app.services.usage import LLMCall


def test_capture_collects_calls():
    with usage.capture() as calls:
        with usage.track("model-a") as call:
            call.prompt_tokens, call.completion_tokens = 100, 20
        with usage.track("model-b") as call:
            call.prompt_tokens, call.completion_tokens = 50, 5
    assert len(calls) == 2
    stats = usage.summarize(calls)
    assert stats["llm_calls"] == 2
    assert stats["prompt_tokens"] == 150
    assert stats["completion_tokens"] == 25
    assert stats["total_tokens"] == 175
    assert stats["models_used"] == ["model-a", "model-b"]


def test_track_records_failure_with_error_type():
    with usage.capture() as calls:
        with pytest.raises(RuntimeError):
            with usage.track("model-a"):
                raise RuntimeError("429 quota")
    stats = usage.summarize(calls)
    assert stats["llm_calls"] == 1
    assert stats["llm_calls_failed"] == 1
    assert stats["failures_by_type"] == {"RuntimeError": 1}
    # Lần gọi hỏng không được tính vào token/chi phí
    assert stats["total_tokens"] == 0


def test_latency_is_measured():
    with usage.capture() as calls:
        with usage.track("model-a"):
            pass
    assert calls[0].latency_ms >= 0


def test_record_outside_capture_is_noop():
    """Gọi LLM ngoài phạm vi capture không được ném lỗi."""
    usage.record(LLMCall(model="model-a", prompt_tokens=10))


def test_nested_capture_is_isolated():
    with usage.capture() as outer:
        with usage.track("outer-model"):
            pass
        with usage.capture() as inner:
            with usage.track("inner-model"):
                pass
        assert len(inner) == 1
        assert inner[0].model == "inner-model"
    assert len(outer) == 1
    assert outer[0].model == "outer-model"


# ------------------------------------------------------------------ chi phí


def test_cost_zero_when_model_unpriced(monkeypatch):
    monkeypatch.delenv("LLM_PRICING_JSON", raising=False)
    with usage.capture() as calls:
        with usage.track("gemini-2.0-flash-lite") as call:
            call.prompt_tokens, call.completion_tokens = 1000, 500
    stats = usage.summarize(calls)
    assert stats["cost_usd"] == 0.0
    # …nhưng phải báo là chưa khai giá, để không tưởng nhầm là thật sự $0
    assert stats["unpriced_calls"] == 1


def test_cost_computed_from_pricing_table(monkeypatch):
    monkeypatch.setenv("LLM_PRICING_JSON", '{"gemini-2.0-flash": [0.10, 0.40]}')
    with usage.capture() as calls:
        with usage.track("gemini-2.0-flash") as call:
            call.prompt_tokens, call.completion_tokens = 1_000_000, 1_000_000
    stats = usage.summarize(calls)
    assert stats["cost_usd"] == pytest.approx(0.50)
    assert stats["unpriced_calls"] == 0


def test_pricing_matches_by_prefix(monkeypatch):
    monkeypatch.setenv("LLM_PRICING_JSON", '{"gemini-2.0-flash": [1.0, 1.0]}')
    # tên model thật thường có hậu tố phiên bản
    assert usage.estimate_cost("gemini-2.0-flash-001", 1_000_000, 0) == pytest.approx(1.0)


def test_malformed_pricing_json_does_not_crash(monkeypatch):
    monkeypatch.setenv("LLM_PRICING_JSON", "{khong-phai-json")
    assert usage.estimate_cost("bat-ky", 1000, 1000) == 0.0


# ------------------------------------------------------------------ async

def test_capture_survives_async_generator():
    """ContextVar.reset() có thể hỏng khi async generator resume ở Context khác.

    stream_answer() bọc capture() quanh một vòng `async for` có yield ra ngoài,
    nên đường này phải chạy được mà không ném ValueError.
    """

    async def fake_stream():
        with usage.capture() as calls:
            async for i in _agen():
                with usage.track(f"model-{i}") as call:
                    call.prompt_tokens = 10
                yield i
        yield usage.summarize(calls)

    async def _agen():
        for i in range(3):
            await asyncio.sleep(0)
            yield i

    async def drive():
        return [item async for item in fake_stream()]

    results = asyncio.run(drive())
    stats = results[-1]
    assert stats["llm_calls"] == 3
    assert stats["total_tokens"] == 30


def test_concurrent_tasks_do_not_mix_usage():
    """Hai request chạy song song phải có sổ ghi riêng."""

    async def one(name: str, n: int) -> dict:
        with usage.capture() as calls:
            for _ in range(n):
                await asyncio.sleep(0)
                with usage.track(name) as call:
                    call.prompt_tokens = 1
            return usage.summarize(calls)

    async def drive():
        return await asyncio.gather(one("a", 2), one("b", 5))

    stats_a, stats_b = asyncio.run(drive())
    assert stats_a["llm_calls"] == 2
    assert stats_b["llm_calls"] == 5
    assert stats_a["models_used"] == ["a"]
    assert stats_b["models_used"] == ["b"]
