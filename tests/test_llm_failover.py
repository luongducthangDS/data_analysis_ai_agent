"""FailoverLLMClient: walk the provider chain, skip dead ones, raise only when all fail."""
from __future__ import annotations

import pytest

from backend.app.services.llm_service import (
    FailoverLLMClient,
    _env_provider_specs,
    _gemini_models,
    _openrouter_models,
)


class _Stub:
    def __init__(self, name, *, fail=False, init_error=False):
        if init_error:
            raise RuntimeError(f"{name} init boom")
        self.name = name
        self.fail = fail
        self.calls = 0

    def generate(self, prompt, **kw):
        self.calls += 1
        if self.fail:
            raise RuntimeError(f"{self.name} runtime 429")
        return f"{self.name}:{prompt}"

    def generate_insights(self, context, **kw):
        return self.generate(context, **kw)

    def answer_question(self, question, context, **kw):
        return self.generate(question, **kw)


def _spec(name, **kw):
    return (name, lambda: _Stub(name, **kw))


def test_first_provider_wins():
    c = FailoverLLMClient([_spec("a"), _spec("b")])
    assert c.generate("hi") == "a:hi"
    assert c.last_provider == "a"


def test_falls_through_to_next_on_runtime_error():
    c = FailoverLLMClient([_spec("a", fail=True), _spec("b")])
    assert c.generate("hi") == "b:hi"
    assert c.last_provider == "b"


def test_skips_provider_that_fails_to_init():
    c = FailoverLLMClient([_spec("a", init_error=True), _spec("b")])
    assert c.generate("hi") == "b:hi"
    assert "a" in c._dead


def test_raises_only_when_every_provider_fails():
    c = FailoverLLMClient([_spec("a", fail=True), _spec("b", fail=True)])
    with pytest.raises(RuntimeError, match="Tất cả LLM provider"):
        c.generate("hi")


def test_dead_provider_not_retried():
    c = FailoverLLMClient([_spec("a", init_error=True), _spec("b")])
    c.generate("one")
    c.generate("two")
    # 'a' factory raised once; after that it's in _dead and never built again
    assert c._dead == {"a"}


def test_all_three_methods_route_through_chain():
    c = FailoverLLMClient([_spec("a", fail=True), _spec("b")])
    assert c.generate_insights("ctx") == "b:ctx"
    assert c.answer_question("q", "ctx") == "b:q"


# ── Env provider chain: Gemini models → OpenRouter models ─────────────────────

def test_model_list_env_override(monkeypatch):
    monkeypatch.setenv("OPENROUTER_MODELS", "x/one:free, y/two:free")
    monkeypatch.setenv("GEMINI_MODELS", "gm-a, gm-b")
    assert _openrouter_models() == ["x/one:free", "y/two:free"]
    assert _gemini_models() == ["gm-a", "gm-b"]


def test_model_list_defaults(monkeypatch):
    for v in ("OPENROUTER_MODELS", "OPENROUTER_MODEL", "GEMINI_MODELS", "GEMINI_MODEL"):
        monkeypatch.delenv(v, raising=False)
    assert _openrouter_models()[0].endswith(":free")
    assert _gemini_models()[0].startswith("gemini-")


def test_env_chain_gemini_models_then_openrouter_models(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("GEMINI_MODELS", "g1,g2")
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    monkeypatch.setenv("OPENROUTER_MODELS", "a/m1:free,b/m2:free")
    names = [name for name, _ in _env_provider_specs("")]
    assert names == ["gemini:g1", "gemini:g2", "openrouter:a/m1:free", "openrouter:b/m2:free"]


def test_env_chain_forced_gemini_skips_openrouter(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("GEMINI_MODELS", "g1,g2")
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    names = [name for name, _ in _env_provider_specs("gemini")]
    assert names == ["gemini:g1", "gemini:g2"]


def test_env_chain_forced_openrouter_skips_gemini(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    monkeypatch.setenv("OPENROUTER_MODELS", "a/m1:free")
    names = [name for name, _ in _env_provider_specs("openrouter")]
    assert names == ["openrouter:a/m1:free"]
