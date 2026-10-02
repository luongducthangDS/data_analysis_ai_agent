from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Callable, TypedDict

import requests
from backend.app.core.config import get_settings  # import này cũng nạp .env (một chỗ duy nhất, xem config.py)

from backend.app.services import usage


_log = logging.getLogger(__name__)


# ── Per-request key override (set by chat route from X-*-Key headers) ─────────
class _RequestKeys(TypedDict, total=False):
    gemini: str
    anthropic: str
    provider: str

_request_keys: ContextVar[_RequestKeys] = ContextVar("_request_keys", default={})


def set_request_keys(gemini: str = "", anthropic: str = "", provider: str = "") -> None:
    """Call at the start of each request to inject user-supplied API keys."""
    keys: _RequestKeys = {}
    if gemini:    keys["gemini"]    = gemini
    if anthropic: keys["anthropic"] = anthropic
    if provider and provider != "auto": keys["provider"] = provider
    _request_keys.set(keys)


class LLMResponseTruncatedError(RuntimeError):
    """Provider dừng vì chạm max_tokens: câu trả lời cụt trông như trả lời thật → coi là lỗi để failover."""


def _raise_if_truncated(model: str, finish_reason: object) -> None:
    # Gemini: FinishReason.MAX_TOKENS (IntEnum, so theo .name); OpenRouter/OpenAI: "length".
    if getattr(finish_reason, "name", finish_reason) in ("MAX_TOKENS", "length"):
        _log.warning("llm: %s cut the answer at max_tokens — treating as failure", model)
        raise LLMResponseTruncatedError(f"{model}: answer truncated at max_tokens")


# ── Hạn chót cho cả lượt hỏi (runner đặt) ─────────────────────────────────────
_deadline: ContextVar[float] = ContextVar("_llm_deadline", default=float("inf"))


@contextmanager
def llm_deadline(seconds: float):
    """Mọi lời gọi LLM trong khối này dừng khi hết `seconds`; caller rơi về fallback tất định."""
    token = _deadline.set(time.monotonic() + seconds)
    try:
        yield
    finally:
        _deadline.reset(token)


def deadline_passed(_retry_state=None) -> bool:
    """Hết hạn lượt hỏi chưa. Cũng dùng làm `stop` của tenacity: hết hạn thì thôi retry, khỏi ngủ backoff."""
    return time.monotonic() >= _deadline.get()


def _call_timeout(default: float) -> float:
    """Timeout của một lời gọi, kẹp theo thời gian còn lại của lượt hỏi."""
    return max(1.0, min(default, _deadline.get() - time.monotonic()))


# ---------------------------------------------------------------------------
# Gemini client (primary) — one instance per model
# ---------------------------------------------------------------------------

# Tried in this order. flash-lite first: RPD 500 on the free tier vs RPD 20 for
# the full flash models, and it's plenty for the JSON plan + short VN brief.
# The heavier flash models sit behind it as a quality/outage backstop.
# Override via GEMINI_MODELS (comma-separated) or GEMINI_MODEL (single).
GEMINI_DEFAULT_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.8-flash",
    "gemini-3.6-flash",
]


def _gemini_models() -> list[str]:
    raw = os.getenv("GEMINI_MODELS") or os.getenv("GEMINI_MODEL") or ""
    models = [m.strip() for m in raw.split(",") if m.strip()]
    return models or list(GEMINI_DEFAULT_MODELS)


class GeminiLLMClient:
    """Google Gemini client. One instance drives ONE model; the failover chain
    holds several so a quota-exhausted (429) model advances to the next."""

    def __init__(self, model: str | None = None):
        self.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", "")
        self.model_name = model or _gemini_models()[0]
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY / GOOGLE_API_KEY not set.")

    def generate(self, prompt: str, max_tokens: int = 700, temperature: float = 0.35, top_p: float = 1.0) -> str:
        import google.generativeai as genai
        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel(
            self.model_name,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                top_p=top_p,
            ),
        )
        with usage.track(self.model_name) as call:
            # Không có timeout, SDK từng treo ~130s mà không ném lỗi → failover không kích hoạt.
            resp = model.generate_content(prompt, request_options={"timeout": _call_timeout(30)})
            meta = getattr(resp, "usage_metadata", None)
            if meta is not None:
                call.prompt_tokens = getattr(meta, "prompt_token_count", 0) or 0
                call.completion_tokens = getattr(meta, "candidates_token_count", 0) or 0
            candidates = getattr(resp, "candidates", None) or []
            if candidates:
                _raise_if_truncated(self.model_name, getattr(candidates[0], "finish_reason", None))
            return (resp.text or "").strip()

    def generate_insights(self, context: str, max_tokens: int = 700, temperature: float = 0.35) -> str:
        system = """Bạn là trợ lý phân tích dữ liệu cho CEO.
Chuyển số liệu thành brief điều hành ngắn, có ưu tiên. Viết tiếng Việt."""
        return self.generate(f"{system}\n\nContext:\n{context}\n\nBản brief:", max_tokens, temperature)

    def answer_question(self, question: str, context: str, max_tokens: int = 500) -> str:
        system = "Bạn là trợ lý phân tích dữ liệu. Trả lời dựa trên context. Không bịa số."
        return self.generate(f"{system}\n\nContext:\n{context}\n\nCâu hỏi: {question}\n\nTrả lời:", max_tokens, 0.25)


# ---------------------------------------------------------------------------
# OpenRouter client (fallback) — one instance per model
# ---------------------------------------------------------------------------

# Free general-purpose models, tried in this order when Gemini is down. All
# multilingual + instruction-following (needed for the VN brief + JSON plan) and
# verified reachable via a plain chat/completions call (some ':free' slugs are
# gated to "agentic harnesses" or no longer free — those are excluded).
# Override via OPENROUTER_MODELS (comma-separated) or OPENROUTER_MODEL (single).
OPENROUTER_DEFAULT_MODELS = [
    "nvidia/nemotron-3-super-120b-a12b:free",
    "google/gemma-4-31b-it:free",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
]


def _openrouter_models() -> list[str]:
    raw = os.getenv("OPENROUTER_MODELS") or os.getenv("OPENROUTER_MODEL") or ""
    models = [m.strip() for m in raw.split(",") if m.strip()]
    return models or list(OPENROUTER_DEFAULT_MODELS)


class OpenRouterLLMClient:
    """OpenRouter chat endpoint (OpenAI-compatible wire format).

    One instance drives ONE model; the failover chain holds several so a
    rate-limited / unavailable free model advances to the next.
    """

    def __init__(self, model: str | None = None):
        self.base_url = get_settings().openrouter_base_url.rstrip("/")
        self.api_token = os.getenv("OPENROUTER_API_KEY", "")
        self.model_id = model or _openrouter_models()[0]
        self.headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/luongducthangDS/data_analysis_ai_agent",
            "X-Title": "SellerLens",
        }
        if not self.api_token:
            raise ValueError("OPENROUTER_API_KEY not set.")

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.35, top_p: float = 1.0) -> str:
        payload = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        with usage.track(self.model_id) as call:
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=_call_timeout(60),
            )
            resp.raise_for_status()
            body = resp.json()
            stats = body.get("usage") or {}
            call.prompt_tokens = int(stats.get("prompt_tokens", 0) or 0)
            call.completion_tokens = int(stats.get("completion_tokens", 0) or 0)
            # OpenRouter returns 200 with an `error` object for upstream failures.
            if body.get("error"):
                raise RuntimeError(f"OpenRouter {self.model_id}: {body['error'].get('message', body['error'])}")
            choice = body["choices"][0]
            _raise_if_truncated(self.model_id, choice.get("finish_reason"))
            return (choice["message"]["content"] or "").strip()

    def generate_insights(self, context: str, max_tokens: int = 700, temperature: float = 0.35) -> str:
        system = """Bạn là trợ lý phân tích dữ liệu cho CEO.
Chuyển số liệu thành brief điều hành ngắn. Viết tiếng Việt."""
        return self.generate(f"{system}\n\nContext:\n{context}\n\nBản brief:", max_tokens, temperature)

    def answer_question(self, question: str, context: str, max_tokens: int = 500) -> str:
        system = "Trả lời câu hỏi dựa trên context. Không bịa số."
        return self.generate(f"{system}\n\nContext:\n{context}\n\nCâu hỏi: {question}\n\nTrả lời:", max_tokens, 0.25)


# ---------------------------------------------------------------------------
# Failover wrapper + Factory — Gemini → OpenAI-compatible
# ---------------------------------------------------------------------------

_ProviderSpec = tuple[str, Callable[[], object]]

COOLDOWN_SECONDS = 60
_COOLDOWN_STATUSES = {429, 503}


def _status_code(exc: Exception) -> int | None:
    """HTTP status của lỗi provider: requests.HTTPError (.response) hoặc google.api_core (.code)."""
    code = getattr(getattr(exc, "response", None), "status_code", None) or getattr(exc, "code", None)
    try:
        return int(code)
    except (TypeError, ValueError):
        return None


class FailoverLLMClient:
    """
    Wraps an ordered list of providers. Every `generate` / `generate_insights` /
    `answer_question` call walks the chain: the first provider that returns wins;
    on ANY runtime error (404 model gone, 429 quota, timeout, network) it logs and
    moves to the next provider. Raises only when every provider in the chain fails
    — that's the signal for callers to drop to their rule-based fallback.

    Provider clients are built lazily and cached; a provider that fails to even
    initialise (missing key) is skipped for the rest of the process.
    """

    def __init__(self, specs: list[_ProviderSpec]):
        self._specs = specs
        self._clients: dict[str, object] = {}
        self._dead: set[str] = set()
        # Provider vừa trả 429/503 thì nghỉ, không tốn thêm round-trip vào nó.
        self._cooldown_until: dict[str, float] = {}
        self.last_provider: str | None = None

    @property
    def chain(self) -> list[str]:
        return [name for name, _ in self._specs]

    def _iter_clients(self):
        for name, factory in self._specs:
            if name in self._dead or time.monotonic() < self._cooldown_until.get(name, 0):
                continue
            client = self._clients.get(name)
            if client is None:
                try:
                    client = factory()
                except Exception as exc:
                    _log.warning("llm: provider %r unavailable (init failed: %s) — skipping", name, exc)
                    self._dead.add(name)
                    continue
                self._clients[name] = client
            yield name, client

    def _call(self, method: str, *args, **kwargs):
        errors: list[str] = []
        for name, client in self._iter_clients():
            if deadline_passed():
                errors.append("hết hạn lượt hỏi")
                break
            try:
                out = getattr(client, method)(*args, **kwargs)
                self.last_provider = name
                if errors:
                    _log.info("llm: recovered on provider %r after %d failure(s)", name, len(errors))
                return out
            except Exception as exc:
                _log.warning("llm: provider %r failed on %s (%s: %s) — trying next",
                             name, method, type(exc).__name__, exc)
                errors.append(f"{name}: {exc}")
                if _status_code(exc) in _COOLDOWN_STATUSES:
                    self._cooldown_until[name] = time.monotonic() + COOLDOWN_SECONDS
        raise RuntimeError(
            f"Tất cả LLM provider đều lỗi cho {method}() [{'; '.join(errors) or 'không có provider khả dụng'}]"
        )

    def generate(self, *args, **kwargs) -> str:
        return self._call("generate", *args, **kwargs)

    def generate_insights(self, *args, **kwargs) -> str:
        return self._call("generate_insights", *args, **kwargs)

    def answer_question(self, *args, **kwargs) -> str:
        return self._call("answer_question", *args, **kwargs)


def _make_request_key_gemini(key: str, model: str | None = None) -> Callable[[], object]:
    def factory():
        c = GeminiLLMClient.__new__(GeminiLLMClient)
        c.api_key = key
        c.model_name = model or _gemini_models()[0]
        return c
    return factory


def _make_request_key_anthropic(key: str) -> Callable[[], object]:
    def factory():
        from backend.app.services.llm_service_anthropic import AnthropicLLMClient  # optional shim
        return AnthropicLLMClient(api_key=key)
    return factory


def _gemini_keys() -> list[str]:
    """Các Gemini key khả dụng, theo thứ tự ưu tiên, đã khử trùng lặp.

    Quota free tier tính theo (PROJECT Google Cloud, model), không theo key: 429 ghi
    `PerDayPerProjectPerModel`. GEMINI_API_KEY2 chỉ nhân đôi hạn mức khi tạo ở project khác.
    """
    keys: list[str] = []
    for var in ("GEMINI_API_KEY", "GEMINI_API_KEY2", "GEMINI_API_KEY3", "GOOGLE_API_KEY"):
        key = (os.getenv(var) or "").strip()
        if key and key not in keys:
            keys.append(key)
    return keys


def _gemini_specs() -> list[_ProviderSpec]:
    """One spec per (model, key) pair.

    Thứ tự: hết quota một model thì thử KEY KHÁC trên cùng model trước, rồi mới
    hạ xuống model kế. Giữ được model tốt nhất lâu nhất thay vì tụt hạng ngay
    khi key đầu chạm trần.
    """
    keys = _gemini_keys()
    if not keys:
        return []
    specs: list[_ProviderSpec] = []
    for model in _gemini_models():
        for idx, key in enumerate(keys, start=1):
            label = f"gemini:{model}" if idx == 1 else f"gemini:{model}#key{idx}"
            specs.append((label, _make_request_key_gemini(key, model)))
    return specs


def _openrouter_specs() -> list[_ProviderSpec]:
    """One spec per configured OpenRouter model."""
    return [
        (f"openrouter:{m}", (lambda m=m: OpenRouterLLMClient(m)))
        for m in _openrouter_models()
    ]


def _env_provider_specs(forced: str) -> list[_ProviderSpec]:
    """Env-key provider chain. `forced` pins one provider's model list
    (gemini | openrouter); otherwise auto = Gemini models → OpenRouter models."""
    if forced == "gemini":
        return _gemini_specs()
    if forced in ("openrouter", "or"):
        return _openrouter_specs()

    # auto: include only providers whose key is present, in fallback order
    specs: list[_ProviderSpec] = []
    if _gemini_keys():
        specs += _gemini_specs()
    if os.getenv("OPENROUTER_API_KEY"):
        specs += _openrouter_specs()
    return specs


_clients: dict[str, FailoverLLMClient] = {}  # theo provider đã ghim ("" = auto)


def get_llm_client() -> FailoverLLMClient:
    """
    Return a FailoverLLMClient.
    Priority: user-supplied header keys (per request) → env-var provider chain.
    Set LLM_PROVIDER env var (or X-LLM-Provider header) to pin one provider.
    """
    rk = _request_keys.get()

    if rk:
        forced = (rk.get("provider") or os.getenv("LLM_PROVIDER", "")).lower()
        specs: list[_ProviderSpec] = []
        if rk.get("gemini") and forced in ("", "gemini"):
            specs.append(("gemini", _make_request_key_gemini(rk["gemini"])))
        if rk.get("anthropic") and forced in ("", "anthropic"):
            specs.append(("anthropic", _make_request_key_anthropic(rk["anthropic"])))
        # keep env providers as a further safety net unless a provider was pinned
        if not forced:
            specs += _env_provider_specs("")
        if specs:
            return FailoverLLMClient(specs)

    forced = os.getenv("LLM_PROVIDER", "").lower()
    if forced in _clients:
        return _clients[forced]
    specs = _env_provider_specs(forced)
    if not specs:
        raise RuntimeError(
            "Không có LLM provider nào khả dụng. "
            "Set GEMINI_API_KEY, OPENROUTER_API_KEY, hoặc nhập API key trong Settings UI."
        )
    _clients[forced] = FailoverLLMClient(specs)
    return _clients[forced]


def get_active_provider() -> str:
    """Name of the provider that last answered, else the head of the chain.
    Model-specific specs ('gemini:<model>' / 'openrouter:<model>') collapse to
    the bare provider name."""
    try:
        client = get_llm_client()
        name = client.last_provider or (client.chain[0] if client.chain else "none")
        return name.split(":", 1)[0]
    except Exception:
        return "none"
