"""Đo token, chi phí và độ trễ của từng lần gọi LLM.

Dùng ContextVar nên an toàn khi nhiều request chạy đồng thời trên cùng
event loop — mỗi request có sổ ghi riêng.

    with capture() as calls:
        ...                       # chạy agent
    summarize(calls)              # {'llm_calls': 3, 'total_tokens': 4210, ...}

Token là số **đo được** do provider trả về. Chi phí là số **suy ra** từ bảng
giá bên dưới — muốn con số đúng phải khai giá qua LLM_PRICING_JSON.
"""
from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field


@dataclass
class LLMCall:
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    failed: bool = False
    error_type: str = ""

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def cost_usd(self) -> float:
        return estimate_cost(self.model, self.prompt_tokens, self.completion_tokens)


_calls: ContextVar[list[LLMCall] | None] = ContextVar("llm_calls", default=None)


# --------------------------------------------------------------------- giá

# USD trên 1 triệu token, dạng {model: (giá_input, giá_output)}.
#
# Mặc định = 0.0 cho toàn bộ chuỗi model hiện tại vì tất cả đang chạy ở
# free tier (Gemini flash-lite/flash free, OpenRouter ':free'). Đây KHÔNG
# phải khẳng định các model này miễn phí vĩnh viễn — khi chuyển sang bậc trả
# phí phải khai giá thật:
#
#     LLM_PRICING_JSON='{"gemini-2.0-flash": [0.10, 0.40]}'
#
# Model không có trong bảng → cost = 0.0 và được đếm vào `unpriced_calls`
# để summary không im lặng báo $0 cho thứ thực ra có mất tiền.
DEFAULT_PRICING: dict[str, tuple[float, float]] = {}


def _pricing() -> dict[str, tuple[float, float]]:
    raw = os.getenv("LLM_PRICING_JSON", "").strip()
    if not raw:
        return DEFAULT_PRICING
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return DEFAULT_PRICING
    return {k: (float(v[0]), float(v[1])) for k, v in parsed.items()}


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    table = _pricing()
    price = table.get(model)
    if price is None:
        # thử khớp tiền tố: "gemini-2.0-flash-lite-001" → "gemini-2.0-flash-lite"
        for key, value in table.items():
            if model.startswith(key):
                price = value
                break
    if price is None:
        return 0.0
    return prompt_tokens / 1e6 * price[0] + completion_tokens / 1e6 * price[1]


def is_priced(model: str) -> bool:
    table = _pricing()
    return model in table or any(model.startswith(k) for k in table)


# ------------------------------------------------------------------ ghi nhận


@contextmanager
def capture():
    """Mở một sổ ghi mới cho khối lệnh bên trong."""
    calls: list[LLMCall] = []
    token = _calls.set(calls)
    try:
        yield calls
    finally:
        try:
            _calls.reset(token)
        except ValueError:
            # Async generator có thể resume ở Context khác với lúc set() —
            # khi đó token không dùng lại được. Dọn bằng cách gán thẳng None;
            # mỗi request FastAPI là một task riêng nên không rò sang request khác.
            _calls.set(None)


def record(call: LLMCall) -> None:
    calls = _calls.get()
    if calls is not None:
        calls.append(call)


@contextmanager
def track(model: str):
    """Đo một lần gọi LLM. Trả về LLMCall để callee điền token vào.

        with track(model_name) as call:
            resp = ...
            call.prompt_tokens = ...
    """
    call = LLMCall(model=model)
    start = time.perf_counter()
    try:
        yield call
    except Exception as exc:
        call.failed = True
        call.error_type = type(exc).__name__
        raise
    finally:
        call.latency_ms = (time.perf_counter() - start) * 1000
        record(call)


# ------------------------------------------------------------------ tổng hợp


def summarize(calls: list[LLMCall]) -> dict:
    ok = [c for c in calls if not c.failed]
    failed = [c for c in calls if c.failed]
    unpriced = [c for c in ok if c.total_tokens and not is_priced(c.model)]

    by_error: dict[str, int] = {}
    for call in failed:
        by_error[call.error_type] = by_error.get(call.error_type, 0) + 1

    return {
        "llm_calls": len(calls),
        "llm_calls_failed": len(failed),
        "prompt_tokens": sum(c.prompt_tokens for c in ok),
        "completion_tokens": sum(c.completion_tokens for c in ok),
        "total_tokens": sum(c.total_tokens for c in ok),
        "cost_usd": round(sum(c.cost_usd for c in ok), 6),
        # Số lần gọi có token nhưng model chưa khai giá — cost_usd bên trên
        # đang bỏ sót chúng. >0 nghĩa là đừng tin con số chi phí.
        "unpriced_calls": len(unpriced),
        "failures_by_type": by_error,
        "models_used": sorted({c.model for c in ok}),
    }


def to_dicts(calls: list[LLMCall]) -> list[dict]:
    return [{**asdict(c), "total_tokens": c.total_tokens, "cost_usd": c.cost_usd} for c in calls]
