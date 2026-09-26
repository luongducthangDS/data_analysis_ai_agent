import asyncio
import json
import logging

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from typing import Optional

from backend.app.agents.runner import AgentOutput, run, stream_answer
from backend.app.api.deps import RATE_LIMIT, limiter, load_owned_session
from backend.app.core.auth import get_current_user
from backend.app.services.llm_service import set_request_keys
from backend.app.schemas import ChatRequest, ChatResponse
from backend.app.services.profiler import build_profile
from backend.app.services.reports import write_markdown_report
from backend.app.services.storage import session_store, DatasetSession

_log = logging.getLogger(__name__)
router = APIRouter()

# Clients get this; the real exception goes to the server log only.
_AGENT_ERROR = "Có lỗi khi phân tích câu hỏi. Vui lòng thử lại."


def _persist_and_report(session: DatasetSession, question: str, answer: str, charts: list, source: str = "llm") -> str:
    """Append turn to history (with source tag) and write markdown report.
    Failures are logged loudly, not raised: the user already has the answer."""
    try:
        session_store.append_messages(session, [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer, "source": source},
        ])
        profile = session.profile or build_profile(session.dataframe)
        session.profile = profile
        report_id, _ = write_markdown_report(answer, profile, charts)
        session.report_id = report_id
        session_store.save(session)
        return report_id
    except Exception:
        _log.exception("persist failed session=%s — turn not saved", session.session_id)
        return ""


def _to_response(session: DatasetSession, output: AgentOutput) -> ChatResponse:
    return ChatResponse(
        session_id=session.session_id,
        answer=output.answer,
        charts=output.charts,
        executed_queries=output.executed_queries,
        query_type=output.intent,
        source=output.source,
        usage=output.usage,
    )


@router.post("/api/chat", response_model=ChatResponse)
@limiter.limit(RATE_LIMIT)
def chat(
    request: Request,
    req: ChatRequest,
    _user: dict = Depends(get_current_user),
) -> ChatResponse:
    """Unified chat endpoint — routes through LangGraph agent."""
    session = load_owned_session(req.session_id, _user)

    try:
        output = run(session.session_id, req.question, session_store.recent_history(session.session_id))
    except Exception as exc:
        _log.exception("chat: agent error session=%s", req.session_id)
        raise HTTPException(status_code=500, detail=_AGENT_ERROR) from exc

    _persist_and_report(session, req.question, output.answer, output.charts, output.source)
    return _to_response(session, output)


@router.post("/api/chat/stream")
@limiter.limit(RATE_LIMIT)
async def chat_stream(
    request: Request,
    req: ChatRequest,
    _user: dict = Depends(get_current_user),
    x_gemini_key: Optional[str] = Header(default=None, alias="X-GEMINI-Key"),
    x_anthropic_key: Optional[str] = Header(default=None, alias="X-ANTHROPIC-Key"),
    x_llm_provider: Optional[str] = Header(default=None, alias="X-LLM-Provider"),
) -> StreamingResponse:
    """
    SSE streaming endpoint. Each event: data: <json>\\n\\n

    Event types:
      {"type": "node",  "node": "<name>"}           — graph node completed
      {"type": "token", "content": "<text>"}         — answer chunk (word-by-word)
      {"type": "done",  "charts": [...], "source": "llm"|"fallback"|...}
      {"type": "error", "detail": "..."}
    """
    # Inject user-supplied keys (from Settings UI) for this request's context
    set_request_keys(
        gemini=x_gemini_key or "",
        anthropic=x_anthropic_key or "",
        provider=x_llm_provider or "",
    )

    session = await run_in_threadpool(load_owned_session, req.session_id, _user)
    session_id = session.session_id
    history_snapshot = await run_in_threadpool(session_store.recent_history, session_id)
    question = req.question

    async def event_generator():
        accumulated_answer = ""
        final_meta: dict = {}
        try:
            async for chunk in stream_answer(session_id, question, history_snapshot):
                if chunk["type"] == "token":
                    accumulated_answer += chunk["content"]
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                    # Small yield to event loop so chunks aren't batched by the network
                    await asyncio.sleep(0)
                elif chunk["type"] == "done":
                    final_meta = chunk
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                else:
                    # node events
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        except Exception:
            _log.exception("chat_stream error session=%s", session_id)
            yield f"data: {json.dumps({'type': 'error', 'detail': _AGENT_ERROR}, ensure_ascii=False)}\n\n"
            return

        # Persist after streaming completes — DB + report file write, off the event loop.
        if accumulated_answer:
            await run_in_threadpool(
                _persist_and_report, session_store.get(session_id), question, accumulated_answer,
                final_meta.get("charts") or [], final_meta.get("source", "llm"),
            )

    return StreamingResponse(event_generator(), media_type="text/event-stream")
