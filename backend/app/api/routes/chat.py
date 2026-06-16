from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from backend.app.agents.runner import AgentOutput, run, stream_answer
from backend.app.core.auth import get_current_user
from backend.app.schemas import (
    AgentChatResponse,
    AnalyzeRequest,
    AnalyzeResponse,
    ChatRequest,
    ChatResponse,
)
from backend.app.services.guardrails import describe_guardrails
from backend.app.services.profiler import build_profile
from backend.app.services.reports import write_markdown_report
from backend.app.services.storage import session_store, DatasetSession

_log = logging.getLogger(__name__)
router = APIRouter()


def _persist_and_report(session: DatasetSession, question: str, answer: str, charts: list, source: str = "llm") -> str:
    """Append turn to history (with source tag) and write markdown report."""
    session.history.append({"role": "user", "content": question})
    session.history.append({"role": "assistant", "content": answer, "source": source})
    if session.profile is None and session.dataframe is not None:
        session.profile = build_profile(session.dataframe)
    profile = session.profile or build_profile(session.dataframe)
    report_id, _ = write_markdown_report(answer, profile, charts)
    session.report_id = report_id
    session_store.save(session)
    return report_id


def _to_response(session: DatasetSession, output: AgentOutput) -> ChatResponse:
    return ChatResponse(
        session_id=session.session_id,
        answer=output.answer,
        charts=output.charts,
        executed_queries=output.executed_queries,
        query_type=output.intent,
        source=output.source,
    )


@router.post("/api/chat", response_model=ChatResponse)
def chat(
    req: ChatRequest,
    _user: dict = Depends(get_current_user),
) -> ChatResponse:
    """Unified chat endpoint — routes through LangGraph agent."""
    try:
        session = session_store.get(req.session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    try:
        output = run(session.session_id, req.question, session.history[-6:])
    except Exception as exc:
        _log.error("chat: agent error session=%s: %s", req.session_id, exc)
        raise HTTPException(status_code=500, detail=f"Agent error: {exc}") from exc

    _persist_and_report(session, req.question, output.answer, output.charts, output.source)
    return _to_response(session, output)


@router.post("/api/chat/stream")
async def chat_stream(
    req: ChatRequest,
    _user: dict = Depends(get_current_user),
) -> StreamingResponse:
    """
    SSE streaming endpoint. Each event: data: <json>\\n\\n

    Event types:
      {"type": "node",  "node": "<name>"}           — graph node completed
      {"type": "token", "content": "<text>"}         — answer chunk (word-by-word)
      {"type": "done",  "charts": [...], "source": "llm"|"fallback"|...}
      {"type": "error", "detail": "..."}
    """
    try:
        session = session_store.get(req.session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    history_snapshot = list(session.history[-6:])
    session_id = session.session_id
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
        except Exception as exc:
            _log.error("chat_stream error session=%s: %s", session_id, exc)
            yield f"data: {json.dumps({'type': 'error', 'detail': str(exc)})}\n\n"
            return

        # Persist after streaming completes
        if accumulated_answer:
            source = final_meta.get("source", "llm")
            charts = final_meta.get("charts") or []
            try:
                sess = session_store.get(session_id)
                _persist_and_report(sess, question, accumulated_answer, charts, source)
            except Exception as exc:
                _log.warning("chat_stream: persist failed session=%s: %s", session_id, exc)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ── Legacy compat endpoints ───────────────────────────────────────────────────

@router.post("/api/analyze", response_model=AnalyzeResponse)
def analyze_compat(
    req: AnalyzeRequest,
    _user: dict = Depends(get_current_user),
) -> AnalyzeResponse:
    """Legacy /api/analyze — now routes through LangGraph agent."""
    try:
        session = session_store.get(req.session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    question = req.question or ""
    if not question:
        raise HTTPException(status_code=400, detail="Question is required.")

    try:
        output = run(session.session_id, question, session.history[-6:])
    except Exception as exc:
        _log.error("analyze_compat: agent error session=%s: %s", req.session_id, exc)
        raise HTTPException(status_code=500, detail=f"Agent error: {exc}") from exc

    profile = session.profile or build_profile(session.dataframe)
    session.profile = profile
    report_id = _persist_and_report(session, question, output.answer, output.charts, output.source)

    return AnalyzeResponse(
        session_id=req.session_id,
        answer=output.answer,
        profile=profile,
        charts=output.charts,
        report_id=report_id,
        executed_queries=output.executed_queries,
        guardrails=describe_guardrails(),
        source=output.source,
    )


@router.post("/api/agent-chat", response_model=AgentChatResponse)
def agent_chat_compat(
    req: ChatRequest,
    _user: dict = Depends(get_current_user),
) -> AgentChatResponse:
    """Legacy /api/agent-chat — now routes through LangGraph agent."""
    try:
        session = session_store.get(req.session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    try:
        output = run(session.session_id, req.question, session.history[-6:])
    except Exception as exc:
        _log.error("agent_chat_compat: agent error session=%s: %s", req.session_id, exc)
        raise HTTPException(status_code=500, detail=f"Agent error: {exc}") from exc

    _persist_and_report(session, req.question, output.answer, output.charts, output.source)
    return AgentChatResponse(
        session_id=req.session_id,
        answer=output.answer,
        charts=output.charts,
        agent_steps=[],
        executed_queries=output.executed_queries,
    )
