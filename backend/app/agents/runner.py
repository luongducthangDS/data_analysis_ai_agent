from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from backend.app.agents.graph import agent_graph
from backend.app.agents.state import AgentState

_log = logging.getLogger(__name__)


def _compute_source(state: AgentState) -> str:
    intent = state.get("intent") or "data_query"
    if intent == "bot_info":
        return "bot_info"
    if intent == "off_topic":
        return "off_topic"
    if state.get("llm_synthesis_failed"):
        return "fallback"
    return "llm"


@dataclass
class AgentOutput:
    answer: str
    charts: list[dict[str, Any]] = field(default_factory=list)
    executed_queries: list[str] = field(default_factory=list)
    intent: str = "data_query"
    llm_plan_failed: bool = False
    llm_synthesis_failed: bool = False
    steps: list[dict[str, Any]] = field(default_factory=list)
    source: str = "llm"  # "llm" | "fallback" | "bot_info" | "off_topic"


def run(
    session_id: str,
    question: str,
    history: list[dict[str, str]] | None = None,
) -> AgentOutput:
    """Run the agent graph synchronously. Returns structured output."""
    initial_state: AgentState = {
        "session_id": session_id,
        "question": question,
        "history": history or [],
        "intent": None,
        "plan": None,
        "llm_plan_failed": False,
        "result_df": None,
        "executed_queries": [],
        "charts": [],
        "answer": None,
        "steps": [],
        "llm_synthesis_failed": False,
        "error": None,
    }

    _log.info("agent.run: session=%s question=%r", session_id, question[:60])
    final_state: AgentState = agent_graph.invoke(initial_state)

    if final_state.get("llm_plan_failed"):
        _log.warning("agent.run: LLM planning fell back to rule-based for question=%r", question[:60])
    if final_state.get("llm_synthesis_failed"):
        _log.warning("agent.run: LLM synthesis fell back to deterministic for question=%r", question[:60])

    return AgentOutput(
        answer=final_state.get("answer") or "Không có kết quả.",
        charts=final_state.get("charts") or [],
        executed_queries=final_state.get("executed_queries") or [],
        intent=final_state.get("intent") or "data_query",
        llm_plan_failed=final_state.get("llm_plan_failed", False),
        llm_synthesis_failed=final_state.get("llm_synthesis_failed", False),
        steps=final_state.get("steps") or [],
        source=_compute_source(final_state),
    )


async def stream_answer(
    session_id: str,
    question: str,
    history: list[dict[str, str]] | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """
    Async generator for SSE streaming. Yields:
      {"type": "node",  "node": "<name>"}           — as each graph node completes
      {"type": "token", "content": "<text chunk>"}  — answer streamed word-by-word
      {"type": "done",  "charts": [...], ...}        — final metadata (no answer — assembled from tokens)
    """
    initial_state: AgentState = {
        "session_id": session_id,
        "question": question,
        "history": history or [],
        "intent": None,
        "plan": None,
        "llm_plan_failed": False,
        "result_df": None,
        "executed_queries": [],
        "charts": [],
        "answer": None,
        "steps": [],
        "llm_synthesis_failed": False,
        "error": None,
    }

    final_state: AgentState = initial_state

    # Stream node completion events as graph runs
    async for event in agent_graph.astream(initial_state):
        for node_name, node_state in event.items():
            final_state = {**final_state, **node_state}
            yield {"type": "node", "node": node_name}

    answer = final_state.get("answer") or ""
    source = _compute_source(final_state)

    # Cosmetic streaming: splits the fully-computed answer into word chunks.
    # The full response is already available before this loop starts, so this
    # does NOT reduce latency — it improves perceived responsiveness only.
    # Real LLM token streaming would require refactoring LLM client interfaces.
    if answer:
        import re
        chunks = re.findall(r'\S+\s*', answer)
        for chunk in chunks:
            yield {"type": "token", "content": chunk}

    yield {
        "type": "done",
        "charts": final_state.get("charts") or [],
        "executed_queries": final_state.get("executed_queries") or [],
        "intent": final_state.get("intent") or "data_query",
        "source": source,
        "llm_plan_failed": final_state.get("llm_plan_failed", False),
        "llm_synthesis_failed": final_state.get("llm_synthesis_failed", False),
    }
