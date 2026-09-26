from __future__ import annotations

import logging

from backend.app.agents.state import AgentState
from backend.app.services.query_classifier import classify_query

_log = logging.getLogger(__name__)


def classify_node(state: AgentState) -> AgentState:
    """Route question to bot_info | off_topic | data_query."""
    question = state.get("question", "")
    intent = classify_query(question)
    if intent == "data_query":
        notice = _missing_cogs(state.get("session_id", ""), question)
        if notice:
            _log.info("classify_node: profit question without COGS → needs_cogs")
            return {**state, "intent": "needs_cogs", "answer": notice}
    _log.info("classify_node: question=%r → intent=%r", question[:60], intent)
    return {**state, "intent": intent}


def _missing_cogs(session_id: str, question: str) -> str | None:
    from backend.app.services.ecommerce_semantic import missing_cogs_notice
    from backend.app.services.storage import session_store
    try:
        return missing_cogs_notice(session_store.get(session_id).dataframe, question)
    except KeyError:
        return None


def route_by_intent(state: AgentState) -> str:
    """Conditional edge: returns the next node name based on intent."""
    return state.get("intent") or "data_query"
