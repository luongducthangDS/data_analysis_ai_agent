from __future__ import annotations

import logging

from backend.app.agents.state import AgentState
from backend.app.services.query_classifier import classify_query

_log = logging.getLogger(__name__)


def classify_node(state: AgentState) -> AgentState:
    """Route question to bot_info | off_topic | data_query."""
    question = state.get("question", "")
    intent = classify_query(question)
    if intent in ("data_query", "data_summary"):
        lost = _lost_sheet(state.get("session_id", ""))
        if lost:
            _log.info("classify_node: active merged sheet lost → sheet_lost")
            return {**state, "intent": "sheet_lost", "answer": lost}
    if intent == "data_query":
        notice = _missing_cogs(state.get("session_id", ""), question)
        if notice:
            _log.info("classify_node: question needs COGS/category the session lacks → needs_cogs")
            return {**state, "intent": "needs_cogs", "answer": notice}
    _log.info("classify_node: question=%r → intent=%r", question[:60], intent)
    return {**state, "intent": intent}


def _lost_sheet(session_id: str) -> str | None:
    from backend.app.services.storage import lost_sheet_notice, session_store
    try:
        return lost_sheet_notice(session_store.get(session_id))
    except KeyError:
        return None


def _missing_cogs(session_id: str, question: str) -> str | None:
    from backend.app.services.ecommerce_semantic import missing_category_notice, missing_cogs_notice
    from backend.app.services.storage import session_store
    try:
        df = session_store.get(session_id).dataframe
        return missing_cogs_notice(df, question) or missing_category_notice(df, question)
    except KeyError:
        return None


def route_by_intent(state: AgentState) -> str:
    """Conditional edge: returns the next node name based on intent."""
    return state.get("intent") or "data_query"
