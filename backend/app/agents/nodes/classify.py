from __future__ import annotations

import logging

from backend.app.agents.state import AgentState
from backend.app.services.query_classifier import classify_query

_log = logging.getLogger(__name__)


def classify_node(state: AgentState) -> AgentState:
    """Route question to bot_info | off_topic | data_query."""
    question = state.get("question", "")
    intent = classify_query(question)
    _log.info("classify_node: question=%r → intent=%r", question[:60], intent)
    return {**state, "intent": intent}


def route_by_intent(state: AgentState) -> str:
    """Conditional edge: returns the next node name based on intent."""
    return state.get("intent") or "data_query"
