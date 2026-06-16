from __future__ import annotations

from typing import Any
from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    # Input
    session_id: str
    question: str
    history: list[dict[str, str]]

    # Routing
    intent: str | None          # "bot_info" | "off_topic" | "data_query"

    # Planning
    plan: dict[str, Any] | None         # JSON plan from LLM
    llm_plan_failed: bool               # True if LLM planning fell back to rule-based

    # Execution
    result_df: Any | None               # pd.DataFrame after execute_plan()
    executed_queries: list[str]

    # Output
    charts: list[dict[str, Any]]
    answer: str | None
    steps: list[dict[str, Any]]         # multi-tool agent steps
    llm_synthesis_failed: bool          # True if synthesis fell back to deterministic

    # Error handling
    error: str | None
