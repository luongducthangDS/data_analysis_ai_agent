from __future__ import annotations

import logging

from backend.app.agents.state import AgentState

_log = logging.getLogger(__name__)


def execute_node(state: AgentState) -> AgentState:
    """
    Run the analysis plan (Pandas operations). Pure deterministic — no LLM.
    Reuses execute_plan() from analysis_planner.
    """
    from backend.app.services.storage import session_store
    from backend.app.services.analysis_planner import execute_plan

    if state.get("plan") is None:
        _log.error("execute_node: no plan available")
        return {**state, "result_df": None, "executed_queries": ["[no_plan]"]}

    plan = state["plan"]
    session_id = state["session_id"]

    try:
        session = session_store.get(session_id)
        df = session.dataframe
        result_df = execute_plan(df, plan)
        _log.info(
            "execute_node: plan executed, result shape=%s, action=%r",
            result_df.shape if result_df is not None else "None",
            plan.get("action"),
        )
        import json
        compact = {k: v for k, v in plan.items() if not k.startswith("_")}
        return {
            **state,
            "result_df": result_df,
            "executed_queries": [json.dumps(compact, ensure_ascii=False)],
        }
    except Exception as exc:
        _log.warning("execute_node: plan execution failed: %s", exc)
        return {**state, "result_df": None, "executed_queries": [f"[execute_failed: {exc}]"], "error": str(exc)}
