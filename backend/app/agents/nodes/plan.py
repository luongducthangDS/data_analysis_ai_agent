from __future__ import annotations

import json
import logging
import re
from typing import Any

import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

from backend.app.agents.state import AgentState

_log = logging.getLogger(__name__)

# Allowed values — replicated here to avoid importing the full planner
_ALLOWED_ACTIONS = {"aggregate", "compare_metrics", "time_series", "profile"}
_ALLOWED_AGGREGATIONS = {"sum", "mean", "median", "min", "max", "count", "nunique"}
_ALLOWED_FILTER_OPS = {"eq", "ne", "gt", "gte", "lt", "lte", "between", "in", "contains"}


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8), reraise=True)
def _call_llm(client, prompt: str) -> str:
    return client.generate(prompt, max_tokens=900, temperature=0.0, top_p=0.9)


def plan_node(state: AgentState) -> AgentState:
    """
    LLM generates a structured JSON analysis plan.
    Falls back to rule-based plan if LLM fails or returns invalid JSON.
    """
    from backend.app.services.llm_service import get_llm_client
    from backend.app.services.storage import session_store
    from backend.app.services.analysis_planner import (
        build_fallback_plan, _build_planner_prompt, _validate_plan_against_dataframe,
        _repair_plan_for_question, _repair_who_plan, _repair_column_names,
    )

    question = state["question"]
    session_id = state["session_id"]
    history = state.get("history", [])

    try:
        session = session_store.get(session_id)
        df = session.dataframe
        profile = session.profile or {}
        client = get_llm_client()
        prompt = _build_planner_prompt(df, question, profile, history, session.ecommerce_col_map or None)
        raw = _call_llm(client, prompt)
        plan = _extract_json(raw)
        plan = _remap_action_aliases(plan)
        plan = _repair_column_names(plan, df)   # fuzzy-resolve LLM column names before validation
        plan = _repair_who_plan(plan, question, df)
        plan = _repair_plan_for_question(plan, question)
        _validate_plan_against_dataframe(df, plan)
        _log.info("plan_node: LLM plan OK action=%r", plan.get("action"))
        return {**state, "plan": plan, "llm_plan_failed": False}
    except Exception as exc:
        _log.warning("plan_node: LLM planning failed (%s) — using rule-based fallback", exc)
        try:
            session = session_store.get(session_id)
            plan = build_fallback_plan(session.dataframe, question)
            plan["_planner_fallback_reason"] = str(exc)
            return {**state, "plan": plan, "llm_plan_failed": True}
        except Exception as exc2:
            _log.error("plan_node: fallback plan also failed: %s", exc2)
            return {**state, "plan": None, "llm_plan_failed": True, "error": str(exc2)}


_ACTION_ALIASES: dict[str, str] = {
    "info":        "profile",
    "summary":     "profile",
    "overview":    "profile",
    "insight":     "profile",
    "describe":    "profile",
    "statistics":  "profile",
    "stats":       "profile",
    "analyze":     "aggregate",
    "group":       "aggregate",
    "group_by":    "aggregate",
    "filter":      "aggregate",
    "trend":       "time_series",
    "timeseries":  "time_series",
    "time":        "time_series",
    "compare":     "compare_metrics",
}

def _remap_action_aliases(plan: dict[str, Any]) -> dict[str, Any]:
    action = plan.get("action", "")
    if action not in {"aggregate", "compare_metrics", "time_series", "profile"}:
        remapped = _ACTION_ALIASES.get(action)
        if remapped:
            _log.info("plan_node: remapping action %r → %r", action, remapped)
            return {**plan, "action": remapped}
    return plan


def _extract_json(raw: str) -> dict[str, Any]:
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"LLM did not return JSON: {raw[:200]}")
    return json.loads(text[start: end + 1])
