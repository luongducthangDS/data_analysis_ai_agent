from __future__ import annotations

from langgraph.graph import StateGraph, END

from backend.app.agents.state import AgentState
from backend.app.agents.nodes.classify import classify_node, route_by_intent
from backend.app.agents.nodes.respond import bot_info_node, off_topic_node, data_summary_node
from backend.app.agents.nodes.plan import plan_node
from backend.app.agents.nodes.execute import execute_node
from backend.app.agents.nodes.synthesize import synthesize_node


def _build_graph() -> StateGraph:
    g = StateGraph(AgentState)

    # Register nodes
    g.add_node("classify", classify_node)
    g.add_node("bot_info", bot_info_node)
    g.add_node("off_topic", off_topic_node)
    g.add_node("data_summary", data_summary_node)
    g.add_node("planner", plan_node)
    g.add_node("execute", execute_node)
    g.add_node("synthesize", synthesize_node)

    # Entry point
    g.set_entry_point("classify")

    # Conditional routing after classify
    g.add_conditional_edges(
        "classify",
        route_by_intent,
        {
            "bot_info":     "bot_info",
            "off_topic":    "off_topic",
            "data_summary": "data_summary",
            "data_query":   "planner",
        },
    )

    # Linear flow for data queries
    g.add_edge("planner", "execute")
    g.add_edge("execute", "synthesize")

    # Terminal nodes
    g.add_edge("bot_info", END)
    g.add_edge("off_topic", END)
    g.add_edge("data_summary", END)
    g.add_edge("synthesize", END)

    return g


# Compile once at module load
agent_graph = _build_graph().compile()
