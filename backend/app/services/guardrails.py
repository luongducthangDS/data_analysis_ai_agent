from __future__ import annotations


ALLOWED_TOOLS = {
    "profile_data",
    "run_duckdb_query",
    "generate_charts",
    "generate_report",
}


def describe_guardrails() -> list[str]:
    return [
        "No arbitrary Python execution in MVP.",
        "LLM can plan analysis, but execution is restricted to validated JSON tool plans.",
        "All analysis actions go through whitelisted tools only.",
        "DuckDB tool accepts SELECT queries only.",
        "Upload parser accepts CSV/XLSX/XLS files only.",
    ]


def assert_allowed_tool(tool_name: str) -> None:
    if tool_name not in ALLOWED_TOOLS:
        raise ValueError(f"Tool is not allowed: {tool_name}")
