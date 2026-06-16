from __future__ import annotations


ALLOWED_TOOLS = {
    "profile_data",
    "execute_pandas_plan",
    "generate_charts",
    "generate_report",
}


def describe_guardrails() -> list[str]:
    return [
        "No arbitrary Python execution — all analysis uses validated JSON plans.",
        "Execution engine: Pandas (no SQL, no DuckDB).",
        "Supported operations: filter, group_by, aggregate, sort, limit, derived columns.",
        "Upload parser accepts CSV/XLSX/XLS files only (10 MB per file).",
        "LLM generates analysis plan; execution is deterministic and sandboxed.",
    ]


def assert_allowed_tool(tool_name: str) -> None:
    if tool_name not in ALLOWED_TOOLS:
        raise ValueError(f"Tool is not allowed: {tool_name}")
