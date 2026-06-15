from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from backend.app.services.agent_tools import AGENT_TOOLS, ToolResult, execute_tool
from backend.app.services.llm_service import get_llm_client


SYSTEM_PROMPT = """Bạn là senior data analyst AI. Nhiệm vụ: phân tích dataset và trả lời câu hỏi của user.

Bạn có 4 tools:
- get_profile: xem cấu trúc dataset (cột, kiểu dữ liệu, thống kê cơ bản)
- analyze_data: chạy aggregate / time_series / compare_metrics / profile
- query_sql: viết SQL SELECT trực tiếp (bảng tên là 'dataset')
- generate_chart: tạo biểu đồ

Quy trình làm việc:
1. Nếu chưa rõ cấu trúc dữ liệu → gọi get_profile trước
2. Gọi tool phù hợp để lấy số liệu
3. Nếu câu hỏi phức tạp, gọi nhiều tool liên tiếp
4. Khi đã đủ thông tin → trả lời bằng text (không gọi thêm tool)

Quy tắc:
- Chỉ dùng TÊN CỘT CHÍNH XÁC từ schema (case-sensitive)
- Không bịa số liệu
- Trả lời tiếng Việt, ngắn gọn, có số liệu cụ thể"""


@dataclass
class AgentStep:
    step: int
    tool_name: str
    arguments: dict[str, Any]
    result_summary: str
    charts: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class AgentResult:
    answer: str
    charts: list[dict[str, Any]]
    agent_steps: list[AgentStep]


def run_agent(
    df: pd.DataFrame,
    question: str,
    profile: dict[str, Any],
    history: list[dict[str, str]] | None = None,
    max_steps: int = 5,
) -> AgentResult:
    schema_summary = _schema_summary(profile)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": f"{SYSTEM_PROMPT}\n\nSCHEMA DATASET:\n{schema_summary}"},
    ]

    for turn in (history or [])[-6:]:
        if turn.get("role") in ("user", "assistant") and turn.get("content"):
            messages.append({"role": turn["role"], "content": str(turn["content"])[:400]})

    messages.append({"role": "user", "content": question})

    client = get_llm_client()
    charts: list[dict[str, Any]] = []
    steps: list[AgentStep] = []

    if not hasattr(client, "generate_with_tools"):
        return _fallback_single_turn(client, df, question, profile)

    for i in range(max_steps):
        try:
            response = client.generate_with_tools(messages, AGENT_TOOLS)
        except Exception as exc:
            return AgentResult(
                answer=f"Lỗi gọi LLM: {exc}",
                charts=charts,
                agent_steps=steps,
            )

        if response["type"] == "text":
            return AgentResult(
                answer=response["content"] or "Không có kết quả.",
                charts=charts,
                agent_steps=steps,
            )

        tool_name = response["name"]
        arguments = response["arguments"]
        tool_result: ToolResult = execute_tool(df, profile, tool_name, arguments)

        step = AgentStep(
            step=i + 1,
            tool_name=tool_name,
            arguments=arguments,
            result_summary=tool_result.summary,
            charts=tool_result.charts,
        )
        steps.append(step)
        charts.extend(tool_result.charts)

        raw_msg = response.get("raw_message")
        if raw_msg is not None:
            messages.append({"role": "assistant", "tool_calls": raw_msg.tool_calls})
        else:
            messages.append({
                "role": "assistant",
                "tool_calls": [{
                    "id": response["id"],
                    "type": "function",
                    "function": {"name": tool_name, "arguments": str(arguments)},
                }],
            })

        messages.append({
            "role": "tool",
            "tool_call_id": response["id"],
            "content": tool_result.summary + ("\n" + tool_result.data_preview if tool_result.data_preview else ""),
        })

    # Hết max_steps — yêu cầu LLM tổng hợp
    messages.append({"role": "user", "content": "Dựa trên kết quả các tools trên, hãy trả lời câu hỏi ban đầu."})
    try:
        final = client.generate_with_tools(messages, [])
        answer = final.get("content") or "Không thể tổng hợp kết quả."
    except Exception:
        answer = _summarize_steps(steps)

    return AgentResult(answer=answer, charts=charts, agent_steps=steps)


def _fallback_single_turn(client: Any, df: pd.DataFrame, question: str, profile: dict[str, Any]) -> AgentResult:
    """Dùng khi LLM client không hỗ trợ tool calling — fallback về generate() thông thường."""
    from backend.app.services.insights import generate_insights
    answer = generate_insights(df, profile, question)
    return AgentResult(answer=answer, charts=[], agent_steps=[])


def _schema_summary(profile: dict[str, Any]) -> str:
    lines = [f"Rows: {profile.get('rows')} | Columns: {profile.get('columns')}"]
    col_types = profile.get("column_types", {})
    for col, dtype in col_types.items():
        missing = profile.get("missing_values", {}).get(col, 0)
        missing_note = f" [{missing} missing]" if missing else ""
        lines.append(f"  - {col} ({dtype}){missing_note}")
    return "\n".join(lines)


def _summarize_steps(steps: list[AgentStep]) -> str:
    if not steps:
        return "Không có kết quả phân tích."
    lines = ["Kết quả phân tích:"]
    for s in steps:
        lines.append(f"- {s.tool_name}: {s.result_summary[:200]}")
    return "\n".join(lines)
