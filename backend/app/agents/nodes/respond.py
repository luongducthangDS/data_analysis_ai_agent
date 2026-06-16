from __future__ import annotations

import logging

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from backend.app.agents.state import AgentState
from backend.app.services.query_classifier import BOT_INFO_RESPONSE, OFF_TOPIC_RESPONSE

_log = logging.getLogger(__name__)


def _is_retryable(exc: Exception) -> bool:
    name = type(exc).__name__
    return any(k in name for k in ("RateLimit", "Timeout", "Connection", "ServiceUnavailable"))


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def _call_llm(client, prompt: str, max_tokens: int = 350, temperature: float = 0.5) -> str:
    return client.generate(prompt, max_tokens=max_tokens, temperature=temperature)


def bot_info_node(state: AgentState) -> AgentState:
    """Generate a contextual bot introduction when data is available, else static."""
    from backend.app.services.storage import session_store
    from backend.app.services.llm_service import get_llm_client

    question = state.get("question", "")
    session_id = state.get("session_id", "")

    try:
        session = session_store.get(session_id)
        df = session.dataframe
        col_names = list(df.columns)[:12]
        n_rows = len(df)
        file_names = ", ".join(session.file_names) if session.file_names else session.filename
        client = get_llm_client()
        prompt = (
            f"Bạn là Data Analysis AI Agent — trợ lý phân tích dữ liệu.\n"
            f"Người dùng vừa upload file: {file_names} ({n_rows:,} dòng, {len(df.columns)} cột).\n"
            f"Các cột: {col_names}.\n"
            f"Họ hỏi: \"{question}\"\n\n"
            "Trả lời thân thiện, ngắn gọn bằng tiếng Việt:\n"
            "1. Xác nhận bạn đã thấy dataset (mention tên file + số dòng).\n"
            "2. Đề xuất 3 câu hỏi cụ thể dựa trên TÊN CỘT thực tế (không bịa).\n"
            "3. Nhắc chế độ: Phân tích / SQL / Agent.\n"
            "Không dùng markdown phức tạp."
        )
        answer = _call_llm(client, prompt)
        if len(answer.strip()) >= 30:
            _log.info("bot_info_node: LLM response OK (%d chars)", len(answer))
            return {**state, "answer": answer.strip(), "charts": [], "executed_queries": ["[bot_info:llm]"]}
    except Exception as exc:
        _log.warning("bot_info_node: LLM failed (%s), using static response", exc)

    return {**state, "answer": BOT_INFO_RESPONSE, "charts": [], "executed_queries": ["[bot_info:static]"]}


def off_topic_node(state: AgentState) -> AgentState:
    """Return static off-topic response."""
    return {**state, "answer": OFF_TOPIC_RESPONSE, "charts": [], "executed_queries": ["[off_topic]"]}
