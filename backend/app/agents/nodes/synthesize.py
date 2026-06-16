from __future__ import annotations

import logging

from tenacity import retry, stop_after_attempt, wait_exponential

from backend.app.agents.state import AgentState

_log = logging.getLogger(__name__)

_BAD_STARTS = ("error", "exception", "traceback", "none", "null", "undefined")


def _is_valid_synthesis(answer: str) -> bool:
    stripped = answer.strip()
    if len(stripped) < 50:
        return False
    if stripped.lower().startswith(_BAD_STARTS):
        return False
    return True


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8), reraise=True)
def _call_llm(client, prompt: str) -> str:
    return client.generate(prompt, max_tokens=500, temperature=0.3)


def synthesize_node(state: AgentState) -> AgentState:
    """
    LLM synthesizes a natural-language answer from execution results.
    Falls back to _deterministic_answer if LLM fails.
    Also builds charts.
    """
    from backend.app.services.llm_service import get_llm_client
    from backend.app.services.storage import session_store
    from backend.app.services.analysis_planner import (
        _deterministic_answer, _build_charts_from_result, _build_currency_warning
    )
    from backend.app.services.insights import generate_insights
    from backend.app.services.profiler import build_profile

    question = state["question"]
    plan = state.get("plan") or {}
    result_df = state.get("result_df")
    session_id = state["session_id"]

    try:
        session = session_store.get(session_id)
        df = session.dataframe
        profile = session.profile or build_profile(df)
    except Exception as exc:
        return {**state, "answer": f"Lỗi phiên làm việc: {exc}", "charts": [], "llm_synthesis_failed": True}

    # Build charts (deterministic — always works)
    charts: list = []
    if result_df is not None and not result_df.empty:
        try:
            charts = _build_charts_from_result(result_df, plan)
        except Exception as exc:
            _log.warning("synthesize_node: chart build failed: %s", exc)

    # If no result (plan failed), use insights fallback
    if result_df is None or result_df.empty:
        _log.info("synthesize_node: no result_df — using generate_insights fallback")
        answer = generate_insights(
            df, profile, question,
            sheets=session.sheets or None,
            sheets_context=session.sheets_context or None,
        )
        return {**state, "answer": answer, "charts": charts, "llm_synthesis_failed": False}

    # Build deterministic fallback first (always available)
    data_summary = _deterministic_answer(question, result_df, plan, source_df=df)

    # Try LLM synthesis
    try:
        client = get_llm_client()
        col_names = list(df.columns)[:15]
        rows_text = result_df.head(15).to_string(index=False, max_colwidth=50)
        currency_note = _build_currency_warning(df, plan) or ""

        prompt = (
            f"Bạn là senior data analyst. Dùng kết quả phân tích sau để trả lời câu hỏi.\n\n"
            f"Câu hỏi: {question}\n\n"
            f"Kết quả phân tích từ dataset ({len(df.columns)} cột: {col_names[:8]}):\n"
            f"{rows_text}\n"
            f"{'LƯU Ý: ' + currency_note if currency_note else ''}\n\n"
            "Yêu cầu:\n"
            "- Trả lời thẳng vào câu hỏi, không giải thích bạn đang làm gì\n"
            "- Nêu số liệu quan trọng nhất trước, sau đó insight và hành động đề xuất\n"
            "- Ngắn gọn (3-5 câu), viết tiếng Việt tự nhiên\n"
            "- Không bắt đầu bằng 'Câu hỏi này...', 'Dựa trên...', hay bất kỳ meta-commentary nào\n"
            "- Không bịa số liệu ngoài kết quả phân tích\n"
            "- Nếu câu hỏi hỏi 'ai'/'người nào': nêu tên entity từ cột đầu tiên của bảng kết quả nếu có\n"
            "- Nếu kết quả không đủ rõ ràng hoặc dữ liệu trống: hãy nói rõ 'Không tìm thấy dữ liệu phù hợp' thay vì đoán"
        )
        answer = _call_llm(client, prompt)
        if _is_valid_synthesis(answer):
            _log.info("synthesize_node: LLM synthesis OK (%d chars)", len(answer))
            return {**state, "answer": answer.strip(), "charts": charts, "llm_synthesis_failed": False}
        _log.warning("synthesize_node: LLM returned invalid response (%d chars)", len(answer.strip()))
    except Exception as exc:
        _log.warning("synthesize_node: LLM synthesis failed (%s: %s)", type(exc).__name__, exc)

    return {**state, "answer": data_summary, "charts": charts, "llm_synthesis_failed": True}
