from __future__ import annotations

import logging
import time

from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from backend.app.agents.state import AgentState
from backend.app.services.llm_service import get_llm_client
from backend.app.services.query_classifier import BOT_INFO_RESPONSE, OFF_TOPIC_RESPONSE
from backend.app.services.storage import session_store

_log = logging.getLogger(__name__)

_MIN_BOT_INFO_LEN = 30
_MIN_SUMMARY_LEN = 50


def _is_retryable(exc: Exception) -> bool:
    name = type(exc).__name__
    return any(k in name for k in ("RateLimit", "Timeout", "Connection", "ServiceUnavailable"))


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception(_is_retryable),
    reraise=True,
)
def _call_llm(client, prompt: str, max_tokens: int = 350, temperature: float = 0.5) -> str:
    return client.generate(prompt, max_tokens=max_tokens, temperature=temperature)


def _build_profile_context(profile: dict, df_len: int, df_cols: int) -> tuple[str, dict]:
    """Build LLM context string and summary stats from session.profile."""
    rows = profile.get("rows", df_len)
    cols = profile.get("columns", df_cols)
    col_types = profile.get("column_types", {})
    numeric_summary = profile.get("numeric_summary", {})
    categorical_summary = profile.get("categorical_summary", {})
    missing = profile.get("missing_values", {})
    missing_total = sum(missing.values())

    n_date = sum(1 for t in col_types.values() if "datetime" in str(t))
    n_num = len(numeric_summary)
    n_cat = len(categorical_summary)

    num_lines = []
    for col, stats in list(numeric_summary.items())[:6]:
        mn, mx, mean = stats.get("min"), stats.get("max"), stats.get("mean")
        if mn is not None:
            num_lines.append(f"  - {col}: min={mn:,.1f}, max={mx:,.1f}, TB={mean:,.1f}")
        else:
            num_lines.append(f"  - {col}")

    cat_lines = []
    for col, vals in list(categorical_summary.items())[:6]:
        top = ", ".join(str(v["value"]) for v in vals[:3])
        cat_lines.append(f"  - {col}: {top}…")

    context = (
        f"Dataset: {rows:,} dòng × {cols} cột "
        f"({n_num} số, {n_cat} danh mục"
        f"{', ' + str(n_date) + ' thời gian' if n_date else ''}).\n"
        f"Missing: {missing_total:,} ô.\n\n"
        f"Cột số:\n" + ("\n".join(num_lines) if num_lines else "  (không có)") + "\n\n"
        f"Cột danh mục:\n" + ("\n".join(cat_lines) if cat_lines else "  (không có)")
    )
    stats = {
        "rows": rows, "cols": cols, "n_num": n_num, "n_cat": n_cat,
        "n_date": n_date, "missing_total": missing_total,
        "num_lines": num_lines, "cat_lines": cat_lines,
    }
    return context, stats


def bot_info_node(state: AgentState) -> AgentState:
    """Generate a contextual bot introduction when data is available, else static."""
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
        t0 = time.perf_counter()
        answer = _call_llm(client, prompt)
        _log.info("bot_info_node: LLM latency=%.2fs session=%s", time.perf_counter() - t0, session_id)
        if len(answer.strip()) >= _MIN_BOT_INFO_LEN:
            return {**state, "answer": answer.strip(), "charts": [], "executed_queries": ["[bot_info:llm]"]}
        _log.warning("bot_info_node: LLM response too short (%d chars) session=%s", len(answer.strip()), session_id)
    except Exception as exc:
        _log.warning("bot_info_node: LLM failed session=%s (%s)", session_id, exc)

    return {**state, "answer": BOT_INFO_RESPONSE, "charts": [], "executed_queries": ["[bot_info:static]"]}


def off_topic_node(state: AgentState) -> AgentState:
    """Return static off-topic response."""
    return {**state, "answer": OFF_TOPIC_RESPONSE, "charts": [], "executed_queries": ["[off_topic]"]}


def data_summary_node(state: AgentState) -> AgentState:
    """
    Answer overview/summary questions using session.profile directly.
    Bypasses planner/execute — profile is already computed on upload.
    """
    session_id = state.get("session_id", "")
    question = state.get("question", "")

    try:
        session = session_store.get(session_id)
        df = session.dataframe
        profile = session.profile or {}
    except Exception as exc:
        _log.warning("data_summary_node: session load failed session=%s (%s)", session_id, exc)
        return {**state, "answer": f"Lỗi phiên: {exc}", "charts": [], "executed_queries": ["[data_summary:error]"]}

    context, s = _build_profile_context(profile, len(df), len(df.columns))

    try:
        client = get_llm_client()
        file_names = ", ".join(session.file_names) if session.file_names else session.filename
        prompt = (
            f"Bạn là senior data analyst. Người dùng muốn hiểu tổng quan về dataset '{file_names}'.\n\n"
            f"Thông tin đã tính sẵn:\n{context}\n\n"
            f"Câu hỏi: {question}\n\n"
            "Trả lời tiếng Việt, súc tích (không dài hơn 10 câu):\n"
            "1. Mô tả kích thước và cấu trúc (số dòng, cột, loại dữ liệu)\n"
            "2. Tóm tắt các cột số quan trọng (range, trung bình)\n"
            "3. Liệt kê các cột danh mục và giá trị tiêu biểu\n"
            "4. Ghi chú missing data (nếu có)\n"
            "5. Đề xuất 3 câu hỏi phân tích cụ thể dựa trên TÊN CỘT THỰC TẾ\n"
            "Không bịa số liệu ngoài context đã cho. Không viết 'Không tìm thấy' khi đã có context."
        )
        t0 = time.perf_counter()
        answer = _call_llm(client, prompt, max_tokens=600, temperature=0.3)
        _log.info(
            "data_summary_node: LLM latency=%.2fs chars=%d session=%s",
            time.perf_counter() - t0, len(answer), session_id,
        )
        if len(answer.strip()) >= _MIN_SUMMARY_LEN:
            return {**state, "answer": answer.strip(), "charts": [], "executed_queries": ["[data_summary:llm]"]}
        _log.warning("data_summary_node: LLM response too short (%d chars) session=%s", len(answer.strip()), session_id)
    except Exception as exc:
        _log.warning("data_summary_node: LLM failed session=%s (%s)", session_id, exc)

    # Deterministic fallback — always works, no LLM needed
    rows, cols = s["rows"], s["cols"]
    n_num, n_cat, n_date = s["n_num"], s["n_cat"], s["n_date"]
    missing_total = s["missing_total"]
    lines = [
        f"Dataset có **{rows:,} dòng** và **{cols} cột** "
        f"({n_num} số, {n_cat} danh mục{', ' + str(n_date) + ' thời gian' if n_date else ''}).",
        "Không có dữ liệu bị thiếu." if missing_total == 0 else f"Tổng **{missing_total:,}** ô bị thiếu.",
    ]
    if s["num_lines"]:
        lines.append("\n**Cột số:**\n" + "\n".join(s["num_lines"]))
    if s["cat_lines"]:
        lines.append("\n**Cột danh mục:**\n" + "\n".join(s["cat_lines"]))
    return {**state, "answer": "\n".join(lines), "charts": [], "executed_queries": ["[data_summary:deterministic]"]}
