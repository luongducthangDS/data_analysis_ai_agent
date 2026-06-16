from __future__ import annotations

import logging
import re

from tenacity import retry, stop_after_attempt, wait_exponential

from backend.app.agents.state import AgentState

_log = logging.getLogger(__name__)

_BAD_STARTS = ("error", "exception", "traceback", "none", "null", "undefined")

# Numbers >= this threshold must be traceable to the result set (anti-hallucination).
# Small ints (years, ranks, counts like "3-5 câu") are ignored to avoid false rejects.
_GROUNDING_MIN = 1000.0
_GROUNDING_REL_TOL = 0.01  # 1% relative tolerance for rounding/formatting


def _is_valid_synthesis(answer: str) -> bool:
    stripped = answer.strip()
    if len(stripped) < 50:
        return False
    if stripped.lower().startswith(_BAD_STARTS):
        return False
    return True


def _parse_numbers(text: str) -> list[float]:
    """Extract numeric values from text, handling VN ('859.045.000') and EN ('8,950.20') formats."""
    out: list[float] = []
    # Drop percentages — they are derived, not raw result values.
    text = re.sub(r"\d[\d.,]*\s*%", " ", text)
    for tok in re.findall(r"\d[\d.,]*\d|\d", text):
        cleaned = tok
        if "," in cleaned and "." in cleaned:
            # EN style: comma=thousands, dot=decimal
            cleaned = cleaned.replace(",", "")
        elif re.fullmatch(r"\d{1,3}(\.\d{3})+", cleaned):
            cleaned = cleaned.replace(".", "")          # VN thousands separator
        elif re.fullmatch(r"\d{1,3}(,\d{3})+", cleaned):
            cleaned = cleaned.replace(",", "")          # EN thousands separator
        else:
            cleaned = cleaned.replace(",", ".")          # lone comma = decimal
        try:
            out.append(float(cleaned))
        except ValueError:
            continue
    return out


def _allowed_values(result_df) -> list[float]:
    """Values the answer may legitimately cite: raw cells + per-column sums + row count."""
    vals: list[float] = [float(len(result_df))]
    for col in result_df.columns:
        series = result_df[col]
        numeric = series.dropna()
        for v in numeric.tolist():
            try:
                vals.append(float(v))
            except (TypeError, ValueError):
                continue
        try:
            vals.append(float(numeric.astype(float).sum()))
        except (TypeError, ValueError):
            continue
    return vals


def _numbers_grounded(answer: str, result_df, extra_allowed: list[float] | None = None) -> bool:
    """
    Reject answers citing large numbers absent from the result set — the most
    common hallucination for a data tool. Conservative: only checks values >= 1000.
    `extra_allowed` lets callers whitelist derived figures (e.g. distribution stats
    computed from the source column, not present in the binned result table).
    """
    if result_df is None or result_df.empty:
        return True
    allowed = _allowed_values(result_df)
    if extra_allowed:
        allowed.extend(extra_allowed)
    for num in _parse_numbers(answer):
        if abs(num) < _GROUNDING_MIN:
            continue
        # Skip bare years (e.g. "năm 2026") — narrative, not a result figure.
        if num.is_integer() and 1900 <= num <= 2100:
            continue
        tol = max(abs(num) * _GROUNDING_REL_TOL, 1.0)
        if not any(abs(num - a) <= tol for a in allowed):
            _log.warning("synthesize_node: ungrounded number in answer: %s", num)
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
        _deterministic_answer, _build_charts_from_result, _build_currency_warning,
        _describe_numeric,
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

    # For distribution questions, compute descriptive stats from the source column
    # (the binned result table alone doesn't carry mean/median/std).
    dist_context = ""
    dist_allowed: list[float] = []
    if plan.get("action") == "distribution":
        stats = _describe_numeric(df, plan.get("column", ""))
        if stats:
            dist_context = (
                f"\nTHỐNG KÊ MÔ TẢ cột '{plan.get('column')}': "
                f"min={stats['min']}, max={stats['max']}, range={stats['range']}, "
                f"trung bình={stats['mean']}, trung vị={stats['median']}, "
                f"độ lệch chuẩn={stats['std']}, Q1={stats['q1']}, Q3={stats['q3']}.\n"
            )
            dist_allowed = [float(v) for v in stats.values() if isinstance(v, (int, float))]

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
            f"{dist_context}"
            f"{'LƯU Ý: ' + currency_note if currency_note else ''}\n\n"
            "Yêu cầu:\n"
            "- Trả lời thẳng vào câu hỏi, không giải thích bạn đang làm gì\n"
            "- Nêu số liệu quan trọng nhất trước, sau đó insight và hành động đề xuất\n"
            "- Ngắn gọn (3-5 câu), viết tiếng Việt tự nhiên\n"
            "- Không bắt đầu bằng 'Câu hỏi này...', 'Dựa trên...', hay bất kỳ meta-commentary nào\n"
            "- Không bịa số liệu ngoài kết quả phân tích\n"
            "- Nếu câu hỏi hỏi 'ai'/'người nào': nêu tên entity từ cột đầu tiên của bảng kết quả nếu có\n"
            "- Nếu là câu hỏi phân phối: mô tả hình dạng (tập trung ở đâu, có lệch không), khoảng giá trị, và độ phân tán\n"
            "- Nếu kết quả không đủ rõ ràng hoặc dữ liệu trống: hãy nói rõ 'Không tìm thấy dữ liệu phù hợp' thay vì đoán"
        )
        answer = _call_llm(client, prompt)
        if _is_valid_synthesis(answer) and _numbers_grounded(answer, result_df, extra_allowed=dist_allowed):
            _log.info("synthesize_node: LLM synthesis OK (%d chars)", len(answer))
            return {**state, "answer": answer.strip(), "charts": charts, "llm_synthesis_failed": False}
        _log.warning("synthesize_node: LLM response rejected (invalid or ungrounded) — using deterministic answer")
    except Exception as exc:
        _log.warning("synthesize_node: LLM synthesis failed (%s: %s)", type(exc).__name__, exc)

    return {**state, "answer": data_summary, "charts": charts, "llm_synthesis_failed": True}
