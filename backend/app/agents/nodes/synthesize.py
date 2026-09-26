from __future__ import annotations

import logging
import re

from tenacity import retry, stop_after_attempt, wait_exponential

from backend.app.agents.state import AgentState
from backend.app.services.numeric_parse import parse_number
from backend.app.services import usage

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
    # Lookbehind loại phần số nằm trong MÃ ĐỊNH DANH: "GL001279" từng bị đọc
    # thành 1279, vượt ngưỡng grounding, không khớp kết quả nào → câu trả lời
    # đúng của LLM bị từ chối và agent rơi xuống bản dự phòng.
    for tok in re.findall(r"(?<![A-Za-zÀ-ỹ0-9_])\d[\d.,]*\d|(?<![A-Za-zÀ-ỹ0-9_])\d", text):
        # Dùng chung bộ đọc số với phần còn lại của hệ thống. Bản cũ coi mọi
        # token có cả "," và "." là kiểu Anh, nên "1.999,52" (kiểu Việt) thành
        # "1.999.52" → float() lỗi → SỐ ĐÓ BỊ BỎ QUA. Hệ quả: một con số bịa
        # viết theo định dạng Việt Nam lọt qua lớp kiểm chứng grounding.
        value = parse_number(tok)
        if value is not None:
            out.append(value)
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
    # _parse_numbers không đọc dấu: "lãi giảm 11.567.470" phải khớp ô −11567470 trong bảng.
    allowed = [abs(a) for a in allowed]
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
    with usage.stage("synthesize"):
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
    from backend.app.services.ecommerce_semantic import profit_notes, seller_questions

    question = state["question"]
    plan = state.get("plan") or {}
    result_df = state.get("result_df")
    session_id = state["session_id"]
    join_warning = state.get("join_warning")

    def _with_warning(ans: str) -> str:
        ans = f"⚠️ {join_warning}\n\n{ans}" if join_warning else ans
        # Ghi chú bắt buộc về lãi (thiếu giá vốn, trước/sau quảng cáo) — tất định, không nhờ LLM nhớ.
        notes = profit_notes(df, plan) if result_df is not None and not result_df.empty else []
        return ans + "".join(f"\n\n⚠️ {n}" for n in notes)

    try:
        session = session_store.get(session_id)
        df = session.dataframe
    except Exception as exc:
        return {**state, "answer": f"Lỗi phiên làm việc: {exc}", "charts": [], "llm_synthesis_failed": True}

    # Build charts (deterministic — always works)
    charts: list = []
    if result_df is not None and not result_df.empty:
        try:
            charts = _build_charts_from_result(result_df, plan)
        except Exception as exc:
            _log.warning("synthesize_node: chart build failed: %s", exc)

    # Không hiểu câu hỏi (plan LLM hỏng + fallback không khớp luật nào) hoặc không có dòng nào khớp:
    # nói thật, không để LLM viết "báo cáo" quanh một kết quả không trả lời câu hỏi.
    unclear = (plan.get("_generic") and state.get("llm_plan_failed")) or result_df is None
    if unclear or result_df is None or result_df.empty:
        queries = list(state.get("executed_queries") or [])
        if unclear:
            answer, tag = _unclear_answer(seller_questions(df)), "[unclear]"
        else:
            answer, tag = _no_rows_answer(df, plan), "[no_rows]"
        _log.info("synthesize_node: %s — deterministic answer", tag)
        return {**state, "answer": answer, "charts": [], "llm_synthesis_failed": False,
                "executed_queries": queries + [tag]}

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
        if plan.get("action") == "profit_bridge":
            # Cột trộn tỷ lệ (0.335) với tiền (4e8) → pandas in dạng 4.219710e+08, LLM dễ đọc sai.
            rows_text = result_df.to_string(index=False, max_colwidth=60,
                                            float_format=lambda v: f"{v:,.0f}" if abs(v) >= 1 else f"{v:.4f}")
        else:
            # Chuỗi thời gian phải thấy cả kỳ cuối: cắt 15 dòng đầu từng làm LLM tưởng "không có dữ liệu tháng 6".
            cap = 45 if plan.get("action") == "time_series" else 15
            rows_text = result_df.head(cap).to_string(index=False, max_colwidth=50)
            if len(result_df) > cap:
                rows_text += f"\n(… còn {len(result_df) - cap} dòng không hiển thị)"
        currency_note = _build_currency_warning(df, plan) or ""

        prompt = (
            f"Bạn là trợ lý phân tích cho chủ shop bán hàng online. Dùng kết quả phân tích sau để trả lời câu hỏi.\n\n"
            f"Câu hỏi: {question}\n\n"
            f"Kết quả phân tích từ dataset ({len(df.columns)} cột: {col_names[:8]}):\n"
            f"{rows_text}\n"
            f"{dist_context}"
            f"{'LƯU Ý: ' + currency_note if currency_note else ''}\n"
            f"{_notes_for_prompt(profit_notes(df, plan))}\n"
            "Yêu cầu:\n"
            "- Trả lời thẳng vào câu hỏi, không giải thích bạn đang làm gì\n"
            "- Nêu số liệu quan trọng nhất trước, sau đó điều đáng chú ý trong số liệu\n"
            "- Chỉ đề xuất hành động khi gắn với một con số/nhóm cụ thể trong kết quả; không khuyên chung chung, "
            "không khen suông (\"rất tích cực\", \"tiếp tục đẩy mạnh\")\n"
            "- Xưng \"em\"; gọi người dùng theo cách họ tự xưng trong câu hỏi (chị/anh/bạn), không rõ thì \"anh/chị\". "
            "Không gọi \"CEO\", \"doanh nghiệp\", \"đội ngũ\"\n"
            "- Viết số kiểu Việt: 13.308.870 đ, 33,5%\n"
            "- Cột loi_nhuan_truoc_qc là lãi TRƯỚC quảng cáo; cột loi_nhuan_rong là lãi SAU quảng cáo — gọi đúng tên\n"
            "- Ngắn gọn (3-5 câu), viết tiếng Việt tự nhiên\n"
            "- Không bắt đầu bằng 'Câu hỏi này...', 'Dựa trên...', hay bất kỳ meta-commentary nào\n"
            "- Không bịa số liệu ngoài kết quả phân tích; không tự tính số mới (chênh lệch, tổng, trung bình) "
            "— chỉ trích số có sẵn trong bảng, so sánh bằng lời\n"
            "- Nếu câu hỏi hỏi 'ai'/'người nào': nêu tên entity từ cột đầu tiên của bảng kết quả nếu có\n"
            "- Nếu là câu hỏi phân phối: mô tả hình dạng (tập trung ở đâu, có lệch không), khoảng giá trị, và độ phân tán\n"
            "- Nếu kết quả không đủ rõ ràng hoặc dữ liệu trống: hãy nói rõ 'Không tìm thấy dữ liệu phù hợp' thay vì đoán"
        )
        answer = _call_llm(client, prompt)
        if _is_valid_synthesis(answer) and _numbers_grounded(answer, result_df, extra_allowed=dist_allowed):
            _log.info("synthesize_node: LLM synthesis OK (%d chars)", len(answer))
            if plan.get("action") == "profit_bridge":
                # Gợi ý tính tất định, không để LLM tự nghĩ hay bỏ sót.
                from backend.app.services.ecommerce_semantic import bridge_actions
                answer += "\n\n**Nên kiểm tra:**\n" + "\n".join(f"- {a}" for a in bridge_actions(result_df))
            return {**state, "answer": _with_warning(answer.strip()), "charts": charts, "llm_synthesis_failed": False}
        _log.warning("synthesize_node: LLM response rejected (invalid or ungrounded) — using deterministic answer")
    except Exception as exc:
        _log.warning("synthesize_node: LLM synthesis failed (%s: %s)", type(exc).__name__, exc)

    return {**state, "answer": _with_warning(data_summary), "charts": charts, "llm_synthesis_failed": True}


def _notes_for_prompt(notes: list[str]) -> str:
    """Ghi chú bắt buộc cũng đưa vào prompt: câu cảnh báo được nối sau, LLM không được viết ngược lại nó."""
    if not notes:
        return ""
    return ("HẠN CHẾ DỮ LIỆU (backend sẽ tự in nguyên văn sau câu trả lời — KHÔNG chép lại, KHÔNG khen/kết luận "
            "tích cực về lãi, không nói ngược lại):\n" + "\n".join(f"- {n}" for n in notes) + "\n")


def _unclear_answer(suggestions: list[str]) -> str:
    tips = suggestions or ["Doanh thu theo tháng", "Top 10 sản phẩm bán chạy"]
    return ("Em chưa hiểu câu này với dữ liệu hiện có, nên không đưa số để tránh trả lời sai. "
            "Anh/chị thử hỏi cụ thể hơn, ví dụ:\n" + "\n".join(f"- {t}" for t in tips[:4]))


def _no_rows_answer(df, plan: dict) -> str:
    """0 dòng khớp: nói điều kiện nào, dữ liệu có gì — thay cho "báo cáo CEO" chung chung (trước đây)."""
    import pandas as pd

    lines = ["Không có dòng nào khớp điều kiện của câu hỏi."]
    for f in plan.get("filters") or []:
        col, value = f.get("column"), f.get("value")
        lines.append(f"- Điều kiện: {col} {f.get('operator')} {value}")
        if col not in df.columns:
            continue
        series = df[col]
        if pd.api.types.is_datetime64_any_dtype(series):
            d = series.dropna()
            if not d.empty:
                lines.append(f"  Dữ liệu chỉ có từ {d.min():%d/%m/%Y} đến {d.max():%d/%m/%Y}.")
        elif not pd.api.types.is_numeric_dtype(series) and series.nunique() <= 20:
            lines.append(f"  Giá trị có trong cột {col}: {', '.join(map(str, series.dropna().unique()))}.")
    return "\n".join(lines)
