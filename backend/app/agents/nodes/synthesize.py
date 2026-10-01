from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import pandas as pd
from tenacity import retry, stop_after_attempt, stop_any, wait_exponential

from backend.app.agents.state import AgentState
from backend.app.services.numeric_parse import parse_number
from backend.app.services import usage
from backend.app.services.llm_service import deadline_passed

_log = logging.getLogger(__name__)

_BAD_STARTS = ("error", "exception", "traceback", "none", "null", "undefined")

# Numbers >= this threshold must be traceable to the result set (anti-hallucination).
# Small plain ints (ranks, counts like "3-5 câu") are ignored to avoid false rejects;
# percentages and numbers with a unit ("99 triệu") are always checked.
_GROUNDING_MIN = 1000.0
_GROUNDING_REL_TOL = 0.01  # 1% relative tolerance for rounding/formatting

_SCALES = {"nghìn": 1e3, "ngàn": 1e3, "triệu": 1e6, "tr": 1e6, "tỷ": 1e9, "tỉ": 1e9}
_L = "A-Za-zÀ-ỹ"
# Lookbehind loại phần số nằm trong MÃ ĐỊNH DANH: "GL001279", "LAN-1270" từng bị đọc
# thành 1279/1270, vượt ngưỡng grounding → câu trả lời đúng bị từ chối.
_CLAIM_RE = re.compile(
    rf"(?<![{_L}0-9_])(?<![{_L}0-9]-)(\d[\d.,]*\d|\d)"
    rf"(\s*%|\s*(?:nghìn|ngàn|triệu|tr|tỷ|tỉ)(?![{_L}]))?",
    re.IGNORECASE,
)
# Con số ngay sau "lãi/lợi nhuận … tăng|giảm|âm" hoặc "lỗ" là lãi (hay Δ lãi) CÓ DẤU.
# ponytail: chỉ xét chủ ngữ lãi — "phí sàn tăng X" có dấu ngược với cột ảnh hưởng lãi.
# "Lãi giảm vì phí tăng X" bị đọc là lãi tăng → từ chối nhầm, rơi về câu tất định (an toàn).
_PROFIT_SIGN_RE = re.compile(
    rf"(?:(?<![{_L}])(?:lãi|lợi nhuận)[^.,;:\n]{{0,40}}?(?<![{_L}])(tăng|giảm|âm)|(?<![{_L}])(lỗ))"
    r"\s+(?:thêm\s+|khoảng\s+|gần\s+|hơn\s+|đến\s+|tới\s+)?$",
    re.IGNORECASE,
)
_YEAR_BEFORE_RE = re.compile(r"(?:năm|/)\s*$", re.IGNORECASE)


@dataclass
class _Claim:
    value: float   # đã nhân đơn vị ("11,6 triệu" → 11_600_000)
    percent: bool
    tol: float     # sai số cho phép: làm tròn theo chữ số cuối đã viết, tối thiểu 1%
    sign: int      # +1 "lãi tăng", -1 "lãi giảm/âm", "lỗ", "-5.000"; 0 = không nói chiều


# Câu trả lời bị cắt ở max_tokens vẫn dài hơn 50 ký tự: khi hết quota flash-lite, gemini-3.8-flash
# (tiêu token cho phần suy nghĩ) trả "… kênh Shopee với 1.63" và nó hiển thị như câu trả lời bình thường.
_COMPLETE_ENDINGS = (".", "!", "?", "…", ")", '"', "”", "»")


def _is_valid_synthesis(answer: str) -> bool:
    stripped = answer.strip()
    if len(stripped) < 50:
        return False
    if stripped.lower().startswith(_BAD_STARTS):
        return False
    if not stripped.rstrip("*_ ").endswith(_COMPLETE_ENDINGS):
        return False
    return True


def _claims(text: str) -> list[_Claim]:
    """Every figure the text asserts, with unit, rounding tolerance and profit direction."""
    out: list[_Claim] = []
    for m in _CLAIM_RE.finditer(text):
        tok, unit = m.group(1), (m.group(2) or "").strip().lower()
        # Dùng chung bộ đọc số với phần còn lại của hệ thống: "1.999,52" (kiểu Việt)
        # từng bị bỏ qua → số bịa viết kiểu Việt lọt qua grounding.
        value = parse_number(tok)
        if value is None:
            continue
        prefix = text[max(0, m.start(1) - 80):m.start(1)]
        if (not unit and value.is_integer() and 1900 <= value <= 2100
                and (_YEAR_BEFORE_RE.search(prefix) or text[m.end(1):m.end(1) + 1] in ("-", "/"))):
            continue   # "năm 2026", "5/2026", "2026-05": năm, không phải số liệu
        frac = re.search(r"[.,](\d+)$", tok)
        decimals = len(frac.group(1)) if frac and not value.is_integer() else 0
        scale = _SCALES.get(unit, 1.0)
        value *= scale
        rounding = 0.5 * 10 ** -decimals * scale
        tol = max(rounding, abs(value) * _GROUNDING_REL_TOL) if unit else max(abs(value) * _GROUNDING_REL_TOL, 1.0)
        sign_m = _PROFIT_SIGN_RE.search(prefix)
        if sign_m:
            sign = 1 if (sign_m.group(1) or "").lower() == "tăng" else -1
        else:
            explicit_minus = prefix.endswith(("-", "−")) and not prefix[-2:-1].isalnum()
            sign = -1 if explicit_minus else 0
        out.append(_Claim(value, unit == "%", tol, sign))
    return out


def _parse_numbers(text: str) -> list[float]:
    """Plain (non-percentage) figures in the text, units applied."""
    return [c.value for c in _claims(text) if not c.percent]


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


def _derived_values(result_df) -> dict[str, list[float]]:
    """Figures derivable from the table, beyond raw cells:
      percents: ratio cells (0.335 or 33.5) and shares of a column total;
      forward:  later − earlier (and its % change / percentage-point gap) between two values of one column or one
                row, in table order — time series ascend, bridge columns go T4 → T5;
      backward: the same pairs the other way round.
    A claim with a profit direction may only use `forward`: "lãi tăng 10 triệu" must
    not pass on a 100 → 90 series because 100 − 90 = +10."""
    out: dict[str, list[float]] = {"percents": [], "forward": [], "forward_pct": [],
                                   "backward": [], "backward_pct": []}
    numeric = result_df.apply(pd.to_numeric, errors="coerce")
    for col in numeric.columns:
        s = numeric[col].dropna()
        out["percents"] += s.tolist() + (s * 100).tolist()
        if s.sum():
            out["percents"] += (s / s.sum() * 100).tolist()
    groups = [numeric[c].dropna().tolist() for c in numeric.columns]
    groups += [row.dropna().tolist() for _, row in numeric.head(60).iterrows()]
    for g in groups:
        if len(g) > 60:   # ponytail: O(n²) cặp; bảng dài hơn thì không nhận số dẫn xuất
            continue
        for i, a in enumerate(g):
            for j, b in enumerate(g):
                if i == j:
                    continue
                key = "forward" if j > i else "backward"
                out[key].append(b - a)
                out[key + "_pct"].append((b - a) * 100)   # chênh điểm %: 0,3452 − 0,3450 → "0,02%"
                if a:
                    out[key + "_pct"].append((b - a) / abs(a) * 100)
    return out


def _numbers_grounded(
    answer: str, result_df, extra_allowed: list[float] | None = None, question: str = "",
    notes: list[str] | None = None,
) -> bool:
    """
    Reject answers citing figures that the result set does not support — the most
    common hallucination for a data tool, and the one that looks right to a seller.
      - plain numbers ≥ 1000 must match a cell, a column sum or a difference of two
        values (±1% or half the last written digit);
      - "99 triệu", "1,5 tỷ" are scaled and checked like plain numbers;
      - "45%" must match a ratio cell, a share of total or a change between values;
      - "lãi tăng X" needs a POSITIVE match, "lãi giảm/âm X", "lỗ X" a NEGATIVE one,
        and differences only count in table order (see _derived_values);
      - numbers the user wrote in the question may be quoted back, and so may numbers in `notes` (the data
        notes we put in the prompt: "3.735 đơn thiếu giá vốn" used to get every mode-C answer rejected).
    `extra_allowed` whitelists derived figures (e.g. distribution stats).
    """
    if result_df is None or result_df.empty:
        return True
    d = _derived_values(result_df)
    values = _allowed_values(result_df) + list(extra_allowed or []) + d["forward"]
    percents = d["percents"] + d["forward_pct"]
    pools = {  # (percent?, signed?) → candidates
        (False, True): values, (False, False): values + d["backward"],
        (True, True): percents, (True, False): percents + d["backward_pct"],
    }
    quoted = [abs(c.value) for c in _claims("\n".join([question, *(notes or [])]))]
    for claim in _claims(answer):
        target = abs(claim.value)
        if not claim.percent and target < _GROUNDING_MIN:
            continue
        if any(abs(q - target) <= claim.tol for q in quoted):
            continue
        pool = pools[(claim.percent, claim.sign != 0)]
        if not any(abs(abs(a) - target) <= claim.tol and (claim.sign == 0 or a * claim.sign > 0) for a in pool):
            _log.warning("synthesize_node: ungrounded %s in answer: %s (sign %+d)",
                         "percentage" if claim.percent else "number", claim.value, claim.sign)
            return False
    return True


@retry(stop=stop_any(stop_after_attempt(3), deadline_passed), wait=wait_exponential(multiplier=1, min=1, max=8), reraise=True)
def _call_llm(client, prompt: str) -> str:
    with usage.stage("synthesize"):
        return client.generate(prompt, max_tokens=500, temperature=0.3)


def synthesize_node(state: AgentState) -> AgentState:
    """
    LLM synthesizes a natural-language answer from execution results.
    Falls back to _deterministic_answer if LLM fails.
    Also builds charts.
    """
    from backend.app.core.config import get_settings
    from backend.app.services.llm_service import get_llm_client
    from backend.app.services.storage import session_store
    from backend.app.services.planner.execute import _describe_numeric
    from backend.app.services.planner.answer import _deterministic_answer, _build_charts_from_result, _build_currency_warning
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
        client = get_llm_client(get_settings().llm_provider_synthesize)
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
        prompt_notes = profit_notes(df, plan)

        prompt = (
            f"Bạn là trợ lý phân tích cho chủ shop bán hàng online. Dùng kết quả phân tích sau để trả lời câu hỏi.\n\n"
            f"Câu hỏi: {question}\n\n"
            f"Kết quả phân tích từ dataset ({len(df.columns)} cột: {col_names[:8]}):\n"
            f"{rows_text}\n"
            f"{dist_context}"
            f"{'LƯU Ý: ' + currency_note if currency_note else ''}\n"
            f"{_notes_for_prompt(prompt_notes)}\n"
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
        if _is_valid_synthesis(answer) and _numbers_grounded(answer, result_df, extra_allowed=dist_allowed,
                                                             question=question, notes=prompt_notes):
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
