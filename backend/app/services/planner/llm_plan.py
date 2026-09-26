"""Planner prompt for the LLM, plus the repairs applied to the plan it returns."""
from __future__ import annotations

import json
import logging
import re
from typing import Any

import pandas as pd

from backend.app.services.ecommerce_semantic import (
    describe_metrics,
)
from backend.app.services.security import sanitize_for_prompt
from backend.app.services.planner.execute import _normalize
from backend.app.services.planner.fallback import _find_name_col, _is_who_question

_log = logging.getLogger(__name__)


def _build_multi_sheet_catalog(session: Any) -> str:
    """Describe all sheets + detected join keys so the LLM can target a cross-sheet `source`."""
    sheets = getattr(session, "sheets", None) or {}
    if len(sheets) <= 1:
        return ""
    active = getattr(session, "active_sheet", None)
    lines = ["", "DANH MỤC SHEET (workbook nhiều sheet):"]
    for key, sdf in sheets.items():
        cols = ", ".join(str(c) for c in list(sdf.columns)[:12])
        mark = "  ← đang phân tích" if key == active else ""
        lines.append(f'  - "{key}" ({len(sdf)} dòng): {cols}{mark}')
    rels = [r for r in getattr(session, "sheet_relationships", []) if getattr(r, "join_key", None)]
    if rels:
        lines.append("QUAN HỆ (join key đã phát hiện):")
        for r in rels:
            lines.append(f'  - "{r.sheet1}" ↔ "{r.sheet2}" join trên "{r.join_key}" ({r.relationship_type})')
    lines.append(
        "Nếu câu hỏi cần dữ liệu từ NHIỀU sheet, thêm field \"source\" vào plan: "
        '{"join":{"base":"<sheet>","with":"<sheet>","on":"<join_key>","how":"left"}}. '
        'Dùng 1 sheet khác: {"sheet":"<tên sheet>"}. '
        "KHÔNG có source = dùng sheet đang phân tích. Chỉ join trên join key đã liệt kê ở trên."
    )
    return "\n".join(lines)


def _build_planner_prompt(
    df: pd.DataFrame,
    question: str,
    profile: dict[str, Any],
    history: list[dict[str, str]] | None = None,
    ecommerce_col_map: dict[str, str] | None = None,
    multi_sheet_catalog: str = "",
) -> str:
    schema_lines = []
    for col, dtype in profile["column_types"].items():
        # Tên cột và giá trị mẫu đến từ file người dùng upload — dữ liệu không
        # tin cậy đi thẳng vào prompt là vector indirect prompt injection.
        examples = [sanitize_for_prompt(v) for v in df[col].dropna().astype(str).head(5)]
        schema_lines.append(f"- {sanitize_for_prompt(col)} ({dtype}): {examples}")

    # Detect currency columns for multi-currency warning in prompt
    currency_cols = [c for c in df.columns if _normalize(c) in ("currency", "tien_te", "don_vi_tien")]
    currency_note = ""
    if currency_cols:
        currencies = df[currency_cols[0]].dropna().unique().tolist()
        if len(currencies) > 1:
            currency_note = f"\nLƯU Ý: Cột '{currency_cols[0]}' có {len(currencies)} loại tiền tệ {currencies}. Nếu câu hỏi hỏi về 1 loại cụ thể, thêm filter currency vào plan.\n"

    return f"""Bạn là senior data analyst / financial analyst AI. Nhiệm vụ: chuyển câu hỏi tự nhiên (tiếng Việt hoặc Anh) thành JSON plan chính xác để backend thực thi bằng Pandas.

QUY TẮC TUYỆT ĐỐI:
1. Chỉ trả về JSON object thuần túy — không markdown, không giải thích, không code block.
2. Chỉ dùng tên cột CHÍNH XÁC từ SCHEMA bên dưới.
3. aggregation hợp lệ: sum | mean | median | min | max | count | nunique
4. operator hợp lệ: eq | ne | gt | gte | lt | lte | between | in | contains
5. Nếu câu hỏi hỏi "bao nhiêu", "số lượng", "count" → aggregation = "count"
6. Nếu câu hỏi hỏi "trung bình", "average" → aggregation = "mean"
7. Nếu câu hỏi hỏi "tỷ lệ %", "phần trăm" → vẫn dùng "sum" hoặc "count" để group, backend tính % từ đó
8. Nếu có cột ngày giờ và câu hỏi hỏi theo tháng/quý/năm → dùng time_series với grain phù hợp
9. Từ "chưa thanh toán" / "pending" → filter Status != "Paid" hoặc Status = "Submitted"/"Approved"
10. Từ "lớn nhất", "top N", "cao nhất" → sort desc, limit = N (mặc định 10)
11. Từ "nhỏ nhất", "thấp nhất", "bottom N" → sort asc, limit = N
12. Câu hỏi chứa "ai", "học sinh nào", "người nào", "nhân viên nào", "khách hàng nào" → PHẢI có group_by trên cột tên/entity + aggregation max/min + sort + limit:1
13. Hỏi "phân phối", "phân bố", "distribution", "histogram" của 1 cột số → action "distribution" với field "column" (tên cột số) và "bins" (mặc định 10)
14. Hỏi "range", "khoảng giá trị", "min max", "biên độ", "spread" của 1 cột số → action "compare_metrics" với 2 metric: min và max của cùng cột đó
{currency_note}
SCHEMA DATASET:
{chr(10).join(schema_lines)}

VÍ DỤ MINH HỌA (kế toán / kiểm toán / DA):

Q: "tổng amount theo category"
{{"action":"aggregate","group_by":["Category"],"metrics":[{{"column":"Amount","aggregation":"sum","label":"Tổng Amount"}}],"sort":[{{"column":"Amount","direction":"desc"}}],"limit":20}}

Q: "top 5 employee chi tiêu nhiều nhất"
{{"action":"aggregate","group_by":["EmployeeID"],"metrics":[{{"column":"Amount","aggregation":"sum","label":"Tổng chi tiêu"}}],"sort":[{{"column":"Amount","direction":"desc"}}],"limit":5}}

Q: "số lượng claim theo trạng thái"
{{"action":"aggregate","group_by":["Status"],"metrics":[{{"column":"ClaimID","aggregation":"count","label":"Số claim"}}],"sort":[{{"column":"Số claim","direction":"desc"}}],"limit":10}}

Q: "claim nào chưa thanh toán trên 500"
{{"action":"aggregate","filters":[{{"column":"Status","operator":"ne","value":"Paid"}},{{"column":"Amount","operator":"gt","value":500}}],"group_by":["Status"],"metrics":[{{"column":"Amount","aggregation":"sum","label":"Tổng Amount chưa TT"}}],"sort":[{{"column":"Amount","direction":"desc"}}],"limit":20}}

Q: "trend expense theo tháng năm 2024"
{{"action":"time_series","filters":[{{"column":"SubmitDate","operator":"between","value":["2024-01-01","2024-12-31"]}}],"time_column":"SubmitDate","grain":"month","metrics":[{{"column":"Amount","aggregation":"sum","label":"Tổng Amount"}}],"sort":[{{"column":"month","direction":"asc"}}],"limit":12}}

Q: "amount trung bình theo manager"
{{"action":"aggregate","group_by":["ApprovedBy"],"metrics":[{{"column":"Amount","aggregation":"mean","label":"Amount trung bình"}}],"sort":[{{"column":"Amount trung bình","direction":"desc"}}],"limit":10}}

Q: "so sánh approved vs rejected"
{{"action":"aggregate","filters":[{{"column":"Status","operator":"in","value":["Approved","Rejected"]}}],"group_by":["Status"],"metrics":[{{"column":"Amount","aggregation":"sum","label":"Tổng Amount"}},{{"column":"ClaimID","aggregation":"count","label":"Số claim"}}],"sort":[{{"column":"Tổng Amount","direction":"desc"}}],"limit":5}}

Q: "expense theo category và status"
{{"action":"aggregate","group_by":["Category","Status"],"metrics":[{{"column":"Amount","aggregation":"sum","label":"Tổng Amount"}}],"sort":[{{"column":"Tổng Amount","direction":"desc"}}],"limit":30}}

Q: "học sinh nào có điểm cao nhất" / "ai có score lớn nhất"
{{"action":"aggregate","group_by":["<cột_tên_entity>"],"metrics":[{{"column":"<score_col>","aggregation":"max","label":"Điểm cao nhất"}}],"sort":[{{"column":"Điểm cao nhất","direction":"desc"}}],"limit":1}}

Q: "số lượng học sinh có điểm dưới 5" / "đếm số bản ghi thỏa điều kiện X < N"
{{"action":"aggregate","filters":[{{"column":"<score_col>","operator":"lt","value":5}}],"metrics":[{{"column":"<score_col>","aggregation":"count","label":"Số học sinh"}}],"limit":1}}

Q: "danh sách học sinh điểm cộng lần 2 dưới 5" / "liệt kê người nào có X < threshold"
{{"action":"aggregate","filters":[{{"column":"<score_col>","operator":"lt","value":5}}],"group_by":["<tên_entity_col>"],"metrics":[{{"column":"<score_col>","aggregation":"max","label":"Điểm"}}],"sort":[{{"column":"Điểm","direction":"asc"}}],"limit":50}}

Q: "phân phối điểm" / "phân bố lương" / "histogram doanh thu"
{{"action":"distribution","column":"<numeric_col>","bins":10}}

Q: "range điểm" / "khoảng giá trị lương" / "biên độ giá"
{{"action":"compare_metrics","metrics":[{{"column":"<numeric_col>","aggregation":"min","label":"Thấp nhất"}},{{"column":"<numeric_col>","aggregation":"max","label":"Cao nhất"}}]}}

Q: "tổng doanh thu theo tên sản phẩm" (doanh thu ở sheet Orders, tên SP ở sheet Items, join order_id)
{{"action":"aggregate","source":{{"join":{{"base":"Orders","with":"Items","on":"order_id","how":"left"}}}},"group_by":["product_name"],"metrics":[{{"column":"revenue","aggregation":"sum","label":"Tổng doanh thu"}}],"sort":[{{"column":"Tổng doanh thu","direction":"desc"}}],"limit":20}}
{multi_sheet_catalog}
LỊCH SỬ HỘI THOẠI GẦN ĐÂY:{_format_history(history)}
{_format_ecommerce_context(ecommerce_col_map)}{describe_metrics(df)}{_date_context(df)}
CÂU HỎI HIỆN TẠI:
{question}""".strip()


def _date_context(df: pd.DataFrame) -> str:
    """Khoảng ngày của dữ liệu để LLM hiểu "tháng này/tháng trước" (trước đây nó bám ví dụ "tháng 5")."""
    date_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns
    if not len(date_cols):
        return ""
    col = date_cols[0]
    dates = pd.to_datetime(df[col], errors="coerce").dropna()
    if dates.empty:
        return ""
    last = dates.max()
    prev = (last.to_period("M") - 1).to_timestamp()
    return (f"\nTHỜI GIAN: cột {col} có dữ liệu từ {dates.min():%Y-%m-%d} đến {last:%Y-%m-%d}. "
            f"\"tháng này\"/\"gần nhất\" = tháng {last:%m/%Y}; \"tháng trước\" = tháng {prev:%m/%Y}. "
            "Không đổi kỳ người dùng hỏi kể cả khi nằm ngoài khoảng này.\n")


def _repair_who_plan(plan: dict[str, Any], question: str, df: pd.DataFrame) -> dict[str, Any]:
    """If question is a 'who' question but plan has no group_by, inject entity column."""
    normalized = _normalize(question)
    is_who = _is_who_question(normalized)
    # profit_bridge tự tìm SKU/kênh gây giảm; ép group_by tên + limit 1 sẽ chỉ còn một nhóm.
    if not is_who or plan.get("group_by") or plan.get("action") == "profit_bridge":
        return plan
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    entity_col = _find_name_col(df, cat_cols)
    if not entity_col:
        return plan
    repaired = dict(plan)
    repaired["group_by"] = [entity_col]
    repaired.setdefault("limit", 1)
    _log.info("_repair_who_plan: injected group_by=%r for who-question", entity_col)
    return repaired


def _repair_id_to_name_group(plan: dict[str, Any], question: str, df: pd.DataFrame) -> dict[str, Any]:
    """Nhóm theo cột TÊN thay vì cột MÃ khi bảng có cả hai.

    Hỏi "sản phẩm nào có giá cao nhất" mà nhóm theo `product_id` thì câu trả
    lời là "P002" — đúng số nhưng vô dụng với người đọc, họ cần "MacBook Air M2".

    Chỉ đổi khi tồn tại cột tên tương ứng và câu hỏi không chủ động hỏi mã.
    """
    group_by = plan.get("group_by")
    if not group_by:
        return plan

    normalized = _normalize(question)
    if any(token in normalized for token in ("ma ", " id", "id ", "code", "ma so")):
        return plan     # người dùng hỏi đúng cái mã

    columns = {str(c).lower(): str(c) for c in df.columns}
    repaired_group, changed = [], False
    for column in group_by:
        match = re.fullmatch(r"(.+?)[_ ]?id", str(column), flags=re.IGNORECASE)
        if not match:
            repaired_group.append(column)
            continue
        prefix = match.group(1).lower()
        for candidate in (f"{prefix}_name", f"{prefix}name", f"ten_{prefix}", f"{prefix}_ten"):
            if candidate in columns:
                repaired_group.append(columns[candidate])
                changed = True
                break
        else:
            repaired_group.append(column)

    if not changed:
        return plan

    repaired = dict(plan)
    repaired["group_by"] = repaired_group
    _log.info("_repair_id_to_name_group: %r → %r", group_by, repaired_group)
    return repaired


def _repair_column_names(plan: dict[str, Any], df: pd.DataFrame) -> dict[str, Any]:
    """
    Fuzzy-resolve LLM-generated column names to actual DataFrame columns.

    The LLM often generates slightly wrong column names (wrong case, missing diacritics,
    truncated, or translated). This prevents _require_column from raising a ValueError
    that would fall back to the dumb profile plan.

    Strategy (in priority order):
      1. Exact match → no change
      2. Normalized match (strip diacritics, lowercase, collapse spaces)
      3. Substring: normalized_llm ⊆ normalized_actual or vice versa (shortest wins)
      4. Word intersection: any word of normalized_llm matches any word of normalized_actual
         (splits on whitespace AND underscores so Diem_cong_2 → {diem, cong, 2})
    """
    norm_map: dict[str, str] = {c: _normalize(c) for c in df.columns}

    def _word_set(s: str) -> set[str]:
        return set(re.sub(r"[_\-]+", " ", s).split())

    def _resolve(col: str) -> str:
        if col in df.columns:
            return col
        n = _normalize(col)
        # 1. Normalized exact
        for actual, norm in norm_map.items():
            if norm == n:
                return actual
        # 2. Substring (prefer actual col whose normalized length is closest to query)
        matches = [actual for actual, norm in norm_map.items() if n in norm or norm in n]
        if matches:
            best = min(matches, key=lambda c: abs(len(norm_map[c]) - len(n)))
            _log.info("_repair_column_names: %r → %r (substring)", col, best)
            return best
        # 3. Word intersection (split on whitespace AND underscores)
        words = _word_set(n)
        scored = [(len(words & _word_set(norm)), actual) for actual, norm in norm_map.items()]
        best_score, best_col = max(scored, key=lambda x: x[0])
        if best_score >= 1:
            _log.info("_repair_column_names: %r → %r (word overlap=%d)", col, best_col, best_score)
            return best_col
        return col  # unresolvable — leave as-is so validation gives a clear error

    repaired = json.loads(json.dumps(plan))  # deep copy
    if isinstance(repaired.get("group_by"), list):
        repaired["group_by"] = [_resolve(c) for c in repaired["group_by"]]
    if isinstance(repaired.get("time_column"), str):
        repaired["time_column"] = _resolve(repaired["time_column"])
    if isinstance(repaired.get("column"), str):  # distribution action
        repaired["column"] = _resolve(repaired["column"])
    for item in repaired.get("filters") or []:
        if isinstance(item.get("column"), str):
            item["column"] = _resolve(item["column"])
    for metric in repaired.get("metrics") or []:
        if isinstance(metric.get("column"), str):
            metric["column"] = _resolve(metric["column"])
    for derived in repaired.get("derived_columns") or []:
        for key in ("source", "quantity", "unit_price", "discount_pct"):
            if isinstance(derived.get(key), str):
                derived[key] = _resolve(derived[key])
    return repaired


def _repair_filter_values(plan: dict[str, Any], df: pd.DataFrame) -> dict[str, Any]:
    """Giá trị filter LLM viết lệch chữ hoa/viết tắt → giá trị thật trong cột ("Tiktok" → "TikTok Shop").

    Trước đây filter kenh = "Tiktok" ra 0 dòng và câu trả lời thành "không có dữ liệu".
    Chỉ đổi khi khớp được DUY NHẤT một giá trị; mơ hồ thì giữ nguyên.
    """
    filters = plan.get("filters") or []
    if not filters:
        return plan
    repaired = json.loads(json.dumps(plan, default=str))

    def _fix(col: str, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        actual = [str(v) for v in df[col].dropna().unique()[:500]]
        if value in actual:
            return value
        n = _normalize(value)
        exact = [a for a in actual if _normalize(a) == n]
        partial = exact or [a for a in actual if n and (n in _normalize(a) or _normalize(a) in n)]
        if len(partial) == 1:
            _log.info("_repair_filter_values: %s %r → %r", col, value, partial[0])
            return partial[0]
        return value

    for item in repaired.get("filters") or []:
        col = item.get("column")
        if item.get("operator") not in {"eq", "ne", "in"} or col not in df.columns:
            continue
        if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_datetime64_any_dtype(df[col]):
            continue
        value = item.get("value")
        item["value"] = [_fix(col, v) for v in value] if isinstance(value, list) else _fix(col, value)
    return repaired


# Câu hỏi nhắm tới CẢ tập dữ liệu, không phải từng nhóm.
_WHOLE_DATASET_MARKERS = (
    "toan bo", "toan he thong", "tong cong", "tat ca", "ca tap",
    "overall", "in total", "of all", "grand total", "entire", "whole",
)
# Dấu hiệu câu hỏi thật sự muốn chia nhóm — nếu có thì đừng đụng vào group_by.
_GROUPING_MARKERS = (
    "theo ", " by ", "moi ", "tung ", " nao", "group", "phan theo",
    "xep hang", "rank", "top ",
)


def _repair_whole_dataset_aggregate(plan: dict[str, Any], question: str) -> dict[str, Any]:
    """Bỏ group_by khi câu hỏi hỏi tổng của TOÀN BỘ dữ liệu.

    "Tổng Debit của toàn bộ sổ cái là bao nhiêu?" hay bị planner dịch thành
    group_by=["AccountName"] + limit 1, và agent trả về tổng của đúng MỘT tài
    khoản trong khi vẫn gọi đó là "toàn bộ sổ cái" — sai số liệu mà nghe rất
    thuyết phục.

    Chỉ can thiệp khi câu hỏi có dấu hiệu "toàn bộ" và KHÔNG có dấu hiệu chia
    nhóm, để không phá các câu xếp hạng ("sản phẩm nào…", "top 5…").
    """
    if plan.get("action") not in {"aggregate", "compare_metrics"}:
        return plan
    if not plan.get("group_by"):
        return plan

    normalized = _normalize(question)
    if not any(marker in normalized for marker in _WHOLE_DATASET_MARKERS):
        return plan
    if any(marker in normalized for marker in _GROUPING_MARKERS):
        return plan

    repaired = dict(plan)
    _log.info("plan repair: bỏ group_by %r vì câu hỏi nhắm toàn bộ dữ liệu", plan.get("group_by"))
    repaired.pop("group_by", None)
    repaired.pop("sort", None)
    repaired["limit"] = 1
    return repaired


def _repair_plan_for_question(plan: dict[str, Any], question: str) -> dict[str, Any]:
    if plan.get("action") == "profit_bridge":
        return plan  # kỳ so sánh nằm trong periods; filter quý chèn thêm sẽ cắt mất một kỳ
    repaired = _repair_whole_dataset_aggregate(dict(plan), question)
    normalized = _normalize(question)
    quarters = _mentioned_quarters(normalized)
    year_match = re.search(r"\b(20\d{2})\b", normalized)
    if not quarters or not year_match:
        return repaired

    year = int(year_match.group(1))
    derived_columns = list(repaired.get("derived_columns", []) or [])
    if not any(item.get("name") == "quarter" for item in derived_columns):
        source = repaired.get("time_column") or "order_date"
        derived_columns.append({"name": "quarter", "operation": "quarter", "source": source})
    repaired["derived_columns"] = derived_columns

    filters = list(repaired.get("filters", []) or [])
    filters.append(
        {
            "column": "quarter",
            "operator": "in",
            "value": [f"{year}Q{quarter}" for quarter in sorted(quarters)],
        }
    )
    repaired["filters"] = filters

    if repaired.get("action") in {"aggregate", "compare_metrics", "time_series"}:
        repaired["group_by"] = ["quarter"]
        repaired["sort"] = [{"column": "quarter", "direction": "asc"}]
        repaired["limit"] = len(quarters)
    return repaired


def _mentioned_quarters(normalized_question: str) -> set[int]:
    quarters = {int(match) for match in re.findall(r"\bq([1-4])\b", normalized_question)}
    quarters.update(int(match) for match in re.findall(r"\bquy\s*([1-4])\b", normalized_question))
    return quarters


def _format_ecommerce_context(col_map: dict[str, str] | None) -> str:
    if not col_map:
        return ""
    lines = ["", "COLUMN MAPPING E-COMMERCE (dùng đúng tên cột này trong plan):"]
    for canonical, actual in col_map.items():
        lines.append(f"  {canonical} → \"{actual}\"")
    lines.append("")
    return "\n".join(lines)


def _format_history(history: list[dict[str, str]] | None) -> str:
    if not history:
        return " (không có)"
    recent = history[-6:]
    lines = []
    for turn in recent:
        role = turn.get("role", "")
        content = str(turn.get("content", ""))[:300]
        if role == "user":
            lines.append(f"User: {content}")
        elif role == "assistant":
            lines.append(f"Assistant: {content}")
    return "\n" + "\n".join(lines) if lines else " (không có)"
