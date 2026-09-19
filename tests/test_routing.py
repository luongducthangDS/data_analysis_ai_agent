"""Đo chất lượng định tuyến intent của `classify_query` (tool-selection accuracy).

Chạy offline — `classify_query` thuần rule-based, không gọi LLM.

    pytest tests/test_routing.py -q
    python tests/test_routing.py          # in bảng per-class + ma trận nhầm lẫn

## Vì sao không báo cáo accuracy trần

85/100 câu trong bộ eval là câu hỏi dữ liệu. Một bộ phân loại luôn trả
`data_query` đạt ngay 85% accuracy mà không phân loại được gì. Nên chỉ số
chính ở đây là **macro-recall** (trung bình recall của từng lớp, mỗi lớp
cân bằng nhau) — nó sụp xuống ngay khi một lớp thiểu số bị bỏ rơi.

## HELD_OUT

Tập này được viết TRƯỚC khi sửa `query_classifier`, để trả lời câu hỏi:
cải thiện trên bộ eval là thật, hay chỉ là học thuộc 100 câu đó.

⚠️ Giới hạn phải nói rõ: trong lúc sửa, danh sách câu sai của tập này ĐÃ
được in ra để phân tích nguyên nhân. Held-out đúng nghĩa thì không được
nhìn. Vì vậy con số trên tập này lạc quan hơn thực tế, và cả hai tập đều
do cùng một người viết nên không thay thế được câu hỏi của người dùng thật.
Nó đủ để chứng minh bộ phân loại không còn chỉ khớp đúng 100 câu eval,
nhưng không đủ để tuyên bố một tỉ lệ chính xác ngoài đời.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app.services.query_classifier import classify_query
from tests.eval_100 import TEST_CASES

# `data_summary` cũng trả lời từ dữ liệu, chỉ khác node xử lý — với mục đích
# định tuyến thì cả hai đều là "câu hỏi về dữ liệu".
DATA_INTENTS = {"data_query", "data_summary"}


def expected_intents(category: str) -> set[str]:
    if category == "bot_info":
        return {"bot_info"}
    if category == "off_topic":
        return {"off_topic"}
    return DATA_INTENTS


# ---------------------------------------------------------------- held-out

# Viết tay, không trùng câu nào trong bộ eval 100.
HELD_OUT: list[tuple[str, str]] = [
    # --- bot_info: hỏi về chính công cụ
    ("Công cụ này làm được những gì?", "bot_info"),
    ("Ứng dụng có đọc được file CSV nặng không?", "bot_info"),
    ("Bạn có thể xuất báo cáo ra PDF không?", "bot_info"),
    ("Tôi dùng phần mềm này thế nào cho hiệu quả?", "bot_info"),
    ("Hệ thống hỗ trợ tiếng Việt chứ?", "bot_info"),
    ("What file formats does this tool support?", "bot_info"),
    ("Can this app draw pie charts?", "bot_info"),
    ("Ai là người phát triển ứng dụng này?", "bot_info"),
    # --- off_topic: ngoài phạm vi phân tích dữ liệu
    ("Dân số Nhật Bản hiện nay khoảng bao nhiêu?", "off_topic"),
    ("Sáng tác giúp tôi một câu chuyện ngắn về biển", "off_topic"),
    ("Nên mua xe máy hãng nào thì bền?", "off_topic"),
    ("Chỉ tôi cách làm bánh mì tại nhà", "off_topic"),
    ("Ai là tác giả của Truyện Kiều?", "off_topic"),
    ("Năm nay có nên mua vàng tích trữ không?", "off_topic"),
    ("Who won the World Cup in 2018?", "off_topic"),
    ("Write me a haiku about the rain", "off_topic"),
    ("Giải thích thuyết tương đối cho tôi", "off_topic"),
    ("Tối nay ăn gì cho ngon?", "off_topic"),
    # --- data: phải KHÔNG bị bắt nhầm sang hai lớp trên
    ("Doanh thu theo quý của năm ngoái", "data_query"),
    ("Sản phẩm nào có tỷ suất lợi nhuận thấp nhất?", "data_query"),
    ("So sánh chi phí giữa hai chi nhánh", "data_query"),
    ("Vẽ biểu đồ xu hướng đơn hàng theo tháng", "data_query"),
    ("Có bao nhiêu khách hàng chưa thanh toán?", "data_query"),
    ("Top 3 nhân viên bán chạy nhất", "data_query"),
    ("Giá trị trung bình của cột số tiền", "data_query"),
    ("Tổng số lượng tồn kho theo kho", "data_query"),
    ("Show me revenue by region", "data_query"),
    ("Which category has the highest average price?", "data_query"),
    # Hai câu dưới có từ dễ gây nhầm ("giúp tôi", "hướng dẫn") nhưng vẫn là
    # câu hỏi dữ liệu — chốt chặn việc nới từ khoá bot_info quá tay.
    ("Giúp tôi tính tổng doanh thu theo vùng", "data_query"),
    ("Hướng dẫn tôi xem doanh số tháng 3 trong file này", "data_query"),
]


# ---------------------------------------------------------------- chấm điểm


def _score(pairs: list[tuple[str, str]]) -> dict:
    """pairs = [(câu hỏi, lớp kỳ vọng)] → thống kê per-class."""
    per_class: dict[str, list[bool]] = {}
    confusion: Counter = Counter()
    misses: list[tuple[str, str, str]] = []

    for question, expected in pairs:
        got = classify_query(question)
        ok = got in (DATA_INTENTS if expected == "data_query" else {expected})
        per_class.setdefault(expected, []).append(ok)
        confusion[(expected, "data_*" if got in DATA_INTENTS else got)] += 1
        if not ok:
            misses.append((question, expected, got))

    recalls = {cls: sum(v) / len(v) for cls, v in per_class.items()}
    total = sum(len(v) for v in per_class.values())
    correct = sum(sum(v) for v in per_class.values())
    return {
        "accuracy": correct / total,
        "macro_recall": sum(recalls.values()) / len(recalls),
        "recalls": recalls,
        "per_class": {k: (sum(v), len(v)) for k, v in per_class.items()},
        "confusion": confusion,
        "misses": misses,
        "total": total,
        "correct": correct,
    }


def _eval_pairs() -> list[tuple[str, str]]:
    out = []
    for tc in TEST_CASES:
        exp = "data_query"
        if tc.category == "bot_info":
            exp = "bot_info"
        elif tc.category == "off_topic":
            exp = "off_topic"
        out.append((tc.question, exp))
    return out


# ---------------------------------------------------------------- ngưỡng

# Ngưỡng đặt SÁT DƯỚI kết quả đo được, để bắt hồi quy chứ không phải để
# tự khen. Nâng ngưỡng khi bộ phân loại thật sự tốt lên.
MIN_MACRO_RECALL_EVAL = 0.95
MIN_MACRO_RECALL_HELD_OUT = 0.95
MIN_DATA_RECALL = 1.0   # không bao giờ được đẩy nhầm câu hỏi dữ liệu đi nơi khác


def test_macro_recall_on_eval_set():
    s = _score(_eval_pairs())
    assert s["macro_recall"] >= MIN_MACRO_RECALL_EVAL, (
        f"macro-recall {s['macro_recall']:.3f} < {MIN_MACRO_RECALL_EVAL}; "
        f"per-class={s['per_class']}"
    )


def test_macro_recall_on_held_out():
    """Tập này không được dùng khi tinh chỉnh — nó đo khả năng khái quát hoá."""
    s = _score(HELD_OUT)
    assert s["macro_recall"] >= MIN_MACRO_RECALL_HELD_OUT, (
        f"macro-recall {s['macro_recall']:.3f} < {MIN_MACRO_RECALL_HELD_OUT}; "
        f"sai: {[m[0] for m in s['misses']]}"
    )


@pytest.mark.parametrize("pairs,name", [(_eval_pairs(), "eval"), (HELD_OUT, "held-out")])
def test_data_questions_never_misrouted(pairs, name):
    """Lỗi tốn kém nhất: câu hỏi dữ liệu bị coi là chit-chat và không được phân tích."""
    s = _score(pairs)
    recall = s["recalls"].get("data_query", 1.0)
    bad = [m for m in s["misses"] if m[1] == "data_query"]
    assert recall >= MIN_DATA_RECALL, f"[{name}] data recall {recall:.3f}; sai: {bad}"


def test_beats_majority_baseline():
    """Phải hơn bộ phân loại ngây thơ luôn trả `data_query`.

    Baseline đó đạt 85% accuracy trên bộ eval mà không phân loại được gì —
    nên accuracy trần là chỉ số vô nghĩa ở đây, phải so bằng macro-recall.
    """
    s = _score(_eval_pairs())
    baseline_macro = 1 / 3      # chỉ lớp data đúng, hai lớp kia recall = 0
    assert s["macro_recall"] > baseline_macro * 1.5


# ---------------------------------------------------------------- báo cáo


def _report(title: str, s: dict) -> None:
    print(f"\n{title}")
    print(f"  accuracy     : {s['correct']}/{s['total']} = {s['accuracy']*100:.1f}%"
          "   <- gây hiểu nhầm khi các lớp mất cân bằng")
    print(f"  macro-recall : {s['macro_recall']*100:.1f}%   <- chỉ số chính")
    for cls, (ok, tot) in sorted(s["per_class"].items()):
        bar = "█" * int(ok / tot * 20)
        print(f"     {cls:12s} {ok:>3}/{tot:<3} [{bar:<20}] {ok/tot*100:5.1f}%")
    if s["misses"]:
        print(f"  {len(s['misses'])} câu sai:")
        for q, exp, got in s["misses"]:
            print(f"     {exp:10s} -> {got:12s} | {q[:54]}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _report("BỘ EVAL (100 câu — dùng để tinh chỉnh)", _score(_eval_pairs()))
    _report("HELD-OUT (30 câu — không dùng khi tinh chỉnh)", _score(HELD_OUT))
