#!/usr/bin/env python3
"""
eval_calibration.py — đối chiếu heuristic scoring (eval_100.py) với nhãn gán thủ công.

3 chiều mềm của eval (`no_meta` bỏ qua vì gần như nhị phân, ít tranh cãi):
  insight · concise · vn_natural

Cách dùng:
    python tests/eval_calibration.py [--answers docs/eval-baseline/results.csv]

In ra, cho từng chiều: n, MAE, % agreement (|Δ| ≤ 0.25), Pearson r giữa
heuristic và nhãn tay. Đây là con số để đánh giá độ tin cậy của heuristic —
KHÔNG dùng để chỉnh heuristic cho khớp nhãn (overfit).

⚠️ 30 nhãn dưới đây do 1 người review từng câu trả lời thật của run 2026-09-10.
   Không phải "human eval" quy mô — chỉ đủ để sanity-check heuristic. Nếu trích
   dẫn con số r cho CV, tự review lại nhãn 1 lượt trước.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

import eval_100 as E

# id -> (insight, concise, vn_natural), thang 0.0–1.0
GOLD: dict[int, tuple[float, float, float]] = {
    1:  (0.75, 1.00, 1.00),
    3:  (0.75, 1.00, 1.00),
    6:  (0.75, 1.00, 1.00),
    8:  (1.00, 1.00, 1.00),
    10: (0.75, 1.00, 1.00),
    13: (0.75, 1.00, 1.00),
    16: (1.00, 1.00, 1.00),
    19: (0.75, 1.00, 1.00),
    22: (0.75, 1.00, 1.00),
    24: (1.00, 0.75, 1.00),
    26: (1.00, 1.00, 1.00),
    29: (0.75, 1.00, 1.00),
    33: (0.25, 1.00, 1.00),   # "không tìm thấy dữ liệu" — deflection
    38: (0.25, 0.75, 1.00),
    41: (0.75, 1.00, 1.00),
    44: (0.50, 0.25, 0.75),   # meta leak "Dựa trên context", template dài, cụt
    50: (1.00, 1.00, 1.00),   # caveat đa tiền tệ là insight thật
    55: (0.75, 1.00, 1.00),
    58: (0.75, 1.00, 1.00),
    62: (0.75, 1.00, 1.00),
    66: (0.75, 1.00, 1.00),
    70: (0.75, 1.00, 1.00),
    73: (1.00, 1.00, 1.00),
    76: (1.00, 0.75, 1.00),   # bot_info — redirect kèm ví dụ, tốt
    79: (0.00, 0.75, 0.50),   # hỏi "vẽ biểu đồ được không" → trả "không có dữ liệu" (sai) + ký tự lỗi
    81: (0.75, 0.75, 1.00),   # off_topic — redirect hơi gượng
    84: (0.50, 0.25, 0.75),   # cụt giữa câu
    88: (0.25, 0.00, 0.50),   # dump "# Executive Data Brief" + tường số
    92: (0.25, 0.00, 0.50),
    97: (0.00, 0.00, 0.50),   # dump, không trả lời đúng câu hỏi
}

DIMS = ["insight", "concise", "vn_natural"]


def load_answers(path: Path) -> dict[int, dict]:
    with open(path, encoding="utf-8") as f:
        return {int(r["id"]): r for r in csv.DictReader(f)}


def heuristic_scores(tc, row) -> dict[str, float]:
    answer = row.get("answer", "") or ""
    llm_fail = str(row.get("llm_failed", "")).lower() in ("true", "1")
    return {
        "insight": E.score_insight(tc, answer),
        "concise": E.score_concise(answer),
        "vn_natural": E.score_vn_natural(answer, llm_failed=llm_fail),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", default="docs/eval-baseline/results.csv")
    args = ap.parse_args()

    E._load_ground_truths()
    answers = load_answers(Path(args.answers))
    by_id = {tc.id: tc for tc in E.TEST_CASES}

    paired: dict[str, list[tuple[float, float]]] = {d: [] for d in DIMS}
    missing = []
    for cid, gold in GOLD.items():
        row = answers.get(cid)
        tc = by_id.get(cid)
        if row is None or tc is None:
            missing.append(cid)
            continue
        h = heuristic_scores(tc, row)
        for d, g in zip(DIMS, gold):
            paired[d].append((h[d], g))

    print("=" * 62)
    print(f"  CALIBRATION — heuristic vs {len(GOLD) - len(missing)} nhãn tay")
    if missing:
        print(f"  (thiếu answer cho id: {missing})")
    print("=" * 62)
    print(f"  {'chiều':<12} {'n':>3}  {'MAE':>6}  {'agree≤0.25':>11}  {'Pearson r':>10}")
    for d in DIMS:
        pr = paired[d]
        n = len(pr)
        mae = sum(abs(a - b) for a, b in pr) / n
        agree = sum(1 for a, b in pr if abs(a - b) <= 0.25) / n
        r = E.pearson([a for a, _ in pr], [b for _, b in pr])
        r_txt = " n/a" if r is None else f"{r:+.2f}"
        print(f"  {d:<12} {n:>3}  {mae:>6.3f}  {agree*100:>9.0f}%  {r_txt:>10}")
    print("=" * 62)

    # In các câu heuristic lệch nhãn nhiều nhất (>0.4) để soi
    print("\n  Lệch lớn (|Δ| > 0.4):")
    for cid, gold in GOLD.items():
        row, tc = answers.get(cid), by_id.get(cid)
        if not row or not tc:
            continue
        h = heuristic_scores(tc, row)
        for d, g in zip(DIMS, gold):
            if abs(h[d] - g) > 0.4:
                print(f"    #{cid:>3} {d:<11} heuristic={h[d]:.2f} gold={g:.2f}  | {row['question'][:45]}")


if __name__ == "__main__":
    main()
