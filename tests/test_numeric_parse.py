"""Kiểm tra bộ đọc số (services/numeric_parse.py)."""
from __future__ import annotations

import pandas as pd
import pytest

from backend.app.services.numeric_parse import (
    coerce_numeric_columns,
    parse_number,
    parse_numeric_series,
    strip_column_names,
)


@pytest.mark.parametrize(
    "raw,expected",
    [
        # dạng có thật trong data/samples/financial_sample.csv
        (" $ 16,185.00 ", 16185.00),
        (" $ (4,533.75) ", -4533.75),
        (" $ -   ", 0.0),
        ("$1,234.56", 1234.56),
        # định dạng châu Âu
        ("1.234,56", 1234.56),
        ("1.234.567,89", 1234567.89),
        # tiền Việt
        ("32.000.000 VNĐ", 32000000.0),
        ("1500đ", 1500.0),
        # số thường
        (42, 42.0),
        (3.14, 3.14),
        ("-17", -17.0),
        ("1e3", 1000.0),
        ("0", 0.0),
        # dấu phẩy thập phân đứng một mình
        ("1,5", 1.5),
        ("1,234", 1234.0),      # 3 chữ số sau dấu phẩy → phân cách nghìn
    ],
)
def test_parse_number_handles_real_formats(raw, expected):
    assert parse_number(raw) == pytest.approx(expected)


@pytest.mark.parametrize(
    "raw",
    [
        "Carretera", "Government", "abc123", "N/A", "", None, float("nan"), "--", "Paseo",
        # Mã tiền tệ đứng một mình là DỮ LIỆU PHÂN LOẠI, không phải số 0.
        # Từng khiến cả cột Currency của general_ledger bị ép thành 0.0.
        "GBP", "EUR", "AUD", "USD", "CAD", "VND", "usd",
    ],
)
def test_non_numeric_returns_none_not_zero(raw):
    """Trả 0.0 cho chữ sẽ biến cột tên sản phẩm thành cột toàn 0."""
    assert parse_number(raw) is None


def test_series_keeps_numeric_dtype_untouched():
    s = pd.Series([1, 2, 3])
    assert parse_numeric_series(s).equals(s)


def test_series_parses_currency_strings():
    s = pd.Series([" $ 1,000.00 ", " $ (250.50) ", " $ -   "])
    out = parse_numeric_series(s)
    assert out.tolist() == [1000.0, -250.5, 0.0]


def test_coerce_skips_text_columns():
    df = pd.DataFrame(
        {
            "product": ["Carretera", "Paseo", "Velo"],
            "amount": [" $ 1,000.00 ", " $ (250.50) ", " $ 30.00 "],
        }
    )
    out = coerce_numeric_columns(df.copy())
    assert out["product"].tolist() == ["Carretera", "Paseo", "Velo"]
    assert out["amount"].tolist() == [1000.0, -250.5, 30.0]


def test_coerce_leaves_mostly_text_column_alone():
    """Cột chữ lẫn vài số không được ép kiểu (dưới ngưỡng 90%)."""
    df = pd.DataFrame({"mixed": ["abc", "def", "ghi", "jkl", "5"]})
    out = coerce_numeric_columns(df.copy())
    assert out["mixed"].tolist() == ["abc", "def", "ghi", "jkl", "5"]


def test_strip_column_names():
    df = pd.DataFrame({" Profit ": [1], "  Sales ": [2], "Year": [3]})
    out = strip_column_names(df.copy())
    assert list(out.columns) == ["Profit", "Sales", "Year"]


def test_strip_column_names_keeps_original_on_collision():
    """Cắt khoảng trắng mà gây trùng tên thì thà giữ tên xấu còn hơn mất cột."""
    df = pd.DataFrame({" A ": [1], "A": [2]})
    out = strip_column_names(df.copy())
    assert list(out.columns) == [" A ", "A"]


def test_financial_sample_totals_match_manual_sum():
    """Chốt chặn trên dữ liệu thật: 63/700 ô từng bị bỏ qua."""
    df = pd.read_csv("data/samples/financial_sample.csv")
    df = strip_column_names(df)
    df = coerce_numeric_columns(df)

    profit = df["Profit"]
    assert profit.notna().sum() == 700, "phải đọc được toàn bộ 700 dòng"
    assert (profit < 0).sum() == 58, "58 dòng ngoặc đơn phải thành số âm"
    assert profit.sum() == pytest.approx(16_893_702.29, abs=0.01)


# ── repair: câu hỏi về toàn bộ dataset ─────────────────────────────────────

from backend.app.services.analysis_planner import _repair_whole_dataset_aggregate  # noqa: E402


def _plan(**kw):
    base = {
        "action": "aggregate",
        "group_by": ["AccountName"],
        "metrics": [{"column": "Debit", "aggregation": "sum", "label": "Tổng"}],
        "sort": [{"column": "Tổng", "direction": "desc"}],
        "limit": 1,
    }
    base.update(kw)
    return base


@pytest.mark.parametrize(
    "question",
    [
        "Tổng Debit của toàn bộ sổ cái là bao nhiêu?",
        "Tổng Credit của toàn bộ sổ cái là bao nhiêu?",
        "Tất cả doanh thu cộng lại là bao nhiêu?",
        "What is the total revenue of all records?",
    ],
)
def test_whole_dataset_question_drops_group_by(question):
    out = _repair_whole_dataset_aggregate(_plan(), question)
    assert "group_by" not in out, f"vẫn còn group_by với câu: {question}"
    assert "sort" not in out


@pytest.mark.parametrize(
    "question",
    [
        "Tổng doanh thu theo vùng của toàn bộ dữ liệu",   # có "theo" → vẫn nhóm
        "Sản phẩm nào mang lại lợi nhuận cao nhất?",       # ranking
        "Top 5 tài khoản có Debit lớn nhất",
        "Doanh thu mỗi chi nhánh",
        "Revenue by segment",
    ],
)
def test_grouping_questions_keep_group_by(question):
    out = _repair_whole_dataset_aggregate(_plan(), question)
    assert out.get("group_by") == ["AccountName"], f"mất group_by ở câu: {question}"


def test_repair_ignores_plans_without_group_by():
    plan = _plan()
    plan.pop("group_by")
    assert _repair_whole_dataset_aggregate(plan, "tổng toàn bộ") == plan


def test_repair_ignores_non_aggregate_actions():
    plan = _plan(action="time_series")
    assert _repair_whole_dataset_aggregate(plan, "tổng toàn bộ") == plan


# ── repair: nhóm theo tên thay vì mã ────────────────────────────────────────

from backend.app.services.analysis_planner import _repair_id_to_name_group  # noqa: E402

_PRODUCTS = pd.DataFrame(
    {"product_id": ["P001"], "product_name": ["MacBook"], "unit_price": [1]}
)


def test_group_by_id_switches_to_name():
    out = _repair_id_to_name_group(
        {"action": "aggregate", "group_by": ["product_id"]},
        "Sản phẩm nào có giá bán cao nhất?",
        _PRODUCTS,
    )
    assert out["group_by"] == ["product_name"]


def test_keeps_id_when_user_asks_for_id():
    out = _repair_id_to_name_group(
        {"action": "aggregate", "group_by": ["product_id"]},
        "Mã sản phẩm nào có giá cao nhất?",
        _PRODUCTS,
    )
    assert out["group_by"] == ["product_id"]


def test_keeps_id_when_no_name_column_exists():
    df = pd.DataFrame({"product_id": ["P001"], "unit_price": [1]})
    plan = {"action": "aggregate", "group_by": ["product_id"]}
    assert _repair_id_to_name_group(plan, "sản phẩm nào", df)["group_by"] == ["product_id"]


def test_non_id_group_by_untouched():
    df = pd.DataFrame({"region": ["Bắc"], "sales": [1]})
    plan = {"action": "aggregate", "group_by": ["region"]}
    assert _repair_id_to_name_group(plan, "doanh thu theo vùng", df) == plan


def test_currency_code_column_is_never_coerced():
    """Chốt chặn hồi quy: cột mã tiền tệ phải giữ nguyên là chữ."""
    df = pd.DataFrame({"Currency": ["GBP", "EUR", "AUD", "USD", "CAD"] * 4})
    out = coerce_numeric_columns(df.copy())
    assert out["Currency"].tolist() == ["GBP", "EUR", "AUD", "USD", "CAD"] * 4
    assert out["Currency"].nunique() == 5


def test_amount_with_currency_suffix_still_parses():
    """Sửa lỗi trên không được làm hỏng việc đọc số có kèm đơn vị."""
    assert parse_number("32.000.000 VNĐ") == pytest.approx(32_000_000.0)
    assert parse_number("1500đ") == pytest.approx(1500.0)
    assert parse_number("250 USD") == pytest.approx(250.0)


# ── grounding: mã định danh không phải số liệu ─────────────────────────────

from backend.app.agents.nodes.synthesize import _numbers_grounded, _parse_numbers  # noqa: E402


@pytest.mark.parametrize(
    "text,expected",
    [
        # Mã định danh KHÔNG được đọc thành số liệu
        ("Giao dịch GL001279 có Debit 1.999,52", [1999.52]),
        ("Mã ORD001 với tổng 8,950.20", [8950.2]),
        ("sản phẩm P002 giá 32.000.000", [32000000.0]),
        # Định dạng số vẫn phải đọc đúng
        ("doanh thu 859.045.000 đồng", [859045000.0]),
        ("số 1.900 sản phẩm", [1900.0]),
    ],
)
def test_grounding_parser_ignores_identifiers(text, expected):
    assert _parse_numbers(text) == expected


def test_percentages_are_dropped():
    """Phần trăm là số dẫn xuất, không phải giá trị trong bảng kết quả."""
    assert _parse_numbers("tỷ lệ 21,23% và số 1.900") == [1900.0]


def test_answer_citing_transaction_id_is_grounded():
    """Hồi quy cho [60]: mã GL001279 từng bị đọc thành 1279 → grounding từ chối
    câu trả lời đúng → agent rơi xuống bản dự phòng bị chấm điểm thấp."""
    df = pd.DataFrame({"GLID": ["GL001279"], "Debit": [1999.52]})
    answer = "Giao dịch GL001279 có giá trị Debit lớn nhất, đạt 1.999,52."
    assert _numbers_grounded(answer, df) is True


def test_hallucinated_vn_number_still_rejected():
    """Sửa lỗi trên không được làm thủng lớp chống bịa số."""
    df = pd.DataFrame({"GLID": ["GL001279"], "Debit": [1999.52]})
    answer = "Giao dịch GL001279 có giá trị Debit lớn nhất, đạt 9.876.543,21."
    assert _numbers_grounded(answer, df) is False


def test_negative_result_cited_as_decrease_is_grounded():
    """Bảng có Δ lãi −11.567.470; LLM viết "giảm 11.567.470" (dấu nằm ở chữ "giảm") — không được từ chối."""
    df = pd.DataFrame({"hang_muc": ["Lãi trước QC"], "anh_huong_lai": [-11567470.0]})
    assert _numbers_grounded("Lãi tháng 5 giảm 11.567.470 so với tháng 4.", df) is True
    assert _numbers_grounded("Lãi tháng 5 giảm 21.567.470 so với tháng 4.", df) is False
