"""T05/T06/T14: plan sai kiểu/sai cấu trúc bị chặn ở validate, không ra số sai trông như đúng.

Trước đây: filter "10" (chuỗi) trên cột số → 0 dòng → "không có dữ liệu"; 2 metric
"Lãi %" và "Lãi" trùng tên sau chuẩn hoá → mất 1 metric; grain "week" → nhóm theo tháng
nhưng cột ghi "week".
"""
import pandas as pd
import pytest

from backend.app.services.planner.execute import (
    PlanValidationError,
    _validate_plan_against_dataframe,
    execute_plan,
)

DF = pd.DataFrame({
    "kenh": ["S", "S", "T"],
    "ma_don": ["101", "102", "103"],
    "lai": [10.0, 20.0, 5.0],
    "phi": [1.0, 2.0, 3.0],
    "ngay": pd.to_datetime(["2026-05-01", "2026-05-09", "2026-06-02"]),
})


def _agg(**kw) -> dict:
    return {"action": "aggregate", "metrics": [{"column": "lai", "aggregation": "sum", "label": "Tổng lãi"}], **kw}


def _run(plan: dict) -> pd.DataFrame:
    _validate_plan_against_dataframe(DF, plan)
    return execute_plan(DF, plan)


@pytest.mark.parametrize("filters, expected", [
    ([{"column": "lai", "operator": "eq", "value": "10"}], 10),              # chuỗi số trên cột số
    ([{"column": "lai", "operator": "between", "value": ["5", "10"]}], 15),
    ([{"column": "lai", "operator": "in", "value": ["10", 20]}], 30),
    ([{"column": "ma_don", "operator": "eq", "value": 101}], 10),            # số trên cột chữ
])
def test_filter_value_is_coerced_to_column_type(filters, expected):
    assert _run(_agg(filters=filters))["Tổng lãi"].tolist() == [expected]


@pytest.mark.parametrize("filters", [
    [{"column": "lai", "operator": "eq", "value": "mười"}],
    [{"column": "lai", "operator": "gt", "value": None}],
    [{"column": "ngay", "operator": "gte", "value": "đầu tháng"}],
])
def test_uncoercible_filter_value_is_rejected(filters):
    with pytest.raises(PlanValidationError):
        _validate_plan_against_dataframe(DF, _agg(filters=filters))
    with pytest.raises(PlanValidationError):
        execute_plan(DF, _agg(filters=filters))


@pytest.mark.parametrize("plan", [
    {"action": "aggregate", "group_by": ["kenh"], "metrics": [
        {"column": "lai", "aggregation": "sum", "label": "Lãi %"},
        {"column": "phi", "aggregation": "sum", "label": "Lãi"}]},
    {"action": "aggregate", "group_by": ["kenh"], "metrics": [{"column": "lai", "label": "kenh"}]},
    {"action": "time_series", "time_column": "ngay", "grain": "month",
     "metrics": [{"column": "lai", "label": "month"}]},
], ids=["labels-collide", "label-is-group-column", "label-is-period-column"])
def test_colliding_output_columns_are_rejected(plan):
    with pytest.raises(PlanValidationError):
        _validate_plan_against_dataframe(DF, plan)
    with pytest.raises(PlanValidationError):
        execute_plan(DF, plan)


@pytest.mark.parametrize("grain", ["week", "day", "hour"])
def test_unknown_grain_is_rejected(grain):
    plan = {"action": "time_series", "time_column": "ngay", "grain": grain, "metrics": [{"column": "lai"}]}
    with pytest.raises(PlanValidationError):
        execute_plan(DF, plan)


def test_validation_error_still_triggers_planner_fallback():
    # plan_node bắt ValueError để rơi xuống rule-based — lỗi mới không được thoát khỏi lưới đó.
    assert issubclass(PlanValidationError, ValueError)
