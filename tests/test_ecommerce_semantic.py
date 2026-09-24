"""Semantic layer TMĐT: metric tính sẵn phải khớp đáp án của bộ dữ liệu mô phỏng shop Chị Lan."""
import json
from pathlib import Path

import pandas as pd
import pytest

from backend.app.services.analysis_planner import _pick_metric_from_question, execute_plan
from backend.app.services.ecommerce_semantic import METRICS, add_metric_columns, describe_metrics
from backend.app.services.storage import SessionStore

SHOP = Path(__file__).resolve().parents[1] / "data" / "samples" / "shop_lan"


@pytest.fixture(scope="module")
def orders():
    # Đi đúng đường nạp file thật của app.
    return SessionStore._normalize_frame(pd.read_csv(SHOP / "shop_lan_orders.csv"))


@pytest.fixture(scope="module")
def truth():
    return json.loads((SHOP / "ground_truth.json").read_text(encoding="utf-8"))


def test_monthly_net_profit_matches_ground_truth(orders, truth):
    ads = pd.read_csv(SHOP / "ads_daily.csv", parse_dates=["ngay"])
    profit = orders.groupby(orders["ngay_dat"].dt.month)["loi_nhuan_truoc_qc"].sum()
    ad_spend = ads.groupby(ads["ngay"].dt.month)["chi_phi_qc"].sum()
    for month, expected in truth["monthly"].items():
        m = int(month)
        assert profit[m] - ad_spend[m] == pytest.approx(expected["loi_nhuan_rong"], abs=1)
        fee_rate = orders.loc[orders["ngay_dat"].dt.month == m, ["phi_san", "doanh_thu_thuan"]].sum()
        assert fee_rate["phi_san"] / fee_rate["doanh_thu_thuan"] == pytest.approx(expected["ty_le_phi"], abs=1e-4)


def test_loss_skus_found_with_plain_aggregate_plan(orders, truth):
    # Đúng loại plan LLM sẽ sinh: không công thức, chỉ sum cột semantic.
    plan = {"action": "aggregate", "group_by": ["sku"],
            "metrics": [{"column": "doanh_thu_thuan", "aggregation": "sum", "label": "Doanh thu"},
                        {"column": "loi_nhuan_truoc_qc", "aggregation": "sum", "label": "Lãi"}],
            "sort": [{"column": "Doanh thu", "direction": "desc"}], "limit": 10}
    top10 = execute_plan(orders, plan)
    assert set(top10.loc[top10["Lãi"] < 0, "sku"]) == {r["sku"] for r in truth["S2_loss_skus"]}


def test_month_filter_keeps_last_day_when_dates_have_time(orders, truth):
    # LLM sinh "between 2026-05-01..2026-05-31"; cột ngày có giờ → phải giữ cả ngày 31/5.
    plan = {"action": "compare_metrics",
            "filters": [{"column": "ngay_dat", "operator": "between", "value": ["2026-05-01", "2026-05-31"]}],
            "metrics": [{"column": "doanh_thu_thuan", "aggregation": "sum", "label": "dt"}]}
    got = execute_plan(orders, plan).iloc[0]["value"]
    assert got == truth["monthly"]["5"]["doanh_thu_thuan"]


def test_status_variants_from_tiktok_export():
    df = pd.DataFrame({"trang_thai": ["Completed", "Cancelled", "Returned"], "doanh_thu": [100, 100, 100],
                       "giam_gia_shop": [10, 0, 0], "phi_hoa_hong": [20, 0, 0], "phi_thanh_toan": [0, 0, 0],
                       "phi_voucher_freeship": [0, 0, 0], "phi_dich_vu": [0, 0, 0], "thue_khau_tru": [0, 0, 0],
                       "gia_von": [40, 40, 40], "chi_phi_hoan": [0, 0, 30]})
    out = add_metric_columns(df)
    assert out["doanh_thu_thuan"].tolist() == [90, 0, 0]
    assert out["loi_nhuan_truoc_qc"].tolist() == [30, 0, -30]


def test_leaves_other_tables_and_user_columns_alone():
    plain = pd.DataFrame({"revenue": [1, 2]})
    assert add_metric_columns(plain) is plain
    assert describe_metrics(plain) == ""


def test_profit_question_routes_to_semantic_metric(orders):
    assert _pick_metric_from_question("vi sao lai thang 5 giam", orders) == "loi_nhuan_truoc_qc"
    assert _pick_metric_from_question("tong phi san theo kenh", orders) == "phi_san"
    assert all(m in describe_metrics(orders) for m in METRICS)
