"""Semantic layer TMĐT: metric tính sẵn phải khớp đáp án của bộ dữ liệu mô phỏng shop Chị Lan."""
import json
from pathlib import Path

import pandas as pd
import pytest

from backend.app.agents.nodes.plan import _ALLOWED_ACTIONS as _NODE_ACTIONS
from backend.app.services.analysis_planner import (
    ALLOWED_ACTIONS, _deterministic_answer, _pick_metric_from_question, _repair_who_plan,
    _validate_plan_against_dataframe, build_fallback_plan, execute_plan,
)
from backend.app.services.ecommerce_columns import detect_ecommerce_columns
from backend.app.services.ecommerce_semantic import (
    FEE_RATE_ROW, METRICS, PROFIT_ROW, RETURN_RATE_ROW, add_metric_columns, attach_cogs, bridge_actions,
    describe_metrics, fmt_pct, month_periods,
)
from backend.app.services.storage import SessionStore
from scripts.gen_shop_lan import PARAMS

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
    assert all(m in describe_metrics(orders) for m in METRICS if m in orders.columns)


@pytest.fixture(scope="module")
def from_exports():
    # Chị Lan tải lên đúng 3 file: 2 file export gốc của sàn + bảng giá vốn.
    store = SessionStore()
    files = ["export_shopee.csv", "export_tiktok.csv", "products.csv"]
    session = store.create_multiple([(f, (SHOP / f).read_bytes()) for f in files])
    store._sessions.pop(session.session_id)  # ép đi đường restore từ DB/đĩa
    return store.get(session.session_id)


def test_raw_exports_plus_catalog_give_same_numbers(from_exports, orders, truth):
    df = from_exports.dataframe
    assert from_exports.active_sheet == "__concat__" and len(df) == len(orders)
    assert df["kenh"].value_counts().to_dict() == orders["kenh"].value_counts().to_dict()
    # Ngày TikTok dạng dd/mm phải đọc đúng, không đảo ngày/tháng.
    dates = df.set_index("ma_don")["ngay_dat"]
    assert (dates.reindex(orders["ma_don"]).values == orders["ngay_dat"].dt.floor("s").values).all()

    month = df["ngay_dat"].dt.month
    for m, expected in truth["monthly"].items():
        got = df[month == int(m)]
        assert got["doanh_thu_thuan"].sum() == expected["doanh_thu_thuan"]
        assert got["phi_san"].sum() == pytest.approx(expected["ty_le_phi"] * expected["doanh_thu_thuan"], rel=1e-4)
    # File sàn không có phần hàng hỏng khi hoàn → lãi từ export cao hơn đúng bằng phần đó.
    returned = orders["trang_thai"] == "Đã trả hàng"
    damage = (orders["chi_phi_hoan"] - PARAMS["return_ship_cost"]).where(returned, 0).sum()
    assert df["loi_nhuan_truoc_qc"].sum() == pytest.approx(orders["loi_nhuan_truoc_qc"].sum() + damage, abs=1)
    assert from_exports.ecommerce_col_map["status_col"] == "trang_thai"
    assert "CẢNH BÁO" not in describe_metrics(df)


def test_normalized_orders_column_map_is_exact(orders):
    col_map = detect_ecommerce_columns(orders)
    assert col_map["status_col"] == "trang_thai" and col_map["revenue_col"] == "doanh_thu"
    assert "shop_col" not in col_map  # trước đây đoán substring ra "giam_gia_shop"


def test_unknown_sku_is_flagged_not_silently_costless():
    order = pd.DataFrame({"ma_don": ["A", "B"], "trang_thai": ["Hoàn thành"] * 2, "sku": ["X", "NEW"],
                          "so_luong": [2, 1], "doanh_thu": [100, 100], "giam_gia_shop": [0, 0],
                          **{c: [0, 0] for c in ("phi_hoa_hong", "phi_thanh_toan", "phi_voucher_freeship",
                                                 "phi_dich_vu", "thue_khau_tru")}})
    sheets = attach_cogs({"don": order, "sp": pd.DataFrame({"sku": [" X "], "gia_von": [30]})})
    out = sheets["don"]
    assert out["loi_nhuan_truoc_qc"].tolist()[0] == 40  # 100 − 2 × 30
    assert "1 đơn" in describe_metrics(out)


def test_profit_question_is_not_mistaken_for_who_question(orders):
    # "lãi" bỏ dấu = "lai" chứa "ai " → từng bị coi là câu hỏi "ai...?" và bị nhét group_by tên sản phẩm.
    plan = {"action": "aggregate", "metrics": [{"column": "loi_nhuan_truoc_qc", "aggregation": "sum"}]}
    assert "group_by" not in _repair_who_plan(plan, "Lãi tháng 5 theo kênh bán", orders)
    assert _repair_who_plan(plan, "Ai bán nhiều nhất?", orders)["group_by"]


def _row(result, name):
    return result.set_index("hang_muc").loc[name]


def test_profit_bridge_explains_fee_hike_s1(orders, truth):
    # Câu hỏi thật của chủ shop → fallback rule-based cũng phải ra đúng phép so tháng 5 với tháng 4.
    plan = build_fallback_plan(orders, "Vì sao lãi tháng 5 giảm?")
    _validate_plan_against_dataframe(orders, plan)
    result = execute_plan(orders, plan)
    s1 = truth["S1_fee_hike"]
    p4, p5 = result.columns[1:3]
    parts = result.head(4)["anh_huong_lai"].sum()
    assert parts == pytest.approx(_row(result, PROFIT_ROW)["anh_huong_lai"], abs=1)  # tách ra cộng lại đúng Δ lãi
    fee = _row(result, FEE_RATE_ROW)
    assert (fee[p4], fee[p5]) == (s1["ty_le_phi_thang_4"], s1["ty_le_phi_thang_5"])
    assert -fee["anh_huong_lai"] == pytest.approx(s1["phi_tang_them_thang_5"], rel=0.01)
    hints = bridge_actions(result)
    assert fmt_pct(s1["tang_gia_de_giu_tien_ve_sau_phi"]) in hints[0] and "CHƯA trừ quảng cáo" in hints[-1]
    assert "Nên kiểm tra" in _deterministic_answer("Vì sao lãi tháng 5 giảm?", result, plan, orders)


def test_profit_bridge_finds_return_spike_segment_s4(orders, truth):
    # Tháng 3 tổng lãi TĂNG nhưng một tỉnh × đơn vị vận chuyển hoàn tăng vọt — vẫn phải chỉ ra nó.
    s4 = truth["S4_return_spike"]
    plan = {"action": "profit_bridge", "periods": month_periods(2026, 3), "group_by": ["tinh", "don_vi_van_chuyen"]}
    result = execute_plan(orders, plan)
    rate = _row(result, RETURN_RATE_ROW)
    assert (rate.iloc[0], rate.iloc[1]) == (s4["ty_le_hoan_thang_2"], s4["ty_le_hoan_thang_3"])
    drags = result[result["hang_muc"].str.contains(":") & (result["anh_huong_lai"] < 0)]
    seg = s4["phan_khuc_gay_tang"]
    assert drags.iloc[0]["hang_muc"].endswith(f"{seg['tinh']} × {seg['don_vi_van_chuyen']}")
    assert any("Tỷ lệ hoàn tăng" in h for h in bridge_actions(result))


def test_profit_bridge_rejects_malformed_periods(orders):
    with pytest.raises(ValueError):
        _validate_plan_against_dataframe(orders, {"action": "profit_bridge", "periods": [["tháng 4"], ["2026-05-01"]]})


def test_planner_and_agent_node_allow_same_actions():
    assert _NODE_ACTIONS == ALLOWED_ACTIONS  # CLAUDE.md: thêm action phải sửa cả hai chỗ


def test_profit_bridge_warns_when_a_period_has_no_data(orders):
    result = execute_plan(orders, {"action": "profit_bridge", "periods": month_periods(2026, 7)})
    assert bridge_actions(result)[0].startswith("CẢNH BÁO: kỳ T7/2026")
