"""Lời hứa "không có số sai trông như đúng" cho seller, trên 3 cách nạp shop mô phỏng (UX feedback 2026-09-26).

A = đủ file (2 export + giá vốn + quảng cáo), B = chỉ 2 export (không giá vốn), C = export + giá vốn thiếu 1/3 SKU.
LLM bị tắt: mọi câu trả lời đi đường tất định, nên chạy được offline trong CI.
"""
import io
import json
from pathlib import Path

import pandas as pd
import pytest

from backend.app.agents.nodes.synthesize import _no_rows_answer
from backend.app.services.analysis_planner import _repair_filter_values, build_fallback_plan, execute_plan
from backend.app.services.ecommerce_semantic import asks_profit, cogs_gap
from backend.app.services.storage import SessionStore
from scripts.gen_shop_lan import PARAMS

SHOP = Path(__file__).resolve().parents[1] / "data" / "samples" / "shop_lan"
EXPORTS = ["export_shopee.csv", "export_tiktok.csv"]


def _load(files: list[tuple[str, bytes]]):
    store = SessionStore()
    session = store.create_multiple(files)
    return store, session


def _files(*names):
    return [(n, (SHOP / n).read_bytes()) for n in names]


def _partial_catalog() -> tuple[str, bytes]:
    products = pd.read_csv(SHOP / "products.csv")
    buf = io.BytesIO()
    products.head(100).to_csv(buf, index=False)  # 150 SKU → thiếu 50
    return "gia_von.csv", buf.getvalue()


@pytest.fixture(scope="module")
def full():
    return _load(_files(*EXPORTS, "products.csv", "ads_daily.csv"))[1]


@pytest.fixture(scope="module")
def no_cogs():
    return _load(_files(*EXPORTS))[1]


@pytest.fixture(scope="module")
def partial():
    return _load(_files(*EXPORTS) + [_partial_catalog()])[1]


@pytest.fixture
def offline(monkeypatch):
    """Tắt LLM: plan rơi xuống rule-based, synthesize rơi xuống câu tất định."""
    from backend.app.agents.nodes import respond

    def boom():
        raise RuntimeError("LLM disabled in test")

    monkeypatch.setattr("backend.app.services.llm_service.get_llm_client", boom)
    monkeypatch.setattr(respond, "get_llm_client", boom)


def _ask(session, question):
    from backend.app.agents.runner import run
    from backend.app.services.storage import session_store
    session_store._sessions[session.session_id] = session
    return run(session.session_id, question, [])


# ── B: không có giá vốn → từ chối có hướng dẫn, không bao giờ gọi doanh thu là lãi ─────────────

@pytest.mark.parametrize("question", ["Tổng lãi 6 tháng là bao nhiêu?", "sku nao lo nhat", "Lãi theo kênh"])
def test_profit_without_cogs_is_refused_not_guessed(no_cogs, offline, question):
    out = _ask(no_cogs, question)
    assert out.intent == "needs_cogs" and out.source == "deterministic"
    assert "Chưa tính được lãi" in out.answer and "sku, gia_von" in out.answer
    revenue = no_cogs.dataframe["doanh_thu"].sum()
    assert f"{revenue:,.0f}".replace(",", ".") not in out.answer


def test_without_cogs_fee_and_revenue_still_answerable(no_cogs):
    df = no_cogs.dataframe
    assert {"doanh_thu_thuan", "phi_san", "la_don_hoan"} <= set(df.columns)
    assert "loi_nhuan_truoc_qc" not in df.columns
    assert cogs_gap(df)["all_missing"]


@pytest.mark.parametrize("question, expected", [
    ("Vì sao lãi tháng 5 giảm?", True), ("sku nao ban chay ma lai lo", True), ("lợi nhuận theo kênh", True),
    ("Xem lại doanh thu tháng 5", False), ("phí sàn theo kênh", False), ("doanh thu thuan thang 5", False),
])
def test_asks_profit_does_not_trip_on_lai_meaning_again(question, expected):
    # "xem lại" bỏ dấu = "xem lai" — không được chặn câu hỏi doanh thu.
    assert asks_profit(question) is expected


# ── C: thiếu giá vốn một phần → mọi câu có lãi đều kèm cảnh báo tất định ─────────────────────

def test_partial_cogs_warning_is_appended(partial, offline):
    out = _ask(partial, "Tổng lãi theo kênh")
    assert "chưa có giá vốn" in out.answer and "CAO hơn thực tế" in out.answer
    assert "TRƯỚC quảng cáo" in out.answer
    gap = cogs_gap(partial.dataframe)
    assert len(gap["skus"]) == 50 and "LAN-127" in gap["skus"]  # SKU lỗ nặng nhất nằm trong phần thiếu


# ── A: quảng cáo được ghép → lãi ròng khớp đáp án ────────────────────────────────────────────

def test_net_profit_after_ads_matches_ground_truth(full):
    truth = json.loads((SHOP / "ground_truth.json").read_text(encoding="utf-8"))
    orders = pd.read_csv(SHOP / "shop_lan_orders.csv", parse_dates=["ngay_dat"])
    # File export không có phần hàng hỏng khi hoàn (test_raw_exports_plus_catalog_give_same_numbers).
    damage = (orders["chi_phi_hoan"] - PARAMS["return_ship_cost"]).where(orders["trang_thai"] == "Đã trả hàng", 0)
    damage = damage.groupby(orders["ngay_dat"].dt.month).sum()
    df = full.dataframe
    net = df.groupby(df["ngay_dat"].dt.month)["loi_nhuan_rong"].sum()
    for month, expected in truth["monthly"].items():
        assert net[int(month)] == pytest.approx(expected["loi_nhuan_rong"] + damage[int(month)], abs=2)
    ads = pd.read_csv(SHOP / "ads_daily.csv")["chi_phi_qc"].sum()
    assert df["chi_phi_qc"].sum() == pytest.approx(ads, abs=1)


def test_shop_summary_says_what_is_linked(full, no_cogs, offline):
    out = _ask(full, "file này có gì vậy em?")
    assert out.source == "deterministic" and "Quảng cáo: đã ghép" in out.answer
    assert "chưa tính được lãi" in _ask(no_cogs, "file này có gì?").answer


# ── Tỷ lệ, bộ lọc, thời gian, không bịa ─────────────────────────────────────────────────────

def test_fee_rate_by_channel_is_ratio_of_sums(full):
    plan = {"action": "aggregate", "group_by": ["kenh"],
            "metrics": [{"column": "phi_san", "aggregation": "sum", "label": "Phí"},
                        {"column": "doanh_thu_thuan", "aggregation": "sum", "label": "DT"}],
            "ratios": [{"label": "Tỷ lệ phí", "numerator": "Phí", "denominator": "DT"}],
            "sort": [{"column": "Tỷ lệ phí", "direction": "desc"}]}
    got = execute_plan(full.dataframe, plan).set_index("kenh")["Tỷ lệ phí"]
    sums = full.dataframe.groupby("kenh")[["phi_san", "doanh_thu_thuan"]].sum()
    assert got.to_dict() == pytest.approx((sums["phi_san"] / sums["doanh_thu_thuan"]).round(4).to_dict())


def test_ratio_must_reference_plan_metrics(full):
    from backend.app.services.analysis_planner import _validate_plan_against_dataframe
    with pytest.raises(ValueError):
        _validate_plan_against_dataframe(full.dataframe, {
            "action": "aggregate", "metrics": [{"column": "phi_san", "aggregation": "sum", "label": "Phí"}],
            "ratios": [{"label": "x", "numerator": "Phí", "denominator": "không có"}]})


def test_filter_value_matches_real_category(full):
    plan = {"action": "aggregate", "filters": [{"column": "kenh", "operator": "eq", "value": "Tiktok"}],
            "metrics": [{"column": "phi_san", "aggregation": "sum"}]}
    assert _repair_filter_values(plan, full.dataframe)["filters"][0]["value"] == "TikTok Shop"


def test_last_month_is_anchored_to_data(full):
    plan = build_fallback_plan(full.dataframe, "vì sao lãi tháng trước giảm")
    assert plan["periods"] == [["2026-04-01", "2026-04-30"], ["2026-05-01", "2026-05-31"]]


def test_out_of_range_period_says_data_range(full):
    plan = {"action": "aggregate",
            "filters": [{"column": "ngay_dat", "operator": "between", "value": ["2026-07-01", "2026-07-31"]}],
            "metrics": [{"column": "loi_nhuan_truoc_qc", "aggregation": "sum"}]}
    # sum() trên 0 dòng từng trả 0 → "lãi tháng 7 là 0 đ". Phải ra bảng rỗng để synthesize nói khoảng dữ liệu.
    assert execute_plan(full.dataframe, plan).empty
    assert "đến 30/06/2026" in _no_rows_answer(full.dataframe, plan)


def test_market_price_question_is_off_topic(full, offline):
    out = _ask(full, "giá vàng hôm nay bao nhiêu")
    assert out.intent == "off_topic" and not any(ch.isdigit() for ch in out.answer)


def test_seller_endpoints(client):
    body = client.post("/api/sample-shop").json()
    assert body["suggested_queries"][0] == "Vì sao lãi tháng này giảm?" and body["cogs_missing"] == 0
    up = client.post("/api/upload", files=[("files", (n, b, "text/csv")) for n, b in _files(*EXPORTS)]).json()
    assert up["cogs_missing"] == 150 and "Chưa có giá vốn" in up["data_notes"][0]
    csv = client.get(f"/api/session/{up['session_id']}/cogs-template.csv")
    template = pd.read_csv(io.BytesIO(csv.content), encoding="utf-8-sig")
    assert list(template.columns) == ["sku", "ten_san_pham", "doanh_thu_thuan", "gia_von"] and len(template) == 150
    dash = client.get(f"/api/dashboard/{up['session_id']}").json()
    assert dash["platform"] == "Shopee + TikTok Shop"
    assert any(k["value"] == "Chưa có giá vốn" and k["is_alert"] for k in dash["kpi_cards"])
