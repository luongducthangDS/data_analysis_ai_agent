"""Lời hứa "không có số sai trông như đúng" cho seller, trên 3 cách nạp shop mô phỏng (UX feedback 2026-09-26).

A = đủ file (2 export + giá vốn + quảng cáo), B = chỉ 2 export (không giá vốn), C = export + giá vốn thiếu 1/3 SKU.
LLM bị tắt: mọi câu trả lời đi đường tất định, nên chạy được offline trong CI.
"""
import io
import json
import re
from pathlib import Path

import pandas as pd
import pytest

from backend.app.agents.nodes.synthesize import _claims, _no_rows_answer, _numbers_grounded
from backend.app.services.planner.answer import _deterministic_answer
from backend.app.services.planner.execute import execute_plan
from backend.app.services.planner.fallback import build_fallback_plan
from backend.app.services.planner.llm_plan import (
    _repair_filter_values, _repair_ratio_denominator, _repair_status_filter,
)
from backend.app.services.ecommerce_semantic import (
    NO_CATEGORY, answer_notes, asks_profit, category_notes, cogs_gap, fmt_num, fmt_pct, missing_category_notice,
    profit_notes, seller_questions,
)
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

    def boom(*_):
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

# ── Lỗi chung mọi model (eval_seller 2026-10-01) ────────────────────────────────────────────

def test_tiktok_status_uses_the_same_values_as_shopee(full):
    df = full.dataframe
    assert set(df["trang_thai"].unique()) == {"Hoàn thành", "Đã trả hàng", "Đã huỷ"}
    plan = {"action": "aggregate", "filters": [{"column": "trang_thai", "operator": "eq", "value": "Hoàn thành"}],
            "metrics": [{"column": "ma_don", "aggregation": "count", "label": "n"}]}
    orders = pd.read_csv(SHOP / "shop_lan_orders.csv")
    # Từng ra 6.307 = riêng Shopee: đơn TikTok mang trạng thái "Completed" (#101, #103, #142).
    assert execute_plan(df, plan)["n"].iloc[0] == (orders["trang_thai"] == "Hoàn thành").sum()


def test_category_comes_from_the_product_table(full, partial):
    orders = pd.read_csv(SHOP / "shop_lan_orders.csv")
    got = full.dataframe.groupby("sku")["danh_muc"].first()
    assert got.to_dict() == orders.groupby("sku")["danh_muc"].first().to_dict()
    # Bảng giá vốn thiếu 50 SKU: các SKU đó vào nhóm riêng, tổng theo danh mục không rơi mất doanh thu nào.
    df = partial.dataframe
    assert set(df.loc[df["danh_muc"] == NO_CATEGORY, "sku"]) == set(cogs_gap(df)["skus"])
    assert df.groupby("danh_muc")["doanh_thu_thuan"].sum().sum() == pytest.approx(df["doanh_thu_thuan"].sum())


def test_category_question_without_product_table_is_refused(full, no_cogs, offline):
    out = _ask(no_cogs, "Doanh thu thuần theo danh mục")
    assert out.source == "deterministic" and "Chưa có danh mục" in out.answer
    assert not any(ch.isdigit() for ch in out.answer)  # không thay danh mục bằng tên sản phẩm rồi đưa số (#60)
    assert missing_category_notice(no_cogs.dataframe, "Doanh thu thuần theo kênh") is None
    assert missing_category_notice(full.dataframe, "Lãi theo danh mục") is None


def test_rate_denominator_is_net_revenue(full):
    df = full.dataframe
    plan = {"action": "aggregate", "group_by": ["kenh"],
            "metrics": [{"column": "phi_san", "aggregation": "sum", "label": "Phí"},
                        {"column": "doanh_thu", "aggregation": "sum"}],  # không label: tham chiếu bằng tên mặc định
            "ratios": [{"label": "Tỷ lệ phí", "numerator": "Phí", "denominator": "sum_doanh_thu"}]}
    got = execute_plan(df, _repair_ratio_denominator(plan, df)).set_index("kenh")["Tỷ lệ phí"]
    sums = df.groupby("kenh")[["phi_san", "doanh_thu_thuan"]].sum()
    # Chia cho doanh_thu gộp cả đơn huỷ/hoàn từng ra TikTok 27,0% thay vì 34,5% (#141).
    assert got.to_dict() == pytest.approx((sums["phi_san"] / sums["doanh_thu_thuan"]).round(4).to_dict())
    plain = {"action": "aggregate", "metrics": [{"column": "doanh_thu", "aggregation": "sum"}]}
    assert _repair_ratio_denominator(plain, df) == plain


def test_profit_ignores_completed_only_filter(full):
    df = full.dataframe
    tiktok = {"column": "kenh", "operator": "eq", "value": "TikTok Shop"}
    done = {"column": "trang_thai", "operator": "eq", "value": "Hoàn thành"}
    plan = {"action": "aggregate", "filters": [tiktok, done],
            "metrics": [{"column": "loi_nhuan_rong", "aggregation": "sum", "label": "Lãi ròng"}]}
    got = execute_plan(df, _repair_status_filter(plan, df))["Lãi ròng"].iloc[0]
    # Lọc đơn hoàn thành bỏ mất phí ship của đơn hoàn: TikTok ra 74,6 tr thay vì 59,8 tr (#41).
    assert got == pytest.approx(df.loc[df["kenh"] == "TikTok Shop", "loi_nhuan_rong"].sum())
    count = {"action": "aggregate", "filters": [done], "metrics": [{"column": "ma_don", "aggregation": "count"}]}
    assert _repair_status_filter(count, df) == count  # "bao nhiêu đơn hoàn thành" vẫn cần filter


@pytest.mark.parametrize("question, keeps", [
    ("Lãi của các đơn hoàn thành tháng 6", True), ("lỗ từ đơn bị huỷ", True), ("lai don da giao", True),
    ("lai tiktok thang 6", False), ("Lãi tháng 5 giảm vì hoàn hàng?", False),
])
def test_status_filter_kept_only_when_the_question_names_a_status(full, question, keeps):
    done = {"column": "trang_thai", "operator": "eq", "value": "Hoàn thành"}
    plan = {"action": "aggregate", "filters": [done], "metrics": [{"column": "loi_nhuan_truoc_qc"}]}
    assert (done in _repair_status_filter(plan, full.dataframe, question)["filters"]) is keeps
    # Tỷ lệ hoàn trên riêng đơn hoàn thành luôn là 0%: bỏ filter kể cả khi câu hỏi nêu trạng thái.
    rate = {"action": "aggregate", "filters": [done], "metrics": [{"column": "la_don_hoan", "aggregation": "mean"}]}
    assert _repair_status_filter(rate, full.dataframe, question)["filters"] == []


def test_category_answer_states_the_unknown_share(full, partial):
    df = partial.dataframe
    by_category = {"action": "aggregate", "group_by": ["danh_muc"], "metrics": [{"column": "doanh_thu_thuan"}]}
    unknown = df["danh_muc"] == NO_CATEGORY
    share = df.loc[unknown, "doanh_thu_thuan"].sum() / df["doanh_thu_thuan"].sum()
    [note] = category_notes(df, by_category)
    assert fmt_pct(share) in note and fmt_num(unknown.sum()) in note
    assert category_notes(df, {"group_by": ["kenh"]}) == []
    assert category_notes(full.dataframe, by_category) == []  # đủ danh mục thì không cảnh báo


def test_note_numbers_all_come_from_the_data(partial):
    # Trích số từ ghi chú chỉ an toàn khi ghi chú không chứa số nào ngoài số tính từ dữ liệu của request.
    df = partial.dataframe
    plan = {"group_by": ["danh_muc"], "metrics": [{"column": "loi_nhuan_truoc_qc"}]}
    gap, unknown = cogs_gap(df), df["danh_muc"] == NO_CATEGORY
    data = {float(gap["orders"]), round(gap["revenue_share"] * 100, 1), float(unknown.sum()),
            round(df.loc[unknown, "doanh_thu_thuan"].sum() / df["doanh_thu_thuan"].sum() * 100, 1)}
    notes = answer_notes(df, plan)
    assert len(notes) == 3  # thiếu giá vốn, trước quảng cáo, chưa có danh mục
    checked = [c for c in _claims("\n".join(notes)) if c.percent or abs(c.value) >= 1000]
    assert checked and all(round(abs(c.value), 1) in data for c in checked)


def test_numbers_from_data_notes_may_be_quoted_but_not_invented(partial):
    df = partial.dataframe
    notes = profit_notes(df, {"metrics": [{"column": "loi_nhuan_truoc_qc"}]})
    result = pd.DataFrame({"kenh": ["Shopee"], "Lãi": [480_593_360.0]})
    quoted = f"Lãi Shopee là 480.593.360 đ, có {fmt_num(cogs_gap(df)['orders'])} đơn chưa có giá vốn."
    # Mode C: mọi câu trả lời của LLM nhắc số đơn thiếu giá vốn đều bị chặn.
    assert _numbers_grounded(quoted, result, notes=notes)
    assert not _numbers_grounded(quoted, result)
    assert not _numbers_grounded("Lãi Shopee là 999.999.999 đ.", result, notes=notes)


def test_rates_print_as_percent_in_deterministic_answer():
    plan = {"action": "aggregate",
            "metrics": [{"column": "phi_san", "aggregation": "sum", "label": "Phí"},
                        {"column": "doanh_thu_thuan", "aggregation": "sum", "label": "DT"}],
            "ratios": [{"label": "Tỷ lệ phí", "numerator": "Phí", "denominator": "DT"}]}
    result = pd.DataFrame({"Phí": [141_360_285.0], "DT": [421_971_000.0], "Tỷ lệ phí": [0.335]})
    text = _deterministic_answer("Tỷ lệ phí sàn tháng 4", result, plan)
    assert "33,5%" in text and "0,335" not in text and "141.360.285" in text
    rate = {"action": "aggregate", "group_by": ["tinh"],
            "metrics": [{"column": "la_don_hoan", "aggregation": "mean", "label": "Tỷ lệ hoàn"}]}
    ranked = pd.DataFrame({"tinh": ["Nghệ An", "Hà Nội"], "Tỷ lệ hoàn": [0.1552, 0.1175]})
    text = _deterministic_answer("Tỷ lệ hoàn theo tỉnh", ranked, rate)
    assert "15,5%" in text and "Tổng" not in text  # cộng/tỷ trọng các tỷ lệ là số vô nghĩa


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
    from backend.app.services.planner.execute import _validate_plan_against_dataframe
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


def test_eval_seller_cases_pass_with_their_own_answer():
    """Mỗi câu của eval_seller phải chấm ĐẠT được bằng chính đáp án: bắt đáp án NaN, số < 1000
    hay rơi vào vùng năm 1900–2100 (bị numbers() bỏ), id trùng/nhảy cóc — không cần chạy server."""
    from tests.eval_seller import REFUSAL, WARNING, cases, ground_truth, heldout_cases, score
    t = ground_truth()
    dev, heldout = cases(t), heldout_cases(t)
    assert [c.id for c in dev] == list(range(1, len(dev) + 1)) and len(dev) >= 150
    assert [c.id for c in heldout] == list(range(1001, 1001 + len(heldout))) and len(heldout) >= 20
    for c in dev + heldout:
        parts = [e[0] if isinstance(e, list) else e for e in c.expect]
        text = " ".join(p if isinstance(p, str) else f"{p:,.0f} đ".replace(",", ".") for p in parts)
        text += {"refuse": " " + REFUSAL[0], "warn": " " + WARNING[0]}.get(c.kind, "")
        assert score(c, text) == (True, False), (c.id, text)


def test_eval_pct_does_not_match_inside_a_bigger_number():
    # Đường luật từng in tỷ trọng tháng "(14,0%)" cạnh lãi; "4,0" (biên lãi ròng T6) khớp chuỗi con → chấm đạt oan.
    from tests.eval_seller import cases, ground_truth, score
    margin = {c.id: c for c in cases(ground_truth())}[47]  # "Biên lợi nhuận tháng 6 là bao nhiêu %?"
    assert score(margin, "6. 2026-06: 48.832.115 (14,0%)")[0] is False
    assert score(margin, "Biên lợi nhuận tháng 6 là 11,6%.")[0] is True


def test_suggested_questions_are_right_without_llm(full, offline):
    """Câu gợi ý trong app phải đúng cả khi LLM không gọi được (hết quota → plan luật + câu tất định).

    Đường luật từng sai 3/6: "doanh thu cao mà đang lỗ" ra top SKU LÃI, tỷ lệ phí sàn "chưa hiểu",
    tỷ lệ hoàn theo tỉnh × ĐVVC ra max cờ 0/1 = "1 (10,0%)" cho mọi tỉnh."""
    from tests.eval_seller import ground_truth
    t = ground_truth()
    mo, ch, sku = t["by"]("thang"), t["by"]("kenh"), t["by"]("sku")
    seg = t["by"]("tinh", "don_vi_van_chuyen").sort_values("hoan", ascending=False)
    (tinh, dvvc), top = seg.index[0], seg.iloc[0]
    loss = set(sku.index[sku.pre < 0])
    rate = (ch.fee / ch.rev).sort_values(ascending=False)  # 34,52% và 34,50%: làm tròn 1 số lẻ ra bằng nhau
    expect = {
        "Vì sao lãi tháng này giảm?": [fmt_num(mo.pre[5] - mo.pre[6])],
        "Lãi từng tháng thế nào?": [fmt_num(v) for v in mo.pre],
        "SKU nào doanh thu cao mà đang lỗ?": sorted(loss),
        "Lãi ròng sau quảng cáo theo tháng": [fmt_num(v) for v in mo.net],
        "Tỷ lệ phí sàn Shopee và TikTok bên nào cao hơn?": [
            fmt_pct(r) for r in rate] + [f"{rate.index[0]} cao hơn {rate.index[1]} {fmt_num((rate[0] - rate[1]) * 100, 2)} điểm %"],
        "Tỉnh và đơn vị vận chuyển nào có tỷ lệ hoàn cao nhất?": [f"{tinh} | {dvvc} | {fmt_pct(top.hoan)}"],
    }
    assert set(expect) == set(seller_questions(full.dataframe))
    answers = {q: _ask(full, q).answer for q in expect}
    for question, parts in expect.items():
        assert all(p in answers[question] for p in parts), (question, answers[question])
    assert set(re.findall(r"LAN-\d+", answers["SKU nào doanh thu cao mà đang lỗ?"])) == loss
    # Tỷ trọng tháng/tổng in cạnh lãi ("34.002.675 (20,5%)") đọc như biên lãi.
    assert "%" not in answers["Lãi từng tháng thế nào?"] + answers["Lãi ròng sau quảng cáo theo tháng"]


@pytest.mark.parametrize("question", [
    "Tỷ lệ hoàn theo đơn vị vận chuyển ở Hà Nội",  # nhắc 1 tỉnh mà không chia theo tỉnh
    "SKU nào đang lỗ ở Nghệ An",
    "Tỷ lệ phí sàn từng kênh tháng 5",
    "Tỷ lệ hoàn bao nhiêu",                       # không có chiều để chia: để luật chung xử lý
])
def test_seller_rule_plan_skips_questions_it_cannot_filter(full, question):
    from backend.app.services.planner.execute import _normalize
    from backend.app.services.planner.fallback import _seller_plan
    assert _seller_plan(full.dataframe, _normalize(question)) is None


def test_having_must_reference_plan_metrics(full):
    from backend.app.services.planner.execute import _validate_plan_against_dataframe
    with pytest.raises(ValueError):
        _validate_plan_against_dataframe(full.dataframe, {
            "action": "aggregate", "group_by": ["sku"],
            "metrics": [{"column": "loi_nhuan_truoc_qc", "aggregation": "sum", "label": "Lãi"}],
            "having": [{"column": "loi_nhuan_truoc_qc", "operator": "lt", "value": 0}]})
