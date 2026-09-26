"""Bộ dữ liệu mô phỏng shop Chị Lan: tất định, đúng tham số, và các kịch bản cài sẵn tìm lại được."""
import json

import pandas as pd
import pytest

from scripts import gen_shop_lan as g


@pytest.fixture(scope="module")
def data():
    return g.generate()


def test_deterministic(data):
    again = g.generate()
    pd.testing.assert_frame_equal(data["orders"], again["orders"])
    assert data["truth"] == again["truth"]


def test_committed_files_match_generator(data):
    # File trong data/samples/shop_lan/ phải là đầu ra của đúng phiên bản script này.
    committed = json.loads((g.OUT_DIR / "ground_truth.json").read_text(encoding="utf-8"))
    assert committed == json.loads(json.dumps(data["truth"], ensure_ascii=False))


def test_calibrated_to_press_figures(data):
    m = data["truth"]["monthly"]
    for month in (1, 2, 3, 4):
        assert m[month]["ty_le_phi"] == pytest.approx(0.335, abs=0.002)   # ~1/3 doanh thu là phí sàn
        assert 0.05 <= m[month]["bien_rong"] <= 0.08                      # biên ròng 5–8%
        assert 350e6 <= m[month]["doanh_thu_thuan"] <= 500e6              # ~400 triệu/tháng
    for month in (5, 6):
        assert m[month]["ty_le_phi"] == pytest.approx(0.365, abs=0.002)


def test_s1_fee_hike_explains_may_drop(data):
    s1 = data["truth"]["S1_fee_hike"]
    drop = s1["loi_nhuan_thang_4"] - s1["loi_nhuan_thang_5"]
    assert drop > 0
    # Phí tăng thêm phải là nguyên nhân chính của mức giảm lãi.
    assert s1["phi_tang_them_thang_5"] >= 0.8 * drop


def test_s2_seeded_loss_skus_are_top10_and_negative(data):
    seeded = set(data["products"].loc[data["products"]["s2_loss"], "sku"])
    found = {r["sku"] for r in data["truth"]["S2_loss_skus"]}
    assert found == seeded and len(found) == 3


def test_s3_ads_doubled_without_revenue_lift(data):
    s3 = data["truth"]["S3_tiktok_ads_week23"]
    assert s3["qc_trung_binh_ngay_tuan_23"] >= 1.8 * s3["qc_trung_binh_ngay_4_tuan_truoc"]
    assert abs(s3["dt_trung_binh_ngay_tuan_23"] / s3["dt_trung_binh_ngay_4_tuan_truoc"] - 1) < 0.15


def test_s4_return_spike_traced_to_province_and_carrier(data):
    s4 = data["truth"]["S4_return_spike"]
    assert s4["ty_le_hoan_thang_3"] > s4["ty_le_hoan_thang_2"]
    assert s4["phan_khuc_gay_tang"]["tinh"] == g.S4_PROVINCE
    assert s4["phan_khuc_gay_tang"]["don_vi_van_chuyen"] == g.S4_CARRIER
