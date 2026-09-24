"""Báo cáo Telegram: số liệu phải khớp đáp án shop Chị Lan, và gửi đúng API Telegram."""
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from backend.app.services.telegram_report import _summary, build_report, load_shop_data, send_telegram
from scripts.gen_shop_lan import PARAMS

SHOP = Path(__file__).resolve().parents[1] / "data" / "samples" / "shop_lan"


@pytest.fixture(scope="module")
def shop():
    # Đi đúng đường của file export gốc: đổi header + ghép giá vốn từ products.csv.
    return load_shop_data([SHOP / f for f in ("export_shopee.csv", "export_tiktok.csv", "products.csv", "ads_daily.csv")])


def test_monthly_net_profit_matches_ground_truth(shop):
    orders, ads = shop
    truth = json.loads((SHOP / "ground_truth.json").read_text(encoding="utf-8"))["monthly"]
    # File sàn không có phần hàng hỏng khi hoàn (xem test_ecommerce_semantic) → lãi cao hơn đúng phần đó.
    full = pd.read_csv(SHOP / "shop_lan_orders.csv", parse_dates=["ngay_dat"])
    damage = (full["chi_phi_hoan"] - PARAMS["return_ship_cost"]).where(full["trang_thai"] == "Đã trả hàng", 0)
    damage_by_month = damage.groupby(full["ngay_dat"].dt.month).sum()
    for month, expected in truth.items():
        start = pd.Timestamp(2026, int(month), 1)
        s = _summary(orders, ads, start, start + pd.offsets.MonthEnd(0))
        assert s["profit"] == pytest.approx(expected["loi_nhuan_rong"] + damage_by_month[int(month)], abs=1)
        assert s["revenue"] == pytest.approx(expected["doanh_thu_thuan"], abs=1)


def test_report_mentions_period_and_channels(shop):
    text = build_report(*shop, days=7)
    assert "24/06 – 30/06/2026" in text
    assert "Shopee" in text and "TikTok Shop" in text
    assert "Lãi ròng (sau QC)" in text


def test_report_without_ads_labels_profit_before_ads(shop):
    assert "Lãi (chưa trừ QC)" in build_report(shop[0], None, days=7)


def test_send_telegram_posts_message(monkeypatch):
    calls = []
    monkeypatch.setattr("backend.app.services.telegram_report.requests.post",
                        lambda url, **kw: calls.append((url, kw)) or SimpleNamespace(ok=True))
    send_telegram("xin chào", "TOKEN", "123")
    (url, kw), = calls
    assert url == "https://api.telegram.org/botTOKEN/sendMessage"
    assert kw["json"] == {"chat_id": "123", "text": "xin chào"}


def test_send_telegram_requires_credentials():
    with pytest.raises(ValueError):
        send_telegram("x", "", "123")
