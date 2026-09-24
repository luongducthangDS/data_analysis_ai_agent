"""
Báo cáo lãi/lỗ định kỳ cho shop TMĐT, gửi qua Telegram.

Số liệu tính tất định từ semantic layer (ecommerce_semantic), không qua LLM:
nạp file → chuẩn hoá như lúc upload → cộng cột đã tính sẵn theo kỳ.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

from backend.app.services.ecommerce_metrics import _format_vnd, _is_cancelled, _is_returned, _normalize_status
from backend.app.services.ecommerce_semantic import attach_cogs
from backend.app.services.storage import SessionStore

TELEGRAM_LIMIT = 4096


def load_shop_data(paths: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """File export sàn + bảng sản phẩm (+ file quảng cáo ngay/kenh/chi_phi_qc) → (đơn hàng, quảng cáo)."""
    sheets = attach_cogs({p.stem: SessionStore._read_dataframe(p) for p in paths})
    orders = [df for df in sheets.values() if "loi_nhuan_truoc_qc" in df.columns]
    if not orders:
        raise ValueError("Không có file đơn hàng nào đủ cột để tính lãi (cần export Shopee/TikTok + bảng sản phẩm có gia_von).")
    ads = [df for df in sheets.values() if {"ngay", "chi_phi_qc"} <= set(df.columns)]
    return pd.concat(orders, ignore_index=True), (pd.concat(ads, ignore_index=True) if ads else None)


def _summary(orders: pd.DataFrame, ads: pd.DataFrame | None, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    day = orders["ngay_dat"].dt.normalize()
    o = orders[(day >= start) & (day <= end)]
    status = _normalize_status(o["trang_thai"])
    per_order = o.assign(_huy=_is_cancelled(status), _hoan=_is_returned(status)).groupby("ma_don")[["_huy", "_hoan"]].any()
    ad_spend = 0.0
    if ads is not None:
        ad_day = pd.to_datetime(ads["ngay"], errors="coerce").dt.normalize()
        ad_spend = float(pd.to_numeric(ads.loc[(ad_day >= start) & (ad_day <= end), "chi_phi_qc"], errors="coerce").sum())
    revenue = float(o["doanh_thu_thuan"].sum())
    profit = float(o["loi_nhuan_truoc_qc"].sum()) - ad_spend
    return {
        "orders": len(per_order),
        "cancelled": int(per_order["_huy"].sum()),
        "returned": int(per_order["_hoan"].sum()),
        "revenue": revenue,
        "fees": float(o["phi_san"].sum()),
        "ads": ad_spend,
        "profit": profit,
        "margin": profit / revenue if revenue else None,
        "rows": o,
    }


def _delta(cur: float, prev: float) -> str:
    if not prev:
        return ""
    pct = (cur - prev) / abs(prev) * 100
    return f" ({'▲' if pct >= 0 else '▼'}{abs(pct):.0f}% so với kỳ trước)"


def build_report(orders: pd.DataFrame, ads: pd.DataFrame | None = None, days: int = 7) -> str:
    """Báo cáo `days` ngày gần nhất (tính theo ngày mới nhất có trong dữ liệu), so với kỳ liền trước."""
    end = orders["ngay_dat"].max().normalize()
    start = end - pd.Timedelta(days=days - 1)
    cur = _summary(orders, ads, start, end)
    prev = _summary(orders, ads, start - pd.Timedelta(days=days), start - pd.Timedelta(days=1))

    profit_label = "Lãi ròng (sau QC)" if ads is not None else "Lãi (chưa trừ QC)"
    lines = [
        f"📊 Báo cáo shop {start:%d/%m} – {end:%d/%m/%Y} ({days} ngày)",
        "",
        f"🧾 Đơn: {cur['orders']:,} | huỷ {cur['cancelled']:,} | hoàn {cur['returned']:,}",
        f"💰 Doanh thu thuần: {_format_vnd(cur['revenue'])}{_delta(cur['revenue'], prev['revenue'])}",
        f"🏪 Phí sàn: {_format_vnd(cur['fees'])}" + (f" ({cur['fees'] / cur['revenue']:.1%} doanh thu)" if cur["revenue"] else ""),
    ]
    if ads is not None:
        lines.append(f"📣 Quảng cáo: {_format_vnd(cur['ads'])}")
    margin = f", biên {cur['margin']:.1%}" if cur["margin"] is not None else ""
    lines.append(f"{'✅' if cur['profit'] >= 0 else '🔴'} {profit_label}: {_format_vnd(cur['profit'])}{margin}"
                 f"{_delta(cur['profit'], prev['profit'])}")

    rows = cur["rows"]
    if "kenh" in rows.columns and rows["kenh"].nunique() > 1:
        lines += ["", "Theo kênh (lãi chưa trừ QC):"]
        by_channel = rows.groupby("kenh")[["doanh_thu_thuan", "loi_nhuan_truoc_qc"]].sum()
        for kenh, r in by_channel.sort_values("doanh_thu_thuan", ascending=False).iterrows():
            lines.append(f"• {kenh}: DT {_format_vnd(r['doanh_thu_thuan'])}, lãi {_format_vnd(r['loi_nhuan_truoc_qc'])}")

    by_sku = rows.groupby("sku")["loi_nhuan_truoc_qc"].sum()
    losers = by_sku[by_sku < 0].sort_values().head(3)
    if not losers.empty:
        lines += ["", "⚠️ SKU đang lỗ (chưa trừ QC):"]
        lines += [f"• {sku}: {_format_vnd(abs(v))} lỗ" for sku, v in losers.items()]

    no_cost = int(rows["gia_von"].isna().sum())
    if no_cost:
        lines += ["", f"❗ {no_cost} dòng đơn không có giá vốn (SKU thiếu trong bảng sản phẩm) → lãi đang bị tính cao hơn thực tế."]
    return "\n".join(lines)


def send_telegram(text: str, token: str, chat_id: str) -> None:
    """Gửi tin nhắn text thuần (không parse_mode, khỏi phải escape tên sản phẩm)."""
    if not token or not chat_id:
        raise ValueError("Thiếu TELEGRAM_BOT_TOKEN hoặc TELEGRAM_CHAT_ID.")
    resp = requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        json={"chat_id": chat_id, "text": text[:TELEGRAM_LIMIT]},
        timeout=15,
    )
    if not resp.ok:
        # Không in URL: trong URL có token.
        raise RuntimeError(f"Telegram trả lỗi {resp.status_code}: {resp.text[:200]}")
