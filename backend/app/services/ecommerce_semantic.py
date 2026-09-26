"""
Semantic layer cho bảng đơn hàng TMĐT đã chuẩn hoá (schema của data/samples/shop_lan/shop_lan_orders.csv).

Metric được định nghĩa MỘT chỗ và tính tất định bằng pandas lúc nạp file. LLM không viết
công thức lợi nhuận, nó chỉ aggregate các cột đã tính sẵn (loi_nhuan_truoc_qc, phi_san...).
"""
from __future__ import annotations

import re

import pandas as pd

from backend.app.services.ecommerce_metrics import _is_cancelled, _is_returned, _normalize_status

FEE_COLS = ("phi_hoa_hong", "phi_thanh_toan", "phi_voucher_freeship", "phi_dich_vu", "thue_khau_tru")
# Không đòi gia_von: file export sàn không có giá vốn nhưng phí sàn, doanh thu thuần vẫn tính được.
REQUIRED_COLS = ("trang_thai", "doanh_thu", "giam_gia_shop", *FEE_COLS)
ADS_COLS = {"ngay", "kenh", "chi_phi_qc"}  # ponytail: đúng schema ads_daily mô phỏng; file QC thật của sàn thêm alias sau

# ponytail: dict Python thay cho YAML, không thêm dependency; tách file khi có >1 schema.
METRICS: dict[str, str] = {
    "doanh_thu_thuan": "doanh thu đơn hoàn thành sau voucher shop (doanh_thu − giam_gia_shop); đơn huỷ/hoàn = 0",
    "phi_san": "tổng phí sàn của đơn: hoa hồng + thanh toán + voucher/freeship + dịch vụ + thuế khấu trừ",
    "la_don_hoan": "1 = đơn bị hoàn/trả, 0 = hoàn thành, trống = huỷ. mean(la_don_hoan) = TỶ LỆ HOÀN của nhóm",
    "loi_nhuan_truoc_qc": "lãi thật của đơn, CHƯA trừ quảng cáo: doanh_thu_thuan − phi_san − gia_von − chi_phi_hoan; "
                          "đơn hoàn thường âm. CHỈ có khi đã tải bảng giá vốn",
    "chi_phi_qc": "chi phí quảng cáo phân bổ về từng đơn theo doanh thu thuần trong cùng ngày × kênh. "
                  "CHỈ có khi đã tải file quảng cáo (ngay, kenh, chi_phi_qc)",
    "loi_nhuan_rong": "lãi SAU quảng cáo = loi_nhuan_truoc_qc − chi_phi_qc",
}
_BASE_METRICS = ("doanh_thu_thuan", "phi_san", "la_don_hoan", "loi_nhuan_truoc_qc")
PROFIT_COLS = ("loi_nhuan_truoc_qc", "loi_nhuan_rong")

# Từ khoá trong câu hỏi (đã bỏ dấu) → metric. Thứ tự quan trọng: cụm dài trước.
_QUESTION_HINTS: tuple[tuple[str, str], ...] = (
    (r"lai rong|loi nhuan rong|sau quang cao|sau qc|net profit", "loi_nhuan_rong"),
    (r"doanh thu thuan|net revenue", "doanh_thu_thuan"),
    (r"phi san|tong phi|platform fee", "phi_san"),
    (r"ty le hoan|\bhoan hang\b|\btra hang\b|return rate", "la_don_hoan"),
    (r"quang cao|\bqc\b|\bads?\b", "chi_phi_qc"),
    (r"\blai\b|\bloi nhuan\b|\blo\b|\bprofit\b|\bmargin\b", "loi_nhuan_truoc_qc"),
)


# Header file export gốc của sàn → schema chuẩn.
# ponytail: đúng 2 định dạng mô phỏng ở data/samples/shop_lan; thêm sàn/phiên bản export = thêm 1 dict.
EXPORT_HEADERS: dict[str, dict[str, str]] = {
    "Shopee": {
        "Mã đơn hàng": "ma_don", "Ngày đặt hàng": "ngay_dat", "Trạng Thái Đơn Hàng": "trang_thai",
        "SKU phân loại hàng": "sku", "Tên sản phẩm": "ten_san_pham", "Số lượng": "so_luong", "Giá gốc": "gia_ban",
        "Tổng giá bán (sản phẩm)": "doanh_thu", "Mã giảm giá của Shop": "giam_gia_shop",
        "Phí cố định": "phi_hoa_hong", "Phí thanh toán": "phi_thanh_toan",
        "Phí Voucher Xtra & Freeship Xtra": "phi_voucher_freeship", "Phí Dịch Vụ": "phi_dich_vu",
        "Thuế khấu trừ": "thue_khau_tru", "Phí vận chuyển trả hàng": "chi_phi_hoan",
        "Tỉnh/Thành phố": "tinh", "Đơn Vị Vận Chuyển": "don_vi_van_chuyen",
    },
    "TikTok Shop": {
        "Order ID": "ma_don", "Created Time": "ngay_dat", "Order Status": "trang_thai",
        "Seller SKU": "sku", "Product Name": "ten_san_pham", "Quantity": "so_luong",
        "SKU Unit Original Price": "gia_ban", "SKU Subtotal Before Discount": "doanh_thu",
        "SKU Seller Discount": "giam_gia_shop", "Commission Fee": "phi_hoa_hong", "Transaction Fee": "phi_thanh_toan",
        "Campaign Service Fee": "phi_voucher_freeship", "Platform Service Fee": "phi_dich_vu",
        "Tax Withheld": "thue_khau_tru", "Return Shipping Fee": "chi_phi_hoan",
        "Province": "tinh", "Shipping Provider Name": "don_vi_van_chuyen",
    },
}
_DAYFIRST_CHANNELS = {"TikTok Shop"}  # "01/02/2026 ..." là 1/2, không phải 2/1
_EXPORT_CORE = {"ma_don", "ngay_dat", "trang_thai", "sku", "doanh_thu", "giam_gia_shop", *FEE_COLS}


def rename_export_columns(df: pd.DataFrame) -> pd.DataFrame:
    """File export Shopee/TikTok nhận ra được → đổi sang tên cột chuẩn và thêm cột kenh. Bảng khác giữ nguyên."""
    for channel, headers in EXPORT_HEADERS.items():
        present = {h: n for h, n in headers.items() if h in df.columns}
        if set(present.values()) >= _EXPORT_CORE:
            out = df.rename(columns=present)
            if channel in _DAYFIRST_CHANNELS:
                out["ngay_dat"] = pd.to_datetime(out["ngay_dat"], dayfirst=True, errors="coerce")
            if "kenh" not in out.columns:
                out["kenh"] = channel
            return out
    return df


def link_sheets(sheets: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Ghép bảng tra cứu vào bảng đơn: giá vốn (lãi trước QC) rồi quảng cáo (lãi ròng)."""
    return attach_ads(attach_cogs(sheets))


def attach_cogs(sheets: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """File export sàn không có giá vốn: ghép từ bảng sản phẩm (cột sku, gia_von = giá vốn/đơn vị) rồi tính metric."""
    catalogs = [df for df in sheets.values() if {"sku", "gia_von"} <= set(df.columns) and "trang_thai" not in df.columns]
    if not catalogs:
        return sheets
    catalog = pd.concat(catalogs)
    unit_cost = pd.to_numeric(catalog["gia_von"], errors="coerce").groupby(catalog["sku"].astype(str).str.strip()).last()
    out = {}
    for key, df in sheets.items():
        if {"sku", "trang_thai"} <= set(df.columns) and "gia_von" not in df.columns:
            qty = pd.to_numeric(df["so_luong"], errors="coerce").fillna(1) if "so_luong" in df.columns else 1
            # SKU không có trong bảng sản phẩm → gia_von NaN, profit_notes báo cho người dùng.
            # Bỏ metric đã tính lúc nạp (chưa có giá vốn) để tính lại đủ cả lãi.
            df = df.drop(columns=[c for c in _BASE_METRICS if c in df.columns])
            df = add_metric_columns(df.assign(gia_von=df["sku"].astype(str).str.strip().map(unit_cost) * qty))
        out[key] = df
    return out


def attach_ads(sheets: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Bảng quảng cáo theo ngày × kênh → chi_phi_qc phân bổ về từng đơn theo doanh thu thuần, và loi_nhuan_rong.

    Tổng chi_phi_qc theo ngày × kênh khớp đúng file quảng cáo, nên lãi ròng theo tháng/kênh là số thật;
    lãi ròng theo SKU/tỉnh là số PHÂN BỔ (giả định QC tỉ lệ với doanh thu).
    """
    ads = [df for df in sheets.values() if ADS_COLS <= set(df.columns)]
    if not ads:
        return sheets
    ad = pd.concat(ads)
    spend = pd.to_numeric(ad["chi_phi_qc"], errors="coerce").fillna(0).groupby(
        [pd.to_datetime(ad["ngay"], errors="coerce").dt.normalize().rename("d"), _channel_key(ad["kenh"]).rename("k")]
    ).sum()
    out = {}
    for key, df in sheets.items():
        if {"loi_nhuan_truoc_qc", "kenh", "ngay_dat"} <= set(df.columns) and "chi_phi_qc" not in df.columns:
            keys = [pd.to_datetime(df["ngay_dat"], errors="coerce").dt.normalize().rename("d"),
                    _channel_key(df["kenh"]).rename("k")]
            rev = df["doanh_thu_thuan"].clip(lower=0)
            day_rev = rev.groupby(keys).transform("sum")
            # Ngày × kênh không có doanh thu thuần (toàn đơn huỷ/hoàn) → chia đều cho số đơn.
            # ponytail: ngày có QC nhưng không có đơn nào thì phần QC đó không phân bổ được (mô phỏng: không xảy ra).
            share = (rev / day_rev).where(day_rev > 0, 1 / rev.groupby(keys).transform("size"))
            day_spend = spend.reindex(pd.MultiIndex.from_arrays(keys)).fillna(0).to_numpy()
            qc = share * day_spend
            df = df.assign(chi_phi_qc=qc, loi_nhuan_rong=df["loi_nhuan_truoc_qc"] - qc)
        out[key] = df
    return out


def _channel_key(s: pd.Series) -> pd.Series:
    """'TikTok Shop' / 'tiktok' / 'Shopee Mall' → 'tiktok' / 'shopee'."""
    return s.astype(str).str.strip().str.casefold().str.split().str[0]


def add_metric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Thêm các cột trong METRICS nếu bảng có đủ REQUIRED_COLS. Không đụng tới cột người dùng đã có.

    Thiếu gia_von → KHÔNG có loi_nhuan_truoc_qc (trước đây LLM lấy doanh thu gọi là "lãi").
    """
    if not set(REQUIRED_COLS) <= set(df.columns) or set(_BASE_METRICS) & set(df.columns):
        return df
    out = df.copy()

    def num(col: str) -> pd.Series:
        return pd.to_numeric(out[col], errors="coerce").fillna(0)

    status = _normalize_status(out["trang_thai"])
    cancelled, returned = _is_cancelled(status), _is_returned(status)
    done = ~(cancelled | returned)
    out["doanh_thu_thuan"] = (num("doanh_thu") - num("giam_gia_shop")).where(done, 0)
    out["phi_san"] = sum(num(c) for c in FEE_COLS).where(done, 0)
    out["la_don_hoan"] = returned.astype(float).where(~cancelled)
    if "gia_von" in out.columns:
        return_cost = num("chi_phi_hoan") if "chi_phi_hoan" in out.columns else 0
        out["loi_nhuan_truoc_qc"] = (out["doanh_thu_thuan"] - out["phi_san"]
                                     - num("gia_von").where(done, 0) - return_cost)
    return out


BRIDGE_REQUIRED = ("doanh_thu_thuan", "phi_san", "loi_nhuan_truoc_qc", "trang_thai")
_BRIDGE_COMPONENTS = (  # (nhãn, cột, dấu khi cộng vào lãi)
    ("Doanh thu thuần", "doanh_thu_thuan", 1),
    ("Phí sàn", "phi_san", -1),
    ("Giá vốn", "_gia_von_tinh", -1),
    ("Chi phí hoàn hàng", "_chi_phi_hoan", -1),
)
PROFIT_ROW, FEE_RATE_ROW, RETURN_RATE_ROW, NO_COST_ROW = "Lãi trước QC", "Tỷ lệ phí sàn", "Tỷ lệ hoàn", "Đơn thiếu giá vốn"
ADS_ROW = "Chi phí quảng cáo"


def month_periods(year: int, month: int, base_month: int | None = None) -> list[list[str]]:
    """[[kỳ gốc], [kỳ so sánh]] dạng ngày cho tháng `month` so với `base_month` (mặc định tháng liền trước)."""
    cur = pd.Period(year=year, month=month, freq="M")
    base = pd.Period(year=year, month=base_month, freq="M") if base_month else cur - 1
    return [[f"{p.start_time:%Y-%m-%d}", f"{p.end_time:%Y-%m-%d}"] for p in (base, cur)]


def profit_bridge(df: pd.DataFrame, plan: dict) -> pd.DataFrame:
    """So lãi trước QC giữa 2 kỳ: chênh lệch tách theo hạng mục + các nhóm kéo lãi đi nhiều nhất.

    Cột: hang_muc | <kỳ gốc> | <kỳ so sánh> | anh_huong_lai. Các hạng mục cộng lại đúng bằng Δ lãi.
    Không có `periods` → tháng cuối cùng trong dữ liệu so với tháng trước đó.
    """
    dates = df[plan.get("time_column") or "ngay_dat"]
    if not pd.api.types.is_datetime64_any_dtype(dates):
        dates = pd.to_datetime(dates, errors="coerce")
    periods = plan.get("periods")
    if not periods:
        last = dates.max()
        periods = month_periods(last.year, last.month)
    (lo0, hi0), (lo1, hi1) = [(pd.Timestamp(lo), pd.Timestamp(hi)) for lo, hi in periods]
    day = dates.dt.normalize()
    frames = [df[day.between(lo0, hi0)], df[day.between(lo1, hi1)]]
    labels = [_period_label(lo0, hi0), _period_label(lo1, hi1)]

    def enrich(part: pd.DataFrame) -> pd.DataFrame:
        hoan = pd.to_numeric(part["chi_phi_hoan"], errors="coerce").fillna(0) if "chi_phi_hoan" in part else 0
        # Giá vốn lấy phần dư để các hạng mục cộng lại khớp tuyệt đối với loi_nhuan_truoc_qc.
        gv = part["doanh_thu_thuan"] - part["phi_san"] - hoan - part["loi_nhuan_truoc_qc"]
        return part.assign(_chi_phi_hoan=hoan, _gia_von_tinh=gv)

    prev, cur = (enrich(f) for f in frames)
    rows = []
    for label, col, sign in _BRIDGE_COMPONENTS:
        a, b = prev[col].sum(), cur[col].sum()
        rows.append((label, a, b, sign * (b - a)))
    a, b = prev["loi_nhuan_truoc_qc"].sum(), cur["loi_nhuan_truoc_qc"].sum()
    rows.append((PROFIT_ROW, a, b, b - a))

    rev0, rev1 = prev["doanh_thu_thuan"].sum(), cur["doanh_thu_thuan"].sum()
    r0 = prev["phi_san"].sum() / rev0 if rev0 else float("nan")
    r1 = cur["phi_san"].sum() / rev1 if rev1 else float("nan")
    # Phần lãi mất/được riêng vì TỶ LỆ phí đổi (phần còn lại của Δ phí là do doanh thu đổi).
    rows.append((FEE_RATE_ROW, round(r0, 4), round(r1, 4), round(-rev1 * (r1 - r0), 0)))  # round(nan) không ndigits sẽ raise
    rows.append((RETURN_RATE_ROW, _return_rate(prev), _return_rate(cur), float("nan")))
    if "chi_phi_qc" in df.columns:
        qa, qb = prev["chi_phi_qc"].sum(), cur["chi_phi_qc"].sum()
        rows.append((ADS_ROW, round(qa), round(qb), round(qa - qb)))
    if "gia_von" in df.columns:
        n0, n1 = int(prev["gia_von"].isna().sum()), int(cur["gia_von"].isna().sum())
        if n0 or n1:
            rows.append((NO_COST_ROW, n0, n1, float("nan")))

    group_by = plan.get("group_by")
    dims = [list(group_by)] if group_by else [[c] for c in ("kenh", "sku") if c in df.columns]
    limit = min(int(plan.get("limit") or 5), 20)
    for dim in dims:
        g0 = prev.groupby(dim)["loi_nhuan_truoc_qc"].sum()
        g1 = cur.groupby(dim)["loi_nhuan_truoc_qc"].sum()
        delta = g1.sub(g0, fill_value=0)
        # Nhóm kéo lãi xuống luôn hiện, kể cả khi tổng lãi tăng (vd hoàn tăng ở 1 tỉnh bị doanh thu che mất).
        top = delta[delta < 0].nsmallest(limit)
        if b > a:
            top = pd.concat([delta[delta > 0].nlargest(limit), top])
        for key, d in top.items():
            parts = key if isinstance(key, tuple) else (key,)
            rows.append((f"{' × '.join(dim)}: {' × '.join(map(str, parts))}", g0.get(key, 0.0), g1.get(key, 0.0), d))

    return pd.DataFrame(rows, columns=["hang_muc", labels[0], labels[1], "anh_huong_lai"])


def bridge_actions(result: pd.DataFrame) -> list[str]:
    """Gợi ý nên kiểm tra gì, suy ra tất định từ bảng profit_bridge. Mọi con số lấy từ chính bảng."""
    # ponytail: ngưỡng cố định (0,5 điểm % phí, 1 điểm % hoàn/giá vốn); cho chỉnh theo shop khi có người dùng thật.
    rows = {r[0]: r[1:] for r in result.itertuples(index=False)}
    p0, p1 = result.columns[1], result.columns[2]
    out = []
    empty = [p for p, rev in zip((p0, p1), rows["Doanh thu thuần"][:2]) if not rev]
    if empty:
        out.append(f"CẢNH BÁO: kỳ {' và '.join(empty)} không có doanh thu trong dữ liệu, nên mức chênh lệch ở trên "
                   "chưa phản ánh thực tế. Kiểm tra lại kỳ so sánh hoặc file đã đủ tháng chưa.")
    if FEE_RATE_ROW in rows:
        r0, r1, lost = rows[FEE_RATE_ROW]
        if r1 - r0 >= 0.005:
            price_up = (1 - r0) / (1 - r1) - 1
            out.append(f"Tỷ lệ phí sàn tăng từ {fmt_pct(r0)} lên {fmt_pct(r1)}, riêng phần này làm lãi giảm {_vnd(-lost)}. "
                       "Kiểm tra biểu phí/hoa hồng mới của sàn và các gói Voucher Xtra/Freeship Xtra đang bật; "
                       f"muốn giữ tiền về như trước cần tăng giá khoảng {fmt_pct(price_up)}.")
    if RETURN_RATE_ROW in rows:
        h0, h1, _ = rows[RETURN_RATE_ROW]
        if h1 - h0 >= 0.01:
            out.append(f"Tỷ lệ hoàn tăng từ {fmt_pct(h0)} lên {fmt_pct(h1)}. Tách lãi theo tỉnh và đơn vị vận chuyển để tìm "
                       "khu vực hoàn tăng; cân nhắc đổi đơn vị vận chuyển hoặc gọi xác nhận đơn COD ở đó.")
    rev0, rev1, _ = rows["Doanh thu thuần"]
    gv0, gv1, _ = rows["Giá vốn"]
    if rev0 and rev1 and gv1 / rev1 - gv0 / rev0 >= 0.01:
        out.append(f"Giá vốn chiếm {fmt_pct(gv1 / rev1)} doanh thu (trước đó {fmt_pct(gv0 / rev0)}). Kiểm tra giá nhập mới "
                   "hoặc cơ cấu bán hàng đang lệch sang SKU biên thấp.")
    for name, (_, now, _) in rows.items():
        if ":" in name and now < 0:
            out.append(f"{name} đang lỗ {_vnd(-now)} trong kỳ {p1}. Xem lại giá bán/giá vốn và tạm dừng quảng cáo cho nhóm này.")
    if NO_COST_ROW in rows:
        out.append(f"{int(rows[NO_COST_ROW][1])} đơn kỳ {p1} thiếu giá vốn nên lãi đang bị tính cao hơn thực tế; "
                   "bổ sung giá vốn cho các SKU này trong bảng sản phẩm.")
    if ADS_ROW in rows:
        qa, qb, _ = rows[ADS_ROW]
        pa, pb, _ = rows[PROFIT_ROW]
        out.append(f"Quảng cáo {p0}: {_vnd(qa)}, {p1}: {_vnd(qb)}. Lãi ròng sau quảng cáo: "
                   f"{p0} {_vnd(pa - qa)} → {p1} {_vnd(pb - qb)}.")
    else:
        out.append(f"Lãi ở đây CHƯA trừ quảng cáo; tải thêm file quảng cáo (ngay, kenh, chi_phi_qc) để có lãi ròng {p0} và {p1}.")
    return out


def _return_rate(part: pd.DataFrame) -> float:
    status = _normalize_status(part["trang_thai"])
    valid = ~_is_cancelled(status)
    return round(float(_is_returned(status)[valid].sum() / valid.sum()), 4) if valid.any() else float("nan")


def _period_label(lo: pd.Timestamp, hi: pd.Timestamp) -> str:
    if lo.day == 1 and hi == lo + pd.offsets.MonthEnd(0):
        return f"T{lo.month}/{lo.year}"
    return f"{lo:%d/%m}–{hi:%d/%m/%Y}"


def fmt_num(value: float, decimals: int = 0) -> str:
    """Kiểu Việt: 13.308.870 và 0,3350 — một định dạng cho cả câu trả lời (trước đây trộn 13,308,870 với 10.985.900)."""
    return f"{value:,.{decimals}f}".translate(str.maketrans(",.", ".,"))


def fmt_pct(ratio: float) -> str:
    return f"{fmt_num(ratio * 100, 1)}%"


def _vnd(value: float) -> str:
    return f"{fmt_num(value)} đ"


def pick_metric(normalized_question: str, df: pd.DataFrame) -> str | None:
    """Metric semantic ứng với câu hỏi (đã bỏ dấu, chữ thường), nếu bảng có cột đó."""
    for pattern, metric in _QUESTION_HINTS:
        if metric in df.columns and re.search(pattern, normalized_question):
            return metric
    return None


def describe_metrics(df: pd.DataFrame) -> str:
    """Khối mô tả metric cho planner prompt; rỗng nếu bảng không có semantic layer."""
    present = [m for m in METRICS if m in df.columns]
    if not present:
        return ""
    lines = ["", "METRIC ĐÃ TÍNH SẴN (semantic layer, ưu tiên dùng thay vì tự ghép công thức):"]
    lines += [f"  - {m}: {METRICS[m]}" for m in present]
    lines += [
        "  Hỏi lãi/lợi nhuận/lỗ → sum loi_nhuan_truoc_qc. Hỏi lãi ròng / sau quảng cáo → sum loi_nhuan_rong.",
        "  Hỏi doanh thu (thực nhận) → sum doanh_thu_thuan. KHÔNG BAO GIỜ dùng doanh_thu làm lãi.",
        "  TỶ LỆ (% phí sàn, biên lãi) = tỷ số của 2 tổng → 2 metric sum + field \"ratios\":",
        '  [{"label":"Tỷ lệ phí sàn","numerator":"<label metric phí>","denominator":"<label metric doanh thu>"}].',
        "  So sàn nào \"ăn phí nhiều hơn\" → so TỶ LỆ phí theo kenh, không so tổng tiền phí.",
        "  Tỷ lệ hoàn theo nhóm → mean la_don_hoan, kèm count ma_don để thấy cỡ nhóm; sort theo tỷ lệ.",
        "  Quảng cáo có gì bất thường / tăng vọt → time_series grain \"date\", sum chi_phi_qc (và doanh_thu_thuan),",
        "  filter kenh + khoảng ngày rộng hơn kỳ hỏi (vd thêm 4 tuần trước đó) để so với mức bình thường.",
        "  Hỏi SKU/kênh nào lỗ → group_by rồi sum loi_nhuan_truoc_qc. KHÔNG filter loi_nhuan_truoc_qc < 0:",
        "  filter chạy trên TỪNG ĐƠN trước khi cộng, sẽ bỏ đơn lãi và cho tổng sai.",
        "  \"Top N doanh thu mà lỗ\" → sort theo doanh_thu_thuan desc, limit N, kèm metric loi_nhuan_truoc_qc.",
        "  \"theo kênh/sàn\" → group_by kenh. Hỏi số của MỘT kỳ → aggregate + filter ngày,",
        "  không dùng time_series (time_series bỏ qua group_by).",
        "  Hỏi VÌ SAO lãi tăng/giảm, lãi kỳ này so với kỳ trước → action profit_bridge (backend tự tách chênh lệch",
        "  theo doanh thu/phí/giá vốn/hoàn và tìm kênh, SKU kéo lãi). periods = [[kỳ gốc], [kỳ so sánh]], ngày YYYY-MM-DD;",
        "  bỏ periods = tháng cuối trong dữ liệu so với tháng trước. group_by chỉ khi người dùng muốn tách theo chiều khác.",
        '  Cú pháp mẫu (tháng M so với tháng M-1): {"action":"profit_bridge","time_column":"ngay_dat",'
        '"periods":[["<YYYY>-<M-1>-01","<cuối tháng M-1>"],["<YYYY>-<M>-01","<cuối tháng M>"]]}',
    ]
    if "doanh_thu_thuan" in df.columns and "loi_nhuan_truoc_qc" not in df.columns:
        lines.append("  CHƯA có bảng giá vốn → KHÔNG có cột lãi; không được trả lời lãi bằng cột khác.")
    gap = cogs_gap(df)
    if gap and not gap["all_missing"]:
        lines.append(f"  CẢNH BÁO: {gap['orders']} đơn có SKU không có trong bảng sản phẩm nên thiếu giá vốn → lãi bị "
                     "tính cao hơn thực tế. Backend tự thêm cảnh báo vào câu trả lời; KHÔNG join bảng giá vốn "
                     "(đã ghép sẵn), cứ dùng loi_nhuan_truoc_qc.")
    lines.append("")
    return "\n".join(lines)


# ── Lớp bảo vệ câu trả lời về lãi (tất định, không qua LLM) ─────────────────────

_PROFIT_ACCENTED = re.compile(r"\b(?:lãi|lời|lợi nhuận|lỗ|profit|margin)\b", re.IGNORECASE)
_PROFIT_PLAIN = re.compile(r"\b(?:lai|loi nhuan|lo|profit|margin)\b")


def asks_profit(question: str) -> bool:
    """Câu hỏi có hỏi lãi/lỗ không. Câu có dấu thì khớp từ có dấu: bỏ dấu thì "xem lại" cũng thành "lai"."""
    q = question.lower()
    plain = _strip_marks(q)
    if plain != q:
        return bool(_PROFIT_ACCENTED.search(q))
    return bool(_PROFIT_PLAIN.search(plain))


def _strip_marks(text: str) -> str:
    import unicodedata
    t = unicodedata.normalize("NFKD", text.replace("đ", "d").replace("Đ", "D"))
    return "".join(ch for ch in t if not unicodedata.combining(ch))


def cogs_gap(df: pd.DataFrame) -> dict | None:
    """Đơn/SKU chưa có giá vốn. None nếu bảng không phải đơn TMĐT hoặc đã đủ giá vốn."""
    if "doanh_thu_thuan" not in df.columns or "sku" not in df.columns:
        return None
    missing = df["gia_von"].isna() if "gia_von" in df.columns else pd.Series(True, index=df.index)
    if not missing.any():
        return None
    rev = df["doanh_thu_thuan"]
    total = rev.sum()
    by_sku = rev[missing].groupby(df.loc[missing, "sku"].astype(str).str.strip()).sum().sort_values(ascending=False)
    return {"orders": int(missing.sum()), "revenue_share": float(rev[missing].sum() / total) if total else 0.0,
            "skus": by_sku.index.tolist(), "all_missing": "gia_von" not in df.columns}


def cogs_template(df: pd.DataFrame) -> pd.DataFrame:
    """sku, ten_san_pham, doanh_thu_thuan, gia_von (để trống) cho SKU chưa có giá vốn — người dùng điền rồi tải lên."""
    cols = ["sku", "ten_san_pham", "doanh_thu_thuan", "gia_von"]
    gap = cogs_gap(df)
    if not gap:
        return pd.DataFrame(columns=cols)
    work = df.assign(sku=df["sku"].astype(str).str.strip())
    work = work[work["sku"].isin(gap["skus"])]
    name = work["ten_san_pham"] if "ten_san_pham" in work.columns else ""
    out = (work.assign(ten_san_pham=name).groupby("sku")
           .agg(ten_san_pham=("ten_san_pham", "first"), doanh_thu_thuan=("doanh_thu_thuan", "sum"))
           .reset_index().sort_values("doanh_thu_thuan", ascending=False))
    return out.assign(gia_von="")[cols]


def missing_cogs_notice(df: pd.DataFrame, question: str) -> str | None:
    """Hỏi lãi mà chưa có giá vốn → câu từ chối có hướng dẫn, thay vì để LLM lấy doanh thu gọi là lãi."""
    if "doanh_thu_thuan" not in df.columns or "loi_nhuan_truoc_qc" in df.columns or not asks_profit(question):
        return None
    skus = (cogs_gap(df) or {"skus": []})["skus"]
    return (
        "**Chưa tính được lãi.** File đơn hàng của sàn không có giá vốn, nên mọi con số \"lãi\" lúc này đều là đoán.\n\n"
        "Cách bổ sung: tải thêm một bảng 2 cột `sku, gia_von` (giá nhập của 1 sản phẩm) cùng lúc với file export. "
        f"Nút **Tải mẫu giá vốn** có sẵn danh sách {len(skus)} SKU của shop, chỉ cần điền cột giá vốn.\n\n"
        + (f"Nên điền trước các SKU doanh thu cao nhất: {', '.join(skus[:5])}.\n\n" if skus else "")
        + "Trong lúc chờ, em vẫn trả lời được doanh thu thuần, phí sàn và tỷ lệ hoàn."
    )


_ITEM_DIMS = {"sku", "ten_san_pham", "danh_muc", "tinh", "don_vi_van_chuyen"}


def profit_notes(df: pd.DataFrame, plan: dict) -> list[str]:
    """Ghi chú bắt buộc cho câu trả lời có dùng cột lãi. Sinh tất định, nối vào cuối câu trả lời."""
    used = {m.get("column") for m in plan.get("metrics") or []} & set(PROFIT_COLS)
    if not used or plan.get("action") == "profit_bridge":  # bridge_actions đã tự nói các ý này
        return []
    notes = []
    gap = cogs_gap(df)
    if gap:
        notes.append(f"{fmt_num(gap['orders'])} đơn ({fmt_pct(gap['revenue_share'])} doanh thu thuần) thuộc "
                     f"{len(gap['skus'])} SKU chưa có giá vốn (vd {', '.join(gap['skus'][:5])}), nên lãi ở trên đang "
                     "CAO hơn thực tế. Bổ sung giá vốn các SKU này (nút Tải mẫu giá vốn) rồi tải lại.")
    if "loi_nhuan_truoc_qc" in used:
        notes.append("Đây là lãi TRƯỚC quảng cáo. " + (
            "Hỏi \"lãi ròng sau quảng cáo\" để trừ quảng cáo." if "loi_nhuan_rong" in df.columns
            else "Tải thêm file quảng cáo (ngay, kenh, chi_phi_qc) để có lãi ròng."))
    if "loi_nhuan_rong" in used and _ITEM_DIMS & set(plan.get("group_by") or []):
        notes.append("Quảng cáo được chia về từng đơn theo doanh thu trong ngày × kênh, "
                     "nên lãi ròng theo SKU/tỉnh là số ước tính.")
    return notes


def seller_notes(df: pd.DataFrame) -> list[str]:
    """Việc còn thiếu để có lãi thật, hiện ngay sau upload (trước khi seller kịp hỏi)."""
    if "doanh_thu_thuan" not in df.columns:
        return []
    gap = cogs_gap(df)
    if "loi_nhuan_truoc_qc" not in df.columns:
        return ["Chưa có giá vốn nên chưa tính được lãi. Tải mẫu giá vốn, điền cột gia_von rồi tải lên cùng file export."]
    notes = []
    if gap:
        notes.append(f"Có giá vốn cho {fmt_pct(1 - gap['revenue_share'])} doanh thu; {len(gap['skus'])} SKU còn thiếu "
                     "nên lãi đang cao hơn thực tế.")
    if "chi_phi_qc" not in df.columns:
        notes.append("Chưa có file quảng cáo (ngay, kenh, chi_phi_qc) nên lãi đang là lãi trước quảng cáo.")
    return notes


def seller_questions(df: pd.DataFrame) -> list[str]:
    """Câu gợi ý cố định cho seller, theo dữ liệu đang có (không gợi ý câu hệ thống chưa trả lời được)."""
    if "doanh_thu_thuan" not in df.columns:
        return []
    if "loi_nhuan_truoc_qc" not in df.columns:
        return ["Doanh thu thuần và phí sàn từng tháng", "Tỷ lệ phí sàn Shopee và TikTok bên nào cao hơn?",
                "Tỉnh và đơn vị vận chuyển nào có tỷ lệ hoàn cao nhất?"]
    out = ["Vì sao lãi tháng này giảm?", "Lãi từng tháng thế nào?", "SKU nào doanh thu cao mà đang lỗ?"]
    if "loi_nhuan_rong" in df.columns:
        out.append("Lãi ròng sau quảng cáo theo tháng")
    out += ["Tỷ lệ phí sàn Shopee và TikTok bên nào cao hơn?", "Tỉnh và đơn vị vận chuyển nào có tỷ lệ hoàn cao nhất?"]
    return out


def shop_summary(df: pd.DataFrame, file_names: list[str]) -> str | None:
    """Trả lời "file này có gì": nạp gì, đã ghép gì, còn thiếu gì. None nếu không phải đơn TMĐT."""
    if "doanh_thu_thuan" not in df.columns:
        return None
    orders = f"**{fmt_num(len(df))} đơn**"
    if "ngay_dat" in df.columns:
        d = pd.to_datetime(df["ngay_dat"], errors="coerce")
        orders += f" từ {d.min():%d/%m/%Y} đến {d.max():%d/%m/%Y}"
    if "kenh" in df.columns:
        orders += " (" + ", ".join(f"{k}: {fmt_num(v)}" for k, v in df["kenh"].value_counts().items()) + ")"
    lines = [f"Đã nạp: {', '.join(file_names)}.",
             f"- Đơn hàng: {orders}. Doanh thu thuần {_vnd(df['doanh_thu_thuan'].sum())}, "
             f"phí sàn {_vnd(df['phi_san'].sum())}."]
    gap = cogs_gap(df)
    if "loi_nhuan_truoc_qc" not in df.columns:
        lines.append("- Giá vốn: **chưa có** → chưa tính được lãi. Tải thêm bảng `sku, gia_von`.")
    elif gap:
        lines.append(f"- Giá vốn: thiếu cho {len(gap['skus'])} SKU ({fmt_pct(gap['revenue_share'])} doanh thu) "
                     "→ lãi đang cao hơn thực tế.")
    else:
        lines.append("- Giá vốn: đã ghép đủ cho mọi SKU.")
    if "chi_phi_qc" in df.columns:
        lines.append(f"- Quảng cáo: đã ghép {_vnd(df['chi_phi_qc'].sum())} → có lãi ròng sau quảng cáo.")
    elif "loi_nhuan_truoc_qc" in df.columns:
        lines.append("- Quảng cáo: chưa có file → lãi đang là lãi trước quảng cáo.")
    questions = seller_questions(df)
    if questions:
        lines += ["", "Có thể hỏi:"] + [f"- {q}" for q in questions[:4]]
    return "\n".join(lines)
