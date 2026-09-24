"""
Sinh dữ liệu MÔ PHỎNG cho persona "Chị Lan", một shop thời trang nữ bán trên Shopee + TikTok Shop.

Đây KHÔNG phải dữ liệu thật. Tham số lấy từ số liệu công bố trên báo chí (xem PARAMS);
cái nào là giả định thì ghi rõ "giả định". Có cài sẵn 4 kịch bản (S1–S4) để eval agent,
và đáp án đúng được tính lại từ chính dữ liệu sinh ra rồi ghi vào ground_truth.json.

Chạy:  python scripts/gen_shop_lan.py            # ghi vào data/samples/shop_lan/
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "samples" / "shop_lan"
SEED = 20260501
START, END = "2026-01-01", "2026-06-30"

# Phí tính trên doanh thu sau voucher của shop. Nguồn ghi cạnh từng tham số.
PARAMS = {
    "commission": 0.16,       # hoa hồng cố định. Nguồn: báo cáo SHS, trích qua VietnamFinance/CafeBiz
    "payment_fee": 0.06,      # phí thanh toán. Nguồn: như trên
    "program_fee": 0.09,      # Voucher Xtra + Freeship Xtra. Nguồn: như trên
    "service_fee": 0.01,      # phí dịch vụ/hạ tầng. Nguồn: như trên
    "tax": 0.015,             # sàn khấu trừ 1% GTGT + 0,5% TNCN cho hộ kinh doanh (NĐ 117/2025)
    "fee_hike_date": "2026-05-01",
    "fee_hike_pp": 0.03,      # phí tăng 2–4% từ 5/2026 (VOV, NLĐ); lấy giữa khoảng
    "cancel_rate": 0.06,      # giả định
    "return_rate": 0.10,      # giả định; tỉ lệ hoàn đồ may mặc quốc tế ~20–30%, VN chưa có số công bố
    "return_ship_cost": 30_000,   # giả định: phí ship chiều về do shop chịu
    "return_damage_share": 0.10,  # giả định: 10% hàng hoàn không bán lại được
    "ads_share": 0.07,        # giả định: ngân sách quảng cáo ~7% doanh thu
    "orders_per_day": 66,     # ~2.000 đơn/tháng (persona)
    "tiktok_share": 0.40,     # giả định
}

CATEGORIES = {"Áo": 45, "Váy/Đầm": 35, "Quần": 30, "Chân váy": 20, "Set đồ": 12, "Phụ kiện": 8}
PROVINCES = {
    "TP.HCM": 0.28, "Hà Nội": 0.24, "Nghệ An": 0.08, "Bình Dương": 0.07, "Đồng Nai": 0.06,
    "Hải Phòng": 0.06, "Đà Nẵng": 0.06, "Cần Thơ": 0.05, "Thanh Hoá": 0.05, "Khánh Hoà": 0.05,
}
CARRIERS = {
    "Shopee": {"SPX Express": 0.45, "J&T Express": 0.35, "GHN": 0.20},
    "TikTok Shop": {"J&T Express": 0.55, "GHN": 0.25, "Best Express": 0.20},
}

# Kịch bản cài sẵn
S2_LOSS_RANKS = (1, 3, 5)          # vị trí phổ biến → SKU doanh thu top 10 nhưng lỗ
S2_VOUCHER, S2_RETURN, S2_COST_RATIO, S2_PRICE = 0.25, 0.30, 0.52, 279_000
S3_WEEK = ("2026-06-01", "2026-06-07")   # ISO tuần 23: quảng cáo TikTok x2.2, doanh thu không đổi
S3_MULT = 2.2
S4_MONTH, S4_PROVINCE, S4_CARRIER, S4_RETURN = 3, "Nghệ An", "J&T Express", 0.75


def _pick(rng, weights: dict, n: int):
    keys = list(weights)
    p = np.array(list(weights.values()), dtype=float)
    return rng.choice(keys, size=n, p=p / p.sum())


def make_products(rng) -> pd.DataFrame:
    rows, i = [], 0
    for cat, n in CATEGORIES.items():
        for _ in range(n):
            i += 1
            price = int(rng.choice(np.arange(99_000, 300_000, 10_000)))
            rows.append({"sku": f"LAN-{i:03d}", "ten_san_pham": f"{cat} mẫu {i:03d}",
                         "danh_muc": cat, "gia_ban": price,
                         "ty_le_gia_von": round(float(rng.uniform(0.42, 0.52)), 3)})
    p = pd.DataFrame(rows)
    # Độ phổ biến kiểu Zipf; các SKU S2 được đặt vào hạng cao và chạy voucher mạnh.
    order = rng.permutation(len(p))
    p["weight"] = 0.0
    p.loc[order, "weight"] = 1.0 / np.arange(1, len(p) + 1) ** 0.6
    p["s2_loss"] = False
    for r in S2_LOSS_RANKS:
        idx = order[r]
        p.loc[idx, ["s2_loss", "ty_le_gia_von", "gia_ban"]] = [True, S2_COST_RATIO, S2_PRICE]
    return p


def make_orders(rng, products: pd.DataFrame) -> pd.DataFrame:
    days = pd.date_range(START, END, freq="D")
    dow = np.where(days.dayofweek >= 5, 1.15, 0.95)
    counts = rng.poisson(PARAMS["orders_per_day"] * dow)
    dates = np.repeat(days.values, counts)
    n = len(dates)

    o = pd.DataFrame({"ngay_dat": pd.to_datetime(dates)})
    o["ngay_dat"] += pd.to_timedelta(rng.integers(7 * 3600, 23 * 3600, n), unit="s")
    o["kenh"] = np.where(rng.random(n) < PARAMS["tiktok_share"], "TikTok Shop", "Shopee")
    o["tinh"] = _pick(rng, PROVINCES, n)
    o["don_vi_van_chuyen"] = ""
    for ch, w in CARRIERS.items():
        m = o["kenh"] == ch
        o.loc[m, "don_vi_van_chuyen"] = _pick(rng, w, int(m.sum()))

    # ponytail: 1 SKU/đơn. Đủ cho P1–P3; tách order_items khi cần phân tích giỏ hàng.
    pi = rng.choice(len(products), size=n, p=products["weight"] / products["weight"].sum())
    prod = products.iloc[pi].reset_index(drop=True)
    o = pd.concat([o, prod[["sku", "ten_san_pham", "danh_muc", "gia_ban", "ty_le_gia_von", "s2_loss"]]], axis=1)
    o["so_luong"] = rng.choice([1, 2, 3], size=n, p=[0.75, 0.20, 0.05])
    o["doanh_thu"] = o["gia_ban"] * o["so_luong"]

    has_v = rng.random(n) < 0.5
    v_rate = np.where(o["s2_loss"], S2_VOUCHER, np.where(has_v, rng.uniform(0.05, 0.10, n), 0.0))
    o["giam_gia_shop"] = (o["doanh_thu"] * v_rate).round(-3)

    # Trạng thái
    p_ret = np.where(o["s2_loss"], S2_RETURN, PARAMS["return_rate"])
    s4 = (o["ngay_dat"].dt.month == S4_MONTH) & (o["tinh"] == S4_PROVINCE) & (o["don_vi_van_chuyen"] == S4_CARRIER)
    p_ret = np.where(s4, S4_RETURN, p_ret)
    u = rng.random(n)
    o["trang_thai"] = np.select(
        [u < PARAMS["cancel_rate"], u < PARAMS["cancel_rate"] + p_ret * (1 - PARAMS["cancel_rate"])],
        ["Đã huỷ", "Đã trả hàng"], "Hoàn thành")

    # Phí: chỉ thu trên đơn hoàn thành, tính trên doanh thu sau voucher shop.
    done = o["trang_thai"] == "Hoàn thành"
    base = (o["doanh_thu"] - o["giam_gia_shop"]).where(done, 0)
    hike = np.where(o["ngay_dat"] >= PARAMS["fee_hike_date"], PARAMS["fee_hike_pp"], 0.0)
    o["phi_hoa_hong"] = (base * (PARAMS["commission"] + hike)).round()
    o["phi_thanh_toan"] = (base * PARAMS["payment_fee"]).round()
    o["phi_voucher_freeship"] = (base * PARAMS["program_fee"]).round()
    o["phi_dich_vu"] = (base * PARAMS["service_fee"]).round()
    o["thue_khau_tru"] = (base * PARAMS["tax"]).round()

    cogs = (o["gia_ban"] * o["ty_le_gia_von"]).round(-2) * o["so_luong"]
    o["gia_von"] = cogs.where(done, 0)
    # ponytail: hàng hỏng tính theo kỳ vọng (10% giá vốn) thay vì bốc thăm từng đơn.
    o["chi_phi_hoan"] = (PARAMS["return_ship_cost"] + PARAMS["return_damage_share"] * cogs).round().where(
        o["trang_thai"] == "Đã trả hàng", 0)

    o = o.sort_values("ngay_dat").reset_index(drop=True)
    o.insert(0, "ma_don", [f"{'SP' if k == 'Shopee' else 'TT'}{260000000 + i}" for i, k in enumerate(o["kenh"])])
    return o.drop(columns=["ty_le_gia_von", "s2_loss"])


def make_ads(rng, orders: pd.DataFrame) -> pd.DataFrame:
    days = pd.date_range(START, END, freq="D")
    rows = []
    for ch in CARRIERS:
        m = (orders["kenh"] == ch) & (orders["trang_thai"] == "Hoàn thành") & (orders["ngay_dat"] < "2026-05-01")
        daily_rev = (orders.loc[m, "doanh_thu"] - orders.loc[m, "giam_gia_shop"]).sum() / 120
        spend = PARAMS["ads_share"] * daily_rev * rng.uniform(0.85, 1.15, len(days))
        if ch == "TikTok Shop":
            spend = np.where((days >= S3_WEEK[0]) & (days <= S3_WEEK[1]), spend * S3_MULT, spend)
        rows.append(pd.DataFrame({"ngay": days.date, "kenh": ch, "chi_phi_qc": spend.round(-3)}))
    return pd.concat(rows, ignore_index=True)


def order_profit(o: pd.DataFrame) -> pd.Series:
    """Lợi nhuận trước quảng cáo của từng đơn. Hoàn thành: doanh thu − voucher − phí − giá vốn; hoàn: −chi phí hoàn; huỷ: 0."""
    fees = o[["phi_hoa_hong", "phi_thanh_toan", "phi_voucher_freeship", "phi_dich_vu", "thue_khau_tru"]].sum(axis=1)
    done = o["trang_thai"] == "Hoàn thành"
    return (o["doanh_thu"] - o["giam_gia_shop"] - fees - o["gia_von"]).where(done, 0) - o["chi_phi_hoan"]


def ground_truth(o: pd.DataFrame, ads: pd.DataFrame) -> dict:
    o = o.assign(thang=o["ngay_dat"].dt.month, loi_nhuan_don=order_profit(o))
    done = o["trang_thai"] == "Hoàn thành"
    o["dt_thuan"] = (o["doanh_thu"] - o["giam_gia_shop"]).where(done, 0)
    ads = ads.assign(thang=pd.to_datetime(ads["ngay"]).dt.month)
    fee_cols = ["phi_hoa_hong", "phi_thanh_toan", "phi_voucher_freeship", "phi_dich_vu", "thue_khau_tru"]

    m = o.groupby("thang").agg(dt_thuan=("dt_thuan", "sum"), ln_don=("loi_nhuan_don", "sum"),
                               phi=("phi_hoa_hong", "sum"))
    m["phi_san"] = o.groupby("thang")[fee_cols].sum().sum(axis=1)
    m["qc"] = ads.groupby("thang")["chi_phi_qc"].sum()
    m["loi_nhuan_rong"] = m["ln_don"] - m["qc"]
    m["bien_rong"] = m["loi_nhuan_rong"] / m["dt_thuan"]
    m["ty_le_phi"] = m["phi_san"] / m["dt_thuan"]
    monthly = {int(k): {"doanh_thu_thuan": round(v.dt_thuan), "phi_san": round(v.phi_san),
                        "ty_le_phi": round(v.ty_le_phi, 4), "quang_cao": round(v.qc),
                        "loi_nhuan_rong": round(v.loi_nhuan_rong), "bien_rong": round(v.bien_rong, 4)}
               for k, v in m.iterrows()}

    apr, may = m.loc[4], m.loc[5]
    may_done = done & (o["thang"] == 5)
    # Phần phí tăng thêm do S1 = điểm % tăng × doanh thu thuần tháng 5.
    s1_extra = PARAMS["fee_hike_pp"] * o.loc[may_done, "dt_thuan"].sum()

    sku = o.groupby(["sku", "ten_san_pham"]).agg(doanh_thu=("dt_thuan", "sum"), ln=("loi_nhuan_don", "sum")).reset_index()
    top10 = sku.nlargest(10, "doanh_thu")
    loss_top10 = top10[top10["ln"] < 0].sort_values("ln")

    wk = ads.assign(ngay=pd.to_datetime(ads["ngay"]))
    tt_ads = wk[wk["kenh"] == "TikTok Shop"].set_index("ngay")["chi_phi_qc"]
    tt_rev = o[o["kenh"] == "TikTok Shop"].set_index("ngay_dat")["dt_thuan"].resample("D").sum()
    s3_start, s3_end = pd.Timestamp(S3_WEEK[0]), pd.Timestamp(S3_WEEK[1])
    prev = (tt_ads.index >= s3_start - pd.Timedelta(days=28)) & (tt_ads.index < s3_start)
    cur = (tt_ads.index >= s3_start) & (tt_ads.index <= s3_end)
    prev_r = (tt_rev.index >= s3_start - pd.Timedelta(days=28)) & (tt_rev.index < s3_start)
    cur_r = (tt_rev.index >= s3_start) & (tt_rev.index <= s3_end)

    ret = o[o["trang_thai"] != "Đã huỷ"].assign(hoan=lambda d: d["trang_thai"] == "Đã trả hàng")
    rr = ret.groupby("thang")["hoan"].mean()
    seg = ret[ret["thang"].isin([2, 3])].groupby(["thang", "tinh", "don_vi_van_chuyen"])["hoan"].agg(["sum", "count"])
    seg_delta = (seg.loc[3]["sum"] - seg.loc[2]["sum"]).sort_values(ascending=False)

    return {
        "note": "Dữ liệu MÔ PHỎNG, đáp án tính lại từ chính dữ liệu sinh ra (seed cố định).",
        "seed": SEED,
        "n_orders": int(len(o)),
        "loi_nhuan_dinh_nghia": "doanh_thu − giam_gia_shop − 5 loại phí − gia_von (đơn Hoàn thành) − chi_phi_hoan (đơn Đã trả hàng) − quảng cáo",
        "monthly": monthly,
        "S1_fee_hike": {
            "from": PARAMS["fee_hike_date"], "hoa_hong_tang_pp": PARAMS["fee_hike_pp"],
            "loi_nhuan_thang_4": round(apr.loi_nhuan_rong), "loi_nhuan_thang_5": round(may.loi_nhuan_rong),
            "ty_le_phi_thang_4": round(apr.ty_le_phi, 4), "ty_le_phi_thang_5": round(may.ty_le_phi, 4),
            "phi_tang_them_thang_5": round(s1_extra),
            "tang_gia_de_giu_tien_ve_sau_phi": round(PARAMS["fee_hike_pp"] / (1 - apr.ty_le_phi - PARAMS["fee_hike_pp"]), 4),
        },
        "S2_loss_skus": [{"sku": r.sku, "doanh_thu_thuan": round(r.doanh_thu), "loi_nhuan_truoc_qc": round(r.ln)}
                         for r in loss_top10.itertuples()],
        "S3_tiktok_ads_week23": {
            "tuan": list(S3_WEEK),
            "qc_trung_binh_ngay_4_tuan_truoc": round(tt_ads[prev].mean()), "qc_trung_binh_ngay_tuan_23": round(tt_ads[cur].mean()),
            "dt_trung_binh_ngay_4_tuan_truoc": round(tt_rev[prev_r].mean()), "dt_trung_binh_ngay_tuan_23": round(tt_rev[cur_r].mean()),
        },
        "S4_return_spike": {
            "thang": S4_MONTH, "ty_le_hoan_thang_2": round(rr[2], 4), "ty_le_hoan_thang_3": round(rr[3], 4),
            "phan_khuc_gay_tang": {"tinh": seg_delta.index[0][0], "don_vi_van_chuyen": seg_delta.index[0][1],
                                   "so_don_hoan_tang_them": int(seg_delta.iloc[0])},
        },
    }


# Tên cột mô phỏng theo cấu trúc file export của từng sàn (không phải file thật).
SHOPEE_COLS = {
    "ma_don": "Mã đơn hàng", "ngay_dat": "Ngày đặt hàng", "trang_thai": "Trạng Thái Đơn Hàng",
    "sku": "SKU phân loại hàng", "ten_san_pham": "Tên sản phẩm", "so_luong": "Số lượng",
    "gia_ban": "Giá gốc", "doanh_thu": "Tổng giá bán (sản phẩm)", "giam_gia_shop": "Mã giảm giá của Shop",
    "phi_hoa_hong": "Phí cố định", "phi_thanh_toan": "Phí thanh toán", "phi_voucher_freeship": "Phí Voucher Xtra & Freeship Xtra",
    "phi_dich_vu": "Phí Dịch Vụ", "thue_khau_tru": "Thuế khấu trừ", "phi_ship_hoan": "Phí vận chuyển trả hàng", "tinh": "Tỉnh/Thành phố",
    "don_vi_van_chuyen": "Đơn Vị Vận Chuyển",
}
TIKTOK_COLS = {
    "ma_don": "Order ID", "ngay_dat": "Created Time", "trang_thai": "Order Status",
    "sku": "Seller SKU", "ten_san_pham": "Product Name", "so_luong": "Quantity",
    "gia_ban": "SKU Unit Original Price", "doanh_thu": "SKU Subtotal Before Discount",
    "giam_gia_shop": "SKU Seller Discount", "phi_hoa_hong": "Commission Fee", "phi_thanh_toan": "Transaction Fee",
    "phi_voucher_freeship": "Campaign Service Fee", "phi_dich_vu": "Platform Service Fee",
    "thue_khau_tru": "Tax Withheld", "phi_ship_hoan": "Return Shipping Fee", "tinh": "Province", "don_vi_van_chuyen": "Shipping Provider Name",
}
TIKTOK_STATUS = {"Hoàn thành": "Completed", "Đã huỷ": "Cancelled", "Đã trả hàng": "Returned"}


def generate(seed: int = SEED) -> dict[str, pd.DataFrame | dict]:
    rng = np.random.default_rng(seed)
    products = make_products(rng)
    orders = make_orders(rng, products)
    ads = make_ads(rng, orders)
    return {"products": products, "orders": orders, "ads": ads, "truth": ground_truth(orders, ads)}


def write(out: Path = OUT_DIR) -> dict:
    g = generate()
    out.mkdir(parents=True, exist_ok=True)
    o = g["orders"]
    o.to_csv(out / "shop_lan_orders.csv", index=False)
    g["ads"].to_csv(out / "ads_daily.csv", index=False)
    p = g["products"]
    p.assign(gia_von=(p["gia_ban"] * p["ty_le_gia_von"]).round(-2))[
        ["sku", "ten_san_pham", "danh_muc", "gia_ban", "gia_von"]].to_csv(out / "products.csv", index=False)

    # Sàn chỉ biết phí ship chiều về; phần hàng hỏng (return_damage_share) shop tự chịu, không có trong export.
    o = o.assign(phi_ship_hoan=np.where(o["trang_thai"] == "Đã trả hàng", PARAMS["return_ship_cost"], 0))
    sp = o[o["kenh"] == "Shopee"][list(SHOPEE_COLS)].rename(columns=SHOPEE_COLS)
    sp.to_csv(out / "export_shopee.csv", index=False)
    tt = o[o["kenh"] == "TikTok Shop"][list(TIKTOK_COLS)].copy()
    tt["trang_thai"] = tt["trang_thai"].map(TIKTOK_STATUS)
    tt["ngay_dat"] = tt["ngay_dat"].dt.strftime("%d/%m/%Y %H:%M:%S")
    tt.rename(columns=TIKTOK_COLS).to_csv(out / "export_tiktok.csv", index=False)

    (out / "ground_truth.json").write_text(json.dumps(g["truth"], ensure_ascii=False, indent=2), encoding="utf-8")
    return g["truth"]


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(write(), ensure_ascii=False, indent=2))
