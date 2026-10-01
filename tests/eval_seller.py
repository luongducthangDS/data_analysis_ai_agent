#!/usr/bin/env python3
"""
eval_seller.py — đo đúng lời hứa của SellerLens trên shop MÔ PHỎNG (data/samples/shop_lan).

167 câu hỏi tự nhiên (có/không dấu), chia theo 3 cách nạp:
  A = đủ file (2 export + giá vốn + quảng cáo), 103 câu   B = chỉ 2 export (không giá vốn), 36 câu
  C = export + bảng giá vốn thiếu 50/150 SKU, 28 câu
Câu 1–25 là bộ gốc (giữ nguyên id để so với lần chạy cũ); câu mới chỉ thêm vào cuối.

Chỉ số (docs/PM_FEEDBACK_2026-09-26.md §5):
  • north star  — % câu về lãi/phí/hoàn trả số khớp đáp án ±2%, hoặc từ chối/cảnh báo đúng khi thiếu dữ liệu (mục tiêu ≥ 90%)
  • số sai trông như đúng — câu có số mà không số nào khớp đáp án và không kèm cảnh báo (mục tiêu 0, chặn release)
  • kịch bản S1–S4 tìm đúng nguyên nhân (mỗi kịch bản hỏi vài cách, mục tiêu đạt hết)

Đáp án tính độc lập bằng pandas từ shop_lan_orders.csv (không đi qua semantic layer của app).
Lãi ròng theo SKU/tỉnh/danh mục chia QC theo doanh thu ngày × kênh, đúng định nghĩa chi_phi_qc của app.
Câu không lọc số (no_numbers) dùng số ≥ 1000 ngoài 1900–2100: số đơn theo tháng rơi vào vùng năm nên không hỏi.
File export không có phần hàng hỏng khi hoàn, nên lãi đường export = đáp án + phần đó (xem test_ecommerce_semantic).

Usage:
    uvicorn backend.app.main:app --port 8000
    python tests/eval_seller.py --base-url http://localhost:8000 [--ids 1,4,14-16] [--out results/seller.json]
"""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend.app.services.numeric_parse import parse_number  # noqa: E402
from scripts.gen_shop_lan import PARAMS  # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

SHOP = Path(__file__).resolve().parents[1] / "data" / "samples" / "shop_lan"
EXPORTS = ["export_shopee.csv", "export_tiktok.csv"]
TOL = 0.02
REFUSAL = ("chưa tính được lãi",)
WARNING = ("chưa có giá vốn", "thiếu giá vốn")


# ── Đáp án ──────────────────────────────────────────────────────────────────────

def ground_truth() -> dict:
    o = pd.read_csv(SHOP / "shop_lan_orders.csv", parse_dates=["ngay_dat"])
    ads = pd.read_csv(SHOP / "ads_daily.csv", parse_dates=["ngay"])
    truth = json.loads((SHOP / "ground_truth.json").read_text(encoding="utf-8"))
    done, ret = o["trang_thai"] == "Hoàn thành", o["trang_thai"] == "Đã trả hàng"
    fees = o[["phi_hoa_hong", "phi_thanh_toan", "phi_voucher_freeship", "phi_dich_vu", "thue_khau_tru"]].sum(axis=1)
    o["rev"] = (o["doanh_thu"] - o["giam_gia_shop"]).where(done, 0)
    o["fee"] = fees.where(done, 0)
    # Lãi trước QC theo đúng thông tin có trong file export: chi phí hoàn = phí ship hoàn.
    o["ship_hoan"] = pd.Series(PARAMS["return_ship_cost"], index=o.index).where(ret, 0)
    o["pre"] = (o["rev"] - o["fee"] - o["gia_von"]).where(done, 0) - o["ship_hoan"]
    # Lãi ròng: QC chia về từng đơn theo doanh thu thuần trong cùng ngày × kênh (định nghĩa chi_phi_qc của app).
    day = [o["ngay_dat"].dt.normalize(), o["kenh"]]
    spend = ads.groupby(["ngay", "kenh"])["chi_phi_qc"].sum().reindex(pd.MultiIndex.from_arrays(day), fill_value=0)
    o["net"] = o["pre"] - o["rev"] / o.groupby(day)["rev"].transform("sum") * spend.to_numpy()
    # Mode C: SKU ngoài 100 dòng đầu bảng sản phẩm không có giá vốn → app tính giá vốn = 0 cho các đơn đó.
    known = set(pd.read_csv(SHOP / "products.csv").head(100)["sku"])
    o["pre_c"] = o["pre"] + o["gia_von"].where(done & ~o["sku"].isin(known), 0)
    # Danh mục cũng đến từ bảng sản phẩm: SKU ngoài bảng → một nhóm riêng (app gọi là "(chưa có danh mục)").
    o["danh_muc_c"] = o["danh_muc"].where(o["sku"].isin(known), "(chưa có danh mục)")
    o["hoan"] = ret.astype(float).where(o["trang_thai"] != "Đã huỷ")  # mean = tỷ lệ hoàn
    o["xong"] = done
    o["thang"], o["quy"] = o["ngay_dat"].dt.month, o["ngay_dat"].dt.quarter
    ads["thang"] = ads["ngay"].dt.month
    wk = ads[(ads["kenh"] == "TikTok Shop") & ads["ngay"].between("2026-06-01", "2026-06-07")]["chi_phi_qc"]
    names = o.groupby("sku")["ten_san_pham"].first()  # "sản phẩm nào lỗ": mã SKU hay tên đều là trả lời đúng

    def by(*cols: str) -> pd.DataFrame:
        return o.groupby(list(cols)).agg(
            rev=("rev", "sum"), fee=("fee", "sum"), pre=("pre", "sum"), net=("net", "sum"), pre_c=("pre_c", "sum"),
            ship_hoan=("ship_hoan", "sum"), hoan=("hoan", "mean"), don=("ma_don", "size"), xong=("xong", "sum"))

    return {
        "by": by, "qc": ads,
        # Giá trị đơn trung bình: sau hay trước voucher shop đều đúng.
        "aov": [o["rev"].sum() / done.sum(), o.loc[done, "doanh_thu"].mean()],
        # S3: tổng/trung bình tuần 23 hoặc chính các ngày tăng vọt đều là câu trả lời đúng.
        "ads_week": [wk.sum(), wk.mean(), *wk.tolist()],
        "s1": truth["S1_fee_hike"], "s2": [[r["sku"], names[r["sku"]]] for r in truth["S2_loss_skus"]], "s4": truth["S4_return_spike"],
    }


def pct(x: float) -> list[str]:
    # Không kèm "%": "34,52%" cũng là đúng khi đáp án làm tròn là 34,5%. Có cả 2 chữ số: 14,555% → app ghi "14,56%".
    # Lệch ±0,06 điểm %: 38,85% nằm đúng ranh giới làm tròn, app ghi "38,8%" vẫn là đúng.
    v = x * 100
    texts = {f"{y:.1f}" for y in (v - 0.06, v, v + 0.06)} | {f"{v:.2f}"}
    out = sorted({t.replace(".", sep) for t in texts for sep in (",", ".")})
    return out + ([f"{v:.0f}%"] if f"{v:.1f}".endswith(".0") else [])  # 12,0% → app ghi "12%"


def month(k: int) -> list[str]:
    return [f"tháng {k}", f"{k:02d}/2026", f"2026-{k:02d}"]


@dataclass
class Case:
    id: int
    mode: str               # A | B | C
    question: str
    kind: str               # number | refuse | warn | contains | no_numbers
    expect: list            # số (number) hoặc chuỗi (contains: phải có ĐỦ; contains_any dùng list lồng)
    profit: bool = False    # tính vào north star
    scenario: str = ""
    # Số đúng nhưng chưa phải câu trả lời (hỏi "tăng bao nhiêu", trả lời phí 2 tháng): trượt, không phải 🚨.
    near: list = field(default_factory=list)


def cases(t: dict) -> list[Case]:
    by, qc = t["by"], t["qc"]
    mo, q, ch, cat, prov, car, sku = (by(c) for c in ("thang", "quy", "kenh", "danh_muc", "tinh", "don_vi_van_chuyen", "sku"))
    chm, tot = by("kenh", "thang"), by("thang").sum()  # tot: chỉ dùng cột cộng được (không dùng hoan)
    cat_c = by("danh_muc_c")  # mode C: danh mục chỉ biết cho SKU có trong bảng giá vốn
    qc_m, qc_chm = qc.groupby("thang")["chi_phi_qc"].sum(), qc.groupby(["kenh", "thang"])["chi_phi_qc"].sum()
    rate, ch_rate = mo.fee / mo.rev, ch.fee / ch.rev
    sh, tt = "Shopee", "TikTok Shop"
    s1, s4 = t["s1"], t["s4"]["phan_khuc_gay_tang"]
    s1x = [[f"{s1['phi_tang_them_thang_5']:,}".replace(",", "."), "13,3 triệu"], pct(s1["tang_gia_de_giu_tien_ve_sau_phi"])]
    s4x = [s4["tinh"], s4["don_vi_van_chuyen"].split()[0]]
    fee_top = [ch_rate.idxmax().split()[0], pct(ch_rate.max())]
    fee_up = [mo.fee[5] - mo.fee[4], s1["phi_tang_them_thang_5"]]  # chênh lệch thô hoặc phần do tăng tỷ lệ phí
    nodata = [["30/06/2026", "không có"]]

    def lai(f: pd.DataFrame, k) -> list[float]:
        """'Lãi' trơn: trước hay sau quảng cáo đều đúng. Hỏi rõ 'lãi ròng'/'trước QC' thì case chỉ nhận một số."""
        return [f.pre[k], f.net[k]]

    def top(f: pd.DataFrame) -> list[float]:
        """"Lãi nhiều nhất": đứng đầu theo lãi trước QC hay sau QC đều đúng (danh mục: Áo vs Váy/Đầm)."""
        return [f.pre.max(), f.net.max()]

    return [
        Case(1, "A", "Tổng lãi 6 tháng là bao nhiêu?", "number", [tot.pre, tot.net], True),
        Case(2, "A", "Lãi tháng 5 là bao nhiêu?", "number", lai(mo, 5), True),
        Case(3, "A", "Lãi ròng sau quảng cáo tháng 6", "number", [mo.net[6]], True),
        Case(4, "A", "Vì sao lãi tháng 5 giảm?", "contains", s1x, True, "S1"),
        Case(5, "A", "SKU nào doanh thu cao mà đang lỗ?", "contains", t["s2"], True, "S2"),
        Case(6, "A", "Tỷ lệ phí sàn Shopee và TikTok bên nào cao hơn?", "contains", fee_top, True),
        Case(7, "A", "Tháng 3 tỉnh nào hoàn hàng tăng mạnh, do đơn vị vận chuyển nào?", "contains", s4x, True, "S4"),
        Case(8, "A", "Quảng cáo TikTok tuần đầu tháng 6 có gì bất thường?", "number", t["ads_week"], False, "S3"),
        Case(9, "A", "Tổng doanh thu thuần 6 tháng", "number", [tot.rev], True),
        Case(10, "A", "tháng này lãi bao nhiêu?", "number", lai(mo, 6), True),
        Case(11, "A", "lai thang 4 bao nhieu", "number", lai(mo, 4), True),
        Case(12, "A", "giá vàng hôm nay bao nhiêu", "no_numbers", []),
        Case(13, "A", "lãi tháng 7 bao nhiêu", "contains", nodata, True),
        Case(14, "A", "Lãi theo kênh", "number", lai(ch, sh) + lai(ch, tt), True),
        Case(15, "A", "Tỷ lệ hoàn tháng 3", "contains", [pct(mo.hoan[3])], True),
        Case(16, "B", "Tổng lãi 6 tháng", "refuse", [], True),
        Case(17, "B", "sku nao lo nhat", "refuse", [], True),
        Case(18, "B", "Lãi theo kênh", "refuse", [], True),
        Case(19, "B", "Doanh thu thuần tháng 5", "number", [mo.rev[5]], True),
        Case(20, "B", "Tổng phí sàn 6 tháng", "number", [tot.fee], True),
        Case(21, "B", "file này có gì?", "contains", ["giá vốn"]),
        Case(22, "C", "Tổng lãi 6 tháng", "warn", [tot.pre_c], True),
        Case(23, "C", "SKU nào lỗ nhiều nhất", "warn", [sku.pre_c.idxmin()], True),
        Case(24, "C", "Lãi theo tháng", "warn", mo.pre_c.tolist(), True),
        Case(25, "C", "Doanh thu thuần theo kênh", "number", [ch.rev[sh], ch.rev[tt]], True),
        # ── A: đủ file — lãi theo tháng/kênh/quý ──
        Case(26, "A", "Lãi tháng 1 là bao nhiêu?", "number", lai(mo, 1), True),
        Case(27, "A", "Lợi nhuận tháng 2", "number", lai(mo, 2), True),
        Case(28, "A", "tháng 3 lời được bao nhiêu", "number", lai(mo, 3), True),
        Case(29, "A", "lai thang 6 bao nhieu", "number", lai(mo, 6), True),
        Case(30, "A", "Lãi tháng trước bao nhiêu?", "number", lai(mo, 5), True),  # dữ liệu hết 30/06 → tháng trước = 5
        Case(31, "A", "Lãi ròng tháng 1 sau khi trừ quảng cáo", "number", [mo.net[1]], True),
        Case(32, "A", "Lợi nhuận ròng tháng 2", "number", [mo.net[2]], True),
        Case(33, "A", "lai rong thang 3", "number", [mo.net[3]], True),
        Case(34, "A", "Lãi sau quảng cáo tháng 4 còn bao nhiêu?", "number", [mo.net[4]], True),
        Case(35, "A", "Lãi ròng tháng 5", "number", [mo.net[5]], True),
        Case(36, "A", "Tổng lãi ròng 6 tháng sau quảng cáo", "number", [tot.net], True),
        Case(37, "A", "Lãi ròng theo tháng", "number", mo.net.tolist(), True),
        Case(38, "A", "Lãi trước quảng cáo tháng 6", "number", [mo.pre[6]], True),
        Case(39, "A", "Lãi trước quảng cáo theo kênh", "number", [ch.pre[sh], ch.pre[tt]], True),
        Case(40, "A", "Shopee lãi bao nhiêu?", "number", lai(ch, sh), True),
        Case(41, "A", "TikTok Shop lãi ròng sau quảng cáo bao nhiêu?", "number", [ch.net[tt]], True),
        Case(42, "A", "Lãi tháng 5 của Shopee", "number", lai(chm, (sh, 5)), True),
        Case(43, "A", "lai tiktok thang 6", "number", lai(chm, (tt, 6)), True),
        Case(44, "A", "Quý 1 lãi bao nhiêu?", "number", lai(q, 1), True),
        Case(45, "A", "Lãi ròng quý 2", "number", [q.net[2]], True),
        Case(46, "A", "Trung bình mỗi tháng lãi bao nhiêu?", "number", [tot.pre / 6, tot.net / 6], True),
        Case(47, "A", "Biên lợi nhuận tháng 6 là bao nhiêu %?", "contains",
             [pct(mo.pre[6] / mo.rev[6]) + pct(mo.net[6] / mo.rev[6])], True),
        Case(48, "A", "Biên lãi ròng tháng 5", "contains", [pct(mo.net[5] / mo.rev[5])], True),
        Case(49, "A", "Tháng nào lãi thấp nhất?", "contains", [month(mo.pre.idxmin())], True),
        Case(50, "A", "Lãi tháng 6 so với tháng 5 thay đổi bao nhiêu?", "number",
             [mo.pre[6] - mo.pre[5], mo.net[6] - mo.net[5]], True),
        Case(51, "A", "Vì sao lãi tháng 6 thấp hơn tháng 4?", "number", [mo.pre[6] - mo.pre[4], mo.net[6] - mo.net[4]], True),
        Case(52, "A", "Tại sao tháng 5 lãi ít hơn tháng 4?", "contains", s1x, True, "S1"),
        Case(53, "A", "lai thang 5 giam vi sao", "contains", s1x, True, "S1"),
        Case(54, "A", "Vì sao lãi tháng trước giảm?", "contains", s1x, True, "S1"),
        # ── A: SKU / danh mục / tỉnh ──
        Case(55, "A", "Những SKU nào đang lỗ?", "contains", t["s2"], True, "S2"),
        Case(56, "A", "Sản phẩm nào bán nhiều mà vẫn lỗ?", "contains", t["s2"], True, "S2"),
        Case(57, "A", "SKU nào lãi nhiều nhất?", "contains", [[sku.pre.idxmax(), sku.net.idxmax()]], True),
        Case(58, "A", "LAN-127 lỗ bao nhiêu?", "number", lai(sku, "LAN-127"), True),
        Case(59, "A", "Lãi của SKU LAN-011", "number", lai(sku, "LAN-011"), True),
        Case(60, "A", "Danh mục nào lãi nhiều nhất?", "number", top(cat), True),
        Case(61, "A", "Danh mục Váy/Đầm lãi bao nhiêu?", "number", lai(cat, "Váy/Đầm"), True),
        Case(62, "A", "Tỉnh nào lãi nhiều nhất?", "number", top(prov), True),
        Case(63, "A", "Lãi ở Hà Nội 6 tháng", "number", lai(prov, "Hà Nội"), True),
        Case(64, "A", "Shopee hay TikTok lãi nhiều hơn?", "number", top(ch), True),
        # ── A: doanh thu thuần ──
        Case(65, "A", "Doanh thu thuần tháng 1", "number", [mo.rev[1]], True),
        Case(66, "A", "Doanh thu thuần tháng 3", "number", [mo.rev[3]], True),
        Case(67, "A", "doanh thu thuan thang 6", "number", [mo.rev[6]], True),
        Case(68, "A", "Doanh thu thuần TikTok tháng 6", "number", [chm.rev[(tt, 6)]], True),
        Case(69, "A", "Doanh thu thuần quý 2", "number", [q.rev[2]], True),
        Case(70, "A", "Tháng nào doanh thu thuần cao nhất?", "contains", [month(mo.rev.idxmax())], True),
        Case(71, "A", "Doanh thu thuần ở TP.HCM", "number", [prov.rev["TP.HCM"]], True),
        Case(72, "A", "Top 3 SKU doanh thu cao nhất", "contains", sku.rev.nlargest(3).index.tolist(), True),
        Case(73, "A", "SKU bán chạy nhất là mã nào?", "contains", [sku.rev.idxmax()]),  # theo doanh thu hay số lượng đều là một mã
        # ── A: phí sàn ──
        Case(74, "A", "Phí sàn tháng 4 hết bao nhiêu?", "number", [mo.fee[4]], True),
        Case(75, "A", "Tổng phí sàn tháng 5", "number", [mo.fee[5]], True),
        Case(76, "A", "phi san thang 6", "number", [mo.fee[6]], True),
        Case(77, "A", "Phí sàn Shopee 6 tháng", "number", [ch.fee[sh]], True),
        Case(78, "A", "Tổng phí sàn TikTok Shop", "number", [ch.fee[tt]], True),
        Case(79, "A", "Tỷ lệ phí sàn tháng 4", "contains", [pct(rate[4])], True),
        Case(80, "A", "Tỷ lệ phí sàn tháng 5 là bao nhiêu %?", "contains", [pct(rate[5])], True),
        Case(81, "A", "Phí sàn tháng 5 tăng bao nhiêu so với tháng 4?", "number", fee_up, True,
             near=[mo.fee[4], mo.fee[5]]),
        Case(82, "A", "Phí sàn trung bình mỗi đơn hoàn thành", "number", [tot.fee / tot.xong], True),
        # ── A: hoàn hàng ──
        Case(83, "A", "Tổng phí ship hoàn hàng 6 tháng", "number", [tot.ship_hoan], True),
        Case(84, "A", "Tỷ lệ hoàn hàng tháng 2", "contains", [pct(mo.hoan[2])], True),
        Case(85, "A", "ty le hoan thang 5", "contains", [pct(mo.hoan[5])], True),
        Case(86, "A", "Tỷ lệ hoàn theo kênh", "contains", [pct(ch.hoan[sh]), pct(ch.hoan[tt])], True),
        Case(87, "A", "Đơn vị vận chuyển nào tỷ lệ hoàn cao nhất?", "contains", [car.hoan.idxmax().split()[0]], True),
        Case(88, "A", "Tỉnh nào tỷ lệ hoàn cao nhất?", "contains", [prov.hoan.idxmax()], True),
        Case(89, "A", "Danh mục nào ít bị hoàn nhất?", "contains", [cat.hoan.idxmin()], True),
        Case(90, "A", "Tỷ lệ hoàn của J&T Express", "contains", [pct(car.hoan["J&T Express"])], True),
        Case(91, "A", "Nghệ An tháng 3 tỷ lệ hoàn bao nhiêu?", "contains", [pct(by("tinh", "thang").hoan[("Nghệ An", 3)])], True),
        Case(92, "A", "Vì sao tỷ lệ hoàn tháng 3 tăng?", "contains", s4x, True, "S4"),
        Case(93, "A", "hoan hang thang 3 tang o tinh nao, don vi van chuyen nao", "contains", s4x, True, "S4"),
        # ── A: quảng cáo, số đơn ──
        Case(94, "A", "Chi phí quảng cáo tháng 6", "number", [qc_m[6]]),
        Case(95, "A", "Tổng tiền quảng cáo 6 tháng", "number", [qc_m.sum()]),
        Case(96, "A", "Quảng cáo TikTok tháng 6 tốn bao nhiêu?", "number", [qc_chm[(tt, 6)]]),
        Case(97, "A", "Chi phí quảng cáo Shopee theo tháng", "number", qc_chm[sh].tolist()),
        Case(98, "A", "Chi phí quảng cáo bằng bao nhiêu % doanh thu thuần?", "contains", [pct(qc_m.sum() / tot.rev)]),
        Case(99, "A", "Quảng cáo TikTok có ngày nào tăng bất thường không?", "number", t["ads_week"], False, "S3"),
        Case(100, "A", "qc tiktok dau thang 6 co tang bat thuong khong", "number", t["ads_week"], False, "S3"),
        # Số đơn theo tháng (~1.800–2.100) trùng vùng bị lọc là năm 1900–2100 nên chỉ hỏi số không rơi vào đó.
        Case(101, "A", "Tháng 3 có bao nhiêu đơn hoàn thành?", "number", [mo.xong[3]]),
        Case(102, "A", "Shopee có bao nhiêu đơn?", "number", [ch.don[sh], ch.xong[sh]]),
        Case(103, "A", "Tổng số đơn hoàn thành 6 tháng", "number", [tot.xong]),
        Case(104, "A", "Giá trị trung bình mỗi đơn hoàn thành", "number", t["aov"], True),
        # ── A: ngoài dữ liệu → không được đưa số ──
        Case(105, "A", "Thời tiết Hà Nội hôm nay thế nào?", "no_numbers", []),
        Case(106, "A", "Tỷ giá đô la hôm nay bao nhiêu?", "no_numbers", []),
        Case(107, "A", "Giá bitcoin bây giờ", "no_numbers", []),
        Case(108, "A", "Lãi suất ngân hàng hiện nay bao nhiêu?", "no_numbers", []),  # "lãi" này không phải lãi shop
        Case(109, "A", "Viết giúp chị một bài thơ về mùa thu", "no_numbers", []),
        Case(110, "A", "Doanh thu thuần tháng 12 năm 2025", "contains", nodata, True),
        Case(111, "A", "Lãi năm 2025", "contains", nodata, True),
        Case(112, "A", "lai thang 8", "contains", nodata, True),
        Case(113, "A", "file này có gì vậy em?", "contains", ["đã ghép"]),
        # ── B: không giá vốn — mọi câu hỏi lãi phải từ chối; doanh thu/phí/hoàn vẫn trả lời ──
        Case(114, "B", "Lãi tháng 5 bao nhiêu?", "refuse", [], True),
        Case(115, "B", "lai thang 6", "refuse", [], True),
        Case(116, "B", "Lợi nhuận theo tháng", "refuse", [], True),
        Case(117, "B", "Shopee lời bao nhiêu?", "refuse", [], True),
        Case(118, "B", "Vì sao lãi tháng 5 giảm?", "refuse", [], True),
        Case(119, "B", "SKU nào đang lỗ?", "refuse", [], True),
        Case(120, "B", "Biên lợi nhuận theo kênh", "refuse", [], True),
        Case(121, "B", "Lãi ròng sau quảng cáo", "refuse", [], True),
        Case(122, "B", "Danh mục nào lãi nhiều nhất?", "refuse", [], True),
        Case(123, "B", "Tỉnh nào lời nhiều nhất?", "refuse", [], True),
        Case(124, "B", "Top 5 sản phẩm lãi cao nhất", "refuse", [], True),
        Case(125, "B", "Tổng lợi nhuận", "refuse", [], True),
        Case(126, "B", "Tháng 6 lỗ hay lãi?", "refuse", [], True),
        Case(127, "B", "profit by channel", "refuse", [], True),
        Case(128, "B", "Mỗi đơn lãi trung bình bao nhiêu?", "refuse", [], True),
        Case(129, "B", "Doanh thu thuần tháng 1", "number", [mo.rev[1]], True),
        Case(130, "B", "doanh thu thuan thang 4", "number", [mo.rev[4]], True),
        Case(131, "B", "Doanh thu thuần quý 1", "number", [q.rev[1]], True),
        Case(132, "B", "Phí sàn tháng 5", "number", [mo.fee[5]], True),
        Case(133, "B", "Phí sàn TikTok 6 tháng", "number", [ch.fee[tt]], True),
        Case(134, "B", "Tỷ lệ phí sàn tháng 6", "contains", [pct(rate[6])], True),
        Case(135, "B", "Phí sàn tháng 5 tăng bao nhiêu so với tháng 4?", "number", fee_up, True,
             near=[mo.fee[4], mo.fee[5]]),
        Case(136, "B", "Tỷ lệ hoàn tháng 3", "contains", [pct(mo.hoan[3])], True),
        Case(137, "B", "Tỉnh nào doanh thu thuần cao nhất?", "number", [prov.rev.max()], True),
        Case(138, "B", "SKU doanh thu cao nhất", "contains", [sku.rev.idxmax()], True),
        Case(139, "B", "Đơn vị vận chuyển nào tỷ lệ hoàn cao nhất?", "contains", [car.hoan.idxmax().split()[0]], True),
        Case(140, "B", "Tháng 3 tỉnh nào hoàn hàng tăng mạnh, do đơn vị vận chuyển nào?", "contains", s4x, True, "S4"),
        Case(141, "B", "Tỷ lệ phí sàn Shopee và TikTok bên nào cao hơn?", "contains", fee_top, True),
        Case(142, "B", "Có bao nhiêu đơn hoàn thành?", "number", [tot.xong]),
        Case(143, "B", "giá vàng hôm nay bao nhiêu", "no_numbers", []),
        # ── C: thiếu giá vốn 50/150 SKU — câu có lãi phải kèm cảnh báo VÀ đúng số theo dữ liệu đang có ──
        Case(144, "C", "Lãi tháng 1", "warn", [mo.pre_c[1]], True),
        Case(145, "C", "Lãi tháng 2 thế nào?", "warn", [mo.pre_c[2]], True),
        Case(146, "C", "Lãi tháng 3 bao nhiêu?", "warn", [mo.pre_c[3]], True),
        Case(147, "C", "lai thang 4 bao nhieu", "warn", [mo.pre_c[4]], True),
        Case(148, "C", "lai thang 5", "warn", [mo.pre_c[5]], True),
        Case(149, "C", "Lợi nhuận tháng 6", "warn", [mo.pre_c[6]], True),
        Case(150, "C", "Lãi theo kênh", "warn", [ch.pre_c[sh], ch.pre_c[tt]], True),
        Case(151, "C", "Shopee lãi bao nhiêu?", "warn", [ch.pre_c[sh]], True),
        Case(152, "C", "TikTok lãi bao nhiêu?", "warn", [ch.pre_c[tt]], True),
        # Mọi SKU Chân váy nằm ngoài bảng giá vốn → app không biết danh mục này; phải nói ra, không đưa số.
        Case(153, "C", "Danh mục Chân váy lãi bao nhiêu?", "contains", ["chưa có danh mục"], True),
        Case(154, "C", "Lãi theo danh mục", "warn", cat_c.pre_c.tolist(), True),
        Case(155, "C", "Tỉnh nào lãi nhiều nhất?", "warn", [prov.pre_c.max()], True),
        Case(156, "C", "SKU nào lãi nhiều nhất?", "warn", [sku.pre_c.idxmax()], True),
        Case(157, "C", "Lãi quý 2", "warn", [q.pre_c[2]], True),
        Case(158, "C", "Tổng lợi nhuận", "warn", [tot.pre_c], True),
        Case(159, "C", "Vì sao lãi tháng 5 giảm?", "warn", [mo.pre_c[5] - mo.pre_c[4]], True),
        Case(160, "C", "Doanh thu thuần tháng 2", "number", [mo.rev[2]], True),
        Case(161, "C", "Phí sàn theo kênh", "number", [ch.fee[sh], ch.fee[tt]], True),
        Case(162, "C", "Tỷ lệ hoàn tháng 5", "contains", [pct(mo.hoan[5])], True),
        Case(163, "C", "Tỷ lệ phí sàn tháng 5", "contains", [pct(rate[5])], True),
        Case(164, "C", "Tổng doanh thu thuần 6 tháng", "number", [tot.rev], True),
        Case(165, "C", "SKU doanh thu cao nhất", "contains", [sku.rev.idxmax()], True),
        Case(166, "C", "file này có gì?", "contains", ["50 SKU"]),
        Case(167, "C", "Danh mục Quần lãi bao nhiêu?", "warn", [cat_c.pre_c["Quần"]], True),  # chỉ phần SKU có trong bảng
    ]


# ── Chấm ────────────────────────────────────────────────────────────────────────

_UNIT = {"tỷ": 1e9, "ty": 1e9, "triệu": 1e6, "trieu": 1e6, "tr": 1e6, "nghìn": 1e3, "ngàn": 1e3, "k": 1e3}
_NUM = re.compile(r"(?<![\w])(-?\d[\d.,]*)\s*(tỷ|ty|triệu|trieu|tr|nghìn|ngàn|k)?(?![\w%])", re.IGNORECASE)


def numbers(text: str) -> list[float]:
    """Mọi số tiền trong câu trả lời, hiểu cả '15,7 triệu', '3,3 tỷ', '13.308.870 đ'. Bỏ %."""
    out = []
    for raw, unit in _NUM.findall(text):
        value = parse_number(raw)
        if value is None:
            continue
        out.append(abs(value) * _UNIT.get((unit or "").lower(), 1))
    return [v for v in out if v >= 1000 and not (v.is_integer() and 1900 <= v <= 2100)]


def score(case: Case, answer: str) -> tuple[bool, bool]:
    """(đạt, số_sai_trông_như_đúng)."""
    low = answer.lower()
    nums = numbers(answer)
    warned = any(w in low for w in WARNING + REFUSAL)
    if case.kind == "refuse":
        ok = any(r in low for r in REFUSAL)
        return ok, (not ok) and bool(nums)
    if case.kind == "warn":
        # Cảnh báo là bắt buộc, nhưng số đi kèm vẫn phải đúng theo dữ liệu đang có.
        strings = [e for e in case.expect if isinstance(e, str)]
        values = [e for e in case.expect if not isinstance(e, str)]
        right = (all(x.lower() in low for x in strings)
                 and (not values or any(abs(n - abs(e)) <= TOL * abs(e) for n in nums for e in values)))
        ok = any(w in low for w in WARNING) and right
        return ok, (not right) and bool(nums)
    if case.kind == "no_numbers":
        return not nums, bool(nums)
    if case.kind == "contains":
        ok = all(any(x.lower() in low for x in (e if isinstance(e, list) else [e])) for e in case.expect)
        return ok, False
    hit = any(abs(n - abs(e)) <= TOL * abs(e) for n in nums for e in case.expect)
    near = any(abs(n - abs(e)) <= TOL * abs(e) for n in nums for e in case.near)
    return hit, bool(nums) and not hit and not warned and not near


# ── Chạy ────────────────────────────────────────────────────────────────────────

def upload(base: str, mode: str) -> str:
    files = [(n, (SHOP / n).read_bytes()) for n in EXPORTS]
    if mode == "A":
        files += [(n, (SHOP / n).read_bytes()) for n in ("products.csv", "ads_daily.csv")]
    elif mode == "C":
        buf = io.BytesIO()
        pd.read_csv(SHOP / "products.csv").head(100).to_csv(buf, index=False)
        files.append(("gia_von.csv", buf.getvalue()))
    r = requests.post(f"{base}/api/upload", files=[("files", (n, b, "text/csv")) for n, b in files], timeout=120)
    r.raise_for_status()
    return r.json()["session_id"]


def ask(base: str, sid: str, question: str) -> tuple[str, dict, float]:
    t0 = time.time()
    r = requests.post(f"{base}/api/chat/stream", json={"session_id": sid, "question": question}, stream=True, timeout=180)
    answer, done = "", {}
    for line in r.iter_lines(decode_unicode=True):
        if line and line.startswith("data: "):
            ev = json.loads(line[6:])
            if ev["type"] == "token":
                answer += ev["content"]
            elif ev["type"] == "done":
                done = ev
            elif ev["type"] == "error":
                answer += f"[ERROR] {ev.get('detail')}"
    return answer, done, time.time() - t0


def _ids(spec: str | None) -> set[int] | None:
    if not spec:
        return None
    out: set[int] = set()
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        out.update(range(int(lo), int(hi or lo) + 1))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--ids")
    ap.add_argument("--out")
    ap.add_argument("--delay", type=float, default=0.0,
                    help="Giãn cách giữa các câu (giây). Gemini free tier 15 request/phút/model, mỗi câu ~2 lượt gọi")
    ap.add_argument("--merge", nargs="+", help="Gộp các file --out (file sau ghi đè câu trùng id), không gọi server")
    args = ap.parse_args()
    if args.merge:  # vd chạy lại các câu rơi xuống fallback vì hết quota rồi gộp với lần chạy chính
        by_id = {r["id"]: r for f in args.merge for r in json.loads(Path(f).read_text(encoding="utf-8"))["rows"]}
        # Chấm lại câu trả lời đã lưu bằng đáp án/bộ chấm hiện tại: sửa bộ chấm không cần gọi lại LLM.
        current = {c.id: c for c in cases(ground_truth())}
        rows = []
        for k in sorted(by_id):
            ok, wrong = score(current[k], by_id[k]["answer"])
            rows.append({**by_id[k], **asdict(current[k]), "ok": ok, "wrong_looks_right": wrong})
    else:
        rows = run(args.base_url.rstrip("/"), _ids(args.ids), args.delay)
    summary = summarize(rows)
    print("\n" + json.dumps(summary, ensure_ascii=False))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=1,
                                             default=str), encoding="utf-8")
    return 1 if summary["wrong_looks_right"] else 0


def run(base: str, wanted: set[int] | None, delay: float) -> list[dict]:
    todo = [c for c in cases(ground_truth()) if not wanted or c.id in wanted]
    sessions = {m: upload(base, m) for m in sorted({c.mode for c in todo})}
    rows = []
    for i, c in enumerate(todo):
        if delay and i:
            time.sleep(delay)  # chạy dồn thì hết quota → câu rơi xuống fallback, không đo được LLM
        answer, done, sec = ask(base, sessions[c.mode], c.question)
        ok, wrong = score(c, answer)
        rows.append({**asdict(c), "answer": answer, "ok": ok, "wrong_looks_right": wrong, "seconds": round(sec, 1),
                     "source": done.get("source"), "queries": done.get("executed_queries")})
        flag = "✅" if ok else ("🚨" if wrong else "❌")
        print(f"{flag} #{c.id:>2} [{c.mode}] {c.question}  ({sec:.1f}s, {done.get('source')})")
        if not ok:
            print("     " + answer.replace("\n", " ")[:300])
    return rows


def summarize(rows: list[dict]) -> dict:
    profit = [r for r in rows if r["profit"]]
    scen = [r for r in rows if r["scenario"].startswith("S")]
    # Từ chối = câu mà đáp án đúng là không đưa số (thiếu giá vốn, câu ngoài dữ liệu).
    refuse = [r for r in rows if r["kind"] in ("refuse", "no_numbers")]
    answered = [r for r in rows if r["kind"] not in ("refuse", "no_numbers")]
    secs = pd.Series([r["seconds"] for r in rows])
    return {
        "north_star": round(sum(r["ok"] for r in profit) / len(profit), 3) if profit else None,
        "wrong_looks_right": sum(r["wrong_looks_right"] for r in rows),
        "scenarios": f"{sum(r['ok'] for r in scen)}/{len(scen)}",
        "passed": f"{sum(r['ok'] for r in rows)}/{len(rows)}",
        "by_mode": {m: f"{sum(r['ok'] for r in rows if r['mode'] == m)}/{sum(r['mode'] == m for r in rows)}"
                    for m in sorted({r["mode"] for r in rows})},
        "answered_ok": f"{sum(r['ok'] for r in answered)}/{len(answered)}",
        "refused_ok": f"{sum(r['ok'] for r in refuse)}/{len(refuse)}",
        "p50_seconds": round(float(secs.median()), 1) if rows else None,
        "p95_seconds": round(float(secs.quantile(0.95)), 1) if rows else None,
    }


if __name__ == "__main__":
    sys.exit(main())
