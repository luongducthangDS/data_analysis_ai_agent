#!/usr/bin/env python3
"""
eval_seller.py — đo đúng lời hứa của SellerLens trên shop MÔ PHỎNG (data/samples/shop_lan).

25 câu hỏi tự nhiên (có/không dấu) × 3 cách nạp:
  A = đủ file (2 export + giá vốn + quảng cáo)   B = chỉ 2 export (không giá vốn)
  C = export + bảng giá vốn thiếu 50/150 SKU

Chỉ số (docs/PM_FEEDBACK_2026-09-26.md §5):
  • north star  — % câu về lãi/phí/hoàn trả số khớp đáp án ±2%, hoặc từ chối/cảnh báo đúng khi thiếu dữ liệu (mục tiêu ≥ 90%)
  • số sai trông như đúng — câu có số mà không số nào khớp đáp án và không kèm cảnh báo (mục tiêu 0, chặn release)
  • kịch bản S1–S4 tìm đúng nguyên nhân (mục tiêu 4/4)

Đáp án tính độc lập bằng pandas từ shop_lan_orders.csv (không đi qua semantic layer của app).
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
from dataclasses import asdict, dataclass
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
    truth = json.loads((SHOP / "ground_truth.json").read_text(encoding="utf-8"))
    done, ret = o["trang_thai"] == "Hoàn thành", o["trang_thai"] == "Đã trả hàng"
    fees = o[["phi_hoa_hong", "phi_thanh_toan", "phi_voucher_freeship", "phi_dich_vu", "thue_khau_tru"]].sum(axis=1)
    o["rev"] = (o["doanh_thu"] - o["giam_gia_shop"]).where(done, 0)
    o["fee"] = fees.where(done, 0)
    # Lãi trước QC theo đúng thông tin có trong file export: chi phí hoàn = phí ship hoàn.
    o["pre"] = (o["rev"] - o["fee"] - o["gia_von"]).where(done, 0) - pd.Series(PARAMS["return_ship_cost"], index=o.index).where(ret, 0)
    ads = pd.read_csv(SHOP / "ads_daily.csv", parse_dates=["ngay"])
    m, am = o["ngay_dat"].dt.month, ads["ngay"].dt.month
    pre_m, ads_m = o.groupby(m)["pre"].sum(), ads.groupby(am)["chi_phi_qc"].sum()
    by_ch = o.groupby("kenh")[["rev", "fee", "pre"]].sum()
    rate = (by_ch["fee"] / by_ch["rev"]).sort_values(ascending=False)
    wk = ads[(ads["kenh"] == "TikTok Shop") & ads["ngay"].between("2026-06-01", "2026-06-07")]["chi_phi_qc"]
    valid3 = o[(m == 3) & (o["trang_thai"] != "Đã huỷ")]
    # Mode C: SKU ngoài 100 dòng đầu bảng sản phẩm không có giá vốn → app tính giá vốn = 0 cho các đơn đó.
    known = set(pd.read_csv(SHOP / "products.csv").head(100)["sku"])
    o["pre_c"] = o["pre"] + o["gia_von"].where(done & ~o["sku"].isin(known), 0)
    by_sku_c = o.groupby("sku")["pre_c"].sum()
    return {
        "pre": pre_m.to_dict(), "net": (pre_m - ads_m).to_dict(), "rev": o.groupby(m)["rev"].sum().to_dict(),
        "pre_total": o["pre"].sum(), "net_total": o["pre"].sum() - ads["chi_phi_qc"].sum(),
        "rev_total": o["rev"].sum(), "fee_total": o["fee"].sum(),
        "by_ch": by_ch, "rate_top": (rate.index[0], rate.iloc[0]),
        # S3: tổng/trung bình tuần 23 hoặc chính các ngày tăng vọt đều là câu trả lời đúng.
        "ads_week": [wk.sum(), wk.mean(), *wk.tolist()], "ret3": (valid3["trang_thai"] == "Đã trả hàng").mean(),
        "pre_c_total": o["pre_c"].sum(), "pre_c_month": o.groupby(m)["pre_c"].sum().tolist(),
        "worst_sku_c": by_sku_c.idxmin(),
        "s1": truth["S1_fee_hike"], "s2": [r["sku"] for r in truth["S2_loss_skus"]], "s4": truth["S4_return_spike"],
    }


def pct(x: float) -> list[str]:
    # Không kèm "%": "34,52%" cũng là đúng khi đáp án làm tròn là 34,5%.
    return [f"{x * 100:.1f}".replace(".", ","), f"{x * 100:.1f}"]


@dataclass
class Case:
    id: int
    mode: str               # A | B | C
    question: str
    kind: str               # number | refuse | warn | contains | no_numbers
    expect: list            # số (number) hoặc chuỗi (contains: phải có ĐỦ; contains_any dùng list lồng)
    profit: bool = False    # tính vào north star
    scenario: str = ""


def cases(t: dict) -> list[Case]:
    ch = t["by_ch"]
    top_ch, top_rate = t["rate_top"]
    s1, s4 = t["s1"], t["s4"]["phan_khuc_gay_tang"]
    return [
        Case(1, "A", "Tổng lãi 6 tháng là bao nhiêu?", "number", [t["pre_total"], t["net_total"]], True),
        Case(2, "A", "Lãi tháng 5 là bao nhiêu?", "number", [t["pre"][5], t["net"][5]], True),
        Case(3, "A", "Lãi ròng sau quảng cáo tháng 6", "number", [t["net"][6]], True),
        Case(4, "A", "Vì sao lãi tháng 5 giảm?", "contains",
             [[f"{s1['phi_tang_them_thang_5']:,}".replace(",", "."), "13,3 triệu"], pct(s1["tang_gia_de_giu_tien_ve_sau_phi"])], True, "S1"),
        Case(5, "A", "SKU nào doanh thu cao mà đang lỗ?", "contains", t["s2"], True, "S2"),
        Case(6, "A", "Tỷ lệ phí sàn Shopee và TikTok bên nào cao hơn?", "contains", [top_ch.split()[0], pct(top_rate)], True),
        Case(7, "A", "Tháng 3 tỉnh nào hoàn hàng tăng mạnh, do đơn vị vận chuyển nào?", "contains",
             [s4["tinh"], s4["don_vi_van_chuyen"].split()[0]], True, "S4"),
        Case(8, "A", "Quảng cáo TikTok tuần đầu tháng 6 có gì bất thường?", "number", t["ads_week"], False, "S3"),
        Case(9, "A", "Tổng doanh thu thuần 6 tháng", "number", [t["rev_total"]], True),
        Case(10, "A", "tháng này lãi bao nhiêu?", "number", [t["pre"][6], t["net"][6]], True),
        Case(11, "A", "lai thang 4 bao nhieu", "number", [t["pre"][4], t["net"][4]], True),
        Case(12, "A", "giá vàng hôm nay bao nhiêu", "no_numbers", []),
        Case(13, "A", "lãi tháng 7 bao nhiêu", "contains", [["30/06/2026", "không có"]], True),
        Case(14, "A", "Lãi theo kênh", "number", [ch.loc["Shopee", "pre"], ch.loc["TikTok Shop", "pre"]], True),
        Case(15, "A", "Tỷ lệ hoàn tháng 3", "contains", [pct(t["ret3"])], True),
        Case(16, "B", "Tổng lãi 6 tháng", "refuse", [], True),
        Case(17, "B", "sku nao lo nhat", "refuse", [], True),
        Case(18, "B", "Lãi theo kênh", "refuse", [], True),
        Case(19, "B", "Doanh thu thuần tháng 5", "number", [t["rev"][5]], True),
        Case(20, "B", "Tổng phí sàn 6 tháng", "number", [t["fee_total"]], True),
        Case(21, "B", "file này có gì?", "contains", ["giá vốn"]),
        Case(22, "C", "Tổng lãi 6 tháng", "warn", [t["pre_c_total"]], True),
        Case(23, "C", "SKU nào lỗ nhiều nhất", "warn", [t["worst_sku_c"]], True),
        Case(24, "C", "Lãi theo tháng", "warn", t["pre_c_month"], True),
        Case(25, "C", "Doanh thu thuần theo kênh", "number", [ch.loc["Shopee", "rev"], ch.loc["TikTok Shop", "rev"]], True),
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
    return hit, bool(nums) and not hit and not warned


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
    args = ap.parse_args()
    base, wanted = args.base_url.rstrip("/"), _ids(args.ids)
    todo = [c for c in cases(ground_truth()) if not wanted or c.id in wanted]
    sessions = {m: upload(base, m) for m in sorted({c.mode for c in todo})}

    rows = []
    for c in todo:
        answer, done, sec = ask(base, sessions[c.mode], c.question)
        ok, wrong = score(c, answer)
        rows.append({**asdict(c), "answer": answer, "ok": ok, "wrong_looks_right": wrong, "seconds": round(sec, 1),
                     "source": done.get("source"), "queries": done.get("executed_queries")})
        flag = "✅" if ok else ("🚨" if wrong else "❌")
        print(f"{flag} #{c.id:>2} [{c.mode}] {c.question}  ({sec:.1f}s, {done.get('source')})")
        if not ok:
            print("     " + answer.replace("\n", " ")[:300])

    profit = [r for r in rows if r["profit"]]
    scen = [r for r in rows if r["scenario"].startswith("S")]
    summary = {
        "north_star": round(sum(r["ok"] for r in profit) / len(profit), 3) if profit else None,
        "wrong_looks_right": sum(r["wrong_looks_right"] for r in rows),
        "scenarios": f"{sum(r['ok'] for r in scen)}/{len(scen)}",
        "passed": f"{sum(r['ok'] for r in rows)}/{len(rows)}",
        "p50_seconds": float(pd.Series([r["seconds"] for r in rows]).median()) if rows else None,
    }
    print("\n" + json.dumps(summary, ensure_ascii=False))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=1,
                                             default=str), encoding="utf-8")
    return 1 if summary["wrong_looks_right"] else 0


if __name__ == "__main__":
    sys.exit(main())
