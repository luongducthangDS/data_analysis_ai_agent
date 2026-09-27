"""T04/T07: định dạng số quyết định THEO CỘT; file export bị Excel lưu lại vẫn ra đúng số lãi.

Trước đây: "0,125" → 125; cùng một cột "1,5" → 1.5 nhưng "1,500" → 1500; file lưu từ
Excel Mac (NFD) mất cột lãi; cp1258 → 400; phân cách ";" → cả file thành 1 cột.
"""
import io
import unicodedata
from pathlib import Path

import pandas as pd
import pytest

from backend.app.services.numeric_parse import coerce_numeric_columns, parse_number, parse_numeric_series
from backend.app.services.storage import session_store


@pytest.mark.parametrize("raw, expected", [
    ("0,125", 0.125), ("0.125", 0.125), ("-0,5", -0.5),
    ("1.234,5", 1234.5), ("1,234.5", 1234.5), ("1.900", 1900.0), ("32.000.000", 32_000_000.0),
])
def test_parse_number_single_cell(raw, expected):
    assert parse_number(raw) == pytest.approx(expected)


COLUMNS = [
    (["1,5", "1,500", "2,25"], [1.5, 1.5, 2.25]),              # phẩy thập phân cả cột
    (["12.5", "12.500", "3.75"], [12.5, 12.5, 3.75]),          # chấm thập phân cả cột
    (["1.234.567", "1,5", "20"], [1_234_567.0, 1.5, 20.0]),    # kiểu Việt: chấm nghìn, phẩy thập phân
    (["1.900", "12.345.000"], [1900.0, 12_345_000.0]),         # chỉ có phân cách nghìn
]


@pytest.mark.parametrize("values, expected", COLUMNS)
def test_one_locale_per_column_at_load(values, expected):
    df = coerce_numeric_columns(pd.DataFrame({"x": values}))
    assert df["x"].tolist() == pytest.approx(expected)


@pytest.mark.parametrize("values, expected", COLUMNS)
def test_one_locale_per_column_at_execution(values, expected):
    assert parse_numeric_series(pd.Series(values, dtype=object)).tolist() == pytest.approx(expected)


# ── file export bị Excel mở rồi lưu lại ────────────────────────────────────

SHOP = Path("data/samples/shop_lan")
FILES = ("export_shopee.csv", "export_tiktok.csv", "products.csv", "ads_daily.csv")
_TONES = "̣̀́̃̉"   # huyền, sắc, ngã, hỏi, nặng


def _cp1258(data: bytes) -> bytes:
    """Mã hoá như Excel Windows tiếng Việt: chữ có mũ/móc dựng sẵn, dấu thanh là ký tự tổ hợp riêng."""
    out = bytearray()
    for ch in data.decode("utf-8"):
        try:
            out += ch.encode("cp1258")
            continue
        except UnicodeEncodeError:
            pass
        base, *marks = unicodedata.normalize("NFD", ch)
        tones = "".join(m for m in marks if m in _TONES)
        rest = "".join(m for m in marks if m not in _TONES)
        out += unicodedata.normalize("NFC", base + rest).encode("cp1258") + tones.encode("cp1258")
    return bytes(out)


def _nfd(data: bytes) -> bytes:
    return unicodedata.normalize("NFD", data.decode("utf-8")).encode("utf-8")


def _bom(data: bytes) -> bytes:
    return b"\xef\xbb\xbf" + data


def _semicolon(data: bytes) -> bytes:
    df = pd.read_csv(io.BytesIO(data), dtype=str, keep_default_na=False)
    return df.to_csv(sep=";", index=False).encode("utf-8")


def _load(transform) -> pd.DataFrame:
    uploads = []
    for name in FILES:
        raw = (SHOP / name).read_bytes()
        uploads.append((name, transform(raw) if name.startswith("export") else raw))
    return session_store.create_multiple(uploads).dataframe


@pytest.fixture(scope="module")
def baseline():
    return _load(lambda b: b)


@pytest.mark.parametrize("transform", [_bom, _nfd, _cp1258, _semicolon],
                         ids=["utf8-bom", "nfd-mac", "cp1258", "semicolon"])
def test_resaved_export_gives_same_profit(baseline, transform):
    df = _load(transform)
    assert len(df) == len(baseline)
    assert df["kenh"].value_counts().to_dict() == baseline["kenh"].value_counts().to_dict()
    for col in ("doanh_thu_thuan", "phi_san", "loi_nhuan_truoc_qc"):
        assert df[col].sum() == pytest.approx(baseline[col].sum()), col
