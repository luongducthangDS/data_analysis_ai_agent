"""T01: một ô CSV dài bất thường không được làm treo parse_number (regex O(n²) giữ GIL)."""
from __future__ import annotations

import time

from backend.app.services.numeric_parse import parse_number


def test_overlong_digit_string_returns_none_fast():
    payload = "1" * 100_000 + "!"   # float() fail, không có chữ cái → rơi xuống _PLAIN_NUMBER_RE.fullmatch
    start = time.perf_counter()
    assert parse_number(payload) is None
    assert time.perf_counter() - start < 0.001


def test_long_but_real_numbers_still_parse():
    assert parse_number("$ (1,234,567,890,123.45)") == -1234567890123.45
    assert parse_number("32.000.000.000 đ") == 32_000_000_000
