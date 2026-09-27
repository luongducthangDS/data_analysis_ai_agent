"""T02/T03: grounding không được để lọt số sai trông như đúng (wrong_looks_right).

Trước đây: dấu bị bỏ (abs) → "lãi TĂNG" khi Δ âm vẫn qua; phần trăm bị bỏ qua;
"99 triệu" là 99 < 1000 nên không kiểm; số 1900–2100 luôn bị coi là năm.
"""
import pandas as pd
import pytest

from backend.app.agents.nodes.synthesize import _numbers_grounded

BRIDGE = pd.DataFrame({"hang_muc": ["Lãi trước QC", "Tỷ lệ phí sàn"],
                       "anh_huong_lai": [-11567470.0, None], "ty_le": [None, 0.335]})
CHANNELS = pd.DataFrame({"kenh": ["Shopee", "TikTok Shop"], "doanh_thu": [64_000_000, 36_000_000]})
MONTHS = pd.DataFrame({"thang": ["2026-04", "2026-05"], "loi_nhuan": [100_000_000, 90_000_000]})


@pytest.mark.parametrize("answer", [
    "Lãi tháng 5 tăng 11.567.470 đ so với tháng 4.",       # đảo chiều: Δ thật là âm
    "Lợi nhuận tăng thêm 11.567.470 đ.",
    "Lãi tháng 5 tăng 11,6 triệu.",
    "Phí sàn chiếm 45% doanh thu.",                        # tỷ lệ thật 33,5%
    "Lãi giảm 99 triệu so với tháng trước.",               # < 1000 nhưng có đơn vị
    "Lãi giảm 12,5 triệu.",                                # sai độ lớn
    "Có 2050 đơn hoàn trong kỳ.",                          # "năm" giả
])
def test_wrong_claims_are_rejected(answer):
    assert _numbers_grounded(answer, BRIDGE) is False


@pytest.mark.parametrize("answer", [
    "Lãi tháng 5 giảm 11.567.470 đ so với tháng 4.",
    "Lãi giảm khoảng 11,6 triệu so với tháng 4.",          # làm tròn 1 chữ số thập phân
    "Lãi tháng 5 giảm 11,57 triệu.",
    "Phí sàn chiếm 33,5% doanh thu.",
    "Phí sàn tăng làm lãi giảm 11.567.470 đ.",             # "tăng" thuộc về phí, không phải lãi
    "Năm 2026, lãi tháng 5 giảm 11.567.470 đ.",
    "Tháng 5/2026 lãi giảm 11.567.470 đ.",
])
def test_correct_claims_pass(answer):
    assert _numbers_grounded(answer, BRIDGE) is True


def test_share_of_total_and_change_between_rows_are_derivable():
    assert _numbers_grounded("Shopee chiếm 64% doanh thu, TikTok Shop 36%.", CHANNELS) is True
    assert _numbers_grounded("Shopee chiếm 70% doanh thu.", CHANNELS) is False
    assert _numbers_grounded("Lợi nhuận tháng 5 giảm 10% so với tháng 4.", MONTHS) is True
    assert _numbers_grounded("Lợi nhuận tháng 5 tăng 10% so với tháng 4.", MONTHS) is False


def test_numbers_quoted_from_the_question_are_allowed():
    answer = "Có 2 SKU có tỷ lệ hoàn trên 15%: LAN-127 và LAN-044."
    assert _numbers_grounded(answer, BRIDGE) is False
    assert _numbers_grounded(answer, BRIDGE, question="SKU nào tỷ lệ hoàn trên 15%?") is True
