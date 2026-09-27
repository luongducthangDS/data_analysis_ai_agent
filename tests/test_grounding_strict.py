"""T02/T03: grounding không được để lọt số sai trông như đúng (wrong_looks_right).

Trước đây: dấu bị bỏ (abs) → "lãi TĂNG" khi Δ âm vẫn qua; phần trăm bị bỏ qua;
"99 triệu" là 99 < 1000 nên không kiểm; số 1900–2100 luôn bị coi là năm.
"""
import pandas as pd
import pytest

from backend.app.agents.nodes.synthesize import _is_valid_synthesis, _numbers_grounded

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


def test_difference_between_two_values_is_derivable():
    # eval_seller #25: "Shopee hơn TikTok khoảng 690 triệu" = 1.632.490.000 − 942.487.000, đúng số học.
    assert _numbers_grounded("Shopee hơn TikTok Shop 28 triệu doanh thu.", CHANNELS) is True
    assert _numbers_grounded("Shopee hơn TikTok Shop 30 triệu doanh thu.", CHANNELS) is False
    assert _numbers_grounded("Lợi nhuận tháng 5 giảm 10 triệu so với tháng 4.", MONTHS) is True
    # Có chiều thì chỉ so theo thứ tự bảng (kỳ trước → kỳ sau): 100 → 90 là GIẢM.
    assert _numbers_grounded("Lợi nhuận tháng 5 tăng 10 triệu so với tháng 4.", MONTHS) is False
    assert _numbers_grounded("Lợi nhuận tháng 5 tăng 11,1% so với tháng 4.", MONTHS) is False
    assert _numbers_grounded("Tháng 4 lợi nhuận cao hơn tháng 5 11,1%.", MONTHS) is True   # không nói chiều lãi


def test_percentage_point_gap_between_ratios_is_derivable():
    # eval_seller #6: "TikTok cao hơn 0,02%" = 34,52% − 34,50%.
    fees = pd.DataFrame({"kenh": ["TikTok Shop", "Shopee"], "ty_le_phi_san": [0.3452, 0.3450]})
    assert _numbers_grounded("Tỷ lệ phí sàn TikTok Shop cao hơn Shopee 0,02%.", fees) is True
    assert _numbers_grounded("Tỷ lệ phí sàn TikTok Shop cao hơn Shopee 0,5%.", fees) is False


@pytest.mark.parametrize("answer", [
    # eval_seller #23–25 khi hết quota: gemini-3.8-flash trả câu bị cắt ở max_tokens.
    "Tổng doanh thu thuần của shop ghi nhận cao nhất ở kênh Shopee với 1.63",
    "Tổng lãi trước quảng cáo theo từng tháng của anh/chị ghi nhận như sau:\n\n-",
    "Hiện tại không tìm thấy dữ liệu phù hợp để xác định SKU nào",
])
def test_truncated_llm_answer_is_invalid(answer):
    assert _is_valid_synthesis(answer) is False


def test_complete_answer_is_valid():
    assert _is_valid_synthesis("Doanh thu thuần kênh Shopee là 1.632.490.000 đ, TikTok Shop là 942.487.000 đ.") is True


def test_numbers_quoted_from_the_question_are_allowed():
    answer = "Có 2 SKU có tỷ lệ hoàn trên 15%: LAN-127 và LAN-044."
    assert _numbers_grounded(answer, BRIDGE) is False
    assert _numbers_grounded(answer, BRIDGE, question="SKU nào tỷ lệ hoàn trên 15%?") is True
