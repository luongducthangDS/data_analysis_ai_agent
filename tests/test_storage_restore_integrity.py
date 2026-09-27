"""T01: nạp lại phiên từ DB phải ra ĐÚNG dữ liệu lúc upload.

Render free ngủ khi nhàn rỗi → cache RAM mất → `_restore_from_db` là đường chạy
thường ngày trên prod, không phải ca hiếm. Trước đây 2 file trùng tên (2 sàn đặt
tên export giống nhau) hay "my file.csv"/"my_file.csv" làm bản restore đổi số âm thầm.
"""
import pytest
from pandas.testing import assert_frame_equal

from backend.app.services.storage import UPLOAD_DIR, session_store

A = b"sku,amount\nA,100\nB,200\n"
B = b"sku,amount\nC,999\n"


def _restore(session_id: str, wipe_disk: bool):
    session_store._sessions.pop(session_id, None)
    if wipe_disk:
        for f in UPLOAD_DIR.glob(f"{session_id}_*"):
            f.unlink()
    return session_store.get(session_id)


@pytest.mark.parametrize("wipe_disk", [False, True], ids=["disk", "db-only"])
@pytest.mark.parametrize("uploads", [
    [("orders.csv", A), ("orders.csv", B)],      # 2 sàn cùng tên file export
    [("my file.csv", A), ("my_file.csv", B)],    # chỉ khác dấu cách → cùng tên trên đĩa
    [("orders.csv", A), ("orders.CSV", B)],      # đĩa Windows không phân biệt hoa thường
], ids=["same-name", "space-vs-underscore", "case"])
def test_restore_equals_create(uploads, wipe_disk):
    fresh = session_store.create_multiple(uploads)
    expected = {k: v.copy() for k, v in fresh.sheets.items()}
    expected_df = fresh.dataframe.copy()

    restored = _restore(fresh.session_id, wipe_disk)

    assert list(restored.sheets) == list(expected)
    for key, df in expected.items():
        assert_frame_equal(restored.sheets[key], df)
    assert_frame_equal(restored.dataframe, expected_df)


def test_restore_keeps_deduplicated_active_sheet():
    s = session_store.create_multiple([("orders.csv", A), ("orders.csv", B)])
    session_store.set_active_sheet(s, "orders_1")

    restored = _restore(s.session_id, wipe_disk=True)

    assert restored.active_sheet == "orders_1"
    assert restored.dataframe["amount"].tolist() == [999]
