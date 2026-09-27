"""T02: join nhiều-nhiều và upload nhiều file không được làm tràn RAM chỉ bằng 1 request."""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest

from backend.app.services import storage
from backend.app.services.storage import DatasetSession, JoinTooLarge, build_source_frame, check_join_size
from tests.conftest import make_csv_bytes

N = 2_000   # 2 000 × 2 000 cùng một khoá = 4 000 000 dòng > 2 000 000


def _same_key_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    return (pd.DataFrame({"k": ["x"] * N, "a": range(N)}),
            pd.DataFrame({"k": ["x"] * N, "b": range(N)}))


def test_merge_sheets_cartesian_join_is_rejected(client):
    a, b = _same_key_frames()
    buf = io.BytesIO()
    with pd.ExcelWriter(buf) as writer:
        a.to_excel(writer, sheet_name="A", index=False)
        b.to_excel(writer, sheet_name="B", index=False)
    up = client.post("/api/upload", files=[("files", ("two.xlsx", buf.getvalue(), "application/octet-stream"))])
    assert up.status_code == 200

    resp = client.post("/api/merge-sheets",
                       json={"session_id": up.json()["session_id"], "sheet_names": ["A", "B"], "join_key": "k"})
    assert resp.status_code == 400
    assert "giới hạn an toàn" in resp.json()["detail"]


def test_build_source_frame_raises_instead_of_silent_fallback():
    a, b = _same_key_frames()
    session = DatasetSession(session_id="t", filename="t.xlsx", file_path=None, dataframe=a, profile={},
                             sheets={"A": a, "B": b})
    with pytest.raises(JoinTooLarge):
        build_source_frame(session, {"join": {"base": "A", "with": "B", "on": "k"}})


def test_estimate_equals_real_left_join_rows(monkeypatch):
    # nhiều-nhiều + khoá không khớp + NaN (pandas ghép NaN với NaN) — ước lượng phải đúng bằng len(merge).
    left = pd.DataFrame({"k": ["x", "x", "y", "z", np.nan, np.nan]})
    right = pd.DataFrame({"k": ["x", "x", "x", "y", "y", np.nan, "w"]})
    real = len(left.merge(right, on="k", how="left"))
    monkeypatch.setattr(storage, "MAX_JOIN_ROWS", real)
    check_join_size(left, right, "k")                      # đúng bằng trần → cho qua
    monkeypatch.setattr(storage, "MAX_JOIN_ROWS", real - 1)
    with pytest.raises(JoinTooLarge):
        check_join_size(left, right, "k")


def test_upload_more_than_max_files_is_rejected(client):
    files = [("files", (f"f{i}.csv", make_csv_bytes(), "text/csv")) for i in range(6)]
    resp = client.post("/api/upload", files=files)
    assert resp.status_code == 400
