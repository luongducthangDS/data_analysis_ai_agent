from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


BASE_DATA_DIR = Path("data")
UPLOAD_DIR = BASE_DATA_DIR / "uploads"
REPORT_DIR = BASE_DATA_DIR / "reports"


@dataclass
class DatasetSession:
    session_id: str
    filename: str
    file_path: Path
    dataframe: pd.DataFrame
    profile: dict[str, Any]
    report_id: str | None = None
    history: list[dict[str, str]] = field(default_factory=list)


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, DatasetSession] = {}
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        REPORT_DIR.mkdir(parents=True, exist_ok=True)

    def create(self, filename: str, content: bytes) -> DatasetSession:
        session_id = uuid.uuid4().hex
        safe_name = Path(filename).name.replace(" ", "_")
        file_path = UPLOAD_DIR / f"{session_id}_{safe_name}"
        file_path.write_bytes(content)
        df = self._read_dataframe(file_path)
        session = DatasetSession(
            session_id=session_id,
            filename=Path(filename).name,
            file_path=file_path,
            dataframe=df,
            profile={},
        )
        self._sessions[session_id] = session
        return session

    def get(self, session_id: str) -> DatasetSession:
        try:
            return self._sessions[session_id]
        except KeyError as exc:
            raise KeyError(f"Unknown session_id: {session_id}") from exc

    def count(self) -> int:
        return len(self._sessions)

    @staticmethod
    def _read_dataframe(file_path: Path) -> pd.DataFrame:
        suffix = file_path.suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(file_path)
        if suffix in {".xlsx", ".xls"}:
            return pd.read_excel(file_path)
        raise ValueError("Only CSV, XLSX, and XLS files are supported.")


session_store = SessionStore()

