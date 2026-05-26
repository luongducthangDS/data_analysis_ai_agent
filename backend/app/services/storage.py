from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import re

import pandas as pd

from backend.app.services.multi_sheet_analyzer import MultiSheetAnalyzer, SheetRelationship


# DATA_DIR env var lets Railway mount a persistent volume at a custom path.
# Falls back to "data/" (relative to CWD = /app) for local dev.
import os as _os
BASE_DATA_DIR = Path(_os.getenv("DATA_DIR", "data"))
UPLOAD_DIR = BASE_DATA_DIR / "uploads"
REPORT_DIR = BASE_DATA_DIR / "reports"


@dataclass
class DatasetSession:
    session_id: str
    filename: str
    file_path: Path | None
    dataframe: pd.DataFrame
    profile: dict[str, Any]
    report_id: str | None = None
    history: list[dict[str, str]] = field(default_factory=list)
    file_names: list[str] = field(default_factory=list)
    sheets: dict[str, pd.DataFrame] = field(default_factory=dict)  # All sheets for Excel files and CSV files
    sheet_relationships: list[SheetRelationship] = field(default_factory=list)
    sheets_context: str = ""  # Text description of sheet structure


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, DatasetSession] = {}
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        REPORT_DIR.mkdir(parents=True, exist_ok=True)

    def create(self, filename: str, content: bytes) -> DatasetSession:
        return self.create_multiple([(filename, content)])

    def create_multiple(self, uploads: list[tuple[str, bytes]]) -> DatasetSession:
        session_id = uuid.uuid4().hex
        file_paths: list[Path] = []
        all_sheets: dict[str, pd.DataFrame] = {}
        file_names: list[str] = []

        for filename, content in uploads:
            safe_name = Path(filename).name.replace(" ", "_")
            file_path = UPLOAD_DIR / f"{session_id}_{safe_name}"
            file_path.write_bytes(content)
            file_paths.append(file_path)
            file_names.append(Path(filename).name)

            suffix = file_path.suffix.lower()
            if suffix == ".csv":
                df = pd.read_csv(file_path)
                df = self._coerce_datetime_columns(df)
                sheet_key = Path(filename).stem
                if sheet_key in all_sheets:
                    suffix_index = 1
                    while f"{sheet_key}_{suffix_index}" in all_sheets:
                        suffix_index += 1
                    sheet_key = f"{sheet_key}_{suffix_index}"
                all_sheets[sheet_key] = df
            elif suffix in {".xlsx", ".xls"}:
                file_sheets = MultiSheetAnalyzer.read_all_sheets(str(file_path))
                for sheet_name, df in file_sheets.items():
                    df = self._coerce_datetime_columns(df)
                    sheet_key = f"{Path(filename).stem}::{sheet_name}"
                    if sheet_key in all_sheets:
                        suffix_index = 1
                        while f"{sheet_key}_{suffix_index}" in all_sheets:
                            suffix_index += 1
                        sheet_key = f"{sheet_key}_{suffix_index}"
                    all_sheets[sheet_key] = df
            else:
                raise ValueError("Only CSV, XLSX, and XLS files are supported.")

        if not all_sheets:
            raise ValueError("No valid sheets found in upload.")

        analysis_df = self._build_analysis_dataframe(all_sheets)
        relationships: list[SheetRelationship] = []
        context = ""
        if len(all_sheets) > 1:
            relationships = MultiSheetAnalyzer.detect_relationships(all_sheets)
            context = MultiSheetAnalyzer.generate_sheets_context(all_sheets, relationships)

        session = DatasetSession(
            session_id=session_id,
            filename=uploads[0][0],
            file_path=file_paths[0] if file_paths else None,
            dataframe=analysis_df,
            profile={},
            file_names=file_names,
            sheets=all_sheets,
            sheet_relationships=relationships,
            sheets_context=context,
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
    def _build_analysis_dataframe(sheets: dict[str, pd.DataFrame]) -> pd.DataFrame:
        if len(sheets) == 1:
            return next(iter(sheets.values()))

        column_sets = {tuple(df.columns.tolist()) for df in sheets.values()}
        if len(column_sets) == 1:
            frames = []
            for sheet_name, df in sheets.items():
                frame = df.copy()
                frame["_source_sheet"] = sheet_name
                frames.append(frame)
            return pd.concat(frames, ignore_index=True)

        best_name, best_df = max(
            sheets.items(),
            key=lambda item: SessionStore._analysis_score(item[1]),
        )
        selected = best_df.copy()
        selected.attrs["source_sheet"] = best_name
        return selected

    @staticmethod
    def _analysis_score(df: pd.DataFrame) -> float:
        numeric_count = len(df.select_dtypes(include="number").columns)
        categorical_count = len(df.select_dtypes(include=["object", "category", "bool"]).columns)
        rows = len(df)
        missing_ratio = float(df.isna().mean().mean()) if rows and len(df.columns) else 1.0
        unnamed_ratio = (
            sum(str(col).lower().startswith("unnamed") for col in df.columns) / len(df.columns)
            if len(df.columns)
            else 1.0
        )
        no_metric_penalty = 25 if numeric_count == 0 else 0
        return (
            numeric_count * 12
            + min(categorical_count, 8) * 1.5
            + min(rows, 10000) / 1000
            - missing_ratio * 10
            - unnamed_ratio * 15
            - no_metric_penalty
        )

    @staticmethod
    def _read_dataframe(file_path: Path) -> pd.DataFrame:
        suffix = file_path.suffix.lower()
        if suffix == ".csv":
            return SessionStore._coerce_datetime_columns(pd.read_csv(file_path))
        if suffix in {".xlsx", ".xls"}:
            return SessionStore._coerce_datetime_columns(pd.read_excel(file_path))
        raise ValueError("Only CSV, XLSX, and XLS files are supported.")

    @staticmethod
    def _coerce_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        date_name_pattern = re.compile(r"(date|time|ngay|thang|nam)", re.IGNORECASE)
        for col in result.select_dtypes(include=["object", "string"]).columns:
            series = result[col]
            non_null = series.dropna()
            if non_null.empty:
                continue
            should_try = bool(date_name_pattern.search(str(col)))
            if not should_try:
                sample = non_null.astype(str).head(20)
                looks_like_date = sample.str.contains(
                    r"\d{4}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}",
                    regex=True,
                ).mean() >= 0.85
                if not looks_like_date:
                    continue
                parsed_sample = pd.to_datetime(sample, errors="coerce", format="mixed")
                should_try = parsed_sample.notna().mean() >= 0.85
            if not should_try:
                continue
            parsed = pd.to_datetime(series, errors="coerce", format="mixed")
            if parsed.notna().sum() / max(len(non_null), 1) >= 0.85:
                result[col] = parsed
        return result


session_store = SessionStore()
