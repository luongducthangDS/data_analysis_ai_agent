from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
import re

import pandas as pd

from backend.app.services.multi_sheet_analyzer import MultiSheetAnalyzer, SheetRelationship
from backend.app.database import init_db, db_session, SessionModel, ChatHistoryModel


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
    sheets: dict[str, pd.DataFrame] = field(default_factory=dict)
    sheet_relationships: list[SheetRelationship] = field(default_factory=list)
    sheets_context: str = ""


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, DatasetSession] = {}
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        init_db()

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
        self._insert_session(session)
        return session

    def get(self, session_id: str) -> DatasetSession:
        if session_id in self._sessions:
            return self._sessions[session_id]
        return self._restore_from_db(session_id)

    def save(self, session: DatasetSession) -> None:
        """Persist profile, report_id, and history to DB. Call after any mutation."""
        self._sessions[session.session_id] = session
        try:
            with db_session() as db:
                row = db.get(SessionModel, session.session_id)
                if row is None:
                    return
                row.profile = session.profile
                row.report_id = session.report_id
                row.updated_at = datetime.utcnow()
                db.query(ChatHistoryModel).filter_by(session_id=session.session_id).delete()
                for turn in session.history:
                    db.add(ChatHistoryModel(
                        session_id=session.session_id,
                        role=turn.get("role", ""),
                        content=turn.get("content", ""),
                    ))
        except Exception:
            pass  # Don't crash the API if DB is temporarily unavailable

    def count(self) -> int:
        try:
            with db_session() as db:
                return db.query(SessionModel).count()
        except Exception:
            return len(self._sessions)

    # ── Private ──────────────────────────────────────────────────────────────

    def _insert_session(self, session: DatasetSession) -> None:
        try:
            with db_session() as db:
                db.add(SessionModel(
                    session_id=session.session_id,
                    filename=session.filename,
                    file_names=session.file_names,
                    profile=session.profile,
                    report_id=session.report_id,
                    sheet_relationships=self._serialize_relationships(session.sheet_relationships),
                    sheets_context=session.sheets_context,
                ))
        except Exception:
            pass

    def _restore_from_db(self, session_id: str) -> DatasetSession:
        """Load a session from DB and reload its DataFrames from disk."""
        with db_session() as db:
            row = db.get(SessionModel, session_id)
            if row is None:
                raise KeyError(f"Unknown session_id: {session_id}")

            file_names: list[str] = row.file_names or []
            all_sheets: dict[str, pd.DataFrame] = {}

            for file_name in file_names:
                safe_name = Path(file_name).name.replace(" ", "_")
                file_path = UPLOAD_DIR / f"{session_id}_{safe_name}"
                if not file_path.exists():
                    raise KeyError(f"Session file missing from disk: {file_path}")

                suffix = file_path.suffix.lower()
                if suffix == ".csv":
                    df = self._coerce_datetime_columns(pd.read_csv(file_path))
                    all_sheets[Path(file_name).stem] = df
                elif suffix in {".xlsx", ".xls"}:
                    for sheet_name, df in MultiSheetAnalyzer.read_all_sheets(str(file_path)).items():
                        all_sheets[f"{Path(file_name).stem}::{sheet_name}"] = self._coerce_datetime_columns(df)

            if not all_sheets:
                raise KeyError(f"No files could be loaded for session: {session_id}")

            analysis_df = self._build_analysis_dataframe(all_sheets)
            history = [{"role": h.role, "content": h.content} for h in row.history]
            file_path_first = (
                UPLOAD_DIR / f"{session_id}_{Path(file_names[0]).name.replace(' ', '_')}"
                if file_names else None
            )

            session = DatasetSession(
                session_id=session_id,
                filename=row.filename,
                file_path=file_path_first,
                dataframe=analysis_df,
                profile=row.profile or {},
                report_id=row.report_id,
                history=history,
                file_names=file_names,
                sheets=all_sheets,
                sheet_relationships=self._deserialize_relationships(row.sheet_relationships),
                sheets_context=row.sheets_context or "",
            )
            self._sessions[session_id] = session
            return session

    @staticmethod
    def _serialize_relationships(rels: list[SheetRelationship]) -> list[dict]:
        return [
            {
                "sheet1": r.sheet1,
                "sheet2": r.sheet2,
                "join_key": r.join_key,
                "relationship_type": r.relationship_type,
                "similarity_score": r.similarity_score,
            }
            for r in rels
        ]

    @staticmethod
    def _deserialize_relationships(data: list[dict] | None) -> list[SheetRelationship]:
        return [SheetRelationship(**d) for d in (data or [])]

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
