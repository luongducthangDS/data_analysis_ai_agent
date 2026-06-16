from __future__ import annotations

import json
import os as _os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
import re

import pandas as pd

from backend.app.services.multi_sheet_analyzer import MultiSheetAnalyzer, SheetRelationship
from backend.app.database import init_db, db_session, SessionModel


# DATA_DIR env var lets Railway mount a persistent volume at a custom path.
# Falls back to "data/" (relative to CWD = /app) for local dev.
BASE_DATA_DIR = Path(_os.getenv("DATA_DIR", "data"))
UPLOAD_DIR = BASE_DATA_DIR / "uploads"
REPORT_DIR = BASE_DATA_DIR / "reports"
HISTORY_DIR = BASE_DATA_DIR / "history"

_MAX_CACHE_SIZE = 200
_CACHE_TTL_SECONDS = 24 * 3600


@dataclass
class DatasetSession:
    session_id: str
    filename: str
    file_path: Path | None
    dataframe: pd.DataFrame
    profile: dict[str, Any]
    owner_id: str = ""
    report_id: str | None = None
    history: list[dict[str, str]] = field(default_factory=list)
    file_names: list[str] = field(default_factory=list)
    sheets: dict[str, pd.DataFrame] = field(default_factory=dict)
    sheet_relationships: list[SheetRelationship] = field(default_factory=list)
    sheets_context: str = ""
    ecommerce_col_map: dict[str, str] = field(default_factory=dict)
    detected_platform: str | None = None


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, DatasetSession] = {}
        self._last_accessed: dict[str, float] = {}
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        init_db()

    def create(self, filename: str, content: bytes, owner_id: str = "") -> DatasetSession:
        return self.create_multiple([(filename, content)], owner_id=owner_id)

    def create_multiple(self, uploads: list[tuple[str, bytes]], owner_id: str = "") -> DatasetSession:
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

        # Detect e-commerce column mapping before persisting
        from backend.app.services.ecommerce_columns import detect_ecommerce_columns, detect_platform
        ecom_col_map = detect_ecommerce_columns(analysis_df)
        detected_platform = detect_platform(analysis_df, ecom_col_map, filename=uploads[0][0])

        session = DatasetSession(
            session_id=session_id,
            filename=uploads[0][0],
            file_path=file_paths[0] if file_paths else None,
            dataframe=analysis_df,
            profile={},
            owner_id=owner_id,
            file_names=file_names,
            sheets=all_sheets,
            sheet_relationships=relationships,
            sheets_context=context,
            ecommerce_col_map=ecom_col_map,
            detected_platform=detected_platform,
        )
        self._sessions[session_id] = session
        self._last_accessed[session_id] = time.time()
        self._insert_session(session)
        return session

    def get(self, session_id: str, owner_id: str = "") -> DatasetSession:
        self._evict_stale()
        if session_id in self._sessions:
            session = self._sessions[session_id]
            self._check_ownership(session, owner_id)
            self._last_accessed[session_id] = time.time()
            return session
        session = self._restore_from_db(session_id)
        self._check_ownership(session, owner_id)
        self._last_accessed[session_id] = time.time()
        return session

    def save(self, session: DatasetSession) -> None:
        """Persist profile, report_id to DB and history to JSON file."""
        self._sessions[session.session_id] = session
        self._last_accessed[session.session_id] = time.time()
        self._save_history(session.session_id, session.history)
        try:
            with db_session() as db:
                row = db.get(SessionModel, session.session_id)
                if row is None:
                    return
                row.profile = session.profile
                row.report_id = session.report_id
                row.updated_at = datetime.utcnow()
        except Exception:
            pass  # Don't crash the API if DB is temporarily unavailable

    def count(self) -> int:
        try:
            with db_session() as db:
                return db.query(SessionModel).count()
        except Exception:
            return len(self._sessions)

    def delete_session(self, session_id: str) -> None:
        """Remove session from cache, DB, and all associated files."""
        self._sessions.pop(session_id, None)
        self._last_accessed.pop(session_id, None)

        # Remove uploaded files
        for f in UPLOAD_DIR.glob(f"{session_id}_*"):
            try:
                f.unlink()
            except Exception:
                pass

        # Remove history file
        history_path = HISTORY_DIR / f"{session_id}.json"
        if history_path.exists():
            try:
                history_path.unlink()
            except Exception:
                pass

        # Remove from DB (cascades to chat_history)
        try:
            with db_session() as db:
                row = db.get(SessionModel, session_id)
                if row:
                    # Delete associated report file if any
                    if row.report_id:
                        report_path = REPORT_DIR / f"{row.report_id}.md"
                        if report_path.exists():
                            try:
                                report_path.unlink()
                            except Exception:
                                pass
                    db.delete(row)
        except Exception:
            pass

    def cleanup_old_sessions(self, max_age_days: int = 7) -> int:
        """Delete sessions older than max_age_days. Returns number of sessions deleted."""
        cutoff = datetime.utcnow() - timedelta(days=max_age_days)
        deleted = 0
        try:
            with db_session() as db:
                old_rows = (
                    db.query(SessionModel)
                    .filter(SessionModel.created_at < cutoff)
                    .all()
                )
                old_ids = [row.session_id for row in old_rows]
            for sid in old_ids:
                self.delete_session(sid)
                deleted += 1
        except Exception:
            pass
        return deleted

    # ── Private ──────────────────────────────────────────────────────────────

    @staticmethod
    def _check_ownership(session: DatasetSession, owner_id: str) -> None:
        """Raise PermissionError if requesting user doesn't own this session."""
        if session.owner_id and owner_id and session.owner_id != owner_id:
            raise PermissionError(f"Access denied to session: {session.session_id}")

    def _evict_stale(self) -> None:
        now = time.time()
        stale = [
            sid for sid, t in self._last_accessed.items()
            if now - t > _CACHE_TTL_SECONDS
        ]
        for sid in stale:
            self._sessions.pop(sid, None)
            self._last_accessed.pop(sid, None)

        # Enforce max cache size via LRU eviction
        if len(self._sessions) > _MAX_CACHE_SIZE:
            oldest = sorted(self._last_accessed, key=lambda s: self._last_accessed[s])
            for sid in oldest[: len(self._sessions) - _MAX_CACHE_SIZE]:
                self._sessions.pop(sid, None)
                self._last_accessed.pop(sid, None)

    def _insert_session(self, session: DatasetSession) -> None:
        try:
            with db_session() as db:
                db.add(SessionModel(
                    session_id=session.session_id,
                    owner_id=session.owner_id,
                    filename=session.filename,
                    file_names=session.file_names,
                    profile=session.profile,
                    report_id=session.report_id,
                    sheet_relationships=self._serialize_relationships(session.sheet_relationships),
                    sheets_context=session.sheets_context,
                    ecommerce_col_map=session.ecommerce_col_map or None,
                    detected_platform=session.detected_platform,
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
            history = self._load_history(session_id)
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
                owner_id=row.owner_id or "",
                report_id=row.report_id,
                history=history,
                file_names=file_names,
                sheets=all_sheets,
                sheet_relationships=self._deserialize_relationships(row.sheet_relationships),
                sheets_context=row.sheets_context or "",
                ecommerce_col_map=row.ecommerce_col_map or {},
                detected_platform=row.detected_platform,
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
    def _save_history(session_id: str, history: list[dict[str, str]]) -> None:
        path = HISTORY_DIR / f"{session_id}.json"
        try:
            path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    @staticmethod
    def _load_history(session_id: str) -> list[dict[str, str]]:
        path = HISTORY_DIR / f"{session_id}.json"
        if not path.exists():
            return []
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return []

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
