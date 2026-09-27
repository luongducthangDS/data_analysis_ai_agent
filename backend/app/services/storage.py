from __future__ import annotations

import io
import logging
import os
import time
import unicodedata
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
import re

import pandas as pd

from backend.app.services.numeric_parse import (
    coerce_numeric_columns,
    strip_column_names,
)
from backend.app.services.multi_sheet_analyzer import MultiSheetAnalyzer, SheetRelationship
from backend.app.core.config import get_settings
from backend.app.database import db_session, ChatHistoryModel, ReportModel, SessionFileModel, SessionModel


# DATA_DIR lets Railway mount a persistent volume at a custom path.
# Falls back to "data/" (relative to CWD = /app) for local dev. Disk is only a
# cache: uploads, history and reports are also stored in DATABASE_URL, so an
# ephemeral disk (Render free) can be rebuilt from Postgres/Supabase.
BASE_DATA_DIR = Path(get_settings().data_dir)
UPLOAD_DIR = BASE_DATA_DIR / "uploads"
REPORT_DIR = BASE_DATA_DIR / "reports"

# In-memory DataFrame LRU; evicted sessions reload from disk/DB on next access
# (merged sheets are not kept).
_MAX_CACHE_SIZE = get_settings().session_cache_size
# RAM is the real limit (Render free = 512 MB), not the number of sessions: one
# 10 MB CSV of text columns can take 100+ MB as a DataFrame.
_MAX_CACHE_BYTES = get_settings().session_cache_mb * 1024 * 1024
_CACHE_TTL_SECONDS = 24 * 3600

_log = logging.getLogger(__name__)

# Sample datasets shipped with the app (data/ is in the Docker image). A sample
# session stores only the set's name; restore reads the files from here.
SAMPLE_SETS = {"shop_lan": Path(__file__).resolve().parents[3] / "data" / "samples" / "shop_lan"}
SAMPLE_FILES = {"shop_lan": ("export_shopee.csv", "export_tiktok.csv", "products.csv", "ads_daily.csv")}


def sample_uploads(sample_set: str, names: list[str] | tuple[str, ...] | None = None) -> list[tuple[str, bytes]]:
    """(name, bytes) of a sample set's files. Unknown set → KeyError (whitelist, no path from input)."""
    folder = SAMPLE_SETS[sample_set]
    return [(name, (folder / Path(name).name).read_bytes()) for name in (names or SAMPLE_FILES[sample_set])]


def resolve_sheet_key(name: str, sheets: dict[str, pd.DataFrame]) -> str:
    """Resolve a user-supplied sheet name to a real key (exact, or by sheet part of 'file::sheet')."""
    if name in sheets:
        return name
    matches = [key for key in sheets if key.split("::", 1)[-1] == name]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous sheet name '{name}'. Use one of: {', '.join(matches)}")
    raise ValueError(f"Unknown sheet name '{name}'.")


def _upload_path(session_id: str, index: int, filename: str) -> Path:
    """Disk cache path of the index-th upload. The index keeps two files with the
    same name — or differing only by space/underscore or case — from overwriting each other."""
    return UPLOAD_DIR / f"{session_id}_{index}_{Path(filename).name.replace(' ', '_')}"


def _write_atomic(path: Path, content: bytes) -> None:
    """Temp file + rename: two requests restoring the same session never read a half-written file."""
    tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    tmp.write_bytes(content)
    os.replace(tmp, path)


def _remove_uploads(session_id: str) -> None:
    for f in UPLOAD_DIR.glob(f"{session_id}_*"):
        f.unlink(missing_ok=True)


def _unique_key(key: str, taken: dict) -> str:
    """"orders" → "orders_1", "orders_2"… when the key is already taken."""
    if key not in taken:
        return key
    i = 1
    while f"{key}_{i}" in taken:
        i += 1
    return f"{key}_{i}"


# Excel tiếng Việt trên Windows lưu CSV bằng cp1258; UTF-8 có BOM là "CSV UTF-8" của Excel.
_CSV_ENCODINGS = ("utf-8-sig", "cp1258")


def read_csv_bytes(content: bytes) -> pd.DataFrame:
    """Read a CSV the way sellers actually send it: UTF-8 (with/without BOM) or
    cp1258, precomposed or combining accents (Excel on Mac saves NFD), "," or ";"
    delimiter (Excel in VN/EU locales). NFC before parsing so headers match
    EXPORT_HEADERS and status values match whichever way they were typed."""
    for encoding in _CSV_ENCODINGS:
        try:
            text = content.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError("Không đọc được bảng mã của file CSV — hãy lưu lại dạng \"CSV UTF-8\".")
    text = unicodedata.normalize("NFC", text)
    header = text.split("\n", 1)[0]
    # ponytail: đoán theo dòng tiêu đề; tên cột chứa dấu ";" sẽ đoán nhầm — thêm csv.Sniffer khi gặp.
    sep = ";" if header.count(";") > header.count(",") else ","
    return pd.read_csv(io.StringIO(text), sep=sep)


MAX_JOIN_ROWS = 2_000_000


class JoinTooLarge(ValueError):
    """Join nhiều-nhiều sẽ sinh quá MAX_JOIN_ROWS dòng — chặn trước khi pandas cấp phát (T02)."""


def check_join_size(left: pd.DataFrame, right: pd.DataFrame, on: str) -> None:
    """Raise JoinTooLarge nếu left join trên `on` sinh quá MAX_JOIN_ROWS dòng.
    Σ count_left[k] · max(count_right[k], 1) đúng bằng số dòng left join (inner join ≤ số đó);
    dropna=False vì pandas ghép NaN với NaN."""
    if len(left) * len(right) <= MAX_JOIN_ROWS:
        return
    left_counts = left[on].value_counts(dropna=False)
    right_counts = right[on].value_counts(dropna=False).reindex(left_counts.index, fill_value=0).clip(lower=1)
    rows = int((left_counts * right_counts).sum())
    if rows > MAX_JOIN_ROWS:
        raise JoinTooLarge(f"Dữ liệu gộp vượt quá giới hạn an toàn ({rows:,} dòng > {MAX_JOIN_ROWS:,}).")


def merge_sheet_frames(
    sheets: dict[str, pd.DataFrame], keys: list[str], on: list[str | None]
) -> tuple[pd.DataFrame, list[str]]:
    """Left-join sheets[keys[0]] with each following sheet on on[i]; None = first
    common column in sorted order (a set's order changes between processes, and a
    restart must pick the same column). Returns the frame and the columns used,
    which is what the recipe stores."""
    merged = sheets[keys[0]].copy()
    used: list[str] = []
    for key, col in zip(keys[1:], on):
        right = sheets[key]
        if not col:
            common = sorted(set(merged.columns) & set(right.columns))
            if not common:
                raise ValueError("No common columns found between sheets to merge on.")
            col = common[0]
        check_join_size(merged, right, col)
        merged = merged.merge(right, on=col, how="left", suffixes=("", "_dup"))
        used.append(col)
    return merged, used


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
    active_sheet: str | None = None  # which sheet drives `dataframe`; None/"__concat__" = auto
    merge_recipes: list[dict[str, Any]] = field(default_factory=list)  # persisted as merged_sheets
    # The persisted active sheet (a merge) could not be rebuilt: `dataframe` is only a
    # placeholder, so chat/dashboard must refuse until the user picks or re-merges a sheet.
    active_sheet_lost: bool = False


class SessionStore:
    """DB is the source of truth; `_sessions` is a per-process DataFrame cache.

    Multi-worker notes: chat history is append-only rows (no lost updates) and
    is read back from the DB for each question. Other fields (profile,
    active_sheet) are last-writer-wins, and a worker keeps its cached copy until
    eviction — so a sheet switch in worker A shows up in worker B only after B's
    cache entry expires. Fine for 1 worker (current deploy); add a cache
    version check if you scale out.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, DatasetSession] = {}
        self._last_accessed: dict[str, float] = {}
        self._nbytes: dict[str, int] = {}
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        REPORT_DIR.mkdir(parents=True, exist_ok=True)

    def create(self, filename: str, content: bytes, owner_id: str = "") -> DatasetSession:
        return self.create_multiple([(filename, content)], owner_id=owner_id)

    def create_multiple(
        self, uploads: list[tuple[str, bytes]], owner_id: str = "", sample_set: str | None = None
    ) -> DatasetSession:
        session_id = uuid.uuid4().hex
        try:
            all_sheets, file_paths = self._build_sheets(session_id, uploads)
        except Exception:
            _remove_uploads(session_id)
            raise
        file_names = [Path(filename).name for filename, _ in uploads]
        analysis_df, active_sheet = self._resolve_active_dataframe(all_sheets)
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
            active_sheet=active_sheet,
        )
        try:
            self._insert_session(session, uploads, sample_set)
        except Exception:
            # Not durable = not created: a RAM-only session would vanish on restart.
            _remove_uploads(session_id)
            raise
        self._remember(session)
        return session

    def _build_sheets(
        self, session_id: str, uploads: list[tuple[str, bytes]]
    ) -> tuple[dict[str, pd.DataFrame], list[Path]]:
        """Uploaded files → sheets. create and restore BOTH go through here, so a
        restored session has the same sheet keys, order and data as at upload
        (restore used to re-derive keys on its own and lost duplicate file names)."""
        sheets: dict[str, pd.DataFrame] = {}
        paths: list[Path] = []
        for index, (filename, content) in enumerate(uploads):
            path = _upload_path(session_id, index, filename)
            if not path.exists():
                _write_atomic(path, content)
            paths.append(path)
            stem, suffix = Path(filename).stem, Path(filename).suffix.lower()
            if suffix == ".csv":
                frames = {stem: read_csv_bytes(content)}
            elif suffix in {".xlsx", ".xls"}:
                frames = {f"{stem}::{name}": df
                          for name, df in MultiSheetAnalyzer.read_all_sheets(str(path)).items()}
            else:
                raise ValueError("Only CSV, XLSX, and XLS files are supported.")
            for key, df in frames.items():
                sheets[_unique_key(key, sheets)] = self._normalize_frame(df)
        if not sheets:
            raise ValueError("No valid sheets found in upload.")

        from backend.app.services.ecommerce_semantic import link_sheets
        return link_sheets(sheets), paths

    def set_active_sheet(self, session: DatasetSession, sheet_key: str) -> DatasetSession:
        """
        Switch which sheet drives analysis. Resolves the key, rebuilds dataframe +
        profile + e-commerce mapping, invalidates the dashboard cache, and persists.
        """
        from backend.app.services.ecommerce_columns import detect_ecommerce_columns, detect_platform
        from backend.app.services.profiler import build_profile

        key = resolve_sheet_key(sheet_key, session.sheets)
        session.dataframe = session.sheets[key].copy()
        session.active_sheet = key
        session.active_sheet_lost = False
        session.profile = build_profile(session.dataframe)
        session.ecommerce_col_map = detect_ecommerce_columns(session.dataframe)
        session.detected_platform = detect_platform(
            session.dataframe, session.ecommerce_col_map, filename=session.filename
        )
        # Invalidate cached dashboard so it recomputes for the new sheet.
        for attr in ("_dashboard_cache", "_dashboard_charts_raw"):
            session.__dict__.pop(attr, None)
        self._nbytes.pop(session.session_id, None)   # new frame (or merged sheet) → recount
        self.save(session)
        return session

    def get(self, session_id: str, owner_id: str = "") -> DatasetSession:
        self._evict_stale(protect=session_id)
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
        """Persist profile, report_id, active sheet and e-commerce mapping.
        History is NOT written here — see append_messages."""
        self._remember(session)
        with db_session() as db:
            row = db.get(SessionModel, session.session_id)
            if row is None:
                raise KeyError(f"Unknown session_id: {session.session_id}")
            row.profile = session.profile
            row.report_id = session.report_id
            row.active_sheet = session.active_sheet
            row.merged_sheets = session.merge_recipes or None
            row.ecommerce_col_map = session.ecommerce_col_map or None
            row.detected_platform = session.detected_platform
            row.updated_at = datetime.utcnow()

    def append_messages(self, session: DatasetSession, messages: list[dict[str, str]]) -> None:
        """Append chat messages as new rows — never rewrites earlier ones, so two
        workers answering the same session can't overwrite each other."""
        session.history.extend(messages)
        with db_session() as db:
            db.add_all(
                ChatHistoryModel(session_id=session.session_id, role=m["role"],
                                 content=m["content"], source=m.get("source"))
                for m in messages
            )

    def recent_history(self, session_id: str, n: int = 6) -> list[dict[str, str]]:
        """Last n messages from the DB (not the RAM copy, which another worker
        may have made stale). ponytail: ignores the legacy chat_log blob — those
        sessions expire within SESSION_TTL_DAYS."""
        with db_session() as db:
            rows = (
                db.query(ChatHistoryModel)
                .filter_by(session_id=session_id)
                .order_by(ChatHistoryModel.id.desc())
                .limit(n)
                .all()
            )
            return [self._message_dict(r) for r in reversed(rows)]

    @staticmethod
    def _message_dict(row: ChatHistoryModel) -> dict[str, str]:
        msg = {"role": row.role, "content": row.content}
        if row.source:
            msg["source"] = row.source
        return msg

    def count(self) -> int:
        try:
            with db_session() as db:
                return db.query(SessionModel).count()
        except Exception:
            return len(self._sessions)

    def delete_session(self, session_id: str) -> None:
        """Remove session from cache, DB, and all associated files."""
        self._forget(session_id)

        # DB first (cascades to chat_history / session_files) — a failure here
        # must reach the caller, not report 204 while the data is still there.
        report_id = None
        with db_session() as db:
            row = db.get(SessionModel, session_id)
            if row:
                report_id = row.report_id
                if report_id and (report := db.get(ReportModel, report_id)):
                    db.delete(report)
                # Every turn writes a report; before session_id existed only the
                # last one was deleted and the rest stayed in the DB forever.
                db.query(ReportModel).filter(ReportModel.session_id == session_id).delete()
                db.delete(row)

        # Disk is only a cache; a leftover file is harmless, so just log it.
        files = list(UPLOAD_DIR.glob(f"{session_id}_*"))
        if report_id:
            files.append(REPORT_DIR / f"{report_id}.md")
        for f in files:
            try:
                f.unlink(missing_ok=True)
            except OSError:
                _log.warning("delete_session: could not remove %s", f, exc_info=True)

    def cleanup_old_sessions(self, max_age_days: int = 7) -> int:
        """Delete sessions older than max_age_days. Returns number of sessions deleted."""
        cutoff = datetime.utcnow() - timedelta(days=max_age_days)
        with db_session() as db:
            old_ids = [
                sid for (sid,) in
                db.query(SessionModel.session_id).filter(SessionModel.created_at < cutoff)
            ]
        deleted = 0
        for sid in old_ids:
            try:
                self.delete_session(sid)
                deleted += 1
            except Exception:
                _log.exception("cleanup: failed to delete session %s", sid)  # retried next run
        return deleted

    # ── Private ──────────────────────────────────────────────────────────────

    @staticmethod
    def _check_ownership(session: DatasetSession, owner_id: str) -> None:
        """Raise PermissionError if requesting user doesn't own this session."""
        if session.owner_id and owner_id and session.owner_id != owner_id:
            raise PermissionError(f"Access denied to session: {session.session_id}")

    def _remember(self, session: DatasetSession) -> None:
        """Put a session in the RAM cache (counting its bytes once) and evict others to fit."""
        sid = session.session_id
        self._sessions[sid] = session
        self._last_accessed[sid] = time.time()
        if sid not in self._nbytes:
            self._nbytes[sid] = self._session_nbytes(session)
        self._evict_stale(protect=sid)

    def _forget(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
        self._last_accessed.pop(session_id, None)
        self._nbytes.pop(session_id, None)

    @staticmethod
    def _session_nbytes(session: DatasetSession) -> int:
        """Deep size of every distinct frame the session holds; `dataframe` is often
        one of `sheets`, so count by identity."""
        frames = {id(df): df for df in [*session.sheets.values(), session.dataframe]}
        return int(sum(df.memory_usage(deep=True).sum() for df in frames.values()))

    def _evict_stale(self, protect: str | None = None) -> None:
        """Drop expired sessions, then least-recently-used ones until both the count
        and the byte budget fit. `protect` (the session in use) is never dropped, so
        a session bigger than the whole budget still works — it just stays alone.
        ponytail: a request still holding an evicted session keeps its RAM until it
        finishes, so the budget is soft; lower SESSION_CACHE_MB if the box still OOMs."""
        now = time.time()
        # Snapshot first: sync routes + the cleanup thread touch this dict
        # concurrently, and iterating it live raises "changed size during iteration".
        accessed = list(self._last_accessed.items())
        for sid, t in accessed:
            if now - t > _CACHE_TTL_SECONDS and sid != protect:
                self._forget(sid)

        lru = [sid for sid, _ in sorted(accessed, key=lambda kv: kv[1])
               if sid in self._sessions and sid != protect]
        total = sum(self._nbytes.get(sid, 0) for sid in list(self._sessions))
        for sid in lru:
            if len(self._sessions) <= _MAX_CACHE_SIZE and total <= _MAX_CACHE_BYTES:
                break
            total -= self._nbytes.get(sid, 0)
            self._forget(sid)

    def _insert_session(
        self, session: DatasetSession, uploads: list[tuple[str, bytes]], sample_set: str | None = None
    ) -> None:
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
                active_sheet=session.active_sheet,
                sample_set=sample_set,
                files=[] if sample_set else [
                    SessionFileModel(name=Path(filename).name, content=content)
                    for filename, content in uploads
                ],
            ))

    def _restore_from_db(self, session_id: str) -> DatasetSession:
        """Rebuild a session from the file bytes stored in the DB (in upload order,
        duplicate names kept) through the same _build_sheets as create.
        ponytail: sessions older than session_files (2026-09-24) are not restorable;
        they expire within SESSION_TTL_DAYS.

        Everything is copied out of the ORM row first: the pooled connection goes
        back before pandas parses (it used to be held for the whole parse)."""
        with db_session() as db:
            row = db.get(SessionModel, session_id)
            if row is None:
                raise KeyError(f"Unknown session_id: {session_id}")
            sample_set = row.sample_set
            blobs = [] if sample_set else [(f.name, f.content) for f in row.files]
            persisted_active = row.active_sheet
            recipes = list(row.merged_sheets or [])
            # Legacy blob (pre append-only) first, then the per-message rows.
            history = list(row.chat_log or []) + [self._message_dict(h) for h in row.history]
            meta = dict(
                filename=row.filename,
                profile=row.profile or {},
                owner_id=row.owner_id or "",
                report_id=row.report_id,
                file_names=list(row.file_names or []),
                sheet_relationships=self._deserialize_relationships(row.sheet_relationships),
                sheets_context=row.sheets_context or "",
                ecommerce_col_map=row.ecommerce_col_map or {},
                detected_platform=row.detected_platform,
            )

        uploads = sample_uploads(sample_set, meta["file_names"]) if sample_set else blobs
        if not uploads:
            raise KeyError(f"No stored files for session: {session_id}")
        all_sheets, file_paths = self._build_sheets(session_id, uploads)
        for recipe in recipes:
            try:
                all_sheets[recipe["name"]], _ = merge_sheet_frames(all_sheets, recipe["sources"], recipe["on"])
            except (KeyError, ValueError):   # JoinTooLarge is a ValueError
                _log.warning("restore: merged sheet %r not rebuilt session=%s",
                             recipe.get("name"), session_id, exc_info=True)
        analysis_df, active_sheet = self._resolve_active_dataframe(all_sheets, active_sheet=persisted_active)
        # A persisted sheet that is gone (merge from before recipes, or a recipe that
        # no longer replays) must not quietly become another sheet: keep its name so
        # save() does not overwrite it, and flag it so answers refuse.
        lost = bool(persisted_active) and persisted_active != "__concat__" and persisted_active not in all_sheets
        session = DatasetSession(
            session_id=session_id,
            file_path=file_paths[0],
            dataframe=analysis_df,
            history=history,
            sheets=all_sheets,
            active_sheet=persisted_active if lost else active_sheet,
            merge_recipes=recipes,
            active_sheet_lost=lost,
            **meta,
        )
        self._remember(session)
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
        df, _ = SessionStore._resolve_active_dataframe(sheets)
        return df

    @staticmethod
    def _resolve_active_dataframe(
        sheets: dict[str, pd.DataFrame], active_sheet: str | None = None
    ) -> tuple[pd.DataFrame, str | None]:
        """
        Pick which frame drives analysis and return (dataframe, active_label).
          - active_sheet given & present → that sheet
          - 1 sheet → that sheet
          - N sheets same schema → vertical concat (label "__concat__")
          - N sheets different schema → highest-scoring sheet (silent-loss made
            visible via the returned label, surfaced to the user as a banner)
        """
        if active_sheet and active_sheet in sheets:
            return sheets[active_sheet].copy(), active_sheet

        if len(sheets) == 1:
            key = next(iter(sheets.keys()))
            return sheets[key], key

        column_sets = {tuple(df.columns.tolist()) for df in sheets.values()}
        # Đơn TMĐT tách theo sàn + bảng sản phẩm tra cứu → vẫn gộp các bảng đơn cùng schema.
        orders = {k: df for k, df in sheets.items() if "doanh_thu_thuan" in df.columns}
        group = sheets if len(column_sets) == 1 else orders
        if len(group) > 1 and len({frozenset(df.columns) for df in group.values()}) == 1:
            frames = []
            for sheet_name, df in group.items():
                frame = df.copy()
                frame["_source_sheet"] = sheet_name
                frames.append(frame)
            return pd.concat(frames, ignore_index=True), "__concat__"

        best_name, best_df = max(
            sheets.items(),
            key=lambda item: SessionStore._analysis_score(item[1]),
        )
        selected = best_df.copy()
        selected.attrs["source_sheet"] = best_name
        return selected, best_name

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
            return SessionStore._normalize_frame(read_csv_bytes(file_path.read_bytes()))
        if suffix in {".xlsx", ".xls"}:
            return SessionStore._normalize_frame(pd.read_excel(file_path))
        raise ValueError("Only CSV, XLSX, and XLS files are supported.")

    @staticmethod
    def _normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
        """Dọn một DataFrame vừa nạp, trước khi mọi thứ khác chạm vào nó.

        Thứ tự có chủ đích:
          1. cắt khoảng trắng thừa ở tên cột (" Profit " → "Profit");
          2. nhận diện cột ngày (dựa vào TÊN cột, nên phải sau bước 1);
          3. đọc cột tiền tệ dạng chữ thành số (" $ (4,533.75) " → -4533.75).

        Bỏ bước 3 thì `sum()` trên cột toàn NaN trả về 0 và agent báo
        "Tổng lợi nhuận là 0" — xem docs/EVALUATION.md.
        """
        from backend.app.services.ecommerce_semantic import add_metric_columns, rename_export_columns

        df = strip_column_names(df)
        # 1b. file export Shopee/TikTok → tên cột chuẩn (phải trước bước 2: ngày TikTok là dd/mm).
        df = rename_export_columns(df)
        df = SessionStore._coerce_datetime_columns(df)
        df = coerce_numeric_columns(df)
        # 4. thêm metric TMĐT tính sẵn (lãi thật, phí sàn) nếu bảng đúng schema đơn hàng.
        return add_metric_columns(df)

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


def build_source_frame(
    session: DatasetSession, source: dict | None
) -> tuple[pd.DataFrame, str | None]:
    """
    Resolve a plan's optional `source` to the DataFrame a query should run on.

      - None / empty            → session.dataframe (active sheet) — backward compatible
      - {"sheet": "<name>"}     → that single sheet
      - {"join": {"base","with","on","how"}} → left/inner join of two sheets

    Safety: only joins on a column present in BOTH sheets; defaults how="left";
    any resolution error falls back to session.dataframe so the query still runs.
    Returns (frame, warning) — warning is a user-facing note (e.g. fan-out) or None.
    """
    if not source or not isinstance(source, dict):
        return session.dataframe, None

    sheets = session.sheets or {}
    if not sheets:
        return session.dataframe, None

    try:
        # Single sheet
        if source.get("sheet"):
            key = resolve_sheet_key(str(source["sheet"]), sheets)
            return sheets[key].copy(), None

        # Join
        join = source.get("join")
        if isinstance(join, dict):
            base_key = resolve_sheet_key(str(join.get("base", "")), sheets)
            with_key = resolve_sheet_key(str(join.get("with", "")), sheets)
            base_df, with_df = sheets[base_key], sheets[with_key]

            on = join.get("on")
            common = list(set(base_df.columns) & set(with_df.columns))
            if not on:
                # Prefer a detected relationship join key, else any common column.
                rel_key = next(
                    (r.join_key for r in session.sheet_relationships
                     if r.join_key and {r.sheet1, r.sheet2} == {base_key, with_key}),
                    None,
                )
                on = rel_key or (common[0] if common else None)
            if not on or on not in base_df.columns or on not in with_df.columns:
                return session.dataframe, None  # can't join safely → active sheet

            how = join.get("how", "left")
            if how not in {"left", "inner"}:
                how = "left"
            check_join_size(base_df, with_df, on)
            merged = base_df.merge(with_df, on=on, how=how, suffixes=("", "_dup"))

            warning = None
            if len(merged) > 3 * max(len(base_df), 1):
                warning = (
                    f"Phép join '{on}' làm số dòng tăng từ {len(base_df)} lên {len(merged)} "
                    "(quan hệ 1-nhiều) — các phép tính tổng/đếm có thể bị nhân lên."
                )
            return merged, warning
    except JoinTooLarge:
        raise   # không lặng lẽ trả sheet active: plan_node sẽ rơi về fallback plan
    except Exception:
        return session.dataframe, None

    return session.dataframe, None



def lost_sheet_notice(session: DatasetSession) -> str | None:
    """Deterministic refusal when the active (merged) sheet could not be restored."""
    if not session.active_sheet_lost:
        return None
    return (
        f"Bảng gộp “{session.active_sheet}” không khôi phục được sau khi máy chủ khởi động lại, "
        "nên tôi không tính trên bảng khác để tránh ra số sai. Hãy bấm “Gộp” lại ở danh sách sheet "
        "(hoặc chọn một sheet) rồi hỏi lại."
    )


session_store = SessionStore()
