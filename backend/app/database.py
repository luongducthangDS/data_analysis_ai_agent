from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from typing import Generator

from sqlalchemy import Column, DateTime, ForeignKey, Integer, JSON, LargeBinary, String, Text, create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, Session, relationship, sessionmaker

from backend.app.core.config import get_settings

DATABASE_URL = get_settings().database_url

# Normalize legacy postgres:// (Railway, Supabase) → postgresql://
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
    pool_pre_ping=True,
)
SessionLocal = sessionmaker(bind=engine)


class Base(DeclarativeBase):
    pass


class SessionModel(Base):
    __tablename__ = "sessions"

    session_id = Column(String(64), primary_key=True)
    owner_id = Column(String(64), nullable=True, default="")
    filename = Column(String(255), nullable=False)
    file_names = Column(JSON, nullable=False, default=list)
    profile = Column(JSON, nullable=True)
    report_id = Column(String(64), nullable=True)
    sheet_relationships = Column(JSON, nullable=False, default=list)
    sheets_context = Column(Text, nullable=True)
    ecommerce_col_map = Column(JSON, nullable=True, default=None)
    detected_platform = Column(String(32), nullable=True, default=None)
    active_sheet = Column(String(255), nullable=True, default=None)
    # Legacy: whole history as one JSON blob, rewritten on every save → lost
    # updates with >1 worker. Now read-only fallback; new turns go to chat_history.
    chat_log = Column(JSON, nullable=True, default=None)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    history = relationship(
        "ChatHistoryModel",
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="ChatHistoryModel.id",
    )
    files = relationship(
        "SessionFileModel",
        cascade="all, delete-orphan",
        order_by="SessionFileModel.id",
    )


class ChatHistoryModel(Base):
    """One row per message, append-only — safe with several workers."""
    __tablename__ = "chat_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(
        String(64),
        ForeignKey("sessions.session_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    source = Column(String(32), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("SessionModel", back_populates="history")


class SessionFileModel(Base):
    """Raw bytes of each uploaded file, so a session can be rebuilt after the
    local disk is wiped. ponytail: bytea in Postgres — Supabase free DB is
    500 MB total; move to Supabase Storage if uploads outgrow that."""
    __tablename__ = "session_files"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(
        String(64),
        ForeignKey("sessions.session_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    name = Column(String(255), nullable=False)
    content = Column(LargeBinary, nullable=False)


class ReportModel(Base):
    __tablename__ = "reports"

    report_id = Column(String(64), primary_key=True)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


def init_db() -> None:
    """Create tables, add missing columns, enable RLS. Idempotent; called from
    the app lifespan (and tests/conftest.py) — not at import time."""
    Base.metadata.create_all(bind=engine)
    _add_missing_columns()
    _enable_rls()


def _add_missing_columns() -> None:
    """create_all never alters existing tables: add any model column the live
    table lacks. ponytail: add-only (nullable columns); renames/type changes
    need a real migration tool (Alembic) — add it when that first happens."""
    existing = inspect(engine)
    with engine.begin() as conn:
        for table in Base.metadata.sorted_tables:
            if not existing.has_table(table.name):
                continue
            have = {c["name"] for c in existing.get_columns(table.name)}
            for col in table.columns:
                if col.name not in have:
                    col_type = col.type.compile(dialect=engine.dialect)
                    conn.execute(text(f'ALTER TABLE "{table.name}" ADD COLUMN "{col.name}" {col_type}'))


def _enable_rls() -> None:
    """Supabase exposes the public schema through its REST API (anon key).
    RLS with no policies blocks that path; the backend connects as the
    table owner, which bypasses RLS."""
    if engine.dialect.name != "postgresql":
        return
    with engine.begin() as conn:
        for table in Base.metadata.tables:
            conn.execute(text(f'ALTER TABLE "{table}" ENABLE ROW LEVEL SECURITY'))


@contextmanager
def db_session() -> Generator[Session, None, None]:
    """Yield a transactional DB session; commit on success, rollback on error."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
