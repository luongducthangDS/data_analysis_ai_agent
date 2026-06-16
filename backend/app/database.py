from __future__ import annotations

import os
from contextlib import contextmanager
from datetime import datetime
from typing import Generator

from sqlalchemy import Column, DateTime, ForeignKey, Integer, JSON, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, relationship, sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/sessions.db")

# Normalize legacy Railway postgres:// → postgresql://
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
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    history = relationship(
        "ChatHistoryModel",
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="ChatHistoryModel.id",
    )


class ChatHistoryModel(Base):
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
    created_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("SessionModel", back_populates="history")


def init_db() -> None:
    """Create all tables if they don't exist. Called once on app startup."""
    Base.metadata.create_all(bind=engine)
    _migrate_add_ecommerce_columns()


def _migrate_add_ecommerce_columns() -> None:
    """Idempotent: add new columns to existing sessions table (SQLite + PostgreSQL safe)."""
    from sqlalchemy import text
    new_cols = [
        ("ecommerce_col_map", "JSON"),
        ("detected_platform", "VARCHAR(32)"),
    ]
    with engine.connect() as conn:
        for col_name, col_type in new_cols:
            try:
                conn.execute(text(f"ALTER TABLE sessions ADD COLUMN {col_name} {col_type}"))
                conn.commit()
            except Exception:
                pass  # column already exists


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
