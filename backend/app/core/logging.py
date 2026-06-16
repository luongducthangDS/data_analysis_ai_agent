from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

_LOG_DIR = Path(__file__).resolve().parents[4] / "logs"
_LOG_FILE = _LOG_DIR / "app.log"
_FMT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DATE_FMT = "%Y-%m-%dT%H:%M:%S"


def setup_logging(debug: bool = False) -> None:
    level = logging.DEBUG if debug else logging.INFO

    _LOG_DIR.mkdir(exist_ok=True)

    file_handler = RotatingFileHandler(
        _LOG_FILE,
        maxBytes=10 * 1024 * 1024,  # 10 MB per file
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(logging.Formatter(_FMT, datefmt=_DATE_FMT))

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(_FMT, datefmt=_DATE_FMT))

    logging.basicConfig(level=level, handlers=[file_handler, console_handler])

    # Quiet noisy third-party loggers
    for name in ("httpx", "httpcore", "groq", "google"):
        logging.getLogger(name).setLevel(logging.WARNING)


def setup_langsmith(api_key: str, project: str) -> bool:
    """Configure LangSmith tracing. Returns True if enabled."""
    if not api_key:
        return False
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = api_key
    os.environ["LANGCHAIN_PROJECT"] = project
    logging.getLogger(__name__).info("LangSmith tracing enabled → project=%r", project)
    return True
