from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from backend.app.api.deps import limiter
from backend.app.api.routes import chat, dashboard, export, health, reports, sessions, sheets, upload
from backend.app.core.config import get_settings
from backend.app.core.logging import setup_langsmith, setup_logging
from backend.app.database import init_db

# ── Bootstrap ────────────────────────────────────────────────────────────────
_settings = get_settings()
setup_logging(_settings.debug)
setup_langsmith(_settings.langsmith_api_key, _settings.langsmith_project)
_log = logging.getLogger(__name__)
if not _settings.allow_no_auth and not _settings.api_keys:
    _log.error("ALLOW_NO_AUTH=false and API_KEYS is empty: every API request will get 401.")

# React build output (frontend/vite.config.ts → outDir: "../dist")
DIST_DIR = Path(__file__).resolve().parents[2] / "dist"


async def _cleanup_loop() -> None:
    from backend.app.services.storage import session_store
    while True:
        await asyncio.sleep(3600)
        try:
            # DB + file deletes are blocking — run them off the event loop.
            await asyncio.to_thread(
                session_store.cleanup_old_sessions, max_age_days=_settings.session_ttl_days
            )
        except Exception:
            _log.exception("session cleanup failed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Schema setup at startup, not at import: importing a module shouldn't hit the DB.
    await asyncio.to_thread(init_db)
    cleanup = asyncio.create_task(_cleanup_loop())   # keep a ref, or it can be GC'd
    yield
    cleanup.cancel()


# ── App factory ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="SellerLens",
    description="Upload CSV/XLSX files and chat with your data using AI.",
    version="1.0.0",
    contact={
        "name": "Luong Duc Thang",
        "url": "https://github.com/luongducthangDS",
        "email": "luongducthang289@gmail.com",
    },
    license_info={"name": "MIT", "url": "https://opensource.org/licenses/MIT"},
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

_origins = [o.strip() for o in _settings.allowed_origins.split(",")]
_allow_creds = _origins != ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=_allow_creds,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── API routers ───────────────────────────────────────────────────────────────
app.include_router(health.router)
app.include_router(upload.router)
app.include_router(chat.router)
app.include_router(sheets.router)
app.include_router(reports.router)
app.include_router(sessions.router)
app.include_router(dashboard.router)
app.include_router(export.router)

# ── Static assets ─────────────────────────────────────────────────────────────
_assets_dir = DIST_DIR / "assets"
if _assets_dir.exists():
    app.mount("/assets", StaticFiles(directory=str(_assets_dir)), name="assets")


# ── SPA routes — MUST be last, after all /api/* routes ───────────────────────
@app.get("/", response_class=HTMLResponse)
def home() -> HTMLResponse:
    index = DIST_DIR / "index.html"
    if index.exists():
        return HTMLResponse(index.read_text(encoding="utf-8"))
    return HTMLResponse("<h2>Frontend not built. Run: cd frontend && npm run build</h2>", status_code=503)


@app.get("/{full_path:path}", response_class=HTMLResponse)
def spa_fallback(full_path: str) -> HTMLResponse:
    if full_path.startswith("api/"):   # mistyped API path → JSON 404, not the SPA with 200
        raise HTTPException(status_code=404, detail="Not Found")
    index = DIST_DIR / "index.html"
    if index.exists():
        return HTMLResponse(index.read_text(encoding="utf-8"))
    return HTMLResponse("<h2>Frontend not built.</h2>", status_code=503)
