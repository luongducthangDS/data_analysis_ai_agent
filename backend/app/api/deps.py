from __future__ import annotations

from fastapi import Depends, HTTPException, Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from backend.app.core.auth import get_current_user, key_id
from backend.app.core.config import get_settings
from backend.app.services.storage import session_store, DatasetSession


def _rate_limit_key(request: Request) -> str:
    # Only a *valid* key gets its own bucket — otherwise a random X-API-Key per
    # request would dodge the limit.
    api_key = request.headers.get("X-API-Key")
    if api_key and api_key in get_settings().api_keys:
        return "key:" + key_id(api_key)
    # ponytail: rightmost X-Forwarded-For hop = client as seen by the platform
    # proxy (leftmost is client-spoofable). With 2+ proxies in front (CDN + LB)
    # this is the inner proxy → users share one bucket; then read the CDN's
    # client-IP header instead.
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[-1].strip()
    return get_remote_address(request)


limiter = Limiter(key_func=_rate_limit_key)
RATE_LIMIT = get_settings().rate_limit


def load_owned_session(session_id: str, user: dict) -> DatasetSession:
    """Resolve session_id → DatasetSession, 404 if not found, 403 if not owned.
    Blocking (may rebuild DataFrames from DB) — call from sync routes or a threadpool."""
    try:
        return session_store.get(session_id, owner_id=user.get("user_id", ""))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


def get_session(session_id: str, _user: dict = Depends(get_current_user)) -> DatasetSession:
    # Sync on purpose: FastAPI runs it in the threadpool, off the event loop.
    return load_owned_session(session_id, _user)
