from __future__ import annotations

from fastapi import Depends, HTTPException, Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from backend.app.core.auth import get_current_user
from backend.app.core.config import get_settings
from backend.app.services.storage import session_store, DatasetSession


def _rate_limit_key(request: Request) -> str:
    api_key = request.headers.get("X-API-Key")
    return api_key if api_key else get_remote_address(request)


limiter = Limiter(key_func=_rate_limit_key)


async def get_session(session_id: str, _user: dict = Depends(get_current_user)) -> DatasetSession:
    """Resolve session_id → DatasetSession, 404 if not found, 403 if not owned."""
    try:
        return session_store.get(session_id, owner_id=_user.get("user_id", ""))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
