from __future__ import annotations

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

from backend.app.core.config import get_settings

_API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


async def get_current_user(api_key: str | None = Security(_API_KEY_HEADER)) -> dict:
    """
    Validate API key from X-API-Key header.
    In dev mode (ALLOW_NO_AUTH=true) or when no API keys are configured, all requests pass.
    """
    settings = get_settings()

    # Dev mode: no auth required
    if settings.allow_no_auth or not settings.api_keys:
        return {"user_id": "anonymous", "api_key": api_key or ""}

    if not api_key or api_key not in settings.api_keys:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key. Provide X-API-Key header.",
        )
    return {"user_id": api_key[:8], "api_key": api_key}
