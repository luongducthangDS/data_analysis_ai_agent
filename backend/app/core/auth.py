from __future__ import annotations

import hashlib

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

from backend.app.core.config import get_settings

_API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def key_id(api_key: str) -> str:
    """Stable, non-secret id for an API key (owner_id, rate-limit bucket)."""
    return hashlib.sha256(api_key.encode()).hexdigest()[:16]


async def get_current_user(api_key: str | None = Security(_API_KEY_HEADER)) -> dict:
    """
    Validate API key from X-API-Key header.
    Fail-closed: only ALLOW_NO_AUTH=true lets requests through without a key.
    ALLOW_NO_AUTH=false with no API_KEYS configured rejects everything.
    """
    settings = get_settings()

    if settings.allow_no_auth:
        return {"user_id": "anonymous", "api_key": ""}

    if not api_key or api_key not in settings.api_keys:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key. Provide X-API-Key header.",
        )
    return {"user_id": key_id(api_key), "api_key": api_key}
