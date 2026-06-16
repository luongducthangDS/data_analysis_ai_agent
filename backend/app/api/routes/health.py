from __future__ import annotations

from fastapi import APIRouter

from backend.app.schemas import HealthResponse
from backend.app.services.llm_service import get_active_provider
from backend.app.services.storage import session_store

router = APIRouter()


@router.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        sessions=session_store.count(),
        llm_provider=get_active_provider(),
    )
