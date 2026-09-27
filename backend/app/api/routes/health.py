from __future__ import annotations

from fastapi import APIRouter

from backend.app.schemas import HealthResponse
from backend.app.services.llm_service import get_active_provider
from backend.app.services.storage import session_store

router = APIRouter()


@router.get("/api/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness probe (Render healthCheckPath): async and no I/O on purpose.
    A sync route shares the 40-slot threadpool with /api/chat; hung LLM calls
    used to fill it, the probe timed out and Render restarted the container.
    `sessions` = sessions cached in this process, not a DB count."""
    return HealthResponse(
        status="ok",
        sessions=len(session_store._sessions),
        llm_provider=get_active_provider(),
    )
