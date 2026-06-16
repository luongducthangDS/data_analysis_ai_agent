from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from backend.app.core.auth import get_current_user
from backend.app.services.storage import REPORT_DIR

router = APIRouter()


@router.get("/api/report/{report_id}")
def download_report(
    report_id: str,
    _user: dict = Depends(get_current_user),
) -> FileResponse:
    path = REPORT_DIR / f"{report_id}.md"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Report not found.")
    return FileResponse(path, media_type="text/markdown", filename=f"report-{report_id}.md")
