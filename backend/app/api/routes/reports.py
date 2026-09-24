from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, Response

from backend.app.core.auth import get_current_user
from backend.app.database import ReportModel, db_session
from backend.app.services.storage import REPORT_DIR

router = APIRouter()


@router.get("/api/report/{report_id}")
def download_report(
    report_id: str,
    _user: dict = Depends(get_current_user),
) -> Response:
    filename = f"report-{report_id}.md"
    path = REPORT_DIR / f"{report_id}.md"
    if path.exists():
        return FileResponse(path, media_type="text/markdown", filename=filename)
    # Disk wiped (ephemeral host) → serve the copy kept in the DB.
    with db_session() as db:
        report = db.get(ReportModel, report_id)
        content = report.content if report else None
    if content is None:
        raise HTTPException(status_code=404, detail="Report not found.")
    return Response(
        content,
        media_type="text/markdown",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
