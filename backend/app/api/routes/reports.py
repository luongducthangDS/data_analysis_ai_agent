from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response

from backend.app.core.auth import get_current_user
from backend.app.database import ReportModel, SessionModel, db_session

router = APIRouter()


@router.get("/api/report/{report_id}")
def download_report(
    report_id: str,
    _user: dict = Depends(get_current_user),
) -> Response:
    """Serve a chat report to the owner of its session. The DB copy is the one
    served: it is always there (disk may be wiped) and carries the owner."""
    with db_session() as db:
        row = (
            db.query(ReportModel.content, SessionModel.owner_id)
            .join(SessionModel, SessionModel.session_id == ReportModel.session_id)
            .filter(ReportModel.report_id == report_id)
            .first()
        )
    # No row = unknown report, a report from before session_id existed, or its
    # session was deleted/expired.
    if row is None:
        raise HTTPException(status_code=404, detail="Report not found.")
    content, owner_id = row
    user_id = _user.get("user_id", "")
    # Same rule as SessionStore._check_ownership.
    if owner_id and user_id and owner_id != user_id:
        raise HTTPException(status_code=403, detail="Access denied to report.")
    return Response(
        content,
        media_type="text/markdown",
        headers={"Content-Disposition": f'attachment; filename="report-{report_id}.md"'},
    )
