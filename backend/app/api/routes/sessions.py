from __future__ import annotations

import io

from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.responses import StreamingResponse

from backend.app.core.auth import get_current_user
from backend.app.services.storage import session_store

router = APIRouter()


@router.delete("/api/session/{session_id}", status_code=204, response_class=Response)
def delete_session(
    session_id: str,
    _user: dict = Depends(get_current_user),
) -> Response:
    """Delete a session and all associated uploaded files and history."""
    owner_id = _user.get("user_id", "")
    try:
        session_store.get(session_id, owner_id=owner_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc

    session_store.delete_session(session_id)
    return Response(status_code=204)


@router.get("/api/session/{session_id}/data.csv")
def export_session_csv(
    session_id: str,
    _user: dict = Depends(get_current_user),
) -> StreamingResponse:
    """Export the session's dataset as CSV."""
    owner_id = _user.get("user_id", "")
    try:
        session = session_store.get(session_id, owner_id=owner_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc

    buf = io.StringIO()
    session.dataframe.to_csv(buf, index=False)
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=data-{session_id[:8]}.csv"},
    )
