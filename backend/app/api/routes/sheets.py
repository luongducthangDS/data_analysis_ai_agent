from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from backend.app.api.deps import get_session
from backend.app.core.auth import get_current_user
from pydantic import BaseModel

from backend.app.schemas import (
    ActiveSheetResponse,
    GetSheetsResponse,
    MergeSheetsRequest,
    MergeSheetsResponse,
    SheetData,
)
from backend.app.api.routes.upload import build_preview
from backend.app.services.storage import session_store, DatasetSession, resolve_sheet_key


class SetActiveSheetRequest(BaseModel):
    sheet_name: str


def _refresh_payload(session: DatasetSession) -> ActiveSheetResponse:
    """Build the post-switch/merge refresh payload (profile + preview) for the frontend."""
    from backend.app.api.routes.upload import _generate_suggested_queries
    cols, rows = build_preview(session.dataframe)
    return ActiveSheetResponse(
        session_id=session.session_id,
        active_sheet=session.active_sheet,
        profile=session.profile,
        preview_columns=cols,
        preview_rows=rows,
        suggested_queries=_generate_suggested_queries(session.dataframe, session.profile),
    )

_log = logging.getLogger(__name__)
router = APIRouter()


@router.get("/api/sheets/{session_id}", response_model=GetSheetsResponse)
def get_sheets(
    session_id: str,
    session: DatasetSession = Depends(get_session),
) -> GetSheetsResponse:
    if not session.sheets:
        raise HTTPException(status_code=400, detail="This dataset has no multiple sheets.")

    sheets_data = []
    for sheet_key, df in session.sheets.items():
        preview = df.head(5).to_dict(orient="records")
        if "::" in sheet_key:
            file_name, sheet_name = sheet_key.split("::", 1)
        else:
            file_name, sheet_name = sheet_key, sheet_key
        sheets_data.append(
            SheetData(
                file_name=file_name,
                name=sheet_name,
                rows=len(df),
                columns=len(df.columns),
                column_names=df.columns.tolist(),
                preview=preview,
            )
        )

    relationships = [
        {
            "sheet1": rel.sheet1,
            "sheet2": rel.sheet2,
            "join_key": rel.join_key,
            "relationship_type": rel.relationship_type,
            "similarity_score": rel.similarity_score,
        }
        for rel in session.sheet_relationships
        if rel.similarity_score > 0.3
    ]

    return GetSheetsResponse(
        session_id=session.session_id,
        files=session.file_names if session.file_names else None,
        sheets=sheets_data,
        relationships=relationships,
    )


@router.post("/api/session/{session_id}/active-sheet", response_model=ActiveSheetResponse)
def set_active_sheet(
    session_id: str,
    req: SetActiveSheetRequest,
    session: DatasetSession = Depends(get_session),
) -> ActiveSheetResponse:
    """Switch which sheet drives analysis; returns refreshed profile + preview."""
    if not session.sheets:
        raise HTTPException(status_code=400, detail="This dataset has no multiple sheets.")
    try:
        session_store.set_active_sheet(session, req.sheet_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _refresh_payload(session)


@router.post("/api/merge-sheets", response_model=MergeSheetsResponse)
def merge_sheets(
    req: MergeSheetsRequest,
    _user: dict = Depends(get_current_user),
) -> MergeSheetsResponse:
    try:
        session = session_store.get(req.session_id, owner_id=_user.get("user_id", ""))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc

    if not session.sheets or len(session.sheets) < 2:
        raise HTTPException(status_code=400, detail="Not enough sheets to merge.")
    if len(req.sheet_names) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 sheets to merge.")

    try:
        resolved = [resolve_sheet_key(n, session.sheets) for n in req.sheet_names]
        sheets_to_merge = {n: session.sheets[n] for n in resolved}

        merged_df = sheets_to_merge[resolved[0]].copy()
        for sheet_name in resolved[1:]:
            df_to_merge = sheets_to_merge[sheet_name]
            if req.join_key:
                merged_df = merged_df.merge(df_to_merge, on=req.join_key, how="left", suffixes=("", "_dup"))
            else:
                common_cols = set(merged_df.columns) & set(df_to_merge.columns)
                if not common_cols:
                    raise ValueError("No common columns found between sheets to merge on.")
                join_col = list(common_cols)[0]
                merged_df = merged_df.merge(df_to_merge, on=join_col, how="left", suffixes=("", "_dup"))

        # Readable label from the sheet parts ("Orders + Items"), not the raw
        # "file::sheet" keys. Keep it unique against existing sheet keys.
        short = " + ".join(k.split("::")[-1] for k in resolved)
        merged_name = short
        i = 2
        while merged_name in session.sheets:
            merged_name = f"{short} ({i})"
            i += 1
        session.sheets[merged_name] = merged_df

        # Make the merged frame the active analysis target (re-profiles + persists),
        # so chat/dashboard actually use it instead of leaving it inert.
        session_store.set_active_sheet(session, merged_name)

        cols, rows = build_preview(session.dataframe)
        return MergeSheetsResponse(
            session_id=session.session_id,
            merged_rows=len(merged_df),
            merged_columns=len(merged_df.columns),
            merged_sheet_name=merged_name,
            active_sheet=session.active_sheet,
            profile=session.profile,
            preview_columns=cols,
            preview_rows=rows,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
