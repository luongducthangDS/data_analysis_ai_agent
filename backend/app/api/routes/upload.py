import logging
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from sqlalchemy.exc import SQLAlchemyError

from backend.app.api.deps import RATE_LIMIT, limiter
from backend.app.core.auth import get_current_user
from backend.app.schemas import ImportUrlRequest, UploadResponse
from backend.app.services.profiler import build_profile
from backend.app.services.storage import session_store
from backend.app.services.workspace_connectors import fetch_from_url

_log = logging.getLogger(__name__)
router = APIRouter()


def _generate_suggested_queries(df, profile: dict) -> list[str]:
    suggestions: list[str] = []
    numeric_cols = list(profile.get("numeric_summary", {}).keys())
    cat_cols = list(profile.get("categorical_summary", {}).keys())

    metric = numeric_cols[0] if numeric_cols else None
    dim1 = cat_cols[0] if cat_cols else None
    dim2 = cat_cols[1] if len(cat_cols) > 1 else None

    status_col = next(
        (c for c in cat_cols if any(kw in c.lower() for kw in ("status", "trang_thai", "state", "stage"))), None
    )
    amount_col = next(
        (c for c in numeric_cols if any(kw in c.lower() for kw in
         ("amount", "value", "total", "sum", "revenue", "cost", "price", "doanh", "tien"))),
        metric,
    )

    if amount_col and dim1:
        suggestions.append(f"tổng {amount_col} theo {dim1}")
    if amount_col and dim2:
        suggestions.append(f"top 10 {dim2} theo {amount_col}")
    if amount_col and status_col:
        suggestions.append(f"{amount_col} trung bình theo {status_col}")
    if status_col:
        suggestions.append(f"số lượng record theo {status_col}")
    if amount_col and dim1 and dim2:
        suggestions.append(f"so sánh {amount_col} theo {dim1} và {dim2}")
    if amount_col:
        suggestions.append(f"top 5 {amount_col} lớn nhất")

    if amount_col and amount_col in df.columns:
        try:
            q95 = float(df[amount_col].quantile(0.95))
            suggestions.append(f"{amount_col} > {q95:,.0f} (top 5%)")
        except Exception:
            pass

    return suggestions[:8]


def _create_session_or_http_error(uploads: list[tuple[str, bytes]], owner_id: str) -> UploadResponse:
    """Map failures to status codes without leaking internals: ValueError is our
    (or pandas') message about the file itself → 400 as-is; a DB failure → 503;
    anything else is logged and gets a generic 400."""
    try:
        return _build_upload_response(uploads, owner_id=owner_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except SQLAlchemyError as exc:
        _log.exception("upload: DB write failed")
        raise HTTPException(status_code=503, detail="Không lưu được dữ liệu, vui lòng thử lại sau.") from exc
    except Exception as exc:
        _log.exception("upload: could not parse %s", [name for name, _ in uploads])
        raise HTTPException(status_code=400, detail="Không đọc được file. Kiểm tra lại định dạng CSV/XLSX.") from exc


def _build_upload_response(uploads: list[tuple[str, bytes]], owner_id: str = "") -> UploadResponse:
    session = session_store.create_multiple(uploads, owner_id=owner_id)
    profile = build_profile(session.dataframe)
    session.profile = profile
    session_store.save(session)

    sheet_names = list(session.sheets.keys()) if session.sheets else None
    file_sheet_map: dict[str, list[str]] = {}
    for sheet_key in sheet_names or []:
        file_name = sheet_key.split("::", 1)[0]   # "file::sheet" → file; plain CSV key → itself
        file_sheet_map.setdefault(file_name, []).append(sheet_key)

    preview_columns, preview_rows = build_preview(session.dataframe)
    suggested_queries = _generate_suggested_queries(session.dataframe, profile)

    return UploadResponse(
        session_id=session.session_id,
        filename=session.filename,
        filenames=session.file_names,
        profile=profile,
        sheet_names=sheet_names,
        file_sheet_map=file_sheet_map if file_sheet_map else None,
        sheets_context=session.sheets_context if sheet_names and len(sheet_names) > 1 else None,
        preview_columns=preview_columns,
        preview_rows=preview_rows,
        suggested_queries=suggested_queries,
        active_sheet=session.active_sheet,
    )


def build_preview(df, n: int = 10) -> tuple[list[str], list[dict]]:
    """Column names + first N rows (stringified, NaN→None) for table preview."""
    preview_df = df.head(n)
    columns = [str(c) for c in preview_df.columns]
    rows = preview_df.astype(str).where(preview_df.notna(), None).to_dict(orient="records")
    return columns, rows


MAX_UPLOAD_BYTES = 10 * 1024 * 1024


@router.post("/api/upload", response_model=UploadResponse)
@limiter.limit(RATE_LIMIT)
def upload_dataset(
    # Sync on purpose: pandas parsing is CPU-bound; FastAPI runs a `def` route
    # in the threadpool so one big upload doesn't freeze the event loop.
    request: Request,
    files: list[UploadFile] = File(...),
    _user: dict = Depends(get_current_user),
) -> UploadResponse:
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    uploads: list[tuple[str, bytes]] = []
    for file in files:
        if not file.filename:
            raise HTTPException(status_code=400, detail="Missing filename.")
        if Path(file.filename).suffix.lower() not in {".csv", ".xlsx", ".xls"}:
            raise HTTPException(status_code=400, detail="Only CSV, XLSX, and XLS files are supported.")
        content = file.file.read(MAX_UPLOAD_BYTES + 1)  # never load more than the limit
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="MVP upload limit is 10MB per file.")
        uploads.append((file.filename, content))

    return _create_session_or_http_error(uploads, _user.get("user_id", ""))


@router.post("/api/import-url", response_model=UploadResponse)
@limiter.limit(RATE_LIMIT)
def import_from_url(
    request: Request,
    req: ImportUrlRequest,
    _user: dict = Depends(get_current_user),
) -> UploadResponse:
    try:
        filename, content = fetch_from_url(req.url)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        _log.exception("import-url: fetch failed")
        raise HTTPException(status_code=400, detail="Không thể tải file từ URL này.") from exc
    return _create_session_or_http_error([(filename, content)], _user.get("user_id", ""))
