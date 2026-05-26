from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from dotenv import load_dotenv

load_dotenv()

from backend.app.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    ChatRequest,
    ChatResponse,
    HealthResponse,
    UploadResponse,
    GetSheetsResponse,
    MergeSheetsRequest,
    MergeSheetsResponse,
    SheetData,
)
from backend.app.services.analysis_intent import infer_grouped_metric_intent
from backend.app.services.charts import generate_question_charts, generate_recommended_charts
from backend.app.services.guardrails import describe_guardrails
from backend.app.services.insights import generate_insights
from backend.app.services.analysis_planner import run_planned_analysis
from backend.app.services.multi_sheet_analyzer import MultiSheetAnalyzer
from backend.app.services.profiler import build_profile
from backend.app.services.query_engine import run_readonly_query, simple_question_to_sql
from backend.app.services.reports import write_markdown_report
from backend.app.services.storage import REPORT_DIR, session_store
from backend.app.ui import render_home


app = FastAPI(
    title="Data Analysis AI Agent",
    description="Upload CSV/XLSX files and generate automatic profiling, charts, insights, and reports.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
def home() -> HTMLResponse:
    return HTMLResponse(render_home())


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", sessions=session_store.count())


@app.post("/api/upload", response_model=UploadResponse)
async def upload_dataset(files: list[UploadFile] = File(...)) -> UploadResponse:
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    uploads: list[tuple[str, bytes]] = []
    for file in files:
        if not file.filename:
            raise HTTPException(status_code=400, detail="Missing filename.")
        if Path(file.filename).suffix.lower() not in {".csv", ".xlsx", ".xls"}:
            raise HTTPException(status_code=400, detail="Only CSV, XLSX, and XLS files are supported.")

        content = await file.read()
        if len(content) > 10 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="MVP upload limit is 10MB per file.")
        uploads.append((file.filename, content))

    try:
        session = session_store.create_multiple(uploads)
        profile = build_profile(session.dataframe)
        session.profile = profile

        sheet_names = list(session.sheets.keys()) if session.sheets else None
        file_sheet_map: dict[str, list[str]] = {}
        for sheet_key in sheet_names or []:
            if "::" in sheet_key:
                file_name, sheet_name = sheet_key.split("::", 1)
            else:
                file_name, sheet_name = sheet_key, sheet_key
            file_sheet_map.setdefault(file_name, []).append(sheet_key)

        preview_df = session.dataframe.head(10)
        preview_columns = [str(c) for c in preview_df.columns]
        preview_rows = preview_df.astype(str).where(preview_df.notna(), None).to_dict(orient="records")
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
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/analyze", response_model=AnalyzeResponse)
def analyze_dataset(req: AnalyzeRequest) -> AnalyzeResponse:
    try:
        session = session_store.get(req.session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    profile = session.profile or build_profile(session.dataframe)
    session.profile = profile
    executed_queries: list[str] = []
    try:
        if req.question:
            planned = run_planned_analysis(session.dataframe, req.question, profile)
            answer = planned.answer
            charts = planned.charts or generate_recommended_charts(session.dataframe)
            executed_queries = planned.executed_queries
        else:
            raise ValueError("No question provided; use automatic analysis.")
    except Exception as exc:
        answer = generate_insights(
            session.dataframe,
            profile,
            req.question,
            sheets=session.sheets if session.sheets else None,
            sheets_context=session.sheets_context if session.sheets_context else None,
        )
        charts = generate_question_charts(session.dataframe, req.question) or generate_recommended_charts(session.dataframe)
        executed_queries = [f"fallback_analysis: {exc}"] if req.question else []
    report_id, _ = write_markdown_report(answer, profile, charts)
    session.report_id = report_id
    session.history.append({"role": "assistant", "content": answer})

    return AnalyzeResponse(
        session_id=session.session_id,
        answer=answer,
        profile=profile,
        charts=charts,
        report_id=report_id,
        executed_queries=executed_queries,
        guardrails=describe_guardrails(),
    )


def _generate_suggested_queries(df, profile: dict) -> list[str]:
    """Generate relevant example queries based on dataset schema."""
    import pandas as pd
    suggestions = []
    numeric_cols = list(profile.get("numeric_summary", {}).keys())
    cat_cols = list(profile.get("categorical_summary", {}).keys())
    datetime_cols = [c for c, t in profile.get("column_types", {}).items() if "datetime" in str(t)]

    metric = numeric_cols[0] if numeric_cols else None
    dim1 = cat_cols[0] if cat_cols else None
    dim2 = cat_cols[1] if len(cat_cols) > 1 else None

    # Status/stage-like column detection
    status_col = next((c for c in cat_cols if any(kw in c.lower() for kw in ("status", "trang_thai", "state", "stage"))), None)
    # Amount/value column
    amount_col = next((c for c in numeric_cols if any(kw in c.lower() for kw in ("amount", "value", "total", "sum", "revenue", "cost", "price", "doanh", "tien"))), metric)

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
    if datetime_cols and amount_col:
        suggestions.append(f"trend {amount_col} theo tháng")
    if amount_col:
        suggestions.append(f"top 5 {amount_col} lớn nhất")

    # Try to detect high-value outliers hint
    if amount_col and amount_col in df.columns:
        try:
            q95 = float(df[amount_col].quantile(0.95))
            suggestions.append(f"{amount_col} > {q95:,.0f} (top 5%)")
        except Exception:
            pass

    return suggestions[:8]


@app.post("/api/chat", response_model=ChatResponse)
def chat_with_dataset(req: ChatRequest) -> ChatResponse:
    try:
        session = session_store.get(req.session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    try:
        profile = session.profile or build_profile(session.dataframe)
        session.profile = profile
        planned = run_planned_analysis(session.dataframe, req.question, profile)
        answer = planned.answer
        executed_queries = planned.executed_queries
    except Exception as exc:
        try:
            sql = simple_question_to_sql(req.question, session.dataframe)
            rows = run_readonly_query(session.dataframe, sql)
            answer = _format_chat_answer(req.question, session.dataframe, rows)
            executed_queries = [sql]
        except Exception as fallback_exc:
            raise HTTPException(status_code=400, detail=str(fallback_exc)) from exc

    session.history.append({"role": "user", "content": req.question})
    session.history.append({"role": "assistant", "content": answer})
    return ChatResponse(session_id=req.session_id, answer=answer, executed_queries=executed_queries)


def _format_chat_answer(question: str, df, rows: list[dict]) -> str:
    grouped_intent = infer_grouped_metric_intent(question, df)
    if grouped_intent and rows:
        metric_key = next((key for key in rows[0] if key.startswith("total_")), None)
        if metric_key:
            total = sum(float(row.get(metric_key) or 0) for row in rows)
            top = rows[0]
            top_value = float(top.get(metric_key) or 0)
            top_share = top_value / total * 100 if total else 0
            lines = [
                f"{grouped_intent.metric_label.capitalize()} theo {grouped_intent.dimension_label}: tổng {total:,.0f}.",
                f"Nhóm dẫn đầu là {top.get(grouped_intent.dimension)} với {top_value:,.0f}, chiếm {top_share:.1f}%.",
                "",
                "Top kết quả:",
            ]
            for index, row in enumerate(rows[:5], start=1):
                value = row.get(grouped_intent.dimension)
                metric_value = float(row.get(metric_key) or 0)
                share = metric_value / total * 100 if total else 0
                lines.append(f"{index}. {value}: {metric_value:,.0f} ({share:.1f}%)")
            return "\n".join(lines)

    if len(rows) == 1 and rows[0]:
        key, value = next(iter(rows[0].items()))
        if isinstance(value, (int, float)):
            return f"Kết quả truy vấn: `{key}` = {value:,.0f}"
        return f"Kết quả truy vấn: `{key}` = {value}"

    preview = rows[:5]
    answer = f"Đã chạy truy vấn an toàn trên dataset và tìm thấy {len(rows)} dòng kết quả."
    if preview:
        answer += f"\nPreview: {preview}"
    return answer


@app.get("/api/sheets/{session_id}", response_model=GetSheetsResponse)
def get_sheets(session_id: str) -> GetSheetsResponse:
    """Get all sheets and their relationships from an Excel file"""
    try:
        session = session_store.get(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

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
        session_id=session_id,
        files=session.file_names if session.file_names else None,
        sheets=sheets_data,
        relationships=relationships,
    )


@app.post("/api/merge-sheets", response_model=MergeSheetsResponse)
def merge_sheets(req: MergeSheetsRequest) -> MergeSheetsResponse:
    """Merge multiple sheets into one"""
    try:
        session = session_store.get(req.session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if not session.sheets or len(session.sheets) < 2:
        raise HTTPException(status_code=400, detail="Not enough sheets to merge.")

    if len(req.sheet_names) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 sheets to merge.")

    try:
        resolved_sheet_names = [_resolve_sheet_key(name, session.sheets) for name in req.sheet_names]
        sheets_to_merge = {name: session.sheets[name] for name in resolved_sheet_names}

        # Start with first sheet
        merged_df = sheets_to_merge[resolved_sheet_names[0]].copy()

        # Merge remaining sheets
        for sheet_name in resolved_sheet_names[1:]:
            df_to_merge = sheets_to_merge[sheet_name]

            if req.join_key:
                # Explicit join key
                merged_df = merged_df.merge(df_to_merge, on=req.join_key, how="left", suffixes=("", "_dup"))
            else:
                # Auto-detect common column
                common_cols = set(merged_df.columns) & set(df_to_merge.columns)
                if not common_cols:
                    raise ValueError(f"No common columns found between sheets to merge on.")

                join_col = list(common_cols)[0]
                merged_df = merged_df.merge(df_to_merge, on=join_col, how="left", suffixes=("", "_dup"))

        # Update session with merged data
        merged_name = "_".join(resolved_sheet_names)
        session.sheets[merged_name] = merged_df

        return MergeSheetsResponse(
            session_id=session.session_id,
            merged_rows=len(merged_df),
            merged_columns=len(merged_df.columns),
            merged_sheet_name=merged_name,
        )

    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _resolve_sheet_key(name: str, sheets: dict) -> str:
    if name in sheets:
        return name

    matches = [key for key in sheets if key.split("::", 1)[-1] == name]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous sheet name '{name}'. Use one of: {', '.join(matches)}")
    raise ValueError(f"Unknown sheet name '{name}'.")


@app.get("/api/report/{report_id}")
def download_report(report_id: str) -> FileResponse:
    path = REPORT_DIR / f"{report_id}.md"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Report not found.")
    return FileResponse(path, media_type="text/markdown", filename=f"report-{report_id}.md")
