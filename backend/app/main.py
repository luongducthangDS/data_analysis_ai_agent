from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse

from backend.app.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    ChatRequest,
    ChatResponse,
    HealthResponse,
    UploadResponse,
)
from backend.app.services.charts import generate_question_charts, generate_recommended_charts
from backend.app.services.guardrails import describe_guardrails
from backend.app.services.insights import generate_insights
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
async def upload_dataset(file: UploadFile = File(...)) -> UploadResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename.")
    if Path(file.filename).suffix.lower() not in {".csv", ".xlsx", ".xls"}:
        raise HTTPException(status_code=400, detail="Only CSV, XLSX, and XLS files are supported.")

    content = await file.read()
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="MVP upload limit is 10MB.")

    try:
        session = session_store.create(file.filename, content)
        profile = build_profile(session.dataframe)
        session.profile = profile
        return UploadResponse(session_id=session.session_id, filename=session.filename, profile=profile)
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
    answer = generate_insights(session.dataframe, profile, req.question)
    charts = generate_question_charts(session.dataframe, req.question) or generate_recommended_charts(session.dataframe)
    report_id, _ = write_markdown_report(answer, profile, charts)
    session.report_id = report_id
    session.history.append({"role": "assistant", "content": answer})

    return AnalyzeResponse(
        session_id=session.session_id,
        answer=answer,
        profile=profile,
        charts=charts,
        report_id=report_id,
        guardrails=describe_guardrails(),
    )


@app.post("/api/chat", response_model=ChatResponse)
def chat_with_dataset(req: ChatRequest) -> ChatResponse:
    try:
        session = session_store.get(req.session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    try:
        sql = simple_question_to_sql(req.question, session.dataframe)
        rows = run_readonly_query(session.dataframe, sql)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    preview = rows[:5]
    answer = f"Đã chạy truy vấn an toàn trên dataset và tìm thấy {len(rows)} dòng kết quả."
    if preview:
        answer += f"\nPreview: {preview}"

    session.history.append({"role": "user", "content": req.question})
    session.history.append({"role": "assistant", "content": answer})
    return ChatResponse(session_id=req.session_id, answer=answer, executed_queries=[sql])


@app.get("/api/report/{report_id}")
def download_report(report_id: str) -> FileResponse:
    path = REPORT_DIR / f"{report_id}.md"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Report not found.")
    return FileResponse(path, media_type="text/markdown", filename=f"report-{report_id}.md")
