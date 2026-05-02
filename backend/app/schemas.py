from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class DatasetProfile(BaseModel):
    rows: int
    columns: int
    column_types: dict[str, str]
    missing_values: dict[str, int]
    numeric_summary: dict[str, dict[str, float | int | None]]
    categorical_summary: dict[str, list[dict[str, Any]]]


class ChartSpec(BaseModel):
    chart_id: str
    title: str
    chart_type: Literal["bar", "line", "histogram", "scatter"]
    x: str | None = None
    y: str | None = None
    plotly_json: dict[str, Any]


class UploadResponse(BaseModel):
    session_id: str
    filename: str
    profile: DatasetProfile


class AnalyzeRequest(BaseModel):
    session_id: str
    question: str | None = Field(
        default=None,
        description="Optional business question. If omitted, the system performs automatic analysis.",
    )


class AnalyzeResponse(BaseModel):
    session_id: str
    answer: str
    profile: DatasetProfile
    charts: list[ChartSpec]
    report_id: str
    executed_queries: list[str] = Field(default_factory=list)
    guardrails: list[str] = Field(default_factory=list)


class ChatRequest(BaseModel):
    session_id: str
    question: str


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    executed_queries: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    sessions: int

