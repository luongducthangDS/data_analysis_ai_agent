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
    filenames: list[str] | None = Field(default=None, description="Original uploaded filenames")
    profile: DatasetProfile
    sheet_names: list[str] | None = Field(default=None, description="List of sheet identifiers from uploaded files")
    file_sheet_map: dict[str, list[str]] | None = Field(default=None, description="Mapping of filename to sheet identifiers")
    sheets_context: str | None = Field(default=None, description="Description of sheet structure")
    preview_columns: list[str] = Field(default_factory=list, description="Column names for table preview")
    preview_rows: list[dict[str, Any]] = Field(default_factory=list, description="First 10 rows for table preview")
    suggested_queries: list[str] = Field(default_factory=list, description="Auto-generated example queries")


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
    source: str = "llm"  # "llm" | "fallback" | "bot_info" | "off_topic"


class ChatRequest(BaseModel):
    session_id: str
    question: str


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    charts: list[ChartSpec] = Field(default_factory=list)
    executed_queries: list[str] = Field(default_factory=list)
    query_type: str = "data_query"  # "data_query" | "bot_info" | "off_topic"
    source: str = "llm"  # "llm" | "fallback" | "bot_info" | "off_topic"


class HealthResponse(BaseModel):
    status: str
    sessions: int
    llm_provider: str = "unknown"


class SheetData(BaseModel):
    file_name: str | None = None
    name: str
    rows: int
    columns: int
    column_names: list[str]
    preview: list[dict[str, Any]] = Field(default_factory=list, description="First 5 rows")


class GetSheetsResponse(BaseModel):
    session_id: str
    files: list[str] | None = Field(default=None, description="Uploaded filenames")
    sheets: list[SheetData]
    relationships: list[dict[str, Any]] = Field(
        default_factory=list, description="Detected relationships between sheets"
    )


class MergeSheetsRequest(BaseModel):
    session_id: str
    sheet_names: list[str] = Field(description="Sheet names to merge")
    join_key: str | None = Field(default=None, description="Column to join on")


class MergeSheetsResponse(BaseModel):
    session_id: str
    merged_rows: int
    merged_columns: int
    merged_sheet_name: str


class ImportUrlRequest(BaseModel):
    url: str


class ImportGSheetRequest(BaseModel):
    url_or_id: str
    sheet_name: str | None = None


class AgentStepSchema(BaseModel):
    step: int
    tool_name: str
    arguments: dict[str, Any]
    result_summary: str
    charts: list[ChartSpec] = Field(default_factory=list)


class AgentChatResponse(BaseModel):
    session_id: str
    answer: str
    charts: list[ChartSpec] = Field(default_factory=list)
    agent_steps: list[AgentStepSchema] = Field(default_factory=list)
    executed_queries: list[str] = Field(default_factory=list)
