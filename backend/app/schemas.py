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
    active_sheet: str | None = Field(default=None, description="Sheet key currently driving analysis ('__concat__' = merged same-schema sheets)")
    data_notes: list[str] = Field(default_factory=list, description="Seller data gaps (missing COGS / ads) shown right after upload")
    cogs_missing: int = Field(default=0, description="Number of SKUs without cost of goods — >0 enables the COGS template download")


class ActiveSheetResponse(BaseModel):
    session_id: str
    active_sheet: str | None = None
    profile: DatasetProfile
    preview_columns: list[str] = Field(default_factory=list)
    preview_rows: list[dict[str, Any]] = Field(default_factory=list)
    suggested_queries: list[str] = Field(default_factory=list)
    data_notes: list[str] = Field(default_factory=list)
    cogs_missing: int = 0


class ChatRequest(BaseModel):
    session_id: str
    question: str


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    charts: list[ChartSpec] = Field(default_factory=list)
    executed_queries: list[str] = Field(default_factory=list)
    query_type: str = "data_query"  # "data_query" | "bot_info" | "off_topic"
    source: str = "llm"  # "llm" | "fallback" | "deterministic" | "bot_info" | "off_topic"
    # Token / chi phí / số lần gọi LLM cho lượt hỏi này. Rỗng khi không gọi LLM
    # (bot_info, off_topic, hoặc toàn bộ rơi xuống rule-based).
    usage: dict = Field(default_factory=dict)


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
    active_sheet: str | None = None
    profile: DatasetProfile | None = None
    preview_columns: list[str] = Field(default_factory=list)
    preview_rows: list[dict[str, Any]] = Field(default_factory=list)


class ImportUrlRequest(BaseModel):
    url: str


# ── E-Commerce Dashboard ──────────────────────────────────────────────────────

class KPICard(BaseModel):
    label: str
    value: str
    delta: str | None = None
    delta_positive: bool | None = None
    formula: str = ""
    is_alert: bool = False


class DashboardResponse(BaseModel):
    session_id: str
    platform: str | None = None
    kpi_cards: list[KPICard] = Field(default_factory=list)
    charts: list[ChartSpec] = Field(default_factory=list)
    top_products: list[dict[str, Any]] = Field(default_factory=list)
    col_map: dict[str, str] = Field(default_factory=dict)
    unmapped_cols: list[str] = Field(default_factory=list)
    is_ecommerce: bool = False
    suggested_queries: list[str] = Field(default_factory=list)
    top_label: str | None = Field(default=None, description="Metric name shown for the Top 10 table")
