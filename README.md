# Data Analysis AI Agent

MVP AI engineering project for uploading CSV/XLSX files and generating automatic dataset profiling, chart recommendations, natural-language insights, and Markdown reports.

The first implementation deliberately avoids arbitrary code execution. Instead, analysis actions go through a small set of whitelisted tools: profiler, read-only DuckDB query engine, chart generator, and report writer. This keeps the MVP safer and easier to evaluate than a free-form Python execution agent.

## MVP Scope

- Upload one or more CSV/XLSX/XLS files up to 10MB each.
- **[NEW] Support for Excel files with multiple sheets and multi-file aggregation** - automatically detect relationships between sheets and across uploaded files.
- Profile rows, columns, dtypes, missing values, numeric stats, and top categorical values.
- Generate up to 3 recommended Plotly charts.
- Answer focused grouped-metric questions such as "doanh thu theo vùng" with a grouped summary and matching bar chart.
- Generate a natural-language insight summary.
- Ask simple dataset questions through read-only DuckDB SQL generation.
- Export Markdown report.
- Serve a no-build HTML demo UI from FastAPI at `/`.
- Include a React/Vite frontend scaffold for the polished UI track.

## Architecture

```mermaid
flowchart LR
    User["User"] --> UI["HTML UI / React UI"]
    UI --> API["FastAPI API"]
    API --> Store["Session Store + Uploaded File"]
    API --> Profiler["Profile Tool"]
    API --> Query["DuckDB Query Tool"]
    API --> Charts["Plotly Chart Tool"]
    API --> Reports["Markdown Report Tool"]
    Profiler --> Insight["Insight Synthesizer"]
    Query --> Insight
    Charts --> Insight
    Reports --> Download["Report Download"]
```

## Safety Model

```mermaid
flowchart TD
    Request["User request"] --> Router["Tool router"]
    Router --> Whitelist{"Allowed tool?"}
    Whitelist -->|No| Block["Block request"]
    Whitelist -->|Yes| SQL{"DuckDB query?"}
    SQL -->|Yes| SelectOnly["Require SELECT only"]
    SelectOnly --> Forbidden{"Forbidden SQL pattern?"}
    Forbidden -->|Yes| Block
    Forbidden -->|No| Execute["Execute on in-memory DuckDB"]
    SQL -->|No| Tool["Run deterministic tool"]
    Execute --> Result["Return bounded result"]
    Tool --> Result
```

## API

### Health

```http
GET /api/health
```

### Upload

```http
POST /api/upload
Content-Type: multipart/form-data
```

This endpoint now supports uploading one or more files in the same request.
It can aggregate across multiple Excel files and all sheets within each uploaded file.

Returns `session_id`, dataset profile, filenames, sheet identifiers, and sheet structure context.

### Analyze

```http
POST /api/analyze
Content-Type: application/json

{
  "session_id": "...",
  "question": "Optional business question"
}
```

Returns insight text, profile, Plotly chart specs, report id, and guardrail notes.

### Chat

```http
POST /api/chat
Content-Type: application/json

{
  "session_id": "...",
  "question": "tổng sales là bao nhiêu?"
}
```

The MVP maps common questions to safe read-only DuckDB queries.

## Multi-Sheet Support

For Excel files with multiple sheets, the system automatically:

1. **Detects relationships** between sheets (based on common columns)
2. **Analyzes sheet structure** (identifies join keys, parent-child relationships)
3. **Merges related data** intelligently for comprehensive analysis

### New API Endpoints

#### Get Sheets Information
```http
GET /api/sheets/{session_id}
```

Response includes:
- List of all sheets with metadata (rows, columns, column names)
- Detected relationships and join keys
- Similarity scores between sheets

Example:
```json
{
  "sheets": [
    {
      "name": "Products",
      "rows": 100,
      "columns": 5,
      "column_names": ["product_id", "name", "category", ...]
    },
    {
      "name": "Sales",
      "rows": 500,
      "columns": 4,
      "column_names": ["product_id", "sales", "revenue", ...]
    }
  ],
  "relationships": [
    {
      "sheet1": "Products",
      "sheet2": "Sales",
      "join_key": "product_id",
      "relationship_type": "parent_child"
    }
  ]
}
```

#### Merge Multiple Sheets
```http
POST /api/merge-sheets
Content-Type: application/json

{
  "session_id": "...",
  "sheet_names": ["Products", "Sales"],
  "join_key": "product_id"
}
```

This creates a merged dataset that can be used for subsequent analysis and reporting.

## Run Backend

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
uvicorn backend.app.main:app --reload --port 8000
```

Open:

```text
http://localhost:8000
```

Swagger docs:

```text
http://localhost:8000/docs
```

## React Frontend Track

Node was not available in the initial local environment, so the source is scaffolded but not installed yet.

```bash
cd frontend
npm install
npm run dev
```

Vite proxies `/api` to `http://localhost:8000`.

## Verification

```bash
python -m compileall backend scripts
python scripts/smoke_test.py
```

## Roadmap

- Add LLM provider abstraction for insight synthesis.
- Add benchmark datasets and 20 evaluation questions.
- Expand question-aware analysis intents beyond grouped metrics.
- Add chart success-rate and latency metrics.
- Add PII detection and masking option.
- Add async job queue for larger files.
- Add PDF export after Markdown/HTML report stabilizes.

## CV Positioning

This project is designed to demonstrate:

- practical AI product thinking;
- safe tool-calling architecture;
- data profiling and analytics automation;
- backend API design with FastAPI;
- chart/report generation;
- evaluation-ready AI engineering workflow.
