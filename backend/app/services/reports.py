from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from jinja2 import Template

from backend.app.services.storage import REPORT_DIR


REPORT_TEMPLATE = """# Data Analysis Report

## Summary

{{ answer }}

## Dataset Profile

- Rows: {{ profile.rows }}
- Columns: {{ profile.columns }}

## Column Types

{% for name, dtype in profile.column_types.items() -%}
- `{{ name }}`: {{ dtype }}
{% endfor %}

## Missing Values

{% for name, count in profile.missing_values.items() -%}
- `{{ name }}`: {{ count }}
{% endfor %}

## Recommended Charts

{% for chart in charts -%}
- {{ chart.title }} (`{{ chart.chart_type }}`)
{% endfor %}
"""


def write_markdown_report(answer: str, profile: dict[str, Any], charts: list[dict[str, Any]]) -> tuple[str, Path]:
    report_id = uuid.uuid4().hex
    path = REPORT_DIR / f"{report_id}.md"
    rendered = Template(REPORT_TEMPLATE).render(answer=answer, profile=profile, charts=charts)
    path.write_text(rendered, encoding="utf-8")
    return report_id, path

