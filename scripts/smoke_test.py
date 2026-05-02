from __future__ import annotations

import tempfile
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.services.charts import generate_recommended_charts
from backend.app.services.insights import generate_insights
from backend.app.services.profiler import build_profile
from backend.app.services.query_engine import run_readonly_query, simple_question_to_sql
from backend.app.services.reports import write_markdown_report


def main() -> None:
    df = pd.DataFrame(
        {
            "category": ["A", "A", "B", "C", "C", "C"],
            "region": ["North", "South", "North", "West", "West", "East"],
            "sales": [100, 120, 90, 200, 210, 190],
            "profit": [20, 25, 15, 60, 65, 55],
        }
    )

    profile = build_profile(df)
    assert profile["rows"] == 6
    assert profile["columns"] == 4
    assert "sales" in profile["numeric_summary"]

    sql = simple_question_to_sql("tổng sales là bao nhiêu?", df)
    rows = run_readonly_query(df, sql)
    assert rows[0]["sum_sales"] == 910

    sql = simple_question_to_sql("doanh thu theo vùng", df)
    rows = run_readonly_query(df, sql)
    assert rows[0]["region"] == "West"
    assert rows[0]["total_sales"] == 410

    charts = generate_recommended_charts(df)
    assert len(charts) >= 2

    answer = generate_insights(df, profile, "doanh thu theo vùng")
    assert "Doanh thu theo vùng" in answer
    assert "West" in answer

    with tempfile.TemporaryDirectory():
        report_id, path = write_markdown_report(answer, profile, charts)
        assert report_id
        assert Path(path).exists()

    print("Smoke test passed.")


if __name__ == "__main__":
    main()
