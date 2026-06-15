"""
scripts/benchmark.py — Data Analysis AI Agent — Benchmark Suite
================================================================
20 test cases covering:
  - Profiling accuracy (column types, null detection)
  - Simple aggregations (sum, mean, count)
  - Grouped queries
  - Multi-sheet join detection
  - Chart generation
  - Insight quality (keyword checks)
  - Edge cases (empty data, all-null columns, single row)

Usage:
  cd data-analysis-ai-agent
  python scripts/benchmark.py

  # Verbose output (show full answers)
  python scripts/benchmark.py --verbose

  # Save results to JSON
  python scripts/benchmark.py --output reports/benchmark_results.json

  # Run specific test categories
  python scripts/benchmark.py --categories profiling aggregation

No external services needed — all tests run fully offline (no HF_TOKEN required).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.services.charts import generate_question_charts, generate_recommended_charts
from backend.app.services.insights import generate_insights
from backend.app.services.multi_sheet_analyzer import MultiSheetAnalyzer
from backend.app.services.profiler import build_profile
from backend.app.services.query_engine import run_readonly_query, simple_question_to_sql


# ── Test result model ─────────────────────────────────────────────────────────

@dataclass
class TestResult:
    test_id: str
    category: str
    description: str
    passed: bool
    latency_ms: float
    error: str = ""
    detail: str = ""


# ── Dataset fixtures ──────────────────────────────────────────────────────────

def _make_sales_df() -> pd.DataFrame:
    """Standard sales dataset used across most tests."""
    return pd.DataFrame({
        "category":   ["Electronics", "Electronics", "Clothing", "Clothing", "Food", "Food", "Food"],
        "region":     ["North", "South", "North", "West",  "West",  "West",  "East"],
        "sales":      [1500.0,  1200.0,  800.0,   950.0,   300.0,   320.0,   280.0],
        "profit":     [300.0,   250.0,   120.0,   140.0,    45.0,    48.0,    42.0],
        "quantity":   [10,      8,       15,      12,       30,      28,      25],
        "month":      ["Jan",   "Feb",   "Jan",   "Feb",    "Jan",   "Feb",   "Mar"],
    })


def _make_df_with_nulls() -> pd.DataFrame:
    """Dataset with controlled null values for null-detection tests."""
    return pd.DataFrame({
        "id":       [1, 2, 3, 4, 5],
        "name":     ["Alice", None, "Charlie", None, "Eve"],
        "score":    [85.0, 92.0, None, 78.0, None],
        "active":   [True, False, True, None, True],
    })


def _make_all_null_df() -> pd.DataFrame:
    """Edge case: a column that is entirely null."""
    return pd.DataFrame({
        "id":       [1, 2, 3],
        "value":    [None, None, None],
        "label":    ["a", "b", "c"],
    })


def _make_empty_df() -> pd.DataFrame:
    """Edge case: zero rows."""
    return pd.DataFrame(columns=["id", "sales", "region"])


def _make_single_row_df() -> pd.DataFrame:
    return pd.DataFrame({"product": ["Widget A"], "revenue": [9999.99], "units": [1]})


def _make_date_df() -> pd.DataFrame:
    """Dataset with datetime column for time-series chart test."""
    dates = pd.date_range("2024-01-01", periods=12, freq="ME")
    return pd.DataFrame({
        "date":    dates,
        "revenue": [100, 120, 115, 130, 145, 160, 155, 170, 180, 195, 210, 225],
        "costs":   [70,  75,  72,  80,  88,  95,  90,  100, 105, 112, 118, 125],
    })


def _make_sheet_a() -> pd.DataFrame:
    return pd.DataFrame({
        "order_id":    [101, 102, 103, 104],
        "customer_id": [1,   2,   1,   3],
        "amount":      [250, 400, 310, 180],
    })


def _make_sheet_b() -> pd.DataFrame:
    return pd.DataFrame({
        "customer_id": [1,       2,       3],
        "name":        ["Alice", "Bob",   "Carol"],
        "city":        ["HCM",   "Hanoi", "DaNang"],
    })


# ── Test runner ───────────────────────────────────────────────────────────────

def run_test(
    test_id: str,
    category: str,
    description: str,
    fn: Callable[[], bool | tuple[bool, str]],
) -> TestResult:
    t0 = time.perf_counter()
    try:
        result = fn()
        if isinstance(result, tuple):
            passed, detail = result
        else:
            passed, detail = result, ""
        latency_ms = (time.perf_counter() - t0) * 1000
        return TestResult(
            test_id=test_id,
            category=category,
            description=description,
            passed=passed,
            latency_ms=latency_ms,
            detail=detail,
        )
    except Exception as exc:
        latency_ms = (time.perf_counter() - t0) * 1000
        return TestResult(
            test_id=test_id,
            category=category,
            description=description,
            passed=False,
            latency_ms=latency_ms,
            error=f"{type(exc).__name__}: {exc}",
            detail=traceback.format_exc(limit=3),
        )


# ── Test definitions ──────────────────────────────────────────────────────────

def _all_tests() -> list[tuple]:
    """Return list of (test_id, category, description, callable)."""

    tests = []

    # ── CATEGORY: profiling ───────────────────────────────────────────────────

    def t01_profile_row_col_count():
        df = _make_sales_df()
        p = build_profile(df)
        assert p["rows"] == 7, f"Expected 7 rows, got {p['rows']}"
        assert p["columns"] == 6, f"Expected 6 cols, got {p['columns']}"
        return True

    tests.append(("T01", "profiling", "Profile: row/column count accuracy", t01_profile_row_col_count))

    def t02_profile_numeric_cols():
        df = _make_sales_df()
        p = build_profile(df)
        assert "sales" in p["numeric_summary"], "sales not in numeric_summary"
        assert "profit" in p["numeric_summary"], "profit not in numeric_summary"
        assert "quantity" in p["numeric_summary"], "quantity not in numeric_summary"
        # sales mean should be ~= 764.28
        sales_mean = p["numeric_summary"]["sales"]["mean"]
        assert abs(sales_mean - 764.2857) < 1.0, f"Sales mean wrong: {sales_mean}"
        return True

    tests.append(("T02", "profiling", "Profile: numeric column stats accuracy", t02_profile_numeric_cols))

    def t03_profile_categorical_cols():
        df = _make_sales_df()
        p = build_profile(df)
        assert "category" in p["categorical_summary"], "category not in categorical_summary"
        assert "region" in p["categorical_summary"], "region not in categorical_summary"
        # Food should appear 3 times
        cat_counts = {item["value"]: item["count"] for item in p["categorical_summary"]["category"]}
        assert cat_counts.get("Food") == 3, f"Food count wrong: {cat_counts}"
        return True

    tests.append(("T03", "profiling", "Profile: categorical top-values accuracy", t03_profile_categorical_cols))

    def t04_profile_null_detection():
        df = _make_df_with_nulls()
        p = build_profile(df)
        assert p["missing_values"]["name"] == 2, f"Expected 2 nulls in name, got {p['missing_values']['name']}"
        assert p["missing_values"]["score"] == 2, f"Expected 2 nulls in score, got {p['missing_values']['score']}"
        return True

    tests.append(("T04", "profiling", "Profile: null/missing value detection", t04_profile_null_detection))

    def t05_profile_column_types():
        df = _make_sales_df()
        p = build_profile(df)
        types = p["column_types"]
        assert "float64" in types["sales"], f"sales type wrong: {types['sales']}"
        assert "int64" in types["quantity"], f"quantity type wrong: {types['quantity']}"
        assert "object" in types["category"], f"category type wrong: {types['category']}"
        return True

    tests.append(("T05", "profiling", "Profile: column dtype classification", t05_profile_column_types))

    # ── CATEGORY: aggregation ─────────────────────────────────────────────────

    def t06_sum_query():
        df = _make_sales_df()
        sql = simple_question_to_sql("tổng sales là bao nhiêu?", df)
        rows = run_readonly_query(df, sql)
        assert len(rows) == 1, f"Expected 1 row, got {len(rows)}"
        # total sales = 1500+1200+800+950+300+320+280 = 5350
        total = rows[0].get("sum_sales") or list(rows[0].values())[0]
        assert abs(float(total) - 5350.0) < 0.01, f"Sum wrong: {total}"
        return True, f"SUM = {rows[0]}"

    tests.append(("T06", "aggregation", "Simple aggregation: SUM query", t06_sum_query))

    def t07_count_query():
        df = _make_sales_df()
        sql = simple_question_to_sql("bao nhiêu dòng?", df)
        rows = run_readonly_query(df, sql)
        count_val = list(rows[0].values())[0]
        assert int(count_val) == 7, f"Count wrong: {count_val}"
        return True, f"COUNT = {count_val}"

    tests.append(("T07", "aggregation", "Simple aggregation: COUNT rows", t07_count_query))

    def t08_mean_query():
        df = _make_sales_df()
        sql = simple_question_to_sql("trung bình profit", df)
        rows = run_readonly_query(df, sql)
        assert rows, "No result returned"
        val = list(rows[0].values())[0]
        # mean profit = (300+250+120+140+45+48+42)/7 = 135
        assert abs(float(val) - 135.0) < 1.0, f"Mean wrong: {val}"
        return True, f"AVG profit = {val}"

    tests.append(("T08", "aggregation", "Simple aggregation: AVG query", t08_mean_query))

    # ── CATEGORY: grouped ─────────────────────────────────────────────────────

    def t09_grouped_sum_by_category():
        df = _make_sales_df()
        sql = simple_question_to_sql("doanh thu theo category", df)
        rows = run_readonly_query(df, sql)
        assert len(rows) >= 3, f"Expected >=3 groups, got {len(rows)}"
        # Top group should be Electronics: 1500+1200=2700
        top_row = rows[0]
        cat_col = next((k for k in top_row if "category" in k.lower()), None)
        assert cat_col, f"No category column found in {top_row}"
        assert top_row[cat_col] == "Electronics", f"Top category wrong: {top_row[cat_col]}"
        return True, f"Top: {rows[0]}"

    tests.append(("T09", "grouped", "Grouped query: sales by category (top group correct)", t09_grouped_sum_by_category))

    def t10_grouped_sum_by_region():
        df = _make_sales_df()
        sql = simple_question_to_sql("doanh thu theo vùng", df)
        rows = run_readonly_query(df, sql)
        assert rows, "No result"
        # North total: Electronics_North(1500) + Clothing_North(800) = 2300 — actual top.
        # (West = 950+300+320 = 1570; a previous version had wrong data assumptions.)
        top_row = rows[0]
        region_col = next((k for k in top_row if "region" in k.lower()), None)
        assert region_col, f"No region column in {top_row}"
        assert top_row[region_col] == "North", f"Top region wrong: {top_row[region_col]}"
        return True, f"Top region: {rows[0]}"

    tests.append(("T10", "grouped", "Grouped query: sales by region (correct leader)", t10_grouped_sum_by_region))

    def t11_grouped_result_ordering():
        """Results should be DESC ordered (highest first)."""
        df = _make_sales_df()
        sql = simple_question_to_sql("doanh thu theo category", df)
        rows = run_readonly_query(df, sql)
        metric_vals = []
        for row in rows:
            for k, v in row.items():
                if k.startswith("total_") or k.startswith("sum_"):
                    metric_vals.append(float(v))
        assert metric_vals == sorted(metric_vals, reverse=True), \
            f"Results not sorted DESC: {metric_vals}"
        return True, f"Order: {metric_vals}"

    tests.append(("T11", "grouped", "Grouped query: results ordered DESC by metric", t11_grouped_result_ordering))

    # ── CATEGORY: multi_sheet ─────────────────────────────────────────────────

    def t12_detect_join_key():
        """MultiSheetAnalyzer should detect customer_id as the join key."""
        sheet_a = _make_sheet_a()
        sheet_b = _make_sheet_b()
        sheets = {"orders": sheet_a, "customers": sheet_b}
        relationships = MultiSheetAnalyzer.detect_relationships(sheets)
        assert relationships, "No relationships detected"
        rel = relationships[0]
        assert rel.join_key == "customer_id", \
            f"Expected join_key=customer_id, got {rel.join_key}"
        return True, f"Join key: {rel.join_key}, score: {rel.similarity_score:.2f}"

    tests.append(("T12", "multi_sheet", "Multi-sheet: detect join key between orders/customers", t12_detect_join_key))

    def t13_relationship_type():
        """Relationship should be parent_child or related (not independent)."""
        sheet_a = _make_sheet_a()
        sheet_b = _make_sheet_b()
        sheets = {"orders": sheet_a, "customers": sheet_b}
        relationships = MultiSheetAnalyzer.detect_relationships(sheets)
        assert relationships, "No relationships detected"
        rel = relationships[0]
        assert rel.relationship_type != "independent", \
            f"Relationship type should not be independent: {rel.relationship_type}"
        return True, f"Type: {rel.relationship_type}"

    tests.append(("T13", "multi_sheet", "Multi-sheet: relationship type not 'independent'", t13_relationship_type))

    def t14_similarity_score():
        """Similarity score for customer_id join should be > 0.3."""
        sheet_a = _make_sheet_a()
        sheet_b = _make_sheet_b()
        sheets = {"orders": sheet_a, "customers": sheet_b}
        relationships = MultiSheetAnalyzer.detect_relationships(sheets)
        assert relationships, "No relationships detected"
        score = relationships[0].similarity_score
        assert score > 0.3, f"Similarity score too low: {score}"
        return True, f"Score: {score:.3f}"

    tests.append(("T14", "multi_sheet", "Multi-sheet: similarity score > 0.3 for shared key", t14_similarity_score))

    # ── CATEGORY: charts ──────────────────────────────────────────────────────

    def t15_recommended_charts_count():
        df = _make_sales_df()
        charts = generate_recommended_charts(df)
        assert len(charts) >= 2, f"Expected >=2 charts, got {len(charts)}"
        return True, f"Generated {len(charts)} charts"

    tests.append(("T15", "charts", "Chart generation: >=2 recommended charts for sales data", t15_recommended_charts_count))

    def t16_question_chart():
        df = _make_sales_df()
        charts = generate_question_charts(df, "doanh thu theo category")
        assert charts, "No chart generated for grouped question"
        # chart dict uses 'chart_type' key (not 'type') and 'plotly_json' for payload
        assert "chart_type" in charts[0], f"Chart missing 'chart_type' key. Keys: {list(charts[0].keys())}"
        assert "plotly_json" in charts[0] or "figure" in charts[0] or "data" in charts[0], \
            f"Chart missing plotly_json/figure/data: {list(charts[0].keys())}"
        assert charts[0]["chart_type"] == "bar", f"Expected bar chart, got: {charts[0]['chart_type']}"
        return True, f"Chart type: {charts[0].get('chart_type')}"

    tests.append(("T16", "charts", "Chart generation: grouped question produces bar chart", t16_question_chart))

    def t17_timeseries_chart():
        """Date column triggers a time-series chart (line/area/scatter)."""
        df = _make_date_df()
        charts = generate_recommended_charts(df)
        # chart dicts use 'chart_type' key (not 'type')
        types = [c.get("chart_type") for c in charts]
        assert any(t in ("line", "area", "scatter", "timeseries") for t in types), \
            f"Expected a time-series chart type in {types}. Chart keys: {[list(c.keys()) for c in charts[:2]]}"
        return True, f"Chart types: {types}"

    tests.append(("T17", "charts", "Chart generation: datetime column triggers time-series chart", t17_timeseries_chart))

    # ── CATEGORY: insights ────────────────────────────────────────────────────

    def t18_insight_mentions_top_group():
        """Insight about category sales should mention Electronics."""
        df = _make_sales_df()
        p = build_profile(df)
        answer = generate_insights(df, p, "doanh thu theo category")
        assert "Electronics" in answer, \
            f"Insight missing top category 'Electronics'. Got: {answer[:300]}"
        return True, f"Answer preview: {answer[:150]}"

    tests.append(("T18", "insights", "Insight quality: top category mentioned in answer", t18_insight_mentions_top_group))

    def t19_insight_contains_numbers():
        """Insight should include at least one numeric value."""
        import re
        df = _make_sales_df()
        p = build_profile(df)
        answer = generate_insights(df, p, "tổng doanh thu")
        has_number = bool(re.search(r"\d[\d,\.]*", answer))
        assert has_number, f"Insight contains no numbers: {answer[:300]}"
        return True, f"Answer preview: {answer[:150]}"

    tests.append(("T19", "insights", "Insight quality: answer contains numeric values", t19_insight_contains_numbers))

    # ── CATEGORY: edge_cases ─────────────────────────────────────────────────

    def t20_empty_dataframe_profile():
        """build_profile on empty DataFrame should not crash."""
        df = _make_empty_df()
        p = build_profile(df)
        assert p["rows"] == 0, f"Expected 0 rows, got {p['rows']}"
        assert p["columns"] == 3, f"Expected 3 cols, got {p['columns']}"
        # No nulls to count (no rows) — missing_values should all be 0
        for col, count in p["missing_values"].items():
            assert count == 0, f"Unexpected null count for {col}: {count}"
        return True

    tests.append(("T20", "edge_cases", "Edge case: empty DataFrame profile (0 rows, no crash)", t20_empty_dataframe_profile))

    return tests


# ── Reporter ──────────────────────────────────────────────────────────────────

def _print_results(results: list[TestResult], verbose: bool = False) -> None:
    passed = sum(1 for r in results if r.passed)
    total = len(results)

    print("\n" + "=" * 72)
    print("  Data Analysis AI Agent — Benchmark Results")
    print("=" * 72)
    print(f"  {'ID':<5} {'Category':<14} {'Pass':<6} {'ms':>6}  Description")
    print("-" * 72)

    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"  {r.test_id:<5} {r.category:<14} {status:<6} {r.latency_ms:>5.0f}  {r.description}")
        if not r.passed:
            print(f"         ERROR: {r.error}")
        if verbose and r.detail:
            print(f"         DETAIL: {r.detail[:120]}")

    print("-" * 72)
    print(f"  Result: {passed}/{total} passed", end="")
    if passed == total:
        print("  — ALL PASS")
    else:
        print(f"  — {total - passed} FAILED")

    # Category breakdown
    categories: dict[str, list[bool]] = {}
    for r in results:
        categories.setdefault(r.category, []).append(r.passed)

    print("\n  By category:")
    for cat, statuses in sorted(categories.items()):
        p = sum(statuses)
        t = len(statuses)
        bar = "#" * p + "." * (t - p)
        print(f"    {cat:<14} [{bar}] {p}/{t}")

    # Latency
    latencies = [r.latency_ms for r in results]
    import statistics
    print(f"\n  Latency: mean={statistics.mean(latencies):.0f}ms  "
          f"p95={sorted(latencies)[int(len(latencies)*0.95)]:.0f}ms  "
          f"max={max(latencies):.0f}ms")
    print("=" * 72 + "\n")


def _save_results(results: list[TestResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "summary": {
            "total": len(results),
            "passed": sum(1 for r in results if r.passed),
            "failed": sum(1 for r in results if not r.passed),
        },
        "results": [
            {
                "id": r.test_id,
                "category": r.category,
                "description": r.description,
                "passed": r.passed,
                "latency_ms": round(r.latency_ms, 1),
                "error": r.error,
            }
            for r in results
        ],
    }
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Results saved → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Data Analysis AI Agent — benchmark suite (20 test cases)"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detail output per test")
    parser.add_argument("--output", type=Path, default=None, help="Save results as JSON")
    parser.add_argument(
        "--categories",
        nargs="+",
        choices=["profiling", "aggregation", "grouped", "multi_sheet", "charts", "insights", "edge_cases"],
        default=None,
        help="Run only specific categories (default: all)",
    )
    args = parser.parse_args()

    all_tests = _all_tests()

    # Filter by category if requested
    if args.categories:
        all_tests = [(tid, cat, desc, fn) for tid, cat, desc, fn in all_tests if cat in args.categories]

    print(f"Running {len(all_tests)} benchmark tests...")
    results = []
    for test_id, category, description, fn in all_tests:
        print(f"  {test_id}: {description[:55]}...", end=" ", flush=True)
        result = run_test(test_id, category, description, fn)
        status = "OK" if result.passed else "FAIL"
        print(f"{status} ({result.latency_ms:.0f}ms)")
        results.append(result)

    _print_results(results, verbose=args.verbose)

    if args.output:
        _save_results(results, args.output)

    failed = sum(1 for r in results if not r.passed)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
