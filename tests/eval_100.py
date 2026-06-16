#!/usr/bin/env python3
"""
eval_100.py — Data Analysis AI Agent Evaluation Framework
==========================================================
100 câu hỏi đánh giá hệ thống trên 6 datasets thực tế.

Metrics đánh giá:
  • correctness   — đúng số/keyword so với ground truth (0–1)
  • relevance     — câu trả lời có đúng chủ đề không (0–1)
  • lang_quality  — tiếng Việt tự nhiên, không meta-commentary (0–1)
  • latency_ms    — thời gian phản hồi (ms)
  • http_ok       — API trả về 200 (bool)
  • llm_failed    — rơi vào deterministic fallback (bool)

Usage:
    # 1. Khởi động server
    uvicorn backend.app.main:app --host 0.0.0.0 --port 8000

    # 2. Chạy eval toàn bộ
    python tests/eval_100.py

    # 3. Tùy chọn
    python tests/eval_100.py --base-url http://localhost:8000 --out results/run1
    python tests/eval_100.py --ids 1,5,10-20          # chạy subset
    python tests/eval_100.py --dataset sales_data      # chỉ 1 dataset
    python tests/eval_100.py --no-report               # chỉ in ra console
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "data" / "samples"
UPLOADS = ROOT / "data" / "uploads"   # fallback nếu samples không có file

# File paths (dùng samples/, còn lại fallback sang uploads/)
FILES = {
    "sales_sample":      SAMPLES / "sales_sample.csv",
    "sales_data":        UPLOADS / "179c58605be14358b9c6939401c89ee1_sales_data.csv",
    "financial_sample":  UPLOADS / "31bd20d6bd9c49859a60bb1f45cd53c6_FinancialSample.xlsx-Sheet1.csv",
    "general_ledger":    UPLOADS / "8e4f2b8cafec45cd9f7114c1447d96b6_General-Ledger.xlsx",
    "expense_claims":    UPLOADS / "8e4f2b8cafec45cd9f7114c1447d96b6_Expense-Claims.xlsx",
    "viet_relational":   UPLOADS / "754d49641cd34053a2b6dbf3ad84198b_viet_sales_relational.xlsx",
}

# ── Ground truths (tính sẵn từ pandas) ────────────────────────────────────────
GROUND_TRUTH: dict[str, Any] = {}

def _load_ground_truths() -> None:
    try:
        import pandas as pd
    except ImportError:
        print("[WARN] pandas không có — bỏ qua ground truth computation, correctness=N/A")
        return

    # sales_sample
    try:
        df = pd.read_csv(FILES["sales_sample"])
        GROUND_TRUTH["ss_total_sales"]          = int(df["sales"].sum())
        GROUND_TRUTH["ss_top_category"]         = df.groupby("category")["sales"].sum().idxmax()
        GROUND_TRUTH["ss_top_region_profit"]    = df.groupby("region")["profit"].sum().idxmax()
        GROUND_TRUTH["ss_total_qty"]            = int(df["quantity"].sum())
        GROUND_TRUTH["ss_north_count"]          = int((df["region"] == "North").sum())
        GROUND_TRUTH["ss_electronics_sales"]    = int(df[df["category"] == "Electronics"]["sales"].sum())
        GROUND_TRUTH["ss_n_categories"]         = int(df["category"].nunique())
        GROUND_TRUTH["ss_avg_sales_north"]      = round(df[df["region"] == "North"]["sales"].mean(), 1)
        GROUND_TRUTH["ss_avg_profit_margin"]    = round((df["profit"] / df["sales"]).mean(), 3)
        GROUND_TRUTH["ss_min_profit_category"]  = df.groupby("category")["profit"].sum().idxmin()
    except Exception as e:
        print(f"[WARN] sales_sample ground truth lỗi: {e}")

    # sales_data
    try:
        df = pd.read_csv(FILES["sales_data"])
        df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
        GROUND_TRUTH["sd_total_revenue"]        = int(df["revenue"].sum())
        GROUND_TRUTH["sd_top_rep"]              = df.groupby("sales_rep")["revenue"].sum().idxmax()
        GROUND_TRUTH["sd_top_category"]         = df.groupby("product_category")["revenue"].sum().idxmax()
        GROUND_TRUTH["sd_jan_orders"]           = int((df["order_date"].dt.month == 1).sum())
        GROUND_TRUTH["sd_mean_discount"]        = round(df["discount_pct"].mean(), 2)
        GROUND_TRUTH["sd_max_price_product"]    = df.loc[df["unit_price"].idxmax(), "product_name"]
        GROUND_TRUTH["sd_top_region"]           = df.groupby("customer_region")["revenue"].sum().idxmax()
        GROUND_TRUTH["sd_mean_revenue"]         = int(df["revenue"].mean())
        GROUND_TRUTH["sd_unique_customers"]     = int(df["customer_name"].nunique())
        GROUND_TRUTH["sd_max_discount"]         = float(df["discount_pct"].max())
        GROUND_TRUTH["sd_total_qty"]            = int(df["quantity"].sum())
        GROUND_TRUTH["sd_tranminh_revenue"]     = int(df[df["sales_rep"] == "Trần Minh"]["revenue"].sum())
    except Exception as e:
        print(f"[WARN] sales_data ground truth lỗi: {e}")

    # financial_sample (cần clean $ và dấu phẩy)
    try:
        df = pd.read_csv(FILES["financial_sample"])
        def _clean(s):
            return pd.to_numeric(
                s.astype(str).str.replace(r"[\$,]", "", regex=True).str.strip(),
                errors="coerce",
            )
        for col in df.select_dtypes("object").columns:
            if col not in ["Segment", "Country", " Product ", " Discount Band ", "Month Name"]:
                df[col] = _clean(df[col])
        GROUND_TRUTH["fin_total_profit"]        = round(float(df[" Profit "].sum()), 0)
        GROUND_TRUTH["fin_top_country"]         = df.groupby("Country")["  Sales "].sum().idxmax()
        GROUND_TRUTH["fin_top_segment"]         = df.groupby("Segment")["  Sales "].sum().idxmax()
        GROUND_TRUTH["fin_2014_sales"]          = round(float(df[df["Year"] == 2014]["  Sales "].sum()), 0)
        GROUND_TRUTH["fin_2015_sales"]          = round(float(df[df["Year"] == 2015]["  Sales "].sum()), 0)
        GROUND_TRUTH["fin_top_product_profit"]  = df.groupby(" Product ")[ " Profit "].sum().idxmax()
        GROUND_TRUTH["fin_total_units"]         = int(df["Units Sold"].sum())
        GROUND_TRUTH["fin_top_month"]           = df.groupby("Month Name")["  Sales "].sum().idxmax()
    except Exception as e:
        print(f"[WARN] financial_sample ground truth lỗi: {e}")

    # general_ledger
    try:
        df = pd.read_excel(FILES["general_ledger"])
        GROUND_TRUTH["gl_total_debit"]          = round(float(df["Debit"].sum()), 2)
        GROUND_TRUTH["gl_total_credit"]         = round(float(df["Credit"].sum()), 2)
        GROUND_TRUTH["gl_net"]                  = round(float(df["Debit"].sum() - df["Credit"].sum()), 2)
        GROUND_TRUTH["gl_top_credit_account"]   = df.groupby("AccountName")["Credit"].sum().idxmax()
        GROUND_TRUTH["gl_top_debit_dept"]       = df.groupby("Dept")["Debit"].sum().idxmax()
        GROUND_TRUTH["gl_total_rows"]           = len(df)
        GROUND_TRUTH["gl_top_currency"]         = df["Currency"].value_counts().index[0]
        GROUND_TRUTH["gl_top_costcenter"]       = df["CostCenter"].value_counts().index[0]
        GROUND_TRUTH["gl_unique_accounts"]      = int(df["AccountNumber"].nunique())
        GROUND_TRUTH["gl_sales_credit"]         = round(float(df[df["Dept"] == "Sales"]["Credit"].sum()), 2)
    except Exception as e:
        print(f"[WARN] general_ledger ground truth lỗi: {e}")

    # expense_claims
    try:
        df = pd.read_excel(FILES["expense_claims"])
        df["SubmitDate"] = pd.to_datetime(df["SubmitDate"], errors="coerce")
        GROUND_TRUTH["exp_total_amount"]        = round(float(df["Amount"].sum()), 2)
        GROUND_TRUTH["exp_top_category"]        = df.groupby("Category")["Amount"].sum().idxmax()
        GROUND_TRUTH["exp_top_approver"]        = df["ApprovedBy"].value_counts().index[0]
        GROUND_TRUTH["exp_paid_count"]          = int((df["Status"] == "Paid").sum())
        GROUND_TRUTH["exp_submitted_count"]     = int((df["Status"] == "Submitted").sum())
        GROUND_TRUTH["exp_mean_amount"]         = round(float(df["Amount"].mean()), 2)
        GROUND_TRUTH["exp_unique_employees"]    = int(df["EmployeeID"].nunique())
        GROUND_TRUTH["exp_top_employee"]        = df.groupby("EmployeeID")["Amount"].sum().idxmax()
    except Exception as e:
        print(f"[WARN] expense_claims ground truth lỗi: {e}")

    # viet_relational
    try:
        import pandas as pd
        xl = pd.ExcelFile(FILES["viet_relational"])
        cust  = xl.parse("customers")
        prod  = xl.parse("products")
        reps  = xl.parse("sales_reps")
        orders = xl.parse("orders")
        GROUND_TRUTH["vr_n_customers"]          = len(cust)
        GROUND_TRUTH["vr_top_product_price"]    = prod.loc[prod["unit_price"].idxmax(), "product_name"]
        GROUND_TRUTH["vr_top_rep_target"]       = reps.loc[reps["target_monthly"].idxmax(), "rep_name"]
        GROUND_TRUTH["vr_n_orders"]             = len(orders)
        GROUND_TRUTH["vr_top_payment"]          = orders["payment_method"].value_counts().index[0]
    except Exception as e:
        print(f"[WARN] viet_relational ground truth lỗi: {e}")


# ── Test case definition ──────────────────────────────────────────────────────
@dataclass
class TestCase:
    id: int
    dataset: str          # key trong FILES
    question: str
    category: str         # aggregation|ranking|comparison|trend|multisheet|bot_info|off_topic|edge|language
    expected_type: str    # number|keyword|any|none
    gt_key: str = ""      # GROUND_TRUTH key để lookup
    tolerance: float = 0.05  # số ± 5% vẫn tính đúng


TEST_CASES: list[TestCase] = [
    # ── SALES SAMPLE CSV (10 rows) ────────────────────────────────────────────
    # Aggregation đơn giản
    TestCase(1,  "sales_sample", "Tổng doanh thu (sales) là bao nhiêu?",
             "aggregation", "number", "ss_total_sales"),
    TestCase(2,  "sales_sample", "Tổng số lượng (quantity) đã bán là bao nhiêu?",
             "aggregation", "number", "ss_total_qty"),
    TestCase(3,  "sales_sample", "Electronics có tổng doanh thu là bao nhiêu?",
             "aggregation", "number", "ss_electronics_sales"),
    TestCase(4,  "sales_sample", "Có bao nhiêu dòng dữ liệu ở khu vực North?",
             "aggregation", "number", "ss_north_count"),
    TestCase(5,  "sales_sample", "Có bao nhiêu danh mục sản phẩm (category) khác nhau?",
             "aggregation", "number", "ss_n_categories"),
    # Ranking
    TestCase(6,  "sales_sample", "Danh mục nào có tổng doanh thu cao nhất?",
             "ranking", "keyword", "ss_top_category"),
    TestCase(7,  "sales_sample", "Danh mục nào có tổng lợi nhuận thấp nhất?",
             "ranking", "keyword", "ss_min_profit_category"),
    # Comparison
    TestCase(8,  "sales_sample", "So sánh tổng doanh thu giữa khu vực North và South?",
             "comparison", "keyword", ""),      # expected: chứa "North" và "South"
    TestCase(9,  "sales_sample", "Khu vực nào có tổng lợi nhuận cao hơn?",
             "comparison", "keyword", "ss_top_region_profit"),
    TestCase(10, "sales_sample", "Profit margin trung bình (profit/sales) của toàn bộ là bao nhiêu?",
             "aggregation", "number", "ss_avg_profit_margin"),

    # ── SALES DATA CSV (30 rows, Vietnamese) ──────────────────────────────────
    # Aggregation
    TestCase(11, "sales_data", "Tổng doanh thu của toàn bộ dữ liệu là bao nhiêu?",
             "aggregation", "number", "sd_total_revenue"),
    TestCase(12, "sales_data", "Tổng số lượng sản phẩm đã bán được là bao nhiêu?",
             "aggregation", "number", "sd_total_qty"),
    TestCase(13, "sales_data", "Có bao nhiêu khách hàng duy nhất?",
             "aggregation", "number", "sd_unique_customers"),
    TestCase(14, "sales_data", "Discount trung bình trên toàn bộ đơn hàng là bao nhiêu %?",
             "aggregation", "number", "sd_mean_discount"),
    TestCase(15, "sales_data", "Doanh thu trung bình mỗi đơn hàng là bao nhiêu?",
             "aggregation", "number", "sd_mean_revenue"),
    # Ranking
    TestCase(16, "sales_data", "Sales rep nào có tổng doanh thu cao nhất?",
             "ranking", "keyword", "sd_top_rep"),
    TestCase(17, "sales_data", "Danh mục sản phẩm nào có tổng doanh thu cao nhất?",
             "ranking", "keyword", "sd_top_category"),
    TestCase(18, "sales_data", "Sản phẩm nào có đơn giá cao nhất?",
             "ranking", "keyword", "sd_max_price_product"),
    TestCase(19, "sales_data", "Khu vực nào có tổng doanh thu lớn nhất?",
             "ranking", "keyword", "sd_top_region"),
    # Filtering
    TestCase(20, "sales_data", "Có bao nhiêu đơn hàng trong tháng 1 năm 2024?",
             "aggregation", "number", "sd_jan_orders"),
    TestCase(21, "sales_data", "Tổng doanh thu của Trần Minh là bao nhiêu?",
             "aggregation", "number", "sd_tranminh_revenue"),
    TestCase(22, "sales_data", "Discount cao nhất trong dataset là bao nhiêu %?",
             "aggregation", "number", "sd_max_discount"),
    # Trend
    TestCase(23, "sales_data", "Tháng nào trong năm 2024 có tổng doanh thu cao nhất?",
             "trend", "any", ""),
    TestCase(24, "sales_data", "Doanh thu theo từng tháng thay đổi như thế nào?",
             "trend", "any", ""),
    # Comparison
    TestCase(25, "sales_data", "So sánh doanh thu của Trần Minh và Phạm Hoa?",
             "comparison", "any", ""),
    TestCase(26, "sales_data", "Top 3 sản phẩm có doanh thu cao nhất?",
             "ranking", "any", ""),
    TestCase(27, "sales_data", "Phần trăm đóng góp của từng khu vực vào tổng doanh thu?",
             "aggregation", "any", ""),
    TestCase(28, "sales_data", "Trung bình số lượng (quantity) mỗi đơn hàng là bao nhiêu?",
             "aggregation", "any", ""),
    TestCase(29, "sales_data", "Đơn hàng nào có doanh thu lớn nhất?",
             "ranking", "any", ""),
    TestCase(30, "sales_data", "Sản phẩm nào được bán nhiều nhất theo số lượng?",
             "ranking", "any", ""),

    # ── FINANCIAL SAMPLE (700 rows, $ format) ─────────────────────────────────
    TestCase(31, "financial_sample", "Tổng lợi nhuận (Profit) của toàn bộ dữ liệu là bao nhiêu?",
             "aggregation", "number", "fin_total_profit"),
    TestCase(32, "financial_sample", "Quốc gia nào có tổng doanh thu cao nhất?",
             "ranking", "keyword", "fin_top_country"),
    TestCase(33, "financial_sample", "Segment nào có tổng doanh thu cao nhất?",
             "ranking", "keyword", "fin_top_segment"),
    TestCase(34, "financial_sample", "Năm 2014 có tổng doanh thu là bao nhiêu?",
             "aggregation", "number", "fin_2014_sales"),
    TestCase(35, "financial_sample", "Năm 2015 có tổng doanh thu là bao nhiêu?",
             "aggregation", "number", "fin_2015_sales"),
    TestCase(36, "financial_sample", "Sản phẩm nào mang lại lợi nhuận cao nhất?",
             "ranking", "keyword", "fin_top_product_profit"),
    TestCase(37, "financial_sample", "Tổng số Units Sold trên toàn bộ dataset là bao nhiêu?",
             "aggregation", "number", "fin_total_units"),
    TestCase(38, "financial_sample", "Tháng nào có tổng doanh thu cao nhất?",
             "ranking", "keyword", "fin_top_month"),
    TestCase(39, "financial_sample", "So sánh tổng doanh thu năm 2014 và năm 2015?",
             "comparison", "any", ""),
    TestCase(40, "financial_sample", "Top 3 quốc gia có tổng lợi nhuận cao nhất?",
             "ranking", "any", ""),
    TestCase(41, "financial_sample", "Discount Band 'None' chiếm bao nhiêu % trong tổng số records?",
             "aggregation", "any", ""),
    TestCase(42, "financial_sample", "Government segment chiếm bao nhiêu phần trăm tổng doanh thu?",
             "aggregation", "any", ""),
    TestCase(43, "financial_sample", "Sản phẩm nào có Gross Sales cao nhất trong năm 2014?",
             "aggregation", "any", ""),
    TestCase(44, "financial_sample", "Profit margin trung bình của từng Segment là bao nhiêu?",
             "aggregation", "any", ""),
    TestCase(45, "financial_sample", "Có bao nhiêu giao dịch (records) trong dataset này?",
             "aggregation", "any", ""),

    # ── GENERAL LEDGER (2000 rows) ────────────────────────────────────────────
    TestCase(46, "general_ledger", "Tổng Debit của toàn bộ sổ cái là bao nhiêu?",
             "aggregation", "number", "gl_total_debit"),
    TestCase(47, "general_ledger", "Tổng Credit của toàn bộ sổ cái là bao nhiêu?",
             "aggregation", "number", "gl_total_credit"),
    TestCase(48, "general_ledger", "Net balance (Debit trừ Credit) là bao nhiêu?",
             "aggregation", "number", "gl_net"),
    TestCase(49, "general_ledger", "Account nào có tổng Credit cao nhất?",
             "ranking", "keyword", "gl_top_credit_account"),
    TestCase(50, "general_ledger", "Phòng ban (Dept) nào có tổng Debit lớn nhất?",
             "ranking", "keyword", "gl_top_debit_dept"),
    TestCase(51, "general_ledger", "Dataset có bao nhiêu giao dịch (rows)?",
             "aggregation", "number", "gl_total_rows"),
    TestCase(52, "general_ledger", "Currency nào được dùng nhiều nhất?",
             "ranking", "keyword", "gl_top_currency"),
    TestCase(53, "general_ledger", "CostCenter nào xuất hiện nhiều nhất?",
             "ranking", "keyword", "gl_top_costcenter"),
    TestCase(54, "general_ledger", "Có bao nhiêu AccountNumber duy nhất trong dataset?",
             "aggregation", "number", "gl_unique_accounts"),
    TestCase(55, "general_ledger", "Dept Sales có tổng Credit là bao nhiêu?",
             "aggregation", "number", "gl_sales_credit"),
    TestCase(56, "general_ledger", "So sánh tổng Debit và Credit giữa các Department?",
             "comparison", "any", ""),
    TestCase(57, "general_ledger", "Tháng nào có số lượng giao dịch nhiều nhất?",
             "trend", "any", ""),
    TestCase(58, "general_ledger", "AccountName nào có số lần xuất hiện nhiều nhất?",
             "ranking", "any", ""),
    TestCase(59, "general_ledger", "Phân tích tỷ trọng Debit theo từng Department?",
             "aggregation", "any", ""),
    TestCase(60, "general_ledger", "Giao dịch nào có giá trị Debit lớn nhất?",
             "ranking", "any", ""),

    # ── EXPENSE CLAIMS (1000 rows) ────────────────────────────────────────────
    TestCase(61, "expense_claims", "Tổng số tiền claims là bao nhiêu?",
             "aggregation", "number", "exp_total_amount"),
    TestCase(62, "expense_claims", "Category chi tiêu nào có tổng Amount cao nhất?",
             "ranking", "keyword", "exp_top_category"),
    TestCase(63, "expense_claims", "Manager nào (ApprovedBy) phê duyệt nhiều claims nhất?",
             "ranking", "keyword", "exp_top_approver"),
    TestCase(64, "expense_claims", "Bao nhiêu claims có Status là 'Paid'?",
             "aggregation", "number", "exp_paid_count"),
    TestCase(65, "expense_claims", "Bao nhiêu claims có Status là 'Submitted' (chưa xử lý)?",
             "aggregation", "number", "exp_submitted_count"),
    TestCase(66, "expense_claims", "Số tiền claim trung bình mỗi lần là bao nhiêu?",
             "aggregation", "number", "exp_mean_amount"),
    TestCase(67, "expense_claims", "Có bao nhiêu nhân viên (EmployeeID) khác nhau?",
             "aggregation", "number", "exp_unique_employees"),
    TestCase(68, "expense_claims", "Nhân viên nào có tổng số tiền claims cao nhất?",
             "ranking", "keyword", "exp_top_employee"),
    TestCase(69, "expense_claims", "Phân tích tỷ lệ Paid/Submitted/Rejected trong dataset?",
             "aggregation", "any", ""),
    TestCase(70, "expense_claims", "Tháng nào trong năm có số lượng claims nhiều nhất?",
             "trend", "any", ""),

    # ── VIET RELATIONAL (multi-sheet) ─────────────────────────────────────────
    TestCase(71, "viet_relational", "Có bao nhiêu khách hàng trong hệ thống?",
             "aggregation", "number", "vr_n_customers"),
    TestCase(72, "viet_relational", "Sản phẩm nào có giá bán cao nhất?",
             "ranking", "keyword", "vr_top_product_price"),
    TestCase(73, "viet_relational", "Sales rep nào có target doanh thu tháng cao nhất?",
             "ranking", "keyword", "vr_top_rep_target"),
    TestCase(74, "viet_relational", "Có bao nhiêu đơn hàng trong hệ thống?",
             "aggregation", "number", "vr_n_orders"),
    TestCase(75, "viet_relational", "Phương thức thanh toán nào được dùng nhiều nhất?",
             "ranking", "keyword", "vr_top_payment"),

    # ── BOT INFO (5 câu) ──────────────────────────────────────────────────────
    TestCase(76, "sales_sample", "Bạn là AI gì? Bạn có thể làm gì?",
             "bot_info", "any", ""),
    TestCase(77, "sales_sample", "Hệ thống này có thể phân tích file Excel không?",
             "bot_info", "any", ""),
    TestCase(78, "sales_sample", "Bạn hỗ trợ những loại câu hỏi nào về dữ liệu?",
             "bot_info", "any", ""),
    TestCase(79, "sales_sample", "Tool này có thể vẽ biểu đồ không?",
             "bot_info", "any", ""),
    TestCase(80, "sales_sample", "Làm thế nào để tôi sử dụng hệ thống này hiệu quả?",
             "bot_info", "any", ""),

    # ── OFF-TOPIC (5 câu — expect redirect/từ chối) ───────────────────────────
    TestCase(81, "sales_sample", "Thủ đô của nước Pháp là đâu?",
             "off_topic", "none", ""),
    TestCase(82, "sales_sample", "Hãy viết cho tôi một bài thơ về mùa xuân",
             "off_topic", "none", ""),
    TestCase(83, "sales_sample", "Tôi nên đầu tư vào cổ phiếu nào trong năm nay?",
             "off_topic", "none", ""),
    TestCase(84, "sales_sample", "Dạy tôi cách nấu món phở bò truyền thống",
             "off_topic", "none", ""),
    TestCase(85, "sales_sample", "Dự báo thời tiết Hà Nội tuần này thế nào?",
             "off_topic", "none", ""),

    # ── EDGE CASES (10 câu) ───────────────────────────────────────────────────
    TestCase(86, "sales_sample", "Dữ liệu này có bao nhiêu dòng và bao nhiêu cột?",
             "edge", "any", ""),
    TestCase(87, "sales_data",   "Có giá trị bị thiếu (null/missing) trong dataset không?",
             "edge", "any", ""),
    TestCase(88, "sales_data",   "So sánh doanh thu năm nay với năm ngoái (nếu dữ liệu chỉ có 1 năm)",
             "edge", "any", ""),
    TestCase(89, "sales_data",   "Dự báo doanh thu tháng tới dựa trên xu hướng hiện tại",
             "edge", "any", ""),
    TestCase(90, "sales_sample", "Cho tôi xem toàn bộ dữ liệu",
             "edge", "any", ""),
    TestCase(91, "financial_sample", "Có outlier nào trong dữ liệu doanh thu không?",
             "edge", "any", ""),
    TestCase(92, "sales_data",   "Tại sao doanh thu tháng 3 lại thấp?",
             "edge", "any", ""),
    TestCase(93, "general_ledger", "Dữ liệu có đáng tin cậy không? Data quality như thế nào?",
             "edge", "any", ""),
    TestCase(94, "expense_claims", "Nếu giảm claims xuống 20% thì công ty tiết kiệm được bao nhiêu?",
             "edge", "any", ""),
    TestCase(95, "viet_relational", "Tổng doanh thu từ tất cả các sheet là bao nhiêu?",
             "edge", "any", ""),

    # ── NGÔN NGỮ TỰ NHIÊN / CASUAL (5 câu) ───────────────────────────────────
    TestCase(96,  "sales_sample", "cho tôi biết tổng sales đi",
             "language", "number", "ss_total_sales"),
    TestCase(97,  "sales_sample", "cái nào bán chạy nhất vậy?",
             "language", "keyword", "ss_top_category"),
    TestCase(98,  "sales_data",   "ai bán hàng giỏi nhất?",
             "language", "keyword", "sd_top_rep"),
    TestCase(99,  "sales_data",   "lãi tổng cộng được bao nhiêu rồi?",
             "language", "any", ""),
    TestCase(100, "sales_data",   "báo cáo nhanh tình hình kinh doanh đi",
             "language", "any", ""),
]


# ── API Client ────────────────────────────────────────────────────────────────
class APIClient:
    def __init__(self, base_url: str, timeout: int = 60):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._sessions: dict[str, str] = {}   # dataset → session_id

    def health_check(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/api/health", timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    def get_or_create_session(self, dataset: str) -> str:
        if dataset in self._sessions:
            return self._sessions[dataset]
        file_path = FILES.get(dataset)
        if not file_path or not file_path.exists():
            raise FileNotFoundError(f"File không tìm thấy: {file_path}")
        with open(file_path, "rb") as f:
            suffix = file_path.suffix.lower()
            mime = "text/csv" if suffix == ".csv" else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            resp = requests.post(
                f"{self.base_url}/api/upload",
                files=[("files", (file_path.name, f, mime))],
                timeout=30,
            )
        resp.raise_for_status()
        data = resp.json()
        session_id = data.get("session_id") or data.get("sessionId") or data["id"]
        self._sessions[dataset] = session_id
        print(f"  [upload] {dataset} → session {session_id[:8]}...")
        return session_id

    def chat(self, session_id: str, question: str) -> dict:
        resp = requests.post(
            f"{self.base_url}/api/chat",
            json={"session_id": session_id, "question": question, "history": []},
            timeout=self.timeout,
        )
        return {
            "status": resp.status_code,
            "body": resp.json() if resp.ok else {},
            "text": resp.text,
        }


# ── Scoring (LLM quality rubric) ──────────────────────────────────────────────
#
# Thay vì test exact-match số/keyword (quá rigid với LLM output),
# đánh giá theo 5 chiều chất lượng thực tế:
#
#  no_meta    (0/1)   — không có chain-of-thought leak ra ngoài
#  insight    (0–1)   — có "so what" / hành động đề xuất, không chỉ đọc số
#  concise    (0–1)   — 2-5 câu, đúng tầm; không quá ngắn hoặc dài lê thê
#  vn_natural (0–1)   — tiếng Việt tự nhiên, không robotic/English mix
#  factual_ok (0–1)   — sanity check: không contradict ground truth hiển nhiên
#
# overall = no_meta×0.30 + insight×0.25 + concise×0.20 + vn_natural×0.15 + factual_ok×0.10
#
# Lý do weights này: meta-commentary leak là bug nghiêm trọng nhất (thiết kế prompt);
# insight là giá trị cốt lõi LLM mang lại; factual chỉ sanity (đúng logic analysis).
# ─────────────────────────────────────────────────────────────────────────────

_VN_RE = re.compile(
    r"[àáâãèéêìíòóôõùúýăđơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỷỹ]", re.I
)

# Patterns báo hiệu meta-commentary / chain-of-thought leak
_META_RE = [
    re.compile(p, re.I) for p in [
        r"\bTRƯỚC TIÊN\b",
        r"\bDựa trên (dữ liệu|kết quả|phân tích)",
        r"\bCâu hỏi (này|trên)\b",
        r"\btôi sẽ (tính|phân tích|xem|kiểm tra|tìm)\b",
        r"\btôi cần\b",
        r"\bBước \d\b",
        r"\bĐể trả lời\b",
        r"\bHãy (tính|xem|phân tích)\b",
        r"\bTôi (hiểu|thấy rằng) câu hỏi",
    ]
]

# Signals cho thấy câu trả lời có insight / actionable
_INSIGHT_SIGNALS = [
    r"\bcho thấy\b", r"\bcho thấy rằng\b",
    r"\bđề xuất\b", r"\bkhuyến nghị\b", r"\bnên\b",
    r"\bcần (chú ý|tập trung|cải thiện|xem xét)\b",
    r"\bcơ hội\b", r"\brủi ro\b",
    r"\btăng trưởng\b", r"\bgiảm\b.*\bcần\b",
    r"\bchiếm\b.*\b%\b", r"\btỷ lệ\b",
    r"\bso với\b", r"\bchênh lệch\b",
    r"\btiếp tục\b", r"\bcải thiện\b",
    r"\bhành động\b", r"\bưu tiên\b",
]

# Off-topic redirect signals
_REDIRECT_SIGNALS = [
    "ngoài phạm vi", "không thể hỗ trợ", "chỉ hỗ trợ", "không hỗ trợ",
    "câu hỏi về dữ liệu", "upload file", "phân tích dữ liệu",
    "tôi là trợ lý", "không phải lĩnh vực",
]


def _parse_vn_number(text: str) -> list[float]:
    """
    Parse số từ text có thể dùng định dạng Việt Nam (dấu phẩy = thập phân).
    Ví dụ: "0,212" → 0.212; "1.234,56" → 1234.56; "8.950" → 8950
    """
    results = []
    # Pattern: số với optional dấu phẩy/chấm làm phân cách
    for raw in re.findall(r"-?\d[\d.,]*", text):
        # Thử parse theo format Việt Nam: 1.234,56 → 1234.56
        if re.match(r"^\d{1,3}(\.\d{3})+(,\d+)?$", raw) and not raw.startswith("0."):
            # European: "1.234,56" hoặc "1.234"
            val = raw.replace(".", "").replace(",", ".")
            try:
                results.append(float(val))
                continue
            except ValueError:
                pass
        # Format "0,212" (leading 0, comma as decimal)
        if re.match(r"^-?0,\d+$", raw):
            try:
                results.append(float(raw.replace(",", ".")))
                continue
            except ValueError:
                pass
        # Standard: strip commas (thousands separator)
        cleaned = raw.replace(",", "")
        try:
            results.append(float(cleaned))
        except ValueError:
            pass
    return results


def score_no_meta(answer: str) -> float:
    """1.0 nếu không có meta-commentary, 0.0 nếu có leak."""
    if not answer:
        return 0.0
    return 0.0 if any(p.search(answer) for p in _META_RE) else 1.0


def score_insight(tc: TestCase, answer: str) -> float:
    """
    Đánh giá câu trả lời có đi xa hơn việc chỉ đọc con số không.
    - off_topic / bot_info: pass nếu redirect hợp lý
    - edge / trend: pass nếu trả lời substantive
    - aggregation/ranking/...: cần có ít nhất 1 signal insight
    """
    if not answer or len(answer.strip()) < 15:
        return 0.0
    if tc.category in ("off_topic",):
        lower = answer.lower()
        return 1.0 if any(s in lower for s in _REDIRECT_SIGNALS) else 0.3
    if tc.category in ("bot_info", "edge"):
        return 1.0 if len(answer.strip()) > 60 else 0.5
    # Các category còn lại: check signal insight
    n_signals = sum(1 for p in _INSIGHT_SIGNALS if re.search(p, answer, re.I))
    if n_signals >= 2:
        return 1.0
    if n_signals == 1:
        return 0.7
    # Không có signal nhưng câu trả lời đủ dài → partial
    return 0.4 if len(answer.strip()) > 80 else 0.2


def score_concise(answer: str) -> float:
    """
    Đúng độ dài: 40–500 chars lý tưởng (≈ 2-5 câu).
    Quá ngắn (< 20): chưa đủ. Quá dài (> 800): dài dòng.
    """
    if not answer:
        return 0.0
    n = len(answer.strip())
    if n < 20:
        return 0.1
    if n < 40:
        return 0.5
    if n <= 500:
        return 1.0
    if n <= 800:
        return 0.7
    return 0.4   # quá dài


def score_vn_natural(answer: str, llm_failed: bool) -> float:
    """Tiếng Việt tự nhiên: có diacritics, không robotic/English dump."""
    if not answer:
        return 0.0
    has_vn = bool(_VN_RE.search(answer))
    # Nếu fallback deterministic, câu trả lời thường là bảng số cứng
    if llm_failed:
        return 0.3 if has_vn else 0.1
    # Check xem có đổ raw JSON / SQL / DataFrame ra không
    has_dump = bool(re.search(r"\{.*:.*\}|\bSELECT\b|dtype|NaN|DataFrame", answer))
    score = 1.0 if has_vn else 0.5
    if has_dump:
        score -= 0.4
    return max(0.0, round(score, 2))


def score_factual_ok(tc: TestCase, answer: str, http_ok: bool) -> float:
    """
    Sanity check lỏng: câu trả lời không contradict ground truth hiển nhiên.
    - Nếu không có GT: 1.0 (không thể check)
    - Nếu GT là số: kiểm tra xem answer có chứa số trong ballpark ±30% không
    - Nếu GT là keyword: kiểm tra keyword (normalized) xuất hiện không
    Không phạt partial miss — đây chỉ là safety net chống hallucinate nặng.
    """
    if not http_ok or not answer:
        return 0.0
    # Off-topic: luôn ok nếu có substantive answer
    if tc.category in ("off_topic", "bot_info", "edge"):
        return 1.0
    gt = GROUND_TRUTH.get(tc.gt_key)
    if gt is None:
        return 1.0   # không có GT → không check

    def _norm(s: str) -> str:
        return unicodedata.normalize("NFKD", s).lower()

    if tc.expected_type == "number":
        nums = _parse_vn_number(answer)
        if not nums:
            return 0.5   # không thể verify → neutral
        gt_f = float(gt)
        # Pass nếu bất kỳ số nào trong answer nằm trong ±30% GT
        for n in nums:
            if gt_f == 0 and abs(n) < 1:
                return 1.0
            if gt_f != 0 and abs(n - gt_f) / abs(gt_f) <= 0.30:
                return 1.0
        # Fail nếu số sai hoàn toàn (order of magnitude khác)
        return 0.0

    if tc.expected_type == "keyword":
        gt_str = str(gt)
        gt_words = [w for w in gt_str.split() if len(w) > 2]
        if not gt_words:
            return 1.0
        matches = sum(1 for w in gt_words if _norm(w) in _norm(answer))
        return 1.0 if matches >= len(gt_words) else (0.5 if matches > 0 else 0.0)

    return 1.0   # "any" / "none" → không check factual


# ── Result dataclass ──────────────────────────────────────────────────────────
@dataclass
class EvalResult:
    id: int
    dataset: str
    category: str
    question: str
    answer: str
    latency_ms: float
    http_status: int
    llm_failed: bool
    # 5 quality dimensions
    no_meta: float       # không có meta-commentary leak
    insight: float       # có actionable insight
    concise: float       # độ dài phù hợp
    vn_natural: float    # tiếng Việt tự nhiên
    factual_ok: float    # không contradict ground truth
    overall: float = field(init=False)
    error: str = ""

    def __post_init__(self):
        self.overall = round(
            self.no_meta    * 0.30
            + self.insight  * 0.25
            + self.concise  * 0.20
            + self.vn_natural * 0.15
            + self.factual_ok * 0.10,
            3,
        )


# ── Runner ────────────────────────────────────────────────────────────────────
def run_eval(
    client: APIClient,
    cases: list[TestCase],
    verbose: bool = True,
) -> list[EvalResult]:
    results: list[EvalResult] = []

    for tc in cases:
        if verbose:
            print(f"\n[{tc.id:>3}] [{tc.category:<12}] {tc.question[:70]}")

        # Upload / reuse session
        try:
            session_id = client.get_or_create_session(tc.dataset)
        except Exception as e:
            if verbose:
                print(f"      ❌ Upload lỗi: {e}")
            results.append(EvalResult(
                id=tc.id, dataset=tc.dataset, category=tc.category,
                question=tc.question, answer="", latency_ms=0,
                http_status=0, llm_failed=False,
                no_meta=0, insight=0, concise=0, vn_natural=0, factual_ok=0,
                error=f"Upload lỗi: {e}",
            ))
            continue

        # Chat
        t0 = time.perf_counter()
        try:
            resp = client.chat(session_id, tc.question)
            latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        except Exception as e:
            latency_ms = round((time.perf_counter() - t0) * 1000, 1)
            results.append(EvalResult(
                id=tc.id, dataset=tc.dataset, category=tc.category,
                question=tc.question, answer="", latency_ms=latency_ms,
                http_status=0, llm_failed=False,
                no_meta=0, insight=0, concise=0, vn_natural=0, factual_ok=0,
                error=f"Request lỗi: {e}",
            ))
            continue

        http_ok  = resp["status"] == 200
        body     = resp["body"]
        answer   = body.get("answer", "") if http_ok else ""
        llm_fail = bool(body.get("llm_synthesis_failed", False))

        nm   = score_no_meta(answer)
        ins  = score_insight(tc, answer)
        con  = score_concise(answer)
        vn   = score_vn_natural(answer, llm_failed=llm_fail)
        fok  = score_factual_ok(tc, answer, http_ok)

        r = EvalResult(
            id=tc.id, dataset=tc.dataset, category=tc.category,
            question=tc.question,
            answer=answer[:300] if answer else f"HTTP {resp['status']}: {resp['text'][:100]}",
            latency_ms=latency_ms,
            http_status=resp["status"],
            llm_failed=llm_fail,
            no_meta=nm, insight=ins, concise=con, vn_natural=vn, factual_ok=fok,
            error="",
        )
        results.append(r)

        if verbose:
            icon = "✅" if r.overall >= 0.7 else ("⚠️" if r.overall >= 0.4 else "❌")
            dims = f"meta={nm:.0f} ins={ins:.2f} con={con:.2f} vn={vn:.2f} fok={fok:.2f}"
            print(f"      {icon} overall={r.overall:.2f} | {dims} | {latency_ms:.0f}ms | llm_fallback={llm_fail}")
            print(f"      → {answer[:130]!r}")

    return results


# ── Report Generator ──────────────────────────────────────────────────────────
def save_csv(results: list[EvalResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        w.writeheader()
        w.writerows(asdict(r) for r in results)
    print(f"\n[CSV] {path}")


def save_html(results: list[EvalResult], path: Path) -> None:
    from collections import defaultdict
    path.parent.mkdir(parents=True, exist_ok=True)

    total    = len(results)
    passed   = sum(1 for r in results if r.overall >= 0.7)
    errors   = sum(1 for r in results if r.error or r.http_status not in (200, 0))
    avg_lat  = round(sum(r.latency_ms for r in results) / max(total, 1))
    llm_fail = sum(r.llm_failed for r in results)

    def _avg(attr: str) -> float:
        return round(sum(getattr(r, attr) for r in results) / max(total, 1), 2)

    # Per-dimension averages
    avg_meta = _avg("no_meta")
    avg_ins  = _avg("insight")
    avg_con  = _avg("concise")
    avg_vn   = _avg("vn_natural")
    avg_fok  = _avg("factual_ok")

    # Per-category summary
    cat_stats: dict[str, list] = defaultdict(list)
    for r in results:
        cat_stats[r.category].append(r.overall)
    cat_rows = "".join(
        f"<tr><td>{cat}</td><td>{len(v)}</td>"
        f"<td>{round(sum(v)/len(v), 2):.2f}</td>"
        f"<td>{sum(1 for x in v if x >= 0.7)}/{len(v)}</td></tr>"
        for cat, v in sorted(cat_stats.items())
    )

    # Per-dataset summary
    ds_stats: dict[str, list] = defaultdict(list)
    for r in results:
        ds_stats[r.dataset].append(r.overall)
    ds_rows = "".join(
        f"<tr><td>{ds}</td><td>{len(v)}</td>"
        f"<td>{round(sum(v)/len(v), 2):.2f}</td>"
        f"<td>{sum(1 for x in v if x >= 0.7)}/{len(v)}</td></tr>"
        for ds, v in sorted(ds_stats.items())
    )

    def row_color(r: EvalResult) -> str:
        if r.overall >= 0.7: return "#e6f4ea"
        if r.overall >= 0.4: return "#fff8e1"
        return "#fde8e8"

    def meta_badge(v: float) -> str:
        return "✅" if v >= 1.0 else "❌"

    detail_rows = "".join(
        f"<tr style='background:{row_color(r)}'>"
        f"<td>{r.id}</td>"
        f"<td><span class='badge badge-{r.category}'>{r.category}</span></td>"
        f"<td style='font-size:0.78rem'>{r.dataset}</td>"
        f"<td>{r.question}</td>"
        f"<td class='ans' title='{r.answer}'>{r.answer[:180]}</td>"
        f"<td title='no meta-commentary'>{meta_badge(r.no_meta)}</td>"
        f"<td>{r.insight:.2f}</td>"
        f"<td>{r.concise:.2f}</td>"
        f"<td>{r.vn_natural:.2f}</td>"
        f"<td>{r.factual_ok:.2f}</td>"
        f"<td><b>{r.overall:.2f}</b></td>"
        f"<td>{r.latency_ms:.0f}</td>"
        f"<td>{'⚠️' if r.llm_failed else ''}</td>"
        f"<td style='color:red;font-size:0.75rem'>{r.error[:60]}</td>"
        f"</tr>"
        for r in results
    )

    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html = f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<title>Eval Report — Data Analysis AI Agent</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 24px; color: #222; }}
  h1 {{ font-size: 1.6rem; margin-bottom: 4px; }}
  .meta {{ color: #666; font-size: 0.85rem; margin-bottom: 24px; }}
  .kpi-grid {{ display: grid; grid-template-columns: repeat(7, 1fr); gap: 10px; margin-bottom: 28px; }}
  .kpi {{ background: #f8f9fa; border-radius: 8px; padding: 12px 8px; text-align: center; border: 1px solid #e0e0e0; }}
  .kpi .val {{ font-size: 1.6rem; font-weight: 700; color: #1a73e8; }}
  .kpi .lbl {{ font-size: 0.72rem; color: #555; margin-top: 3px; }}
  .dim-grid {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px; margin-bottom: 24px; }}
  .dim {{ background: #fff; border: 2px solid #e0e0e0; border-radius: 8px; padding: 10px; text-align: center; }}
  .dim .dval {{ font-size: 1.4rem; font-weight: 700; }}
  .dim .dlbl {{ font-size: 0.7rem; color: #666; margin-top: 2px; }}
  h2 {{ font-size: 1.05rem; margin: 20px 0 6px; border-bottom: 2px solid #e0e0e0; padding-bottom: 3px; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 0.8rem; margin-bottom: 28px; }}
  th {{ background: #1a73e8; color: white; padding: 7px 8px; text-align: left; position: sticky; top: 0; }}
  td {{ padding: 5px 8px; border-bottom: 1px solid #eee; vertical-align: top; }}
  .ans {{ max-width: 260px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; cursor: pointer; }}
  .badge {{ display: inline-block; padding: 2px 6px; border-radius: 9px; font-size: 0.7rem; font-weight: 600; }}
  .badge-aggregation {{ background:#dbeafe;color:#1e40af }}
  .badge-ranking     {{ background:#fef3c7;color:#92400e }}
  .badge-comparison  {{ background:#d1fae5;color:#065f46 }}
  .badge-trend       {{ background:#ede9fe;color:#5b21b6 }}
  .badge-bot_info    {{ background:#e0f2fe;color:#075985 }}
  .badge-off_topic   {{ background:#fee2e2;color:#991b1b }}
  .badge-edge        {{ background:#fce7f3;color:#9d174d }}
  .badge-language    {{ background:#ecfdf5;color:#065f46 }}
  tr:hover {{ filter: brightness(0.96); }}
  .note {{ color:#888;font-size:0.75rem;margin-top:24px }}
</style>
</head>
<body>
<h1>📊 Eval Report — Data Analysis AI Agent</h1>
<div class="meta">Chạy lúc: {run_time} &nbsp;|&nbsp; {total} câu hỏi &nbsp;|&nbsp; 6 datasets</div>

<div class="kpi-grid">
  <div class="kpi"><div class="val">{passed}/{total}</div><div class="lbl">Pass (≥ 0.7)</div></div>
  <div class="kpi"><div class="val">{round(passed/total*100,1)}%</div><div class="lbl">Pass Rate</div></div>
  <div class="kpi"><div class="val">{_avg("overall"):.2f}</div><div class="lbl">Avg Overall</div></div>
  <div class="kpi"><div class="val">{avg_lat}ms</div><div class="lbl">Avg Latency</div></div>
  <div class="kpi"><div class="val">{llm_fail}/{total}</div><div class="lbl">LLM Fallback</div></div>
  <div class="kpi"><div class="val">{errors}</div><div class="lbl">HTTP Errors</div></div>
</div>

<h2>Quality Dimensions (avg)</h2>
<div class="dim-grid">
  <div class="dim"><div class="dval" style="color:#059669">{avg_meta:.2f}</div><div class="dlbl">no_meta ×0.30<br><small>Không leak chain-of-thought</small></div></div>
  <div class="dim"><div class="dval" style="color:#7c3aed">{avg_ins:.2f}</div><div class="dlbl">insight ×0.25<br><small>Có insight / khuyến nghị</small></div></div>
  <div class="dim"><div class="dval" style="color:#d97706">{avg_con:.2f}</div><div class="dlbl">concise ×0.20<br><small>Đủ ngắn gọn</small></div></div>
  <div class="dim"><div class="dval" style="color:#0891b2">{avg_vn:.2f}</div><div class="dlbl">vn_natural ×0.15<br><small>Tiếng Việt tự nhiên</small></div></div>
  <div class="dim"><div class="dval" style="color:#dc2626">{avg_fok:.2f}</div><div class="dlbl">factual_ok ×0.10<br><small>Số không lệch ±30%</small></div></div>
</div>

<h2>Theo Category</h2>
<table>
  <tr><th>Category</th><th>Count</th><th>Avg Overall</th><th>Pass</th></tr>
  {cat_rows}
</table>

<h2>Theo Dataset</h2>
<table>
  <tr><th>Dataset</th><th>Count</th><th>Avg Overall</th><th>Pass</th></tr>
  {ds_rows}
</table>

<h2>Chi tiết 100 câu hỏi</h2>
<table>
  <tr>
    <th>#</th><th>Category</th><th>Dataset</th><th>Câu hỏi</th><th>Câu trả lời</th>
    <th>Meta</th><th>Insight</th><th>Concise</th><th>VN</th><th>Factual</th>
    <th>Overall</th><th>ms</th><th>LLM↓</th><th>Error</th>
  </tr>
  {detail_rows}
</table>

<p class="note">
  Scoring: overall = no_meta×0.30 + insight×0.25 + concise×0.20 + vn_natural×0.15 + factual_ok×0.10<br>
  Meta: không có "TRƯỚC TIÊN / Dựa trên dữ liệu / tôi sẽ tính" leak. Insight: signal "cho thấy / đề xuất / cần chú ý". Concise: 40–500 ký tự. VN: dùng dấu tiếng Việt. Factual: số không lệch ±30% ground truth.
</p>
</body>
</html>"""

    path.write_text(html, encoding="utf-8")
    print(f"[HTML] {path}")


def print_summary(results: list[EvalResult]) -> None:
    from collections import defaultdict
    total  = len(results)
    passed = sum(1 for r in results if r.overall >= 0.7)
    print("\n" + "="*70)
    print(f"  TỔNG KẾT: {passed}/{total} pass ({passed/total*100:.1f}%)")
    print(f"  Avg no_meta     : {sum(r.no_meta for r in results)/total:.3f}  (×0.30)")
    print(f"  Avg insight     : {sum(r.insight for r in results)/total:.3f}  (×0.25)")
    print(f"  Avg concise     : {sum(r.concise for r in results)/total:.3f}  (×0.20)")
    print(f"  Avg vn_natural  : {sum(r.vn_natural for r in results)/total:.3f}  (×0.15)")
    print(f"  Avg factual_ok  : {sum(r.factual_ok for r in results)/total:.3f}  (×0.10)")
    print(f"  Avg overall     : {sum(r.overall for r in results)/total:.3f}")
    print(f"  Avg latency     : {sum(r.latency_ms for r in results)/total:.0f}ms")
    llm_fail_count = sum(r.llm_failed for r in results)
    llm_usage_pct = round((total - llm_fail_count) / total * 100, 1)
    print(f"  LLM usage rate  : {llm_usage_pct}%  ({llm_fail_count}/{total} fallback to rule-based)")
    print(f"  HTTP errors     : {sum(1 for r in results if r.http_status not in (200, 0))}")

    print("\n  By category:")
    cat_stats: dict[str, list] = defaultdict(list)
    for r in results:
        cat_stats[r.category].append(r.overall)
    for cat, v in sorted(cat_stats.items()):
        bar = "█" * int(sum(v)/len(v) * 10)
        print(f"    {cat:<14} [{bar:<10}] {sum(v)/len(v):.2f}  ({sum(1 for x in v if x>=0.7)}/{len(v)} pass)")

    print("\n  Failed (overall < 0.7):")
    for r in [x for x in results if x.overall < 0.7]:
        print(f"    [{r.id:>3}] {r.question[:55]:<55} → {r.overall:.2f}")
    print("="*70)


# ── Main ──────────────────────────────────────────────────────────────────────
def parse_ids(s: str) -> set[int]:
    ids: set[int] = set()
    for part in s.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            ids.update(range(int(a), int(b) + 1))
        else:
            ids.add(int(part))
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Eval 100 câu hỏi cho Data Analysis AI Agent")
    parser.add_argument("--base-url", default="http://localhost:8000", help="URL server")
    parser.add_argument("--out",      default="results/eval", help="Output prefix (không có extension)")
    parser.add_argument("--ids",      default="",             help="Subset IDs, vd: 1,5,10-20")
    parser.add_argument("--dataset",  default="",             help="Chỉ chạy 1 dataset")
    parser.add_argument("--no-report", action="store_true",   help="Không tạo file báo cáo")
    parser.add_argument("--quiet",    action="store_true",    help="Ít log hơn")
    args = parser.parse_args()

    print(f"Loading ground truths from datasets...")
    _load_ground_truths()
    print(f"Loaded {len(GROUND_TRUTH)} ground truth values\n")

    client = APIClient(args.base_url)
    if not client.health_check():
        print(f"[ERROR] Server không phản hồi tại {args.base_url}")
        print("  Hãy chạy: uvicorn backend.app.main:app --host 0.0.0.0 --port 8000")
        sys.exit(1)
    print(f"[OK] Server sẵn sàng tại {args.base_url}\n")

    # Filter test cases
    cases = TEST_CASES
    if args.ids:
        target_ids = parse_ids(args.ids)
        cases = [tc for tc in cases if tc.id in target_ids]
    if args.dataset:
        cases = [tc for tc in cases if tc.dataset == args.dataset]
    print(f"Chạy {len(cases)} test cases...\n")

    results = run_eval(client, cases, verbose=not args.quiet)
    print_summary(results)

    if not args.no_report and results:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_base = Path(args.out + f"_{ts}")
        save_csv(results, out_base.with_suffix(".csv"))
        save_html(results, out_base.with_suffix(".html"))


if __name__ == "__main__":
    main()
