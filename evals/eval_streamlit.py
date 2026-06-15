"""
Data Analysis AI Agent — Demo & Eval Dashboard
===============================================
Tab 1 "💬 Demo": iframe nhúng trực tiếp React app đang chạy trên backend
  → giao diện y hệt deployed app (dark theme, sidebar, chat bubbles, ...)
Tab 2 "🧪 Eval": 100 câu hỏi tự động với metrics đầy đủ
Tab 3 "ℹ️ About": breakdown câu hỏi + scoring rubric

Cài đặt:   pip install streamlit requests pandas plotly
Chạy:      streamlit run evals/eval_streamlit.py
           (backend phải chạy trước: uvicorn backend.app.main:app --reload)
"""
from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="DataAgent — Demo & Eval",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Constants ─────────────────────────────────────────────────────────────────
DEFAULT_API = "http://localhost:8000"
DEFAULT_CSV = str(Path(__file__).parent.parent / "sales_data.csv")
DEMO_HEIGHT = 780  # iframe height in px

CATEGORY_COLORS = {
    "aggregation": "#4CAF50", "filter": "#2196F3", "ranking": "#FF9800",
    "time_series": "#9C27B0", "comparison": "#F44336", "count": "#00BCD4",
    "complex": "#E91E63", "schema": "#607D8B",
    "bot_info": "#009688", "off_topic": "#795548",
}

# ── API helpers ────────────────────────────────────────────────────────────────

def _upload_file(api_url: str, name: str, data: bytes, mime: str) -> dict:
    resp = requests.post(
        f"{api_url}/api/upload",
        files=[("files", (name, data, mime))],
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def _call(api_url: str, session_id: str, question: str, mode: str) -> dict:
    ep = {"analyze": "/api/analyze", "chat": "/api/chat", "agent": "/api/agent-chat"}[mode]
    t0 = time.perf_counter()
    try:
        resp = requests.post(
            f"{api_url}{ep}",
            json={"session_id": session_id, "question": question},
            timeout=90,
        )
        ms = int((time.perf_counter() - t0) * 1000)
        data = resp.json() if resp.content else {}
        return {
            "ok": resp.ok, "status": resp.status_code, "ms": ms,
            "answer": data.get("answer", ""),
            "charts": data.get("charts", []),
            "agent_steps": data.get("agent_steps", []),
            "executed_queries": data.get("executed_queries", []),
            "query_type": data.get("query_type", "data_query"),
            "error": data.get("detail") if not resp.ok else None,
        }
    except Exception as exc:
        return {"ok": False, "status": 0, "ms": int((time.perf_counter()-t0)*1000),
                "answer": "", "charts": [], "agent_steps": [], "executed_queries": [],
                "query_type": "error", "error": str(exc)}

# ── Top bar (thay sidebar — collapsed) ────────────────────────────────────────

def topbar() -> str:
    c1, c2, c3 = st.columns([3, 1, 1])
    api_url = c1.text_input("🔗 Backend URL", DEFAULT_API, label_visibility="collapsed",
                            placeholder="http://localhost:8000")
    if c2.button("🏥 Health", use_container_width=True):
        try:
            r = requests.get(f"{api_url}/api/health", timeout=5)
            if r.ok:
                j = r.json()
                c3.success(f"✅ {j.get('llm_provider','?')[:20]}")
            else:
                c3.error(f"❌ {r.status_code}")
        except Exception as exc:
            c3.error(f"❌ offline")
    return api_url

# ── Tab 1: Demo — iframe vào React app thực ───────────────────────────────────

def tab_demo(api_url: str) -> None:
    """
    Nhúng trực tiếp React frontend thông qua iframe.
    Backend đã serve React build tại gốc ("/"), nên iframe trỏ vào api_url
    là cách duy nhất đảm bảo UI giống 100% với deployed app.
    """
    # Kiểm tra backend còn sống không
    alive = False
    try:
        r = requests.get(f"{api_url}/api/health", timeout=3)
        alive = r.ok
    except Exception:
        pass

    if not alive:
        st.warning(
            f"**Backend không phản hồi** tại `{api_url}`\n\n"
            "Chạy backend trước:\n"
            "```\nuvicorn backend.app.main:app --reload --port 8000\n```\n"
            "Sau đó refresh trang này."
        )
        return

    # Inject CSS để xóa padding mặc định của Streamlit, iframe chiếm toàn bộ chiều cao
    st.markdown(
        """
        <style>
        /* Xóa padding của block container để iframe full-width */
        [data-testid="stMainBlockContainer"] { padding: 0 !important; max-width: 100% !important; }
        [data-testid="block-container"]      { padding: 0 !important; max-width: 100% !important; }
        .stApp > header                       { display: none; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Tính chiều cao viewport: dùng window.innerHeight qua HTML trick
    height = DEMO_HEIGHT
    components.iframe(api_url, height=height, scrolling=False)

# ── Question bank (100 questions) ─────────────────────────────────────────────
# kw:    keywords — ≥40% phải xuất hiện trong answer để pass
# vals:  giá trị số kỳ vọng — ít nhất 1 phải xuất hiện (nếu list rỗng → skip)
# chart: True nếu câu hỏi nên sinh biểu đồ
# qt:    query_type kỳ vọng từ API ("data_query" | "bot_info" | "off_topic")
QUESTIONS: list[dict[str, Any]] = [
    # ── Aggregation (20) ──────────────────────────────────────────────────────
    {"id":  1,"cat":"aggregation","diff":"easy",  "ep":"chat","qt":"data_query",
     "q":"Tổng revenue là bao nhiêu?",
     "kw":["revenue","tổng"],"vals":["859045000","859,045","859"],"chart":False},
    {"id":  2,"cat":"aggregation","diff":"easy",  "ep":"chat","qt":"data_query",
     "q":"Tổng revenue theo product_category?",
     "kw":["laptop","điện thoại","441","301"],"vals":[],"chart":True},
    {"id":  3,"cat":"aggregation","diff":"easy",  "ep":"chat","qt":"data_query",
     "q":"Revenue theo customer_region?",
     "kw":["tp.hcm","hà nội","đà nẵng"],"vals":["431","224","98"],"chart":True},
    {"id":  4,"cat":"aggregation","diff":"easy",  "ep":"chat","qt":"data_query",
     "q":"Tổng revenue của Phạm Hoa?",
     "kw":["phạm hoa","431"],"vals":["431850000","431,850","431"],"chart":False},
    {"id":  5,"cat":"aggregation","diff":"easy",  "ep":"chat","qt":"data_query",
     "q":"Tổng revenue của Laptop?",
     "kw":["laptop","441"],"vals":["441175000","441,175","441"],"chart":False},
    {"id":  6,"cat":"aggregation","diff":"easy",  "ep":"chat","qt":"data_query",
     "q":"Revenue của TP.HCM?",
     "kw":["tp.hcm","431"],"vals":["431850000","431,850","431"],"chart":False},
    {"id":  7,"cat":"aggregation","diff":"easy",  "ep":"chat","qt":"data_query",
     "q":"Revenue của Hà Nội?",
     "kw":["hà nội","224"],"vals":["224650000","224,650","224"],"chart":False},
    {"id":  8,"cat":"aggregation","diff":"easy",  "ep":"chat","qt":"data_query",
     "q":"Revenue lớn nhất từ một đơn hàng?",
     "kw":["83","max"],"vals":["83600000","83,600","83"],"chart":False},
    {"id":  9,"cat":"aggregation","diff":"medium","ep":"chat","qt":"data_query",
     "q":"Discount trung bình là bao nhiêu phần trăm?",
     "kw":["discount","trung bình"],"vals":["4.33","4,33","4.3","4%"],"chart":False},
    {"id": 10,"cat":"aggregation","diff":"medium","ep":"chat","qt":"data_query",
     "q":"Tổng revenue của từng sales_rep?",
     "kw":["phạm hoa","trần minh","nguyễn lân"],"vals":["431","224"],"chart":True},
    {"id": 11,"cat":"aggregation","diff":"easy",  "ep":"chat","qt":"data_query",
     "q":"Revenue tháng 1/2024?",
     "kw":["tháng 1","148"],"vals":["148300000","148,300","148"],"chart":False},
    {"id": 12,"cat":"aggregation","diff":"easy",  "ep":"chat","qt":"data_query",
     "q":"Revenue tháng 2/2024?",
     "kw":["tháng 2","168"],"vals":["168925000","168,925","168"],"chart":False},
    {"id": 13,"cat":"aggregation","diff":"medium","ep":"chat","qt":"data_query",
     "q":"Tổng revenue của Nguyễn Lân?",
     "kw":["nguyễn lân","116"],"vals":["116475000","116,475","116"],"chart":False},
    {"id": 14,"cat":"aggregation","diff":"medium","ep":"chat","qt":"data_query",
     "q":"Revenue của Đà Nẵng?",
     "kw":["đà nẵng","98"],"vals":["98100000","98,100","98"],"chart":False},
    {"id": 15,"cat":"aggregation","diff":"medium","ep":"chat","qt":"data_query",
     "q":"Tổng revenue của Màn hình?",
     "kw":["màn hình","72"],"vals":["72995000","72,995","72"],"chart":False},
    {"id": 16,"cat":"aggregation","diff":"medium","ep":"chat","qt":"data_query",
     "q":"Revenue trung bình mỗi đơn theo từng vùng?",
     "kw":["vùng","trung bình"],"vals":[],"chart":True},
    {"id": 17,"cat":"aggregation","diff":"easy",  "ep":"chat","qt":"data_query",
     "q":"Revenue tháng 4/2024?",
     "kw":["tháng 4","161"],"vals":["161620000","161,620","161"],"chart":False},
    {"id": 18,"cat":"aggregation","diff":"easy",  "ep":"chat","qt":"data_query",
     "q":"Revenue tháng 5/2024?",
     "kw":["tháng 5","75"],"vals":["75775000","75,775","75"],"chart":False},
    {"id": 19,"cat":"aggregation","diff":"medium","ep":"chat","qt":"data_query",
     "q":"Discount trung bình theo từng category?",
     "kw":["discount","category","trung bình"],"vals":[],"chart":True},
    {"id": 20,"cat":"aggregation","diff":"medium","ep":"chat","qt":"data_query",
     "q":"Unit price trung bình của tất cả sản phẩm?",
     "kw":["unit_price","trung bình"],"vals":["14300000","14,300","14"],"chart":False},

    # ── Filter (12) ───────────────────────────────────────────────────────────
    {"id": 21,"cat":"filter","diff":"medium","ep":"chat","qt":"data_query",
     "q":"Đơn hàng ở TP.HCM có tổng revenue bao nhiêu?",
     "kw":["tp.hcm","431"],"vals":["431850000","431,850","431"],"chart":False},
    {"id": 22,"cat":"filter","diff":"medium","ep":"chat","qt":"data_query",
     "q":"Đơn hàng nào có discount > 10%?",
     "kw":["discount","15%"],"vals":["15"],"chart":False},
    {"id": 23,"cat":"filter","diff":"medium","ep":"chat","qt":"data_query",
     "q":"Đơn hàng nào không có discount (discount = 0)?",
     "kw":["discount","0"],"vals":[],"chart":False},
    {"id": 24,"cat":"filter","diff":"hard",  "ep":"chat","qt":"data_query",
     "q":"Laptop bán ở Hà Nội có tổng revenue bao nhiêu?",
     "kw":["laptop","hà nội"],"vals":[],"chart":False},
    {"id": 25,"cat":"filter","diff":"easy",  "ep":"chat","qt":"data_query",
     "q":"MacBook Air M2 bán được tổng revenue bao nhiêu?",
     "kw":["macbook"],"vals":[],"chart":False},
    {"id": 26,"cat":"filter","diff":"medium","ep":"chat","qt":"data_query",
     "q":"Tổng revenue của đơn hàng có revenue > 50,000,000?",
     "kw":["50","revenue"],"vals":[],"chart":False},
    {"id": 27,"cat":"filter","diff":"medium","ep":"chat","qt":"data_query",
     "q":"Đơn hàng của khách hàng Công ty ABC?",
     "kw":["abc"],"vals":[],"chart":False},
    {"id": 28,"cat":"filter","diff":"medium","ep":"chat","qt":"data_query",
     "q":"Tổng revenue của Samsung Galaxy S24?",
     "kw":["samsung"],"vals":[],"chart":False},
    {"id": 29,"cat":"filter","diff":"hard",  "ep":"chat","qt":"data_query",
     "q":"Revenue của Trần Minh tại Hà Nội?",
     "kw":["trần minh","hà nội"],"vals":[],"chart":False},
    {"id": 30,"cat":"filter","diff":"easy",  "ep":"chat","qt":"data_query",
     "q":"Đơn hàng nào bán trong tháng 6/2024?",
     "kw":["tháng 6","2024"],"vals":[],"chart":False},
    {"id": 31,"cat":"filter","diff":"medium","ep":"chat","qt":"data_query",
     "q":"Revenue của Phụ kiện theo từng vùng địa lý?",
     "kw":["phụ kiện"],"vals":[],"chart":True},
    {"id": 32,"cat":"filter","diff":"hard",  "ep":"chat","qt":"data_query",
     "q":"Laptop nào có discount cao nhất?",
     "kw":["laptop","discount"],"vals":["15","10"],"chart":False},

    # ── Ranking (12) ──────────────────────────────────────────────────────────
    {"id": 33,"cat":"ranking","diff":"easy",  "ep":"chat","qt":"data_query",
     "q":"Top 3 category theo tổng revenue?",
     "kw":["laptop","điện thoại","441"],"vals":[],"chart":True},
    {"id": 34,"cat":"ranking","diff":"easy",  "ep":"chat","qt":"data_query",
     "q":"Sales rep nào có revenue cao nhất?",
     "kw":["phạm hoa","431"],"vals":["431"],"chart":False},
    {"id": 35,"cat":"ranking","diff":"easy",  "ep":"chat","qt":"data_query",
     "q":"Vùng nào có revenue cao nhất?",
     "kw":["tp.hcm","431"],"vals":["431"],"chart":False},
    {"id": 36,"cat":"ranking","diff":"easy",  "ep":"chat","qt":"data_query",
     "q":"Top 5 đơn hàng có revenue cao nhất?",
     "kw":["83","revenue"],"vals":["83600000","83"],"chart":False},
    {"id": 37,"cat":"ranking","diff":"medium","ep":"chat","qt":"data_query",
     "q":"Top 3 sản phẩm theo tổng revenue?",
     "kw":["iphone","revenue"],"vals":[],"chart":True},
    {"id": 38,"cat":"ranking","diff":"easy",  "ep":"chat","qt":"data_query",
     "q":"Category có revenue thấp nhất?",
     "kw":["phụ kiện","43"],"vals":["43"],"chart":False},
    {"id": 39,"cat":"ranking","diff":"easy",  "ep":"chat","qt":"data_query",
     "q":"Vùng có revenue thấp nhất?",
     "kw":["huế","18"],"vals":["18"],"chart":False},
    {"id": 40,"cat":"ranking","diff":"medium","ep":"chat","qt":"data_query",
     "q":"Tháng nào có doanh thu cao nhất?",
     "kw":["tháng 2","168"],"vals":["168"],"chart":False},
    {"id": 41,"cat":"ranking","diff":"medium","ep":"chat","qt":"data_query",
     "q":"Sản phẩm đắt nhất (unit_price cao nhất)?",
     "kw":["macbook","32"],"vals":["32000000","32"],"chart":False},
    {"id": 42,"cat":"ranking","diff":"hard",  "ep":"chat","qt":"data_query",
     "q":"Sales rep nào có discount trung bình cao nhất?",
     "kw":["discount","trung bình"],"vals":[],"chart":False},
    {"id": 43,"cat":"ranking","diff":"medium","ep":"chat","qt":"data_query",
     "q":"Top 3 tháng có revenue cao nhất?",
     "kw":["tháng 2","168"],"vals":[],"chart":True},
    {"id": 44,"cat":"ranking","diff":"medium","ep":"chat","qt":"data_query",
     "q":"Top 5 khách hàng theo tổng revenue?",
     "kw":["revenue","khách hàng"],"vals":[],"chart":True},

    # ── Time series (8) ───────────────────────────────────────────────────────
    {"id": 45,"cat":"time_series","diff":"easy",  "ep":"analyze","qt":"data_query",
     "q":"Revenue theo từng tháng năm 2024?",
     "kw":["tháng","2024"],"vals":["148","168","128"],"chart":True},
    {"id": 46,"cat":"time_series","diff":"easy",  "ep":"analyze","qt":"data_query",
     "q":"Xu hướng doanh thu theo thời gian?",
     "kw":["tháng","revenue"],"vals":[],"chart":True},
    {"id": 47,"cat":"time_series","diff":"medium","ep":"chat","qt":"data_query",
     "q":"Tháng nào có doanh thu thấp nhất?",
     "kw":["tháng 7","14"],"vals":["14000000","14"],"chart":False},
    {"id": 48,"cat":"time_series","diff":"hard",  "ep":"chat","qt":"data_query",
     "q":"Tổng revenue quý 1/2024 (tháng 1, 2, 3)?",
     "kw":["quý 1","q1","tháng 1"],"vals":["445225000","445"],"chart":False},
    {"id": 49,"cat":"time_series","diff":"hard",  "ep":"chat","qt":"data_query",
     "q":"Tổng revenue quý 2/2024 (tháng 4, 5, 6)?",
     "kw":["quý 2","q2","tháng 4"],"vals":["399820000","399"],"chart":False},
    {"id": 50,"cat":"time_series","diff":"medium","ep":"chat","qt":"data_query",
     "q":"Revenue tháng 6/2024?",
     "kw":["tháng 6","162"],"vals":["162425000","162,425","162"],"chart":False},
    {"id": 51,"cat":"time_series","diff":"hard",  "ep":"chat","qt":"data_query",
     "q":"Số đơn hàng mỗi tháng trong năm 2024?",
     "kw":["tháng","đơn"],"vals":[],"chart":True},
    {"id": 52,"cat":"time_series","diff":"hard",  "ep":"analyze","qt":"data_query",
     "q":"Trend discount trung bình theo tháng 2024?",
     "kw":["discount","tháng"],"vals":[],"chart":True},

    # ── Comparison (8) ────────────────────────────────────────────────────────
    {"id": 53,"cat":"comparison","diff":"easy",  "ep":"chat","qt":"data_query",
     "q":"So sánh revenue của Laptop và Điện thoại?",
     "kw":["laptop","điện thoại","441","301"],"vals":[],"chart":True},
    {"id": 54,"cat":"comparison","diff":"easy",  "ep":"chat","qt":"data_query",
     "q":"So sánh revenue TP.HCM và Hà Nội?",
     "kw":["tp.hcm","hà nội","431","224"],"vals":[],"chart":True},
    {"id": 55,"cat":"comparison","diff":"medium","ep":"chat","qt":"data_query",
     "q":"So sánh Phạm Hoa và Trần Minh về revenue?",
     "kw":["phạm hoa","trần minh"],"vals":["431","224"],"chart":True},
    {"id": 56,"cat":"comparison","diff":"easy",  "ep":"chat","qt":"data_query",
     "q":"Laptop vs Màn hình, category nào revenue cao hơn?",
     "kw":["laptop","màn hình","441"],"vals":[],"chart":False},
    {"id": 57,"cat":"comparison","diff":"medium","ep":"chat","qt":"data_query",
     "q":"So sánh revenue tháng 1 và tháng 2/2024?",
     "kw":["tháng 1","tháng 2","148","168"],"vals":[],"chart":False},
    {"id": 58,"cat":"comparison","diff":"hard",  "ep":"chat","qt":"data_query",
     "q":"So sánh Nguyễn Lân và Lê Quân về revenue?",
     "kw":["nguyễn lân","lê quân"],"vals":["116","86"],"chart":False},
    {"id": 59,"cat":"comparison","diff":"hard",  "ep":"chat","qt":"data_query",
     "q":"iPhone 15 vs MacBook Air M2, cái nào revenue tổng cao hơn?",
     "kw":["iphone","macbook"],"vals":[],"chart":False},
    {"id": 60,"cat":"comparison","diff":"medium","ep":"chat","qt":"data_query",
     "q":"TP.HCM vs Đà Nẵng: chênh lệch revenue bao nhiêu?",
     "kw":["tp.hcm","đà nẵng"],"vals":["431","98"],"chart":False},

    # ── Count (7) ─────────────────────────────────────────────────────────────
    {"id": 61,"cat":"count","diff":"easy",  "ep":"chat","qt":"data_query",
     "q":"Dataset có bao nhiêu đơn hàng?",
     "kw":["30"],"vals":["30"],"chart":False},
    {"id": 62,"cat":"count","diff":"easy",  "ep":"chat","qt":"data_query",
     "q":"Có bao nhiêu loại category sản phẩm?",
     "kw":["4","category"],"vals":["4"],"chart":False},
    {"id": 63,"cat":"count","diff":"easy",  "ep":"chat","qt":"data_query",
     "q":"Có bao nhiêu vùng địa lý?",
     "kw":["6","vùng"],"vals":["6"],"chart":False},
    {"id": 64,"cat":"count","diff":"easy",  "ep":"chat","qt":"data_query",
     "q":"Dataset có bao nhiêu cột?",
     "kw":["11"],"vals":["11"],"chart":False},
    {"id": 65,"cat":"count","diff":"medium","ep":"chat","qt":"data_query",
     "q":"Có bao nhiêu khách hàng khác nhau?",
     "kw":["28","khách hàng"],"vals":["28","27"],"chart":False},
    {"id": 66,"cat":"count","diff":"medium","ep":"chat","qt":"data_query",
     "q":"Có bao nhiêu sản phẩm khác nhau?",
     "kw":["10","sản phẩm"],"vals":["10"],"chart":False},
    {"id": 67,"cat":"count","diff":"hard",  "ep":"chat","qt":"data_query",
     "q":"Tháng 3/2024 có bao nhiêu đơn hàng?",
     "kw":["tháng 3","5"],"vals":["5"],"chart":False},

    # ── Complex (8) ───────────────────────────────────────────────────────────
    {"id": 68,"cat":"complex","diff":"hard","ep":"chat","qt":"data_query",
     "q":"Laptop có discount > 0, tổng revenue là bao nhiêu?",
     "kw":["laptop","discount","revenue"],"vals":[],"chart":False},
    {"id": 69,"cat":"complex","diff":"hard","ep":"chat","qt":"data_query",
     "q":"Top 3 sản phẩm Laptop theo revenue?",
     "kw":["laptop","macbook","dell"],"vals":[],"chart":True},
    {"id": 70,"cat":"complex","diff":"hard","ep":"chat","qt":"data_query",
     "q":"Revenue trung bình mỗi đơn theo từng sales rep?",
     "kw":["phạm hoa","trần minh","trung bình"],"vals":[],"chart":True},
    {"id": 71,"cat":"complex","diff":"hard","ep":"chat","qt":"data_query",
     "q":"Category nào có discount trung bình cao nhất?",
     "kw":["discount","category","trung bình"],"vals":[],"chart":False},
    {"id": 72,"cat":"complex","diff":"hard","ep":"chat","qt":"data_query",
     "q":"Sản phẩm nào được bán nhiều nhất theo tổng quantity?",
     "kw":["quantity","sản phẩm"],"vals":[],"chart":False},
    {"id": 73,"cat":"complex","diff":"hard","ep":"analyze","qt":"data_query",
     "q":"Tổng quantity của Laptop theo từng tháng?",
     "kw":["laptop","quantity","tháng"],"vals":[],"chart":True},
    {"id": 74,"cat":"complex","diff":"hard","ep":"chat","qt":"data_query",
     "q":"Khách hàng nào mua nhiều nhất (revenue cao nhất)?",
     "kw":["khách hàng","revenue"],"vals":[],"chart":False},
    {"id": 75,"cat":"complex","diff":"hard","ep":"chat","qt":"data_query",
     "q":"Top 5 đơn hàng revenue lớn nhất tại TP.HCM?",
     "kw":["tp.hcm","revenue"],"vals":[],"chart":False},

    # ── Schema (5) ────────────────────────────────────────────────────────────
    {"id": 76,"cat":"schema","diff":"easy",  "ep":"analyze","qt":"data_query",
     "q":"Tóm tắt cấu trúc và thống kê cơ bản của dataset?",
     "kw":["rows","columns","30"],"vals":["30","11"],"chart":False},
    {"id": 77,"cat":"schema","diff":"easy",  "ep":"chat","qt":"data_query",
     "q":"Cột nào trong dataset là kiểu số?",
     "kw":["revenue","quantity","unit_price"],"vals":[],"chart":False},
    {"id": 78,"cat":"schema","diff":"medium","ep":"chat","qt":"data_query",
     "q":"Dataset có missing values không?",
     "kw":["missing"],"vals":[],"chart":False},
    {"id": 79,"cat":"schema","diff":"easy",  "ep":"chat","qt":"data_query",
     "q":"Phạm vi thời gian dữ liệu từ khi nào đến khi nào?",
     "kw":["2024"],"vals":["2024-01","2024-07"],"chart":False},
    {"id": 80,"cat":"schema","diff":"medium","ep":"analyze","qt":"data_query",
     "q":"Mô tả thống kê (min, max, mean) của các cột số?",
     "kw":["revenue","mean","min","max"],"vals":["83","3"],"chart":False},

    # ── Bot info (10) — kiểm tra classifier route đúng ─────────────────────
    {"id": 81,"cat":"bot_info","diff":"easy","ep":"chat","qt":"bot_info",
     "q":"Bạn là ai?",
     "kw":["phân tích","upload","csv"],"vals":[],"chart":False},
    {"id": 82,"cat":"bot_info","diff":"easy","ep":"chat","qt":"bot_info",
     "q":"Bạn tên gì?",
     "kw":["phân tích","agent"],"vals":[],"chart":False},
    {"id": 83,"cat":"bot_info","diff":"easy","ep":"chat","qt":"bot_info",
     "q":"Bạn có thể làm gì cho tôi?",
     "kw":["phân tích","upload"],"vals":[],"chart":False},
    {"id": 84,"cat":"bot_info","diff":"easy","ep":"chat","qt":"bot_info",
     "q":"Hướng dẫn sử dụng?",
     "kw":["upload","csv","câu hỏi"],"vals":[],"chart":False},
    {"id": 85,"cat":"bot_info","diff":"easy","ep":"chat","qt":"bot_info",
     "q":"Bạn được tạo bởi ai?",
     "kw":["phân tích","dữ liệu"],"vals":[],"chart":False},
    {"id": 86,"cat":"bot_info","diff":"easy","ep":"chat","qt":"bot_info",
     "q":"Who are you?",
     "kw":["phân tích","agent"],"vals":[],"chart":False},
    {"id": 87,"cat":"bot_info","diff":"easy","ep":"chat","qt":"bot_info",
     "q":"What can you do?",
     "kw":["phân tích","upload"],"vals":[],"chart":False},
    {"id": 88,"cat":"bot_info","diff":"easy","ep":"chat","qt":"bot_info",
     "q":"Tính năng của bạn là gì?",
     "kw":["phân tích","biểu đồ"],"vals":[],"chart":False},
    {"id": 89,"cat":"bot_info","diff":"easy","ep":"chat","qt":"bot_info",
     "q":"Cách dùng tool này như thế nào?",
     "kw":["upload","csv"],"vals":[],"chart":False},
    {"id": 90,"cat":"bot_info","diff":"easy","ep":"chat","qt":"bot_info",
     "q":"Tell me about yourself",
     "kw":["phân tích","dữ liệu","agent"],"vals":[],"chart":False},

    # ── Off-topic (10) — kiểm tra classifier chặn đúng ────────────────────
    {"id":  91,"cat":"off_topic","diff":"easy","ep":"chat","qt":"off_topic",
     "q":"Hôm nay thời tiết Hà Nội thế nào?",
     "kw":["phân tích","dữ liệu","upload"],"vals":[],"chart":False},
    {"id":  92,"cat":"off_topic","diff":"easy","ep":"chat","qt":"off_topic",
     "q":"Giá bitcoin hiện tại là bao nhiêu?",
     "kw":["phân tích","dữ liệu"],"vals":[],"chart":False},
    {"id":  93,"cat":"off_topic","diff":"easy","ep":"chat","qt":"off_topic",
     "q":"Cho tôi xem tin tức mới nhất?",
     "kw":["phân tích","dữ liệu","upload"],"vals":[],"chart":False},
    {"id":  94,"cat":"off_topic","diff":"easy","ep":"chat","qt":"off_topic",
     "q":"Bóng đá tối nay trận gì?",
     "kw":["phân tích","dữ liệu"],"vals":[],"chart":False},
    {"id":  95,"cat":"off_topic","diff":"easy","ep":"chat","qt":"off_topic",
     "q":"Viết thơ về mùa xuân cho tôi?",
     "kw":["phân tích","dữ liệu"],"vals":[],"chart":False},
    {"id":  96,"cat":"off_topic","diff":"easy","ep":"chat","qt":"off_topic",
     "q":"What's the weather today?",
     "kw":["phân tích","dữ liệu"],"vals":[],"chart":False},
    {"id":  97,"cat":"off_topic","diff":"easy","ep":"chat","qt":"off_topic",
     "q":"Dịch bài văn này sang tiếng Anh?",
     "kw":["phân tích","dữ liệu"],"vals":[],"chart":False},
    {"id":  98,"cat":"off_topic","diff":"easy","ep":"chat","qt":"off_topic",
     "q":"Kể cho tôi nghe câu chuyện vui?",
     "kw":["phân tích","dữ liệu"],"vals":[],"chart":False},
    {"id":  99,"cat":"off_topic","diff":"easy","ep":"chat","qt":"off_topic",
     "q":"Công thức nấu phở bò?",
     "kw":["phân tích","dữ liệu"],"vals":[],"chart":False},
    {"id": 100,"cat":"off_topic","diff":"easy","ep":"chat","qt":"off_topic",
     "q":"Recommend me some travel destinations",
     "kw":["phân tích","dữ liệu"],"vals":[],"chart":False},
]

assert len(QUESTIONS) == 100, f"Expected 100, got {len(QUESTIONS)}"

# ── Scoring ────────────────────────────────────────────────────────────────────

def _norm(t: str) -> str:
    t = t.lower()
    t = re.sub(r"(\d)[,.](\d{3})(?!\d)", r"\1\2", t)
    return t

def _kw(answer: str, kws: list[str]) -> tuple[int, int]:
    n = _norm(answer)
    return sum(1 for k in kws if _norm(k) in n), len(kws)

def _val(answer: str, vals: list[str]) -> bool:
    if not vals:
        return True
    n = _norm(answer)
    return any(_norm(v) in n for v in vals)

def _llm(answer: str) -> bool:
    if len(answer.strip()) < 30:
        return False
    return bool(re.search(
        r"\d{1,3}[,.]\d{3}|tổng|theo|trung bình|phân tích|#{1,3} |\*\*|\d+\s*%",
        answer, re.I | re.M,
    ))

def _score(raw: dict) -> dict:
    if raw.get("error"):
        return {**raw, "pass_http":False,"pass_content":False,"kw_m":0,"kw_t":0,
                "pass_kw":False,"pass_val":False,"pass_qt":False,
                "llm":False,"overall":False,"score":0}
    q = raw["_q"]
    answer = raw.get("answer","") or ""
    pass_http    = raw.get("status",0) == 200
    pass_content = len(answer.strip()) >= 30
    kw_m, kw_t  = _kw(answer, q["kw"])
    pass_kw      = (kw_m/kw_t >= 0.4) if kw_t else True
    pass_val     = _val(answer, q["vals"])
    pass_qt      = raw.get("query_type","") == q["qt"]
    llm          = _llm(answer)
    has_chart    = len(raw.get("charts",[]) or []) > 0
    chart_ok     = (not q["chart"]) or has_chart

    # For bot_info / off_topic: query_type match is the main signal (35pt)
    # For data_query: keywords + value are main
    is_routing = q["qt"] in ("bot_info","off_topic")
    overall = pass_http and pass_content and (pass_qt if is_routing else (pass_kw or pass_val))

    score = 0
    if pass_http:                              score += 15
    if pass_content:                           score += 15
    if pass_qt:                                score += 30
    if pass_kw:                                score += 20
    if pass_val:                               score += 10
    if llm:                                    score += 5
    if chart_ok and q["chart"]:                score += 5

    return {**raw,
            "pass_http":pass_http,"pass_content":pass_content,
            "kw_m":kw_m,"kw_t":kw_t,"pass_kw":pass_kw,
            "pass_val":pass_val,"pass_qt":pass_qt,
            "llm":llm,"has_chart":has_chart,"overall":overall,"score":score}

# ── Tab 2: Eval ────────────────────────────────────────────────────────────────

def tab_eval(api_url: str) -> None:
    # Filter controls
    cats   = ["all"] + sorted({q["cat"]  for q in QUESTIONS})
    fc, fd, fe = st.columns(3)
    fcat   = fc.selectbox("Category", cats,  key="ev_cat")
    fdiff  = fd.selectbox("Difficulty", ["all","easy","medium","hard"], key="ev_diff")
    fep    = fe.selectbox("Endpoint",   ["all","chat","analyze","agent"], key="ev_ep")

    active = [q for q in QUESTIONS
              if (fcat == "all" or q["cat"] == fcat)
              and (fdiff == "all" or q["diff"] == fdiff)
              and (fep  == "all" or q["ep"]  == fep)]

    bc1, bc2, bc3 = st.columns([2,2,1])
    delay = bc3.slider("Delay (ms)", 0, 2000, 300, 100, key="ev_delay")
    run   = bc1.button("▶ Run Evaluation", type="primary", use_container_width=True)
    clear = bc2.button("🗑 Clear", use_container_width=True)

    bc1.info(f"**{len(active)} câu hỏi** — {fcat} / {fdiff} / {fep}")

    if clear:
        st.session_state.pop("eval_results", None)
        st.session_state.pop("eval_session_id", None)
        st.rerun()

    if run:
        # Upload CSV for eval session
        csv_path = DEFAULT_CSV
        if not Path(csv_path).exists():
            st.error(f"Không tìm thấy `{csv_path}`"); st.stop()
        with st.spinner("Uploading dataset..."):
            try:
                r = _upload_file(api_url, "sales_data.csv",
                                 Path(csv_path).read_bytes(), "text/csv")
                sid = r["session_id"]
                st.session_state["eval_session_id"] = sid
                st.success(f"✅ session: `{sid[:20]}...`")
            except Exception as exc:
                st.error(f"Upload failed: {exc}"); st.stop()

        results, prog, live = [], st.progress(0.0,"Starting…"), st.empty()
        for i, q in enumerate(active):
            prog.progress((i+1)/len(active), f"[{i+1}/{len(active)}] Q{q['id']}: {q['q'][:50]}…")
            raw = _call(api_url, sid, q["q"], q["ep"])
            raw["_q"] = q; raw["id"]=q["id"]; raw["cat"]=q["cat"]
            raw["diff"]=q["diff"]; raw["ep"]=q["ep"]; raw["q"]=q["q"]
            results.append(_score(raw))
            if (i+1) % 5 == 0 or i == len(active)-1:
                pv = pd.DataFrame([{"ID":r["id"],"Cat":r["cat"],"Diff":r["diff"],
                    "Pass":"✅" if r["overall"] else "❌","Score":r["score"],
                    "QType":r.get("query_type","?"),"ms":r.get("ms",0),
                    "Q":r["q"][:50]+"…"} for r in results])
                live.dataframe(pv, use_container_width=True, height=220)
            if delay > 0:
                time.sleep(delay/1000)

        prog.progress(1.0,"✅ Done!")
        st.session_state["eval_results"] = results
        pn = sum(1 for r in results if r["overall"])
        sc = sum(r["score"] for r in results)/len(results)
        st.success(f"**{pn}/{len(results)} passed** | avg score: **{sc:.1f}/100**")

    # Results dashboard
    results = st.session_state.get("eval_results")
    if not results:
        st.info("Chạy eval ở trên để xem kết quả.")
        return

    df = pd.DataFrame(results)
    n   = len(df)
    pn  = int(df["overall"].sum())
    sc  = df["score"].mean()
    http= (df["status"]==200).mean()*100
    llm = df["llm"].mean()*100
    ms  = df["ms"].mean()
    qt  = df["pass_qt"].mean()*100

    c = st.columns(7)
    c[0].metric("Total",        n)
    c[1].metric("Pass Rate",    f"{pn/n*100:.1f}%", f"{pn}/{n}")
    c[2].metric("Avg Score",    f"{sc:.1f}/100")
    c[3].metric("HTTP OK",      f"{http:.1f}%")
    c[4].metric("Query Route",  f"{qt:.1f}%", help="pass_qt: API trả đúng query_type")
    c[5].metric("LLM Engaged",  f"{llm:.1f}%")
    c[6].metric("Avg Latency",  f"{ms:.0f} ms")

    st.divider()
    ch1, ch2 = st.columns(2)
    with ch1:
        cat_df = df.groupby("cat").agg(pass_rate=("overall","mean"),n=("id","count"),
                                        avg_score=("score","mean")).reset_index()
        fig = px.bar(cat_df,x="cat",y="pass_rate",color="cat",text="pass_rate",
                     title="Pass Rate by Category",color_discrete_map=CATEGORY_COLORS)
        fig.update_traces(texttemplate="%{text:.0%}",textposition="outside")
        fig.update_layout(yaxis=dict(tickformat=".0%",range=[0,1.15]),showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with ch2:
        fig2 = px.histogram(df,x="ms",color="cat",nbins=25,
                            title="Latency (ms)",color_discrete_map=CATEGORY_COLORS)
        fig2.update_layout(barmode="overlay")
        st.plotly_chart(fig2, use_container_width=True)

    ch3, ch4 = st.columns(2)
    with ch3:
        qt_df = df.groupby("query_type").agg(n=("id","count"),pass_rate=("overall","mean")).reset_index()
        fig3 = px.bar(qt_df,x="query_type",y="pass_rate",color="query_type",text="pass_rate",
                      title="Pass Rate by query_type (routing accuracy)")
        fig3.update_traces(texttemplate="%{text:.0%}",textposition="outside")
        fig3.update_layout(yaxis=dict(tickformat=".0%",range=[0,1.15]),showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)
    with ch4:
        fig4 = px.scatter(df,x="ms",y="score",color="cat",hover_data=["id","q","overall"],
                          title="Latency vs Score",color_discrete_map=CATEGORY_COLORS)
        st.plotly_chart(fig4, use_container_width=True)

    st.divider()
    st.subheader("📋 All Results")
    disp = df[["id","cat","diff","ep","q","overall","score","pass_qt",
               "pass_kw","pass_val","llm","has_chart","ms","status","query_type"]] \
              .rename(columns={"id":"ID","cat":"Cat","diff":"Diff","ep":"EP","q":"Question",
                               "overall":"Pass","score":"Score","pass_qt":"RouteOK",
                               "pass_kw":"KW","pass_val":"Val","llm":"LLM",
                               "has_chart":"Chart","ms":"ms","status":"HTTP",
                               "query_type":"QueryType"}).copy()
    show_fail = st.checkbox("❌ Failed only")
    if show_fail:
        disp = disp[~disp["Pass"]]

    def _row_color(row):
        c = "#d4edda" if row["Pass"] else "#f8d7da"
        return [f"background-color: {c}"]*len(row)

    st.dataframe(disp.style.apply(_row_color,axis=1), use_container_width=True, height=500)
    st.download_button("📥 Export CSV", disp.to_csv(index=False), "eval_results.csv","text/csv")

    with st.expander(f"❌ Failed ({len(df[~df['overall']])})"):
        f = df[~df["overall"]][["id","cat","diff","q","score","error","query_type"]]
        st.dataframe(f, use_container_width=True)

# ── Tab 3: About ──────────────────────────────────────────────────────────────

def tab_about() -> None:
    st.subheader("Bộ câu hỏi eval (100 questions)")
    df_q = pd.DataFrame([{"ID":q["id"],"Category":q["cat"],"Difficulty":q["diff"],
                           "Endpoint":q["ep"],"ExpectedQType":q["qt"],
                           "Question":q["q"][:85]} for q in QUESTIONS])

    c1,c2,c3 = st.columns(3)
    with c1:
        p1 = px.pie(df_q,names="Category",title="By Category",
                    color="Category",color_discrete_map=CATEGORY_COLORS)
        st.plotly_chart(p1,use_container_width=True)
    with c2:
        p2 = px.pie(df_q,names="Difficulty",title="By Difficulty",
                    color_discrete_sequence=["#4CAF50","#FF9800","#F44336"])
        st.plotly_chart(p2,use_container_width=True)
    with c3:
        p3 = px.pie(df_q,names="ExpectedQType",title="By Query Type")
        st.plotly_chart(p3,use_container_width=True)

    st.subheader("Toàn bộ câu hỏi")
    st.dataframe(df_q, use_container_width=True, height=420)

    st.subheader("Routing flow")
    st.markdown("""
```
/api/chat  (hoặc /api/analyze)
    │
    ├── classify_query(question)
    │       │
    │       ├── "bot_info"   → BOT_INFO_RESPONSE (describe capabilities)
    │       ├── "off_topic"  → OFF_TOPIC_RESPONSE (polite redirect)
    │       └── "data_query" → run_planned_analysis() → LLM plan + charts
    │
    └── query_type trả về trong response → eval kiểm tra RouteOK
```
""")

    st.subheader("Scoring (tổng 100pt)")
    st.markdown("""
| Metric | Pts | Mô tả |
|--------|-----|-------|
| HTTP 200 | 15 | API trả status thành công |
| Content OK | 15 | Answer ≥ 30 ký tự |
| RouteOK | 30 | `query_type` trả về đúng (bot_info/off_topic/data_query) |
| Keywords match | 20 | ≥40% từ khóa kỳ vọng xuất hiện |
| Value found | 10 | Giá trị số kỳ vọng tìm thấy |
| LLM engaged | 5 | Heuristic: số có format, markdown, phân tích |
| Chart OK | 5 | Chart tạo ra khi câu hỏi yêu cầu |

**Overall Pass** = HTTP OK ∧ Content OK ∧ (RouteOK nếu routing; KW ∨ Val nếu data)

**Ground truth** (sales_data.csv):
- 30 orders · 11 cols · 2024-01 → 2024-07
- Total revenue: **859,045,000 VND**
- Top category: Laptop 441M · Điện thoại 302M · Màn hình 73M · Phụ kiện 43M
- Top region: TP.HCM 432M · HN 225M · ĐN 98M
- Peak month: 2024-02 (168,925,000) · Lowest: 2024-07 (14,000,000)
    """)

# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # Hide default Streamlit menu + footer
    st.markdown(
        "<style>#MainMenu{visibility:hidden}footer{visibility:hidden}</style>",
        unsafe_allow_html=True,
    )

    api_url = topbar()

    t_demo, t_eval, t_about = st.tabs(["💬 Demo (React App)", "🧪 Eval (100Q)", "ℹ️ About"])

    with t_demo:
        tab_demo(api_url)
    with t_eval:
        tab_eval(api_url)
    with t_about:
        tab_about()


main()
