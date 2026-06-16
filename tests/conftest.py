import os
import io
import tempfile

# MUST be set before any backend module is imported
_TEST_DIR = tempfile.mkdtemp(prefix="data_agent_test_")
os.environ["DATA_DIR"] = _TEST_DIR
os.environ["DATABASE_URL"] = f"sqlite:///{_TEST_DIR}/test.db"
os.environ.setdefault("GROQ_API_KEY", "test-key")
os.environ.setdefault("GEMINI_API_KEY", "test-key")

import pytest
import pandas as pd
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services.storage import session_store


def make_csv_bytes(df: pd.DataFrame | None = None) -> bytes:
    if df is None:
        df = pd.DataFrame({
            "date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
            "category": ["A", "B", "A", "C", "B"],
            "amount": [100.0, 200.0, 150.0, 300.0, 250.0],
            "count": [1, 2, 3, 4, 5],
        })
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


def make_xlsx_bytes(df: pd.DataFrame | None = None) -> bytes:
    if df is None:
        df = pd.DataFrame({
            "product": ["X", "Y", "Z"],
            "sales": [1000, 2000, 1500],
            "region": ["North", "South", "East"],
        })
    buf = io.BytesIO()
    df.to_excel(buf, index=False)
    return buf.getvalue()


@pytest.fixture(scope="session")
def sample_df():
    return pd.DataFrame({
        "date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
        "category": ["A", "B", "A", "C", "B"],
        "amount": [100.0, 200.0, 150.0, 300.0, 250.0],
        "count": [1, 2, 3, 4, 5],
    })


@pytest.fixture(scope="session")
def sample_csv_bytes(sample_df):
    buf = io.BytesIO()
    sample_df.to_csv(buf, index=False)
    return buf.getvalue()


@pytest.fixture(scope="session")
def sample_xlsx_bytes():
    return make_xlsx_bytes()


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def uploaded_session_id(client, sample_csv_bytes):
    resp = client.post(
        "/api/upload",
        files=[("files", ("test.csv", sample_csv_bytes, "text/csv"))],
    )
    assert resp.status_code == 200
    return resp.json()["session_id"]


@pytest.fixture
def mock_planned_analysis(mocker):
    from backend.app.agents.runner import AgentOutput
    result = AgentOutput(
        answer="Tổng amount là 1000.",
        charts=[],
        executed_queries=["SELECT SUM(amount) FROM dataset"],
        intent="data_query",
    )
    return mocker.patch("backend.app.api.routes.chat.run", return_value=result)


@pytest.fixture
def mock_agent_run(mocker):
    from backend.app.agents.runner import AgentOutput
    result = AgentOutput(
        answer="Agent phân tích xong.",
        charts=[],
        executed_queries=[],
        intent="data_query",
    )
    return mocker.patch("backend.app.api.routes.chat.run", return_value=result)
