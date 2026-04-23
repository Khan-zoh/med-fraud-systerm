"""Tests for the FastAPI scoring API."""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from cms_fwa.serving.api import app, state


@pytest.fixture(autouse=True)
def mock_state():
    """Set up a fake risk table for testing."""
    state.risk_table = pd.DataFrame({
        "npi": ["1234567890", "0987654321", "1111111111"],
        "provider_last_name": ["Doe", "Smith", "Jones"],
        "provider_first_name": ["John", "Jane", "Bob"],
        "provider_specialty": ["Cardiology", "Cardiology", "Dermatology"],
        "provider_state": ["TX", "TX", "CA"],
        "risk_score": [85.2, 42.1, 15.0],
        "risk_tier": ["Critical", "Moderate", "Low"],
        "is_excluded": [True, False, False],
        "if_anomaly_score": [0.9, 0.4, 0.1],
        "xgb_fraud_prob": [0.85, 0.35, 0.05],
        "graph_anomaly_score": [0.7, 0.3, 0.2],
    })
    state.ready = True
    yield
    state.risk_table = None
    state.ready = False


client = TestClient(app)


def test_health_endpoint() -> None:
    """Health check should return healthy status."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["models_loaded"] is True
    assert data["providers_scored"] == 3


def test_score_valid_npi() -> None:
    """Scoring a known NPI should return risk details."""
    resp = client.post("/score", json={"npi": "1234567890"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["npi"] == "1234567890"
    assert data["risk_score"] == 85.2
    assert data["risk_tier"] == "Critical"
    assert data["model_scores"]["xgboost"] == 0.85
    assert "disclaimer" in data


def test_score_unknown_npi() -> None:
    """Scoring an unknown NPI should return 404."""
    resp = client.post("/score", json={"npi": "9999999999"})
    assert resp.status_code == 404


def test_score_invalid_npi_format() -> None:
    """Invalid NPI format should return 422."""
    resp = client.post("/score", json={"npi": "abc"})
    assert resp.status_code == 422


def test_get_provider() -> None:
    """GET /provider/{npi} should return the same data as POST /score."""
    resp = client.get("/provider/1234567890")
    assert resp.status_code == 200
    assert resp.json()["risk_score"] == 85.2


def test_top_risk_default() -> None:
    """Top risk should return providers sorted by score."""
    resp = client.get("/top-risk")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 3
    assert data[0]["risk_score"] >= data[1]["risk_score"]


def test_top_risk_filter_specialty() -> None:
    """Specialty filter should narrow results."""
    resp = client.get("/top-risk", params={"specialty": "Cardiology"})
    data = resp.json()
    assert len(data) == 2
    assert all(p["specialty"] == "Cardiology" for p in data)


def test_top_risk_filter_min_score() -> None:
    """Min score filter should exclude low-risk providers."""
    resp = client.get("/top-risk", params={"min_score": 40})
    data = resp.json()
    assert all(p["risk_score"] >= 40 for p in data)


def test_stats_endpoint() -> None:
    """Stats should return summary info."""
    resp = client.get("/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_providers"] == 3
    assert data["total_excluded"] == 1


def test_unloaded_state_returns_503() -> None:
    """When models aren't loaded, endpoints should return 503."""
    state.ready = False
    resp = client.post("/score", json={"npi": "1234567890"})
    assert resp.status_code == 503
