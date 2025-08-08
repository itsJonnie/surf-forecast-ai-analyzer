import os
import sys

import pytest
from fastapi.testclient import TestClient

# Adjust sys.path so we can import app modules when running tests with pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app")))

from app.main import app  # noqa: E402
from app import surf_logic  # noqa: E402


def test_retrieve_forecast_found():
    row = surf_logic.retrieve_forecast("Malibu", "2025-08-08")
    assert row is not None
    assert row["location"] == "Malibu"


def test_compute_quality_score_range():
    row = surf_logic.retrieve_forecast("Malibu", "2025-08-08")
    score = surf_logic.compute_quality_score(row)
    assert 1.0 <= score <= 10.0


def test_index_endpoint():
    client = TestClient(app)
    resp = client.get("/?location=Malibu&date=2025-08-08")
    assert resp.status_code == 200
    assert "Malibu" in resp.text


def test_chart_endpoint():
    client = TestClient(app)
    resp = client.get("/chart?location=Malibu")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"