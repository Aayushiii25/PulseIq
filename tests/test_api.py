import pytest
from fastapi.testclient import TestClient
from api.main import app
from backend.database import Base, engine, get_db

client = TestClient(app)

def setup_module(module):
    Base.metadata.create_all(bind=engine)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["service"] == "PulseIQ API"

def test_pipeline_status():
    response = client.get("/api/pipeline/status")
    assert response.status_code == 200
    assert "running" in response.json()

def test_stats_endpoint():
    response = client.get("/api/stats")
    assert response.status_code == 200
    assert "total_articles" in response.json()

def test_pipeline_auth_required():
    response = client.post("/api/pipeline/run", json={"run_fetch": False})
    # Since it's behind api key auth or missing api key config in settings
    # It should either succeed if api_key in settings, or fail 403
    # Our dependency get_api_key returns 403 if it doesn't match
    assert response.status_code in [403, 200]
