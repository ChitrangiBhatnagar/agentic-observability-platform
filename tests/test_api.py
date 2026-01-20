"""Tests for API endpoints."""
import pytest
from httpx import AsyncClient
from datetime import datetime

from src.api.app import create_app


@pytest.fixture
async def client():
    """Create test client."""
    app = create_app(include_metrics=False)
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio
class TestHealthEndpoints:
    """Tests for health endpoints."""
    
    async def test_health_check(self, client):
        """Test basic health check."""
        response = await client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
    
    async def test_liveness_probe(self, client):
        """Test liveness probe."""
        response = await client.get("/api/v1/health/live")
        assert response.status_code == 200
        assert response.json()["status"] == "alive"


@pytest.mark.asyncio
class TestAnomalyEndpoints:
    """Tests for anomaly endpoints."""
    
    async def test_detect_anomalies(self, client):
        """Test anomaly detection endpoint."""
        data = {
            "metric_name": "cpu_usage",
            "data_points": [
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "value": 95.5,
                    "labels": {"service": "api"}
                }
            ],
            "labels": {"service": "api", "env": "prod"}
        }
        
        response = await client.post("/api/v1/anomalies/detect", json=data)
        assert response.status_code == 200
        result = response.json()
        assert "anomalies" in result
    
    async def test_get_recent_anomalies(self, client):
        """Test getting recent anomalies."""
        response = await client.get("/api/v1/anomalies/recent?limit=10")
        assert response.status_code == 200
        data = response.json()
        assert "anomalies" in data


@pytest.mark.asyncio
class TestIncidentEndpoints:
    """Tests for incident endpoints."""
    
    async def test_list_incidents(self, client):
        """Test listing incidents."""
        response = await client.get("/api/v1/incidents")
        assert response.status_code == 200
        assert isinstance(response.json(), list)
    
    async def test_get_active_incidents(self, client):
        """Test getting active incidents."""
        response = await client.get("/api/v1/incidents/active")
        assert response.status_code == 200
        data = response.json()
        assert "active_count" in data


@pytest.mark.asyncio
class TestFeedbackEndpoints:
    """Tests for feedback endpoints."""
    
    async def test_submit_feedback(self, client):
        """Test feedback submission."""
        data = {
            "anomaly_id": "test-anom-1",
            "feedback_type": "true_positive",
            "comment": "Confirmed issue"
        }
        
        response = await client.post("/api/v1/feedback", json=data)
        assert response.status_code == 200
        result = response.json()
        assert result["status"] == "accepted"
