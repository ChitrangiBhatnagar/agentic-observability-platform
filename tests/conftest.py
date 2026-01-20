"""Pytest configuration and fixtures."""
import pytest
import asyncio
from typing import AsyncGenerator, Generator
from datetime import datetime, timedelta
import numpy as np

from config import Settings
from src.agents import AgentOrchestrator, create_default_orchestrator
from src.types import Anomaly, Severity, AnomalyType


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_settings() -> Settings:
    """Test configuration."""
    return Settings(
        environment="testing",
        log_level="DEBUG",
        database_url="postgresql://test:test@localhost:5432/test_db",
        redis_url="redis://localhost:6379/1",
    )


@pytest.fixture
async def orchestrator() -> AsyncGenerator[AgentOrchestrator, None]:
    """Create test orchestrator."""
    orch = await create_default_orchestrator({
        "anomaly_threshold": 0.7,
        "correlation_window": 300,
        "auto_retrain": False,
    })
    await orch.start()
    yield orch
    await orch.stop()


@pytest.fixture
def sample_anomaly() -> Anomaly:
    """Sample anomaly for testing."""
    return Anomaly(
        id="test-anomaly-1",
        metric_name="cpu_usage",
        labels={"service": "api", "instance": "prod-1"},
        anomaly_type=AnomalyType.SPIKE,
        severity=Severity.HIGH,
        ensemble_score=0.85,
        confidence=0.9,
        value=95.5,
        expected_value=45.2,
        deviation=0.52,
        timestamp=datetime.utcnow(),
    )


@pytest.fixture
def sample_time_series() -> np.ndarray:
    """Generate sample time series data."""
    np.random.seed(42)
    t = np.linspace(0, 100, 1000)
    # Normal pattern + seasonal + noise
    trend = 50 + 0.1 * t
    seasonal = 10 * np.sin(2 * np.pi * t / 24)
    noise = np.random.normal(0, 2, len(t))
    return trend + seasonal + noise


@pytest.fixture
def anomalous_time_series() -> np.ndarray:
    """Generate time series with anomalies."""
    np.random.seed(42)
    t = np.linspace(0, 100, 1000)
    normal = 50 + 10 * np.sin(2 * np.pi * t / 24) + np.random.normal(0, 2, len(t))
    
    # Inject anomalies
    normal[500:510] += 30  # Spike
    normal[700:720] -= 25  # Drop
    
    return normal
