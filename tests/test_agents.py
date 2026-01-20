"""Tests for agents."""
import pytest
from datetime import datetime

from src.agents import (
    DetectionAgent,
    CorrelationAgent,
    RootCauseAgent,
    RecommendationAgent,
    FeedbackAgent,
)
from src.types import FeedbackType, Severity


@pytest.mark.asyncio
class TestDetectionAgent:
    """Tests for Detection Agent."""
    
    async def test_process_anomaly(self, sample_anomaly):
        """Test anomaly processing."""
        agent = DetectionAgent()
        
        result = await agent.process({
            "metric_name": "cpu_usage",
            "value": 95.5,
            "expected_value": 45.2,
        })
        
        assert result is not None
    
    async def test_model_selection(self):
        """Test contextual model selection."""
        agent = DetectionAgent()
        
        # Seasonal metric
        model = agent._select_model({
            "has_seasonality": True,
            "dimensionality": "low",
        })
        assert model in ["stl_esd", "zscore"]


@pytest.mark.asyncio
class TestCorrelationAgent:
    """Tests for Correlation Agent."""
    
    async def test_correlation_calculation(self, sample_anomaly):
        """Test anomaly correlation."""
        agent = CorrelationAgent()
        
        # Create correlated anomaly
        anomaly2 = sample_anomaly.model_copy()
        anomaly2.id = "test-anomaly-2"
        anomaly2.metric_name = "memory_usage"
        
        await agent._process_anomaly(sample_anomaly)
        await agent._process_anomaly(anomaly2)
        
        assert len(agent._active_anomalies) == 2
    
    async def test_cluster_creation(self):
        """Test anomaly clustering."""
        agent = CorrelationAgent(min_correlation_score=0.5)
        
        stats = agent.get_correlation_stats()
        assert "active_anomalies" in stats


@pytest.mark.asyncio
class TestRootCauseAgent:
    """Tests for Root Cause Agent."""
    
    async def test_pattern_matching(self):
        """Test root cause pattern matching."""
        agent = RootCauseAgent()
        
        causes = await agent._analyze_root_causes(
            "incident-1",
            [
                {
                    "metric_name": "memory_usage_bytes",
                    "severity": "high",
                    "labels": {"service": "api"},
                }
            ]
        )
        
        assert len(causes) > 0
        assert causes[0].category is not None
    
    async def test_topology_analysis(self):
        """Test topology-based analysis."""
        topology = {
            "backend": ["database", "cache"],
            "frontend": ["backend"],
        }
        
        agent = RootCauseAgent(service_topology=topology)
        
        causes = await agent._analyze_with_topology(["frontend", "backend"])
        # Should identify backend as potential root
        assert any("backend" in c.description for c in causes)


@pytest.mark.asyncio
class TestRecommendationAgent:
    """Tests for Recommendation Agent."""
    
    async def test_recommendation_generation(self):
        """Test recommendation generation."""
        agent = RecommendationAgent()
        
        from src.agents.recommendation_agent import RecommendationContext
        
        context = RecommendationContext(
            incident_id="inc-1",
            causes=[
                {
                    "category": "capacity",
                    "description": "High CPU usage",
                    "probability": 0.8,
                }
            ],
            affected_services=["api"],
            severity=Severity.HIGH,
            time_since_start=300,
        )
        
        recs = await agent._generate_recommendations(context)
        assert len(recs) > 0
        assert recs[0].action_type is not None
    
    async def test_action_templates(self):
        """Test action templates."""
        agent = RecommendationAgent()
        
        templates = agent.get_templates()
        assert len(templates) > 0
        assert any(t["name"] == "Restart Service" for t in templates)


@pytest.mark.asyncio
class TestFeedbackAgent:
    """Tests for Feedback Agent."""
    
    async def test_feedback_processing(self):
        """Test feedback submission."""
        agent = FeedbackAgent()
        
        result = await agent._process_feedback({
            "anomaly_id": "anom-1",
            "type": FeedbackType.TRUE_POSITIVE.value,
            "comment": "Confirmed anomaly",
            "features": {"cpu": 95.5, "memory": 80.2},
            "metric": "cpu_usage",
        })
        
        assert result.decision is not None
    
    async def test_model_metrics_update(self):
        """Test model performance tracking."""
        agent = FeedbackAgent()
        
        agent._update_model_metrics("test_model", FeedbackType.TRUE_POSITIVE)
        agent._update_model_metrics("test_model", FeedbackType.FALSE_POSITIVE)
        
        perf = agent.get_model_performance("test_model")
        assert perf["true_positives"] == 1
        assert perf["false_positives"] == 1
        assert perf["precision"] > 0
