"""
Anomaly detection endpoints.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel, Field

from src.api.app import get_orchestrator
from src.agents import AgentOrchestrator
from src.types import Severity, AnomalyType
from src.utils import generate_id, now_utc

router = APIRouter()


# Request/Response Models
class MetricDataPoint(BaseModel):
    """A single metric data point."""
    timestamp: datetime
    value: float
    labels: Dict[str, str] = Field(default_factory=dict)


class AnomalyDetectionRequest(BaseModel):
    """Request for anomaly detection."""
    metric_name: str
    data_points: List[MetricDataPoint]
    labels: Dict[str, str] = Field(default_factory=dict)
    detect_type: Optional[str] = None  # spike, drop, trend, etc.


class AnomalyResponse(BaseModel):
    """Response for detected anomaly."""
    id: str
    metric_name: str
    labels: Dict[str, str]
    anomaly_type: str
    severity: str
    score: float
    confidence: float
    value: float
    expected_value: Optional[float]
    deviation: float
    timestamp: datetime
    explanation: Optional[str] = None


class DetectionResultResponse(BaseModel):
    """Response for detection request."""
    request_id: str
    status: str
    anomalies: List[AnomalyResponse]
    processing_time_ms: float


class IngestMetricsRequest(BaseModel):
    """Request for ingesting metrics."""
    metrics: List[Dict[str, Any]]


# Endpoints
@router.post("/anomalies/detect", response_model=DetectionResultResponse)
async def detect_anomalies(
    request: AnomalyDetectionRequest,
    background_tasks: BackgroundTasks
) -> DetectionResultResponse:
    """
    Detect anomalies in provided metric data.
    
    Runs the multi-agent detection pipeline on the provided data
    and returns detected anomalies with explanations.
    """
    start_time = datetime.now()
    orchestrator = get_orchestrator()
    
    # Prepare data for processing
    anomaly_data = {
        "metric_name": request.metric_name,
        "labels": request.labels,
        "data_points": [
            {
                "timestamp": dp.timestamp.isoformat(),
                "value": dp.value,
                "labels": dp.labels
            }
            for dp in request.data_points
        ],
        "detect_type": request.detect_type
    }
    
    # Process through orchestrator
    result = await orchestrator.process_anomaly(anomaly_data)
    
    # Calculate processing time
    processing_time = (datetime.now() - start_time).total_seconds() * 1000
    
    # Build response
    anomalies = []
    for stage in result.get("pipeline_stages", []):
        if stage.get("agent") == "detection" and "anomalies" in stage:
            for anom in stage["anomalies"]:
                anomalies.append(AnomalyResponse(
                    id=anom.get("id", generate_id("anom")),
                    metric_name=request.metric_name,
                    labels=request.labels,
                    anomaly_type=anom.get("type", "unknown"),
                    severity=anom.get("severity", "medium"),
                    score=anom.get("score", 0),
                    confidence=anom.get("confidence", 0),
                    value=anom.get("value", 0),
                    expected_value=anom.get("expected_value"),
                    deviation=anom.get("deviation", 0),
                    timestamp=datetime.fromisoformat(anom.get("timestamp", now_utc().isoformat())),
                    explanation=anom.get("explanation")
                ))
    
    return DetectionResultResponse(
        request_id=generate_id("req"),
        status=result.get("status", "completed"),
        anomalies=anomalies,
        processing_time_ms=processing_time
    )


@router.post("/anomalies/ingest")
async def ingest_metrics(
    request: IngestMetricsRequest,
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """
    Ingest metrics for continuous anomaly detection.
    
    Metrics are queued for processing by the detection pipeline.
    """
    orchestrator = get_orchestrator()
    
    ingested_count = 0
    errors = []
    
    for metric in request.metrics:
        try:
            await orchestrator.process_anomaly(metric)
            ingested_count += 1
        except Exception as e:
            errors.append({"metric": metric.get("name"), "error": str(e)})
    
    return {
        "status": "accepted",
        "ingested": ingested_count,
        "errors": errors,
        "timestamp": now_utc().isoformat()
    }


@router.get("/anomalies/recent")
async def get_recent_anomalies(
    limit: int = Query(default=50, ge=1, le=500),
    severity: Optional[str] = Query(default=None),
    service: Optional[str] = Query(default=None),
    since_minutes: int = Query(default=60, ge=1, le=1440)
) -> Dict[str, Any]:
    """
    Get recently detected anomalies.
    
    Returns anomalies from the last N minutes with optional filters.
    """
    orchestrator = get_orchestrator()
    
    # Get from correlation agent
    stats = orchestrator.get_agent_stats()
    correlation_stats = stats.get("correlation", {})
    
    # Note: In a real implementation, this would query a database
    # For now, return stats-based info
    return {
        "anomalies": [],  # Would be populated from database
        "total": correlation_stats.get("active_anomalies", 0),
        "filters": {
            "limit": limit,
            "severity": severity,
            "service": service,
            "since_minutes": since_minutes
        }
    }


@router.get("/anomalies/{anomaly_id}")
async def get_anomaly(anomaly_id: str) -> Dict[str, Any]:
    """
    Get details of a specific anomaly.
    
    Returns full anomaly information including explanation.
    """
    orchestrator = get_orchestrator()
    
    # Note: Would query database in real implementation
    return {
        "id": anomaly_id,
        "status": "not_found",
        "message": "Anomaly details would be retrieved from database"
    }


@router.get("/anomalies/{anomaly_id}/explain")
async def explain_anomaly(anomaly_id: str) -> Dict[str, Any]:
    """
    Get detailed explanation for an anomaly.
    
    Uses SHAP and natural language to explain why
    the anomaly was detected.
    """
    # Would use explainability module
    return {
        "anomaly_id": anomaly_id,
        "explanation": {
            "summary": "Explanation would be generated here",
            "contributing_features": [],
            "similar_past_anomalies": []
        }
    }


@router.get("/anomalies/stats")
async def get_anomaly_stats() -> Dict[str, Any]:
    """
    Get anomaly detection statistics.
    
    Returns counts, rates, and trends.
    """
    orchestrator = get_orchestrator()
    stats = orchestrator.get_agent_stats()
    
    return {
        "detection": stats.get("detection", {}),
        "correlation": stats.get("correlation", {}),
        "timestamp": now_utc().isoformat()
    }
