"""
Feedback submission endpoints.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from src.api.app import get_orchestrator
from src.types import FeedbackType
from src.utils import generate_id, now_utc

router = APIRouter()


# Request/Response Models
class FeedbackRequest(BaseModel):
    """Request to submit feedback on an anomaly."""
    anomaly_id: str
    feedback_type: str = Field(
        ...,
        description="Type of feedback: true_positive, false_positive, false_negative, missed_anomaly"
    )
    comment: Optional[str] = None
    model_id: Optional[str] = None
    features: Optional[Dict[str, float]] = None
    metric: Optional[str] = None
    labeler: Optional[str] = None


class FeedbackResponse(BaseModel):
    """Response for submitted feedback."""
    id: str
    status: str
    message: str


class ActionOutcomeRequest(BaseModel):
    """Request to record action outcome."""
    recommendation_id: str
    success: bool
    time_to_resolve_seconds: Optional[float] = None
    notes: Optional[str] = None


class ModelPerformanceResponse(BaseModel):
    """Response for model performance metrics."""
    model_id: str
    precision: float
    recall: float
    f1_score: float
    true_positives: int
    false_positives: int
    false_negatives: int


# Endpoints
@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest) -> FeedbackResponse:
    """
    Submit feedback on an anomaly detection.
    
    This feedback is used to improve model accuracy
    and may trigger model retraining.
    """
    orchestrator = get_orchestrator()
    
    # Validate feedback type
    try:
        feedback_type = FeedbackType(request.feedback_type)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid feedback type. Must be one of: {[t.value for t in FeedbackType]}"
        )
    
    # Submit through orchestrator
    result = await orchestrator.submit_feedback({
        "anomaly_id": request.anomaly_id,
        "type": request.feedback_type,
        "comment": request.comment,
        "model_id": request.model_id,
        "features": request.features,
        "metric": request.metric,
        "labeler": request.labeler or "api"
    })
    
    return FeedbackResponse(
        id=generate_id("fb"),
        status="accepted",
        message=result.get("decision", "Feedback submitted successfully")
    )


@router.post("/feedback/action-outcome")
async def record_action_outcome(
    request: ActionOutcomeRequest
) -> Dict[str, Any]:
    """
    Record the outcome of a recommended action.
    
    This feedback improves recommendation effectiveness scores.
    """
    orchestrator = get_orchestrator()
    
    # Get recommendation agent
    rec_agent = orchestrator._agents.get("recommendation")
    if rec_agent and hasattr(rec_agent, 'record_outcome'):
        rec_agent.record_outcome(
            request.recommendation_id,
            request.success,
            request.time_to_resolve_seconds
        )
    
    return {
        "status": "recorded",
        "recommendation_id": request.recommendation_id,
        "success": request.success,
        "timestamp": now_utc().isoformat()
    }


@router.post("/feedback/root-cause")
async def record_root_cause_feedback(
    cause_id: str,
    was_correct: bool,
    notes: Optional[str] = None
) -> Dict[str, Any]:
    """
    Record feedback on a root cause prediction.
    
    This feedback improves root cause analysis accuracy.
    """
    orchestrator = get_orchestrator()
    
    # Get root cause agent
    rca_agent = orchestrator._agents.get("root_cause")
    if rca_agent and hasattr(rca_agent, 'record_feedback'):
        rca_agent.record_feedback(cause_id, was_correct)
    
    return {
        "status": "recorded",
        "cause_id": cause_id,
        "was_correct": was_correct,
        "timestamp": now_utc().isoformat()
    }


@router.get("/feedback/summary")
async def get_feedback_summary() -> Dict[str, Any]:
    """
    Get summary of submitted feedback.
    
    Returns feedback counts by type and recent activity.
    """
    orchestrator = get_orchestrator()
    stats = orchestrator.get_agent_stats()
    
    feedback_stats = stats.get("feedback", {})
    
    return {
        "summary": feedback_stats.get("feedback_summary", {}),
        "timestamp": now_utc().isoformat()
    }


@router.get("/feedback/model-performance")
async def get_model_performance(
    model_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get model performance based on feedback.
    
    Returns precision, recall, and F1 scores.
    """
    orchestrator = get_orchestrator()
    stats = orchestrator.get_agent_stats()
    
    feedback_stats = stats.get("feedback", {})
    model_perf = feedback_stats.get("model_performance", {})
    
    if model_id:
        return model_perf.get(model_id, {
            "precision": 0,
            "recall": 0,
            "f1_score": 0,
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": 0
        })
    
    return model_perf


@router.get("/feedback/retrain-status")
async def get_retrain_status() -> Dict[str, Any]:
    """
    Get status of model retraining requests.
    
    Returns pending and completed retrain requests.
    """
    orchestrator = get_orchestrator()
    stats = orchestrator.get_agent_stats()
    
    feedback_stats = stats.get("feedback", {})
    
    return {
        "retrain_requests": feedback_stats.get("retrain_requests", []),
        "pending_count": feedback_stats.get("feedback_summary", {}).get("pending_retrains", 0),
        "timestamp": now_utc().isoformat()
    }


@router.post("/feedback/trigger-retrain")
async def trigger_retrain(
    model_id: str
) -> Dict[str, Any]:
    """
    Manually trigger model retraining.
    
    Forces a retrain for the specified model.
    """
    orchestrator = get_orchestrator()
    
    # Get feedback agent
    feedback_agent = orchestrator._agents.get("feedback")
    if feedback_agent:
        result = await feedback_agent.process({
            "action": "trigger_retrain",
            "model_id": model_id
        })
        
        return {
            "status": "triggered",
            "model_id": model_id,
            "timestamp": now_utc().isoformat()
        }
    
    raise HTTPException(
        status_code=503,
        detail="Feedback agent not available"
    )


@router.get("/feedback/samples")
async def get_labeled_samples(
    metric: Optional[str] = None,
    limit: int = Query(default=100, ge=1, le=1000)
) -> Dict[str, Any]:
    """
    Get labeled samples for a metric.
    
    Returns samples that have been labeled through feedback.
    """
    orchestrator = get_orchestrator()
    
    # Get feedback agent
    feedback_agent = orchestrator._agents.get("feedback")
    if feedback_agent:
        result = await feedback_agent.process({
            "action": "get_samples",
            "metric": metric,
            "limit": limit
        })
        
        return {
            "samples": [],  # Would return actual samples
            "total": 0,
            "filters": {
                "metric": metric,
                "limit": limit
            }
        }
    
    return {
        "samples": [],
        "total": 0
    }
