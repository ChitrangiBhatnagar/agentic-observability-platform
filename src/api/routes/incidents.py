"""
Incident management endpoints.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from src.api.app import get_orchestrator
from src.types import Severity
from src.utils import generate_id, now_utc

router = APIRouter()


# Request/Response Models
class IncidentResponse(BaseModel):
    """Response for an incident."""
    id: str
    title: str
    description: str
    severity: str
    status: str
    anomaly_ids: List[str]
    affected_services: List[str]
    created_at: datetime
    updated_at: Optional[datetime] = None


class RootCauseResponse(BaseModel):
    """Response for a root cause."""
    id: str
    category: str
    description: str
    probability: float
    evidence: List[str]
    affected_components: List[str]
    suggested_investigation: List[str]


class RecommendationResponse(BaseModel):
    """Response for a recommendation."""
    id: str
    title: str
    description: str
    action_type: str
    risk_level: str
    confidence: float
    expected_impact: str
    status: str = "pending"


class UpdateIncidentRequest(BaseModel):
    """Request to update an incident."""
    status: Optional[str] = None
    severity: Optional[str] = None
    notes: Optional[str] = None


# Endpoints
@router.get("/incidents", response_model=List[IncidentResponse])
async def list_incidents(
    status: Optional[str] = Query(default=None),
    severity: Optional[str] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200)
) -> List[IncidentResponse]:
    """
    List all incidents.
    
    Optionally filter by status and severity.
    """
    orchestrator = get_orchestrator()
    
    # Get from correlation agent
    stats = orchestrator.get_agent_stats()
    
    # Note: Would query database in real implementation
    return []


@router.get("/incidents/active")
async def get_active_incidents() -> Dict[str, Any]:
    """
    Get all active (open) incidents.
    
    Returns incidents requiring attention.
    """
    orchestrator = get_orchestrator()
    stats = orchestrator.get_agent_stats()
    correlation_stats = stats.get("correlation", {})
    
    return {
        "active_count": correlation_stats.get("active_incidents", 0),
        "incidents": [],  # Would be from database
        "timestamp": now_utc().isoformat()
    }


@router.get("/incidents/{incident_id}")
async def get_incident(incident_id: str) -> Dict[str, Any]:
    """
    Get details of a specific incident.
    
    Includes related anomalies, root causes, and recommendations.
    """
    orchestrator = get_orchestrator()
    
    # Get root causes
    root_causes = await orchestrator.get_root_causes(incident_id)
    
    # Get recommendations
    recommendations = await orchestrator.get_recommendations(incident_id)
    
    return {
        "id": incident_id,
        "root_causes": root_causes,
        "recommendations": [
            {
                "id": r.id,
                "title": r.title,
                "action_type": r.action_type.value if hasattr(r.action_type, 'value') else str(r.action_type),
                "confidence": r.confidence,
                "risk_level": r.risk_level
            }
            for r in recommendations
        ] if recommendations else [],
        "timeline": [],  # Would include timeline events
        "status": "active"
    }


@router.patch("/incidents/{incident_id}")
async def update_incident(
    incident_id: str,
    update: UpdateIncidentRequest
) -> Dict[str, Any]:
    """
    Update an incident.
    
    Can update status, severity, and add notes.
    """
    # Note: Would update in database
    return {
        "id": incident_id,
        "updated": True,
        "changes": update.model_dump(exclude_none=True),
        "timestamp": now_utc().isoformat()
    }


@router.post("/incidents/{incident_id}/acknowledge")
async def acknowledge_incident(incident_id: str) -> Dict[str, Any]:
    """
    Acknowledge an incident.
    
    Marks that someone is looking at the incident.
    """
    return {
        "id": incident_id,
        "status": "acknowledged",
        "acknowledged_at": now_utc().isoformat()
    }


@router.post("/incidents/{incident_id}/resolve")
async def resolve_incident(
    incident_id: str,
    resolution_notes: Optional[str] = None
) -> Dict[str, Any]:
    """
    Resolve an incident.
    
    Marks the incident as resolved.
    """
    return {
        "id": incident_id,
        "status": "resolved",
        "resolved_at": now_utc().isoformat(),
        "resolution_notes": resolution_notes
    }


@router.get("/incidents/{incident_id}/root-causes")
async def get_incident_root_causes(
    incident_id: str
) -> List[RootCauseResponse]:
    """
    Get root causes for an incident.
    
    Returns ranked probable causes with evidence.
    """
    orchestrator = get_orchestrator()
    causes = await orchestrator.get_root_causes(incident_id)
    
    if isinstance(causes, list):
        return [
            RootCauseResponse(
                id=c.id if hasattr(c, 'id') else generate_id("cause"),
                category=c.category.value if hasattr(c.category, 'value') else str(c.category),
                description=c.description,
                probability=c.probability,
                evidence=c.evidence,
                affected_components=c.affected_components,
                suggested_investigation=c.suggested_investigation
            )
            for c in causes
        ]
    return []


@router.get("/incidents/{incident_id}/recommendations")
async def get_incident_recommendations(
    incident_id: str
) -> List[RecommendationResponse]:
    """
    Get recommendations for an incident.
    
    Returns ranked remediation actions.
    """
    orchestrator = get_orchestrator()
    recommendations = await orchestrator.get_recommendations(incident_id)
    
    if isinstance(recommendations, list):
        return [
            RecommendationResponse(
                id=r.id,
                title=r.title,
                description=r.description,
                action_type=r.action_type.value if hasattr(r.action_type, 'value') else str(r.action_type),
                risk_level=r.risk_level,
                confidence=r.confidence,
                expected_impact=r.expected_impact,
                status=r.parameters.get("status", "pending") if hasattr(r, 'parameters') else "pending"
            )
            for r in recommendations
        ]
    return []


@router.post("/incidents/{incident_id}/recommendations/{rec_id}/approve")
async def approve_recommendation(
    incident_id: str,
    rec_id: str
) -> Dict[str, Any]:
    """
    Approve a recommendation for execution.
    """
    orchestrator = get_orchestrator()
    
    # Get recommendation agent
    rec_agent = orchestrator._agents.get("recommendation")
    if rec_agent:
        await rec_agent.process({
            "action": "approve",
            "recommendation_id": rec_id
        })
    
    return {
        "recommendation_id": rec_id,
        "status": "approved",
        "approved_at": now_utc().isoformat()
    }


@router.post("/incidents/{incident_id}/recommendations/{rec_id}/reject")
async def reject_recommendation(
    incident_id: str,
    rec_id: str,
    reason: Optional[str] = None
) -> Dict[str, Any]:
    """
    Reject a recommendation.
    """
    orchestrator = get_orchestrator()
    
    # Get recommendation agent
    rec_agent = orchestrator._agents.get("recommendation")
    if rec_agent:
        rec_agent.process({
            "action": "reject",
            "recommendation_id": rec_id,
            "reason": reason or "Rejected by operator"
        })
    
    return {
        "recommendation_id": rec_id,
        "status": "rejected",
        "reason": reason,
        "rejected_at": now_utc().isoformat()
    }


@router.get("/incidents/{incident_id}/timeline")
async def get_incident_timeline(incident_id: str) -> Dict[str, Any]:
    """
    Get timeline of events for an incident.
    
    Returns chronological sequence of events.
    """
    from src.explainability import TimelineReconstructor
    
    # Would fetch anomalies and events from database
    # Then use TimelineReconstructor to build timeline
    
    return {
        "incident_id": incident_id,
        "timeline": [],  # Would be populated
        "phases": [],
        "summary_stats": {}
    }


@router.get("/incidents/stats")
async def get_incident_stats() -> Dict[str, Any]:
    """
    Get incident statistics.
    
    Returns counts, MTTR, trends, etc.
    """
    orchestrator = get_orchestrator()
    stats = orchestrator.get_agent_stats()
    
    return {
        "total_incidents": 0,  # Would be from database
        "open_incidents": stats.get("correlation", {}).get("active_incidents", 0),
        "mttr_minutes": 0,  # Mean time to resolve
        "by_severity": {},
        "by_service": {},
        "timestamp": now_utc().isoformat()
    }
