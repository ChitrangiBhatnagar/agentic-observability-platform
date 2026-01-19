"""
Health check endpoints.
"""

from datetime import datetime
from typing import Dict, Any

from fastapi import APIRouter, Depends

from src.api.app import get_orchestrator, get_settings, app_state
from src.agents import AgentOrchestrator

router = APIRouter()


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Basic health check endpoint.
    
    Returns service health status.
    """
    return {
        "status": "healthy" if app_state.startup_complete else "starting",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }


@router.get("/health/live")
async def liveness_probe() -> Dict[str, str]:
    """
    Kubernetes liveness probe.
    
    Returns 200 if the service is alive.
    """
    return {"status": "alive"}


@router.get("/health/ready")
async def readiness_probe() -> Dict[str, Any]:
    """
    Kubernetes readiness probe.
    
    Returns 200 if the service is ready to accept traffic.
    """
    if not app_state.startup_complete:
        return {"status": "not_ready", "reason": "startup_incomplete"}
    
    if app_state.orchestrator is None:
        return {"status": "not_ready", "reason": "orchestrator_not_initialized"}
    
    return {"status": "ready"}


@router.get("/health/agents")
async def agent_health() -> Dict[str, Any]:
    """
    Get health status of all agents.
    
    Returns detailed health information for each agent.
    """
    orchestrator = get_orchestrator()
    return orchestrator.get_health()


@router.get("/metrics/agents")
async def agent_metrics() -> Dict[str, Any]:
    """
    Get metrics from the agent orchestrator.
    
    Returns message routing stats, decision counts, etc.
    """
    orchestrator = get_orchestrator()
    return orchestrator.get_metrics()


@router.get("/metrics/agents/stats")
async def agent_stats() -> Dict[str, Any]:
    """
    Get detailed statistics from all agents.
    
    Returns per-agent statistics and performance metrics.
    """
    orchestrator = get_orchestrator()
    return orchestrator.get_agent_stats()


@router.get("/config")
async def get_config() -> Dict[str, Any]:
    """
    Get current configuration (non-sensitive).
    
    Returns configuration values safe for display.
    """
    settings = get_settings()
    
    return {
        "environment": settings.environment,
        "log_level": settings.log_level,
        "ml": {
            "anomaly_threshold": settings.ml.anomaly_threshold,
            "ensemble_models": settings.ml.ensemble_models,
            "training_window_hours": settings.ml.training_window_hours,
        },
        "api": {
            "rate_limit_requests": settings.api.rate_limit_requests,
            "rate_limit_window": settings.api.rate_limit_window,
        },
        "agent": {
            "max_concurrent_agents": settings.agent.max_concurrent_agents,
            "message_queue_size": settings.agent.message_queue_size,
        }
    }
