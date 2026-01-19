"""
FastAPI Application Factory.
"""

from contextlib import asynccontextmanager
from typing import Optional
import asyncio

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from prometheus_client import make_asgi_app
import structlog

from config import Settings
from src.agents import create_default_orchestrator, AgentOrchestrator
from src.utils import setup_logging, get_logger

logger = get_logger(__name__)


class AppState:
    """Application state holder."""
    
    def __init__(self):
        self.settings: Optional[Settings] = None
        self.orchestrator: Optional[AgentOrchestrator] = None
        self.startup_complete: bool = False


# Global app state
app_state = AppState()


def get_orchestrator() -> AgentOrchestrator:
    """Get the agent orchestrator instance."""
    if app_state.orchestrator is None:
        raise RuntimeError("Orchestrator not initialized")
    return app_state.orchestrator


def get_settings() -> Settings:
    """Get settings instance."""
    if app_state.settings is None:
        app_state.settings = Settings()
    return app_state.settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    settings = get_settings()
    
    # Setup logging
    setup_logging(settings.log_level, settings.environment == "development")
    
    logger.info("Starting Agentic Observability Platform...")
    
    # Initialize orchestrator
    config = {
        "anomaly_threshold": settings.ml.anomaly_threshold,
        "correlation_window": settings.agent.memory_ttl_seconds,
        "auto_approve_low_risk": False,
        "min_samples_for_retrain": settings.ml.min_training_samples,
        "auto_retrain": True,
        "service_topology": {},  # Can be loaded from config
    }
    
    app_state.orchestrator = await create_default_orchestrator(config)
    await app_state.orchestrator.start()
    
    app_state.startup_complete = True
    logger.info("Platform started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down platform...")
    if app_state.orchestrator:
        await app_state.orchestrator.stop()
    
    logger.info("Platform shutdown complete")


def create_app(
    settings: Optional[Settings] = None,
    include_metrics: bool = True
) -> FastAPI:
    """
    Create FastAPI application.
    
    Args:
        settings: Optional settings override
        include_metrics: Include Prometheus metrics endpoint
        
    Returns:
        Configured FastAPI application
    """
    if settings:
        app_state.settings = settings
    else:
        settings = get_settings()
    
    app = FastAPI(
        title="Agentic Observability Platform",
        description=(
            "AI-Driven Observability & Anomaly Detection Platform with "
            "multi-agent architecture for intelligent alerting and root cause analysis."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan
    )
    
    # Add middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Include routers
    from src.api.routes import anomalies, health, feedback, incidents
    
    app.include_router(health.router, prefix="/api/v1", tags=["Health"])
    app.include_router(anomalies.router, prefix="/api/v1", tags=["Anomalies"])
    app.include_router(incidents.router, prefix="/api/v1", tags=["Incidents"])
    app.include_router(feedback.router, prefix="/api/v1", tags=["Feedback"])
    
    # Add Prometheus metrics endpoint
    if include_metrics:
        metrics_app = make_asgi_app()
        app.mount("/metrics", metrics_app)
    
    # Root endpoint
    @app.get("/", include_in_schema=False)
    async def root():
        return {
            "name": "Agentic Observability Platform",
            "version": "1.0.0",
            "status": "running" if app_state.startup_complete else "starting",
            "docs": "/docs"
        }
    
    return app


# Create default app instance
app = create_app()
