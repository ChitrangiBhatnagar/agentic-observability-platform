"""Main entry point for the Agentic Observability Platform."""
import uvicorn
from config import get_settings


def main():
    """Run the application server."""
    settings = get_settings()
    
    uvicorn.run(
        "src.api.app:create_app",
        factory=True,
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.environment == "development",
        workers=settings.api_workers if settings.environment == "production" else 1,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
