"""Database initialization and utilities."""
import asyncio
from typing import Optional
import asyncpg
from contextlib import asynccontextmanager

from config import get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


class Database:
    """Database connection manager."""
    
    def __init__(self, url: str):
        self.url = url
        self.pool: Optional[asyncpg.Pool] = None
    
    async def connect(self, min_size: int = 5, max_size: int = 20):
        """Create connection pool."""
        logger.info("Creating database connection pool")
        self.pool = await asyncpg.create_pool(
            self.url,
            min_size=min_size,
            max_size=max_size,
            command_timeout=60,
        )
        logger.info(f"Database pool created (min={min_size}, max={max_size})")
    
    async def disconnect(self):
        """Close connection pool."""
        if self.pool:
            logger.info("Closing database connection pool")
            await self.pool.close()
            self.pool = None
    
    @asynccontextmanager
    async def acquire(self):
        """Acquire a connection from the pool."""
        async with self.pool.acquire() as conn:
            yield conn
    
    async def execute(self, query: str, *args):
        """Execute a query."""
        async with self.acquire() as conn:
            return await conn.execute(query, *args)
    
    async def fetch(self, query: str, *args):
        """Fetch multiple rows."""
        async with self.acquire() as conn:
            return await conn.fetch(query, *args)
    
    async def fetchrow(self, query: str, *args):
        """Fetch a single row."""
        async with self.acquire() as conn:
            return await conn.fetchrow(query, *args)
    
    async def fetchval(self, query: str, *args):
        """Fetch a single value."""
        async with self.acquire() as conn:
            return await conn.fetchval(query, *args)


# Repository classes for data access

class AnomalyRepository:
    """Repository for anomaly data."""
    
    def __init__(self, db: Database):
        self.db = db
    
    async def create(self, anomaly_data: dict) -> str:
        """Create a new anomaly record."""
        query = """
            INSERT INTO anomalies (
                id, metric_name, labels, anomaly_type, severity,
                ensemble_score, confidence, value, expected_value,
                deviation, timestamp, created_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, NOW())
            RETURNING id
        """
        return await self.db.fetchval(
            query,
            anomaly_data["id"],
            anomaly_data["metric_name"],
            anomaly_data["labels"],
            anomaly_data["anomaly_type"],
            anomaly_data["severity"],
            anomaly_data["ensemble_score"],
            anomaly_data["confidence"],
            anomaly_data["value"],
            anomaly_data["expected_value"],
            anomaly_data["deviation"],
            anomaly_data["timestamp"],
        )
    
    async def get_recent(self, limit: int = 100, severity: str = None):
        """Get recent anomalies."""
        query = "SELECT * FROM anomalies"
        args = []
        
        if severity:
            query += " WHERE severity = $1"
            args.append(severity)
        
        query += " ORDER BY timestamp DESC LIMIT $" + str(len(args) + 1)
        args.append(limit)
        
        return await self.db.fetch(query, *args)
    
    async def get_by_id(self, anomaly_id: str):
        """Get anomaly by ID."""
        query = "SELECT * FROM anomalies WHERE id = $1"
        return await self.db.fetchrow(query, anomaly_id)


class IncidentRepository:
    """Repository for incident data."""
    
    def __init__(self, db: Database):
        self.db = db
    
    async def create(self, incident_data: dict) -> str:
        """Create a new incident."""
        query = """
            INSERT INTO incidents (
                id, title, severity, status, affected_services,
                anomaly_count, correlation_score, started_at, created_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, NOW())
            RETURNING id
        """
        return await self.db.fetchval(
            query,
            incident_data["id"],
            incident_data["title"],
            incident_data["severity"],
            incident_data.get("status", "open"),
            incident_data["affected_services"],
            incident_data["anomaly_count"],
            incident_data["correlation_score"],
            incident_data["started_at"],
        )
    
    async def get_active(self):
        """Get active incidents."""
        query = """
            SELECT * FROM incidents 
            WHERE status IN ('open', 'investigating', 'acknowledged')
            ORDER BY severity DESC, started_at DESC
        """
        return await self.db.fetch(query)
    
    async def update_status(self, incident_id: str, status: str):
        """Update incident status."""
        query = "UPDATE incidents SET status = $1, updated_at = NOW() WHERE id = $2"
        await self.db.execute(query, status, incident_id)


class FeedbackRepository:
    """Repository for feedback data."""
    
    def __init__(self, db: Database):
        self.db = db
    
    async def create(self, feedback_data: dict) -> str:
        """Create feedback record."""
        query = """
            INSERT INTO feedback (
                id, anomaly_id, feedback_type, comment,
                operator_id, created_at
            ) VALUES ($1, $2, $3, $4, $5, NOW())
            RETURNING id
        """
        return await self.db.fetchval(
            query,
            feedback_data["id"],
            feedback_data["anomaly_id"],
            feedback_data["feedback_type"],
            feedback_data.get("comment"),
            feedback_data.get("operator_id"),
        )
    
    async def get_summary(self, days: int = 7):
        """Get feedback summary."""
        query = """
            SELECT 
                feedback_type,
                COUNT(*) as count
            FROM feedback
            WHERE created_at >= NOW() - INTERVAL '%s days'
            GROUP BY feedback_type
        """ % days
        return await self.db.fetch(query)


async def init_database():
    """Initialize database connection and tables."""
    settings = get_settings()
    db = Database(settings.database_url)
    
    try:
        await db.connect()
        logger.info("Database initialized successfully")
        return db
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise


async def close_database(db: Database):
    """Close database connection."""
    if db:
        await db.disconnect()


# Export
__all__ = [
    "Database",
    "AnomalyRepository",
    "IncidentRepository",
    "FeedbackRepository",
    "init_database",
    "close_database",
]
