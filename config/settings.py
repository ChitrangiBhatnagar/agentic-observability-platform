"""
Configuration settings for the Agentic Observability Platform.
Uses pydantic-settings for type-safe configuration management.
"""

from enum import Enum
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    """Deployment environment."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(str, Enum):
    """Log levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class DatabaseSettings(BaseSettings):
    """Database configuration."""
    model_config = SettingsConfigDict(env_prefix="DB_")
    
    host: str = Field(default="localhost", description="Database host")
    port: int = Field(default=5432, description="Database port")
    name: str = Field(default="observability", description="Database name")
    user: str = Field(default="postgres", description="Database user")
    password: str = Field(default="postgres", description="Database password")
    pool_size: int = Field(default=10, description="Connection pool size")
    
    @property
    def url(self) -> str:
        """Get database connection URL."""
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"


class RedisSettings(BaseSettings):
    """Redis configuration for caching and streaming."""
    model_config = SettingsConfigDict(env_prefix="REDIS_")
    
    host: str = Field(default="localhost", description="Redis host")
    port: int = Field(default=6379, description="Redis port")
    password: Optional[str] = Field(default=None, description="Redis password")
    db: int = Field(default=0, description="Redis database number")
    
    @property
    def url(self) -> str:
        """Get Redis connection URL."""
        if self.password:
            return f"redis://:{self.password}@{self.host}:{self.port}/{self.db}"
        return f"redis://{self.host}:{self.port}/{self.db}"


class PrometheusSettings(BaseSettings):
    """Prometheus configuration."""
    model_config = SettingsConfigDict(env_prefix="PROMETHEUS_")
    
    url: str = Field(default="http://localhost:9090", description="Prometheus server URL")
    scrape_interval: int = Field(default=15, description="Scrape interval in seconds")
    query_timeout: int = Field(default=30, description="Query timeout in seconds")


class KafkaSettings(BaseSettings):
    """Kafka configuration for event streaming."""
    model_config = SettingsConfigDict(env_prefix="KAFKA_")
    
    bootstrap_servers: str = Field(default="localhost:9092", description="Kafka bootstrap servers")
    consumer_group: str = Field(default="observability-platform", description="Consumer group ID")
    topics_metrics: str = Field(default="metrics", description="Metrics topic")
    topics_alerts: str = Field(default="alerts", description="Alerts topic")
    topics_anomalies: str = Field(default="anomalies", description="Anomalies topic")


class MLSettings(BaseSettings):
    """Machine Learning configuration."""
    model_config = SettingsConfigDict(env_prefix="ML_")
    
    model_registry_path: str = Field(default="./models/registry", description="Model registry path")
    trained_models_path: str = Field(default="./models/trained", description="Trained models path")
    batch_size: int = Field(default=64, description="Default batch size")
    window_size: int = Field(default=60, description="Default sliding window size")
    anomaly_threshold: float = Field(default=0.95, description="Anomaly threshold")
    retraining_interval: int = Field(default=86400, description="Retraining interval in seconds")
    
    # Model-specific settings
    isolation_forest_contamination: float = Field(default=0.01, description="IF contamination")
    autoencoder_latent_dim: int = Field(default=32, description="Autoencoder latent dimension")
    lstm_hidden_size: int = Field(default=64, description="LSTM hidden size")


class AgentSettings(BaseSettings):
    """Agentic AI configuration."""
    model_config = SettingsConfigDict(env_prefix="AGENT_")
    
    memory_ttl: int = Field(default=3600, description="Agent memory TTL in seconds")
    decision_confidence_threshold: float = Field(default=0.7, description="Decision confidence threshold")
    max_concurrent_detections: int = Field(default=100, description="Max concurrent detections")
    correlation_window: int = Field(default=300, description="Correlation window in seconds")
    root_cause_max_depth: int = Field(default=5, description="Max depth for root cause analysis")
    feedback_weight: float = Field(default=0.1, description="Weight for operator feedback")


class APISettings(BaseSettings):
    """API server configuration."""
    model_config = SettingsConfigDict(env_prefix="API_")
    
    host: str = Field(default="0.0.0.0", description="API host")
    port: int = Field(default=8000, description="API port")
    workers: int = Field(default=4, description="Number of workers")
    reload: bool = Field(default=False, description="Auto-reload on changes")
    cors_origins: str = Field(default="*", description="CORS origins")


class Settings(BaseSettings):
    """Main application settings."""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    # Application
    app_name: str = Field(default="Agentic Observability Platform", description="Application name")
    app_version: str = Field(default="1.0.0", description="Application version")
    environment: Environment = Field(default=Environment.DEVELOPMENT, description="Environment")
    debug: bool = Field(default=True, description="Debug mode")
    log_level: LogLevel = Field(default=LogLevel.INFO, description="Log level")
    
    # Sub-configurations
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    prometheus: PrometheusSettings = Field(default_factory=PrometheusSettings)
    kafka: KafkaSettings = Field(default_factory=KafkaSettings)
    ml: MLSettings = Field(default_factory=MLSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    api: APISettings = Field(default_factory=APISettings)


# Global settings instance
settings = Settings()
