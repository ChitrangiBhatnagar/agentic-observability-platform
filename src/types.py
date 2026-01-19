"""
Core data types and schemas for the Agentic Observability Platform.
Defines all the Pydantic models for metrics, anomalies, agents, and events.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, field_validator
import uuid


# =============================================================================
# Enums
# =============================================================================

class MetricType(str, Enum):
    """Type of metric being observed."""
    GAUGE = "gauge"
    COUNTER = "counter"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class AnomalyType(str, Enum):
    """Type of detected anomaly."""
    SPIKE = "spike"
    DROP = "drop"
    TREND_CHANGE = "trend_change"
    PATTERN_VIOLATION = "pattern_violation"
    LEVEL_SHIFT = "level_shift"
    VARIANCE_CHANGE = "variance_change"
    OUTLIER = "outlier"


class Severity(str, Enum):
    """Severity level of an anomaly or alert."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AgentType(str, Enum):
    """Types of agents in the system."""
    DETECTION = "detection"
    CORRELATION = "correlation"
    ROOT_CAUSE = "root_cause"
    RECOMMENDATION = "recommendation"
    FEEDBACK = "feedback"


class ModelType(str, Enum):
    """Types of anomaly detection models."""
    ZSCORE = "zscore"
    STL_ESD = "stl_esd"
    ISOLATION_FOREST = "isolation_forest"
    ONE_CLASS_SVM = "one_class_svm"
    LSTM_AUTOENCODER = "lstm_autoencoder"
    TRANSFORMER_AUTOENCODER = "transformer_autoencoder"
    PROPHET = "prophet"
    ENSEMBLE = "ensemble"


class ActionType(str, Enum):
    """Types of remediation actions."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    RESTART = "restart"
    ROLLBACK = "rollback"
    CONFIG_CHANGE = "config_change"
    ALERT = "alert"
    INVESTIGATE = "investigate"
    NO_ACTION = "no_action"


class FeedbackType(str, Enum):
    """Types of operator feedback."""
    TRUE_POSITIVE = "true_positive"
    FALSE_POSITIVE = "false_positive"
    MISSED_DETECTION = "missed_detection"
    SEVERITY_ADJUST = "severity_adjust"
    LABEL_CORRECTION = "label_correction"


# =============================================================================
# Base Models
# =============================================================================

class TimestampedModel(BaseModel):
    """Base model with timestamp."""
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class IdentifiedModel(BaseModel):
    """Base model with ID."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))


# =============================================================================
# Metric Models
# =============================================================================

class MetricLabel(BaseModel):
    """Label attached to a metric."""
    key: str
    value: str


class MetricDataPoint(TimestampedModel):
    """A single metric data point."""
    value: float
    labels: Dict[str, str] = Field(default_factory=dict)


class Metric(IdentifiedModel, TimestampedModel):
    """A metric with its metadata and current value."""
    name: str
    type: MetricType
    value: float
    labels: Dict[str, str] = Field(default_factory=dict)
    unit: Optional[str] = None
    description: Optional[str] = None
    
    @property
    def full_name(self) -> str:
        """Get full metric name with labels."""
        if not self.labels:
            return self.name
        label_str = ",".join(f'{k}="{v}"' for k, v in sorted(self.labels.items()))
        return f"{self.name}{{{label_str}}}"


class MetricSeries(BaseModel):
    """Time series of metric values."""
    metric_name: str
    labels: Dict[str, str] = Field(default_factory=dict)
    data_points: List[MetricDataPoint] = Field(default_factory=list)
    
    @property
    def values(self) -> List[float]:
        """Get list of values."""
        return [dp.value for dp in self.data_points]
    
    @property
    def timestamps(self) -> List[datetime]:
        """Get list of timestamps."""
        return [dp.timestamp for dp in self.data_points]


# =============================================================================
# Feature Models
# =============================================================================

class FeatureVector(TimestampedModel):
    """Extracted features for a time window."""
    metric_name: str
    labels: Dict[str, str] = Field(default_factory=dict)
    
    # Statistical features
    mean: float
    std: float
    min: float
    max: float
    median: float
    skewness: float
    kurtosis: float
    
    # Rolling features
    rolling_mean_5: float
    rolling_mean_15: float
    rolling_std_5: float
    rolling_std_15: float
    
    # Change features
    change_rate: float
    trend_slope: float
    
    # Seasonal features
    seasonal_component: Optional[float] = None
    residual_component: Optional[float] = None
    
    # Lag features
    lag_1: Optional[float] = None
    lag_5: Optional[float] = None
    lag_15: Optional[float] = None
    
    # Additional context
    window_size: int
    raw_values: List[float] = Field(default_factory=list)


# =============================================================================
# Anomaly Models
# =============================================================================

class ContributingFeature(BaseModel):
    """A feature that contributed to anomaly detection."""
    name: str
    value: float
    importance: float  # 0-1
    expected_range: tuple[float, float]


class AnomalyScore(BaseModel):
    """Anomaly score from a single model."""
    model_type: ModelType
    score: float  # 0-1
    threshold: float
    is_anomaly: bool
    confidence: float  # 0-1
    contributing_features: List[ContributingFeature] = Field(default_factory=list)


class Anomaly(IdentifiedModel, TimestampedModel):
    """A detected anomaly."""
    metric_name: str
    labels: Dict[str, str] = Field(default_factory=dict)
    anomaly_type: AnomalyType
    severity: Severity
    
    # Detection details
    scores: List[AnomalyScore] = Field(default_factory=list)
    ensemble_score: float
    confidence: float
    
    # Context
    value: float
    expected_value: float
    deviation: float
    
    # Analysis
    contributing_features: List[ContributingFeature] = Field(default_factory=list)
    explanation: Optional[str] = None
    
    # Correlation
    correlated_anomaly_ids: List[str] = Field(default_factory=list)
    root_cause_hypothesis: Optional[str] = None
    
    # Status
    acknowledged: bool = False
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    
    @property
    def is_high_priority(self) -> bool:
        """Check if this is a high priority anomaly."""
        return self.severity in [Severity.HIGH, Severity.CRITICAL] and self.confidence > 0.8


# =============================================================================
# Agent Models
# =============================================================================

class AgentMemoryEntry(TimestampedModel):
    """An entry in an agent's memory."""
    key: str
    value: Any
    ttl: Optional[int] = None  # Time to live in seconds
    access_count: int = 0


class AgentDecision(IdentifiedModel, TimestampedModel):
    """A decision made by an agent."""
    agent_type: AgentType
    agent_id: str
    
    # Decision details
    decision: str
    reasoning: str
    confidence: float
    
    # Context
    input_data: Dict[str, Any] = Field(default_factory=dict)
    alternatives_considered: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Outcome
    outcome: Optional[str] = None
    feedback: Optional[FeedbackType] = None


class AgentState(BaseModel):
    """Current state of an agent."""
    agent_type: AgentType
    agent_id: str
    is_active: bool = True
    
    # Performance metrics
    decisions_made: int = 0
    correct_decisions: int = 0
    average_confidence: float = 0.0
    
    # Memory
    short_term_memory: List[AgentMemoryEntry] = Field(default_factory=list)
    long_term_memory_size: int = 0
    
    # Timestamps
    last_active: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


# =============================================================================
# Recommendation Models
# =============================================================================

class RecommendedAction(IdentifiedModel, TimestampedModel):
    """A recommended remediation action."""
    action_type: ActionType
    target_service: str
    target_resource: Optional[str] = None
    
    # Details
    description: str
    reasoning: str
    confidence: float
    
    # Impact assessment
    expected_impact: str
    risk_level: Severity
    estimated_time: int  # seconds
    
    # Dependencies
    prerequisite_actions: List[str] = Field(default_factory=list)
    rollback_action: Optional[str] = None
    
    # Status
    approved: bool = False
    executed: bool = False
    execution_result: Optional[str] = None


# =============================================================================
# Incident Models
# =============================================================================

class IncidentEvent(TimestampedModel):
    """An event in an incident timeline."""
    event_type: str
    description: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Incident(IdentifiedModel, TimestampedModel):
    """A correlated incident grouping multiple anomalies."""
    title: str
    description: str
    severity: Severity
    
    # Related entities
    anomaly_ids: List[str] = Field(default_factory=list)
    affected_services: List[str] = Field(default_factory=list)
    
    # Analysis
    root_cause: Optional[str] = None
    root_cause_confidence: float = 0.0
    recommended_actions: List[RecommendedAction] = Field(default_factory=list)
    
    # Timeline
    events: List[IncidentEvent] = Field(default_factory=list)
    
    # Status
    status: str = "open"  # open, investigating, mitigating, resolved
    assigned_to: Optional[str] = None
    resolved_at: Optional[datetime] = None
    
    # Postmortem
    postmortem: Optional[str] = None
    lessons_learned: List[str] = Field(default_factory=list)


# =============================================================================
# Feedback Models
# =============================================================================

class OperatorFeedback(IdentifiedModel, TimestampedModel):
    """Feedback from an operator on a detection."""
    anomaly_id: str
    feedback_type: FeedbackType
    
    # Details
    comment: Optional[str] = None
    correct_severity: Optional[Severity] = None
    correct_label: Optional[str] = None
    
    # Operator info
    operator_id: str
    
    # Impact on model
    applied_to_training: bool = False


# =============================================================================
# Explanation Models
# =============================================================================

class FeatureExplanation(BaseModel):
    """SHAP-style explanation for a feature."""
    feature_name: str
    feature_value: float
    shap_value: float
    contribution: str  # "increases" or "decreases"


class AnomalyExplanation(BaseModel):
    """Human-readable explanation of an anomaly."""
    anomaly_id: str
    
    # Summary
    summary: str
    detailed_explanation: str
    
    # Feature contributions
    feature_explanations: List[FeatureExplanation] = Field(default_factory=list)
    
    # Visual aids
    attention_weights: Optional[Dict[str, float]] = None
    
    # Narrative
    why_anomalous: str
    historical_context: str
    similar_past_incidents: List[str] = Field(default_factory=list)


# =============================================================================
# System Models
# =============================================================================

class SystemHealth(TimestampedModel):
    """Overall system health status."""
    status: str  # healthy, degraded, unhealthy
    
    # Components
    components: Dict[str, str] = Field(default_factory=dict)
    
    # Metrics
    total_metrics_monitored: int
    total_anomalies_detected_24h: int
    false_positive_rate_24h: float
    mean_detection_time: float  # seconds
    
    # Agents
    active_agents: int
    agent_decisions_24h: int


class ModelPerformance(BaseModel):
    """Performance metrics for a model."""
    model_type: ModelType
    model_version: str
    
    # Metrics
    precision: float
    recall: float
    f1_score: float
    false_positive_rate: float
    
    # Timing
    avg_inference_time: float  # ms
    throughput: float  # predictions/second
    
    # Drift
    feature_drift_score: float
    prediction_drift_score: float
    
    # Meta
    training_data_size: int
    last_trained: datetime
    last_evaluated: datetime
