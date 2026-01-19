"""
Base classes for anomaly detection models.
Defines the interface all detectors must implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pickle
from pathlib import Path

from src.types import ModelType, AnomalyType, Severity, ContributingFeature
from src.utils import get_logger, now_utc, generate_id

logger = get_logger(__name__)


@dataclass
class DetectionResult:
    """Result from an anomaly detection model."""
    
    # Core result
    is_anomaly: bool
    anomaly_score: float  # 0-1, higher = more anomalous
    confidence: float  # 0-1, model confidence
    
    # Classification
    anomaly_type: Optional[AnomalyType] = None
    severity: Severity = Severity.LOW
    
    # Context
    threshold: float = 0.5
    expected_value: Optional[float] = None
    deviation: Optional[float] = None
    
    # Explanation
    contributing_features: List[ContributingFeature] = field(default_factory=list)
    explanation: Optional[str] = None
    
    # Metadata
    model_type: Optional[ModelType] = None
    model_version: Optional[str] = None
    detection_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_anomaly": self.is_anomaly,
            "anomaly_score": self.anomaly_score,
            "confidence": self.confidence,
            "anomaly_type": self.anomaly_type.value if self.anomaly_type else None,
            "severity": self.severity.value,
            "threshold": self.threshold,
            "expected_value": self.expected_value,
            "deviation": self.deviation,
            "contributing_features": [
                {
                    "name": cf.name,
                    "value": cf.value,
                    "importance": cf.importance,
                    "expected_range": cf.expected_range,
                }
                for cf in self.contributing_features
            ],
            "explanation": self.explanation,
            "model_type": self.model_type.value if self.model_type else None,
            "model_version": self.model_version,
            "detection_time_ms": self.detection_time_ms,
        }


@dataclass
class ModelMetadata:
    """Metadata for a trained model."""
    model_id: str
    model_type: ModelType
    version: str
    created_at: datetime
    
    # Training info
    training_samples: int = 0
    training_duration_s: float = 0.0
    
    # Performance metrics
    validation_precision: float = 0.0
    validation_recall: float = 0.0
    validation_f1: float = 0.0
    
    # Configuration
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    feature_names: List[str] = field(default_factory=list)
    
    # Status
    is_active: bool = True
    last_used: Optional[datetime] = None


class BaseAnomalyDetector(ABC):
    """
    Abstract base class for all anomaly detection models.
    
    Defines the common interface for:
    - Training
    - Prediction
    - Model persistence
    - Performance tracking
    """
    
    def __init__(
        self,
        model_type: ModelType,
        version: str = "1.0.0",
        threshold: float = 0.5,
        **kwargs
    ):
        """
        Initialize the detector.
        
        Args:
            model_type: Type of the model
            version: Model version
            threshold: Anomaly detection threshold
            **kwargs: Additional model-specific parameters
        """
        self.model_type = model_type
        self.version = version
        self.threshold = threshold
        self.model_id = generate_id(f"{model_type.value}")
        
        # Model state
        self._is_fitted = False
        self._feature_names: List[str] = []
        self._training_stats: Dict[str, float] = {}
        
        # Metadata
        self.metadata = ModelMetadata(
            model_id=self.model_id,
            model_type=model_type,
            version=version,
            created_at=now_utc(),
            hyperparameters=kwargs,
        )
        
        logger.info(
            "Initialized detector",
            model_type=model_type.value,
            model_id=self.model_id
        )
    
    @property
    def is_fitted(self) -> bool:
        """Check if model is fitted."""
        return self._is_fitted
    
    @property
    def feature_names(self) -> List[str]:
        """Get feature names used for training."""
        return self._feature_names
    
    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None
    ) -> "BaseAnomalyDetector":
        """
        Fit the model on training data.
        
        Args:
            X: Training features (n_samples, n_features)
            y: Optional labels (for supervised methods)
            feature_names: Names of features
            
        Returns:
            Self
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomaly scores.
        
        Args:
            X: Input features (n_samples, n_features)
            
        Returns:
            Anomaly scores (n_samples,) in range [0, 1]
        """
        pass
    
    def detect(
        self,
        X: np.ndarray,
        return_details: bool = True
    ) -> List[DetectionResult]:
        """
        Detect anomalies and return detailed results.
        
        Args:
            X: Input features (n_samples, n_features)
            return_details: Whether to include detailed explanations
            
        Returns:
            List of DetectionResult objects
        """
        import time
        start_time = time.time()
        
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before detection")
        
        X = np.atleast_2d(X)
        scores = self.predict(X)
        
        detection_time = (time.time() - start_time) * 1000 / len(X)
        
        results = []
        for i, score in enumerate(scores):
            is_anomaly = score >= self.threshold
            
            # Calculate severity based on score
            if score >= 0.95:
                severity = Severity.CRITICAL
            elif score >= 0.85:
                severity = Severity.HIGH
            elif score >= 0.7:
                severity = Severity.MEDIUM
            else:
                severity = Severity.LOW
            
            # Calculate confidence
            confidence = self._calculate_confidence(score)
            
            # Get contributing features if available
            contributing = []
            if return_details and hasattr(self, '_get_contributing_features'):
                contributing = self._get_contributing_features(X[i])
            
            result = DetectionResult(
                is_anomaly=is_anomaly,
                anomaly_score=float(score),
                confidence=confidence,
                severity=severity if is_anomaly else Severity.LOW,
                threshold=self.threshold,
                model_type=self.model_type,
                model_version=self.version,
                detection_time_ms=detection_time,
                contributing_features=contributing,
            )
            results.append(result)
        
        return results
    
    def detect_single(
        self,
        x: np.ndarray,
        return_details: bool = True
    ) -> DetectionResult:
        """
        Detect anomaly for a single sample.
        
        Args:
            x: Input features (n_features,)
            return_details: Whether to include detailed explanations
            
        Returns:
            DetectionResult object
        """
        x = np.atleast_2d(x)
        results = self.detect(x, return_details)
        return results[0]
    
    def _calculate_confidence(self, score: float) -> float:
        """
        Calculate detection confidence based on score.
        
        Higher scores far from threshold = higher confidence.
        """
        distance_from_threshold = abs(score - self.threshold)
        max_distance = max(self.threshold, 1 - self.threshold)
        confidence = min(1.0, distance_from_threshold / max_distance + 0.5)
        return confidence
    
    def partial_fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None
    ) -> "BaseAnomalyDetector":
        """
        Incrementally fit the model (online learning).
        
        Default implementation raises NotImplementedError.
        Override in subclasses that support online learning.
        
        Args:
            X: New training data
            y: Optional labels
            
        Returns:
            Self
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support online learning"
        )
    
    def save(self, path: str) -> None:
        """
        Save model to disk.
        
        Args:
            path: File path for saving
        """
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            "model_type": self.model_type.value,
            "version": self.version,
            "model_id": self.model_id,
            "threshold": self.threshold,
            "is_fitted": self._is_fitted,
            "feature_names": self._feature_names,
            "training_stats": self._training_stats,
            "metadata": self.metadata,
            "model_state": self._get_model_state(),
        }
        
        with open(save_path, "wb") as f:
            pickle.dump(model_data, f)
        
        logger.info("Saved model", path=str(save_path), model_id=self.model_id)
    
    @classmethod
    def load(cls, path: str) -> "BaseAnomalyDetector":
        """
        Load model from disk.
        
        Args:
            path: File path to load from
            
        Returns:
            Loaded model instance
        """
        with open(path, "rb") as f:
            model_data = pickle.load(f)
        
        # Create instance
        instance = cls(
            model_type=ModelType(model_data["model_type"]),
            version=model_data["version"],
            threshold=model_data["threshold"],
        )
        
        instance.model_id = model_data["model_id"]
        instance._is_fitted = model_data["is_fitted"]
        instance._feature_names = model_data["feature_names"]
        instance._training_stats = model_data["training_stats"]
        instance.metadata = model_data["metadata"]
        instance._set_model_state(model_data["model_state"])
        
        logger.info("Loaded model", path=path, model_id=instance.model_id)
        return instance
    
    @abstractmethod
    def _get_model_state(self) -> Dict[str, Any]:
        """Get model-specific state for serialization."""
        pass
    
    @abstractmethod
    def _set_model_state(self, state: Dict[str, Any]) -> None:
        """Set model-specific state from deserialization."""
        pass
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return {
            "model_type": self.model_type.value,
            "version": self.version,
            "threshold": self.threshold,
            "model_id": self.model_id,
            "is_fitted": self._is_fitted,
        }
    
    def set_threshold(self, threshold: float) -> None:
        """Set the anomaly detection threshold."""
        if not 0 <= threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1")
        self.threshold = threshold
    
    def update_metadata(self, **kwargs) -> None:
        """Update model metadata."""
        for key, value in kwargs.items():
            if hasattr(self.metadata, key):
                setattr(self.metadata, key, value)


class AdaptiveThresholdMixin:
    """
    Mixin for adaptive threshold adjustment.
    
    Automatically adjusts detection threshold based on
    recent detection patterns.
    """
    
    def __init__(
        self,
        target_anomaly_rate: float = 0.01,
        adaptation_rate: float = 0.1,
        min_threshold: float = 0.3,
        max_threshold: float = 0.95,
        **kwargs
    ):
        self.target_anomaly_rate = target_anomaly_rate
        self.adaptation_rate = adaptation_rate
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self._recent_scores: List[float] = []
        self._score_window = 1000
        super().__init__(**kwargs)
    
    def adapt_threshold(self, scores: np.ndarray) -> float:
        """
        Adapt threshold based on recent scores.
        
        Args:
            scores: Recent anomaly scores
            
        Returns:
            New threshold value
        """
        self._recent_scores.extend(scores.tolist())
        if len(self._recent_scores) > self._score_window:
            self._recent_scores = self._recent_scores[-self._score_window:]
        
        if len(self._recent_scores) < 100:
            return self.threshold
        
        # Calculate current anomaly rate
        current_rate = np.mean(np.array(self._recent_scores) >= self.threshold)
        
        # Adjust threshold
        if current_rate > self.target_anomaly_rate:
            # Too many anomalies, raise threshold
            new_threshold = self.threshold + self.adaptation_rate * (current_rate - self.target_anomaly_rate)
        else:
            # Too few anomalies, lower threshold
            new_threshold = self.threshold - self.adaptation_rate * (self.target_anomaly_rate - current_rate)
        
        # Clamp to valid range
        new_threshold = max(self.min_threshold, min(self.max_threshold, new_threshold))
        
        if abs(new_threshold - self.threshold) > 0.01:
            logger.info(
                "Adapted threshold",
                old_threshold=self.threshold,
                new_threshold=new_threshold,
                current_rate=current_rate
            )
            self.threshold = new_threshold
        
        return self.threshold
