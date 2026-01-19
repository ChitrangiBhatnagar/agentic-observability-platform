"""
Detection Agent - Dynamically selects the best model for each metric.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.types import AgentType, AgentDecision, ModelType, Anomaly, AnomalyType, Severity
from src.models import (
    BaseAnomalyDetector,
    DetectionResult,
    ZScoreDetector,
    STLESDDetector,
    IsolationForestDetector,
    OneClassSVMDetector,
    LSTMAutoencoderDetector,
    EnsembleDetector,
)
from src.utils import get_logger, generate_id, now_utc
from .base import BaseAgent, AgentMessage, MessagePriority

logger = get_logger(__name__)


class DetectionAgent(BaseAgent):
    """
    Detection Agent that dynamically selects the best anomaly detection model.
    
    Responsibilities:
    - Maintain a portfolio of detection models
    - Select optimal model based on metric characteristics
    - Adapt model selection based on feedback
    - Report detected anomalies to other agents
    """
    
    def __init__(
        self,
        models: Optional[Dict[ModelType, BaseAnomalyDetector]] = None,
        default_model: ModelType = ModelType.ENSEMBLE,
        selection_strategy: str = "adaptive",
        **kwargs
    ):
        """
        Initialize Detection Agent.
        
        Args:
            models: Dictionary of available models
            default_model: Default model type
            selection_strategy: 'fixed', 'adaptive', 'contextual'
        """
        super().__init__(agent_type=AgentType.DETECTION, **kwargs)
        
        self.selection_strategy = selection_strategy
        self.default_model = default_model
        
        # Initialize models
        self.models: Dict[ModelType, BaseAnomalyDetector] = models or {}
        if not self.models:
            self._initialize_default_models()
        
        # Model performance tracking per metric
        self._model_performance: Dict[str, Dict[ModelType, Dict[str, float]]] = {}
        
        # Metric characteristics cache
        self._metric_characteristics: Dict[str, Dict[str, Any]] = {}
        
        # Detection history
        self._detection_counts: Dict[str, int] = {}
    
    def _initialize_default_models(self) -> None:
        """Initialize default set of models."""
        self.models = {
            ModelType.ZSCORE: ZScoreDetector(threshold=0.5),
            ModelType.STL_ESD: STLESDDetector(threshold=0.5),
            ModelType.ISOLATION_FOREST: IsolationForestDetector(threshold=0.5),
            ModelType.ONE_CLASS_SVM: OneClassSVMDetector(threshold=0.5),
        }
        
        # Create ensemble from available models
        ensemble = EnsembleDetector(
            models=list(self.models.values()),
            strategy="weighted",
            threshold=0.5
        )
        self.models[ModelType.ENSEMBLE] = ensemble
        
        logger.info(
            "Initialized default models",
            model_count=len(self.models)
        )
    
    async def process(self, data: Dict[str, Any]) -> Optional[AgentDecision]:
        """
        Process incoming data and detect anomalies.
        
        Args:
            data: Dictionary containing:
                - metric_name: Name of the metric
                - features: Feature vector
                - raw_values: Raw time series values
                - labels: Metric labels
                
        Returns:
            Agent decision about detection
        """
        metric_name = data.get("metric_name", "unknown")
        features = data.get("features")
        raw_values = data.get("raw_values")
        labels = data.get("labels", {})
        
        if features is None and raw_values is None:
            return None
        
        # Select appropriate model
        selected_model, selection_reasoning = self._select_model(
            metric_name, features, raw_values
        )
        
        # Run detection
        try:
            if features is not None:
                X = np.atleast_2d(features)
            else:
                X = np.atleast_2d(raw_values)
            
            results = selected_model.detect(X, return_details=True)
            result = results[0]
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return None
        
        # Analyze result
        if result.is_anomaly:
            anomaly_type = self._classify_anomaly_type(features, raw_values)
            
            # Create anomaly record
            anomaly = Anomaly(
                metric_name=metric_name,
                labels=labels,
                anomaly_type=anomaly_type,
                severity=result.severity,
                ensemble_score=result.anomaly_score,
                confidence=result.confidence,
                value=raw_values[-1] if raw_values else 0,
                expected_value=np.mean(raw_values[:-10]) if raw_values and len(raw_values) > 10 else 0,
                deviation=result.anomaly_score,
                contributing_features=result.contributing_features,
            )
            
            # Notify other agents
            self.send_message(
                recipient=AgentType.CORRELATION.value,
                message_type="anomaly_detected",
                payload={
                    "anomaly": anomaly.model_dump() if hasattr(anomaly, 'model_dump') else vars(anomaly),
                    "model_used": selected_model.model_type.value,
                },
                priority=MessagePriority.HIGH
            )
            
            # Update detection count
            self._detection_counts[metric_name] = self._detection_counts.get(metric_name, 0) + 1
        
        # Record decision
        decision = self.record_decision(
            decision=f"{'Anomaly detected' if result.is_anomaly else 'Normal'} using {selected_model.model_type.value}",
            reasoning=selection_reasoning,
            confidence=result.confidence,
            input_data={
                "metric_name": metric_name,
                "anomaly_score": result.anomaly_score,
                "model_used": selected_model.model_type.value,
            },
            alternatives=[
                {"model": m.value, "available": True}
                for m in self.models.keys()
            ]
        )
        
        # Store result in memory
        self.memory.remember(
            f"detection_{metric_name}",
            {
                "is_anomaly": result.is_anomaly,
                "score": result.anomaly_score,
                "model": selected_model.model_type.value,
            }
        )
        
        return decision
    
    def _select_model(
        self,
        metric_name: str,
        features: Optional[np.ndarray],
        raw_values: Optional[np.ndarray]
    ) -> Tuple[BaseAnomalyDetector, str]:
        """
        Select the best model for this metric.
        
        Returns:
            Tuple of (selected model, reasoning)
        """
        if self.selection_strategy == "fixed":
            model = self.models.get(self.default_model, list(self.models.values())[0])
            return model, f"Using fixed model: {self.default_model.value}"
        
        # Analyze metric characteristics
        characteristics = self._analyze_characteristics(metric_name, raw_values)
        
        # Get historical performance
        perf = self._model_performance.get(metric_name, {})
        
        if self.selection_strategy == "adaptive" and perf:
            # Select based on past performance
            best_model_type = max(
                perf.keys(),
                key=lambda m: perf[m].get("accuracy", 0.5)
            )
            model = self.models.get(best_model_type, self.models[self.default_model])
            return model, f"Adaptive selection based on past accuracy: {best_model_type.value}"
        
        # Contextual selection based on metric characteristics
        if characteristics.get("has_seasonality", False):
            model = self.models.get(ModelType.STL_ESD, self.models[self.default_model])
            return model, "Selected STL+ESD due to detected seasonality"
        
        if characteristics.get("high_dimensionality", False):
            model = self.models.get(ModelType.ISOLATION_FOREST, self.models[self.default_model])
            return model, "Selected Isolation Forest for high-dimensional data"
        
        if characteristics.get("has_trend", False):
            model = self.models.get(ModelType.STL_ESD, self.models[self.default_model])
            return model, "Selected STL+ESD due to detected trend"
        
        # Default to ensemble
        model = self.models.get(ModelType.ENSEMBLE, list(self.models.values())[0])
        return model, "Using ensemble for robust detection"
    
    def _analyze_characteristics(
        self,
        metric_name: str,
        raw_values: Optional[np.ndarray]
    ) -> Dict[str, Any]:
        """Analyze time series characteristics."""
        if metric_name in self._metric_characteristics:
            return self._metric_characteristics[metric_name]
        
        characteristics = {
            "has_seasonality": False,
            "has_trend": False,
            "is_stationary": True,
            "high_dimensionality": False,
            "sample_size": 0,
        }
        
        if raw_values is not None and len(raw_values) > 10:
            values = np.asarray(raw_values).flatten()
            characteristics["sample_size"] = len(values)
            
            # Simple seasonality check using autocorrelation
            if len(values) > 48:
                acf = np.correlate(values - np.mean(values), values - np.mean(values), mode='full')
                acf = acf[len(acf)//2:]
                acf = acf / acf[0]
                
                # Check for peaks in ACF (suggests seasonality)
                if len(acf) > 24 and np.max(acf[20:30]) > 0.5:
                    characteristics["has_seasonality"] = True
            
            # Trend check
            x = np.arange(len(values))
            slope, _ = np.polyfit(x, values, 1)
            if abs(slope) > 0.01 * np.std(values):
                characteristics["has_trend"] = True
        
        self._metric_characteristics[metric_name] = characteristics
        return characteristics
    
    def _classify_anomaly_type(
        self,
        features: Optional[np.ndarray],
        raw_values: Optional[np.ndarray]
    ) -> AnomalyType:
        """Classify the type of anomaly detected."""
        if raw_values is None or len(raw_values) < 2:
            return AnomalyType.OUTLIER
        
        values = np.asarray(raw_values).flatten()
        last_value = values[-1]
        recent_mean = np.mean(values[-10:]) if len(values) >= 10 else np.mean(values)
        overall_mean = np.mean(values)
        overall_std = np.std(values)
        
        # Spike detection
        if last_value > overall_mean + 3 * overall_std:
            return AnomalyType.SPIKE
        
        # Drop detection
        if last_value < overall_mean - 3 * overall_std:
            return AnomalyType.DROP
        
        # Trend change detection
        if len(values) > 20:
            recent_slope, _ = np.polyfit(range(10), values[-10:], 1)
            prev_slope, _ = np.polyfit(range(10), values[-20:-10], 1)
            if np.sign(recent_slope) != np.sign(prev_slope):
                return AnomalyType.TREND_CHANGE
        
        # Level shift detection
        if len(values) > 10:
            if abs(recent_mean - overall_mean) > 2 * overall_std:
                return AnomalyType.LEVEL_SHIFT
        
        # Variance change
        if len(values) > 20:
            recent_var = np.var(values[-10:])
            prev_var = np.var(values[-20:-10])
            if recent_var > 2 * prev_var or recent_var < 0.5 * prev_var:
                return AnomalyType.VARIANCE_CHANGE
        
        return AnomalyType.OUTLIER
    
    def train_models(
        self,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> None:
        """
        Train all models on training data.
        
        Args:
            X: Training features
            feature_names: Feature names
        """
        for model_type, model in self.models.items():
            try:
                if model_type != ModelType.ENSEMBLE:
                    model.fit(X, feature_names=feature_names)
                    logger.info(f"Trained model: {model_type.value}")
            except Exception as e:
                logger.error(f"Failed to train {model_type.value}: {e}")
        
        # Train ensemble last
        if ModelType.ENSEMBLE in self.models:
            try:
                self.models[ModelType.ENSEMBLE].fit(X, feature_names=feature_names)
                logger.info("Trained ensemble model")
            except Exception as e:
                logger.error(f"Failed to train ensemble: {e}")
    
    def update_model_performance(
        self,
        metric_name: str,
        model_type: ModelType,
        accuracy: float,
        precision: float,
        recall: float
    ) -> None:
        """Update performance metrics for a model on a specific metric."""
        if metric_name not in self._model_performance:
            self._model_performance[metric_name] = {}
        
        self._model_performance[metric_name][model_type] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "updated_at": now_utc().isoformat(),
        }
    
    def get_detection_stats(self) -> Dict[str, Any]:
        """Get detection statistics."""
        return {
            "total_detections": sum(self._detection_counts.values()),
            "detections_by_metric": dict(self._detection_counts),
            "model_performance": {
                k: {m.value: v for m, v in perf.items()}
                for k, perf in self._model_performance.items()
            },
            "decisions_made": self._decisions_made,
            "accuracy": self.get_accuracy(),
        }
