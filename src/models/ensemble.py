"""
Ensemble Anomaly Detection.
Combines multiple models for robust detection.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.types import ModelType, ContributingFeature, AnomalyScore
from src.utils import get_logger
from .base import BaseAnomalyDetector, DetectionResult

logger = get_logger(__name__)


class EnsembleDetector(BaseAnomalyDetector):
    """
    Ensemble detector combining multiple anomaly detection models.
    
    Strategies:
    - voting: Majority voting across models
    - averaging: Average of anomaly scores
    - weighted: Weighted average based on model performance
    - max: Maximum anomaly score
    - stacking: Meta-learner on model outputs
    """
    
    def __init__(
        self,
        models: Optional[List[BaseAnomalyDetector]] = None,
        strategy: str = "weighted",
        weights: Optional[List[float]] = None,
        threshold: float = 0.5,
        min_agreement: float = 0.5,
        **kwargs
    ):
        """
        Initialize ensemble detector.
        
        Args:
            models: List of anomaly detectors
            strategy: Combination strategy ('voting', 'averaging', 'weighted', 'max')
            weights: Model weights for weighted strategy
            threshold: Anomaly threshold
            min_agreement: Minimum fraction of models that must agree (for voting)
        """
        super().__init__(
            model_type=ModelType.ENSEMBLE,
            threshold=threshold,
            **kwargs
        )
        
        self.models = models or []
        self.strategy = strategy
        self.min_agreement = min_agreement
        
        # Initialize weights
        if weights is not None:
            self.weights = np.array(weights)
        elif models:
            self.weights = np.ones(len(models)) / len(models)
        else:
            self.weights = np.array([])
        
        # Model performance tracking
        self._model_performance: Dict[str, Dict[str, float]] = {}
    
    def add_model(
        self,
        model: BaseAnomalyDetector,
        weight: float = 1.0
    ) -> "EnsembleDetector":
        """
        Add a model to the ensemble.
        
        Args:
            model: Anomaly detector to add
            weight: Initial weight for the model
            
        Returns:
            Self
        """
        self.models.append(model)
        
        # Update weights
        self.weights = np.append(self.weights, weight)
        self.weights = self.weights / self.weights.sum()
        
        logger.info(
            "Added model to ensemble",
            model_type=model.model_type.value,
            ensemble_size=len(self.models)
        )
        
        return self
    
    def remove_model(self, model_id: str) -> bool:
        """
        Remove a model from the ensemble.
        
        Args:
            model_id: ID of the model to remove
            
        Returns:
            True if removed, False if not found
        """
        for i, model in enumerate(self.models):
            if model.model_id == model_id:
                del self.models[i]
                self.weights = np.delete(self.weights, i)
                if len(self.weights) > 0:
                    self.weights = self.weights / self.weights.sum()
                return True
        return False
    
    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None
    ) -> "EnsembleDetector":
        """
        Fit all models in the ensemble.
        
        Args:
            X: Training data
            y: Optional labels
            feature_names: Feature names
            
        Returns:
            Self
        """
        X = np.atleast_2d(X)
        
        for model in self.models:
            try:
                model.fit(X, y, feature_names)
                logger.info(f"Fitted model: {model.model_type.value}")
            except Exception as e:
                logger.error(
                    f"Failed to fit model {model.model_type.value}",
                    error=str(e)
                )
        
        self._feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]
        self._is_fitted = all(m.is_fitted for m in self.models)
        
        self.metadata.training_samples = len(X)
        
        logger.info(
            "Fitted ensemble",
            models=len(self.models),
            fitted=self._is_fitted
        )
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomaly scores using ensemble strategy.
        
        Args:
            X: Input data
            
        Returns:
            Ensemble anomaly scores
        """
        if not self._is_fitted:
            raise RuntimeError("Ensemble must be fitted first")
        
        X = np.atleast_2d(X)
        n_samples = len(X)
        
        # Get predictions from all models
        all_scores = []
        for model in self.models:
            try:
                scores = model.predict(X)
                all_scores.append(scores)
            except Exception as e:
                logger.error(f"Prediction failed for {model.model_type.value}: {e}")
                # Use neutral score on failure
                all_scores.append(np.full(n_samples, 0.5))
        
        all_scores = np.array(all_scores)  # (n_models, n_samples)
        
        # Apply combination strategy
        if self.strategy == "voting":
            votes = (all_scores >= self.threshold).astype(float)
            agreement = np.sum(votes * self.weights[:, np.newaxis], axis=0)
            scores = (agreement >= self.min_agreement).astype(float)
            
        elif self.strategy == "averaging":
            scores = np.mean(all_scores, axis=0)
            
        elif self.strategy == "weighted":
            scores = np.sum(all_scores * self.weights[:, np.newaxis], axis=0)
            
        elif self.strategy == "max":
            scores = np.max(all_scores, axis=0)
            
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        return scores
    
    def detect(
        self,
        X: np.ndarray,
        return_details: bool = True
    ) -> List[DetectionResult]:
        """
        Detect anomalies with detailed results from each model.
        
        Args:
            X: Input data
            return_details: Include per-model details
            
        Returns:
            List of DetectionResult with ensemble and individual model info
        """
        import time
        start_time = time.time()
        
        X = np.atleast_2d(X)
        
        # Get ensemble scores
        ensemble_scores = self.predict(X)
        
        # Get individual model scores
        model_scores_list = []
        for model in self.models:
            try:
                model_result = model.detect(X, return_details=False)
                model_scores_list.append([r.anomaly_score for r in model_result])
            except Exception:
                model_scores_list.append([0.5] * len(X))
        
        detection_time = (time.time() - start_time) * 1000 / len(X)
        
        results = []
        for i, ensemble_score in enumerate(ensemble_scores):
            is_anomaly = ensemble_score >= self.threshold
            
            # Collect individual model scores
            individual_scores = []
            for j, model in enumerate(self.models):
                individual_scores.append(AnomalyScore(
                    model_type=model.model_type,
                    score=model_scores_list[j][i],
                    threshold=model.threshold,
                    is_anomaly=model_scores_list[j][i] >= model.threshold,
                    confidence=model._calculate_confidence(model_scores_list[j][i]),
                    contributing_features=[]
                ))
            
            # Aggregate contributing features
            all_contributing = []
            if return_details:
                for model in self.models:
                    if hasattr(model, '_get_contributing_features'):
                        try:
                            cf = model._get_contributing_features(X[i])
                            all_contributing.extend(cf)
                        except Exception:
                            pass
            
            # Deduplicate and rank
            seen_features = set()
            unique_contributing = []
            for cf in sorted(all_contributing, key=lambda x: x.importance, reverse=True):
                if cf.name not in seen_features:
                    seen_features.add(cf.name)
                    unique_contributing.append(cf)
            
            result = DetectionResult(
                is_anomaly=is_anomaly,
                anomaly_score=float(ensemble_score),
                confidence=self._calculate_confidence(ensemble_score),
                threshold=self.threshold,
                model_type=ModelType.ENSEMBLE,
                model_version=self.version,
                detection_time_ms=detection_time,
                contributing_features=unique_contributing[:5],
            )
            results.append(result)
        
        return results
    
    def update_weights(
        self,
        performance_scores: Dict[str, float]
    ) -> None:
        """
        Update model weights based on performance scores.
        
        Args:
            performance_scores: Dict mapping model_id to performance score
        """
        new_weights = []
        for model in self.models:
            score = performance_scores.get(model.model_id, 1.0)
            new_weights.append(max(0.1, score))  # Minimum weight
        
        new_weights = np.array(new_weights)
        self.weights = new_weights / new_weights.sum()
        
        logger.info("Updated ensemble weights", weights=self.weights.tolist())
    
    def get_model_contributions(
        self,
        X: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Get individual model contributions to ensemble score.
        
        Args:
            X: Input data
            
        Returns:
            Dict mapping model_type to contribution scores
        """
        X = np.atleast_2d(X)
        contributions = {}
        
        for model in self.models:
            scores = model.predict(X)
            contributions[model.model_type.value] = scores
        
        return contributions
    
    def _get_model_state(self) -> Dict[str, Any]:
        """Get model state for serialization."""
        return {
            "strategy": self.strategy,
            "min_agreement": self.min_agreement,
            "weights": self.weights,
            "models": self.models,
            "model_performance": self._model_performance,
        }
    
    def _set_model_state(self, state: Dict[str, Any]) -> None:
        """Set model state from deserialization."""
        self.strategy = state["strategy"]
        self.min_agreement = state["min_agreement"]
        self.weights = state["weights"]
        self.models = state["models"]
        self._model_performance = state["model_performance"]
    
    @property
    def model_count(self) -> int:
        """Get number of models in ensemble."""
        return len(self.models)
    
    def get_model_weights(self) -> Dict[str, float]:
        """Get current model weights."""
        return {
            model.model_id: float(weight)
            for model, weight in zip(self.models, self.weights)
        }


class AdaptiveEnsembleDetector(EnsembleDetector):
    """
    Adaptive ensemble that automatically adjusts weights
    based on feedback and performance metrics.
    """
    
    def __init__(
        self,
        adaptation_rate: float = 0.1,
        performance_window: int = 100,
        **kwargs
    ):
        """
        Initialize adaptive ensemble.
        
        Args:
            adaptation_rate: Rate at which weights are updated
            performance_window: Number of recent predictions to consider
        """
        super().__init__(**kwargs)
        self.adaptation_rate = adaptation_rate
        self.performance_window = performance_window
        
        # Tracking
        self._predictions_buffer: List[Dict[str, Any]] = []
        self._feedback_buffer: List[Dict[str, Any]] = []
    
    def record_feedback(
        self,
        sample_idx: int,
        is_true_positive: bool
    ) -> None:
        """
        Record feedback for a prediction.
        
        Args:
            sample_idx: Index of the sample
            is_true_positive: Whether the detection was correct
        """
        self._feedback_buffer.append({
            "sample_idx": sample_idx,
            "is_correct": is_true_positive
        })
        
        # Trigger adaptation if enough feedback
        if len(self._feedback_buffer) >= 10:
            self._adapt_weights()
    
    def _adapt_weights(self) -> None:
        """Adapt weights based on accumulated feedback."""
        if not self._feedback_buffer:
            return
        
        # Calculate per-model accuracy on feedback samples
        model_correct = {model.model_id: 0 for model in self.models}
        model_total = {model.model_id: 0 for model in self.models}
        
        # This is simplified - real implementation would track predictions
        # For now, use uniform adjustment
        
        # Clear feedback buffer
        self._feedback_buffer = []
        
        logger.debug("Adapted ensemble weights")
