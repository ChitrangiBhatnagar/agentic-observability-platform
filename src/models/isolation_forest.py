"""
Isolation Forest Anomaly Detection.
Tree-based anomaly detection that isolates anomalies.
"""

from typing import Any, Dict, List, Optional
import numpy as np
from sklearn.ensemble import IsolationForest

from src.types import ModelType, ContributingFeature
from src.utils import get_logger
from .base import BaseAnomalyDetector, AdaptiveThresholdMixin

logger = get_logger(__name__)


class IsolationForestDetector(AdaptiveThresholdMixin, BaseAnomalyDetector):
    """
    Isolation Forest based anomaly detection.
    
    Excellent for high-dimensional data. Works by isolating observations
    by randomly selecting features and split values.
    
    Key insight: Anomalies are easier to isolate (require fewer splits).
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_samples: str = "auto",
        contamination: float = 0.01,
        max_features: float = 1.0,
        bootstrap: bool = False,
        random_state: int = 42,
        threshold: float = 0.5,
        **kwargs
    ):
        """
        Initialize Isolation Forest detector.
        
        Args:
            n_estimators: Number of trees
            max_samples: Number of samples per tree
            contamination: Expected proportion of anomalies
            max_features: Number of features per tree
            bootstrap: Whether to use bootstrap sampling
            random_state: Random seed
            threshold: Anomaly threshold
        """
        super().__init__(
            model_type=ModelType.ISOLATION_FOREST,
            threshold=threshold,
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=contamination,
            max_features=max_features,
            bootstrap=bootstrap,
            random_state=random_state,
            **kwargs
        )
        
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        
        self._model: Optional[IsolationForest] = None
        self._feature_importances: Optional[np.ndarray] = None
    
    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None
    ) -> "IsolationForestDetector":
        """
        Fit the Isolation Forest model.
        
        Args:
            X: Training data (n_samples, n_features)
            y: Ignored (unsupervised method)
            feature_names: Feature names
            
        Returns:
            Self
        """
        X = np.atleast_2d(X)
        
        self._model = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            contamination=self.contamination,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            random_state=self.random_state,
            n_jobs=-1,
        )
        
        self._model.fit(X)
        
        # Estimate feature importances using mean decrease in path length
        self._feature_importances = self._compute_feature_importances(X)
        
        self._feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]
        self._is_fitted = True
        
        self.metadata.training_samples = len(X)
        self.metadata.hyperparameters.update({
            "n_estimators": self.n_estimators,
            "contamination": self.contamination,
        })
        
        logger.info(
            "Fitted Isolation Forest",
            samples=len(X),
            features=X.shape[1],
            n_estimators=self.n_estimators
        )
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomaly scores.
        
        Args:
            X: Input data (n_samples, n_features)
            
        Returns:
            Anomaly scores (n_samples,) in range [0, 1]
        """
        if not self._is_fitted or self._model is None:
            raise RuntimeError("Detector must be fitted first")
        
        X = np.atleast_2d(X)
        
        # Get decision function (negative = anomaly)
        decision = self._model.decision_function(X)
        
        # Convert to anomaly score [0, 1]
        # More negative = more anomalous
        # Normalize so that 0 = normal, 1 = highly anomalous
        scores = 1 / (1 + np.exp(decision * 2))
        
        return scores
    
    def _compute_feature_importances(self, X: np.ndarray) -> np.ndarray:
        """
        Compute feature importances by measuring path length contribution.
        
        Uses permutation-based approach.
        """
        if self._model is None:
            return np.ones(X.shape[1]) / X.shape[1]
        
        baseline_scores = self._model.decision_function(X)
        importances = np.zeros(X.shape[1])
        
        for i in range(X.shape[1]):
            X_permuted = X.copy()
            np.random.shuffle(X_permuted[:, i])
            permuted_scores = self._model.decision_function(X_permuted)
            
            # Importance = change in scores when feature is randomized
            importances[i] = np.mean(np.abs(baseline_scores - permuted_scores))
        
        # Normalize
        if np.sum(importances) > 0:
            importances = importances / np.sum(importances)
        else:
            importances = np.ones(X.shape[1]) / X.shape[1]
        
        return importances
    
    def _get_contributing_features(self, x: np.ndarray) -> List[ContributingFeature]:
        """Get features contributing to anomaly."""
        x = np.atleast_1d(x)
        
        if self._feature_importances is None:
            return []
        
        contributions = []
        for i, (name, importance, val) in enumerate(
            zip(self._feature_names, self._feature_importances, x)
        ):
            if importance > 0.05:  # Only significant features
                contributions.append(ContributingFeature(
                    name=name,
                    value=float(val),
                    importance=float(importance),
                    expected_range=(0.0, 0.0)  # Not applicable for IF
                ))
        
        contributions.sort(key=lambda c: c.importance, reverse=True)
        return contributions[:5]
    
    def _get_model_state(self) -> Dict[str, Any]:
        """Get model state for serialization."""
        return {
            "n_estimators": self.n_estimators,
            "max_samples": self.max_samples,
            "contamination": self.contamination,
            "max_features": self.max_features,
            "bootstrap": self.bootstrap,
            "random_state": self.random_state,
            "model": self._model,
            "feature_importances": self._feature_importances,
        }
    
    def _set_model_state(self, state: Dict[str, Any]) -> None:
        """Set model state from deserialization."""
        self.n_estimators = state["n_estimators"]
        self.max_samples = state["max_samples"]
        self.contamination = state["contamination"]
        self.max_features = state["max_features"]
        self.bootstrap = state["bootstrap"]
        self.random_state = state["random_state"]
        self._model = state["model"]
        self._feature_importances = state["feature_importances"]
