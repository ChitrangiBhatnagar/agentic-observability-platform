"""
One-Class SVM Anomaly Detection.
Support vector machine based novelty detection.
"""

from typing import Any, Dict, List, Optional
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

from src.types import ModelType, ContributingFeature
from src.utils import get_logger
from .base import BaseAnomalyDetector

logger = get_logger(__name__)


class OneClassSVMDetector(BaseAnomalyDetector):
    """
    One-Class SVM based anomaly detection.
    
    Learns a boundary around normal data in feature space.
    Good for small to medium datasets with clear separation.
    """
    
    def __init__(
        self,
        kernel: str = "rbf",
        nu: float = 0.01,
        gamma: str = "scale",
        threshold: float = 0.5,
        **kwargs
    ):
        """
        Initialize One-Class SVM detector.
        
        Args:
            kernel: Kernel type ('rbf', 'linear', 'poly', 'sigmoid')
            nu: Upper bound on training errors fraction
            gamma: Kernel coefficient
            threshold: Anomaly threshold
        """
        super().__init__(
            model_type=ModelType.ONE_CLASS_SVM,
            threshold=threshold,
            kernel=kernel,
            nu=nu,
            gamma=gamma,
            **kwargs
        )
        
        self.kernel = kernel
        self.nu = nu
        self.gamma = gamma
        
        self._model: Optional[OneClassSVM] = None
        self._scaler: Optional[StandardScaler] = None
        self._support_vector_indices: Optional[np.ndarray] = None
    
    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None
    ) -> "OneClassSVMDetector":
        """
        Fit the One-Class SVM model.
        
        Args:
            X: Training data (n_samples, n_features)
            y: Ignored
            feature_names: Feature names
            
        Returns:
            Self
        """
        X = np.atleast_2d(X)
        
        # Scale features (important for SVM)
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)
        
        self._model = OneClassSVM(
            kernel=self.kernel,
            nu=self.nu,
            gamma=self.gamma,
        )
        
        self._model.fit(X_scaled)
        
        # Store support vector info
        self._support_vector_indices = self._model.support_
        
        self._feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]
        self._is_fitted = True
        
        self.metadata.training_samples = len(X)
        self.metadata.hyperparameters.update({
            "kernel": self.kernel,
            "nu": self.nu,
            "n_support_vectors": len(self._support_vector_indices),
        })
        
        logger.info(
            "Fitted One-Class SVM",
            samples=len(X),
            features=X.shape[1],
            kernel=self.kernel,
            support_vectors=len(self._support_vector_indices)
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
        if not self._is_fitted or self._model is None or self._scaler is None:
            raise RuntimeError("Detector must be fitted first")
        
        X = np.atleast_2d(X)
        X_scaled = self._scaler.transform(X)
        
        # Get decision function (negative = anomaly)
        decision = self._model.decision_function(X_scaled)
        
        # Normalize to [0, 1]
        # Use sigmoid transformation
        scores = 1 / (1 + np.exp(decision * 0.5))
        
        return scores
    
    def _get_contributing_features(self, x: np.ndarray) -> List[ContributingFeature]:
        """Get features contributing to anomaly."""
        if self._scaler is None:
            return []
        
        x = np.atleast_1d(x)
        x_scaled = self._scaler.transform(x.reshape(1, -1)).flatten()
        
        contributions = []
        for i, (name, scaled_val, original_val) in enumerate(
            zip(self._feature_names, x_scaled, x)
        ):
            # For SVM, deviation from mean is a proxy for contribution
            importance = abs(scaled_val) / (abs(scaled_val).sum() + 1e-10)
            
            if importance > 0.05:
                contributions.append(ContributingFeature(
                    name=name,
                    value=float(original_val),
                    importance=float(importance),
                    expected_range=(
                        float(self._scaler.mean_[i] - 2 * self._scaler.scale_[i]),
                        float(self._scaler.mean_[i] + 2 * self._scaler.scale_[i])
                    )
                ))
        
        contributions.sort(key=lambda c: c.importance, reverse=True)
        return contributions[:5]
    
    def _get_model_state(self) -> Dict[str, Any]:
        """Get model state for serialization."""
        return {
            "kernel": self.kernel,
            "nu": self.nu,
            "gamma": self.gamma,
            "model": self._model,
            "scaler": self._scaler,
            "support_vector_indices": self._support_vector_indices,
        }
    
    def _set_model_state(self, state: Dict[str, Any]) -> None:
        """Set model state from deserialization."""
        self.kernel = state["kernel"]
        self.nu = state["nu"]
        self.gamma = state["gamma"]
        self._model = state["model"]
        self._scaler = state["scaler"]
        self._support_vector_indices = state["support_vector_indices"]
