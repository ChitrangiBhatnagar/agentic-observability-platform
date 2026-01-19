"""
SHAP-based Explainability for Anomaly Detection Models.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from src.types import ContributingFeature, AnomalyExplanation
from src.utils import get_logger, generate_id, now_utc

logger = get_logger(__name__)


class SHAPExplainer:
    """
    SHAP (SHapley Additive exPlanations) based explainer for anomaly detection.
    
    Provides feature importance explanations using SHAP values,
    helping operators understand why an anomaly was detected.
    """
    
    def __init__(
        self,
        background_samples: Optional[np.ndarray] = None,
        max_background_size: int = 100,
        approximate: bool = True
    ):
        """
        Initialize SHAP Explainer.
        
        Args:
            background_samples: Background dataset for SHAP calculations
            max_background_size: Maximum background samples to use
            approximate: Use approximate SHAP (faster)
        """
        self.background_samples = background_samples
        self.max_background_size = max_background_size
        self.approximate = approximate
        
        self._shap_available = self._check_shap_available()
        self._explainer = None
    
    def _check_shap_available(self) -> bool:
        """Check if SHAP library is available."""
        try:
            import shap
            return True
        except ImportError:
            logger.warning("SHAP library not available, using fallback explanations")
            return False
    
    def set_background(self, samples: np.ndarray) -> None:
        """Set background samples for SHAP calculations."""
        if len(samples) > self.max_background_size:
            # Sample randomly
            indices = np.random.choice(
                len(samples),
                self.max_background_size,
                replace=False
            )
            samples = samples[indices]
        
        self.background_samples = samples
        self._explainer = None  # Reset explainer
    
    def explain(
        self,
        model: Any,
        features: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> List[ContributingFeature]:
        """
        Explain model predictions using SHAP.
        
        Args:
            model: Trained model with predict method
            features: Feature vector(s) to explain
            feature_names: Names for each feature
            
        Returns:
            List of contributing features with importance
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        n_features = features.shape[1]
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(n_features)]
        
        if self._shap_available:
            return self._explain_with_shap(model, features, feature_names)
        else:
            return self._explain_fallback(model, features, feature_names)
    
    def _explain_with_shap(
        self,
        model: Any,
        features: np.ndarray,
        feature_names: List[str]
    ) -> List[ContributingFeature]:
        """Explain using SHAP library."""
        import shap
        
        try:
            # Create explainer if needed
            if self._explainer is None:
                if self.background_samples is not None:
                    if self.approximate:
                        self._explainer = shap.KernelExplainer(
                            model.predict if hasattr(model, 'predict') else model,
                            self.background_samples
                        )
                    else:
                        # Try TreeExplainer for tree-based models
                        try:
                            self._explainer = shap.TreeExplainer(model)
                        except Exception:
                            self._explainer = shap.KernelExplainer(
                                model.predict if hasattr(model, 'predict') else model,
                                self.background_samples
                            )
                else:
                    # Use background summary
                    background = shap.sample(features, min(10, len(features)))
                    self._explainer = shap.KernelExplainer(
                        model.predict if hasattr(model, 'predict') else model,
                        background
                    )
            
            # Calculate SHAP values
            shap_values = self._explainer.shap_values(features)
            
            # Handle multi-output
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            
            # Get values for first sample
            values = shap_values[0] if shap_values.ndim > 1 else shap_values
            
            return self._create_contributing_features(
                values, feature_names, features[0]
            )
            
        except Exception as e:
            logger.error(f"SHAP explanation failed: {e}")
            return self._explain_fallback(model, features, feature_names)
    
    def _explain_fallback(
        self,
        model: Any,
        features: np.ndarray,
        feature_names: List[str]
    ) -> List[ContributingFeature]:
        """
        Fallback explanation using perturbation-based importance.
        
        Simple approach: perturb each feature and measure prediction change.
        """
        base_pred = self._get_prediction(model, features[0])
        
        importances = []
        
        for i in range(len(feature_names)):
            # Create perturbed version
            perturbed = features[0].copy()
            
            # Perturb by setting to zero or flipping sign
            original_value = perturbed[i]
            perturbed[i] = 0 if original_value != 0 else 1
            
            # Get new prediction
            new_pred = self._get_prediction(model, perturbed)
            
            # Importance is change in prediction
            importance = abs(base_pred - new_pred)
            importances.append(importance)
        
        # Normalize
        total = sum(importances) or 1
        importances = [i / total for i in importances]
        
        return self._create_contributing_features(
            np.array(importances),
            feature_names,
            features[0]
        )
    
    def _get_prediction(self, model: Any, features: np.ndarray) -> float:
        """Get model prediction for features."""
        features = features.reshape(1, -1)
        
        if hasattr(model, 'predict'):
            pred = model.predict(features)
        elif hasattr(model, 'decision_function'):
            pred = model.decision_function(features)
        elif callable(model):
            pred = model(features)
        else:
            raise ValueError("Model must have predict, decision_function, or be callable")
        
        return float(pred[0]) if hasattr(pred, '__len__') else float(pred)
    
    def _create_contributing_features(
        self,
        importances: np.ndarray,
        feature_names: List[str],
        feature_values: np.ndarray
    ) -> List[ContributingFeature]:
        """Create ContributingFeature objects from importances."""
        features = []
        
        # Sort by absolute importance
        indices = np.argsort(np.abs(importances))[::-1]
        
        for idx in indices:
            importance = float(importances[idx])
            
            # Skip very low importance features
            if abs(importance) < 0.01:
                continue
            
            features.append(ContributingFeature(
                name=feature_names[idx],
                value=float(feature_values[idx]),
                contribution=importance,
                description=self._describe_contribution(
                    feature_names[idx],
                    feature_values[idx],
                    importance
                )
            ))
        
        return features[:10]  # Top 10 features
    
    def _describe_contribution(
        self,
        name: str,
        value: float,
        contribution: float
    ) -> str:
        """Generate human-readable description of contribution."""
        direction = "increased" if contribution > 0 else "decreased"
        magnitude = "significantly" if abs(contribution) > 0.2 else "slightly"
        
        return f"{name}={value:.2f} {magnitude} {direction} anomaly score"
    
    def explain_ensemble(
        self,
        models: Dict[str, Any],
        features: np.ndarray,
        feature_names: Optional[List[str]] = None,
        model_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, List[ContributingFeature]]:
        """
        Explain predictions from an ensemble of models.
        
        Args:
            models: Dictionary of model_name -> model
            features: Feature vector to explain
            feature_names: Names for features
            model_weights: Optional weights for each model
            
        Returns:
            Dictionary of model_name -> contributing features
        """
        explanations = {}
        
        for model_name, model in models.items():
            try:
                explanation = self.explain(model, features, feature_names)
                explanations[model_name] = explanation
            except Exception as e:
                logger.error(f"Failed to explain {model_name}: {e}")
                explanations[model_name] = []
        
        return explanations
    
    def aggregate_explanations(
        self,
        explanations: Dict[str, List[ContributingFeature]],
        model_weights: Optional[Dict[str, float]] = None
    ) -> List[ContributingFeature]:
        """
        Aggregate explanations from multiple models.
        
        Args:
            explanations: Model explanations
            model_weights: Weights for each model
            
        Returns:
            Aggregated contributing features
        """
        if model_weights is None:
            model_weights = {m: 1.0 for m in explanations.keys()}
        
        # Normalize weights
        total_weight = sum(model_weights.values())
        model_weights = {m: w / total_weight for m, w in model_weights.items()}
        
        # Aggregate by feature name
        aggregated = {}
        
        for model_name, features in explanations.items():
            weight = model_weights.get(model_name, 0)
            
            for feature in features:
                if feature.name not in aggregated:
                    aggregated[feature.name] = {
                        'value': feature.value,
                        'contribution': 0,
                        'descriptions': []
                    }
                
                aggregated[feature.name]['contribution'] += feature.contribution * weight
                if feature.description:
                    aggregated[feature.name]['descriptions'].append(feature.description)
        
        # Convert to ContributingFeature
        result = []
        for name, data in aggregated.items():
            result.append(ContributingFeature(
                name=name,
                value=data['value'],
                contribution=data['contribution'],
                description=data['descriptions'][0] if data['descriptions'] else ""
            ))
        
        # Sort by absolute contribution
        result.sort(key=lambda f: abs(f.contribution), reverse=True)
        
        return result[:10]
    
    def get_feature_importance_summary(
        self,
        features_list: List[List[ContributingFeature]]
    ) -> Dict[str, float]:
        """
        Get aggregate feature importance across multiple explanations.
        
        Args:
            features_list: List of explanation feature lists
            
        Returns:
            Dictionary of feature_name -> average importance
        """
        importance_sums = {}
        importance_counts = {}
        
        for features in features_list:
            for feature in features:
                if feature.name not in importance_sums:
                    importance_sums[feature.name] = 0
                    importance_counts[feature.name] = 0
                
                importance_sums[feature.name] += abs(feature.contribution)
                importance_counts[feature.name] += 1
        
        return {
            name: importance_sums[name] / importance_counts[name]
            for name in importance_sums
        }
