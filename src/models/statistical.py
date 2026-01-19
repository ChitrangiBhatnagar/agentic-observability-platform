"""
Statistical Anomaly Detection Models.
Implements Z-Score and STL+ESD based detection.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy import stats

from src.types import ModelType, AnomalyType, ContributingFeature
from src.utils import get_logger
from .base import BaseAnomalyDetector, DetectionResult, AdaptiveThresholdMixin

logger = get_logger(__name__)


class ZScoreDetector(AdaptiveThresholdMixin, BaseAnomalyDetector):
    """
    Z-Score based anomaly detection.
    
    Simple but effective baseline for detecting outliers
    based on deviation from mean.
    """
    
    def __init__(
        self,
        z_threshold: float = 3.0,
        threshold: float = 0.5,
        use_mad: bool = False,
        **kwargs
    ):
        """
        Initialize Z-Score detector.
        
        Args:
            z_threshold: Z-score threshold for anomaly (e.g., 3.0 = 3 std deviations)
            threshold: Normalized anomaly threshold (0-1)
            use_mad: Use Median Absolute Deviation (more robust)
        """
        super().__init__(
            model_type=ModelType.ZSCORE,
            threshold=threshold,
            **kwargs
        )
        self.z_threshold = z_threshold
        self.use_mad = use_mad
        
        # Learned statistics
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None
        self._median: Optional[np.ndarray] = None
        self._mad: Optional[np.ndarray] = None
    
    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None
    ) -> "ZScoreDetector":
        """
        Fit the detector by computing statistics.
        
        Args:
            X: Training data (n_samples, n_features)
            y: Ignored
            feature_names: Feature names
            
        Returns:
            Self
        """
        X = np.atleast_2d(X)
        
        if self.use_mad:
            self._median = np.median(X, axis=0)
            self._mad = np.median(np.abs(X - self._median), axis=0)
            # Avoid division by zero
            self._mad = np.maximum(self._mad, 1e-10)
        else:
            self._mean = np.mean(X, axis=0)
            self._std = np.std(X, axis=0)
            # Avoid division by zero
            self._std = np.maximum(self._std, 1e-10)
        
        self._feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]
        self._is_fitted = True
        
        self.metadata.training_samples = len(X)
        
        logger.info(
            "Fitted Z-Score detector",
            samples=len(X),
            features=X.shape[1],
            use_mad=self.use_mad
        )
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores based on Z-scores.
        
        Args:
            X: Input data (n_samples, n_features)
            
        Returns:
            Anomaly scores (n_samples,) normalized to [0, 1]
        """
        if not self._is_fitted:
            raise RuntimeError("Detector must be fitted first")
        
        X = np.atleast_2d(X)
        
        if self.use_mad:
            # Modified Z-score using MAD
            z_scores = 0.6745 * (X - self._median) / self._mad
        else:
            z_scores = (X - self._mean) / self._std
        
        # Take maximum absolute Z-score across features
        max_z = np.max(np.abs(z_scores), axis=1)
        
        # Normalize to [0, 1] using sigmoid-like function
        # Maps z_threshold to ~0.5, higher z-scores approach 1
        scores = 1 / (1 + np.exp(-(max_z - self.z_threshold)))
        
        return scores
    
    def _get_contributing_features(self, x: np.ndarray) -> List[ContributingFeature]:
        """Get features contributing to anomaly."""
        x = np.atleast_1d(x)
        
        if self.use_mad:
            z_scores = 0.6745 * (x - self._median) / self._mad
        else:
            z_scores = (x - self._mean) / self._std
        
        contributions = []
        for i, (name, z, val) in enumerate(zip(self._feature_names, z_scores, x)):
            if abs(z) > self.z_threshold * 0.5:  # Include moderately contributing features
                if self.use_mad:
                    expected = self._median[i]
                    low = expected - 2 * self._mad[i] / 0.6745
                    high = expected + 2 * self._mad[i] / 0.6745
                else:
                    expected = self._mean[i]
                    low = expected - 2 * self._std[i]
                    high = expected + 2 * self._std[i]
                
                contributions.append(ContributingFeature(
                    name=name,
                    value=float(val),
                    importance=float(abs(z) / self.z_threshold),
                    expected_range=(float(low), float(high))
                ))
        
        # Sort by importance
        contributions.sort(key=lambda c: c.importance, reverse=True)
        return contributions[:5]  # Top 5
    
    def partial_fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None
    ) -> "ZScoreDetector":
        """
        Update statistics incrementally.
        
        Uses Welford's online algorithm for numerical stability.
        """
        X = np.atleast_2d(X)
        
        if not self._is_fitted:
            return self.fit(X)
        
        # Update using exponential moving average
        alpha = 0.1  # Learning rate
        
        if self.use_mad:
            new_median = np.median(X, axis=0)
            new_mad = np.median(np.abs(X - new_median), axis=0)
            
            self._median = (1 - alpha) * self._median + alpha * new_median
            self._mad = (1 - alpha) * self._mad + alpha * new_mad
            self._mad = np.maximum(self._mad, 1e-10)
        else:
            new_mean = np.mean(X, axis=0)
            new_std = np.std(X, axis=0)
            
            self._mean = (1 - alpha) * self._mean + alpha * new_mean
            self._std = (1 - alpha) * self._std + alpha * new_std
            self._std = np.maximum(self._std, 1e-10)
        
        self.metadata.training_samples += len(X)
        
        return self
    
    def _get_model_state(self) -> Dict[str, Any]:
        """Get model state for serialization."""
        return {
            "z_threshold": self.z_threshold,
            "use_mad": self.use_mad,
            "mean": self._mean,
            "std": self._std,
            "median": self._median,
            "mad": self._mad,
        }
    
    def _set_model_state(self, state: Dict[str, Any]) -> None:
        """Set model state from deserialization."""
        self.z_threshold = state["z_threshold"]
        self.use_mad = state["use_mad"]
        self._mean = state["mean"]
        self._std = state["std"]
        self._median = state["median"]
        self._mad = state["mad"]


class STLESDDetector(BaseAnomalyDetector):
    """
    STL + ESD (Seasonal-Trend decomposition + Extreme Studentized Deviate) detector.
    
    Decomposes time series into trend, seasonal, and residual components,
    then applies ESD test on residuals to detect anomalies.
    
    Excellent for time series with strong seasonality.
    """
    
    def __init__(
        self,
        period: int = 24,
        max_anomalies: float = 0.05,
        alpha: float = 0.05,
        threshold: float = 0.5,
        **kwargs
    ):
        """
        Initialize STL+ESD detector.
        
        Args:
            period: Seasonal period (e.g., 24 for hourly data with daily seasonality)
            max_anomalies: Maximum proportion of anomalies expected
            alpha: Significance level for ESD test
            threshold: Normalized threshold
        """
        super().__init__(
            model_type=ModelType.STL_ESD,
            threshold=threshold,
            **kwargs
        )
        self.period = period
        self.max_anomalies = max_anomalies
        self.alpha = alpha
        
        # Learned parameters
        self._residual_mean: float = 0.0
        self._residual_std: float = 1.0
        self._seasonal_pattern: Optional[np.ndarray] = None
        self._trend_coefficients: Optional[np.ndarray] = None
    
    def _stl_decompose(self, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform STL decomposition.
        
        Returns:
            Tuple of (trend, seasonal, residual)
        """
        try:
            from statsmodels.tsa.seasonal import STL
            
            stl = STL(values, period=self.period, robust=True)
            result = stl.fit()
            
            return result.trend, result.seasonal, result.resid
            
        except Exception as e:
            logger.warning(f"STL decomposition failed, using simple method: {e}")
            # Fallback: simple moving average decomposition
            trend = np.convolve(values, np.ones(self.period)/self.period, mode='same')
            detrended = values - trend
            
            # Estimate seasonal component
            seasonal = np.zeros_like(values)
            for i in range(self.period):
                indices = np.arange(i, len(values), self.period)
                seasonal[indices] = np.mean(detrended[indices])
            
            residual = values - trend - seasonal
            
            return trend, seasonal, residual
    
    def _esd_test(
        self,
        residuals: np.ndarray,
        max_outliers: int
    ) -> List[int]:
        """
        Perform Generalized ESD test.
        
        Returns:
            List of anomaly indices
        """
        n = len(residuals)
        anomaly_indices = []
        
        working_data = residuals.copy()
        working_indices = np.arange(n)
        
        for i in range(max_outliers):
            if len(working_data) < 3:
                break
            
            mean = np.mean(working_data)
            std = np.std(working_data)
            
            if std < 1e-10:
                break
            
            # Find maximum deviation
            deviations = np.abs(working_data - mean) / std
            max_idx = np.argmax(deviations)
            max_deviation = deviations[max_idx]
            
            # Critical value
            p = 1 - self.alpha / (2 * (n - i))
            t_crit = stats.t.ppf(p, n - i - 2)
            lambda_crit = ((n - i - 1) * t_crit) / np.sqrt((n - i - 2 + t_crit**2) * (n - i))
            
            if max_deviation > lambda_crit:
                anomaly_indices.append(working_indices[max_idx])
                working_data = np.delete(working_data, max_idx)
                working_indices = np.delete(working_indices, max_idx)
            else:
                break
        
        return anomaly_indices
    
    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None
    ) -> "STLESDDetector":
        """
        Fit the detector on training data.
        
        Expects X to be a time series (n_samples,) or (n_samples, 1).
        """
        X = np.atleast_1d(X).flatten()
        
        if len(X) < self.period * 2:
            logger.warning("Not enough data for STL decomposition, using Z-score fallback")
            self._residual_mean = np.mean(X)
            self._residual_std = np.std(X)
        else:
            trend, seasonal, residual = self._stl_decompose(X)
            
            self._seasonal_pattern = np.array([
                np.mean(seasonal[i::self.period])
                for i in range(self.period)
            ])
            
            # Fit linear trend
            x = np.arange(len(trend))
            valid = ~np.isnan(trend)
            if np.sum(valid) > 1:
                self._trend_coefficients = np.polyfit(x[valid], trend[valid], 1)
            else:
                self._trend_coefficients = np.array([0, np.mean(X)])
            
            # Store residual statistics
            valid_resid = residual[~np.isnan(residual)]
            self._residual_mean = np.mean(valid_resid)
            self._residual_std = np.std(valid_resid)
        
        self._residual_std = max(self._residual_std, 1e-10)
        self._is_fitted = True
        self._feature_names = feature_names or ["value"]
        
        self.metadata.training_samples = len(X)
        
        logger.info(
            "Fitted STL+ESD detector",
            samples=len(X),
            period=self.period
        )
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomaly scores for new data.
        
        Args:
            X: Time series data (n_samples,) or (n_samples, 1)
            
        Returns:
            Anomaly scores (n_samples,)
        """
        if not self._is_fitted:
            raise RuntimeError("Detector must be fitted first")
        
        X = np.atleast_1d(X).flatten()
        n = len(X)
        
        if n < self.period * 2:
            # Not enough data for decomposition, use simple Z-score
            z_scores = np.abs(X - self._residual_mean) / self._residual_std
            scores = 1 / (1 + np.exp(-(z_scores - 3)))
            return scores
        
        # Decompose
        trend, seasonal, residual = self._stl_decompose(X)
        
        # Compute anomaly scores based on residuals
        z_scores = np.abs(residual - self._residual_mean) / self._residual_std
        
        # Normalize to [0, 1]
        scores = 1 / (1 + np.exp(-(z_scores - 3)))
        
        # Apply ESD test for final decision
        max_outliers = int(len(X) * self.max_anomalies)
        if max_outliers > 0:
            valid_resid = np.where(np.isnan(residual), 0, residual)
            esd_anomalies = set(self._esd_test(valid_resid, max_outliers))
            
            # Boost scores for ESD-detected anomalies
            for idx in esd_anomalies:
                scores[idx] = min(1.0, scores[idx] + 0.2)
        
        return scores
    
    def _get_contributing_features(self, x: np.ndarray) -> List[ContributingFeature]:
        """Get features contributing to anomaly."""
        return [ContributingFeature(
            name="residual_deviation",
            value=float(x) if np.isscalar(x) else float(x[0]),
            importance=1.0,
            expected_range=(
                float(self._residual_mean - 2*self._residual_std),
                float(self._residual_mean + 2*self._residual_std)
            )
        )]
    
    def _get_model_state(self) -> Dict[str, Any]:
        """Get model state for serialization."""
        return {
            "period": self.period,
            "max_anomalies": self.max_anomalies,
            "alpha": self.alpha,
            "residual_mean": self._residual_mean,
            "residual_std": self._residual_std,
            "seasonal_pattern": self._seasonal_pattern,
            "trend_coefficients": self._trend_coefficients,
        }
    
    def _set_model_state(self, state: Dict[str, Any]) -> None:
        """Set model state from deserialization."""
        self.period = state["period"]
        self.max_anomalies = state["max_anomalies"]
        self.alpha = state["alpha"]
        self._residual_mean = state["residual_mean"]
        self._residual_std = state["residual_std"]
        self._seasonal_pattern = state["seasonal_pattern"]
        self._trend_coefficients = state["trend_coefficients"]
