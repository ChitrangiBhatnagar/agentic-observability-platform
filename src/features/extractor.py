"""
Feature Extractor for time series data.
Orchestrates multiple transformers to create comprehensive feature vectors.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
import numpy as np

from src.types import FeatureVector, MetricSeries
from src.utils import get_logger, now_utc
from .transformers import (
    BaseTransformer,
    StatisticalFeatures,
    SeasonalDecomposer,
    TrendAnalyzer,
    ChangePointDetector,
    RollingFeatures,
    LagFeatures,
)

logger = get_logger(__name__)


class FeatureExtractor:
    """
    Main feature extractor that combines multiple transformers.
    
    Produces comprehensive feature vectors for anomaly detection.
    """
    
    def __init__(
        self,
        include_statistical: bool = True,
        include_seasonal: bool = True,
        include_trend: bool = True,
        include_change_point: bool = True,
        include_rolling: bool = True,
        include_lag: bool = True,
        seasonal_period: int = 24,
        rolling_windows: List[int] = [5, 10, 15, 30],
        lag_values: List[int] = [1, 5, 10, 15, 30, 60],
    ):
        """
        Initialize the feature extractor.
        
        Args:
            include_statistical: Include statistical features
            include_seasonal: Include seasonal decomposition features
            include_trend: Include trend features
            include_change_point: Include change point features
            include_rolling: Include rolling window features
            include_lag: Include lag features
            seasonal_period: Period for seasonal decomposition
            rolling_windows: Window sizes for rolling features
            lag_values: Lag values for lag features
        """
        self.transformers: List[BaseTransformer] = []
        
        if include_statistical:
            self.transformers.append(StatisticalFeatures())
        
        if include_seasonal:
            self.transformers.append(SeasonalDecomposer(period=seasonal_period))
        
        if include_trend:
            self.transformers.append(TrendAnalyzer())
        
        if include_change_point:
            self.transformers.append(ChangePointDetector())
        
        if include_rolling:
            self.transformers.append(RollingFeatures(windows=rolling_windows))
        
        if include_lag:
            self.transformers.append(LagFeatures(lags=lag_values))
        
        logger.info(
            "Initialized feature extractor",
            num_transformers=len(self.transformers),
            total_features=len(self.get_feature_names())
        )
    
    def get_feature_names(self) -> List[str]:
        """Get all feature names from all transformers."""
        names = []
        for transformer in self.transformers:
            names.extend(transformer.feature_names)
        return names
    
    def extract(
        self,
        values: List[float],
        metric_name: str = "",
        labels: Optional[Dict[str, str]] = None,
        timestamp: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Extract features from a time series.
        
        Args:
            values: Time series values
            metric_name: Name of the metric
            labels: Metric labels
            timestamp: Timestamp for the feature vector
            
        Returns:
            Dictionary of all extracted features
        """
        if not values:
            logger.warning("Empty values provided to feature extractor")
            return {name: 0.0 for name in self.get_feature_names()}
        
        values_array = np.asarray(values, dtype=np.float64)
        
        features = {
            "metric_name": metric_name,
            "labels": labels or {},
            "timestamp": timestamp or now_utc(),
            "window_size": len(values),
        }
        
        # Apply all transformers
        for transformer in self.transformers:
            try:
                transformer_features = transformer.transform(values_array)
                features.update(transformer_features)
            except Exception as e:
                logger.error(
                    f"Transformer failed: {transformer.__class__.__name__}",
                    error=str(e)
                )
                # Add zeros for failed transformer
                for name in transformer.feature_names:
                    features[name] = 0.0
        
        return features
    
    def extract_from_series(
        self,
        series: MetricSeries
    ) -> Dict[str, Any]:
        """
        Extract features from a MetricSeries object.
        
        Args:
            series: MetricSeries to extract features from
            
        Returns:
            Dictionary of extracted features
        """
        return self.extract(
            values=series.values,
            metric_name=series.metric_name,
            labels=series.labels,
            timestamp=series.timestamps[-1] if series.timestamps else None
        )
    
    def extract_to_vector(
        self,
        values: List[float],
        metric_name: str = "",
        labels: Optional[Dict[str, str]] = None,
    ) -> FeatureVector:
        """
        Extract features and return a FeatureVector object.
        
        Args:
            values: Time series values
            metric_name: Name of the metric
            labels: Metric labels
            
        Returns:
            FeatureVector with extracted features
        """
        features = self.extract(values, metric_name, labels)
        
        # Map to FeatureVector fields
        return FeatureVector(
            metric_name=metric_name,
            labels=labels or {},
            timestamp=features.get("timestamp", now_utc()),
            
            # Statistical features
            mean=features.get("mean", 0.0),
            std=features.get("std", 0.0),
            min=features.get("min", 0.0),
            max=features.get("max", 0.0),
            median=features.get("median", 0.0),
            skewness=features.get("skewness", 0.0),
            kurtosis=features.get("kurtosis", 0.0),
            
            # Rolling features
            rolling_mean_5=features.get("rolling_mean_5", 0.0),
            rolling_mean_15=features.get("rolling_mean_15", 0.0),
            rolling_std_5=features.get("rolling_std_5", 0.0),
            rolling_std_15=features.get("rolling_std_15", 0.0),
            
            # Change features
            change_rate=features.get("rate_of_change_short", 0.0),
            trend_slope=features.get("linear_slope", 0.0),
            
            # Seasonal features
            seasonal_component=features.get("seasonal_mean", None),
            residual_component=features.get("residual_mean", None),
            
            # Lag features
            lag_1=features.get("lag_1", None),
            lag_5=features.get("lag_5", None),
            lag_15=features.get("lag_15", None),
            
            # Meta
            window_size=features.get("window_size", len(values)),
            raw_values=values[-100:] if len(values) > 100 else values,  # Keep last 100
        )
    
    def extract_batch(
        self,
        series_list: List[MetricSeries]
    ) -> List[Dict[str, Any]]:
        """
        Extract features from multiple series.
        
        Args:
            series_list: List of MetricSeries
            
        Returns:
            List of feature dictionaries
        """
        return [self.extract_from_series(series) for series in series_list]
    
    def to_numpy(
        self,
        features: Dict[str, Any],
        feature_names: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Convert feature dictionary to numpy array.
        
        Args:
            features: Feature dictionary
            feature_names: Ordered list of feature names (defaults to all)
            
        Returns:
            Numpy array of feature values
        """
        if feature_names is None:
            feature_names = self.get_feature_names()
        
        return np.array([features.get(name, 0.0) for name in feature_names])
    
    def batch_to_numpy(
        self,
        features_list: List[Dict[str, Any]],
        feature_names: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Convert list of feature dictionaries to 2D numpy array.
        
        Args:
            features_list: List of feature dictionaries
            feature_names: Ordered list of feature names
            
        Returns:
            2D numpy array (samples x features)
        """
        if not features_list:
            return np.array([])
        
        if feature_names is None:
            feature_names = self.get_feature_names()
        
        return np.array([
            [f.get(name, 0.0) for name in feature_names]
            for f in features_list
        ])


class IncrementalFeatureExtractor(FeatureExtractor):
    """
    Incremental feature extractor for streaming scenarios.
    
    Maintains state to efficiently compute features as new data arrives.
    """
    
    def __init__(
        self,
        window_size: int = 60,
        **kwargs
    ):
        """
        Initialize incremental feature extractor.
        
        Args:
            window_size: Size of the sliding window
            **kwargs: Arguments passed to FeatureExtractor
        """
        super().__init__(**kwargs)
        self.window_size = window_size
        self._buffers: Dict[str, List[float]] = {}
    
    def _get_buffer_key(self, metric_name: str, labels: Dict[str, str]) -> str:
        """Generate buffer key from metric name and labels."""
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{metric_name}|{label_str}"
    
    def update(
        self,
        metric_name: str,
        labels: Dict[str, str],
        value: float
    ) -> Optional[Dict[str, Any]]:
        """
        Update with a new value and optionally extract features.
        
        Args:
            metric_name: Name of the metric
            labels: Metric labels
            value: New value
            
        Returns:
            Features if window is full, None otherwise
        """
        key = self._get_buffer_key(metric_name, labels)
        
        if key not in self._buffers:
            self._buffers[key] = []
        
        buffer = self._buffers[key]
        buffer.append(value)
        
        # Maintain window size
        if len(buffer) > self.window_size:
            buffer.pop(0)
        
        # Only extract features when window is full
        if len(buffer) >= self.window_size:
            return self.extract(
                values=buffer,
                metric_name=metric_name,
                labels=labels
            )
        
        return None
    
    def get_buffer(
        self,
        metric_name: str,
        labels: Dict[str, str]
    ) -> List[float]:
        """Get the current buffer for a metric."""
        key = self._get_buffer_key(metric_name, labels)
        return self._buffers.get(key, [])
    
    def clear_buffer(
        self,
        metric_name: str,
        labels: Dict[str, str]
    ) -> None:
        """Clear the buffer for a metric."""
        key = self._get_buffer_key(metric_name, labels)
        if key in self._buffers:
            del self._buffers[key]
    
    def clear_all_buffers(self) -> None:
        """Clear all buffers."""
        self._buffers.clear()
