"""
Feature Transformers for time series data.
Implements various feature extraction techniques for anomaly detection.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from scipy import stats
from scipy.signal import find_peaks

from src.utils import get_logger

logger = get_logger(__name__)


class BaseTransformer(ABC):
    """Base class for all feature transformers."""
    
    @abstractmethod
    def transform(self, values: np.ndarray) -> Dict[str, Any]:
        """
        Transform input values to features.
        
        Args:
            values: Input time series values
            
        Returns:
            Dictionary of feature names to values
        """
        pass
    
    @property
    @abstractmethod
    def feature_names(self) -> List[str]:
        """Get list of feature names produced by this transformer."""
        pass


class StatisticalFeatures(BaseTransformer):
    """
    Extract statistical features from time series.
    
    Features:
    - mean, std, min, max, median
    - skewness, kurtosis
    - percentiles (25, 75, 95, 99)
    - coefficient of variation
    - interquartile range
    """
    
    def __init__(self, percentiles: List[float] = [25, 75, 95, 99]):
        """
        Initialize statistical feature extractor.
        
        Args:
            percentiles: List of percentiles to compute
        """
        self.percentiles = percentiles
    
    @property
    def feature_names(self) -> List[str]:
        """Get feature names."""
        base = ["mean", "std", "min", "max", "median", "skewness", "kurtosis", "cv", "iqr"]
        pct_names = [f"p{int(p)}" for p in self.percentiles]
        return base + pct_names
    
    def transform(self, values: np.ndarray) -> Dict[str, Any]:
        """
        Compute statistical features.
        
        Args:
            values: Input values
            
        Returns:
            Dictionary of statistical features
        """
        if len(values) == 0:
            return {name: 0.0 for name in self.feature_names}
        
        values = np.asarray(values, dtype=np.float64)
        
        # Handle NaN values
        valid_values = values[~np.isnan(values)]
        if len(valid_values) == 0:
            return {name: 0.0 for name in self.feature_names}
        
        mean = np.mean(valid_values)
        std = np.std(valid_values)
        
        features = {
            "mean": float(mean),
            "std": float(std),
            "min": float(np.min(valid_values)),
            "max": float(np.max(valid_values)),
            "median": float(np.median(valid_values)),
            "skewness": float(stats.skew(valid_values)) if len(valid_values) > 2 else 0.0,
            "kurtosis": float(stats.kurtosis(valid_values)) if len(valid_values) > 3 else 0.0,
            "cv": float(std / mean) if mean != 0 else 0.0,
            "iqr": float(np.percentile(valid_values, 75) - np.percentile(valid_values, 25)),
        }
        
        # Add percentiles
        for p in self.percentiles:
            features[f"p{int(p)}"] = float(np.percentile(valid_values, p))
        
        return features


class SeasonalDecomposer(BaseTransformer):
    """
    Decompose time series into trend, seasonal, and residual components.
    Uses STL (Seasonal and Trend decomposition using Loess).
    """
    
    def __init__(
        self,
        period: int = 24,  # Default: daily seasonality with hourly data
        robust: bool = True
    ):
        """
        Initialize seasonal decomposer.
        
        Args:
            period: Seasonality period
            robust: Use robust fitting
        """
        self.period = period
        self.robust = robust
    
    @property
    def feature_names(self) -> List[str]:
        """Get feature names."""
        return [
            "trend_mean", "trend_slope", "trend_std",
            "seasonal_amplitude", "seasonal_mean",
            "residual_mean", "residual_std", "residual_max_abs",
            "seasonality_strength", "trend_strength"
        ]
    
    def transform(self, values: np.ndarray) -> Dict[str, Any]:
        """
        Decompose time series and extract features.
        
        Args:
            values: Input values
            
        Returns:
            Dictionary of decomposition features
        """
        if len(values) < self.period * 2:
            # Not enough data for decomposition
            return {name: 0.0 for name in self.feature_names}
        
        try:
            from statsmodels.tsa.seasonal import STL
            
            # Perform STL decomposition
            stl = STL(values, period=self.period, robust=self.robust)
            result = stl.fit()
            
            trend = result.trend
            seasonal = result.seasonal
            residual = result.resid
            
            # Calculate trend slope
            x = np.arange(len(trend))
            valid_idx = ~np.isnan(trend)
            if np.sum(valid_idx) > 1:
                slope, _ = np.polyfit(x[valid_idx], trend[valid_idx], 1)
            else:
                slope = 0.0
            
            # Calculate strength metrics
            var_resid = np.var(residual[~np.isnan(residual)])
            var_resid_trend = np.var(residual[~np.isnan(residual)] + trend[~np.isnan(trend)])
            var_resid_seasonal = np.var(residual[~np.isnan(residual)] + seasonal[~np.isnan(seasonal)])
            
            trend_strength = max(0, 1 - var_resid / var_resid_trend) if var_resid_trend > 0 else 0
            seasonality_strength = max(0, 1 - var_resid / var_resid_seasonal) if var_resid_seasonal > 0 else 0
            
            return {
                "trend_mean": float(np.nanmean(trend)),
                "trend_slope": float(slope),
                "trend_std": float(np.nanstd(trend)),
                "seasonal_amplitude": float(np.nanmax(seasonal) - np.nanmin(seasonal)),
                "seasonal_mean": float(np.nanmean(np.abs(seasonal))),
                "residual_mean": float(np.nanmean(residual)),
                "residual_std": float(np.nanstd(residual)),
                "residual_max_abs": float(np.nanmax(np.abs(residual))),
                "seasonality_strength": float(seasonality_strength),
                "trend_strength": float(trend_strength),
            }
            
        except Exception as e:
            logger.warning(f"STL decomposition failed: {e}")
            return {name: 0.0 for name in self.feature_names}


class TrendAnalyzer(BaseTransformer):
    """
    Analyze trends in time series data.
    
    Features:
    - Linear regression slope
    - Trend direction
    - Change rate
    - Momentum indicators
    """
    
    def __init__(
        self,
        short_window: int = 5,
        medium_window: int = 15,
        long_window: int = 30
    ):
        """
        Initialize trend analyzer.
        
        Args:
            short_window: Short-term window size
            medium_window: Medium-term window size
            long_window: Long-term window size
        """
        self.short_window = short_window
        self.medium_window = medium_window
        self.long_window = long_window
    
    @property
    def feature_names(self) -> List[str]:
        """Get feature names."""
        return [
            "linear_slope", "linear_intercept", "linear_r2",
            "exp_moving_avg_short", "exp_moving_avg_medium", "exp_moving_avg_long",
            "momentum_short", "momentum_medium",
            "rate_of_change_short", "rate_of_change_medium",
            "trend_direction"
        ]
    
    def _exponential_moving_average(self, values: np.ndarray, span: int) -> float:
        """Calculate exponential moving average."""
        if len(values) == 0:
            return 0.0
        alpha = 2 / (span + 1)
        ema = values[0]
        for v in values[1:]:
            ema = alpha * v + (1 - alpha) * ema
        return float(ema)
    
    def transform(self, values: np.ndarray) -> Dict[str, Any]:
        """
        Analyze trends in the time series.
        
        Args:
            values: Input values
            
        Returns:
            Dictionary of trend features
        """
        if len(values) < 3:
            return {name: 0.0 for name in self.feature_names}
        
        values = np.asarray(values, dtype=np.float64)
        valid_values = values[~np.isnan(values)]
        
        if len(valid_values) < 3:
            return {name: 0.0 for name in self.feature_names}
        
        # Linear regression
        x = np.arange(len(valid_values))
        slope, intercept = np.polyfit(x, valid_values, 1)
        
        # R-squared
        y_pred = slope * x + intercept
        ss_res = np.sum((valid_values - y_pred) ** 2)
        ss_tot = np.sum((valid_values - np.mean(valid_values)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Exponential moving averages
        ema_short = self._exponential_moving_average(valid_values, min(self.short_window, len(valid_values)))
        ema_medium = self._exponential_moving_average(valid_values, min(self.medium_window, len(valid_values)))
        ema_long = self._exponential_moving_average(valid_values, min(self.long_window, len(valid_values)))
        
        # Momentum
        momentum_short = valid_values[-1] - valid_values[-min(self.short_window, len(valid_values))]
        momentum_medium = valid_values[-1] - valid_values[-min(self.medium_window, len(valid_values))]
        
        # Rate of change
        roc_short_idx = min(self.short_window, len(valid_values) - 1)
        roc_medium_idx = min(self.medium_window, len(valid_values) - 1)
        
        roc_short = (valid_values[-1] / valid_values[-roc_short_idx] - 1) * 100 if valid_values[-roc_short_idx] != 0 else 0
        roc_medium = (valid_values[-1] / valid_values[-roc_medium_idx] - 1) * 100 if valid_values[-roc_medium_idx] != 0 else 0
        
        # Trend direction: 1 = up, 0 = flat, -1 = down
        trend_direction = 1 if slope > 0.01 else (-1 if slope < -0.01 else 0)
        
        return {
            "linear_slope": float(slope),
            "linear_intercept": float(intercept),
            "linear_r2": float(r2),
            "exp_moving_avg_short": float(ema_short),
            "exp_moving_avg_medium": float(ema_medium),
            "exp_moving_avg_long": float(ema_long),
            "momentum_short": float(momentum_short),
            "momentum_medium": float(momentum_medium),
            "rate_of_change_short": float(roc_short),
            "rate_of_change_medium": float(roc_medium),
            "trend_direction": float(trend_direction),
        }


class ChangePointDetector(BaseTransformer):
    """
    Detect change points and level shifts in time series.
    
    Uses CUSUM (Cumulative Sum) and other statistical methods.
    """
    
    def __init__(
        self,
        threshold: float = 3.0,
        min_segment_length: int = 5
    ):
        """
        Initialize change point detector.
        
        Args:
            threshold: Detection threshold in standard deviations
            min_segment_length: Minimum segment length between change points
        """
        self.threshold = threshold
        self.min_segment_length = min_segment_length
    
    @property
    def feature_names(self) -> List[str]:
        """Get feature names."""
        return [
            "num_change_points", "first_change_point_idx", "last_change_point_idx",
            "max_level_shift", "mean_segment_length",
            "cusum_max", "cusum_min", "cusum_range",
            "variance_change", "has_recent_change"
        ]
    
    def _cusum(self, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate CUSUM statistics."""
        mean = np.mean(values)
        std = np.std(values)
        if std == 0:
            std = 1
        
        normalized = (values - mean) / std
        
        cusum_pos = np.zeros(len(values))
        cusum_neg = np.zeros(len(values))
        
        for i in range(1, len(values)):
            cusum_pos[i] = max(0, cusum_pos[i-1] + normalized[i] - 0.5)
            cusum_neg[i] = min(0, cusum_neg[i-1] + normalized[i] + 0.5)
        
        return cusum_pos, cusum_neg
    
    def _detect_change_points(self, values: np.ndarray) -> List[int]:
        """Detect change points using CUSUM."""
        if len(values) < self.min_segment_length * 2:
            return []
        
        cusum_pos, cusum_neg = self._cusum(values)
        
        # Find peaks in CUSUM
        change_points = []
        
        # Check positive CUSUM
        peaks_pos, _ = find_peaks(cusum_pos, height=self.threshold, distance=self.min_segment_length)
        change_points.extend(peaks_pos.tolist())
        
        # Check negative CUSUM
        peaks_neg, _ = find_peaks(-cusum_neg, height=self.threshold, distance=self.min_segment_length)
        change_points.extend(peaks_neg.tolist())
        
        return sorted(set(change_points))
    
    def transform(self, values: np.ndarray) -> Dict[str, Any]:
        """
        Detect change points and extract features.
        
        Args:
            values: Input values
            
        Returns:
            Dictionary of change point features
        """
        if len(values) < self.min_segment_length * 2:
            return {name: 0.0 for name in self.feature_names}
        
        values = np.asarray(values, dtype=np.float64)
        valid_values = values[~np.isnan(values)]
        
        if len(valid_values) < self.min_segment_length * 2:
            return {name: 0.0 for name in self.feature_names}
        
        # Detect change points
        change_points = self._detect_change_points(valid_values)
        
        # CUSUM statistics
        cusum_pos, cusum_neg = self._cusum(valid_values)
        
        # Calculate segment lengths
        if change_points:
            segments = [change_points[0]] + [
                change_points[i+1] - change_points[i]
                for i in range(len(change_points) - 1)
            ] + [len(valid_values) - change_points[-1]]
            mean_segment_length = np.mean(segments)
        else:
            mean_segment_length = len(valid_values)
        
        # Calculate max level shift
        max_level_shift = 0.0
        if len(change_points) > 0:
            for cp in change_points:
                if cp > 0 and cp < len(valid_values) - 1:
                    before = np.mean(valid_values[max(0, cp-self.min_segment_length):cp])
                    after = np.mean(valid_values[cp:min(len(valid_values), cp+self.min_segment_length)])
                    shift = abs(after - before)
                    max_level_shift = max(max_level_shift, shift)
        
        # Variance change
        mid = len(valid_values) // 2
        var_first_half = np.var(valid_values[:mid]) if mid > 0 else 0
        var_second_half = np.var(valid_values[mid:]) if mid < len(valid_values) else 0
        variance_change = abs(var_second_half - var_first_half) / max(var_first_half, 1e-10)
        
        # Recent change
        recent_threshold = max(1, int(len(valid_values) * 0.1))
        has_recent_change = any(cp >= len(valid_values) - recent_threshold for cp in change_points)
        
        return {
            "num_change_points": float(len(change_points)),
            "first_change_point_idx": float(change_points[0]) if change_points else -1.0,
            "last_change_point_idx": float(change_points[-1]) if change_points else -1.0,
            "max_level_shift": float(max_level_shift),
            "mean_segment_length": float(mean_segment_length),
            "cusum_max": float(np.max(cusum_pos)),
            "cusum_min": float(np.min(cusum_neg)),
            "cusum_range": float(np.max(cusum_pos) - np.min(cusum_neg)),
            "variance_change": float(variance_change),
            "has_recent_change": float(has_recent_change),
        }


class RollingFeatures(BaseTransformer):
    """
    Extract rolling window features.
    
    Features computed over multiple window sizes.
    """
    
    def __init__(
        self,
        windows: List[int] = [5, 10, 15, 30]
    ):
        """
        Initialize rolling features extractor.
        
        Args:
            windows: List of window sizes
        """
        self.windows = windows
    
    @property
    def feature_names(self) -> List[str]:
        """Get feature names."""
        names = []
        for w in self.windows:
            names.extend([
                f"rolling_mean_{w}",
                f"rolling_std_{w}",
                f"rolling_min_{w}",
                f"rolling_max_{w}",
            ])
        return names
    
    def transform(self, values: np.ndarray) -> Dict[str, Any]:
        """
        Compute rolling features.
        
        Args:
            values: Input values
            
        Returns:
            Dictionary of rolling features
        """
        if len(values) == 0:
            return {name: 0.0 for name in self.feature_names}
        
        values = np.asarray(values, dtype=np.float64)
        features = {}
        
        for w in self.windows:
            if len(values) >= w:
                window = values[-w:]
            else:
                window = values
            
            features[f"rolling_mean_{w}"] = float(np.nanmean(window))
            features[f"rolling_std_{w}"] = float(np.nanstd(window))
            features[f"rolling_min_{w}"] = float(np.nanmin(window))
            features[f"rolling_max_{w}"] = float(np.nanmax(window))
        
        return features


class LagFeatures(BaseTransformer):
    """
    Extract lag features for time series.
    
    Creates features from past values at specific lag intervals.
    """
    
    def __init__(
        self,
        lags: List[int] = [1, 5, 10, 15, 30, 60]
    ):
        """
        Initialize lag features extractor.
        
        Args:
            lags: List of lag values
        """
        self.lags = lags
    
    @property
    def feature_names(self) -> List[str]:
        """Get feature names."""
        names = [f"lag_{l}" for l in self.lags]
        names.extend([f"lag_diff_{l}" for l in self.lags])
        names.extend([f"lag_ratio_{l}" for l in self.lags if l > 0])
        return names
    
    def transform(self, values: np.ndarray) -> Dict[str, Any]:
        """
        Compute lag features.
        
        Args:
            values: Input values
            
        Returns:
            Dictionary of lag features
        """
        if len(values) == 0:
            return {name: 0.0 for name in self.feature_names}
        
        values = np.asarray(values, dtype=np.float64)
        current = values[-1]
        features = {}
        
        for l in self.lags:
            idx = -(l + 1)
            if abs(idx) <= len(values):
                lag_value = values[idx]
            else:
                lag_value = values[0]
            
            features[f"lag_{l}"] = float(lag_value)
            features[f"lag_diff_{l}"] = float(current - lag_value)
            
            if l > 0:
                features[f"lag_ratio_{l}"] = float(current / lag_value) if lag_value != 0 else 1.0
        
        return features
