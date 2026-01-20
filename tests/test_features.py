"""Tests for feature engineering."""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.features import (
    StatisticalFeatures,
    SeasonalDecomposer,
    TrendAnalyzer,
    FeatureExtractor,
)


class TestStatisticalFeatures:
    """Tests for statistical feature transformer."""
    
    def test_basic_statistics(self, sample_time_series):
        """Test basic statistical features."""
        transformer = StatisticalFeatures(window_size=50)
        features = transformer.transform(sample_time_series)
        
        assert "mean" in features
        assert "std" in features
        assert "min" in features
        assert "max" in features
        assert features["mean"] is not None
    
    def test_percentiles(self, sample_time_series):
        """Test percentile calculation."""
        transformer = StatisticalFeatures(percentiles=[25, 50, 75])
        features = transformer.transform(sample_time_series)
        
        assert "p25" in features
        assert "p50" in features
        assert "p75" in features


class TestSeasonalDecomposer:
    """Tests for seasonal decomposition."""
    
    def test_stl_decomposition(self, sample_time_series):
        """Test STL decomposition."""
        transformer = SeasonalDecomposer(period=24, method="stl")
        features = transformer.transform(sample_time_series)
        
        assert "trend" in features
        assert "seasonal" in features
        assert "residual" in features
    
    def test_seasonal_strength(self, sample_time_series):
        """Test seasonal strength calculation."""
        transformer = SeasonalDecomposer(period=24)
        features = transformer.transform(sample_time_series)
        
        assert "seasonal_strength" in features
        assert 0 <= features["seasonal_strength"] <= 1


class TestTrendAnalyzer:
    """Tests for trend analysis."""
    
    def test_linear_trend(self):
        """Test linear trend detection."""
        # Create data with clear trend
        t = np.linspace(0, 100, 1000)
        data = 2 * t + 50 + np.random.normal(0, 1, len(t))
        
        transformer = TrendAnalyzer(window_size=100)
        features = transformer.transform(data)
        
        assert "trend_slope" in features
        assert features["trend_slope"] > 0  # Positive trend
    
    def test_momentum(self, sample_time_series):
        """Test momentum calculation."""
        transformer = TrendAnalyzer(momentum_periods=[10, 20])
        features = transformer.transform(sample_time_series)
        
        assert "momentum_10" in features
        assert "momentum_20" in features


class TestFeatureExtractor:
    """Tests for feature extractor."""
    
    def test_extract_all_features(self, sample_time_series):
        """Test extracting all features."""
        extractor = FeatureExtractor()
        features = extractor.extract(sample_time_series)
        
        assert len(features) > 0
        # Should have stats, seasonal, trend features
        assert any("mean" in k for k in features.keys())
        assert any("seasonal" in k for k in features.keys())
        assert any("trend" in k for k in features.keys())
    
    def test_feature_vector(self, sample_time_series):
        """Test feature vector conversion."""
        extractor = FeatureExtractor()
        features = extractor.extract(sample_time_series)
        vector = extractor.to_vector(features)
        
        assert isinstance(vector, np.ndarray)
        assert len(vector) > 0
