"""Tests for ML models."""
import pytest
import numpy as np
from datetime import datetime

from src.models import (
    ZScoreDetector,
    STLESDDetector,
    IsolationForestDetector,
    OneClassSVMDetector,
    EnsembleDetector,
)


class TestZScoreDetector:
    """Tests for Z-Score detector."""
    
    def test_fit_predict(self, sample_time_series):
        """Test basic fit and predict."""
        detector = ZScoreDetector(threshold=3.0)
        detector.fit(sample_time_series)
        
        result = detector.detect(sample_time_series[-100:])
        assert result.anomaly_scores is not None
        assert len(result.anomaly_scores) == 100
    
    def test_online_learning(self, sample_time_series):
        """Test online learning mode."""
        detector = ZScoreDetector(online_learning=True)
        detector.fit(sample_time_series[:500])
        
        # Detect on new data
        result = detector.detect(sample_time_series[500:600])
        assert result.anomaly_scores is not None
    
    def test_mad_variant(self, sample_time_series):
        """Test MAD-based z-score."""
        detector = ZScoreDetector(use_mad=True)
        detector.fit(sample_time_series)
        
        result = detector.detect(sample_time_series[-50:])
        assert result.anomaly_scores is not None


class TestSTLESDDetector:
    """Tests for STL+ESD detector."""
    
    def test_seasonal_decomposition(self, sample_time_series):
        """Test seasonal decomposition."""
        detector = STLESDDetector(period=24)
        detector.fit(sample_time_series)
        
        assert detector.metadata.get("trend") is not None
        assert detector.metadata.get("seasonal") is not None
    
    def test_detect_anomalies(self, anomalous_time_series):
        """Test anomaly detection on series with injected anomalies."""
        detector = STLESDDetector(period=24)
        detector.fit(anomalous_time_series)
        
        result = detector.detect(anomalous_time_series)
        # Should detect the injected spike and drop
        assert np.any(result.anomaly_scores > 0.7)


class TestIsolationForest:
    """Tests for Isolation Forest detector."""
    
    def test_fit_predict(self, sample_time_series):
        """Test basic functionality."""
        # Reshape for sklearn
        X = sample_time_series.reshape(-1, 1)
        
        detector = IsolationForestDetector(contamination=0.1)
        detector.fit(X)
        
        result = detector.detect(X[-100:])
        assert result.anomaly_scores is not None
        assert len(result.anomaly_scores) == 100
    
    def test_feature_importance(self):
        """Test feature importance calculation."""
        np.random.seed(42)
        X = np.random.randn(1000, 5)
        
        detector = IsolationForestDetector()
        detector.fit(X)
        
        result = detector.detect(X[-10:])
        # Feature importance should be computed
        assert result.metadata.get("feature_importance") is not None


class TestEnsembleDetector:
    """Tests for ensemble detector."""
    
    def test_voting_strategy(self, sample_time_series):
        """Test voting ensemble."""
        X = sample_time_series.reshape(-1, 1)
        
        detectors = {
            "zscore": ZScoreDetector(),
            "iforest": IsolationForestDetector(),
        }
        
        ensemble = EnsembleDetector(detectors, strategy="voting")
        ensemble.fit(X)
        
        result = ensemble.detect(X[-50:])
        assert result.anomaly_scores is not None
    
    def test_weighted_strategy(self, sample_time_series):
        """Test weighted ensemble."""
        X = sample_time_series.reshape(-1, 1)
        
        detectors = {
            "zscore": ZScoreDetector(),
            "iforest": IsolationForestDetector(),
        }
        weights = {"zscore": 0.6, "iforest": 0.4}
        
        ensemble = EnsembleDetector(detectors, strategy="weighted", weights=weights)
        ensemble.fit(X)
        
        result = ensemble.detect(X[-50:])
        assert result.anomaly_scores is not None
