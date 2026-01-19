"""
Feature Pipeline for end-to-end feature processing.
Orchestrates data flow from raw metrics to feature vectors ready for ML.
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional
import numpy as np

from config.settings import settings
from src.types import MetricSeries, FeatureVector
from src.utils import get_logger, now_utc
from src.ingestion import StreamProcessor, StreamWindow
from .extractor import FeatureExtractor, IncrementalFeatureExtractor

logger = get_logger(__name__)


@dataclass
class PipelineStats:
    """Statistics for the feature pipeline."""
    total_processed: int = 0
    features_generated: int = 0
    processing_errors: int = 0
    avg_processing_time_ms: float = 0.0
    last_processed: Optional[datetime] = None


class FeaturePipeline:
    """
    End-to-end feature processing pipeline.
    
    Connects stream processing with feature extraction and provides
    a clean interface for downstream ML models.
    """
    
    def __init__(
        self,
        stream_processor: Optional[StreamProcessor] = None,
        feature_extractor: Optional[FeatureExtractor] = None,
        window_size: int = 60,
        batch_size: int = 32,
        enable_caching: bool = True,
        cache_ttl: int = 300,  # 5 minutes
    ):
        """
        Initialize the feature pipeline.
        
        Args:
            stream_processor: Stream processor instance
            feature_extractor: Feature extractor instance
            window_size: Window size for feature extraction
            batch_size: Batch size for processing
            enable_caching: Whether to cache feature vectors
            cache_ttl: Cache TTL in seconds
        """
        self.stream_processor = stream_processor or StreamProcessor(window_size=window_size)
        self.feature_extractor = feature_extractor or FeatureExtractor()
        self.window_size = window_size
        self.batch_size = batch_size
        self.enable_caching = enable_caching
        self.cache_ttl = cache_ttl
        
        # Cache for feature vectors
        self._cache: Dict[str, tuple[Dict[str, Any], datetime]] = {}
        
        # Statistics
        self._stats = PipelineStats()
        
        # Callbacks
        self._on_features_ready: List[Callable[[Dict[str, Any]], None]] = []
        self._on_batch_ready: List[Callable[[List[Dict[str, Any]]], None]] = []
        
        # Register with stream processor
        self.stream_processor.on_window_complete(self._process_window)
        self.stream_processor.on_batch_ready(self._process_batch)
        
        self._running = False
    
    def _get_cache_key(self, metric_name: str, labels: Dict[str, str]) -> str:
        """Generate cache key."""
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{metric_name}|{label_str}"
    
    def _process_window(self, window: StreamWindow) -> None:
        """Process a single window when complete."""
        try:
            start_time = asyncio.get_event_loop().time()
            
            values = window.get_values()
            if not values:
                return
            
            features = self.feature_extractor.extract(
                values=values,
                metric_name=window.metric_name,
                labels=window.labels
            )
            
            # Cache if enabled
            if self.enable_caching:
                cache_key = self._get_cache_key(window.metric_name, window.labels)
                self._cache[cache_key] = (features, now_utc())
            
            # Update stats
            processing_time = (asyncio.get_event_loop().time() - start_time) * 1000
            self._stats.total_processed += 1
            self._stats.features_generated += 1
            self._stats.last_processed = now_utc()
            self._stats.avg_processing_time_ms = (
                self._stats.avg_processing_time_ms * 0.9 + processing_time * 0.1
            )
            
            # Trigger callbacks
            for callback in self._on_features_ready:
                try:
                    callback(features)
                except Exception as e:
                    logger.error("Feature callback error", error=str(e))
                    
        except Exception as e:
            logger.error("Window processing error", error=str(e))
            self._stats.processing_errors += 1
    
    def _process_batch(self, windows: List[StreamWindow]) -> None:
        """Process a batch of windows."""
        try:
            features_batch = []
            
            for window in windows:
                values = window.get_values()
                if values:
                    features = self.feature_extractor.extract(
                        values=values,
                        metric_name=window.metric_name,
                        labels=window.labels
                    )
                    features_batch.append(features)
                    
                    # Cache
                    if self.enable_caching:
                        cache_key = self._get_cache_key(window.metric_name, window.labels)
                        self._cache[cache_key] = (features, now_utc())
            
            self._stats.total_processed += len(windows)
            self._stats.features_generated += len(features_batch)
            
            # Trigger batch callbacks
            if features_batch:
                for callback in self._on_batch_ready:
                    try:
                        callback(features_batch)
                    except Exception as e:
                        logger.error("Batch callback error", error=str(e))
                        
        except Exception as e:
            logger.error("Batch processing error", error=str(e))
            self._stats.processing_errors += 1
    
    async def start(self) -> None:
        """Start the feature pipeline."""
        if self._running:
            return
        
        self._running = True
        await self.stream_processor.start()
        
        # Start cache cleanup task
        asyncio.create_task(self._cleanup_cache_loop())
        
        logger.info("Feature pipeline started")
    
    async def stop(self) -> None:
        """Stop the feature pipeline."""
        if not self._running:
            return
        
        self._running = False
        await self.stream_processor.stop()
        
        logger.info("Feature pipeline stopped")
    
    async def _cleanup_cache_loop(self) -> None:
        """Periodically clean up expired cache entries."""
        while self._running:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                current_time = now_utc()
                expired_keys = []
                
                for key, (_, timestamp) in self._cache.items():
                    age = (current_time - timestamp).total_seconds()
                    if age > self.cache_ttl:
                        expired_keys.append(key)
                
                for key in expired_keys:
                    del self._cache[key]
                
                if expired_keys:
                    logger.debug("Cleaned up cache entries", count=len(expired_keys))
                    
            except Exception as e:
                logger.error("Cache cleanup error", error=str(e))
    
    async def ingest(
        self,
        metric_name: str,
        labels: Dict[str, str],
        value: float,
        timestamp: Optional[datetime] = None
    ) -> bool:
        """
        Ingest a data point into the pipeline.
        
        Args:
            metric_name: Metric name
            labels: Metric labels
            value: Value
            timestamp: Optional timestamp
            
        Returns:
            True if ingested successfully
        """
        return await self.stream_processor.ingest(
            metric_name=metric_name,
            labels=labels,
            value=value,
            timestamp=timestamp
        )
    
    async def ingest_series(self, series: MetricSeries) -> int:
        """
        Ingest a metric series.
        
        Args:
            series: MetricSeries to ingest
            
        Returns:
            Number of points ingested
        """
        return await self.stream_processor.ingest_batch(series)
    
    def get_features(
        self,
        metric_name: str,
        labels: Dict[str, str]
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached features for a metric.
        
        Args:
            metric_name: Metric name
            labels: Metric labels
            
        Returns:
            Feature dictionary or None
        """
        cache_key = self._get_cache_key(metric_name, labels)
        cached = self._cache.get(cache_key)
        
        if cached:
            features, timestamp = cached
            age = (now_utc() - timestamp).total_seconds()
            if age <= self.cache_ttl:
                return features
        
        return None
    
    def get_features_for_window(
        self,
        metric_name: str,
        labels: Dict[str, str]
    ) -> Optional[Dict[str, Any]]:
        """
        Get features for the current window of a metric.
        
        Args:
            metric_name: Metric name
            labels: Metric labels
            
        Returns:
            Feature dictionary or None
        """
        window = self.stream_processor.get_window(metric_name, labels)
        if window and window.is_full():
            return self.feature_extractor.extract(
                values=window.get_values(),
                metric_name=metric_name,
                labels=labels
            )
        return None
    
    def get_all_current_features(self) -> List[Dict[str, Any]]:
        """
        Get current features for all active windows.
        
        Returns:
            List of feature dictionaries
        """
        features_list = []
        
        for window in self.stream_processor.get_all_windows():
            if window.is_full():
                features = self.feature_extractor.extract(
                    values=window.get_values(),
                    metric_name=window.metric_name,
                    labels=window.labels
                )
                features_list.append(features)
        
        return features_list
    
    def on_features_ready(
        self,
        callback: Callable[[Dict[str, Any]], None]
    ) -> None:
        """Register callback for when features are ready."""
        self._on_features_ready.append(callback)
    
    def on_batch_ready(
        self,
        callback: Callable[[List[Dict[str, Any]]], None]
    ) -> None:
        """Register callback for when a batch of features is ready."""
        self._on_batch_ready.append(callback)
    
    def get_feature_matrix(
        self,
        feature_names: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Get feature matrix for all current windows.
        
        Args:
            feature_names: Specific features to include
            
        Returns:
            2D numpy array (samples x features)
        """
        features_list = self.get_all_current_features()
        return self.feature_extractor.batch_to_numpy(features_list, feature_names)
    
    @property
    def stats(self) -> PipelineStats:
        """Get pipeline statistics."""
        return self._stats
    
    @property
    def cache_size(self) -> int:
        """Get current cache size."""
        return len(self._cache)
    
    @property
    def active_windows(self) -> int:
        """Get number of active windows."""
        return self.stream_processor.window_count


class RealtimeFeaturePipeline(FeaturePipeline):
    """
    Real-time feature pipeline optimized for low-latency scenarios.
    
    Uses incremental feature extraction for better performance.
    """
    
    def __init__(
        self,
        window_size: int = 60,
        **kwargs
    ):
        """
        Initialize real-time pipeline.
        
        Args:
            window_size: Window size
            **kwargs: Arguments passed to FeaturePipeline
        """
        super().__init__(window_size=window_size, **kwargs)
        
        # Override with incremental extractor
        self.incremental_extractor = IncrementalFeatureExtractor(
            window_size=window_size
        )
    
    async def ingest_and_extract(
        self,
        metric_name: str,
        labels: Dict[str, str],
        value: float
    ) -> Optional[Dict[str, Any]]:
        """
        Ingest a value and immediately extract features if window is full.
        
        Args:
            metric_name: Metric name
            labels: Metric labels
            value: Value
            
        Returns:
            Features if window is full, None otherwise
        """
        # Update incremental extractor
        features = self.incremental_extractor.update(
            metric_name=metric_name,
            labels=labels,
            value=value
        )
        
        if features:
            # Cache and trigger callbacks
            if self.enable_caching:
                cache_key = self._get_cache_key(metric_name, labels)
                self._cache[cache_key] = (features, now_utc())
            
            for callback in self._on_features_ready:
                try:
                    callback(features)
                except Exception as e:
                    logger.error("Callback error", error=str(e))
            
            self._stats.features_generated += 1
        
        return features
