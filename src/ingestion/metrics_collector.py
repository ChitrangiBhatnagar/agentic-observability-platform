"""
Metrics Collector for continuous metric ingestion.
Orchestrates metric collection from multiple sources and manages the ingestion pipeline.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Set
from dataclasses import dataclass, field

from config.settings import settings
from src.types import Metric, MetricSeries, MetricType
from src.utils import get_logger, generate_id, now_utc
from .prometheus_client import PrometheusClient

logger = get_logger(__name__)


@dataclass
class MetricConfig:
    """Configuration for a metric to collect."""
    name: str
    query: str
    type: MetricType = MetricType.GAUGE
    collection_interval: int = 15  # seconds
    labels: Dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    priority: int = 1  # 1 = highest priority


@dataclass
class CollectionStats:
    """Statistics for metric collection."""
    total_collections: int = 0
    successful_collections: int = 0
    failed_collections: int = 0
    last_collection_time: Optional[datetime] = None
    last_error: Optional[str] = None
    avg_collection_duration_ms: float = 0.0


class MetricsCollector:
    """
    Continuous metrics collector with support for:
    - Multiple metric sources
    - Configurable collection intervals
    - Parallel collection
    - Error handling and retries
    - Collection statistics
    """
    
    def __init__(
        self,
        prometheus_url: Optional[str] = None,
        max_concurrent_collections: int = 50
    ):
        """
        Initialize the metrics collector.
        
        Args:
            prometheus_url: Prometheus server URL
            max_concurrent_collections: Maximum concurrent metric collections
        """
        self.prometheus_client = PrometheusClient(url=prometheus_url)
        self.max_concurrent = max_concurrent_collections
        
        # Metric configurations
        self._metric_configs: Dict[str, MetricConfig] = {}
        
        # Collection state
        self._running = False
        self._collection_task: Optional[asyncio.Task] = None
        self._semaphore = asyncio.Semaphore(max_concurrent_collections)
        
        # Statistics
        self._stats: Dict[str, CollectionStats] = {}
        
        # Callbacks
        self._on_metrics_collected: List[Callable[[List[MetricSeries]], None]] = []
        self._on_error: List[Callable[[str, Exception], None]] = []
    
    def register_metric(self, config: MetricConfig) -> None:
        """
        Register a metric for collection.
        
        Args:
            config: Metric configuration
        """
        self._metric_configs[config.name] = config
        self._stats[config.name] = CollectionStats()
        logger.info("Registered metric for collection", metric_name=config.name)
    
    def unregister_metric(self, metric_name: str) -> None:
        """
        Unregister a metric from collection.
        
        Args:
            metric_name: Name of the metric to unregister
        """
        if metric_name in self._metric_configs:
            del self._metric_configs[metric_name]
            del self._stats[metric_name]
            logger.info("Unregistered metric from collection", metric_name=metric_name)
    
    def on_metrics_collected(
        self,
        callback: Callable[[List[MetricSeries]], None]
    ) -> None:
        """
        Register a callback for when metrics are collected.
        
        Args:
            callback: Function to call with collected metrics
        """
        self._on_metrics_collected.append(callback)
    
    def on_error(
        self,
        callback: Callable[[str, Exception], None]
    ) -> None:
        """
        Register a callback for collection errors.
        
        Args:
            callback: Function to call with metric name and exception
        """
        self._on_error.append(callback)
    
    async def collect_metric(
        self,
        config: MetricConfig,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None
    ) -> List[MetricSeries]:
        """
        Collect a single metric.
        
        Args:
            config: Metric configuration
            start: Start time for range query
            end: End time for range query
            
        Returns:
            List of collected metric series
        """
        async with self._semaphore:
            start_time = asyncio.get_event_loop().time()
            stats = self._stats.get(config.name, CollectionStats())
            
            try:
                end = end or now_utc()
                start = start or end - timedelta(minutes=5)
                
                series = await self.prometheus_client.get_metric_series(
                    metric_name=config.name,
                    labels=config.labels,
                    start=start,
                    end=end,
                    step=f"{config.collection_interval}s"
                )
                
                # Update statistics
                duration_ms = (asyncio.get_event_loop().time() - start_time) * 1000
                stats.total_collections += 1
                stats.successful_collections += 1
                stats.last_collection_time = now_utc()
                stats.avg_collection_duration_ms = (
                    stats.avg_collection_duration_ms * 0.9 + duration_ms * 0.1
                )
                
                logger.debug(
                    "Collected metric",
                    metric_name=config.name,
                    series_count=len(series),
                    duration_ms=duration_ms
                )
                
                return series
                
            except Exception as e:
                stats.total_collections += 1
                stats.failed_collections += 1
                stats.last_error = str(e)
                
                logger.error(
                    "Failed to collect metric",
                    metric_name=config.name,
                    error=str(e)
                )
                
                for callback in self._on_error:
                    try:
                        callback(config.name, e)
                    except Exception:
                        pass
                
                return []
    
    async def collect_all_metrics(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None
    ) -> Dict[str, List[MetricSeries]]:
        """
        Collect all registered metrics.
        
        Args:
            start: Start time for range query
            end: End time for range query
            
        Returns:
            Dictionary mapping metric names to their series
        """
        enabled_configs = [
            config for config in self._metric_configs.values()
            if config.enabled
        ]
        
        # Sort by priority
        enabled_configs.sort(key=lambda c: c.priority)
        
        # Collect in parallel
        tasks = [
            self.collect_metric(config, start, end)
            for config in enabled_configs
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        collected = {}
        for config, result in zip(enabled_configs, results):
            if isinstance(result, Exception):
                logger.error(
                    "Collection failed",
                    metric_name=config.name,
                    error=str(result)
                )
                collected[config.name] = []
            else:
                collected[config.name] = result
                
                # Trigger callbacks
                if result:
                    for callback in self._on_metrics_collected:
                        try:
                            callback(result)
                        except Exception as e:
                            logger.error("Callback error", error=str(e))
        
        return collected
    
    async def start_continuous_collection(
        self,
        interval: int = 15
    ) -> None:
        """
        Start continuous metric collection.
        
        Args:
            interval: Collection interval in seconds
        """
        if self._running:
            logger.warning("Collection already running")
            return
        
        self._running = True
        logger.info("Starting continuous metric collection", interval=interval)
        
        async def collection_loop():
            while self._running:
                try:
                    await self.collect_all_metrics()
                except Exception as e:
                    logger.error("Collection cycle failed", error=str(e))
                
                await asyncio.sleep(interval)
        
        self._collection_task = asyncio.create_task(collection_loop())
    
    async def stop_continuous_collection(self) -> None:
        """Stop continuous metric collection."""
        if not self._running:
            return
        
        self._running = False
        if self._collection_task:
            self._collection_task.cancel()
            try:
                await self._collection_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Stopped continuous metric collection")
    
    def get_stats(self, metric_name: Optional[str] = None) -> Dict[str, CollectionStats]:
        """
        Get collection statistics.
        
        Args:
            metric_name: Specific metric name or None for all
            
        Returns:
            Dictionary of statistics
        """
        if metric_name:
            return {metric_name: self._stats.get(metric_name, CollectionStats())}
        return dict(self._stats)
    
    async def discover_metrics(
        self,
        pattern: Optional[str] = None
    ) -> List[str]:
        """
        Discover available metrics from Prometheus.
        
        Args:
            pattern: Optional regex pattern to filter metrics
            
        Returns:
            List of metric names
        """
        import re
        
        all_metrics = await self.prometheus_client.get_all_metrics()
        
        if pattern:
            regex = re.compile(pattern)
            return [m for m in all_metrics if regex.match(m)]
        
        return all_metrics
    
    async def auto_register_metrics(
        self,
        pattern: str,
        collection_interval: int = 15
    ) -> int:
        """
        Automatically register metrics matching a pattern.
        
        Args:
            pattern: Regex pattern for metric names
            collection_interval: Collection interval for registered metrics
            
        Returns:
            Number of metrics registered
        """
        metrics = await self.discover_metrics(pattern)
        
        for metric_name in metrics:
            config = MetricConfig(
                name=metric_name,
                query=metric_name,
                collection_interval=collection_interval
            )
            self.register_metric(config)
        
        logger.info(
            "Auto-registered metrics",
            pattern=pattern,
            count=len(metrics)
        )
        
        return len(metrics)
    
    async def close(self) -> None:
        """Clean up resources."""
        await self.stop_continuous_collection()
        await self.prometheus_client.close()


# Default metric configurations for common observability patterns
DEFAULT_METRICS = [
    MetricConfig(
        name="node_cpu_seconds_total",
        query='rate(node_cpu_seconds_total{mode!="idle"}[5m])',
        type=MetricType.COUNTER,
        priority=1
    ),
    MetricConfig(
        name="node_memory_MemAvailable_bytes",
        query="node_memory_MemAvailable_bytes",
        type=MetricType.GAUGE,
        priority=1
    ),
    MetricConfig(
        name="node_disk_io_time_seconds_total",
        query="rate(node_disk_io_time_seconds_total[5m])",
        type=MetricType.COUNTER,
        priority=2
    ),
    MetricConfig(
        name="node_network_receive_bytes_total",
        query="rate(node_network_receive_bytes_total[5m])",
        type=MetricType.COUNTER,
        priority=2
    ),
    MetricConfig(
        name="http_requests_total",
        query='rate(http_requests_total[5m])',
        type=MetricType.COUNTER,
        priority=1
    ),
    MetricConfig(
        name="http_request_duration_seconds",
        query='histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))',
        type=MetricType.HISTOGRAM,
        priority=1
    ),
    MetricConfig(
        name="container_cpu_usage_seconds_total",
        query='rate(container_cpu_usage_seconds_total[5m])',
        type=MetricType.COUNTER,
        priority=1
    ),
    MetricConfig(
        name="container_memory_usage_bytes",
        query="container_memory_usage_bytes",
        type=MetricType.GAUGE,
        priority=1
    ),
]
