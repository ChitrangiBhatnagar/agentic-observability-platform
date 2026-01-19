"""
Stream Processor for real-time metric processing.
Handles high-throughput metric streams with windowing and buffering.
"""

import asyncio
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Deque, Dict, List, Optional, Set

from config.settings import settings
from src.types import MetricDataPoint, MetricSeries
from src.utils import get_logger, generate_id, now_utc

logger = get_logger(__name__)


@dataclass
class StreamWindow:
    """A sliding window of metric data points."""
    metric_name: str
    labels: Dict[str, str]
    window_size: int  # Number of data points
    data: Deque[MetricDataPoint] = field(default_factory=deque)
    
    def add(self, data_point: MetricDataPoint) -> Optional[MetricDataPoint]:
        """
        Add a data point to the window.
        
        Args:
            data_point: Data point to add
            
        Returns:
            Evicted data point if window was full, None otherwise
        """
        evicted = None
        if len(self.data) >= self.window_size:
            evicted = self.data.popleft()
        self.data.append(data_point)
        return evicted
    
    def get_values(self) -> List[float]:
        """Get all values in the window."""
        return [dp.value for dp in self.data]
    
    def get_timestamps(self) -> List[datetime]:
        """Get all timestamps in the window."""
        return [dp.timestamp for dp in self.data]
    
    def is_full(self) -> bool:
        """Check if window is full."""
        return len(self.data) >= self.window_size
    
    def clear(self) -> None:
        """Clear the window."""
        self.data.clear()
    
    @property
    def latest(self) -> Optional[MetricDataPoint]:
        """Get the latest data point."""
        return self.data[-1] if self.data else None
    
    @property
    def oldest(self) -> Optional[MetricDataPoint]:
        """Get the oldest data point."""
        return self.data[0] if self.data else None


@dataclass
class StreamStats:
    """Statistics for a metric stream."""
    total_points: int = 0
    points_per_second: float = 0.0
    last_update: Optional[datetime] = None
    errors: int = 0


class StreamProcessor:
    """
    Real-time stream processor for metrics.
    
    Features:
    - Sliding windows per metric
    - Buffered batch processing
    - Async event handlers
    - Backpressure handling
    """
    
    def __init__(
        self,
        window_size: int = 60,
        batch_size: int = 100,
        flush_interval: float = 1.0,
        max_buffer_size: int = 10000
    ):
        """
        Initialize the stream processor.
        
        Args:
            window_size: Default window size for metrics
            batch_size: Batch size for processing
            flush_interval: Interval for flushing buffers
            max_buffer_size: Maximum buffer size before backpressure
        """
        self.window_size = window_size
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.max_buffer_size = max_buffer_size
        
        # Windows per metric signature
        self._windows: Dict[str, StreamWindow] = {}
        
        # Input buffer
        self._buffer: Deque[tuple[str, MetricDataPoint]] = deque(maxlen=max_buffer_size)
        self._buffer_lock = asyncio.Lock()
        
        # Processing state
        self._running = False
        self._processing_task: Optional[asyncio.Task] = None
        
        # Statistics
        self._stats: Dict[str, StreamStats] = {}
        self._global_stats = StreamStats()
        
        # Event handlers
        self._on_window_complete: List[Callable[[StreamWindow], None]] = []
        self._on_batch_ready: List[Callable[[List[StreamWindow]], None]] = []
        self._on_anomaly_candidate: List[Callable[[str, MetricDataPoint], None]] = []
    
    def _get_signature(self, metric_name: str, labels: Dict[str, str]) -> str:
        """Generate a unique signature for a metric."""
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{metric_name}|{label_str}"
    
    def get_or_create_window(
        self,
        metric_name: str,
        labels: Dict[str, str],
        window_size: Optional[int] = None
    ) -> StreamWindow:
        """
        Get or create a window for a metric.
        
        Args:
            metric_name: Metric name
            labels: Metric labels
            window_size: Custom window size
            
        Returns:
            StreamWindow instance
        """
        signature = self._get_signature(metric_name, labels)
        
        if signature not in self._windows:
            self._windows[signature] = StreamWindow(
                metric_name=metric_name,
                labels=labels,
                window_size=window_size or self.window_size
            )
            self._stats[signature] = StreamStats()
        
        return self._windows[signature]
    
    async def ingest(
        self,
        metric_name: str,
        labels: Dict[str, str],
        value: float,
        timestamp: Optional[datetime] = None
    ) -> bool:
        """
        Ingest a single data point.
        
        Args:
            metric_name: Metric name
            labels: Metric labels
            value: Metric value
            timestamp: Data point timestamp
            
        Returns:
            True if successfully buffered, False if backpressure
        """
        if len(self._buffer) >= self.max_buffer_size:
            logger.warning("Buffer full, applying backpressure")
            return False
        
        data_point = MetricDataPoint(
            value=value,
            timestamp=timestamp or now_utc(),
            labels=labels
        )
        
        signature = self._get_signature(metric_name, labels)
        
        async with self._buffer_lock:
            self._buffer.append((signature, data_point))
        
        # Update stats
        self._global_stats.total_points += 1
        self._global_stats.last_update = now_utc()
        
        return True
    
    async def ingest_batch(
        self,
        series: MetricSeries
    ) -> int:
        """
        Ingest a batch of data points from a series.
        
        Args:
            series: MetricSeries to ingest
            
        Returns:
            Number of points ingested
        """
        ingested = 0
        for data_point in series.data_points:
            success = await self.ingest(
                metric_name=series.metric_name,
                labels=series.labels,
                value=data_point.value,
                timestamp=data_point.timestamp
            )
            if success:
                ingested += 1
            else:
                break
        
        return ingested
    
    async def _process_buffer(self) -> int:
        """
        Process buffered data points.
        
        Returns:
            Number of points processed
        """
        processed = 0
        batch_windows: Set[str] = set()
        
        async with self._buffer_lock:
            to_process = min(len(self._buffer), self.batch_size)
            
            for _ in range(to_process):
                if not self._buffer:
                    break
                
                signature, data_point = self._buffer.popleft()
                
                # Get or create window
                parts = signature.split("|", 1)
                metric_name = parts[0]
                labels = {}
                if len(parts) > 1 and parts[1]:
                    for pair in parts[1].split(","):
                        if "=" in pair:
                            k, v = pair.split("=", 1)
                            labels[k] = v
                
                window = self.get_or_create_window(metric_name, labels)
                window.add(data_point)
                
                # Update stats
                stats = self._stats[signature]
                stats.total_points += 1
                stats.last_update = now_utc()
                
                batch_windows.add(signature)
                processed += 1
        
        # Trigger batch callback
        if batch_windows and self._on_batch_ready:
            ready_windows = [
                self._windows[sig] for sig in batch_windows
                if self._windows[sig].is_full()
            ]
            if ready_windows:
                for callback in self._on_batch_ready:
                    try:
                        callback(ready_windows)
                    except Exception as e:
                        logger.error("Batch callback error", error=str(e))
        
        return processed
    
    async def _processing_loop(self) -> None:
        """Main processing loop."""
        while self._running:
            try:
                processed = await self._process_buffer()
                if processed > 0:
                    logger.debug("Processed data points", count=processed)
            except Exception as e:
                logger.error("Processing error", error=str(e))
            
            await asyncio.sleep(self.flush_interval)
    
    async def start(self) -> None:
        """Start the stream processor."""
        if self._running:
            logger.warning("Stream processor already running")
            return
        
        self._running = True
        self._processing_task = asyncio.create_task(self._processing_loop())
        logger.info("Stream processor started")
    
    async def stop(self) -> None:
        """Stop the stream processor."""
        if not self._running:
            return
        
        self._running = False
        if self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass
        
        # Process remaining buffer
        while self._buffer:
            await self._process_buffer()
        
        logger.info("Stream processor stopped")
    
    def on_window_complete(
        self,
        callback: Callable[[StreamWindow], None]
    ) -> None:
        """Register callback for when a window becomes complete."""
        self._on_window_complete.append(callback)
    
    def on_batch_ready(
        self,
        callback: Callable[[List[StreamWindow]], None]
    ) -> None:
        """Register callback for when a batch is ready for processing."""
        self._on_batch_ready.append(callback)
    
    def on_anomaly_candidate(
        self,
        callback: Callable[[str, MetricDataPoint], None]
    ) -> None:
        """Register callback for potential anomaly candidates."""
        self._on_anomaly_candidate.append(callback)
    
    def get_window(
        self,
        metric_name: str,
        labels: Dict[str, str]
    ) -> Optional[StreamWindow]:
        """
        Get a window for a metric.
        
        Args:
            metric_name: Metric name
            labels: Metric labels
            
        Returns:
            StreamWindow if exists, None otherwise
        """
        signature = self._get_signature(metric_name, labels)
        return self._windows.get(signature)
    
    def get_all_windows(self) -> List[StreamWindow]:
        """Get all active windows."""
        return list(self._windows.values())
    
    def get_stats(
        self,
        metric_name: Optional[str] = None
    ) -> Dict[str, StreamStats]:
        """
        Get stream statistics.
        
        Args:
            metric_name: Specific metric or None for all
            
        Returns:
            Dictionary of statistics
        """
        if metric_name:
            matching = {
                k: v for k, v in self._stats.items()
                if k.startswith(f"{metric_name}|")
            }
            return matching
        return dict(self._stats)
    
    def clear_window(
        self,
        metric_name: str,
        labels: Dict[str, str]
    ) -> bool:
        """
        Clear a specific window.
        
        Args:
            metric_name: Metric name
            labels: Metric labels
            
        Returns:
            True if window was cleared, False if not found
        """
        signature = self._get_signature(metric_name, labels)
        if signature in self._windows:
            self._windows[signature].clear()
            return True
        return False
    
    def clear_all_windows(self) -> None:
        """Clear all windows."""
        for window in self._windows.values():
            window.clear()
        logger.info("Cleared all windows")
    
    @property
    def buffer_size(self) -> int:
        """Get current buffer size."""
        return len(self._buffer)
    
    @property
    def window_count(self) -> int:
        """Get number of active windows."""
        return len(self._windows)
    
    @property
    def global_stats(self) -> StreamStats:
        """Get global stream statistics."""
        return self._global_stats
