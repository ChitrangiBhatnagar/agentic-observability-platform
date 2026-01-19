"""
Prometheus Client for querying metrics.
Handles connection to Prometheus server and executes PromQL queries.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin

import httpx

from config.settings import settings
from src.types import Metric, MetricDataPoint, MetricSeries, MetricType
from src.utils import get_logger, async_retry

logger = get_logger(__name__)


class PrometheusClient:
    """
    Async client for Prometheus API.
    
    Supports:
    - Instant queries
    - Range queries
    - Metadata queries
    - Series discovery
    """
    
    def __init__(
        self,
        url: Optional[str] = None,
        timeout: Optional[int] = None
    ):
        """
        Initialize Prometheus client.
        
        Args:
            url: Prometheus server URL (defaults to settings)
            timeout: Query timeout in seconds (defaults to settings)
        """
        self.url = url or settings.prometheus.url
        self.timeout = timeout or settings.prometheus.query_timeout
        self._client: Optional[httpx.AsyncClient] = None
    
    async def __aenter__(self) -> "PrometheusClient":
        """Async context manager entry."""
        self._client = httpx.AsyncClient(
            base_url=self.url,
            timeout=self.timeout
        )
        return self
    
    async def __aexit__(self, *args) -> None:
        """Async context manager exit."""
        if self._client:
            await self._client.aclose()
    
    @property
    def client(self) -> httpx.AsyncClient:
        """Get HTTP client, creating if needed."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.url,
                timeout=self.timeout
            )
        return self._client
    
    @async_retry(max_attempts=3, base_delay=1.0)
    async def query(
        self,
        query: str,
        time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute an instant query.
        
        Args:
            query: PromQL query string
            time: Evaluation timestamp (defaults to now)
            
        Returns:
            List of result dictionaries
        """
        params = {"query": query}
        if time:
            params["time"] = time.timestamp()
        
        response = await self.client.get("/api/v1/query", params=params)
        response.raise_for_status()
        
        data = response.json()
        if data["status"] != "success":
            raise ValueError(f"Prometheus query failed: {data.get('error', 'Unknown error')}")
        
        return data["data"]["result"]
    
    @async_retry(max_attempts=3, base_delay=1.0)
    async def query_range(
        self,
        query: str,
        start: datetime,
        end: datetime,
        step: str = "15s"
    ) -> List[Dict[str, Any]]:
        """
        Execute a range query.
        
        Args:
            query: PromQL query string
            start: Start time
            end: End time
            step: Query resolution step
            
        Returns:
            List of result dictionaries with values over time
        """
        params = {
            "query": query,
            "start": start.timestamp(),
            "end": end.timestamp(),
            "step": step
        }
        
        response = await self.client.get("/api/v1/query_range", params=params)
        response.raise_for_status()
        
        data = response.json()
        if data["status"] != "success":
            raise ValueError(f"Prometheus range query failed: {data.get('error', 'Unknown error')}")
        
        return data["data"]["result"]
    
    async def get_metric_series(
        self,
        metric_name: str,
        labels: Optional[Dict[str, str]] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        step: str = "15s"
    ) -> List[MetricSeries]:
        """
        Get time series data for a metric.
        
        Args:
            metric_name: Name of the metric
            labels: Label filters
            start: Start time (defaults to 1 hour ago)
            end: End time (defaults to now)
            step: Query resolution step
            
        Returns:
            List of MetricSeries objects
        """
        end = end or datetime.utcnow()
        start = start or end - timedelta(hours=1)
        
        # Build PromQL query with label filters
        if labels:
            label_selectors = ",".join(f'{k}="{v}"' for k, v in labels.items())
            query = f"{metric_name}{{{label_selectors}}}"
        else:
            query = metric_name
        
        results = await self.query_range(query, start, end, step)
        
        series_list = []
        for result in results:
            metric_labels = result.get("metric", {})
            metric_labels.pop("__name__", None)
            
            data_points = [
                MetricDataPoint(
                    timestamp=datetime.fromtimestamp(ts),
                    value=float(val)
                )
                for ts, val in result.get("values", [])
            ]
            
            series = MetricSeries(
                metric_name=metric_name,
                labels=metric_labels,
                data_points=data_points
            )
            series_list.append(series)
        
        return series_list
    
    async def get_all_metrics(self) -> List[str]:
        """
        Get list of all available metric names.
        
        Returns:
            List of metric names
        """
        response = await self.client.get("/api/v1/label/__name__/values")
        response.raise_for_status()
        
        data = response.json()
        if data["status"] != "success":
            raise ValueError(f"Failed to get metrics: {data.get('error', 'Unknown error')}")
        
        return data["data"]
    
    async def get_metric_metadata(
        self,
        metric_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get metadata for metrics.
        
        Args:
            metric_name: Specific metric name or None for all
            
        Returns:
            Dictionary of metric metadata
        """
        params = {}
        if metric_name:
            params["metric"] = metric_name
        
        response = await self.client.get("/api/v1/metadata", params=params)
        response.raise_for_status()
        
        data = response.json()
        if data["status"] != "success":
            raise ValueError(f"Failed to get metadata: {data.get('error', 'Unknown error')}")
        
        return data["data"]
    
    async def get_targets(self) -> Dict[str, Any]:
        """
        Get information about scrape targets.
        
        Returns:
            Dictionary with active and dropped targets
        """
        response = await self.client.get("/api/v1/targets")
        response.raise_for_status()
        
        data = response.json()
        if data["status"] != "success":
            raise ValueError(f"Failed to get targets: {data.get('error', 'Unknown error')}")
        
        return data["data"]
    
    async def get_label_values(
        self,
        label_name: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None
    ) -> List[str]:
        """
        Get all values for a specific label.
        
        Args:
            label_name: Name of the label
            start: Start time filter
            end: End time filter
            
        Returns:
            List of label values
        """
        params = {}
        if start:
            params["start"] = start.timestamp()
        if end:
            params["end"] = end.timestamp()
        
        response = await self.client.get(
            f"/api/v1/label/{label_name}/values",
            params=params
        )
        response.raise_for_status()
        
        data = response.json()
        if data["status"] != "success":
            raise ValueError(f"Failed to get label values: {data.get('error', 'Unknown error')}")
        
        return data["data"]
    
    async def get_series(
        self,
        match: List[str],
        start: Optional[datetime] = None,
        end: Optional[datetime] = None
    ) -> List[Dict[str, str]]:
        """
        Get series matching the specified selectors.
        
        Args:
            match: List of series selectors
            start: Start time
            end: End time
            
        Returns:
            List of series label sets
        """
        params = {"match[]": match}
        if start:
            params["start"] = start.timestamp()
        if end:
            params["end"] = end.timestamp()
        
        response = await self.client.get("/api/v1/series", params=params)
        response.raise_for_status()
        
        data = response.json()
        if data["status"] != "success":
            raise ValueError(f"Failed to get series: {data.get('error', 'Unknown error')}")
        
        return data["data"]
    
    async def check_health(self) -> bool:
        """
        Check if Prometheus is healthy.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            response = await self.client.get("/-/healthy")
            return response.status_code == 200
        except Exception as e:
            logger.error("Prometheus health check failed", error=str(e))
            return False
    
    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
