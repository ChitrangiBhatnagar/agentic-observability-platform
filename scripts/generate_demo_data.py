"""Synthetic data generator for testing and demos."""
import asyncio
import random
from datetime import datetime, timedelta
from typing import List, Dict
import math
import httpx

from src.utils.logging import get_logger

logger = get_logger(__name__)


class MetricGenerator:
    """Generate realistic synthetic metrics with anomalies."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=30.0)
        
        # Service configurations
        self.services = {
            "api-gateway": {
                "cpu_baseline": 45,
                "memory_baseline": 60,
                "latency_baseline": 100,
                "error_rate_baseline": 0.01,
            },
            "user-service": {
                "cpu_baseline": 30,
                "memory_baseline": 50,
                "latency_baseline": 50,
                "error_rate_baseline": 0.005,
            },
            "payment-service": {
                "cpu_baseline": 35,
                "memory_baseline": 55,
                "latency_baseline": 150,
                "error_rate_baseline": 0.002,
            },
            "database": {
                "cpu_baseline": 50,
                "memory_baseline": 70,
                "latency_baseline": 20,
                "error_rate_baseline": 0.001,
            },
        }
        
        # Anomaly scenarios
        self.anomaly_scenarios = [
            self._memory_leak_scenario,
            self._traffic_spike_scenario,
            self._database_slowdown_scenario,
            self._cascading_failure_scenario,
        ]
        
        self.current_anomaly = None
        self.anomaly_start = None
    
    def _generate_normal_value(
        self, 
        baseline: float, 
        timestamp: datetime,
        seasonality_period: float = 24.0,  # hours
        noise_level: float = 0.1
    ) -> float:
        """Generate normal metric value with seasonality and noise."""
        # Time-based seasonality (daily pattern)
        hour = timestamp.hour + timestamp.minute / 60.0
        seasonal = baseline * 0.2 * math.sin(2 * math.pi * hour / seasonality_period)
        
        # Random noise
        noise = random.gauss(0, baseline * noise_level)
        
        return max(0, baseline + seasonal + noise)
    
    def _memory_leak_scenario(self, service: str, metric: str, baseline: float, timestamp: datetime) -> float:
        """Simulate memory leak."""
        if metric == "memory_usage" and service in ["user-service", "api-gateway"]:
            minutes_since_start = (timestamp - self.anomaly_start).total_seconds() / 60
            # Gradual increase over time
            leak_increase = minutes_since_start * 0.5
            return baseline + leak_increase
        return self._generate_normal_value(baseline, timestamp)
    
    def _traffic_spike_scenario(self, service: str, metric: str, baseline: float, timestamp: datetime) -> float:
        """Simulate traffic spike."""
        if metric in ["cpu_usage", "latency_ms"] and service == "api-gateway":
            # Sharp spike
            return baseline * random.uniform(2.5, 3.5)
        return self._generate_normal_value(baseline, timestamp)
    
    def _database_slowdown_scenario(self, service: str, metric: str, baseline: float, timestamp: datetime) -> float:
        """Simulate database performance degradation."""
        if metric == "latency_ms" and service == "database":
            return baseline * random.uniform(5, 10)
        elif metric == "latency_ms" and service in ["user-service", "payment-service"]:
            # Dependent services also affected
            return baseline * random.uniform(2, 3)
        return self._generate_normal_value(baseline, timestamp)
    
    def _cascading_failure_scenario(self, service: str, metric: str, baseline: float, timestamp: datetime) -> float:
        """Simulate cascading failure."""
        minutes_since_start = (timestamp - self.anomaly_start).total_seconds() / 60
        
        # Start with payment service
        if minutes_since_start < 5:
            if service == "payment-service" and metric == "error_rate":
                return baseline * 100
        # Cascade to user service
        elif minutes_since_start < 10:
            if service in ["payment-service", "user-service"] and metric == "error_rate":
                return baseline * 50
        # Eventually affects api-gateway
        else:
            if metric == "error_rate":
                return baseline * 20
        
        return self._generate_normal_value(baseline, timestamp)
    
    def _should_inject_anomaly(self) -> bool:
        """Decide whether to inject anomaly."""
        if self.current_anomaly is None:
            # 5% chance to start anomaly
            return random.random() < 0.05
        else:
            # End anomaly after 10-20 minutes
            duration = (datetime.utcnow() - self.anomaly_start).total_seconds() / 60
            if duration > random.uniform(10, 20):
                logger.info(f"Ending anomaly scenario: {self.current_anomaly.__name__}")
                self.current_anomaly = None
                self.anomaly_start = None
                return False
        return False
    
    def generate_metric_value(self, service: str, metric: str, timestamp: datetime) -> float:
        """Generate a single metric value."""
        baseline = self.services[service].get(f"{metric}_baseline", 50)
        
        # Check if we should inject anomaly
        if self._should_inject_anomaly():
            self.current_anomaly = random.choice(self.anomaly_scenarios)
            self.anomaly_start = timestamp
            logger.info(f"Starting anomaly scenario: {self.current_anomaly.__name__}")
        
        # Generate value
        if self.current_anomaly:
            value = self.current_anomaly(service, metric, baseline, timestamp)
        else:
            value = self._generate_normal_value(baseline, timestamp)
        
        return round(value, 2)
    
    async def send_metrics(self, metrics: List[Dict]) -> None:
        """Send metrics to the platform API."""
        try:
            response = await self.client.post(
                f"{self.base_url}/api/v1/anomalies/ingest",
                json={"metrics": metrics}
            )
            if response.status_code == 200:
                logger.debug(f"Sent {len(metrics)} metrics")
            else:
                logger.warning(f"Failed to send metrics: {response.status_code}")
        except Exception as e:
            logger.error(f"Error sending metrics: {e}")
    
    async def run(self, interval: int = 15, duration: int = None):
        """Run the metric generator."""
        logger.info(f"Starting metric generator (interval={interval}s)")
        
        start_time = datetime.utcnow()
        iteration = 0
        
        try:
            while True:
                if duration and (datetime.utcnow() - start_time).total_seconds() > duration:
                    break
                
                timestamp = datetime.utcnow()
                metrics = []
                
                # Generate metrics for all services
                for service, config in self.services.items():
                    for metric_type in ["cpu_usage", "memory_usage", "latency_ms", "error_rate"]:
                        value = self.generate_metric_value(service, metric_type, timestamp)
                        
                        metrics.append({
                            "metric_name": metric_type,
                            "value": value,
                            "timestamp": timestamp.isoformat(),
                            "labels": {
                                "service": service,
                                "environment": "production",
                                "region": "us-east-1",
                            }
                        })
                
                # Send metrics
                await self.send_metrics(metrics)
                
                iteration += 1
                if iteration % 10 == 0:
                    logger.info(f"Generated {iteration * len(metrics)} total metrics")
                
                await asyncio.sleep(interval)
        
        except KeyboardInterrupt:
            logger.info("Stopping metric generator")
        finally:
            await self.client.aclose()


async def main():
    """Main entry point for demo generator."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate synthetic metrics")
    parser.add_argument("--url", default="http://localhost:8000", help="Platform API URL")
    parser.add_argument("--interval", type=int, default=15, help="Metric generation interval (seconds)")
    parser.add_argument("--duration", type=int, default=None, help="Run duration (seconds)")
    
    args = parser.parse_args()
    
    generator = MetricGenerator(base_url=args.url)
    await generator.run(interval=args.interval, duration=args.duration)


if __name__ == "__main__":
    asyncio.run(main())
