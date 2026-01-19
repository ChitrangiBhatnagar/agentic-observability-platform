"""
Root Cause Agent - Ranks probable causes using causal inference.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

from src.types import AgentType, AgentDecision, Severity
from src.utils import get_logger, generate_id, now_utc
from .base import BaseAgent, AgentMessage, MessagePriority

logger = get_logger(__name__)


class CauseCategory(str, Enum):
    """Categories of root causes."""
    INFRASTRUCTURE = "infrastructure"
    APPLICATION = "application"
    NETWORK = "network"
    EXTERNAL = "external"
    CAPACITY = "capacity"
    CONFIGURATION = "configuration"
    DEPENDENCY = "dependency"
    UNKNOWN = "unknown"


@dataclass
class RootCause:
    """A potential root cause for an incident."""
    id: str
    category: CauseCategory
    description: str
    probability: float
    evidence: List[str]
    affected_components: List[str]
    suggested_investigation: List[str]
    timestamp: datetime = field(default_factory=now_utc)


@dataclass
class CausalLink:
    """A causal relationship between metrics/services."""
    source: str
    target: str
    strength: float
    lag_seconds: float
    confidence: float


class RootCauseAgent(BaseAgent):
    """
    Root Cause Agent that ranks probable causes.
    
    Responsibilities:
    - Receive incident notifications from Correlation Agent
    - Build causal graphs from metric relationships
    - Rank root causes by probability
    - Provide investigation guidance
    """
    
    def __init__(
        self,
        service_topology: Optional[Dict[str, List[str]]] = None,
        known_patterns: Optional[Dict[str, Dict]] = None,
        **kwargs
    ):
        """
        Initialize Root Cause Agent.
        
        Args:
            service_topology: Service dependency graph
            known_patterns: Known problem patterns for matching
        """
        super().__init__(agent_type=AgentType.ROOT_CAUSE, **kwargs)
        
        self.service_topology = service_topology or {}
        self.known_patterns = known_patterns or self._get_default_patterns()
        
        # Causal graph
        self._causal_links: Dict[str, List[CausalLink]] = defaultdict(list)
        
        # Analysis cache
        self._analysis_cache: Dict[str, List[RootCause]] = {}
        
        # Historical cause statistics
        self._cause_stats: Dict[CauseCategory, Dict[str, float]] = defaultdict(
            lambda: {"count": 0, "success_rate": 0.5}
        )
        
        # Register message handlers
        self.register_handler("incident_created", self._handle_incident)
        self.register_handler("anomaly_context", self._handle_context)
    
    def _get_default_patterns(self) -> Dict[str, Dict]:
        """Get default problem patterns for matching."""
        return {
            "memory_exhaustion": {
                "indicators": ["memory_usage", "oom", "heap"],
                "category": CauseCategory.CAPACITY,
                "description": "Memory exhaustion detected",
                "investigation": [
                    "Check for memory leaks",
                    "Review recent deployments",
                    "Analyze heap dumps"
                ]
            },
            "cpu_saturation": {
                "indicators": ["cpu_usage", "load_average", "throttl"],
                "category": CauseCategory.CAPACITY,
                "description": "CPU saturation detected",
                "investigation": [
                    "Identify CPU-intensive processes",
                    "Check for infinite loops",
                    "Review resource limits"
                ]
            },
            "network_latency": {
                "indicators": ["latency", "response_time", "timeout", "connection"],
                "category": CauseCategory.NETWORK,
                "description": "Network latency issues",
                "investigation": [
                    "Check network connectivity",
                    "Review DNS resolution",
                    "Analyze packet loss"
                ]
            },
            "disk_pressure": {
                "indicators": ["disk", "iops", "io_wait", "storage"],
                "category": CauseCategory.INFRASTRUCTURE,
                "description": "Disk I/O pressure",
                "investigation": [
                    "Check disk utilization",
                    "Review I/O patterns",
                    "Consider disk cleanup"
                ]
            },
            "dependency_failure": {
                "indicators": ["upstream", "downstream", "external", "third_party"],
                "category": CauseCategory.DEPENDENCY,
                "description": "External dependency failure",
                "investigation": [
                    "Check dependency status pages",
                    "Review circuit breaker states",
                    "Test connectivity to dependencies"
                ]
            },
            "error_spike": {
                "indicators": ["error", "exception", "5xx", "4xx"],
                "category": CauseCategory.APPLICATION,
                "description": "Application error spike",
                "investigation": [
                    "Review error logs",
                    "Check recent deployments",
                    "Analyze error patterns"
                ]
            }
        }
    
    async def process(self, data: Dict[str, Any]) -> Optional[AgentDecision]:
        """
        Process root cause analysis request.
        
        Args:
            data: May contain:
                - incident_id: Incident to analyze
                - anomalies: Anomaly data
                - action: 'analyze', 'get_causes', 'update_graph'
        """
        action = data.get("action", "analyze")
        
        if action == "analyze":
            incident_id = data.get("incident_id")
            anomalies = data.get("anomalies", [])
            
            causes = await self._analyze_root_causes(incident_id, anomalies)
            
            return self.record_decision(
                decision=f"Identified {len(causes)} potential root causes",
                reasoning=self._format_causes_reasoning(causes),
                confidence=causes[0].probability if causes else 0.0,
                input_data=data
            )
        
        elif action == "update_causal_link":
            source = data.get("source")
            target = data.get("target")
            strength = data.get("strength", 0.5)
            lag = data.get("lag_seconds", 0)
            
            self._add_causal_link(source, target, strength, lag)
        
        elif action == "get_causes":
            incident_id = data.get("incident_id")
            return self._analysis_cache.get(incident_id, [])
        
        return None
    
    async def _handle_incident(self, message: AgentMessage) -> None:
        """Handle incident notification from Correlation Agent."""
        incident_id = message.payload.get("incident_id")
        anomaly_ids = message.payload.get("anomaly_ids", [])
        affected_services = message.payload.get("affected_services", [])
        
        logger.info(
            "Analyzing incident for root causes",
            incident_id=incident_id,
            anomaly_count=len(anomaly_ids)
        )
        
        # Request anomaly details
        self.send_message(
            recipient=AgentType.CORRELATION.value,
            message_type="get_anomaly_details",
            payload={"anomaly_ids": anomaly_ids}
        )
        
        # Perform preliminary analysis
        causes = await self._analyze_with_topology(affected_services)
        self._analysis_cache[incident_id] = causes
        
        # Notify Recommendation Agent
        if causes:
            self.send_message(
                recipient=AgentType.RECOMMENDATION.value,
                message_type="root_causes_identified",
                payload={
                    "incident_id": incident_id,
                    "causes": [
                        {
                            "id": c.id,
                            "category": c.category.value,
                            "description": c.description,
                            "probability": c.probability,
                        }
                        for c in causes[:5]
                    ]
                },
                priority=MessagePriority.HIGH
            )
    
    async def _handle_context(self, message: AgentMessage) -> None:
        """Handle additional context for analysis."""
        anomaly_data = message.payload.get("anomalies", [])
        incident_id = message.payload.get("incident_id")
        
        if incident_id and anomaly_data:
            causes = await self._analyze_root_causes(incident_id, anomaly_data)
            self._analysis_cache[incident_id] = causes
    
    async def _analyze_root_causes(
        self,
        incident_id: Optional[str],
        anomalies: List[Dict[str, Any]]
    ) -> List[RootCause]:
        """Analyze root causes from anomaly data."""
        causes: List[RootCause] = []
        
        # Extract features from anomalies
        metrics = [a.get("metric_name", "") for a in anomalies]
        services = [a.get("labels", {}).get("service", "") for a in anomalies]
        severities = [a.get("severity", "low") for a in anomalies]
        timestamps = [a.get("timestamp", now_utc()) for a in anomalies]
        
        # Pattern matching
        pattern_matches = self._match_patterns(metrics)
        for pattern_name, match_score in pattern_matches:
            pattern = self.known_patterns[pattern_name]
            causes.append(RootCause(
                id=generate_id("cause"),
                category=pattern["category"],
                description=pattern["description"],
                probability=match_score,
                evidence=[f"Matched pattern: {pattern_name}"],
                affected_components=list(set(services)),
                suggested_investigation=pattern["investigation"]
            ))
        
        # Topology-based analysis
        if services:
            topology_causes = await self._analyze_with_topology(services)
            causes.extend(topology_causes)
        
        # Temporal analysis
        temporal_causes = self._analyze_temporal_patterns(anomalies)
        causes.extend(temporal_causes)
        
        # Deduplicate and rank
        causes = self._deduplicate_causes(causes)
        causes = self._rank_causes(causes)
        
        return causes[:10]  # Top 10 causes
    
    def _match_patterns(
        self,
        metrics: List[str]
    ) -> List[Tuple[str, float]]:
        """Match metrics against known patterns."""
        matches = []
        
        for pattern_name, pattern in self.known_patterns.items():
            indicators = pattern["indicators"]
            
            match_count = 0
            for metric in metrics:
                metric_lower = metric.lower()
                for indicator in indicators:
                    if indicator.lower() in metric_lower:
                        match_count += 1
                        break
            
            if match_count > 0:
                score = min(1.0, match_count / max(len(metrics), 1) * 2)
                matches.append((pattern_name, score))
        
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches
    
    async def _analyze_with_topology(
        self,
        affected_services: List[str]
    ) -> List[RootCause]:
        """Analyze using service topology."""
        causes = []
        
        # Find common upstream services
        upstream_counts = defaultdict(int)
        
        for service in affected_services:
            # Find services that this service depends on
            for potential_root, dependents in self.service_topology.items():
                if service in dependents:
                    upstream_counts[potential_root] += 1
        
        # Rank by impact
        for upstream, count in sorted(
            upstream_counts.items(),
            key=lambda x: x[1],
            reverse=True
        ):
            if count >= 2:  # At least 2 affected services depend on it
                probability = min(0.9, count / len(affected_services))
                causes.append(RootCause(
                    id=generate_id("cause"),
                    category=CauseCategory.DEPENDENCY,
                    description=f"Upstream service '{upstream}' may be the root cause",
                    probability=probability,
                    evidence=[
                        f"{count} affected services depend on {upstream}",
                        "Topology analysis suggests cascading failure"
                    ],
                    affected_components=[upstream] + affected_services,
                    suggested_investigation=[
                        f"Check health of {upstream}",
                        f"Review logs and metrics for {upstream}",
                        "Check circuit breaker states"
                    ]
                ))
        
        return causes
    
    def _analyze_temporal_patterns(
        self,
        anomalies: List[Dict[str, Any]]
    ) -> List[RootCause]:
        """Analyze temporal patterns in anomalies."""
        causes = []
        
        if len(anomalies) < 2:
            return causes
        
        # Sort by timestamp
        sorted_anomalies = sorted(
            anomalies,
            key=lambda x: x.get("timestamp", datetime.min)
        )
        
        # Find the first anomaly (potential root cause)
        first = sorted_anomalies[0]
        first_metric = first.get("metric_name", "unknown")
        first_service = first.get("labels", {}).get("service", "unknown")
        
        # Calculate time spread
        if len(sorted_anomalies) > 1:
            timestamps = [a.get("timestamp") for a in sorted_anomalies if a.get("timestamp")]
            if timestamps and len(timestamps) > 1:
                time_spread = (timestamps[-1] - timestamps[0]).total_seconds()
                
                if time_spread > 0:
                    # If anomalies spread over time, first one is more likely root cause
                    probability = min(0.85, 0.5 + (time_spread / 600))  # Max 10 min spread
                    
                    causes.append(RootCause(
                        id=generate_id("cause"),
                        category=CauseCategory.APPLICATION,
                        description=f"Initial anomaly in {first_service}: {first_metric}",
                        probability=probability,
                        evidence=[
                            f"First anomaly occurred in {first_service}",
                            f"Anomalies spread over {time_spread:.1f} seconds",
                            "Temporal ordering suggests cascading impact"
                        ],
                        affected_components=[first_service],
                        suggested_investigation=[
                            f"Review {first_service} logs around anomaly time",
                            f"Check {first_metric} metric history",
                            "Look for deployment or config changes"
                        ]
                    ))
        
        return causes
    
    def _add_causal_link(
        self,
        source: str,
        target: str,
        strength: float,
        lag_seconds: float
    ) -> None:
        """Add or update a causal link."""
        link = CausalLink(
            source=source,
            target=target,
            strength=strength,
            lag_seconds=lag_seconds,
            confidence=0.5
        )
        
        # Update existing or add new
        existing = None
        for existing_link in self._causal_links[source]:
            if existing_link.target == target:
                existing = existing_link
                break
        
        if existing:
            # Update with exponential moving average
            existing.strength = 0.9 * existing.strength + 0.1 * strength
            existing.lag_seconds = 0.9 * existing.lag_seconds + 0.1 * lag_seconds
            existing.confidence = min(1.0, existing.confidence + 0.1)
        else:
            self._causal_links[source].append(link)
    
    def _deduplicate_causes(
        self,
        causes: List[RootCause]
    ) -> List[RootCause]:
        """Remove duplicate causes, keeping highest probability."""
        unique = {}
        
        for cause in causes:
            key = (cause.category, cause.description[:50])
            if key not in unique or cause.probability > unique[key].probability:
                unique[key] = cause
        
        return list(unique.values())
    
    def _rank_causes(self, causes: List[RootCause]) -> List[RootCause]:
        """Rank causes by probability and historical success."""
        def score(cause: RootCause) -> float:
            base_score = cause.probability
            
            # Adjust by historical success rate
            category_stats = self._cause_stats[cause.category]
            success_factor = category_stats.get("success_rate", 0.5)
            
            return base_score * (0.7 + 0.3 * success_factor)
        
        return sorted(causes, key=score, reverse=True)
    
    def _format_causes_reasoning(self, causes: List[RootCause]) -> str:
        """Format causes for decision reasoning."""
        if not causes:
            return "No root causes identified"
        
        lines = []
        for i, cause in enumerate(causes[:3], 1):
            lines.append(
                f"{i}. {cause.description} (p={cause.probability:.2f}, "
                f"category={cause.category.value})"
            )
        
        return "; ".join(lines)
    
    def record_feedback(
        self,
        cause_id: str,
        was_correct: bool
    ) -> None:
        """Record feedback on a root cause prediction."""
        # Find the cause
        for causes in self._analysis_cache.values():
            for cause in causes:
                if cause.id == cause_id:
                    # Update category statistics
                    stats = self._cause_stats[cause.category]
                    stats["count"] += 1
                    alpha = 0.1
                    current_rate = stats["success_rate"]
                    stats["success_rate"] = alpha * (1.0 if was_correct else 0.0) + (1 - alpha) * current_rate
                    
                    logger.info(
                        "Root cause feedback recorded",
                        cause_id=cause_id,
                        was_correct=was_correct,
                        category=cause.category.value
                    )
                    return
    
    def get_causal_graph(self) -> Dict[str, Any]:
        """Get the causal graph for visualization."""
        nodes = set()
        edges = []
        
        for source, links in self._causal_links.items():
            nodes.add(source)
            for link in links:
                nodes.add(link.target)
                edges.append({
                    "source": source,
                    "target": link.target,
                    "strength": link.strength,
                    "lag": link.lag_seconds
                })
        
        return {
            "nodes": [{"id": n} for n in nodes],
            "edges": edges
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get root cause analysis statistics."""
        return {
            "total_analyses": len(self._analysis_cache),
            "causal_links": sum(len(v) for v in self._causal_links.values()),
            "category_stats": {
                cat.value: dict(stats)
                for cat, stats in self._cause_stats.items()
            }
        }
