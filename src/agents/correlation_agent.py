"""
Correlation Agent - Links anomalies across services and metrics.
"""

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

from src.types import AgentType, AgentDecision, Anomaly, Incident, Severity
from src.utils import get_logger, generate_id, now_utc
from .base import BaseAgent, AgentMessage, MessagePriority

logger = get_logger(__name__)


@dataclass
class AnomalyCluster:
    """A cluster of correlated anomalies."""
    id: str
    anomaly_ids: List[str]
    services: Set[str]
    metrics: Set[str]
    start_time: datetime
    end_time: datetime
    severity: Severity
    correlation_score: float


class CorrelationAgent(BaseAgent):
    """
    Correlation Agent that links anomalies across services.
    
    Responsibilities:
    - Receive anomaly notifications from Detection Agent
    - Correlate anomalies based on time, service topology, and patterns
    - Create incidents from correlated anomalies
    - Build correlation graphs for root cause analysis
    """
    
    def __init__(
        self,
        correlation_window: int = 300,  # 5 minutes
        min_correlation_score: float = 0.6,
        service_topology: Optional[Dict[str, List[str]]] = None,
        **kwargs
    ):
        """
        Initialize Correlation Agent.
        
        Args:
            correlation_window: Time window for correlating anomalies (seconds)
            min_correlation_score: Minimum score to consider anomalies correlated
            service_topology: Service dependency graph
        """
        super().__init__(agent_type=AgentType.CORRELATION, **kwargs)
        
        self.correlation_window = correlation_window
        self.min_correlation_score = min_correlation_score
        self.service_topology = service_topology or {}
        
        # Active anomalies within correlation window
        self._active_anomalies: Dict[str, Anomaly] = {}
        
        # Anomaly clusters
        self._clusters: Dict[str, AnomalyCluster] = {}
        
        # Correlation history
        self._correlation_matrix: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # Incidents
        self._incidents: Dict[str, Incident] = {}
        
        # Register message handlers
        self.register_handler("anomaly_detected", self._handle_anomaly)
    
    async def process(self, data: Dict[str, Any]) -> Optional[AgentDecision]:
        """
        Process correlation request.
        
        Args:
            data: May contain:
                - anomalies: List of anomalies to correlate
                - action: 'correlate', 'get_incidents', 'get_clusters'
        """
        action = data.get("action", "correlate")
        
        if action == "correlate":
            anomalies = data.get("anomalies", [])
            for anomaly_data in anomalies:
                anomaly = self._dict_to_anomaly(anomaly_data)
                await self._process_anomaly(anomaly)
        
        elif action == "get_incidents":
            return self.record_decision(
                decision="Retrieved incidents",
                reasoning="Responding to incident query",
                confidence=1.0,
                input_data={"action": action}
            )
        
        elif action == "cleanup":
            cleaned = self._cleanup_old_anomalies()
            return self.record_decision(
                decision=f"Cleaned up {cleaned} old anomalies",
                reasoning="Periodic maintenance",
                confidence=1.0,
                input_data={"cleaned": cleaned}
            )
        
        return None
    
    async def _handle_anomaly(self, message: AgentMessage) -> None:
        """Handle incoming anomaly from Detection Agent."""
        anomaly_data = message.payload.get("anomaly")
        if anomaly_data:
            anomaly = self._dict_to_anomaly(anomaly_data)
            await self._process_anomaly(anomaly)
    
    def _dict_to_anomaly(self, data: Dict[str, Any]) -> Anomaly:
        """Convert dictionary to Anomaly object."""
        # Handle potential missing fields
        return Anomaly(
            id=data.get("id", generate_id("anom")),
            metric_name=data.get("metric_name", "unknown"),
            labels=data.get("labels", {}),
            anomaly_type=data.get("anomaly_type", "outlier"),
            severity=data.get("severity", Severity.LOW),
            ensemble_score=data.get("ensemble_score", 0.5),
            confidence=data.get("confidence", 0.5),
            value=data.get("value", 0),
            expected_value=data.get("expected_value", 0),
            deviation=data.get("deviation", 0),
            timestamp=data.get("timestamp", now_utc()),
        )
    
    async def _process_anomaly(self, anomaly: Anomaly) -> None:
        """Process a new anomaly and find correlations."""
        # Add to active anomalies
        self._active_anomalies[anomaly.id] = anomaly
        
        # Find correlated anomalies
        correlations = self._find_correlations(anomaly)
        
        if correlations:
            # Create or update cluster
            cluster = self._update_clusters(anomaly, correlations)
            
            # Check if this warrants an incident
            if self._should_create_incident(cluster):
                incident = self._create_incident(cluster)
                
                # Notify Root Cause Agent
                self.send_message(
                    recipient=AgentType.ROOT_CAUSE.value,
                    message_type="incident_created",
                    payload={
                        "incident_id": incident.id,
                        "anomaly_ids": incident.anomaly_ids,
                        "affected_services": incident.affected_services,
                    },
                    priority=MessagePriority.HIGH
                )
                
                logger.info(
                    "Created incident from correlation",
                    incident_id=incident.id,
                    anomaly_count=len(incident.anomaly_ids)
                )
        
        # Store in memory
        self.memory.remember(
            f"anomaly_{anomaly.id}",
            {
                "metric": anomaly.metric_name,
                "severity": anomaly.severity.value if hasattr(anomaly.severity, 'value') else str(anomaly.severity),
                "correlations": len(correlations),
            }
        )
    
    def _find_correlations(
        self,
        anomaly: Anomaly
    ) -> List[Tuple[str, float]]:
        """
        Find anomalies correlated with the given anomaly.
        
        Returns:
            List of (anomaly_id, correlation_score) tuples
        """
        correlations = []
        anomaly_time = anomaly.timestamp
        
        for other_id, other in self._active_anomalies.items():
            if other_id == anomaly.id:
                continue
            
            # Calculate correlation score
            score = self._calculate_correlation_score(anomaly, other)
            
            if score >= self.min_correlation_score:
                correlations.append((other_id, score))
                
                # Update correlation matrix
                self._update_correlation_matrix(anomaly, other, score)
        
        # Sort by score
        correlations.sort(key=lambda x: x[1], reverse=True)
        
        return correlations
    
    def _calculate_correlation_score(
        self,
        anomaly1: Anomaly,
        anomaly2: Anomaly
    ) -> float:
        """
        Calculate correlation score between two anomalies.
        
        Considers:
        - Time proximity
        - Service relationship
        - Metric similarity
        - Severity alignment
        """
        scores = []
        
        # Time proximity (exponential decay)
        time_diff = abs((anomaly1.timestamp - anomaly2.timestamp).total_seconds())
        time_score = np.exp(-time_diff / self.correlation_window)
        scores.append(time_score * 0.3)  # 30% weight
        
        # Service relationship
        service1 = anomaly1.labels.get("service", anomaly1.labels.get("job", ""))
        service2 = anomaly2.labels.get("service", anomaly2.labels.get("job", ""))
        
        if service1 and service2:
            if service1 == service2:
                service_score = 1.0
            elif self._are_services_connected(service1, service2):
                service_score = 0.8
            else:
                service_score = 0.2
        else:
            service_score = 0.5
        scores.append(service_score * 0.3)  # 30% weight
        
        # Metric similarity
        metric_score = self._calculate_metric_similarity(
            anomaly1.metric_name,
            anomaly2.metric_name
        )
        scores.append(metric_score * 0.2)  # 20% weight
        
        # Severity alignment
        sev_map = {Severity.LOW: 1, Severity.MEDIUM: 2, Severity.HIGH: 3, Severity.CRITICAL: 4}
        sev1 = sev_map.get(anomaly1.severity, 1)
        sev2 = sev_map.get(anomaly2.severity, 1)
        severity_score = 1 - abs(sev1 - sev2) / 3
        scores.append(severity_score * 0.2)  # 20% weight
        
        return sum(scores)
    
    def _are_services_connected(self, service1: str, service2: str) -> bool:
        """Check if two services are connected in topology."""
        if service1 in self.service_topology:
            if service2 in self.service_topology[service1]:
                return True
        if service2 in self.service_topology:
            if service1 in self.service_topology[service2]:
                return True
        return False
    
    def _calculate_metric_similarity(self, metric1: str, metric2: str) -> float:
        """Calculate similarity between metric names."""
        # Simple similarity based on common prefixes/suffixes
        parts1 = set(metric1.replace("_", " ").replace(".", " ").split())
        parts2 = set(metric2.replace("_", " ").replace(".", " ").split())
        
        if not parts1 or not parts2:
            return 0.0
        
        intersection = parts1 & parts2
        union = parts1 | parts2
        
        return len(intersection) / len(union)
    
    def _update_correlation_matrix(
        self,
        anomaly1: Anomaly,
        anomaly2: Anomaly,
        score: float
    ) -> None:
        """Update the correlation matrix with new observation."""
        key1 = f"{anomaly1.metric_name}|{anomaly1.labels.get('service', '')}"
        key2 = f"{anomaly2.metric_name}|{anomaly2.labels.get('service', '')}"
        
        # Exponential moving average
        alpha = 0.1
        current = self._correlation_matrix[key1].get(key2, 0.5)
        self._correlation_matrix[key1][key2] = alpha * score + (1 - alpha) * current
        self._correlation_matrix[key2][key1] = self._correlation_matrix[key1][key2]
    
    def _update_clusters(
        self,
        anomaly: Anomaly,
        correlations: List[Tuple[str, float]]
    ) -> AnomalyCluster:
        """Update or create anomaly clusters."""
        # Find existing cluster containing correlated anomalies
        existing_cluster = None
        correlated_ids = [c[0] for c in correlations]
        
        for cluster_id, cluster in self._clusters.items():
            if any(aid in cluster.anomaly_ids for aid in correlated_ids):
                existing_cluster = cluster
                break
        
        if existing_cluster:
            # Update existing cluster
            existing_cluster.anomaly_ids.append(anomaly.id)
            existing_cluster.metrics.add(anomaly.metric_name)
            
            service = anomaly.labels.get("service", anomaly.labels.get("job", ""))
            if service:
                existing_cluster.services.add(service)
            
            existing_cluster.end_time = anomaly.timestamp
            
            # Update severity to highest
            sev_order = [Severity.LOW, Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL]
            if sev_order.index(anomaly.severity) > sev_order.index(existing_cluster.severity):
                existing_cluster.severity = anomaly.severity
            
            return existing_cluster
        else:
            # Create new cluster
            service = anomaly.labels.get("service", anomaly.labels.get("job", ""))
            cluster = AnomalyCluster(
                id=generate_id("cluster"),
                anomaly_ids=[anomaly.id] + correlated_ids,
                services={service} if service else set(),
                metrics={anomaly.metric_name},
                start_time=anomaly.timestamp,
                end_time=anomaly.timestamp,
                severity=anomaly.severity,
                correlation_score=np.mean([c[1] for c in correlations])
            )
            
            self._clusters[cluster.id] = cluster
            return cluster
    
    def _should_create_incident(self, cluster: AnomalyCluster) -> bool:
        """Determine if a cluster warrants an incident."""
        # Create incident if:
        # - Multiple services affected
        # - High severity
        # - Many anomalies
        
        if len(cluster.services) >= 2:
            return True
        if cluster.severity in [Severity.HIGH, Severity.CRITICAL]:
            return True
        if len(cluster.anomaly_ids) >= 5:
            return True
        
        return False
    
    def _create_incident(self, cluster: AnomalyCluster) -> Incident:
        """Create an incident from a cluster."""
        incident = Incident(
            id=generate_id("inc"),
            title=f"Correlated anomalies in {', '.join(list(cluster.services)[:3])}",
            description=f"Detected {len(cluster.anomaly_ids)} correlated anomalies across {len(cluster.services)} services",
            severity=cluster.severity,
            anomaly_ids=cluster.anomaly_ids,
            affected_services=list(cluster.services),
            timestamp=cluster.start_time,
            status="open",
        )
        
        self._incidents[incident.id] = incident
        
        # Update anomalies with correlation info
        for aid in cluster.anomaly_ids:
            if aid in self._active_anomalies:
                self._active_anomalies[aid].correlated_anomaly_ids = [
                    x for x in cluster.anomaly_ids if x != aid
                ]
        
        return incident
    
    def _cleanup_old_anomalies(self) -> int:
        """Remove anomalies outside the correlation window."""
        cutoff = now_utc() - timedelta(seconds=self.correlation_window * 2)
        
        to_remove = [
            aid for aid, anomaly in self._active_anomalies.items()
            if anomaly.timestamp < cutoff
        ]
        
        for aid in to_remove:
            del self._active_anomalies[aid]
        
        return len(to_remove)
    
    def get_active_incidents(self) -> List[Incident]:
        """Get all active incidents."""
        return [i for i in self._incidents.values() if i.status == "open"]
    
    def get_correlation_graph(self) -> Dict[str, Any]:
        """Get the correlation graph for visualization."""
        nodes = []
        edges = []
        
        seen_nodes = set()
        
        for key1, correlations in self._correlation_matrix.items():
            if key1 not in seen_nodes:
                nodes.append({"id": key1, "label": key1.split("|")[0]})
                seen_nodes.add(key1)
            
            for key2, score in correlations.items():
                if key2 not in seen_nodes:
                    nodes.append({"id": key2, "label": key2.split("|")[0]})
                    seen_nodes.add(key2)
                
                if score >= self.min_correlation_score:
                    edges.append({
                        "source": key1,
                        "target": key2,
                        "weight": score
                    })
        
        return {"nodes": nodes, "edges": edges}
    
    def get_correlation_stats(self) -> Dict[str, Any]:
        """Get correlation statistics."""
        return {
            "active_anomalies": len(self._active_anomalies),
            "clusters": len(self._clusters),
            "incidents": len(self._incidents),
            "active_incidents": len(self.get_active_incidents()),
            "correlation_pairs": sum(
                len(v) for v in self._correlation_matrix.values()
            ) // 2,
        }
