"""
Agent Orchestrator - Coordinates all agents in the system.
"""

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Type, Callable
import traceback

from src.types import AgentType, Severity
from src.utils import get_logger, generate_id, now_utc
from .base import BaseAgent, AgentMessage, MessagePriority

logger = get_logger(__name__)


class OrchestratorState(str, Enum):
    """State of the orchestrator."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"


@dataclass
class AgentHealth:
    """Health status of an agent."""
    agent_type: AgentType
    is_healthy: bool
    last_activity: datetime
    message_queue_size: int
    error_count: int
    last_error: Optional[str] = None


@dataclass
class OrchestratorMetrics:
    """Metrics for the orchestrator."""
    messages_routed: int = 0
    messages_failed: int = 0
    total_decisions: int = 0
    processing_time_ms: float = 0
    start_time: datetime = field(default_factory=now_utc)


class AgentOrchestrator:
    """
    Agent Orchestrator that coordinates all agents.
    
    Responsibilities:
    - Initialize and manage agent lifecycle
    - Route messages between agents
    - Handle agent failures and recovery
    - Monitor agent health
    - Provide unified interface for external systems
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the Agent Orchestrator.
        
        Args:
            config: Configuration for agents and orchestrator
        """
        self.config = config or {}
        
        # Agents registry
        self._agents: Dict[str, BaseAgent] = {}
        
        # Message queues
        self._message_queues: Dict[str, asyncio.Queue] = defaultdict(
            lambda: asyncio.Queue(maxsize=1000)
        )
        
        # State
        self._state = OrchestratorState.STOPPED
        self._metrics = OrchestratorMetrics()
        
        # Health tracking
        self._agent_health: Dict[str, AgentHealth] = {}
        
        # Background tasks
        self._tasks: List[asyncio.Task] = []
        
        # Event handlers
        self._event_handlers: Dict[str, List[Callable]] = defaultdict(list)
        
        # Message history for debugging
        self._message_history: List[Dict[str, Any]] = []
        self._max_history = 1000
    
    def register_agent(
        self,
        agent: BaseAgent,
        agent_id: Optional[str] = None
    ) -> str:
        """
        Register an agent with the orchestrator.
        
        Args:
            agent: Agent instance to register
            agent_id: Optional custom ID (defaults to agent type)
            
        Returns:
            Agent ID
        """
        aid = agent_id or agent.agent_type.value
        
        if aid in self._agents:
            logger.warning("Replacing existing agent", agent_id=aid)
        
        self._agents[aid] = agent
        
        # Initialize health tracking
        self._agent_health[aid] = AgentHealth(
            agent_type=agent.agent_type,
            is_healthy=True,
            last_activity=now_utc(),
            message_queue_size=0,
            error_count=0
        )
        
        # Set up message sending callback
        agent.send_message = self._create_send_callback(aid)
        
        logger.info("Agent registered", agent_id=aid, agent_type=agent.agent_type.value)
        
        return aid
    
    def _create_send_callback(self, sender_id: str) -> Callable:
        """Create a message sending callback for an agent."""
        def send_message(
            recipient: str,
            message_type: str,
            payload: Dict[str, Any],
            priority: MessagePriority = MessagePriority.NORMAL
        ) -> None:
            message = AgentMessage(
                id=generate_id("msg"),
                sender=sender_id,
                recipient=recipient,
                message_type=message_type,
                payload=payload,
                priority=priority,
                timestamp=now_utc()
            )
            
            # Queue message for delivery
            asyncio.create_task(self._route_message(message))
        
        return send_message
    
    async def _route_message(self, message: AgentMessage) -> bool:
        """Route a message to its recipient."""
        try:
            recipient = message.recipient
            
            # Track in history
            self._message_history.append({
                "id": message.id,
                "sender": message.sender,
                "recipient": recipient,
                "type": message.message_type,
                "timestamp": message.timestamp.isoformat()
            })
            
            if len(self._message_history) > self._max_history:
                self._message_history = self._message_history[-self._max_history:]
            
            # Find recipient agent
            if recipient not in self._agents:
                logger.warning(
                    "Message recipient not found",
                    recipient=recipient,
                    sender=message.sender
                )
                self._metrics.messages_failed += 1
                return False
            
            # Add to recipient's queue
            queue = self._message_queues[recipient]
            
            try:
                queue.put_nowait(message)
                self._metrics.messages_routed += 1
                return True
            except asyncio.QueueFull:
                logger.error("Message queue full", recipient=recipient)
                self._metrics.messages_failed += 1
                return False
                
        except Exception as e:
            logger.error("Message routing failed", error=str(e))
            self._metrics.messages_failed += 1
            return False
    
    async def start(self) -> None:
        """Start the orchestrator and all agents."""
        if self._state == OrchestratorState.RUNNING:
            logger.warning("Orchestrator already running")
            return
        
        self._state = OrchestratorState.STARTING
        logger.info("Starting orchestrator...")
        
        # Start message processing for each agent
        for agent_id in self._agents:
            task = asyncio.create_task(
                self._process_agent_messages(agent_id)
            )
            self._tasks.append(task)
        
        # Start health monitoring
        health_task = asyncio.create_task(self._health_monitor())
        self._tasks.append(health_task)
        
        self._state = OrchestratorState.RUNNING
        self._metrics.start_time = now_utc()
        
        logger.info(
            "Orchestrator started",
            agent_count=len(self._agents)
        )
        
        # Emit start event
        await self._emit_event("orchestrator_started", {
            "agents": list(self._agents.keys())
        })
    
    async def stop(self) -> None:
        """Stop the orchestrator and all agents."""
        if self._state == OrchestratorState.STOPPED:
            return
        
        self._state = OrchestratorState.STOPPING
        logger.info("Stopping orchestrator...")
        
        # Cancel all tasks
        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        self._tasks.clear()
        self._state = OrchestratorState.STOPPED
        
        logger.info("Orchestrator stopped")
        
        # Emit stop event
        await self._emit_event("orchestrator_stopped", {})
    
    async def pause(self) -> None:
        """Pause message processing."""
        if self._state == OrchestratorState.RUNNING:
            self._state = OrchestratorState.PAUSED
            logger.info("Orchestrator paused")
    
    async def resume(self) -> None:
        """Resume message processing."""
        if self._state == OrchestratorState.PAUSED:
            self._state = OrchestratorState.RUNNING
            logger.info("Orchestrator resumed")
    
    async def _process_agent_messages(self, agent_id: str) -> None:
        """Process messages for an agent."""
        agent = self._agents[agent_id]
        queue = self._message_queues[agent_id]
        
        while True:
            try:
                # Wait for message
                message = await queue.get()
                
                # Skip if paused
                if self._state == OrchestratorState.PAUSED:
                    await queue.put(message)
                    await asyncio.sleep(0.1)
                    continue
                
                # Update health
                self._agent_health[agent_id].last_activity = now_utc()
                self._agent_health[agent_id].message_queue_size = queue.qsize()
                
                # Process message
                start_time = datetime.now()
                
                try:
                    await agent.handle_message(message)
                except Exception as e:
                    logger.error(
                        "Agent message handling failed",
                        agent_id=agent_id,
                        message_type=message.message_type,
                        error=str(e)
                    )
                    self._agent_health[agent_id].error_count += 1
                    self._agent_health[agent_id].last_error = str(e)
                
                # Update metrics
                elapsed = (datetime.now() - start_time).total_seconds() * 1000
                self._metrics.processing_time_ms = (
                    self._metrics.processing_time_ms * 0.9 + elapsed * 0.1
                )
                
                queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    "Message processing error",
                    agent_id=agent_id,
                    error=str(e)
                )
                await asyncio.sleep(1)
    
    async def _health_monitor(self) -> None:
        """Monitor agent health."""
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                now = now_utc()
                
                for agent_id, health in self._agent_health.items():
                    # Check for inactivity
                    inactive_time = (now - health.last_activity).total_seconds()
                    
                    if inactive_time > 300:  # 5 minutes
                        health.is_healthy = False
                        logger.warning(
                            "Agent inactive",
                            agent_id=agent_id,
                            inactive_seconds=inactive_time
                        )
                    else:
                        # Check error rate
                        if health.error_count > 10:
                            health.is_healthy = False
                        else:
                            health.is_healthy = True
                    
                    # Update queue size
                    health.message_queue_size = self._message_queues[agent_id].qsize()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Health monitor error", error=str(e))
    
    async def process_anomaly(
        self,
        anomaly_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process an anomaly through the agent pipeline.
        
        This is the main entry point for external systems.
        
        Args:
            anomaly_data: Anomaly data to process
            
        Returns:
            Processing result including recommendations
        """
        if self._state != OrchestratorState.RUNNING:
            raise RuntimeError("Orchestrator not running")
        
        result = {
            "status": "processing",
            "anomaly_id": anomaly_data.get("id"),
            "timestamp": now_utc().isoformat(),
            "pipeline_stages": []
        }
        
        # Start with Detection Agent
        detection_agent = self._agents.get(AgentType.DETECTION.value)
        if detection_agent:
            try:
                decision = await detection_agent.process(anomaly_data)
                if decision:
                    result["pipeline_stages"].append({
                        "agent": "detection",
                        "decision": decision.decision,
                        "confidence": decision.confidence
                    })
                    self._metrics.total_decisions += 1
            except Exception as e:
                logger.error("Detection processing failed", error=str(e))
                result["pipeline_stages"].append({
                    "agent": "detection",
                    "error": str(e)
                })
        
        # The rest happens asynchronously through message passing
        result["status"] = "submitted"
        
        return result
    
    async def submit_feedback(
        self,
        feedback_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Submit operator feedback.
        
        Args:
            feedback_data: Feedback to process
            
        Returns:
            Processing result
        """
        feedback_agent = self._agents.get(AgentType.FEEDBACK.value)
        if not feedback_agent:
            return {"error": "Feedback agent not available"}
        
        try:
            decision = await feedback_agent.process({
                "action": "submit_feedback",
                "feedback": feedback_data
            })
            
            return {
                "status": "accepted",
                "decision": decision.decision if decision else None
            }
        except Exception as e:
            logger.error("Feedback processing failed", error=str(e))
            return {"error": str(e)}
    
    async def get_recommendations(
        self,
        incident_id: str
    ) -> List[Dict[str, Any]]:
        """
        Get recommendations for an incident.
        
        Args:
            incident_id: Incident ID
            
        Returns:
            List of recommendations
        """
        rec_agent = self._agents.get(AgentType.RECOMMENDATION.value)
        if not rec_agent:
            return []
        
        result = await rec_agent.process({
            "action": "get_recommendations",
            "incident_id": incident_id
        })
        
        if isinstance(result, list):
            return result
        return []
    
    async def get_root_causes(
        self,
        incident_id: str
    ) -> List[Dict[str, Any]]:
        """
        Get root causes for an incident.
        
        Args:
            incident_id: Incident ID
            
        Returns:
            List of root causes
        """
        rca_agent = self._agents.get(AgentType.ROOT_CAUSE.value)
        if not rca_agent:
            return []
        
        result = await rca_agent.process({
            "action": "get_causes",
            "incident_id": incident_id
        })
        
        if isinstance(result, list):
            return result
        return []
    
    def on_event(
        self,
        event_type: str,
        handler: Callable
    ) -> None:
        """Register an event handler."""
        self._event_handlers[event_type].append(handler)
    
    async def _emit_event(
        self,
        event_type: str,
        data: Dict[str, Any]
    ) -> None:
        """Emit an event to handlers."""
        for handler in self._event_handlers.get(event_type, []):
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(data)
                else:
                    handler(data)
            except Exception as e:
                logger.error(
                    "Event handler error",
                    event_type=event_type,
                    error=str(e)
                )
    
    def get_health(self) -> Dict[str, Any]:
        """Get health status of all agents."""
        return {
            "orchestrator_state": self._state.value,
            "agents": {
                agent_id: {
                    "type": health.agent_type.value,
                    "is_healthy": health.is_healthy,
                    "last_activity": health.last_activity.isoformat(),
                    "queue_size": health.message_queue_size,
                    "error_count": health.error_count,
                    "last_error": health.last_error
                }
                for agent_id, health in self._agent_health.items()
            }
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get orchestrator metrics."""
        uptime = (now_utc() - self._metrics.start_time).total_seconds()
        
        return {
            "state": self._state.value,
            "uptime_seconds": uptime,
            "messages_routed": self._metrics.messages_routed,
            "messages_failed": self._metrics.messages_failed,
            "total_decisions": self._metrics.total_decisions,
            "avg_processing_time_ms": self._metrics.processing_time_ms,
            "agent_count": len(self._agents),
            "message_throughput": (
                self._metrics.messages_routed / uptime if uptime > 0 else 0
            )
        }
    
    def get_message_history(
        self,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get recent message history."""
        return self._message_history[-limit:]
    
    def get_agent_stats(self) -> Dict[str, Any]:
        """Get statistics from all agents."""
        stats = {}
        
        for agent_id, agent in self._agents.items():
            if hasattr(agent, 'get_stats'):
                try:
                    stats[agent_id] = agent.get_stats()
                except Exception as e:
                    stats[agent_id] = {"error": str(e)}
        
        return stats


async def create_default_orchestrator(
    config: Optional[Dict[str, Any]] = None
) -> AgentOrchestrator:
    """
    Create an orchestrator with all default agents.
    
    Args:
        config: Optional configuration
        
    Returns:
        Configured orchestrator
    """
    from .detection_agent import DetectionAgent
    from .correlation_agent import CorrelationAgent
    from .root_cause_agent import RootCauseAgent
    from .recommendation_agent import RecommendationAgent
    from .feedback_agent import FeedbackAgent
    
    config = config or {}
    
    orchestrator = AgentOrchestrator(config)
    
    # Create and register agents
    detection = DetectionAgent(
        anomaly_threshold=config.get("anomaly_threshold", 0.7)
    )
    orchestrator.register_agent(detection)
    
    correlation = CorrelationAgent(
        correlation_window=config.get("correlation_window", 300),
        service_topology=config.get("service_topology", {})
    )
    orchestrator.register_agent(correlation)
    
    root_cause = RootCauseAgent(
        service_topology=config.get("service_topology", {})
    )
    orchestrator.register_agent(root_cause)
    
    recommendation = RecommendationAgent(
        auto_approve_low_risk=config.get("auto_approve_low_risk", False)
    )
    orchestrator.register_agent(recommendation)
    
    feedback = FeedbackAgent(
        min_samples_for_retrain=config.get("min_samples_for_retrain", 100),
        auto_retrain=config.get("auto_retrain", True)
    )
    orchestrator.register_agent(feedback)
    
    logger.info("Default orchestrator created with all agents")
    
    return orchestrator
