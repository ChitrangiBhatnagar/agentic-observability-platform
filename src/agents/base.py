"""
Base Agent Infrastructure.
Defines the foundation for all agents in the system.
"""

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
import asyncio
import json

from src.types import AgentType, AgentState, AgentDecision, AgentMemoryEntry
from src.utils import get_logger, generate_id, now_utc

logger = get_logger(__name__)


class MessagePriority(Enum):
    """Priority levels for agent messages."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class AgentMessage:
    """Message passed between agents."""
    id: str = field(default_factory=lambda: generate_id("msg"))
    sender: str = ""
    recipient: str = ""
    message_type: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    priority: MessagePriority = MessagePriority.NORMAL
    timestamp: datetime = field(default_factory=now_utc)
    correlation_id: Optional[str] = None
    requires_response: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "sender": self.sender,
            "recipient": self.recipient,
            "message_type": self.message_type,
            "payload": self.payload,
            "priority": self.priority.value,
            "timestamp": self.timestamp.isoformat(),
            "correlation_id": self.correlation_id,
            "requires_response": self.requires_response,
        }


class AgentMemory:
    """
    Agent memory system with short-term and long-term storage.
    
    Short-term: Recent events, decisions, observations (limited capacity)
    Long-term: Learned patterns, aggregated knowledge (persistent)
    """
    
    def __init__(
        self,
        short_term_capacity: int = 1000,
        short_term_ttl: int = 3600,  # 1 hour
        long_term_capacity: int = 10000
    ):
        """
        Initialize agent memory.
        
        Args:
            short_term_capacity: Maximum short-term entries
            short_term_ttl: TTL for short-term entries in seconds
            long_term_capacity: Maximum long-term entries
        """
        self.short_term_capacity = short_term_capacity
        self.short_term_ttl = short_term_ttl
        self.long_term_capacity = long_term_capacity
        
        # Short-term memory (recent events)
        self._short_term: deque[AgentMemoryEntry] = deque(maxlen=short_term_capacity)
        
        # Long-term memory (patterns, knowledge)
        self._long_term: Dict[str, AgentMemoryEntry] = {}
        
        # Index for fast lookups
        self._short_term_index: Dict[str, List[int]] = {}
    
    def remember(
        self,
        key: str,
        value: Any,
        long_term: bool = False,
        ttl: Optional[int] = None
    ) -> None:
        """
        Store a memory.
        
        Args:
            key: Memory key
            value: Memory value
            long_term: Store in long-term memory
            ttl: Custom TTL in seconds
        """
        entry = AgentMemoryEntry(
            key=key,
            value=value,
            ttl=ttl or (None if long_term else self.short_term_ttl),
            timestamp=now_utc()
        )
        
        if long_term:
            self._long_term[key] = entry
            # Evict if over capacity
            if len(self._long_term) > self.long_term_capacity:
                oldest_key = min(
                    self._long_term.keys(),
                    key=lambda k: self._long_term[k].timestamp
                )
                del self._long_term[oldest_key]
        else:
            self._short_term.append(entry)
    
    def recall(
        self,
        key: str,
        default: Any = None
    ) -> Any:
        """
        Retrieve a memory.
        
        Args:
            key: Memory key
            default: Default if not found
            
        Returns:
            Memory value or default
        """
        # Check long-term first
        if key in self._long_term:
            entry = self._long_term[key]
            entry.access_count += 1
            return entry.value
        
        # Check short-term
        current_time = now_utc()
        for entry in reversed(self._short_term):
            if entry.key == key:
                # Check TTL
                if entry.ttl:
                    age = (current_time - entry.timestamp).total_seconds()
                    if age > entry.ttl:
                        continue
                entry.access_count += 1
                return entry.value
        
        return default
    
    def recall_recent(
        self,
        key: str,
        limit: int = 10,
        within_seconds: Optional[int] = None
    ) -> List[Any]:
        """
        Recall recent memories matching a key.
        
        Args:
            key: Memory key to match
            limit: Maximum number of memories
            within_seconds: Only memories within this time window
            
        Returns:
            List of memory values
        """
        current_time = now_utc()
        results = []
        
        for entry in reversed(self._short_term):
            if entry.key == key:
                if within_seconds:
                    age = (current_time - entry.timestamp).total_seconds()
                    if age > within_seconds:
                        continue
                results.append(entry.value)
                if len(results) >= limit:
                    break
        
        return results
    
    def forget(self, key: str) -> bool:
        """
        Remove a memory.
        
        Args:
            key: Memory key
            
        Returns:
            True if removed
        """
        if key in self._long_term:
            del self._long_term[key]
            return True
        
        # Remove from short-term
        original_len = len(self._short_term)
        self._short_term = deque(
            [e for e in self._short_term if e.key != key],
            maxlen=self.short_term_capacity
        )
        return len(self._short_term) < original_len
    
    def clear_short_term(self) -> None:
        """Clear all short-term memories."""
        self._short_term.clear()
        self._short_term_index.clear()
    
    def cleanup_expired(self) -> int:
        """
        Remove expired short-term memories.
        
        Returns:
            Number of entries removed
        """
        current_time = now_utc()
        original_len = len(self._short_term)
        
        self._short_term = deque(
            [
                e for e in self._short_term
                if not e.ttl or (current_time - e.timestamp).total_seconds() <= e.ttl
            ],
            maxlen=self.short_term_capacity
        )
        
        return original_len - len(self._short_term)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            "short_term_count": len(self._short_term),
            "short_term_capacity": self.short_term_capacity,
            "long_term_count": len(self._long_term),
            "long_term_capacity": self.long_term_capacity,
        }
    
    def to_context(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Export recent memories as context for decision-making.
        
        Args:
            limit: Maximum entries to include
            
        Returns:
            List of memory dictionaries
        """
        context = []
        
        # Add recent short-term
        for entry in list(self._short_term)[-limit:]:
            context.append({
                "key": entry.key,
                "value": entry.value,
                "timestamp": entry.timestamp.isoformat(),
                "type": "short_term"
            })
        
        return context


class BaseAgent(ABC):
    """
    Base class for all agents in the system.
    
    Agents are autonomous units that:
    - Observe the system state
    - Make decisions based on observations
    - Communicate with other agents
    - Learn from feedback
    """
    
    def __init__(
        self,
        agent_type: AgentType,
        name: Optional[str] = None,
        memory_capacity: int = 1000,
        decision_confidence_threshold: float = 0.7
    ):
        """
        Initialize the agent.
        
        Args:
            agent_type: Type of agent
            name: Agent name
            memory_capacity: Memory capacity
            decision_confidence_threshold: Minimum confidence for decisions
        """
        self.agent_type = agent_type
        self.agent_id = generate_id(f"agent_{agent_type.value}")
        self.name = name or f"{agent_type.value}_agent"
        self.decision_confidence_threshold = decision_confidence_threshold
        
        # State
        self._is_active = True
        self._created_at = now_utc()
        self._last_active = now_utc()
        
        # Memory
        self.memory = AgentMemory(short_term_capacity=memory_capacity)
        
        # Decision tracking
        self._decisions_made = 0
        self._correct_decisions = 0
        self._decision_history: deque[AgentDecision] = deque(maxlen=100)
        
        # Message handling
        self._message_queue: asyncio.Queue[AgentMessage] = asyncio.Queue()
        self._message_handlers: Dict[str, Callable] = {}
        
        # Callbacks for outgoing messages
        self._on_message: Optional[Callable[[AgentMessage], None]] = None
        
        logger.info(
            "Initialized agent",
            agent_type=agent_type.value,
            agent_id=self.agent_id
        )
    
    @property
    def state(self) -> AgentState:
        """Get current agent state."""
        return AgentState(
            agent_type=self.agent_type,
            agent_id=self.agent_id,
            is_active=self._is_active,
            decisions_made=self._decisions_made,
            correct_decisions=self._correct_decisions,
            average_confidence=self._calculate_average_confidence(),
            short_term_memory=[],  # Not exposing for privacy
            long_term_memory_size=len(self.memory._long_term),
            last_active=self._last_active,
            created_at=self._created_at,
        )
    
    def _calculate_average_confidence(self) -> float:
        """Calculate average decision confidence."""
        if not self._decision_history:
            return 0.0
        return sum(d.confidence for d in self._decision_history) / len(self._decision_history)
    
    @abstractmethod
    async def process(self, data: Dict[str, Any]) -> Optional[AgentDecision]:
        """
        Main processing method - to be implemented by subclasses.
        
        Args:
            data: Input data for processing
            
        Returns:
            Agent decision or None if no decision made
        """
        pass
    
    def register_handler(
        self,
        message_type: str,
        handler: Callable[[AgentMessage], None]
    ) -> None:
        """
        Register a handler for a message type.
        
        Args:
            message_type: Type of message to handle
            handler: Handler function
        """
        self._message_handlers[message_type] = handler
    
    async def receive_message(self, message: AgentMessage) -> None:
        """
        Receive a message from another agent.
        
        Args:
            message: Incoming message
        """
        await self._message_queue.put(message)
        logger.debug(
            "Received message",
            agent_id=self.agent_id,
            message_type=message.message_type,
            sender=message.sender
        )
    
    async def process_messages(self) -> None:
        """Process all queued messages."""
        while not self._message_queue.empty():
            message = await self._message_queue.get()
            
            handler = self._message_handlers.get(message.message_type)
            if handler:
                try:
                    await handler(message)
                except Exception as e:
                    logger.error(
                        "Message handler error",
                        error=str(e),
                        message_type=message.message_type
                    )
            else:
                logger.warning(
                    "No handler for message type",
                    message_type=message.message_type
                )
    
    def send_message(
        self,
        recipient: str,
        message_type: str,
        payload: Dict[str, Any],
        priority: MessagePriority = MessagePriority.NORMAL,
        requires_response: bool = False
    ) -> AgentMessage:
        """
        Send a message to another agent.
        
        Args:
            recipient: Recipient agent ID or type
            message_type: Type of message
            payload: Message payload
            priority: Message priority
            requires_response: Whether response is needed
            
        Returns:
            The sent message
        """
        message = AgentMessage(
            sender=self.agent_id,
            recipient=recipient,
            message_type=message_type,
            payload=payload,
            priority=priority,
            requires_response=requires_response,
        )
        
        if self._on_message:
            self._on_message(message)
        
        logger.debug(
            "Sent message",
            agent_id=self.agent_id,
            recipient=recipient,
            message_type=message_type
        )
        
        return message
    
    def record_decision(
        self,
        decision: str,
        reasoning: str,
        confidence: float,
        input_data: Dict[str, Any],
        alternatives: Optional[List[Dict[str, Any]]] = None
    ) -> AgentDecision:
        """
        Record a decision made by the agent.
        
        Args:
            decision: The decision made
            reasoning: Reasoning behind the decision
            confidence: Confidence level
            input_data: Input data used for decision
            alternatives: Alternative options considered
            
        Returns:
            The recorded decision
        """
        agent_decision = AgentDecision(
            agent_type=self.agent_type,
            agent_id=self.agent_id,
            decision=decision,
            reasoning=reasoning,
            confidence=confidence,
            input_data=input_data,
            alternatives_considered=alternatives or [],
        )
        
        self._decision_history.append(agent_decision)
        self._decisions_made += 1
        self._last_active = now_utc()
        
        # Store in memory
        self.memory.remember(
            f"decision_{agent_decision.id}",
            agent_decision.model_dump() if hasattr(agent_decision, 'model_dump') else vars(agent_decision),
            long_term=confidence > 0.9
        )
        
        return agent_decision
    
    def receive_feedback(
        self,
        decision_id: str,
        is_correct: bool,
        feedback_details: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Receive feedback on a previous decision.
        
        Args:
            decision_id: ID of the decision
            is_correct: Whether the decision was correct
            feedback_details: Additional feedback details
        """
        if is_correct:
            self._correct_decisions += 1
        
        # Store feedback in memory
        self.memory.remember(
            f"feedback_{decision_id}",
            {
                "decision_id": decision_id,
                "is_correct": is_correct,
                "details": feedback_details
            },
            long_term=True
        )
        
        logger.info(
            "Received feedback",
            agent_id=self.agent_id,
            decision_id=decision_id,
            is_correct=is_correct
        )
    
    def get_accuracy(self) -> float:
        """Get decision accuracy rate."""
        if self._decisions_made == 0:
            return 0.0
        return self._correct_decisions / self._decisions_made
    
    def explain_decision(
        self,
        decision: AgentDecision
    ) -> str:
        """
        Generate a human-readable explanation of a decision.
        
        Args:
            decision: The decision to explain
            
        Returns:
            Human-readable explanation
        """
        return (
            f"Agent {self.name} decided: {decision.decision}\n"
            f"Reasoning: {decision.reasoning}\n"
            f"Confidence: {decision.confidence:.1%}\n"
            f"Alternatives considered: {len(decision.alternatives_considered)}"
        )
    
    async def start(self) -> None:
        """Start the agent."""
        self._is_active = True
        logger.info("Agent started", agent_id=self.agent_id)
    
    async def stop(self) -> None:
        """Stop the agent."""
        self._is_active = False
        logger.info("Agent stopped", agent_id=self.agent_id)
    
    def is_active(self) -> bool:
        """Check if agent is active."""
        return self._is_active
