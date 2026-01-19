"""
Recommendation Agent - Proposes remediation actions.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set
import numpy as np

from src.types import AgentType, AgentDecision, RecommendedAction, ActionType, Severity
from src.utils import get_logger, generate_id, now_utc
from .base import BaseAgent, AgentMessage, MessagePriority
from .root_cause_agent import CauseCategory

logger = get_logger(__name__)


class ActionRisk(str, Enum):
    """Risk level of an action."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ActionStatus(str, Enum):
    """Status of a recommended action."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXECUTED = "executed"
    FAILED = "failed"


@dataclass
class ActionTemplate:
    """Template for a remediation action."""
    id: str
    name: str
    description: str
    action_type: ActionType
    risk: ActionRisk
    applicable_categories: List[CauseCategory]
    prerequisites: List[str]
    steps: List[str]
    rollback_steps: List[str]
    estimated_impact: str
    automation_ready: bool = False


@dataclass
class RecommendationContext:
    """Context for generating recommendations."""
    incident_id: str
    causes: List[Dict[str, Any]]
    affected_services: List[str]
    severity: Severity
    time_since_start: float


class RecommendationAgent(BaseAgent):
    """
    Recommendation Agent that proposes remediation actions.
    
    Responsibilities:
    - Receive root cause analysis from Root Cause Agent
    - Match causes to remediation playbooks
    - Rank actions by effectiveness and risk
    - Track action outcomes for learning
    """
    
    def __init__(
        self,
        auto_approve_low_risk: bool = False,
        max_recommendations: int = 5,
        **kwargs
    ):
        """
        Initialize Recommendation Agent.
        
        Args:
            auto_approve_low_risk: Auto-approve low-risk actions
            max_recommendations: Maximum recommendations per incident
        """
        super().__init__(agent_type=AgentType.RECOMMENDATION, **kwargs)
        
        self.auto_approve_low_risk = auto_approve_low_risk
        self.max_recommendations = max_recommendations
        
        # Action templates
        self._templates: Dict[str, ActionTemplate] = self._load_templates()
        
        # Active recommendations
        self._recommendations: Dict[str, List[RecommendedAction]] = {}
        
        # Action history for learning
        self._action_history: List[Dict[str, Any]] = []
        
        # Effectiveness scores
        self._effectiveness: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"success_count": 0, "total_count": 0, "avg_time_to_resolve": 0}
        )
        
        # Register message handlers
        self.register_handler("root_causes_identified", self._handle_root_causes)
    
    def _load_templates(self) -> Dict[str, ActionTemplate]:
        """Load action templates."""
        return {
            "restart_service": ActionTemplate(
                id="restart_service",
                name="Restart Service",
                description="Restart the affected service to clear transient state",
                action_type=ActionType.RESTART,
                risk=ActionRisk.MEDIUM,
                applicable_categories=[CauseCategory.APPLICATION, CauseCategory.CAPACITY],
                prerequisites=["Service is restartable", "No data loss on restart"],
                steps=[
                    "Drain connections gracefully",
                    "Stop service instances",
                    "Clear local caches",
                    "Start service instances",
                    "Verify health checks"
                ],
                rollback_steps=["N/A - service will be in previous state"],
                estimated_impact="Brief service interruption (30s-2min)",
                automation_ready=True
            ),
            "scale_out": ActionTemplate(
                id="scale_out",
                name="Scale Out",
                description="Add more instances to handle load",
                action_type=ActionType.SCALE,
                risk=ActionRisk.LOW,
                applicable_categories=[CauseCategory.CAPACITY],
                prerequisites=["Auto-scaling enabled", "Resources available"],
                steps=[
                    "Calculate required additional capacity",
                    "Provision new instances",
                    "Wait for health checks",
                    "Update load balancer"
                ],
                rollback_steps=[
                    "Remove additional instances",
                    "Update load balancer"
                ],
                estimated_impact="Gradual improvement over 5-10 minutes",
                automation_ready=True
            ),
            "failover": ActionTemplate(
                id="failover",
                name="Failover to Backup",
                description="Switch to backup/standby system",
                action_type=ActionType.FAILOVER,
                risk=ActionRisk.HIGH,
                applicable_categories=[CauseCategory.INFRASTRUCTURE, CauseCategory.DEPENDENCY],
                prerequisites=["Backup system available", "Data replicated"],
                steps=[
                    "Verify backup system health",
                    "Update DNS/routing",
                    "Redirect traffic",
                    "Verify service restoration"
                ],
                rollback_steps=[
                    "Revert DNS/routing",
                    "Redirect traffic back"
                ],
                estimated_impact="Potential brief disruption during switch",
                automation_ready=False
            ),
            "rollback_deployment": ActionTemplate(
                id="rollback_deployment",
                name="Rollback Deployment",
                description="Revert to previous stable version",
                action_type=ActionType.ROLLBACK,
                risk=ActionRisk.MEDIUM,
                applicable_categories=[CauseCategory.APPLICATION, CauseCategory.CONFIGURATION],
                prerequisites=["Previous version available", "Rollback tested"],
                steps=[
                    "Identify previous stable version",
                    "Deploy previous version",
                    "Run smoke tests",
                    "Update traffic routing"
                ],
                rollback_steps=["Re-deploy current version"],
                estimated_impact="5-15 minute deployment time",
                automation_ready=True
            ),
            "increase_resources": ActionTemplate(
                id="increase_resources",
                name="Increase Resource Limits",
                description="Increase CPU/memory limits for the service",
                action_type=ActionType.SCALE,
                risk=ActionRisk.LOW,
                applicable_categories=[CauseCategory.CAPACITY],
                prerequisites=["Resources available in cluster"],
                steps=[
                    "Update resource limits in config",
                    "Apply configuration",
                    "Restart affected pods"
                ],
                rollback_steps=["Revert resource limits"],
                estimated_impact="Brief restart during apply",
                automation_ready=True
            ),
            "enable_circuit_breaker": ActionTemplate(
                id="enable_circuit_breaker",
                name="Enable Circuit Breaker",
                description="Enable circuit breaker for failing dependency",
                action_type=ActionType.RATE_LIMIT,
                risk=ActionRisk.LOW,
                applicable_categories=[CauseCategory.DEPENDENCY, CauseCategory.NETWORK],
                prerequisites=["Circuit breaker configured"],
                steps=[
                    "Identify failing dependency",
                    "Enable circuit breaker",
                    "Configure fallback behavior"
                ],
                rollback_steps=["Disable circuit breaker"],
                estimated_impact="Improved resilience, graceful degradation",
                automation_ready=True
            ),
            "clear_cache": ActionTemplate(
                id="clear_cache",
                name="Clear Cache",
                description="Clear application caches to refresh state",
                action_type=ActionType.RESTART,
                risk=ActionRisk.LOW,
                applicable_categories=[CauseCategory.APPLICATION],
                prerequisites=["Cache invalidation endpoint available"],
                steps=[
                    "Identify affected caches",
                    "Invalidate cache entries",
                    "Warm up critical paths"
                ],
                rollback_steps=["N/A - cache will rebuild"],
                estimated_impact="Temporary performance degradation",
                automation_ready=True
            ),
            "investigate_logs": ActionTemplate(
                id="investigate_logs",
                name="Investigate Logs",
                description="Manual investigation of logs and metrics",
                action_type=ActionType.INVESTIGATE,
                risk=ActionRisk.LOW,
                applicable_categories=[
                    CauseCategory.APPLICATION, CauseCategory.INFRASTRUCTURE,
                    CauseCategory.UNKNOWN
                ],
                prerequisites=["Log access available"],
                steps=[
                    "Access centralized logging",
                    "Filter by affected services and time range",
                    "Identify error patterns",
                    "Correlate with metrics"
                ],
                rollback_steps=["N/A - investigation only"],
                estimated_impact="No direct impact, diagnostic only",
                automation_ready=False
            ),
            "escalate": ActionTemplate(
                id="escalate",
                name="Escalate to On-Call",
                description="Escalate incident to on-call engineer",
                action_type=ActionType.ESCALATE,
                risk=ActionRisk.LOW,
                applicable_categories=[CauseCategory.UNKNOWN, CauseCategory.EXTERNAL],
                prerequisites=["On-call rotation configured"],
                steps=[
                    "Gather incident summary",
                    "Page on-call engineer",
                    "Provide context and access"
                ],
                rollback_steps=["N/A - human escalation"],
                estimated_impact="Human intervention required",
                automation_ready=True
            )
        }
    
    async def process(self, data: Dict[str, Any]) -> Optional[AgentDecision]:
        """
        Process recommendation request.
        
        Args:
            data: May contain:
                - incident_id: Incident to recommend for
                - causes: Root causes
                - action: 'recommend', 'approve', 'reject', 'get_recommendations'
        """
        action = data.get("action", "recommend")
        
        if action == "recommend":
            incident_id = data.get("incident_id")
            causes = data.get("causes", [])
            affected_services = data.get("affected_services", [])
            severity = data.get("severity", Severity.MEDIUM)
            
            context = RecommendationContext(
                incident_id=incident_id,
                causes=causes,
                affected_services=affected_services,
                severity=severity,
                time_since_start=data.get("time_since_start", 0)
            )
            
            recommendations = await self._generate_recommendations(context)
            
            return self.record_decision(
                decision=f"Generated {len(recommendations)} recommendations",
                reasoning=self._format_recommendations(recommendations),
                confidence=0.8,
                input_data=data
            )
        
        elif action == "approve":
            recommendation_id = data.get("recommendation_id")
            return await self._approve_action(recommendation_id)
        
        elif action == "reject":
            recommendation_id = data.get("recommendation_id")
            reason = data.get("reason", "")
            return self._reject_action(recommendation_id, reason)
        
        elif action == "get_recommendations":
            incident_id = data.get("incident_id")
            return self._recommendations.get(incident_id, [])
        
        return None
    
    async def _handle_root_causes(self, message: AgentMessage) -> None:
        """Handle root causes from Root Cause Agent."""
        incident_id = message.payload.get("incident_id")
        causes = message.payload.get("causes", [])
        
        logger.info(
            "Generating recommendations for incident",
            incident_id=incident_id,
            cause_count=len(causes)
        )
        
        context = RecommendationContext(
            incident_id=incident_id,
            causes=causes,
            affected_services=[],
            severity=Severity.MEDIUM,
            time_since_start=0
        )
        
        recommendations = await self._generate_recommendations(context)
        
        # Notify operators
        if recommendations:
            self.send_message(
                recipient=AgentType.FEEDBACK.value,
                message_type="recommendations_ready",
                payload={
                    "incident_id": incident_id,
                    "recommendations": [
                        {
                            "id": r.id,
                            "title": r.title,
                            "action_type": r.action_type.value,
                            "risk": r.risk.value if hasattr(r, 'risk') else "medium",
                            "confidence": r.confidence
                        }
                        for r in recommendations
                    ]
                },
                priority=MessagePriority.HIGH
            )
    
    async def _generate_recommendations(
        self,
        context: RecommendationContext
    ) -> List[RecommendedAction]:
        """Generate recommendations based on context."""
        candidates = []
        
        for cause in context.causes:
            category = CauseCategory(cause.get("category", "unknown"))
            
            # Find applicable templates
            for template in self._templates.values():
                if category in template.applicable_categories:
                    score = self._calculate_action_score(template, cause, context)
                    candidates.append((template, score, cause))
        
        # Sort by score
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Create recommendations
        recommendations = []
        seen_actions = set()
        
        for template, score, cause in candidates:
            if template.id in seen_actions:
                continue
            seen_actions.add(template.id)
            
            recommendation = RecommendedAction(
                id=generate_id("rec"),
                title=template.name,
                description=f"{template.description}. Related to: {cause.get('description', 'Unknown cause')}",
                action_type=template.action_type,
                target_service=context.affected_services[0] if context.affected_services else "unknown",
                parameters={
                    "template_id": template.id,
                    "steps": template.steps,
                    "rollback_steps": template.rollback_steps
                },
                confidence=score,
                risk_level=template.risk.value,
                expected_impact=template.estimated_impact
            )
            
            recommendations.append(recommendation)
            
            if len(recommendations) >= self.max_recommendations:
                break
        
        # Store recommendations
        if context.incident_id:
            self._recommendations[context.incident_id] = recommendations
        
        # Auto-approve low-risk if enabled
        if self.auto_approve_low_risk:
            for rec in recommendations:
                if rec.risk_level == ActionRisk.LOW.value:
                    await self._approve_action(rec.id, auto=True)
        
        return recommendations
    
    def _calculate_action_score(
        self,
        template: ActionTemplate,
        cause: Dict[str, Any],
        context: RecommendationContext
    ) -> float:
        """Calculate action score based on multiple factors."""
        scores = []
        
        # Cause probability
        cause_prob = cause.get("probability", 0.5)
        scores.append(cause_prob * 0.4)
        
        # Risk adjustment (lower risk is better)
        risk_weights = {
            ActionRisk.LOW: 1.0,
            ActionRisk.MEDIUM: 0.8,
            ActionRisk.HIGH: 0.5,
            ActionRisk.CRITICAL: 0.2
        }
        risk_score = risk_weights.get(template.risk, 0.5)
        scores.append(risk_score * 0.2)
        
        # Historical effectiveness
        effectiveness = self._effectiveness.get(template.id, {})
        total = effectiveness.get("total_count", 0)
        if total > 0:
            success_rate = effectiveness.get("success_count", 0) / total
        else:
            success_rate = 0.5  # Neutral if no history
        scores.append(success_rate * 0.3)
        
        # Automation readiness (bonus for automated actions)
        if template.automation_ready:
            scores.append(0.1)
        
        return sum(scores)
    
    async def _approve_action(
        self,
        recommendation_id: str,
        auto: bool = False
    ) -> AgentDecision:
        """Approve a recommended action."""
        for incident_id, recommendations in self._recommendations.items():
            for rec in recommendations:
                if rec.id == recommendation_id:
                    # Update status
                    rec.parameters["status"] = ActionStatus.APPROVED.value
                    rec.parameters["approved_at"] = now_utc().isoformat()
                    rec.parameters["auto_approved"] = auto
                    
                    logger.info(
                        "Action approved",
                        recommendation_id=recommendation_id,
                        auto=auto
                    )
                    
                    return self.record_decision(
                        decision=f"Approved action: {rec.title}",
                        reasoning="Action approved" + (" automatically" if auto else " by operator"),
                        confidence=1.0,
                        input_data={"recommendation_id": recommendation_id}
                    )
        
        return self.record_decision(
            decision="Recommendation not found",
            reasoning=f"No recommendation with id {recommendation_id}",
            confidence=0.0,
            input_data={"recommendation_id": recommendation_id}
        )
    
    def _reject_action(
        self,
        recommendation_id: str,
        reason: str
    ) -> AgentDecision:
        """Reject a recommended action."""
        for incident_id, recommendations in self._recommendations.items():
            for rec in recommendations:
                if rec.id == recommendation_id:
                    rec.parameters["status"] = ActionStatus.REJECTED.value
                    rec.parameters["rejection_reason"] = reason
                    
                    logger.info(
                        "Action rejected",
                        recommendation_id=recommendation_id,
                        reason=reason
                    )
                    
                    return self.record_decision(
                        decision=f"Rejected action: {rec.title}",
                        reasoning=f"Rejection reason: {reason}",
                        confidence=1.0,
                        input_data={"recommendation_id": recommendation_id}
                    )
        
        return self.record_decision(
            decision="Recommendation not found",
            reasoning=f"No recommendation with id {recommendation_id}",
            confidence=0.0,
            input_data={"recommendation_id": recommendation_id}
        )
    
    def record_outcome(
        self,
        recommendation_id: str,
        success: bool,
        time_to_resolve: Optional[float] = None
    ) -> None:
        """Record the outcome of an action for learning."""
        for recommendations in self._recommendations.values():
            for rec in recommendations:
                if rec.id == recommendation_id:
                    template_id = rec.parameters.get("template_id")
                    if template_id:
                        stats = self._effectiveness[template_id]
                        stats["total_count"] += 1
                        if success:
                            stats["success_count"] += 1
                        if time_to_resolve:
                            current_avg = stats.get("avg_time_to_resolve", 0)
                            count = stats["total_count"]
                            stats["avg_time_to_resolve"] = (
                                current_avg * (count - 1) + time_to_resolve
                            ) / count
                        
                        # Update action history
                        self._action_history.append({
                            "recommendation_id": recommendation_id,
                            "template_id": template_id,
                            "success": success,
                            "time_to_resolve": time_to_resolve,
                            "timestamp": now_utc().isoformat()
                        })
                        
                        logger.info(
                            "Action outcome recorded",
                            template_id=template_id,
                            success=success
                        )
                    return
    
    def _format_recommendations(
        self,
        recommendations: List[RecommendedAction]
    ) -> str:
        """Format recommendations for logging."""
        if not recommendations:
            return "No recommendations generated"
        
        lines = []
        for i, rec in enumerate(recommendations[:3], 1):
            lines.append(
                f"{i}. {rec.title} (risk={rec.risk_level}, conf={rec.confidence:.2f})"
            )
        
        return "; ".join(lines)
    
    def get_templates(self) -> List[Dict[str, Any]]:
        """Get available action templates."""
        return [
            {
                "id": t.id,
                "name": t.name,
                "description": t.description,
                "risk": t.risk.value,
                "automation_ready": t.automation_ready
            }
            for t in self._templates.values()
        ]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get recommendation statistics."""
        total_recommendations = sum(
            len(recs) for recs in self._recommendations.values()
        )
        
        return {
            "total_recommendations": total_recommendations,
            "active_incidents": len(self._recommendations),
            "templates_count": len(self._templates),
            "action_history_count": len(self._action_history),
            "effectiveness": {
                template_id: {
                    "success_rate": (
                        stats["success_count"] / stats["total_count"]
                        if stats["total_count"] > 0 else 0
                    ),
                    "total_count": stats["total_count"],
                    "avg_time_to_resolve": stats.get("avg_time_to_resolve", 0)
                }
                for template_id, stats in self._effectiveness.items()
            }
        }
