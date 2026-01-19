"""
Feedback Agent - Processes human feedback and retrains models.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Callable
import numpy as np

from src.types import AgentType, AgentDecision, FeedbackType, OperatorFeedback
from src.utils import get_logger, generate_id, now_utc
from .base import BaseAgent, AgentMessage, MessagePriority

logger = get_logger(__name__)


class LabelQuality(str, Enum):
    """Quality of a feedback label."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class LabeledSample:
    """A labeled sample for training."""
    id: str
    anomaly_id: str
    features: Dict[str, float]
    label: bool  # True = real anomaly, False = false positive
    confidence: float
    labeler: str
    timestamp: datetime = field(default_factory=now_utc)


@dataclass
class RetrainingRequest:
    """Request to retrain a model."""
    id: str
    model_id: str
    trigger: str
    samples_count: int
    timestamp: datetime = field(default_factory=now_utc)
    status: str = "pending"


class FeedbackAgent(BaseAgent):
    """
    Feedback Agent that processes human feedback and retrains models.
    
    Responsibilities:
    - Collect feedback from operators on anomalies and recommendations
    - Validate and quality-check feedback
    - Trigger model retraining when appropriate
    - Track feedback statistics and model performance
    """
    
    def __init__(
        self,
        min_samples_for_retrain: int = 100,
        feedback_window_days: int = 7,
        auto_retrain: bool = True,
        **kwargs
    ):
        """
        Initialize Feedback Agent.
        
        Args:
            min_samples_for_retrain: Minimum samples before retraining
            feedback_window_days: Days to keep feedback samples
            auto_retrain: Whether to auto-trigger retraining
        """
        super().__init__(agent_type=AgentType.FEEDBACK, **kwargs)
        
        self.min_samples_for_retrain = min_samples_for_retrain
        self.feedback_window_days = feedback_window_days
        self.auto_retrain = auto_retrain
        
        # Labeled samples by metric
        self._samples: Dict[str, List[LabeledSample]] = defaultdict(list)
        
        # Feedback history
        self._feedback_history: List[OperatorFeedback] = []
        
        # Retraining requests
        self._retrain_requests: Dict[str, RetrainingRequest] = {}
        
        # Model performance tracking
        self._model_metrics: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {
                "true_positives": 0,
                "false_positives": 0,
                "false_negatives": 0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0
            }
        )
        
        # Callbacks for model retraining
        self._retrain_callbacks: Dict[str, Callable] = {}
        
        # Register message handlers
        self.register_handler("recommendations_ready", self._handle_recommendations)
        self.register_handler("anomaly_feedback", self._handle_anomaly_feedback)
    
    async def process(self, data: Dict[str, Any]) -> Optional[AgentDecision]:
        """
        Process feedback request.
        
        Args:
            data: May contain:
                - action: 'submit_feedback', 'get_samples', 'trigger_retrain', 'get_stats'
                - feedback: Feedback data
        """
        action = data.get("action", "submit_feedback")
        
        if action == "submit_feedback":
            feedback_data = data.get("feedback", {})
            result = await self._process_feedback(feedback_data)
            return result
        
        elif action == "get_samples":
            metric = data.get("metric")
            limit = data.get("limit", 100)
            samples = self._get_samples(metric, limit)
            return self.record_decision(
                decision=f"Retrieved {len(samples)} samples",
                reasoning="Responding to sample query",
                confidence=1.0,
                input_data=data
            )
        
        elif action == "trigger_retrain":
            model_id = data.get("model_id")
            return await self._trigger_retrain(model_id, "manual")
        
        elif action == "cleanup":
            cleaned = self._cleanup_old_samples()
            return self.record_decision(
                decision=f"Cleaned up {cleaned} old samples",
                reasoning="Periodic maintenance",
                confidence=1.0,
                input_data=data
            )
        
        return None
    
    async def _handle_recommendations(self, message: AgentMessage) -> None:
        """Handle recommendations notification for feedback tracking."""
        incident_id = message.payload.get("incident_id")
        recommendations = message.payload.get("recommendations", [])
        
        # Store for later feedback collection
        self.memory.remember(
            f"pending_feedback_{incident_id}",
            {
                "recommendations": recommendations,
                "timestamp": now_utc().isoformat(),
                "status": "pending"
            }
        )
        
        logger.info(
            "Tracking recommendations for feedback",
            incident_id=incident_id,
            recommendation_count=len(recommendations)
        )
    
    async def _handle_anomaly_feedback(self, message: AgentMessage) -> None:
        """Handle anomaly feedback submission."""
        feedback_data = message.payload.get("feedback", {})
        await self._process_feedback(feedback_data)
    
    async def _process_feedback(
        self,
        feedback_data: Dict[str, Any]
    ) -> AgentDecision:
        """Process submitted feedback."""
        # Create OperatorFeedback object
        feedback = OperatorFeedback(
            id=generate_id("fb"),
            anomaly_id=feedback_data.get("anomaly_id"),
            feedback_type=FeedbackType(feedback_data.get("type", "true_positive")),
            comment=feedback_data.get("comment", ""),
            timestamp=now_utc()
        )
        
        self._feedback_history.append(feedback)
        
        # Validate feedback quality
        quality = self._assess_feedback_quality(feedback)
        
        # Create labeled sample if applicable
        if feedback_data.get("features"):
            sample = LabeledSample(
                id=generate_id("sample"),
                anomaly_id=feedback.anomaly_id,
                features=feedback_data.get("features", {}),
                label=feedback.feedback_type in [
                    FeedbackType.TRUE_POSITIVE,
                    FeedbackType.MISSED_ANOMALY
                ],
                confidence=self._quality_to_confidence(quality),
                labeler=feedback_data.get("labeler", "operator")
            )
            
            metric = feedback_data.get("metric", "unknown")
            self._samples[metric].append(sample)
        
        # Update model metrics
        model_id = feedback_data.get("model_id", "default")
        self._update_model_metrics(model_id, feedback.feedback_type)
        
        # Notify other agents
        self._notify_feedback(feedback)
        
        # Check if retraining is needed
        if self.auto_retrain:
            await self._check_retrain_trigger(model_id)
        
        return self.record_decision(
            decision=f"Processed {feedback.feedback_type.value} feedback",
            reasoning=f"Quality: {quality.value}, Anomaly: {feedback.anomaly_id}",
            confidence=self._quality_to_confidence(quality),
            input_data={"feedback_id": feedback.id}
        )
    
    def _assess_feedback_quality(
        self,
        feedback: OperatorFeedback
    ) -> LabelQuality:
        """Assess the quality of submitted feedback."""
        quality_score = 0.5
        
        # Has comment
        if feedback.comment and len(feedback.comment) > 10:
            quality_score += 0.2
        
        # Recent feedback
        # (implicitly high quality if just submitted)
        quality_score += 0.1
        
        # Consistent with previous feedback on same anomaly
        similar = [
            f for f in self._feedback_history
            if f.anomaly_id == feedback.anomaly_id and f.id != feedback.id
        ]
        if similar:
            if all(f.feedback_type == feedback.feedback_type for f in similar):
                quality_score += 0.2
            else:
                quality_score -= 0.2  # Inconsistent feedback
        
        if quality_score >= 0.8:
            return LabelQuality.HIGH
        elif quality_score >= 0.5:
            return LabelQuality.MEDIUM
        else:
            return LabelQuality.LOW
    
    def _quality_to_confidence(self, quality: LabelQuality) -> float:
        """Convert quality to confidence score."""
        return {
            LabelQuality.HIGH: 0.95,
            LabelQuality.MEDIUM: 0.75,
            LabelQuality.LOW: 0.5
        }[quality]
    
    def _update_model_metrics(
        self,
        model_id: str,
        feedback_type: FeedbackType
    ) -> None:
        """Update model performance metrics based on feedback."""
        metrics = self._model_metrics[model_id]
        
        if feedback_type == FeedbackType.TRUE_POSITIVE:
            metrics["true_positives"] += 1
        elif feedback_type == FeedbackType.FALSE_POSITIVE:
            metrics["false_positives"] += 1
        elif feedback_type == FeedbackType.MISSED_ANOMALY:
            metrics["false_negatives"] += 1
        # FALSE_NEGATIVE handled same as MISSED_ANOMALY
        elif feedback_type == FeedbackType.FALSE_NEGATIVE:
            metrics["false_negatives"] += 1
        
        # Recalculate metrics
        tp = metrics["true_positives"]
        fp = metrics["false_positives"]
        fn = metrics["false_negatives"]
        
        if tp + fp > 0:
            metrics["precision"] = tp / (tp + fp)
        if tp + fn > 0:
            metrics["recall"] = tp / (tp + fn)
        if metrics["precision"] + metrics["recall"] > 0:
            metrics["f1_score"] = (
                2 * metrics["precision"] * metrics["recall"] /
                (metrics["precision"] + metrics["recall"])
            )
    
    def _notify_feedback(self, feedback: OperatorFeedback) -> None:
        """Notify other agents about feedback."""
        # Notify Detection Agent about feedback
        self.send_message(
            recipient=AgentType.DETECTION.value,
            message_type="feedback_received",
            payload={
                "anomaly_id": feedback.anomaly_id,
                "feedback_type": feedback.feedback_type.value,
                "is_true_anomaly": feedback.feedback_type in [
                    FeedbackType.TRUE_POSITIVE,
                    FeedbackType.MISSED_ANOMALY
                ]
            }
        )
        
        # If it was a false positive, notify Correlation Agent
        if feedback.feedback_type == FeedbackType.FALSE_POSITIVE:
            self.send_message(
                recipient=AgentType.CORRELATION.value,
                message_type="false_positive_reported",
                payload={
                    "anomaly_id": feedback.anomaly_id
                }
            )
    
    async def _check_retrain_trigger(self, model_id: str) -> None:
        """Check if retraining should be triggered."""
        # Count recent samples
        total_samples = sum(len(samples) for samples in self._samples.values())
        
        if total_samples < self.min_samples_for_retrain:
            return
        
        # Check model performance degradation
        metrics = self._model_metrics[model_id]
        
        should_retrain = False
        trigger_reason = ""
        
        # Low precision (too many false positives)
        if metrics["precision"] < 0.7 and metrics["true_positives"] + metrics["false_positives"] >= 20:
            should_retrain = True
            trigger_reason = f"Low precision: {metrics['precision']:.2f}"
        
        # Low recall (missing anomalies)
        if metrics["recall"] < 0.8 and metrics["true_positives"] + metrics["false_negatives"] >= 20:
            should_retrain = True
            trigger_reason = f"Low recall: {metrics['recall']:.2f}"
        
        # Enough samples accumulated
        if total_samples >= self.min_samples_for_retrain * 2:
            should_retrain = True
            trigger_reason = f"Sample threshold reached: {total_samples}"
        
        if should_retrain:
            await self._trigger_retrain(model_id, trigger_reason)
    
    async def _trigger_retrain(
        self,
        model_id: str,
        trigger: str
    ) -> AgentDecision:
        """Trigger model retraining."""
        # Count samples
        total_samples = sum(len(samples) for samples in self._samples.values())
        
        request = RetrainingRequest(
            id=generate_id("retrain"),
            model_id=model_id,
            trigger=trigger,
            samples_count=total_samples
        )
        
        self._retrain_requests[request.id] = request
        
        logger.info(
            "Triggered model retraining",
            model_id=model_id,
            trigger=trigger,
            samples=total_samples
        )
        
        # Call retrain callback if registered
        if model_id in self._retrain_callbacks:
            try:
                samples = self._get_training_data()
                await self._retrain_callbacks[model_id](samples)
                request.status = "completed"
            except Exception as e:
                logger.error("Retraining failed", error=str(e))
                request.status = "failed"
        else:
            request.status = "pending_callback"
        
        # Notify Detection Agent
        self.send_message(
            recipient=AgentType.DETECTION.value,
            message_type="retrain_triggered",
            payload={
                "request_id": request.id,
                "model_id": model_id,
                "trigger": trigger
            },
            priority=MessagePriority.HIGH
        )
        
        return self.record_decision(
            decision=f"Triggered retraining for {model_id}",
            reasoning=trigger,
            confidence=0.9,
            input_data={"request_id": request.id}
        )
    
    def register_retrain_callback(
        self,
        model_id: str,
        callback: Callable
    ) -> None:
        """Register a callback for model retraining."""
        self._retrain_callbacks[model_id] = callback
        logger.info("Registered retrain callback", model_id=model_id)
    
    def _get_samples(
        self,
        metric: Optional[str] = None,
        limit: int = 100
    ) -> List[LabeledSample]:
        """Get labeled samples for training."""
        if metric:
            samples = self._samples.get(metric, [])
        else:
            samples = [s for ss in self._samples.values() for s in ss]
        
        # Sort by timestamp and return most recent
        samples = sorted(samples, key=lambda s: s.timestamp, reverse=True)
        return samples[:limit]
    
    def _get_training_data(self) -> Dict[str, Any]:
        """Get training data for model retraining."""
        all_samples = [s for ss in self._samples.values() for s in ss]
        
        # Separate by label
        positive_samples = [s for s in all_samples if s.label]
        negative_samples = [s for s in all_samples if not s.label]
        
        return {
            "positive_samples": [
                {"features": s.features, "confidence": s.confidence}
                for s in positive_samples
            ],
            "negative_samples": [
                {"features": s.features, "confidence": s.confidence}
                for s in negative_samples
            ],
            "total_count": len(all_samples),
            "positive_count": len(positive_samples),
            "negative_count": len(negative_samples)
        }
    
    def _cleanup_old_samples(self) -> int:
        """Remove old feedback samples."""
        cutoff = now_utc() - timedelta(days=self.feedback_window_days)
        
        removed_count = 0
        for metric in list(self._samples.keys()):
            original_count = len(self._samples[metric])
            self._samples[metric] = [
                s for s in self._samples[metric]
                if s.timestamp >= cutoff
            ]
            removed_count += original_count - len(self._samples[metric])
        
        return removed_count
    
    def get_model_performance(
        self,
        model_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get model performance metrics."""
        if model_id:
            return dict(self._model_metrics.get(model_id, {}))
        return {
            model: dict(metrics)
            for model, metrics in self._model_metrics.items()
        }
    
    def get_feedback_summary(self) -> Dict[str, Any]:
        """Get feedback summary statistics."""
        # Count by type
        type_counts = defaultdict(int)
        for fb in self._feedback_history:
            type_counts[fb.feedback_type.value] += 1
        
        # Recent feedback rate
        recent_cutoff = now_utc() - timedelta(hours=24)
        recent_count = sum(
            1 for fb in self._feedback_history
            if fb.timestamp >= recent_cutoff
        )
        
        return {
            "total_feedback": len(self._feedback_history),
            "feedback_by_type": dict(type_counts),
            "recent_24h": recent_count,
            "total_samples": sum(len(s) for s in self._samples.values()),
            "samples_by_metric": {
                metric: len(samples)
                for metric, samples in self._samples.items()
            },
            "pending_retrains": sum(
                1 for r in self._retrain_requests.values()
                if r.status in ["pending", "pending_callback"]
            )
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive feedback agent statistics."""
        return {
            "feedback_summary": self.get_feedback_summary(),
            "model_performance": self.get_model_performance(),
            "retrain_requests": [
                {
                    "id": r.id,
                    "model_id": r.model_id,
                    "status": r.status,
                    "trigger": r.trigger
                }
                for r in self._retrain_requests.values()
            ]
        }
