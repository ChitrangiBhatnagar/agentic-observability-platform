"""
Natural Language Explanation Generator for Anomalies.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional
import re

from src.types import (
    Anomaly, AnomalyType, Severity, ContributingFeature,
    AnomalyExplanation, RecommendedAction
)
from src.utils import get_logger, generate_id, now_utc

logger = get_logger(__name__)


@dataclass
class ExplanationTemplate:
    """Template for generating explanations."""
    pattern: str
    severity_modifiers: Dict[Severity, str]
    anomaly_type_phrases: Dict[str, str]


class NaturalLanguageExplainer:
    """
    Generates human-readable explanations for detected anomalies.
    
    Transforms technical anomaly data into clear, actionable
    descriptions for operators and stakeholders.
    """
    
    def __init__(
        self,
        detail_level: str = "standard",  # brief, standard, detailed
        include_technical: bool = True,
        language: str = "en"
    ):
        """
        Initialize Natural Language Explainer.
        
        Args:
            detail_level: Level of detail in explanations
            include_technical: Include technical details
            language: Language for explanations
        """
        self.detail_level = detail_level
        self.include_technical = include_technical
        self.language = language
        
        self._templates = self._load_templates()
        self._severity_colors = {
            Severity.LOW: "🟢",
            Severity.MEDIUM: "🟡",
            Severity.HIGH: "🟠",
            Severity.CRITICAL: "🔴"
        }
    
    def _load_templates(self) -> Dict[str, ExplanationTemplate]:
        """Load explanation templates."""
        return {
            "anomaly_detected": ExplanationTemplate(
                pattern=(
                    "{severity_indicator} {severity_word} anomaly detected in "
                    "{metric_name} ({service}): {anomaly_description}. "
                    "{value_description}"
                ),
                severity_modifiers={
                    Severity.LOW: "Minor",
                    Severity.MEDIUM: "Moderate",
                    Severity.HIGH: "Significant",
                    Severity.CRITICAL: "Critical"
                },
                anomaly_type_phrases={
                    "spike": "sudden spike detected",
                    "drop": "unexpected drop observed",
                    "trend_change": "trend deviation identified",
                    "level_shift": "baseline shift detected",
                    "variance_change": "unusual variability pattern",
                    "outlier": "outlier value detected",
                    "seasonal": "seasonal pattern deviation"
                }
            ),
            "root_cause": ExplanationTemplate(
                pattern=(
                    "Probable root cause: {cause_description} "
                    "(confidence: {probability:.0%}). "
                    "{evidence_summary}"
                ),
                severity_modifiers={},
                anomaly_type_phrases={}
            ),
            "recommendation": ExplanationTemplate(
                pattern=(
                    "Recommended action: {action_title}. "
                    "{action_description} "
                    "Risk: {risk_level}. Expected impact: {expected_impact}."
                ),
                severity_modifiers={},
                anomaly_type_phrases={}
            )
        }
    
    def explain_anomaly(
        self,
        anomaly: Anomaly,
        contributing_features: Optional[List[ContributingFeature]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> AnomalyExplanation:
        """
        Generate explanation for an anomaly.
        
        Args:
            anomaly: Anomaly to explain
            contributing_features: Feature contributions
            context: Additional context
            
        Returns:
            Comprehensive anomaly explanation
        """
        # Build summary
        summary = self._build_summary(anomaly)
        
        # Build detailed explanation
        details = self._build_details(anomaly, contributing_features, context)
        
        # Get feature contributions
        features = contributing_features or []
        
        # Format for display
        formatted = self._format_explanation(summary, details, anomaly.severity)
        
        return AnomalyExplanation(
            anomaly_id=anomaly.id,
            summary=summary,
            details=details,
            contributing_features=features,
            confidence=anomaly.confidence
        )
    
    def _build_summary(self, anomaly: Anomaly) -> str:
        """Build summary explanation."""
        template = self._templates["anomaly_detected"]
        
        # Get service from labels
        service = anomaly.labels.get("service", anomaly.labels.get("job", "unknown"))
        
        # Get severity word
        severity_word = template.severity_modifiers.get(anomaly.severity, "Unknown")
        
        # Get anomaly type description
        anomaly_type_str = anomaly.anomaly_type.value if hasattr(anomaly.anomaly_type, 'value') else str(anomaly.anomaly_type)
        anomaly_desc = template.anomaly_type_phrases.get(
            anomaly_type_str,
            "anomalous behavior detected"
        )
        
        # Build value description
        if anomaly.expected_value:
            deviation_pct = abs(anomaly.deviation) * 100
            direction = "above" if anomaly.value > anomaly.expected_value else "below"
            value_desc = (
                f"Current value ({anomaly.value:.2f}) is {deviation_pct:.1f}% "
                f"{direction} expected ({anomaly.expected_value:.2f})."
            )
        else:
            value_desc = f"Current value: {anomaly.value:.2f}"
        
        # Format summary
        summary = template.pattern.format(
            severity_indicator=self._severity_colors.get(anomaly.severity, "⚪"),
            severity_word=severity_word,
            metric_name=self._format_metric_name(anomaly.metric_name),
            service=service,
            anomaly_description=anomaly_desc,
            value_description=value_desc
        )
        
        return summary
    
    def _build_details(
        self,
        anomaly: Anomaly,
        contributing_features: Optional[List[ContributingFeature]],
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Build detailed explanation."""
        details = []
        
        # Timing information
        details.append(f"**Detected at:** {anomaly.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        
        # Metric information
        details.append(f"**Metric:** {anomaly.metric_name}")
        
        # Labels
        if anomaly.labels:
            label_str = ", ".join(f"{k}={v}" for k, v in anomaly.labels.items())
            details.append(f"**Labels:** {label_str}")
        
        # Scores
        details.append(f"**Anomaly Score:** {anomaly.ensemble_score:.2f}")
        details.append(f"**Confidence:** {anomaly.confidence:.0%}")
        
        # Contributing features
        if contributing_features and self.detail_level in ["standard", "detailed"]:
            details.append("\n**Contributing Factors:**")
            for i, feature in enumerate(contributing_features[:5], 1):
                contribution_pct = abs(feature.contribution) * 100
                direction = "↑" if feature.contribution > 0 else "↓"
                details.append(
                    f"  {i}. {feature.name}: {direction} {contribution_pct:.1f}% contribution"
                )
        
        # Technical details
        if self.include_technical and self.detail_level == "detailed":
            details.append("\n**Technical Details:**")
            details.append(f"  - Anomaly Type: {anomaly.anomaly_type}")
            details.append(f"  - Deviation: {anomaly.deviation:.4f}")
            if context:
                for key, value in context.items():
                    details.append(f"  - {key}: {value}")
        
        # Correlated anomalies
        if anomaly.correlated_anomaly_ids:
            count = len(anomaly.correlated_anomaly_ids)
            details.append(f"\n**Correlated with:** {count} other anomalies")
        
        return "\n".join(details)
    
    def _format_explanation(
        self,
        summary: str,
        details: str,
        severity: Severity
    ) -> str:
        """Format complete explanation for display."""
        if self.detail_level == "brief":
            return summary
        elif self.detail_level == "standard":
            return f"{summary}\n\n{details}"
        else:  # detailed
            header = self._get_severity_header(severity)
            return f"{header}\n\n{summary}\n\n{details}"
    
    def _get_severity_header(self, severity: Severity) -> str:
        """Get severity header for detailed explanations."""
        headers = {
            Severity.LOW: "ℹ️ LOW SEVERITY ANOMALY",
            Severity.MEDIUM: "⚠️ MEDIUM SEVERITY ANOMALY",
            Severity.HIGH: "🚨 HIGH SEVERITY ANOMALY",
            Severity.CRITICAL: "🔥 CRITICAL SEVERITY ANOMALY"
        }
        return headers.get(severity, "ANOMALY DETECTED")
    
    def _format_metric_name(self, name: str) -> str:
        """Format metric name for readability."""
        # Replace underscores and dots with spaces
        readable = name.replace("_", " ").replace(".", " ")
        
        # Capitalize words
        words = readable.split()
        readable = " ".join(w.capitalize() for w in words)
        
        return readable
    
    def explain_root_cause(
        self,
        cause_data: Dict[str, Any]
    ) -> str:
        """Generate explanation for root cause."""
        template = self._templates["root_cause"]
        
        evidence = cause_data.get("evidence", [])
        evidence_summary = ". ".join(evidence[:3]) if evidence else "Based on pattern analysis."
        
        return template.pattern.format(
            cause_description=cause_data.get("description", "Unknown cause"),
            probability=cause_data.get("probability", 0),
            evidence_summary=evidence_summary
        )
    
    def explain_recommendation(
        self,
        recommendation: RecommendedAction
    ) -> str:
        """Generate explanation for recommendation."""
        template = self._templates["recommendation"]
        
        return template.pattern.format(
            action_title=recommendation.title,
            action_description=recommendation.description,
            risk_level=recommendation.risk_level,
            expected_impact=recommendation.expected_impact
        )
    
    def generate_incident_summary(
        self,
        anomalies: List[Anomaly],
        root_causes: List[Dict[str, Any]],
        recommendations: List[RecommendedAction]
    ) -> str:
        """
        Generate comprehensive incident summary.
        
        Args:
            anomalies: Related anomalies
            root_causes: Identified root causes
            recommendations: Proposed actions
            
        Returns:
            Formatted incident summary
        """
        lines = ["# Incident Summary\n"]
        
        # Overview
        max_severity = max(a.severity for a in anomalies) if anomalies else Severity.LOW
        lines.append(f"**Status:** {self._severity_colors.get(max_severity, '⚪')} {max_severity.value.upper()}")
        lines.append(f"**Anomalies:** {len(anomalies)}")
        lines.append(f"**Time:** {now_utc().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        lines.append("")
        
        # Affected services
        services = set()
        for a in anomalies:
            service = a.labels.get("service", a.labels.get("job"))
            if service:
                services.add(service)
        
        if services:
            lines.append(f"**Affected Services:** {', '.join(services)}")
            lines.append("")
        
        # Anomalies section
        lines.append("## Detected Anomalies\n")
        for i, anomaly in enumerate(anomalies[:5], 1):
            summary = self._build_summary(anomaly)
            lines.append(f"{i}. {summary}")
        
        if len(anomalies) > 5:
            lines.append(f"\n*...and {len(anomalies) - 5} more*")
        lines.append("")
        
        # Root causes section
        if root_causes:
            lines.append("## Probable Root Causes\n")
            for i, cause in enumerate(root_causes[:3], 1):
                explanation = self.explain_root_cause(cause)
                lines.append(f"{i}. {explanation}")
            lines.append("")
        
        # Recommendations section
        if recommendations:
            lines.append("## Recommended Actions\n")
            for i, rec in enumerate(recommendations[:3], 1):
                explanation = self.explain_recommendation(rec)
                lines.append(f"{i}. {explanation}")
            lines.append("")
        
        # Footer
        lines.append("---")
        lines.append("*Generated by Agentic Observability Platform*")
        
        return "\n".join(lines)
    
    def generate_alert_message(
        self,
        anomaly: Anomaly,
        channel: str = "slack"  # slack, email, pagerduty
    ) -> Dict[str, Any]:
        """
        Generate alert message for different channels.
        
        Args:
            anomaly: Anomaly to alert about
            channel: Target channel
            
        Returns:
            Formatted alert for the channel
        """
        summary = self._build_summary(anomaly)
        
        if channel == "slack":
            return self._format_slack_alert(anomaly, summary)
        elif channel == "email":
            return self._format_email_alert(anomaly, summary)
        elif channel == "pagerduty":
            return self._format_pagerduty_alert(anomaly, summary)
        else:
            return {"message": summary}
    
    def _format_slack_alert(
        self,
        anomaly: Anomaly,
        summary: str
    ) -> Dict[str, Any]:
        """Format alert for Slack."""
        color_map = {
            Severity.LOW: "#36a64f",
            Severity.MEDIUM: "#ffcc00",
            Severity.HIGH: "#ff9900",
            Severity.CRITICAL: "#ff0000"
        }
        
        service = anomaly.labels.get("service", "unknown")
        
        return {
            "attachments": [
                {
                    "color": color_map.get(anomaly.severity, "#808080"),
                    "title": f"Anomaly Detected: {anomaly.metric_name}",
                    "text": summary,
                    "fields": [
                        {"title": "Service", "value": service, "short": True},
                        {"title": "Severity", "value": anomaly.severity.value, "short": True},
                        {"title": "Score", "value": f"{anomaly.ensemble_score:.2f}", "short": True},
                        {"title": "Confidence", "value": f"{anomaly.confidence:.0%}", "short": True}
                    ],
                    "footer": "Agentic Observability Platform",
                    "ts": int(anomaly.timestamp.timestamp())
                }
            ]
        }
    
    def _format_email_alert(
        self,
        anomaly: Anomaly,
        summary: str
    ) -> Dict[str, Any]:
        """Format alert for email."""
        service = anomaly.labels.get("service", "unknown")
        details = self._build_details(anomaly, None, None)
        
        return {
            "subject": f"[{anomaly.severity.value.upper()}] Anomaly: {anomaly.metric_name} ({service})",
            "body_text": f"{summary}\n\n{details}",
            "body_html": f"""
            <h2>{self._severity_colors.get(anomaly.severity)} Anomaly Detected</h2>
            <p>{summary}</p>
            <hr>
            <pre>{details}</pre>
            <hr>
            <p><em>Agentic Observability Platform</em></p>
            """
        }
    
    def _format_pagerduty_alert(
        self,
        anomaly: Anomaly,
        summary: str
    ) -> Dict[str, Any]:
        """Format alert for PagerDuty."""
        severity_map = {
            Severity.LOW: "info",
            Severity.MEDIUM: "warning",
            Severity.HIGH: "error",
            Severity.CRITICAL: "critical"
        }
        
        return {
            "routing_key": "",  # To be filled by caller
            "event_action": "trigger",
            "dedup_key": anomaly.id,
            "payload": {
                "summary": summary[:1024],  # PD has 1024 char limit
                "source": anomaly.labels.get("service", "observability-platform"),
                "severity": severity_map.get(anomaly.severity, "info"),
                "timestamp": anomaly.timestamp.isoformat(),
                "custom_details": {
                    "metric": anomaly.metric_name,
                    "value": anomaly.value,
                    "score": anomaly.ensemble_score,
                    "type": str(anomaly.anomaly_type)
                }
            }
        }
