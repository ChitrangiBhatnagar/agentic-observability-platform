"""
Timeline Reconstruction for Incident Analysis.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

from src.types import Anomaly, Severity
from src.utils import get_logger, generate_id, now_utc

logger = get_logger(__name__)


@dataclass
class TimelineEvent:
    """An event in the incident timeline."""
    id: str
    timestamp: datetime
    event_type: str  # anomaly, alert, action, resolution
    title: str
    description: str
    severity: Optional[Severity] = None
    source: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TimelinePhase:
    """A phase in the incident timeline."""
    name: str
    start_time: datetime
    end_time: Optional[datetime]
    events: List[TimelineEvent]
    description: str


class TimelineReconstructor:
    """
    Reconstructs incident timelines from anomalies and events.
    
    Creates chronological narratives of incidents for
    post-mortem analysis and understanding.
    """
    
    def __init__(
        self,
        merge_window_seconds: int = 60,
        phase_gap_seconds: int = 300
    ):
        """
        Initialize Timeline Reconstructor.
        
        Args:
            merge_window_seconds: Window for merging close events
            phase_gap_seconds: Gap to start a new phase
        """
        self.merge_window_seconds = merge_window_seconds
        self.phase_gap_seconds = phase_gap_seconds
    
    def reconstruct(
        self,
        anomalies: List[Anomaly],
        additional_events: Optional[List[Dict[str, Any]]] = None
    ) -> List[TimelineEvent]:
        """
        Reconstruct timeline from anomalies and events.
        
        Args:
            anomalies: List of anomalies
            additional_events: Additional events to include
            
        Returns:
            Ordered list of timeline events
        """
        events = []
        
        # Convert anomalies to events
        for anomaly in anomalies:
            event = self._anomaly_to_event(anomaly)
            events.append(event)
        
        # Add additional events
        if additional_events:
            for event_data in additional_events:
                event = self._dict_to_event(event_data)
                events.append(event)
        
        # Sort by timestamp
        events.sort(key=lambda e: e.timestamp)
        
        # Merge close events
        events = self._merge_close_events(events)
        
        return events
    
    def _anomaly_to_event(self, anomaly: Anomaly) -> TimelineEvent:
        """Convert anomaly to timeline event."""
        service = anomaly.labels.get("service", anomaly.labels.get("job", "unknown"))
        
        # Build description
        anomaly_type = anomaly.anomaly_type.value if hasattr(anomaly.anomaly_type, 'value') else str(anomaly.anomaly_type)
        
        if anomaly.expected_value:
            deviation_pct = abs(anomaly.deviation) * 100
            description = (
                f"{anomaly_type.replace('_', ' ').title()} in {anomaly.metric_name}. "
                f"Value: {anomaly.value:.2f} (expected: {anomaly.expected_value:.2f}, "
                f"deviation: {deviation_pct:.1f}%)"
            )
        else:
            description = (
                f"{anomaly_type.replace('_', ' ').title()} in {anomaly.metric_name}. "
                f"Value: {anomaly.value:.2f}, Score: {anomaly.ensemble_score:.2f}"
            )
        
        return TimelineEvent(
            id=anomaly.id,
            timestamp=anomaly.timestamp,
            event_type="anomaly",
            title=f"Anomaly: {anomaly.metric_name}",
            description=description,
            severity=anomaly.severity,
            source=service,
            metadata={
                "metric": anomaly.metric_name,
                "score": anomaly.ensemble_score,
                "confidence": anomaly.confidence,
                "labels": anomaly.labels
            }
        )
    
    def _dict_to_event(self, data: Dict[str, Any]) -> TimelineEvent:
        """Convert dictionary to timeline event."""
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        elif timestamp is None:
            timestamp = now_utc()
        
        severity = data.get("severity")
        if isinstance(severity, str):
            severity = Severity(severity)
        
        return TimelineEvent(
            id=data.get("id", generate_id("evt")),
            timestamp=timestamp,
            event_type=data.get("event_type", "event"),
            title=data.get("title", "Event"),
            description=data.get("description", ""),
            severity=severity,
            source=data.get("source"),
            metadata=data.get("metadata", {})
        )
    
    def _merge_close_events(
        self,
        events: List[TimelineEvent]
    ) -> List[TimelineEvent]:
        """Merge events that are very close in time."""
        if len(events) <= 1:
            return events
        
        merged = []
        current_group = [events[0]]
        
        for i in range(1, len(events)):
            event = events[i]
            prev_event = current_group[-1]
            
            # Check if within merge window
            time_diff = (event.timestamp - prev_event.timestamp).total_seconds()
            
            if time_diff <= self.merge_window_seconds and \
               event.source == prev_event.source and \
               event.event_type == prev_event.event_type:
                current_group.append(event)
            else:
                # Merge current group and start new
                merged.append(self._merge_event_group(current_group))
                current_group = [event]
        
        # Don't forget last group
        if current_group:
            merged.append(self._merge_event_group(current_group))
        
        return merged
    
    def _merge_event_group(
        self,
        events: List[TimelineEvent]
    ) -> TimelineEvent:
        """Merge a group of events into one."""
        if len(events) == 1:
            return events[0]
        
        # Use first event as base
        base = events[0]
        
        # Combine descriptions
        if len(events) <= 3:
            descriptions = [e.description for e in events]
            combined_desc = " | ".join(descriptions)
        else:
            combined_desc = f"{base.description} (+{len(events)-1} related events)"
        
        # Use highest severity
        severities = [e.severity for e in events if e.severity]
        if severities:
            severity_order = [Severity.LOW, Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL]
            max_severity = max(severities, key=lambda s: severity_order.index(s) if s in severity_order else -1)
        else:
            max_severity = base.severity
        
        return TimelineEvent(
            id=base.id,
            timestamp=base.timestamp,
            event_type=base.event_type,
            title=f"{base.title} ({len(events)} events)",
            description=combined_desc,
            severity=max_severity,
            source=base.source,
            metadata={
                "merged_count": len(events),
                "event_ids": [e.id for e in events]
            }
        )
    
    def identify_phases(
        self,
        events: List[TimelineEvent]
    ) -> List[TimelinePhase]:
        """
        Identify phases in the incident timeline.
        
        Args:
            events: Timeline events
            
        Returns:
            List of timeline phases
        """
        if not events:
            return []
        
        phases = []
        current_phase_events = [events[0]]
        phase_start = events[0].timestamp
        
        for i in range(1, len(events)):
            event = events[i]
            prev_event = events[i - 1]
            
            # Check for phase break
            time_gap = (event.timestamp - prev_event.timestamp).total_seconds()
            
            if time_gap > self.phase_gap_seconds:
                # End current phase, start new
                phases.append(self._create_phase(
                    current_phase_events,
                    phase_start,
                    prev_event.timestamp
                ))
                current_phase_events = [event]
                phase_start = event.timestamp
            else:
                current_phase_events.append(event)
        
        # Last phase
        if current_phase_events:
            phases.append(self._create_phase(
                current_phase_events,
                phase_start,
                current_phase_events[-1].timestamp
            ))
        
        # Name phases
        for i, phase in enumerate(phases):
            phase.name = self._get_phase_name(i, len(phases), phase)
        
        return phases
    
    def _create_phase(
        self,
        events: List[TimelineEvent],
        start_time: datetime,
        end_time: datetime
    ) -> TimelinePhase:
        """Create a phase from events."""
        # Summarize events
        event_types = defaultdict(int)
        for e in events:
            event_types[e.event_type] += 1
        
        type_summary = ", ".join(f"{count} {t}s" for t, count in event_types.items())
        
        return TimelinePhase(
            name="",  # Set later
            start_time=start_time,
            end_time=end_time,
            events=events,
            description=f"{len(events)} events: {type_summary}"
        )
    
    def _get_phase_name(
        self,
        index: int,
        total: int,
        phase: TimelinePhase
    ) -> str:
        """Determine phase name based on position and content."""
        # Check event types to determine phase character
        event_types = [e.event_type for e in phase.events]
        severities = [e.severity for e in phase.events if e.severity]
        
        if index == 0:
            return "Initial Detection"
        elif index == total - 1:
            if "resolution" in event_types:
                return "Resolution"
            else:
                return "Latest Activity"
        else:
            # Middle phases
            if severities:
                max_sev = max(severities, key=lambda s: [
                    Severity.LOW, Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL
                ].index(s) if s in [Severity.LOW, Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL] else -1)
                
                if max_sev == Severity.CRITICAL:
                    return "Escalation"
                elif max_sev == Severity.HIGH:
                    return "Peak Impact"
            
            return f"Phase {index + 1}"
    
    def format_timeline(
        self,
        events: List[TimelineEvent],
        format: str = "text"  # text, markdown, html
    ) -> str:
        """
        Format timeline for display.
        
        Args:
            events: Timeline events
            format: Output format
            
        Returns:
            Formatted timeline string
        """
        if format == "markdown":
            return self._format_markdown(events)
        elif format == "html":
            return self._format_html(events)
        else:
            return self._format_text(events)
    
    def _format_text(self, events: List[TimelineEvent]) -> str:
        """Format timeline as plain text."""
        lines = ["INCIDENT TIMELINE", "=" * 50, ""]
        
        for event in events:
            time_str = event.timestamp.strftime("%H:%M:%S")
            severity_str = f"[{event.severity.value}]" if event.severity else ""
            source_str = f"({event.source})" if event.source else ""
            
            lines.append(f"{time_str} {severity_str} {event.title} {source_str}")
            lines.append(f"         {event.description}")
            lines.append("")
        
        return "\n".join(lines)
    
    def _format_markdown(self, events: List[TimelineEvent]) -> str:
        """Format timeline as Markdown."""
        lines = ["# Incident Timeline\n"]
        
        # Get phases
        phases = self.identify_phases(events)
        
        for phase in phases:
            duration = (phase.end_time - phase.start_time).total_seconds() if phase.end_time else 0
            lines.append(f"## {phase.name}")
            lines.append(f"*{phase.start_time.strftime('%H:%M:%S')} - Duration: {duration:.0f}s*\n")
            
            for event in phase.events:
                severity_emoji = {
                    Severity.LOW: "🟢",
                    Severity.MEDIUM: "🟡",
                    Severity.HIGH: "🟠",
                    Severity.CRITICAL: "🔴"
                }.get(event.severity, "⚪")
                
                time_str = event.timestamp.strftime("%H:%M:%S")
                source_str = f" `{event.source}`" if event.source else ""
                
                lines.append(f"- **{time_str}** {severity_emoji}{source_str}: {event.title}")
                lines.append(f"  - {event.description}")
            
            lines.append("")
        
        return "\n".join(lines)
    
    def _format_html(self, events: List[TimelineEvent]) -> str:
        """Format timeline as HTML."""
        severity_colors = {
            Severity.LOW: "#28a745",
            Severity.MEDIUM: "#ffc107",
            Severity.HIGH: "#fd7e14",
            Severity.CRITICAL: "#dc3545"
        }
        
        html = ['<div class="timeline">']
        
        for event in events:
            color = severity_colors.get(event.severity, "#6c757d")
            time_str = event.timestamp.strftime("%H:%M:%S")
            
            html.append(f'''
            <div class="timeline-event" style="border-left: 3px solid {color}; padding-left: 15px; margin-bottom: 15px;">
                <div class="event-time" style="font-weight: bold; color: #666;">{time_str}</div>
                <div class="event-title" style="font-size: 1.1em; font-weight: bold;">{event.title}</div>
                <div class="event-source" style="color: #888; font-size: 0.9em;">{event.source or ''}</div>
                <div class="event-description">{event.description}</div>
            </div>
            ''')
        
        html.append('</div>')
        
        return "\n".join(html)
    
    def get_summary_stats(
        self,
        events: List[TimelineEvent]
    ) -> Dict[str, Any]:
        """
        Get summary statistics for timeline.
        
        Args:
            events: Timeline events
            
        Returns:
            Summary statistics
        """
        if not events:
            return {}
        
        # Time span
        first_event = events[0]
        last_event = events[-1]
        duration = (last_event.timestamp - first_event.timestamp).total_seconds()
        
        # Severity distribution
        severity_counts = defaultdict(int)
        for e in events:
            if e.severity:
                severity_counts[e.severity.value] += 1
        
        # Event type distribution
        type_counts = defaultdict(int)
        for e in events:
            type_counts[e.event_type] += 1
        
        # Source distribution
        source_counts = defaultdict(int)
        for e in events:
            if e.source:
                source_counts[e.source] += 1
        
        # Find peak (most events in 1-minute window)
        peak_count = 0
        peak_time = None
        for i, event in enumerate(events):
            window_end = event.timestamp + timedelta(minutes=1)
            count = sum(1 for e in events[i:] if e.timestamp <= window_end)
            if count > peak_count:
                peak_count = count
                peak_time = event.timestamp
        
        return {
            "total_events": len(events),
            "duration_seconds": duration,
            "start_time": first_event.timestamp.isoformat(),
            "end_time": last_event.timestamp.isoformat(),
            "severity_distribution": dict(severity_counts),
            "event_types": dict(type_counts),
            "affected_sources": list(source_counts.keys()),
            "source_distribution": dict(source_counts),
            "peak_time": peak_time.isoformat() if peak_time else None,
            "peak_events_per_minute": peak_count
        }
