"""
Real-time monitoring system for QECC-aware QML training.

Provides real-time event monitoring, alerting, and system health tracking.
"""

import threading
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from queue import Queue, Empty
from collections import deque
import json


class EventType(Enum):
    """Types of monitoring events."""
    TRAINING_START = "training_start"
    TRAINING_END = "training_end"
    EPOCH_COMPLETE = "epoch_complete"
    QECC_ADAPTATION = "qecc_adaptation"
    HARDWARE_ALERT = "hardware_alert"
    PERFORMANCE_ALERT = "performance_alert"
    SYSTEM_ERROR = "system_error"
    METRIC_THRESHOLD = "metric_threshold"
    CUSTOM = "custom"


@dataclass
class MonitoringEvent:
    """Single monitoring event."""
    event_type: EventType
    timestamp: float
    source: str
    message: str
    data: Dict[str, Any] = field(default_factory=dict)
    severity: str = "info"  # debug, info, warning, error, critical
    tags: Dict[str, str] = field(default_factory=dict)


class RealTimeMonitor:
    """
    Real-time monitoring system for QECC-aware QML.
    
    Monitors system events, triggers alerts, and provides real-time
    feedback on training progress and system health.
    """
    
    def __init__(
        self,
        max_events: int = 10000,
        event_retention_hours: float = 24.0,
        processing_interval: float = 0.1,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize real-time monitor.
        
        Args:
            max_events: Maximum number of events to keep in memory
            event_retention_hours: How long to keep events
            processing_interval: How often to process events (seconds)
            logger: Optional logger instance
        """
        self.max_events = max_events
        self.event_retention_seconds = event_retention_hours * 3600
        self.processing_interval = processing_interval
        self.logger = logger or logging.getLogger(__name__)
        
        # Event storage
        self.events: deque = deque(maxlen=max_events)
        self.event_queue: Queue = Queue()
        
        # Event handlers
        self.event_handlers: Dict[EventType, List[Callable]] = {}
        for event_type in EventType:
            self.event_handlers[event_type] = []
        
        # Processing
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.processor_thread: Optional[threading.Thread] = None
        
        # Statistics
        self.start_time = time.time()
        self.event_counts: Dict[EventType, int] = {event_type: 0 for event_type in EventType}
        self.events_processed = 0
        
        # Metrics tracking
        self.metric_watchers: List[Dict[str, Any]] = []
        self.active_alerts: Dict[str, MonitoringEvent] = {}
        
        # Thread safety
        self.lock = threading.RLock()
    
    def start(self):
        """Start real-time monitoring."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        
        # Start event processor thread
        self.processor_thread = threading.Thread(target=self._process_events_loop)
        self.processor_thread.daemon = True
        self.processor_thread.start()
        
        # Start monitoring thread (for periodic tasks)
        self.monitor_thread = threading.Thread(target=self._monitoring_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        self.logger.info("Real-time monitoring started")
        
        # Send start event
        self.emit_event(
            EventType.CUSTOM,
            source="monitor",
            message="Real-time monitoring started",
            severity="info"
        )
    
    def stop(self):
        """Stop real-time monitoring."""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        
        # Wait for threads to complete
        if self.processor_thread:
            self.processor_thread.join(timeout=5.0)
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        
        self.logger.info("Real-time monitoring stopped")
        
        # Send stop event
        self.emit_event(
            EventType.CUSTOM,
            source="monitor",
            message="Real-time monitoring stopped",
            severity="info"
        )
    
    def emit_event(
        self,
        event_type: EventType,
        source: str,
        message: str,
        data: Optional[Dict[str, Any]] = None,
        severity: str = "info",
        tags: Optional[Dict[str, str]] = None
    ):
        """
        Emit a monitoring event.
        
        Args:
            event_type: Type of event
            source: Source component
            message: Event message
            data: Additional event data
            severity: Event severity level
            tags: Event tags
        """
        event = MonitoringEvent(
            event_type=event_type,
            timestamp=time.time(),
            source=source,
            message=message,
            data=data or {},
            severity=severity,
            tags=tags or {}
        )
        
        # Add to queue for processing
        try:
            self.event_queue.put_nowait(event)
        except:
            self.logger.warning("Event queue full, dropping event")
    
    def _process_events_loop(self):
        """Main event processing loop."""
        while self.is_monitoring:
            try:
                # Process events from queue
                while not self.event_queue.empty():
                    try:
                        event = self.event_queue.get(timeout=0.1)
                        self._process_single_event(event)
                        self.event_queue.task_done()
                    except Empty:
                        break
                    except Exception as e:
                        self.logger.error(f"Error processing event: {e}")
                
                time.sleep(self.processing_interval)
                
            except Exception as e:
                self.logger.error(f"Error in event processing loop: {e}")
                time.sleep(1.0)
    
    def _process_single_event(self, event: MonitoringEvent):
        """Process a single event."""
        with self.lock:
            # Store event
            self.events.append(event)
            
            # Update statistics
            self.event_counts[event.event_type] += 1
            self.events_processed += 1
            
            # Handle alerts
            if event.severity in ["warning", "error", "critical"]:
                self._handle_alert(event)
            
            # Call event handlers
            for handler in self.event_handlers.get(event.event_type, []):
                try:
                    handler(event)
                except Exception as e:
                    self.logger.error(f"Event handler error: {e}")
            
            # Call wildcard handlers
            for handler in self.event_handlers.get("*", []):
                try:
                    handler(event)
                except Exception as e:
                    self.logger.error(f"Wildcard event handler error: {e}")
            
            # Log event
            log_level = {
                "debug": logging.DEBUG,
                "info": logging.INFO,
                "warning": logging.WARNING,
                "error": logging.ERROR,
                "critical": logging.CRITICAL
            }.get(event.severity, logging.INFO)
            
            self.logger.log(
                log_level,
                f"[{event.source}] {event.event_type.value}: {event.message}"
            )
    
    def _monitoring_loop(self):
        """Main monitoring loop for periodic tasks."""
        while self.is_monitoring:
            try:
                # Clean old events
                self._cleanup_old_events()
                
                # Check metric watchers
                self._check_metric_watchers()
                
                # Resolve old alerts
                self._resolve_old_alerts()
                
                time.sleep(60.0)  # Run every minute
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
    
    def _cleanup_old_events(self):
        """Remove old events to free memory."""
        if not self.events:
            return
        
        cutoff_time = time.time() - self.event_retention_seconds
        
        # Remove old events (from left side of deque)
        while self.events and self.events[0].timestamp < cutoff_time:
            self.events.popleft()
    
    def _check_metric_watchers(self):
        """Check all registered metric watchers."""
        # This would integrate with MetricsCollector to check thresholds
        # For now, just a placeholder
        pass
    
    def _resolve_old_alerts(self):
        """Resolve alerts that are no longer active."""
        current_time = time.time()
        resolved_alerts = []
        
        for alert_id, alert in self.active_alerts.items():
            # Auto-resolve alerts older than 1 hour
            if current_time - alert.timestamp > 3600:
                resolved_alerts.append(alert_id)
        
        for alert_id in resolved_alerts:
            del self.active_alerts[alert_id]
            self.emit_event(
                EventType.CUSTOM,
                source="monitor",
                message=f"Alert auto-resolved: {alert_id}",
                severity="info"
            )
    
    def _handle_alert(self, event: MonitoringEvent):
        """Handle alert events."""
        alert_id = f"{event.source}_{event.event_type.value}_{int(event.timestamp)}"
        
        if event.severity in ["error", "critical"]:
            self.active_alerts[alert_id] = event
        
        # Could integrate with external alerting systems here
        # (email, Slack, PagerDuty, etc.)
    
    def add_event_handler(self, event_type: EventType, handler: Callable[[MonitoringEvent], None]):
        """
        Add event handler for specific event type.
        
        Args:
            event_type: Type of event to handle
            handler: Callable that takes MonitoringEvent
        """
        self.event_handlers[event_type].append(handler)
        self.logger.info(f"Added event handler for {event_type.value}")
    
    def remove_event_handler(self, event_type: EventType, handler: Callable):
        """Remove event handler."""
        if handler in self.event_handlers[event_type]:
            self.event_handlers[event_type].remove(handler)
            self.logger.info(f"Removed event handler for {event_type.value}")
    
    def add_metric_watcher(
        self,
        metric_name: str,
        threshold: float,
        condition: str = "greater_than",
        alert_message: Optional[str] = None
    ):
        """
        Add metric threshold watcher.
        
        Args:
            metric_name: Name of metric to watch
            threshold: Threshold value
            condition: Condition ("greater_than", "less_than", "equal", "not_equal")
            alert_message: Custom alert message
        """
        watcher = {
            'metric_name': metric_name,
            'threshold': threshold,
            'condition': condition,
            'alert_message': alert_message or f"{metric_name} threshold violation",
            'last_triggered': 0
        }
        
        self.metric_watchers.append(watcher)
        self.logger.info(f"Added metric watcher: {metric_name} {condition} {threshold}")
    
    def get_recent_events(
        self,
        count: Optional[int] = None,
        event_type: Optional[EventType] = None,
        severity: Optional[str] = None,
        source: Optional[str] = None,
        since: Optional[float] = None
    ) -> List[MonitoringEvent]:
        """
        Get recent events with optional filtering.
        
        Args:
            count: Maximum number of events to return
            event_type: Filter by event type
            severity: Filter by severity
            source: Filter by source
            since: Only events after this timestamp
            
        Returns:
            List of matching events
        """
        with self.lock:
            events = list(self.events)
            
            # Apply filters
            if event_type is not None:
                events = [e for e in events if e.event_type == event_type]
            
            if severity is not None:
                events = [e for e in events if e.severity == severity]
            
            if source is not None:
                events = [e for e in events if e.source == source]
            
            if since is not None:
                events = [e for e in events if e.timestamp >= since]
            
            # Sort by timestamp (most recent first)
            events.sort(key=lambda e: e.timestamp, reverse=True)
            
            # Limit count
            if count is not None:
                events = events[:count]
            
            return events
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get list of active alerts."""
        with self.lock:
            return [
                {
                    'id': alert_id,
                    'timestamp': alert.timestamp,
                    'event_type': alert.event_type.value,
                    'source': alert.source,
                    'message': alert.message,
                    'severity': alert.severity,
                    'data': alert.data,
                    'age_seconds': time.time() - alert.timestamp
                }
                for alert_id, alert in self.active_alerts.items()
            ]
    
    def resolve_alert(self, alert_id: str) -> bool:
        """
        Manually resolve an alert.
        
        Args:
            alert_id: ID of alert to resolve
            
        Returns:
            True if alert was resolved
        """
        with self.lock:
            if alert_id in self.active_alerts:
                del self.active_alerts[alert_id]
                
                self.emit_event(
                    EventType.CUSTOM,
                    source="monitor",
                    message=f"Alert manually resolved: {alert_id}",
                    severity="info"
                )
                
                return True
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get monitoring statistics."""
        with self.lock:
            uptime = time.time() - self.start_time
            
            return {
                'is_monitoring': self.is_monitoring,
                'uptime_seconds': uptime,
                'events_processed': self.events_processed,
                'events_per_second': self.events_processed / uptime if uptime > 0 else 0,
                'event_counts_by_type': {k.value: v for k, v in self.event_counts.items()},
                'active_alerts_count': len(self.active_alerts),
                'total_events_in_memory': len(self.events),
                'metric_watchers_count': len(self.metric_watchers),
                'event_handlers_count': sum(len(handlers) for handlers in self.event_handlers.values())
            }
    
    def export_events(
        self,
        format: str = "json",
        count: Optional[int] = None,
        **filter_kwargs
    ) -> str:
        """
        Export events data.
        
        Args:
            format: Export format ("json", "csv")
            count: Maximum number of events to export
            **filter_kwargs: Filtering arguments for get_recent_events()
            
        Returns:
            Exported data as string
        """
        events = self.get_recent_events(count=count, **filter_kwargs)
        
        if format.lower() == "json":
            event_data = []
            for event in events:
                event_data.append({
                    'timestamp': event.timestamp,
                    'event_type': event.event_type.value,
                    'source': event.source,
                    'message': event.message,
                    'severity': event.severity,
                    'data': event.data,
                    'tags': event.tags
                })
            return json.dumps(event_data, indent=2, default=str)
        
        elif format.lower() == "csv":
            import csv
            import io
            
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Header
            writer.writerow(['timestamp', 'event_type', 'source', 'message', 'severity'])
            
            # Data
            for event in events:
                writer.writerow([
                    event.timestamp,
                    event.event_type.value,
                    event.source,
                    event.message,
                    event.severity
                ])
            
            return output.getvalue()
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def clear_events(self):
        """Clear all stored events."""
        with self.lock:
            self.events.clear()
            self.active_alerts.clear()
            self.event_counts = {event_type: 0 for event_type in EventType}
            self.events_processed = 0
            self.start_time = time.time()
            
            self.logger.info("Cleared all monitoring events")
            
            self.emit_event(
                EventType.CUSTOM,
                source="monitor",
                message="Event history cleared",
                severity="info"
            )