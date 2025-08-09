"""
Alert management system for QECC-aware QML monitoring.

Provides intelligent alerting with configurable rules, escalation,
and integration with external notification systems.
"""

from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import time
import logging
import threading
from collections import deque
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


class AlertSeverity(Enum):
    """Alert severity levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertState(Enum):
    """Alert states."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


@dataclass
class AlertRule:
    """Alert rule configuration."""
    name: str
    condition: str  # e.g., "metric > threshold"
    metric_name: str
    threshold: float
    severity: AlertSeverity
    message_template: str
    enabled: bool = True
    cooldown_seconds: float = 300.0  # 5 minutes
    evaluation_window: int = 5  # Number of samples to evaluate
    condition_type: str = "threshold"  # threshold, anomaly, trend
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Alert:
    """Active alert instance."""
    id: str
    rule_name: str
    severity: AlertSeverity
    message: str
    timestamp: float
    state: AlertState
    metric_name: str
    metric_value: Any
    threshold: Optional[float]
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    acknowledgment_info: Optional[Dict[str, Any]] = None
    resolution_info: Optional[Dict[str, Any]] = None


class AlertManager:
    """
    Comprehensive alert management system.
    
    Manages alert rules, evaluates conditions, sends notifications,
    and tracks alert lifecycle.
    """
    
    def __init__(
        self,
        max_alerts: int = 10000,
        alert_retention_hours: float = 168.0,  # 1 week
        evaluation_interval: float = 30.0,  # seconds
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize alert manager.
        
        Args:
            max_alerts: Maximum number of alerts to keep in memory
            alert_retention_hours: How long to keep resolved alerts
            evaluation_interval: How often to evaluate rules
            logger: Optional logger instance
        """
        self.max_alerts = max_alerts
        self.alert_retention_seconds = alert_retention_hours * 3600
        self.evaluation_interval = evaluation_interval
        self.logger = logger or logging.getLogger(__name__)
        
        # Alert storage
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=max_alerts)
        
        # Rules management
        self.rules: Dict[str, AlertRule] = {}
        self.rule_last_triggered: Dict[str, float] = {}
        
        # Notification channels
        self.notification_channels: Dict[str, Callable[[Alert], None]] = {}
        
        # Metrics integration
        self.metrics_collector = None  # Will be set by external system
        
        # Processing
        self.is_running = False
        self.evaluation_thread: Optional[threading.Thread] = None
        
        # Statistics
        self.start_time = time.time()
        self.alerts_created = 0
        self.alerts_by_severity: Dict[AlertSeverity, int] = {sev: 0 for sev in AlertSeverity}
        self.rules_evaluated = 0
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Initialize default rules
        self._initialize_default_rules()
    
    def _initialize_default_rules(self):
        """Initialize default alert rules for QECC systems."""
        default_rules = [
            AlertRule(
                name="high_logical_error_rate",
                condition="logical_error_rate > 0.01",
                metric_name="logical_error_rate",
                threshold=0.01,
                severity=AlertSeverity.WARNING,
                message_template="Logical error rate exceeded threshold: {value} > {threshold}",
                cooldown_seconds=300.0
            ),
            AlertRule(
                name="critical_logical_error_rate",
                condition="logical_error_rate > 0.05",
                metric_name="logical_error_rate",
                threshold=0.05,
                severity=AlertSeverity.CRITICAL,
                message_template="Critical logical error rate: {value} > {threshold}",
                cooldown_seconds=60.0
            ),
            AlertRule(
                name="low_fidelity",
                condition="fidelity < 0.8",
                metric_name="fidelity",
                threshold=0.8,
                severity=AlertSeverity.WARNING,
                message_template="Circuit fidelity below threshold: {value} < {threshold}",
                cooldown_seconds=300.0
            ),
            AlertRule(
                name="training_stalled",
                condition="loss_improvement < 0.001",
                metric_name="loss",
                threshold=0.001,
                severity=AlertSeverity.WARNING,
                message_template="Training progress stalled: loss improvement < {threshold}",
                condition_type="trend",
                cooldown_seconds=600.0
            ),
            AlertRule(
                name="high_gate_error_rate",
                condition="gate_error_rate > 0.01",
                metric_name="gate_error_rate",
                threshold=0.01,
                severity=AlertSeverity.ERROR,
                message_template="Hardware gate error rate too high: {value} > {threshold}",
                cooldown_seconds=180.0
            ),
            AlertRule(
                name="low_coherence_time",
                condition="coherence_t1 < 20e-6",
                metric_name="coherence_t1",
                threshold=20e-6,
                severity=AlertSeverity.ERROR,
                message_template="Low coherence time detected: T1 = {value:.1f}μs < {threshold:.1f}μs",
                cooldown_seconds=300.0
            )
        ]
        
        for rule in default_rules:
            self.add_rule(rule)
    
    def start(self):
        """Start alert evaluation."""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start evaluation thread
        self.evaluation_thread = threading.Thread(target=self._evaluation_loop)
        self.evaluation_thread.daemon = True
        self.evaluation_thread.start()
        
        self.logger.info("Alert manager started")
    
    def stop(self):
        """Stop alert evaluation."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self.evaluation_thread:
            self.evaluation_thread.join(timeout=5.0)
        
        self.logger.info("Alert manager stopped")
    
    def add_rule(self, rule: AlertRule):
        """Add an alert rule."""
        with self.lock:
            self.rules[rule.name] = rule
            self.rule_last_triggered[rule.name] = 0
            
            self.logger.info(f"Added alert rule: {rule.name}")
    
    def remove_rule(self, rule_name: str) -> bool:
        """Remove an alert rule."""
        with self.lock:
            if rule_name in self.rules:
                del self.rules[rule_name]
                if rule_name in self.rule_last_triggered:
                    del self.rule_last_triggered[rule_name]
                
                self.logger.info(f"Removed alert rule: {rule_name}")
                return True
            return False
    
    def enable_rule(self, rule_name: str) -> bool:
        """Enable an alert rule."""
        with self.lock:
            if rule_name in self.rules:
                self.rules[rule_name].enabled = True
                self.logger.info(f"Enabled alert rule: {rule_name}")
                return True
            return False
    
    def disable_rule(self, rule_name: str) -> bool:
        """Disable an alert rule."""
        with self.lock:
            if rule_name in self.rules:
                self.rules[rule_name].enabled = False
                self.logger.info(f"Disabled alert rule: {rule_name}")
                return True
            return False
    
    def _evaluation_loop(self):
        """Main alert evaluation loop."""
        while self.is_running:
            try:
                self._evaluate_all_rules()
                self._cleanup_old_alerts()
                time.sleep(self.evaluation_interval)
            except Exception as e:
                self.logger.error(f"Error in alert evaluation loop: {e}")
                time.sleep(5.0)
    
    def _evaluate_all_rules(self):
        """Evaluate all active rules."""
        if not self.metrics_collector:
            return
        
        current_time = time.time()
        
        with self.lock:
            for rule_name, rule in self.rules.items():
                if not rule.enabled:
                    continue
                
                # Check cooldown
                last_triggered = self.rule_last_triggered.get(rule_name, 0)
                if current_time - last_triggered < rule.cooldown_seconds:
                    continue
                
                try:
                    if self._evaluate_rule(rule, current_time):
                        self.rule_last_triggered[rule_name] = current_time
                        self.rules_evaluated += 1
                except Exception as e:
                    self.logger.error(f"Error evaluating rule {rule_name}: {e}")
    
    def _evaluate_rule(self, rule: AlertRule, current_time: float) -> bool:
        """
        Evaluate a single rule.
        
        Returns:
            True if alert was triggered
        """
        if rule.condition_type == "threshold":
            return self._evaluate_threshold_rule(rule, current_time)
        elif rule.condition_type == "anomaly":
            return self._evaluate_anomaly_rule(rule, current_time)
        elif rule.condition_type == "trend":
            return self._evaluate_trend_rule(rule, current_time)
        else:
            self.logger.warning(f"Unknown condition type: {rule.condition_type}")
            return False
    
    def _evaluate_threshold_rule(self, rule: AlertRule, current_time: float) -> bool:
        """Evaluate threshold-based rule."""
        # Get recent metric values
        metric_history = self.metrics_collector.get_metric_values(
            rule.metric_name, 
            limit=rule.evaluation_window
        )
        
        if not metric_history:
            return False
        
        # Use most recent value or average
        if rule.evaluation_window == 1:
            current_value = metric_history[-1]
        else:
            # Use average of recent values for stability
            try:
                current_value = sum(float(v) for v in metric_history) / len(metric_history)
            except (ValueError, TypeError):
                return False
        
        # Evaluate condition
        triggered = False
        
        if ">" in rule.condition:
            triggered = current_value > rule.threshold
        elif "<" in rule.condition:
            triggered = current_value < rule.threshold
        elif "==" in rule.condition:
            triggered = abs(current_value - rule.threshold) < 1e-10
        elif "!=" in rule.condition:
            triggered = abs(current_value - rule.threshold) >= 1e-10
        
        if triggered:
            self._create_alert(rule, current_value, current_time)
            return True
        
        return False
    
    def _evaluate_anomaly_rule(self, rule: AlertRule, current_time: float) -> bool:
        """Evaluate anomaly detection rule."""
        # Get anomalies from metrics collector
        if hasattr(self.metrics_collector, 'detect_anomalies'):
            anomalies = self.metrics_collector.detect_anomalies(rule.metric_name)
            
            if anomalies:
                # Get the most recent anomaly
                latest_anomaly = max(anomalies, key=lambda m: m.timestamp)
                
                # Check if it's recent enough to trigger
                if current_time - latest_anomaly.timestamp < rule.cooldown_seconds:
                    self._create_alert(rule, latest_anomaly.value, current_time)
                    return True
        
        return False
    
    def _evaluate_trend_rule(self, rule: AlertRule, current_time: float) -> bool:
        """Evaluate trend-based rule."""
        # Get sufficient history for trend analysis
        metric_history = self.metrics_collector.get_metric_values(
            rule.metric_name,
            limit=max(20, rule.evaluation_window * 2)
        )
        
        if len(metric_history) < 10:
            return False
        
        try:
            values = [float(v) for v in metric_history]
        except (ValueError, TypeError):
            return False
        
        # Simple trend detection: compare recent vs older values
        recent_avg = sum(values[-5:]) / 5
        older_avg = sum(values[-15:-10]) / 5
        
        if rule.metric_name == "loss" and "improvement" in rule.condition:
            # Check for loss improvement stagnation
            improvement = (older_avg - recent_avg) / older_avg if older_avg != 0 else 0
            
            if improvement < rule.threshold:
                self._create_alert(rule, improvement, current_time)
                return True
        
        return False
    
    def _create_alert(self, rule: AlertRule, metric_value: Any, timestamp: float):
        """Create a new alert."""
        alert_id = f"{rule.name}_{int(timestamp)}"
        
        # Format message
        message = rule.message_template.format(
            value=metric_value,
            threshold=rule.threshold,
            metric=rule.metric_name
        )
        
        alert = Alert(
            id=alert_id,
            rule_name=rule.name,
            severity=rule.severity,
            message=message,
            timestamp=timestamp,
            state=AlertState.ACTIVE,
            metric_name=rule.metric_name,
            metric_value=metric_value,
            threshold=rule.threshold,
            source="alert_manager",
            metadata=rule.metadata.copy()
        )
        
        with self.lock:
            self.active_alerts[alert_id] = alert
            self.alert_history.append(alert)
            
            # Update statistics
            self.alerts_created += 1
            self.alerts_by_severity[rule.severity] += 1
        
        # Send notifications
        self._send_notifications(alert)
        
        self.logger.warning(f"Alert triggered: {alert.message}")
    
    def _send_notifications(self, alert: Alert):
        """Send alert notifications through all channels."""
        for channel_name, channel_func in self.notification_channels.items():
            try:
                channel_func(alert)
            except Exception as e:
                self.logger.error(f"Failed to send notification via {channel_name}: {e}")
    
    def _cleanup_old_alerts(self):
        """Remove old resolved alerts."""
        current_time = time.time()
        cutoff_time = current_time - self.alert_retention_seconds
        
        # Clean up alert history
        while (self.alert_history and 
               self.alert_history[0].timestamp < cutoff_time and
               self.alert_history[0].state == AlertState.RESOLVED):
            self.alert_history.popleft()
    
    def acknowledge_alert(self, alert_id: str, user: str, note: Optional[str] = None) -> bool:
        """Acknowledge an alert."""
        with self.lock:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.state = AlertState.ACKNOWLEDGED
                alert.acknowledgment_info = {
                    'user': user,
                    'timestamp': time.time(),
                    'note': note
                }
                
                self.logger.info(f"Alert acknowledged by {user}: {alert_id}")
                return True
            return False
    
    def resolve_alert(self, alert_id: str, user: str, note: Optional[str] = None) -> bool:
        """Resolve an alert."""
        with self.lock:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.state = AlertState.RESOLVED
                alert.resolution_info = {
                    'user': user,
                    'timestamp': time.time(),
                    'note': note
                }
                
                # Move from active to history only
                del self.active_alerts[alert_id]
                
                self.logger.info(f"Alert resolved by {user}: {alert_id}")
                return True
            return False
    
    def suppress_alert(self, alert_id: str, user: str, duration_seconds: float = 3600.0) -> bool:
        """Temporarily suppress an alert."""
        with self.lock:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.state = AlertState.SUPPRESSED
                alert.metadata['suppressed_until'] = time.time() + duration_seconds
                alert.metadata['suppressed_by'] = user
                
                self.logger.info(f"Alert suppressed by {user} for {duration_seconds}s: {alert_id}")
                return True
            return False
    
    def add_notification_channel(self, name: str, channel_func: Callable[[Alert], None]):
        """Add a notification channel."""
        self.notification_channels[name] = channel_func
        self.logger.info(f"Added notification channel: {name}")
    
    def remove_notification_channel(self, name: str) -> bool:
        """Remove a notification channel."""
        if name in self.notification_channels:
            del self.notification_channels[name]
            self.logger.info(f"Removed notification channel: {name}")
            return True
        return False
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get all active alerts."""
        with self.lock:
            return [
                {
                    'id': alert.id,
                    'rule_name': alert.rule_name,
                    'severity': alert.severity.value,
                    'message': alert.message,
                    'timestamp': alert.timestamp,
                    'state': alert.state.value,
                    'metric_name': alert.metric_name,
                    'metric_value': alert.metric_value,
                    'threshold': alert.threshold,
                    'age_seconds': time.time() - alert.timestamp,
                    'metadata': alert.metadata
                }
                for alert in self.active_alerts.values()
            ]
    
    def get_alert_history(
        self,
        count: Optional[int] = None,
        severity: Optional[AlertSeverity] = None
    ) -> List[Dict[str, Any]]:
        """Get alert history."""
        with self.lock:
            alerts = list(self.alert_history)
            
            # Filter by severity
            if severity:
                alerts = [a for a in alerts if a.severity == severity]
            
            # Sort by timestamp (most recent first)
            alerts.sort(key=lambda a: a.timestamp, reverse=True)
            
            # Limit count
            if count:
                alerts = alerts[:count]
            
            return [
                {
                    'id': alert.id,
                    'rule_name': alert.rule_name,
                    'severity': alert.severity.value,
                    'message': alert.message,
                    'timestamp': alert.timestamp,
                    'state': alert.state.value,
                    'metric_name': alert.metric_name,
                    'metric_value': alert.metric_value,
                    'resolution_info': alert.resolution_info,
                    'acknowledgment_info': alert.acknowledgment_info
                }
                for alert in alerts
            ]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get alert manager statistics."""
        with self.lock:
            uptime = time.time() - self.start_time
            
            return {
                'is_running': self.is_running,
                'uptime_seconds': uptime,
                'total_rules': len(self.rules),
                'enabled_rules': sum(1 for r in self.rules.values() if r.enabled),
                'active_alerts': len(self.active_alerts),
                'total_alerts_created': self.alerts_created,
                'alerts_per_hour': (self.alerts_created / uptime * 3600) if uptime > 0 else 0,
                'alerts_by_severity': {k.value: v for k, v in self.alerts_by_severity.items()},
                'rules_evaluated': self.rules_evaluated,
                'notification_channels': list(self.notification_channels.keys())
            }
    
    def set_metrics_collector(self, metrics_collector):
        """Set the metrics collector for rule evaluation."""
        self.metrics_collector = metrics_collector
        self.logger.info("Metrics collector connected to alert manager")


# Notification channel implementations

def email_notification_channel(
    smtp_server: str,
    smtp_port: int,
    username: str,
    password: str,
    recipients: List[str],
    sender_email: Optional[str] = None
) -> Callable[[Alert], None]:
    """
    Create email notification channel.
    
    Args:
        smtp_server: SMTP server address
        smtp_port: SMTP server port
        username: SMTP username
        password: SMTP password
        recipients: List of recipient email addresses
        sender_email: Sender email (defaults to username)
        
    Returns:
        Email notification function
    """
    if sender_email is None:
        sender_email = username
    
    def send_email_alert(alert: Alert):
        """Send alert via email."""
        try:
            msg = MIMEMultipart()
            msg['From'] = sender_email
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = f"[{alert.severity.value.upper()}] QECC Alert: {alert.rule_name}"
            
            body = f"""
            Alert Details:
            
            Rule: {alert.rule_name}
            Severity: {alert.severity.value}
            Message: {alert.message}
            Metric: {alert.metric_name}
            Value: {alert.metric_value}
            Threshold: {alert.threshold}
            Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(alert.timestamp))}
            
            Alert ID: {alert.id}
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(username, password)
                server.sendmail(sender_email, recipients, msg.as_string())
                
        except Exception as e:
            logging.getLogger(__name__).error(f"Failed to send email alert: {e}")
    
    return send_email_alert


def slack_notification_channel(webhook_url: str) -> Callable[[Alert], None]:
    """
    Create Slack notification channel.
    
    Args:
        webhook_url: Slack webhook URL
        
    Returns:
        Slack notification function
    """
    def send_slack_alert(alert: Alert):
        """Send alert to Slack."""
        try:
            import requests
            
            severity_colors = {
                AlertSeverity.DEBUG: "#36a64f",
                AlertSeverity.INFO: "#2196F3",
                AlertSeverity.WARNING: "#ff9800",
                AlertSeverity.ERROR: "#f44336",
                AlertSeverity.CRITICAL: "#9c27b0"
            }
            
            payload = {
                "attachments": [
                    {
                        "color": severity_colors.get(alert.severity, "#36a64f"),
                        "title": f"QECC Alert: {alert.rule_name}",
                        "text": alert.message,
                        "fields": [
                            {
                                "title": "Severity",
                                "value": alert.severity.value.upper(),
                                "short": True
                            },
                            {
                                "title": "Metric",
                                "value": f"{alert.metric_name}: {alert.metric_value}",
                                "short": True
                            },
                            {
                                "title": "Threshold",
                                "value": str(alert.threshold),
                                "short": True
                            },
                            {
                                "title": "Alert ID",
                                "value": alert.id,
                                "short": True
                            }
                        ],
                        "ts": int(alert.timestamp)
                    }
                ]
            }
            
            response = requests.post(webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            
        except Exception as e:
            logging.getLogger(__name__).error(f"Failed to send Slack alert: {e}")
    
    return send_slack_alert