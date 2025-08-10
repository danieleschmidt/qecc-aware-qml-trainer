"""
Comprehensive system health monitoring for QECC-QML operations.
"""

import time
import threading
import psutil
import numpy as np
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from collections import deque
import json

from ..utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class HealthMetric:
    """Health metric with timestamp and metadata."""
    name: str
    value: float
    timestamp: float
    unit: str = ""
    threshold: Optional[float] = None
    status: str = "normal"  # normal, warning, critical


class HealthMonitor:
    """
    Monitors system health during quantum ML training and execution.
    """
    
    def __init__(self, 
                 monitoring_interval: float = 1.0,
                 history_size: int = 1000,
                 enable_alerts: bool = True):
        self.monitoring_interval = monitoring_interval
        self.history_size = history_size
        self.enable_alerts = enable_alerts
        
        # Health metrics storage
        self.metrics_history = {}
        self.current_metrics = {}
        self.thresholds = self._initialize_thresholds()
        
        # Monitoring state
        self.is_monitoring = False
        self.monitor_thread = None
        
        # Alert callbacks
        self.alert_callbacks = []
        
        # Performance tracking
        self.start_time = time.time()
        self.last_alert_time = {}
        
    def _initialize_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Initialize health metric thresholds."""
        return {
            "cpu_usage": {"warning": 80.0, "critical": 95.0},
            "memory_usage": {"warning": 80.0, "critical": 95.0},
            "gpu_memory": {"warning": 85.0, "critical": 95.0},
            "circuit_fidelity": {"warning": 0.8, "critical": 0.6},
            "error_rate": {"warning": 0.01, "critical": 0.05},
            "queue_size": {"warning": 100, "critical": 1000},
            "response_time": {"warning": 5.0, "critical": 10.0},
            "disk_usage": {"warning": 85.0, "critical": 95.0},
            "temperature": {"warning": 75.0, "critical": 85.0},
        }
        
    def start_monitoring(self):
        """Start health monitoring in background thread."""
        if self.is_monitoring:
            logger.warning("Health monitoring already active")
            return
            
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Health monitoring started")
        
    def stop_monitoring(self):
        """Stop health monitoring."""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        logger.info("Health monitoring stopped")
        
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                self._collect_system_metrics()
                self._check_thresholds()
                time.sleep(self.monitoring_interval)
            except Exception as e:
                logger.error(f"Health monitoring error: {str(e)}")
                time.sleep(self.monitoring_interval)
                
    def _collect_system_metrics(self):
        """Collect system health metrics."""
        current_time = time.time()
        
        # System metrics
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=None)
            self._record_metric("cpu_usage", cpu_percent, current_time, "%")
            
            # Memory usage
            memory = psutil.virtual_memory()
            self._record_metric("memory_usage", memory.percent, current_time, "%")
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            self._record_metric("disk_usage", disk_percent, current_time, "%")
            
            # Network I/O
            net_io = psutil.net_io_counters()
            self._record_metric("network_bytes_sent", net_io.bytes_sent, current_time, "bytes")
            self._record_metric("network_bytes_recv", net_io.bytes_recv, current_time, "bytes")
            
            # Process-specific metrics
            process = psutil.Process()
            self._record_metric("process_memory", process.memory_percent(), current_time, "%")
            self._record_metric("process_cpu", process.cpu_percent(), current_time, "%")
            
            # Quantum-specific metrics would be added here
            # self._collect_quantum_metrics()
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {str(e)}")
            
    def _collect_quantum_metrics(self):
        """Collect quantum computing specific metrics."""
        # This would be implemented to collect metrics from quantum backends
        # For now, simulate some metrics
        current_time = time.time()
        
        # Simulated quantum metrics
        fake_fidelity = 0.95 + 0.05 * np.random.random()
        fake_error_rate = 0.001 + 0.01 * np.random.random()
        
        self._record_metric("circuit_fidelity", fake_fidelity, current_time)
        self._record_metric("error_rate", fake_error_rate, current_time)
        
    def _record_metric(self, name: str, value: float, timestamp: float, unit: str = ""):
        """Record a health metric."""
        
        # Initialize history for new metrics
        if name not in self.metrics_history:
            self.metrics_history[name] = deque(maxlen=self.history_size)
            
        # Create metric object
        threshold = self.thresholds.get(name, {})
        status = self._determine_status(value, threshold)
        
        metric = HealthMetric(
            name=name,
            value=value,
            timestamp=timestamp,
            unit=unit,
            threshold=threshold.get("critical"),
            status=status
        )
        
        # Store metric
        self.metrics_history[name].append(metric)
        self.current_metrics[name] = metric
        
    def _determine_status(self, value: float, thresholds: Dict[str, float]) -> str:
        """Determine status based on thresholds."""
        if not thresholds:
            return "normal"
            
        # Handle different comparison directions
        reverse_metrics = ["circuit_fidelity"]  # Lower values are worse
        
        if any(metric in thresholds for metric in reverse_metrics):
            # For reverse metrics (like fidelity), lower is worse
            if "critical" in thresholds and value < thresholds["critical"]:
                return "critical"
            elif "warning" in thresholds and value < thresholds["warning"]:
                return "warning"
        else:
            # For normal metrics, higher is worse
            if "critical" in thresholds and value > thresholds["critical"]:
                return "critical"
            elif "warning" in thresholds and value > thresholds["warning"]:
                return "warning"
                
        return "normal"
        
    def _check_thresholds(self):
        """Check all metrics against thresholds and trigger alerts."""
        if not self.enable_alerts:
            return
            
        current_time = time.time()
        
        for name, metric in self.current_metrics.items():
            if metric.status in ["warning", "critical"]:
                # Rate limit alerts (don't spam)
                last_alert = self.last_alert_time.get(name, 0)
                if current_time - last_alert > 60:  # 1 minute cooldown
                    self._trigger_alert(metric)
                    self.last_alert_time[name] = current_time
                    
    def _trigger_alert(self, metric: HealthMetric):
        """Trigger alert for problematic metric."""
        alert_data = {
            "metric": metric.name,
            "value": metric.value,
            "status": metric.status,
            "timestamp": metric.timestamp,
            "unit": metric.unit,
            "threshold": metric.threshold
        }
        
        logger.warning(f"Health Alert: {metric.name} = {metric.value}{metric.unit} "
                      f"(Status: {metric.status})")
        
        # Call registered alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert_data)
            except Exception as e:
                logger.error(f"Alert callback error: {str(e)}")
                
    def register_alert_callback(self, callback: Callable[[Dict], None]):
        """Register callback for health alerts."""
        self.alert_callbacks.append(callback)
        
    def get_current_health(self) -> Dict[str, Any]:
        """Get current system health status."""
        health_status = {
            "timestamp": time.time(),
            "uptime": time.time() - self.start_time,
            "overall_status": "normal",
            "metrics": {}
        }
        
        critical_count = 0
        warning_count = 0
        
        for name, metric in self.current_metrics.items():
            health_status["metrics"][name] = {
                "value": metric.value,
                "status": metric.status,
                "unit": metric.unit
            }
            
            if metric.status == "critical":
                critical_count += 1
            elif metric.status == "warning":
                warning_count += 1
                
        # Determine overall status
        if critical_count > 0:
            health_status["overall_status"] = "critical"
        elif warning_count > 0:
            health_status["overall_status"] = "warning"
            
        health_status["alert_counts"] = {
            "critical": critical_count,
            "warning": warning_count,
            "normal": len(self.current_metrics) - critical_count - warning_count
        }
        
        return health_status
        
    def get_metric_history(self, metric_name: str, 
                          duration_seconds: Optional[int] = None) -> List[HealthMetric]:
        """Get historical data for a specific metric."""
        if metric_name not in self.metrics_history:
            return []
            
        metrics = list(self.metrics_history[metric_name])
        
        if duration_seconds:
            cutoff_time = time.time() - duration_seconds
            metrics = [m for m in metrics if m.timestamp >= cutoff_time]
            
        return metrics
        
    def export_health_report(self, filepath: str):
        """Export comprehensive health report to file."""
        report = {
            "report_timestamp": time.time(),
            "monitoring_config": {
                "interval": self.monitoring_interval,
                "history_size": self.history_size,
                "alerts_enabled": self.enable_alerts
            },
            "current_health": self.get_current_health(),
            "thresholds": self.thresholds,
            "metrics_summary": {}
        }
        
        # Add metrics summary
        for name, history in self.metrics_history.items():
            if history:
                values = [m.value for m in history]
                report["metrics_summary"][name] = {
                    "count": len(values),
                    "min": min(values),
                    "max": max(values),
                    "mean": np.mean(values),
                    "std": np.std(values) if len(values) > 1 else 0.0,
                    "latest": history[-1].value
                }
                
        try:
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Health report exported to {filepath}")
        except Exception as e:
            logger.error(f"Failed to export health report: {str(e)}")
            
    def set_threshold(self, metric_name: str, warning: float, critical: float):
        """Set custom thresholds for a metric."""
        self.thresholds[metric_name] = {
            "warning": warning,
            "critical": critical
        }
        logger.info(f"Updated thresholds for {metric_name}: warning={warning}, critical={critical}")


class AlertManager:
    """
    Manages different types of alerts and notifications.
    """
    
    def __init__(self):
        self.alert_handlers = {}
        
    def add_handler(self, handler_type: str, handler_func: Callable):
        """Add alert handler."""
        self.alert_handlers[handler_type] = handler_func
        
    def send_alert(self, alert_data: Dict[str, Any], handler_types: List[str] = None):
        """Send alert using specified handlers."""
        if handler_types is None:
            handler_types = list(self.alert_handlers.keys())
            
        for handler_type in handler_types:
            if handler_type in self.alert_handlers:
                try:
                    self.alert_handlers[handler_type](alert_data)
                except Exception as e:
                    logger.error(f"Alert handler {handler_type} failed: {str(e)}")
                    
    def email_handler(self, alert_data: Dict[str, Any]):
        """Email alert handler (placeholder)."""
        logger.info(f"EMAIL ALERT: {alert_data}")
        
    def slack_handler(self, alert_data: Dict[str, Any]):
        """Slack alert handler (placeholder)."""
        logger.info(f"SLACK ALERT: {alert_data}")
        
    def webhook_handler(self, alert_data: Dict[str, Any]):
        """Webhook alert handler (placeholder)."""
        logger.info(f"WEBHOOK ALERT: {alert_data}")