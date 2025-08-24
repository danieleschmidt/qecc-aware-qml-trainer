"""
Comprehensive Health Monitoring for Quantum Machine Learning Systems.

This module provides real-time monitoring, alerting, and performance tracking
for quantum machine learning operations with predictive health analysis.
"""

# Import with fallback support
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from qecc_qml.core.fallback_imports import create_fallback_implementations
    create_fallback_implementations()
except ImportError:
    pass
import time
import threading
import psutil
import logging
import json
try:
    import numpy as np
except ImportError:
    import sys
    if 'numpy' in sys.modules:
        np = sys.modules['numpy']
    else:
        class MockNumPy:
            @staticmethod
            def array(x): return list(x) if isinstance(x, (list, tuple)) else x
            @staticmethod
            def zeros(shape): return [0] * (shape if isinstance(shape, int) else shape[0])
            @staticmethod  
            def ones(shape): return [1] * (shape if isinstance(shape, int) else shape[0])
            ndarray = list
        np = MockNumPy()
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor
import warnings
from pathlib import Path

from ..utils.logging_config import get_logger

logger = get_logger(__name__)

@dataclass
class HealthMetrics:
    """Comprehensive health metrics."""
    timestamp: float = 0.0
    
    # System metrics
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    disk_usage: float = 0.0
    
    # Application metrics
    active_circuits: int = 0
    total_executions: int = 0
    error_rate: float = 0.0
    average_execution_time: float = 0.0
    
    # Quantum-specific metrics
    fidelity_average: float = 0.0
    gate_error_rate: float = 0.0
    decoherence_rate: float = 0.0
    
    # Performance metrics
    throughput: float = 0.0
    latency_p95: float = 0.0
    queue_length: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'timestamp': self.timestamp,
            'cpu_usage': self.cpu_usage,
            'memory_usage': self.memory_usage,
            'disk_usage': self.disk_usage,
            'active_circuits': self.active_circuits,
            'total_executions': self.total_executions,
            'error_rate': self.error_rate,
            'average_execution_time': self.average_execution_time,
            'fidelity_average': self.fidelity_average,
            'gate_error_rate': self.gate_error_rate,
            'decoherence_rate': self.decoherence_rate,
            'throughput': self.throughput,
            'latency_p95': self.latency_p95,
            'queue_length': self.queue_length
        }

@dataclass
class HealthAlert:
    """Health alert definition."""
    alert_id: str
    severity: str  # 'info', 'warning', 'error', 'critical'
    title: str
    description: str
    metric_name: str
    threshold: float
    current_value: float
    timestamp: float
    resolved: bool = False
    resolution_time: Optional[float] = None

@dataclass
class MonitoringConfig:
    """Configuration for health monitoring."""
    collection_interval: float = 10.0  # seconds
    metric_retention_hours: int = 24
    alert_thresholds: Dict[str, Dict[str, float]] = field(default_factory=dict)
    enable_predictive_alerts: bool = True
    enable_auto_healing: bool = False
    notification_callbacks: List[Callable] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.alert_thresholds:
            self.alert_thresholds = {
                'cpu_usage': {'warning': 70.0, 'critical': 90.0},
                'memory_usage': {'warning': 80.0, 'critical': 95.0},
                'error_rate': {'warning': 0.05, 'critical': 0.15},
                'fidelity_average': {'warning': 0.8, 'critical': 0.6},
                'latency_p95': {'warning': 5.0, 'critical': 10.0}
            }

class PredictiveAnalyzer:
    """Predictive analysis for health metrics."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metric_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
    
    def add_metric(self, metric_name: str, value: float, timestamp: float) -> None:
        """Add metric value for analysis."""
        self.metric_history[metric_name].append((timestamp, value))
    
    def predict_trend(self, metric_name: str, look_ahead_minutes: int = 30) -> Dict[str, Any]:
        """Predict metric trend using simple linear regression."""
        if metric_name not in self.metric_history or len(self.metric_history[metric_name]) < 10:
            return {'prediction': None, 'confidence': 0.0, 'trend': 'unknown'}
        
        data = list(self.metric_history[metric_name])
        times = np.array([point[0] for point in data])
        values = np.array([point[1] for point in data])
        
        # Normalize time for better numerical stability
        times_norm = (times - times[0]) / 3600.0  # Convert to hours
        
        # Simple linear regression
        try:
            coeffs = np.polyfit(times_norm, values, 1)
            slope, intercept = coeffs
            
            # Predict future value
            future_time = times_norm[-1] + (look_ahead_minutes / 60.0)
            predicted_value = slope * future_time + intercept
            
            # Calculate trend
            if abs(slope) < 0.001:
                trend = 'stable'
            elif slope > 0:
                trend = 'increasing'
            else:
                trend = 'decreasing'
            
            # Simple confidence based on variance
            residuals = values - (slope * times_norm + intercept)
            variance = np.var(residuals)
            confidence = max(0.0, min(1.0, 1.0 - variance / np.var(values)))
            
            return {
                'prediction': predicted_value,
                'confidence': confidence,
                'trend': trend,
                'slope': slope
            }
            
        except Exception as e:
            logger.warning(f"Prediction error for {metric_name}: {e}")
            return {'prediction': None, 'confidence': 0.0, 'trend': 'unknown'}
    
    def detect_anomalies(self, metric_name: str, current_value: float) -> Dict[str, Any]:
        """Detect anomalies in metric values."""
        if metric_name not in self.metric_history or len(self.metric_history[metric_name]) < 20:
            return {'is_anomaly': False, 'severity': 'normal', 'score': 0.0}
        
        values = np.array([point[1] for point in self.metric_history[metric_name]])
        
        try:
            mean_val = np.mean(values)
            std_val = np.std(values)
            
            if std_val == 0:
                return {'is_anomaly': False, 'severity': 'normal', 'score': 0.0}
            
            # Z-score based anomaly detection
            z_score = abs(current_value - mean_val) / std_val
            
            if z_score > 3:
                severity = 'critical'
                is_anomaly = True
            elif z_score > 2:
                severity = 'warning'
                is_anomaly = True
            else:
                severity = 'normal'
                is_anomaly = False
            
            return {
                'is_anomaly': is_anomaly,
                'severity': severity,
                'score': z_score,
                'mean': mean_val,
                'std': std_val
            }
            
        except Exception as e:
            logger.warning(f"Anomaly detection error for {metric_name}: {e}")
            return {'is_anomaly': False, 'severity': 'normal', 'score': 0.0}

class ComprehensiveHealthMonitor:
    """
    Comprehensive health monitoring system for quantum machine learning.
    
    Provides real-time monitoring, alerting, predictive analysis, and
    automated health management for quantum computing systems.
    """
    
    def __init__(self, config: Optional[MonitoringConfig] = None):
        """Initialize comprehensive health monitor."""
        self.config = config or MonitoringConfig()
        self.logger = get_logger(__name__)
        self.is_running = False
        self.monitoring_thread: Optional[threading.Thread] = None
        
        # Metrics storage
        self.current_metrics = HealthMetrics()
        self.metrics_history: deque = deque(maxlen=int(self.config.metric_retention_hours * 3600 / self.config.collection_interval))
        
        # Alerting system
        self.active_alerts: Dict[str, HealthAlert] = {}
        self.alert_history: List[HealthAlert] = []
        
        # Predictive analysis
        self.predictor = PredictiveAnalyzer()
        
        # Performance tracking
        self.execution_times: deque = deque(maxlen=1000)
        self.error_counts: deque = deque(maxlen=1000)
        self.circuit_queue: List[Any] = []
        
        # Auto-healing
        self.healing_actions: Dict[str, Callable] = {}
        
        self.logger.info("Comprehensive health monitor initialized")
    
    def start_monitoring(self) -> None:
        """Start background health monitoring."""
        if self.is_running:
            self.logger.warning("Monitoring already running")
            return
        
        self.is_running = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        self.logger.info("Health monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop health monitoring."""
        self.is_running = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        self.logger.info("Health monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.is_running:
            try:
                # Collect metrics
                self.collect_metrics()
                
                # Check for alerts
                self.check_alerts()
                
                # Predictive analysis
                if self.config.enable_predictive_alerts:
                    self.run_predictive_analysis()
                
                # Auto-healing
                if self.config.enable_auto_healing:
                    self.attempt_auto_healing()
                
                # Clean up old data
                self.cleanup_old_data()
                
                time.sleep(self.config.collection_interval)
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                time.sleep(self.config.collection_interval)
    
    def collect_metrics(self) -> HealthMetrics:
        """Collect comprehensive health metrics."""
        current_time = time.time()
        metrics = HealthMetrics(timestamp=current_time)
        
        try:
            # System metrics
            metrics.cpu_usage = psutil.cpu_percent(interval=1)
            metrics.memory_usage = psutil.virtual_memory().percent
            metrics.disk_usage = psutil.disk_usage('/').percent
            
            # Application metrics
            metrics.active_circuits = len(self.circuit_queue)
            metrics.total_executions = len(self.execution_times)
            
            # Calculate error rate
            if self.error_counts:
                recent_errors = sum(1 for timestamp in self.error_counts if timestamp > current_time - 3600)
                recent_executions = sum(1 for timestamp in self.execution_times if timestamp > current_time - 3600)
                metrics.error_rate = recent_errors / max(recent_executions, 1)
            
            # Calculate average execution time
            if self.execution_times:
                recent_times = [t for t in self.execution_times if t > current_time - 3600]
                if recent_times:
                    metrics.average_execution_time = np.mean(recent_times)
            
            # Quantum-specific metrics (simulated)
            metrics.fidelity_average = self._calculate_average_fidelity()
            metrics.gate_error_rate = self._estimate_gate_error_rate()
            metrics.decoherence_rate = self._estimate_decoherence_rate()
            
            # Performance metrics
            metrics.throughput = self._calculate_throughput()
            metrics.latency_p95 = self._calculate_latency_percentile(95)
            metrics.queue_length = len(self.circuit_queue)
            
            # Store metrics
            self.current_metrics = metrics
            self.metrics_history.append(metrics)
            
            # Add to predictive analyzer
            for attr_name in ['cpu_usage', 'memory_usage', 'error_rate', 'fidelity_average']:
                value = getattr(metrics, attr_name)
                self.predictor.add_metric(attr_name, value, current_time)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error collecting metrics: {e}")
            return metrics
    
    def _calculate_average_fidelity(self) -> float:
        """Calculate average fidelity (simulated)."""
        # In a real implementation, this would aggregate fidelity from actual quantum executions
        base_fidelity = 0.95
        noise = np.random.normal(0, 0.02)  # Small random variation
        return max(0.0, min(1.0, base_fidelity + noise))
    
    def _estimate_gate_error_rate(self) -> float:
        """Estimate gate error rate (simulated)."""
        # Simulated gate error rate with some correlation to system load
        base_error = 0.001
        load_factor = self.current_metrics.cpu_usage / 100.0
        return base_error * (1 + load_factor * 0.5)
    
    def _estimate_decoherence_rate(self) -> float:
        """Estimate decoherence rate (simulated)."""
        # Simulated decoherence rate
        return np.random.exponential(0.01)
    
    def _calculate_throughput(self) -> float:
        """Calculate operations per second."""
        current_time = time.time()
        recent_executions = sum(1 for timestamp in self.execution_times if timestamp > current_time - 60)
        return recent_executions / 60.0
    
    def _calculate_latency_percentile(self, percentile: int) -> float:
        """Calculate latency percentile."""
        if not self.execution_times:
            return 0.0
        
        current_time = time.time()
        recent_times = [t for t in self.execution_times if t > current_time - 3600]
        
        if not recent_times:
            return 0.0
        
        return np.percentile(recent_times, percentile)
    
    def check_alerts(self) -> None:
        """Check for alert conditions."""
        current_time = time.time()
        
        for metric_name, thresholds in self.config.alert_thresholds.items():
            if not hasattr(self.current_metrics, metric_name):
                continue
            
            current_value = getattr(self.current_metrics, metric_name)
            
            # Check thresholds
            for severity, threshold in thresholds.items():
                alert_id = f"{metric_name}_{severity}"
                
                should_alert = False
                if metric_name in ['cpu_usage', 'memory_usage', 'error_rate', 'latency_p95']:
                    should_alert = current_value > threshold
                elif metric_name == 'fidelity_average':
                    should_alert = current_value < threshold
                
                if should_alert and alert_id not in self.active_alerts:
                    # Create new alert
                    alert = HealthAlert(
                        alert_id=alert_id,
                        severity=severity,
                        title=f"{metric_name.replace('_', ' ').title()} {severity.title()}",
                        description=f"{metric_name} is {current_value:.2f}, threshold: {threshold}",
                        metric_name=metric_name,
                        threshold=threshold,
                        current_value=current_value,
                        timestamp=current_time
                    )
                    
                    self.active_alerts[alert_id] = alert
                    self.alert_history.append(alert)
                    self._send_alert_notification(alert)
                    
                elif not should_alert and alert_id in self.active_alerts:
                    # Resolve alert
                    alert = self.active_alerts[alert_id]
                    alert.resolved = True
                    alert.resolution_time = current_time
                    del self.active_alerts[alert_id]
                    self._send_resolution_notification(alert)
    
    def run_predictive_analysis(self) -> None:
        """Run predictive analysis for early warning."""
        for metric_name in ['cpu_usage', 'memory_usage', 'error_rate']:
            prediction = self.predictor.predict_trend(metric_name, look_ahead_minutes=30)
            
            if prediction['prediction'] is not None and prediction['confidence'] > 0.7:
                predicted_value = prediction['prediction']
                
                # Check if prediction exceeds thresholds
                if metric_name in self.config.alert_thresholds:
                    warning_threshold = self.config.alert_thresholds[metric_name].get('warning', float('inf'))
                    
                    if predicted_value > warning_threshold:
                        alert_id = f"predictive_{metric_name}_warning"
                        
                        if alert_id not in self.active_alerts:
                            alert = HealthAlert(
                                alert_id=alert_id,
                                severity='info',
                                title=f"Predictive Alert: {metric_name.replace('_', ' ').title()}",
                                description=f"Predicted {metric_name} will reach {predicted_value:.2f} in 30 minutes (trend: {prediction['trend']})",
                                metric_name=metric_name,
                                threshold=warning_threshold,
                                current_value=predicted_value,
                                timestamp=time.time()
                            )
                            
                            self.active_alerts[alert_id] = alert
                            self._send_alert_notification(alert)
            
            # Anomaly detection
            current_value = getattr(self.current_metrics, metric_name, 0)
            anomaly = self.predictor.detect_anomalies(metric_name, current_value)
            
            if anomaly['is_anomaly']:
                alert_id = f"anomaly_{metric_name}"
                
                if alert_id not in self.active_alerts:
                    alert = HealthAlert(
                        alert_id=alert_id,
                        severity=anomaly['severity'],
                        title=f"Anomaly Detected: {metric_name.replace('_', ' ').title()}",
                        description=f"Anomalous {metric_name}: {current_value:.2f} (z-score: {anomaly['score']:.2f})",
                        metric_name=metric_name,
                        threshold=anomaly['mean'] + 2 * anomaly['std'],
                        current_value=current_value,
                        timestamp=time.time()
                    )
                    
                    self.active_alerts[alert_id] = alert
                    self._send_alert_notification(alert)
    
    def attempt_auto_healing(self) -> None:
        """Attempt automatic healing actions."""
        for alert_id, alert in self.active_alerts.items():
            if alert.severity in ['error', 'critical'] and alert.metric_name in self.healing_actions:
                try:
                    healing_action = self.healing_actions[alert.metric_name]
                    self.logger.info(f"Attempting auto-healing for {alert.metric_name}")
                    
                    success = healing_action(alert)
                    
                    if success:
                        self.logger.info(f"Auto-healing successful for {alert.metric_name}")
                    else:
                        self.logger.warning(f"Auto-healing failed for {alert.metric_name}")
                        
                except Exception as e:
                    self.logger.error(f"Auto-healing error for {alert.metric_name}: {e}")
    
    def _send_alert_notification(self, alert: HealthAlert) -> None:
        """Send alert notification."""
        self.logger.warning(f"ALERT [{alert.severity.upper()}]: {alert.title} - {alert.description}")
        
        # Send to configured notification callbacks
        for callback in self.config.notification_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Notification callback error: {e}")
    
    def _send_resolution_notification(self, alert: HealthAlert) -> None:
        """Send alert resolution notification."""
        self.logger.info(f"RESOLVED: {alert.title}")
    
    def cleanup_old_data(self) -> None:
        """Clean up old data to prevent memory leaks."""
        current_time = time.time()
        
        # Clean execution times
        cutoff_time = current_time - 3600  # 1 hour
        self.execution_times = deque(
            [t for t in self.execution_times if t > cutoff_time],
            maxlen=1000
        )
        
        # Clean error counts
        self.error_counts = deque(
            [t for t in self.error_counts if t > cutoff_time],
            maxlen=1000
        )
        
        # Clean alert history (keep last 100)
        if len(self.alert_history) > 100:
            self.alert_history = self.alert_history[-100:]
    
    def add_execution_time(self, execution_time: float) -> None:
        """Add execution time for monitoring."""
        self.execution_times.append(execution_time)
    
    def add_error(self) -> None:
        """Record an error occurrence."""
        self.error_counts.append(time.time())
    
    def add_circuit_to_queue(self, circuit: Any) -> None:
        """Add circuit to execution queue."""
        self.circuit_queue.append(circuit)
    
    def remove_circuit_from_queue(self, circuit: Any) -> None:
        """Remove circuit from execution queue."""
        if circuit in self.circuit_queue:
            self.circuit_queue.remove(circuit)
    
    def register_healing_action(self, metric_name: str, action: Callable[[HealthAlert], bool]) -> None:
        """Register auto-healing action for a metric."""
        self.healing_actions[metric_name] = action
        self.logger.info(f"Registered healing action for {metric_name}")
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get comprehensive health summary."""
        return {
            'current_metrics': self.current_metrics.to_dict(),
            'active_alerts': len(self.active_alerts),
            'total_alerts_today': len([a for a in self.alert_history if a.timestamp > time.time() - 86400]),
            'system_status': 'healthy' if not self.active_alerts else 'degraded',
            'monitoring_uptime': time.time() - (self.metrics_history[0].timestamp if self.metrics_history else time.time()),
            'prediction_accuracy': self._calculate_prediction_accuracy()
        }
    
    def _calculate_prediction_accuracy(self) -> float:
        """Calculate prediction accuracy (simplified)."""
        # In a real implementation, this would compare predictions with actual values
        return 0.85  # Placeholder
    
    def export_health_report(self, filepath: Path) -> None:
        """Export comprehensive health report."""
        report = {
            'timestamp': time.time(),
            'health_summary': self.get_health_summary(),
            'recent_metrics': [m.to_dict() for m in list(self.metrics_history)[-100:]],
            'active_alerts': [
                {
                    'alert_id': alert.alert_id,
                    'severity': alert.severity,
                    'title': alert.title,
                    'description': alert.description,
                    'timestamp': alert.timestamp
                }
                for alert in self.active_alerts.values()
            ],
            'alert_history': [
                {
                    'alert_id': alert.alert_id,
                    'severity': alert.severity,
                    'title': alert.title,
                    'timestamp': alert.timestamp,
                    'resolved': alert.resolved
                }
                for alert in self.alert_history[-50:]  # Last 50 alerts
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Health report exported to {filepath}")

# Global health monitor instance
_global_monitor = None

def get_health_monitor(config: Optional[MonitoringConfig] = None) -> ComprehensiveHealthMonitor:
    """Get global health monitor instance."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = ComprehensiveHealthMonitor(config)
    return _global_monitor