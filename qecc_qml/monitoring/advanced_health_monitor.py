"""
Advanced Health Monitoring System for QECC-Aware QML

This module provides real-time health monitoring, anomaly detection, and 
predictive maintenance for quantum machine learning systems with error correction.
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
import time
import json
import threading
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from collections import deque
import warnings
from datetime import datetime, timedelta


class HealthStatus(Enum):
    """System health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    FAILURE = "failure"


class MetricType(Enum):
    """Types of metrics being monitored."""
    QUANTUM_FIDELITY = "quantum_fidelity"
    ERROR_RATE = "error_rate"
    GATE_FIDELITY = "gate_fidelity"
    COHERENCE_TIME = "coherence_time"
    SYNDROME_EXTRACTION_SUCCESS = "syndrome_extraction_success"
    DECODING_SUCCESS_RATE = "decoding_success_rate"
    TRAINING_LOSS = "training_loss"
    TRAINING_ACCURACY = "training_accuracy"
    RESOURCE_UTILIZATION = "resource_utilization"
    RESPONSE_TIME = "response_time"


@dataclass
class HealthMetric:
    """Individual health metric data point."""
    timestamp: datetime
    metric_type: MetricType
    value: float
    source: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class HealthAlert:
    """Health monitoring alert."""
    timestamp: datetime
    severity: HealthStatus
    metric_type: MetricType
    message: str
    current_value: float
    threshold: float
    recommendation: Optional[str] = None
    auto_recoverable: bool = False


@dataclass
class SystemHealthReport:
    """Comprehensive system health report."""
    timestamp: datetime
    overall_status: HealthStatus
    component_statuses: Dict[str, HealthStatus]
    active_alerts: List[HealthAlert]
    performance_summary: Dict[str, float]
    trends: Dict[str, str]  # "improving", "stable", "degrading"
    recommendations: List[str]


class HealthThreshold:
    """Dynamic health threshold with adaptive capabilities."""
    
    def __init__(self, metric_type: MetricType, 
                 warning_threshold: float,
                 critical_threshold: float,
                 adaptive: bool = True):
        self.metric_type = metric_type
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.adaptive = adaptive
        self.baseline_window = deque(maxlen=100)
        self.adaptation_factor = 0.1
        
    def update_baseline(self, value: float):
        """Update baseline for adaptive thresholds."""
        if self.adaptive:
            self.baseline_window.append(value)
            
    def get_thresholds(self) -> Tuple[float, float]:
        """Get current warning and critical thresholds."""
        if not self.adaptive or len(self.baseline_window) < 10:
            return self.warning_threshold, self.critical_threshold
        
        # Adaptive thresholds based on recent baseline
        baseline_mean = np.mean(self.baseline_window)
        baseline_std = np.std(self.baseline_window)
        
        # Adjust thresholds based on baseline statistics
        if self.metric_type in [MetricType.ERROR_RATE, MetricType.RESPONSE_TIME]:
            # For metrics where lower is better
            warning = baseline_mean + 2 * baseline_std
            critical = baseline_mean + 3 * baseline_std
        else:
            # For metrics where higher is better
            warning = baseline_mean - 2 * baseline_std
            critical = baseline_mean - 3 * baseline_std
        
        # Smooth threshold changes
        current_warning, current_critical = self.warning_threshold, self.critical_threshold
        self.warning_threshold = (1 - self.adaptation_factor) * current_warning + self.adaptation_factor * warning
        self.critical_threshold = (1 - self.adaptation_factor) * current_critical + self.adaptation_factor * critical
        
        return self.warning_threshold, self.critical_threshold


class AnomalyDetector:
    """Advanced anomaly detection for quantum system metrics."""
    
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.metric_windows = {}
        self.seasonal_patterns = {}
        
    def add_metric(self, metric: HealthMetric) -> Optional[float]:
        """Add metric and return anomaly score (0-1, higher = more anomalous)."""
        key = f"{metric.metric_type.value}_{metric.source}"
        
        if key not in self.metric_windows:
            self.metric_windows[key] = deque(maxlen=self.window_size)
        
        window = self.metric_windows[key]
        window.append((metric.timestamp, metric.value))
        
        if len(window) < 10:
            return None
        
        # Calculate anomaly score using multiple techniques
        scores = []
        
        # Statistical anomaly detection
        values = [v for _, v in window]
        mean_val = np.mean(values)
        std_val = np.std(values)
        if std_val > 0:
            z_score = abs(metric.value - mean_val) / std_val
            scores.append(min(1.0, z_score / 3.0))  # Normalize to 0-1
        
        # Trend-based anomaly detection
        if len(window) >= 20:
            recent_values = values[-10:]
            older_values = values[-20:-10]
            
            recent_mean = np.mean(recent_values)
            older_mean = np.mean(older_values)
            
            if older_mean != 0:
                trend_change = abs(recent_mean - older_mean) / abs(older_mean)
                scores.append(min(1.0, trend_change))
        
        # Periodicity-based anomaly detection
        periodicity_score = self._detect_periodicity_anomaly(key, metric)
        if periodicity_score is not None:
            scores.append(periodicity_score)
        
        return np.mean(scores) if scores else 0.0
    
    def _detect_periodicity_anomaly(self, key: str, metric: HealthMetric) -> Optional[float]:
        """Detect anomalies based on expected periodic patterns."""
        if key not in self.seasonal_patterns:
            return None
        
        # This would implement more sophisticated periodicity detection
        # For now, return a simple placeholder
        return 0.0


class PredictiveMaintenanceEngine:
    """Predictive maintenance using machine learning."""
    
    def __init__(self):
        self.failure_patterns = {}
        self.maintenance_history = []
        self.prediction_window = timedelta(hours=24)
        
    def train_failure_prediction(self, historical_metrics: List[HealthMetric],
                               failure_events: List[datetime]):
        """Train failure prediction model on historical data."""
        # Simplified training - in practice would use sophisticated ML
        print("🤖 Training predictive maintenance model...")
        
        # Extract features from metrics before failures
        for failure_time in failure_events:
            pre_failure_window = failure_time - timedelta(hours=6)
            
            pre_failure_metrics = [
                m for m in historical_metrics 
                if pre_failure_window <= m.timestamp <= failure_time
            ]
            
            # Extract patterns
            self._extract_failure_patterns(pre_failure_metrics)
    
    def _extract_failure_patterns(self, metrics: List[HealthMetric]):
        """Extract patterns that precede failures."""
        # Group metrics by type
        by_type = {}
        for metric in metrics:
            if metric.metric_type not in by_type:
                by_type[metric.metric_type] = []
            by_type[metric.metric_type].append(metric.value)
        
        # Analyze patterns
        for metric_type, values in by_type.items():
            if len(values) >= 3:
                trend = np.polyfit(range(len(values)), values, 1)[0]
                volatility = np.std(values)
                
                if metric_type not in self.failure_patterns:
                    self.failure_patterns[metric_type] = []
                
                self.failure_patterns[metric_type].append({
                    'trend': trend,
                    'volatility': volatility,
                    'final_value': values[-1]
                })
    
    def predict_maintenance_need(self, recent_metrics: List[HealthMetric]) -> Dict[str, float]:
        """Predict maintenance needs in next 24 hours."""
        predictions = {}
        
        # Group recent metrics by type
        by_type = {}
        for metric in recent_metrics:
            if metric.metric_type not in by_type:
                by_type[metric.metric_type] = []
            by_type[metric.metric_type].append(metric.value)
        
        # Compare with known failure patterns
        for metric_type, values in by_type.items():
            if metric_type in self.failure_patterns and len(values) >= 3:
                current_trend = np.polyfit(range(len(values)), values, 1)[0]
                current_volatility = np.std(values)
                current_value = values[-1]
                
                # Calculate similarity to failure patterns
                similarities = []
                for pattern in self.failure_patterns[metric_type]:
                    trend_sim = 1.0 - abs(current_trend - pattern['trend']) / (abs(pattern['trend']) + 1e-6)
                    volatility_sim = 1.0 - abs(current_volatility - pattern['volatility']) / (pattern['volatility'] + 1e-6)
                    value_sim = 1.0 - abs(current_value - pattern['final_value']) / (abs(pattern['final_value']) + 1e-6)
                    
                    similarity = np.mean([trend_sim, volatility_sim, value_sim])
                    similarities.append(max(0.0, similarity))
                
                if similarities:
                    predictions[metric_type.value] = max(similarities)
        
        return predictions


class AdvancedHealthMonitor:
    """
    Advanced health monitoring system for QECC-aware QML.
    
    Features:
    - Real-time metric collection and analysis
    - Adaptive threshold management
    - Anomaly detection using multiple techniques
    - Predictive maintenance
    - Automatic recovery recommendations
    """
    
    def __init__(self, monitoring_interval: float = 1.0):
        self.monitoring_interval = monitoring_interval
        self.is_monitoring = False
        self.monitor_thread = None
        
        # Metric storage
        self.metrics_history = deque(maxlen=10000)
        self.active_alerts = []
        
        # Components
        self.thresholds = self._initialize_thresholds()
        self.anomaly_detector = AnomalyDetector()
        self.predictive_engine = PredictiveMaintenanceEngine()
        
        # Callbacks
        self.alert_callbacks = []
        self.metric_callbacks = []
        
        # Performance tracking
        self.component_health = {
            'quantum_circuit': HealthStatus.HEALTHY,
            'error_correction': HealthStatus.HEALTHY,
            'training_process': HealthStatus.HEALTHY,
            'hardware_backend': HealthStatus.HEALTHY,
            'monitoring_system': HealthStatus.HEALTHY
        }
        
    def _initialize_thresholds(self) -> Dict[MetricType, HealthThreshold]:
        """Initialize health thresholds for different metrics."""
        return {
            MetricType.QUANTUM_FIDELITY: HealthThreshold(
                MetricType.QUANTUM_FIDELITY, 0.95, 0.90
            ),
            MetricType.ERROR_RATE: HealthThreshold(
                MetricType.ERROR_RATE, 0.05, 0.10
            ),
            MetricType.GATE_FIDELITY: HealthThreshold(
                MetricType.GATE_FIDELITY, 0.99, 0.95
            ),
            MetricType.COHERENCE_TIME: HealthThreshold(
                MetricType.COHERENCE_TIME, 50e-6, 20e-6  # microseconds
            ),
            MetricType.SYNDROME_EXTRACTION_SUCCESS: HealthThreshold(
                MetricType.SYNDROME_EXTRACTION_SUCCESS, 0.95, 0.85
            ),
            MetricType.DECODING_SUCCESS_RATE: HealthThreshold(
                MetricType.DECODING_SUCCESS_RATE, 0.90, 0.80
            ),
            MetricType.TRAINING_ACCURACY: HealthThreshold(
                MetricType.TRAINING_ACCURACY, 0.85, 0.70
            ),
            MetricType.RESPONSE_TIME: HealthThreshold(
                MetricType.RESPONSE_TIME, 1.0, 5.0  # seconds
            )
        }
    
    def start_monitoring(self):
        """Start the health monitoring system."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        print("🔍 Advanced health monitoring started")
    
    def stop_monitoring(self):
        """Stop the health monitoring system."""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        print("⏹️  Health monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                # Collect system metrics
                self._collect_system_metrics()
                
                # Process recent metrics
                self._process_recent_metrics()
                
                # Check for maintenance needs
                self._check_predictive_maintenance()
                
                # Update component health
                self._update_component_health()
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                print(f"❌ Monitoring error: {e}")
                time.sleep(self.monitoring_interval)
    
    def _collect_system_metrics(self):
        """Collect current system metrics."""
        timestamp = datetime.now()
        
        # Simulate metric collection (in real system, these would be actual measurements)
        metrics = [
            HealthMetric(
                timestamp=timestamp,
                metric_type=MetricType.QUANTUM_FIDELITY,
                value=0.95 + 0.05 * np.random.randn(),
                source="quantum_system"
            ),
            HealthMetric(
                timestamp=timestamp,
                metric_type=MetricType.ERROR_RATE,
                value=max(0.001, 0.01 + 0.005 * np.random.randn()),
                source="error_correction"
            ),
            HealthMetric(
                timestamp=timestamp,
                metric_type=MetricType.SYNDROME_EXTRACTION_SUCCESS,
                value=min(1.0, 0.98 + 0.02 * np.random.randn()),
                source="syndrome_extractor"
            ),
            HealthMetric(
                timestamp=timestamp,
                metric_type=MetricType.RESPONSE_TIME,
                value=max(0.1, 0.5 + 0.2 * np.random.randn()),
                source="api_server"
            )
        ]
        
        for metric in metrics:
            self.add_metric(metric)
    
    def add_metric(self, metric: HealthMetric):
        """Add a new metric measurement."""
        # Store metric
        self.metrics_history.append(metric)
        
        # Update threshold baseline
        if metric.metric_type in self.thresholds:
            self.thresholds[metric.metric_type].update_baseline(metric.value)
        
        # Check for anomalies
        anomaly_score = self.anomaly_detector.add_metric(metric)
        if anomaly_score and anomaly_score > 0.7:
            self._create_anomaly_alert(metric, anomaly_score)
        
        # Check thresholds
        self._check_thresholds(metric)
        
        # Call metric callbacks
        for callback in self.metric_callbacks:
            try:
                callback(metric)
            except Exception as e:
                print(f"❌ Metric callback error: {e}")
    
    def _check_thresholds(self, metric: HealthMetric):
        """Check if metric violates thresholds."""
        if metric.metric_type not in self.thresholds:
            return
        
        threshold = self.thresholds[metric.metric_type]
        warning_thresh, critical_thresh = threshold.get_thresholds()
        
        # Determine severity based on metric type and thresholds
        severity = None
        message = ""
        
        if metric.metric_type in [MetricType.ERROR_RATE, MetricType.RESPONSE_TIME]:
            # Higher values are worse
            if metric.value >= critical_thresh:
                severity = HealthStatus.CRITICAL
                message = f"{metric.metric_type.value} critically high: {metric.value:.4f}"
            elif metric.value >= warning_thresh:
                severity = HealthStatus.WARNING
                message = f"{metric.metric_type.value} above warning threshold: {metric.value:.4f}"
        else:
            # Lower values are worse
            if metric.value <= critical_thresh:
                severity = HealthStatus.CRITICAL
                message = f"{metric.metric_type.value} critically low: {metric.value:.4f}"
            elif metric.value <= warning_thresh:
                severity = HealthStatus.WARNING
                message = f"{metric.metric_type.value} below warning threshold: {metric.value:.4f}"
        
        if severity:
            alert = HealthAlert(
                timestamp=metric.timestamp,
                severity=severity,
                metric_type=metric.metric_type,
                message=message,
                current_value=metric.value,
                threshold=warning_thresh if severity == HealthStatus.WARNING else critical_thresh,
                recommendation=self._get_threshold_recommendation(metric.metric_type, severity)
            )
            self._add_alert(alert)
    
    def _create_anomaly_alert(self, metric: HealthMetric, anomaly_score: float):
        """Create alert for detected anomaly."""
        alert = HealthAlert(
            timestamp=metric.timestamp,
            severity=HealthStatus.WARNING,
            metric_type=metric.metric_type,
            message=f"Anomaly detected in {metric.metric_type.value} (score: {anomaly_score:.2f})",
            current_value=metric.value,
            threshold=anomaly_score,
            recommendation="Investigate recent system changes or external factors."
        )
        self._add_alert(alert)
    
    def _add_alert(self, alert: HealthAlert):
        """Add new alert and notify callbacks."""
        # Check for duplicate alerts (same metric type and severity)
        existing = [a for a in self.active_alerts 
                   if a.metric_type == alert.metric_type and a.severity == alert.severity]
        
        if not existing:
            self.active_alerts.append(alert)
            
            # Call alert callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    print(f"❌ Alert callback error: {e}")
            
            print(f"🚨 ALERT: {alert.message}")
    
    def _get_threshold_recommendation(self, metric_type: MetricType, 
                                    severity: HealthStatus) -> str:
        """Get recommendation for threshold violation."""
        recommendations = {
            MetricType.QUANTUM_FIDELITY: {
                HealthStatus.WARNING: "Check quantum circuit optimization and noise mitigation",
                HealthStatus.CRITICAL: "Immediate intervention required - check hardware calibration"
            },
            MetricType.ERROR_RATE: {
                HealthStatus.WARNING: "Consider increasing error correction strength",
                HealthStatus.CRITICAL: "Critical error rate - suspend operations and investigate"
            },
            MetricType.SYNDROME_EXTRACTION_SUCCESS: {
                HealthStatus.WARNING: "Review syndrome extraction frequency and method",
                HealthStatus.CRITICAL: "Syndrome extraction failing - check measurement circuits"
            },
            MetricType.RESPONSE_TIME: {
                HealthStatus.WARNING: "System load increasing - consider optimization",
                HealthStatus.CRITICAL: "System overloaded - immediate scaling required"
            }
        }
        
        return recommendations.get(metric_type, {}).get(
            severity, "Monitor closely and investigate if condition persists"
        )
    
    def _process_recent_metrics(self):
        """Process recent metrics for trends and patterns."""
        # Clear resolved alerts
        current_time = datetime.now()
        self.active_alerts = [
            alert for alert in self.active_alerts 
            if current_time - alert.timestamp < timedelta(minutes=30)
        ]
    
    def _check_predictive_maintenance(self):
        """Check for predictive maintenance needs."""
        if len(self.metrics_history) < 50:
            return
        
        # Get recent metrics for prediction
        recent_metrics = list(self.metrics_history)[-50:]
        predictions = self.predictive_engine.predict_maintenance_need(recent_metrics)
        
        for metric_type, probability in predictions.items():
            if probability > 0.8:
                alert = HealthAlert(
                    timestamp=datetime.now(),
                    severity=HealthStatus.WARNING,
                    metric_type=MetricType(metric_type),
                    message=f"Predictive maintenance alert: {metric_type} (probability: {probability:.2f})",
                    current_value=probability,
                    threshold=0.8,
                    recommendation="Schedule preventive maintenance in next 24 hours"
                )
                self._add_alert(alert)
    
    def _update_component_health(self):
        """Update overall component health status."""
        # Analyze recent alerts by component
        recent_alerts = [
            alert for alert in self.active_alerts
            if datetime.now() - alert.timestamp < timedelta(minutes=10)
        ]
        
        # Reset all to healthy
        for component in self.component_health:
            self.component_health[component] = HealthStatus.HEALTHY
        
        # Update based on active alerts
        for alert in recent_alerts:
            if alert.metric_type in [MetricType.QUANTUM_FIDELITY, MetricType.GATE_FIDELITY]:
                self.component_health['quantum_circuit'] = max(
                    self.component_health['quantum_circuit'], alert.severity,
                    key=lambda x: list(HealthStatus).index(x)
                )
            elif alert.metric_type in [MetricType.ERROR_RATE, MetricType.SYNDROME_EXTRACTION_SUCCESS]:
                self.component_health['error_correction'] = max(
                    self.component_health['error_correction'], alert.severity,
                    key=lambda x: list(HealthStatus).index(x)
                )
    
    def get_health_report(self) -> SystemHealthReport:
        """Generate comprehensive health report."""
        current_time = datetime.now()
        
        # Determine overall status
        component_statuses = list(self.component_health.values())
        if HealthStatus.CRITICAL in component_statuses:
            overall_status = HealthStatus.CRITICAL
        elif HealthStatus.DEGRADED in component_statuses:
            overall_status = HealthStatus.DEGRADED
        elif HealthStatus.WARNING in component_statuses:
            overall_status = HealthStatus.WARNING
        else:
            overall_status = HealthStatus.HEALTHY
        
        # Calculate performance summary
        recent_metrics = [
            m for m in self.metrics_history 
            if current_time - m.timestamp < timedelta(minutes=10)
        ]
        
        performance_summary = {}
        for metric_type in MetricType:
            relevant_metrics = [m for m in recent_metrics if m.metric_type == metric_type]
            if relevant_metrics:
                performance_summary[metric_type.value] = np.mean([m.value for m in relevant_metrics])
        
        # Analyze trends
        trends = self._analyze_trends()
        
        # Generate recommendations
        recommendations = self._generate_health_recommendations()
        
        return SystemHealthReport(
            timestamp=current_time,
            overall_status=overall_status,
            component_statuses=self.component_health.copy(),
            active_alerts=self.active_alerts.copy(),
            performance_summary=performance_summary,
            trends=trends,
            recommendations=recommendations
        )
    
    def _analyze_trends(self) -> Dict[str, str]:
        """Analyze metric trends over time."""
        trends = {}
        current_time = datetime.now()
        
        for metric_type in MetricType:
            # Get metrics from last hour
            hour_metrics = [
                m for m in self.metrics_history 
                if m.metric_type == metric_type and 
                   current_time - m.timestamp < timedelta(hours=1)
            ]
            
            if len(hour_metrics) >= 10:
                values = [m.value for m in hour_metrics]
                trend_slope = np.polyfit(range(len(values)), values, 1)[0]
                
                if abs(trend_slope) < 0.001:
                    trends[metric_type.value] = "stable"
                elif trend_slope > 0:
                    if metric_type in [MetricType.ERROR_RATE, MetricType.RESPONSE_TIME]:
                        trends[metric_type.value] = "degrading"
                    else:
                        trends[metric_type.value] = "improving"
                else:
                    if metric_type in [MetricType.ERROR_RATE, MetricType.RESPONSE_TIME]:
                        trends[metric_type.value] = "improving"
                    else:
                        trends[metric_type.value] = "degrading"
        
        return trends
    
    def _generate_health_recommendations(self) -> List[str]:
        """Generate health recommendations based on current state."""
        recommendations = []
        
        # Recommendations based on active alerts
        critical_alerts = [a for a in self.active_alerts if a.severity == HealthStatus.CRITICAL]
        warning_alerts = [a for a in self.active_alerts if a.severity == HealthStatus.WARNING]
        
        if critical_alerts:
            recommendations.append("🚨 Critical issues detected - immediate intervention required")
            for alert in critical_alerts[:3]:  # Top 3 critical issues
                if alert.recommendation:
                    recommendations.append(f"• {alert.recommendation}")
        
        if warning_alerts:
            recommendations.append("⚠️ Warning conditions detected - monitor closely")
        
        # Performance recommendations
        if not self.active_alerts:
            recommendations.append("✅ System operating normally - continue monitoring")
        
        return recommendations
    
    def register_alert_callback(self, callback: Callable[[HealthAlert], None]):
        """Register callback for alert notifications."""
        self.alert_callbacks.append(callback)
    
    def register_metric_callback(self, callback: Callable[[HealthMetric], None]):
        """Register callback for metric updates."""
        self.metric_callbacks.append(callback)


def demo_advanced_health_monitoring():
    """Demonstrate advanced health monitoring capabilities."""
    print("🏥 Advanced Health Monitoring Demo")
    print("=" * 50)
    
    # Initialize monitor
    monitor = AdvancedHealthMonitor(monitoring_interval=0.5)
    
    # Register alert callback
    def alert_handler(alert: HealthAlert):
        emoji = {"warning": "⚠️", "critical": "🚨", "degraded": "⬇️"}
        print(f"{emoji.get(alert.severity.value, '🔔')} Alert: {alert.message}")
        if alert.recommendation:
            print(f"    💡 Recommendation: {alert.recommendation}")
    
    monitor.register_alert_callback(alert_handler)
    
    # Start monitoring
    monitor.start_monitoring()
    
    # Simulate some events
    print("\n📊 Monitoring for 10 seconds...")
    time.sleep(3)
    
    # Inject some problematic metrics
    print("\n⚠️ Injecting degraded performance...")
    bad_metrics = [
        HealthMetric(
            timestamp=datetime.now(),
            metric_type=MetricType.ERROR_RATE,
            value=0.15,  # High error rate
            source="test_injection"
        ),
        HealthMetric(
            timestamp=datetime.now(),
            metric_type=MetricType.QUANTUM_FIDELITY,
            value=0.85,  # Low fidelity
            source="test_injection"
        )
    ]
    
    for metric in bad_metrics:
        monitor.add_metric(metric)
    
    time.sleep(2)
    
    # Generate health report
    report = monitor.get_health_report()
    print(f"\n🏥 Health Report")
    print(f"Overall Status: {report.overall_status.value.upper()}")
    print(f"Active Alerts: {len(report.active_alerts)}")
    print(f"Component Health:")
    for component, status in report.component_statuses.items():
        emoji = {"healthy": "✅", "warning": "⚠️", "critical": "🚨", "degraded": "⬇️"}
        print(f"  {emoji.get(status.value, '❓')} {component}: {status.value}")
    
    print(f"\nRecommendations:")
    for rec in report.recommendations:
        print(f"  {rec}")
    
    # Stop monitoring
    monitor.stop_monitoring()
    
    return monitor


if __name__ == "__main__":
    demo_advanced_health_monitoring()