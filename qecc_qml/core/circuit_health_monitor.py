"""
Advanced Circuit Health Monitoring with Real-time Anomaly Detection
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
import threading
import logging
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
import warnings

try:
    from qiskit_aer import AerSimulator
    from qiskit import transpile
    from qiskit.quantum_info import random_statevector
except ImportError:
    from .fallback_imports import AerSimulator, transpile, random_statevector


@dataclass
class HealthMetric:
    """Represents a circuit health metric."""
    name: str
    value: float
    timestamp: float
    threshold_min: Optional[float] = None
    threshold_max: Optional[float] = None
    is_critical: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthAlert:
    """Represents a health alert."""
    severity: str  # 'warning', 'error', 'critical'
    message: str
    timestamp: float
    metric_name: str
    current_value: float
    expected_range: Tuple[Optional[float], Optional[float]]
    suggested_action: str = ""


class CircuitHealthMonitor:
    """
    Advanced circuit health monitoring with predictive failure detection.
    
    Features:
    - Real-time circuit fidelity tracking
    - Anomaly detection using statistical methods
    - Predictive failure analysis
    - Automatic circuit repair recommendations
    - Performance regression detection
    """
    
    def __init__(
        self,
        history_size: int = 1000,
        alert_threshold: float = 0.95,
        enable_predictions: bool = True,
        monitoring_interval: float = 1.0
    ):
        self.history_size = history_size
        self.alert_threshold = alert_threshold
        self.enable_predictions = enable_predictions
        self.monitoring_interval = monitoring_interval
        
        # Metric storage
        self.metrics_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=history_size)
        )
        self.current_metrics: Dict[str, HealthMetric] = {}
        self.alerts: List[HealthAlert] = []
        
        # Monitoring control
        self._monitoring_active = False
        self._monitoring_thread = None
        self._callbacks: List[Callable[[HealthAlert], None]] = []
        
        # Statistical models for anomaly detection
        self._baseline_stats: Dict[str, Dict[str, float]] = {}
        self._anomaly_scores: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=100)
        )
        
        # Circuit cache for repeated evaluations
        self._circuit_cache: Dict[str, Any] = {}
        
        # Logger setup
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging configuration."""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def start_monitoring(self, circuits: Optional[List[Any]] = None):
        """Start continuous health monitoring."""
        if self._monitoring_active:
            self.logger.warning("Monitoring already active")
            return
        
        self._monitoring_active = True
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(circuits,)
        )
        self._monitoring_thread.daemon = True
        self._monitoring_thread.start()
        
        self.logger.info("Health monitoring started")
    
    def stop_monitoring(self):
        """Stop continuous health monitoring."""
        self._monitoring_active = False
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self._monitoring_thread.join(timeout=5.0)
        
        self.logger.info("Health monitoring stopped")
    
    def _monitoring_loop(self, circuits: Optional[List[Any]]):
        """Main monitoring loop."""
        while self._monitoring_active:
            try:
                if circuits:
                    for circuit in circuits:
                        self._evaluate_circuit_health(circuit)
                
                # Update baseline statistics
                self._update_baseline_stats()
                
                # Detect anomalies
                self._detect_anomalies()
                
                # Generate predictions if enabled
                if self.enable_predictions:
                    self._generate_predictions()
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitoring_interval)
    
    def evaluate_circuit(self, circuit: Any, shots: int = 1024) -> Dict[str, float]:
        """
        Evaluate a single circuit's health metrics.
        
        Args:
            circuit: Quantum circuit to evaluate
            shots: Number of measurement shots
            
        Returns:
            Dictionary of health metrics
        """
        circuit_hash = self._get_circuit_hash(circuit)
        
        # Check cache first
        if circuit_hash in self._circuit_cache:
            cached_result = self._circuit_cache[circuit_hash]
            if time.time() - cached_result['timestamp'] < 60:  # 1-minute cache
                return cached_result['metrics']
        
        metrics = {}
        timestamp = time.time()
        
        try:
            # Fidelity estimation
            fidelity = self._estimate_circuit_fidelity(circuit, shots)
            metrics['fidelity'] = fidelity
            
            # Circuit complexity metrics
            depth = getattr(circuit, 'depth', lambda: 0)()
            metrics['depth'] = depth
            
            gate_count = len(getattr(circuit, 'data', []))
            metrics['gate_count'] = gate_count
            
            # Entanglement metrics
            entanglement_measure = self._calculate_entanglement_measure(circuit)
            metrics['entanglement'] = entanglement_measure
            
            # Error probability estimation
            error_prob = self._estimate_error_probability(circuit)
            metrics['error_probability'] = error_prob
            
            # Success probability
            success_prob = 1.0 - error_prob
            metrics['success_probability'] = success_prob
            
            # Store in cache
            self._circuit_cache[circuit_hash] = {
                'metrics': metrics,
                'timestamp': timestamp
            }
            
            # Update health metrics
            for name, value in metrics.items():
                metric = HealthMetric(
                    name=name,
                    value=value,
                    timestamp=timestamp,
                    is_critical=(name in ['fidelity', 'success_probability'])
                )
                self._update_metric(metric)
            
        except Exception as e:
            self.logger.error(f"Error evaluating circuit health: {e}")
            metrics['evaluation_error'] = 1.0
        
        return metrics
    
    def _evaluate_circuit_health(self, circuit: Any):
        """Internal method to evaluate circuit health during monitoring."""
        self.evaluate_circuit(circuit)
    
    def _estimate_circuit_fidelity(self, circuit: Any, shots: int) -> float:
        """Estimate circuit fidelity through sampling."""
        try:
            backend = AerSimulator()
            
            # Create a test circuit
            if hasattr(circuit, 'copy'):
                test_circuit = circuit.copy()
            else:
                test_circuit = circuit
            
            # Add measurements if not present
            if not hasattr(test_circuit, 'clbits') or len(test_circuit.clbits) == 0:
                from qiskit import ClassicalRegister
                num_qubits = test_circuit.num_qubits
                creg = ClassicalRegister(num_qubits, 'c')
                test_circuit.add_register(creg)
                test_circuit.measure_all()
            
            # Execute and measure fidelity
            job = backend.run(test_circuit, shots=shots)
            result = job.result()
            counts = result.get_counts(0)
            
            # Estimate fidelity from measurement statistics
            total_shots = sum(counts.values())
            max_count = max(counts.values()) if counts else 0
            fidelity = max_count / total_shots if total_shots > 0 else 0.0
            
            return min(fidelity, 1.0)
            
        except Exception as e:
            self.logger.warning(f"Could not estimate fidelity: {e}")
            return 0.5  # Default conservative estimate
    
    def _calculate_entanglement_measure(self, circuit: Any) -> float:
        """Calculate a simple entanglement measure based on two-qubit gates."""
        try:
            if not hasattr(circuit, 'data'):
                return 0.0
            
            two_qubit_gates = 0
            total_gates = len(circuit.data)
            
            for instruction in circuit.data:
                if len(instruction.qubits) >= 2:
                    two_qubit_gates += 1
            
            return two_qubit_gates / max(total_gates, 1)
            
        except Exception:
            return 0.0
    
    def _estimate_error_probability(self, circuit: Any) -> float:
        """Estimate overall error probability based on circuit structure."""
        try:
            depth = getattr(circuit, 'depth', lambda: 0)()
            gate_count = len(getattr(circuit, 'data', []))
            
            # Simple error model: assume 0.1% error per gate
            base_error_rate = 0.001
            total_error_prob = 1 - (1 - base_error_rate) ** gate_count
            
            # Depth contributes to decoherence
            decoherence_factor = 1 + depth * 0.0001
            
            return min(total_error_prob * decoherence_factor, 1.0)
            
        except Exception:
            return 0.1  # Default conservative estimate
    
    def _get_circuit_hash(self, circuit: Any) -> str:
        """Generate a hash for circuit caching."""
        try:
            # Simple hash based on circuit structure
            circuit_str = str(circuit)
            return str(hash(circuit_str))
        except Exception:
            return str(id(circuit))
    
    def _update_metric(self, metric: HealthMetric):
        """Update a health metric and check for alerts."""
        self.current_metrics[metric.name] = metric
        self.metrics_history[metric.name].append(metric)
        
        # Check for threshold violations
        self._check_metric_thresholds(metric)
    
    def _check_metric_thresholds(self, metric: HealthMetric):
        """Check if a metric violates its thresholds."""
        alerts_generated = []
        
        if metric.threshold_min is not None and metric.value < metric.threshold_min:
            alert = HealthAlert(
                severity='critical' if metric.is_critical else 'warning',
                message=f"{metric.name} below minimum threshold",
                timestamp=metric.timestamp,
                metric_name=metric.name,
                current_value=metric.value,
                expected_range=(metric.threshold_min, metric.threshold_max),
                suggested_action=self._get_suggested_action(metric.name, 'low')
            )
            alerts_generated.append(alert)
        
        if metric.threshold_max is not None and metric.value > metric.threshold_max:
            alert = HealthAlert(
                severity='critical' if metric.is_critical else 'warning',
                message=f"{metric.name} above maximum threshold",
                timestamp=metric.timestamp,
                metric_name=metric.name,
                current_value=metric.value,
                expected_range=(metric.threshold_min, metric.threshold_max),
                suggested_action=self._get_suggested_action(metric.name, 'high')
            )
            alerts_generated.append(alert)
        
        # Add alerts and trigger callbacks
        for alert in alerts_generated:
            self.alerts.append(alert)
            self._trigger_alert_callbacks(alert)
    
    def _get_suggested_action(self, metric_name: str, condition: str) -> str:
        """Get suggested action for a metric condition."""
        suggestions = {
            ('fidelity', 'low'): "Reduce circuit depth, add error correction, or use noise mitigation",
            ('success_probability', 'low'): "Simplify circuit or improve error correction",
            ('error_probability', 'high'): "Add error correction or reduce circuit complexity",
            ('depth', 'high'): "Optimize circuit depth or use circuit compilation",
            ('gate_count', 'high'): "Optimize gate count or use more efficient gate sequences"
        }
        
        return suggestions.get((metric_name, condition), "Review circuit parameters")
    
    def _update_baseline_stats(self):
        """Update baseline statistics for anomaly detection."""
        for metric_name, history in self.metrics_history.items():
            if len(history) >= 10:  # Minimum samples for statistics
                values = [m.value for m in history]
                self._baseline_stats[metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values)
                }
    
    def _detect_anomalies(self):
        """Detect anomalies in metrics using statistical methods."""
        for metric_name, current_metric in self.current_metrics.items():
            if metric_name in self._baseline_stats:
                stats = self._baseline_stats[metric_name]
                
                # Z-score anomaly detection
                if stats['std'] > 0:
                    z_score = abs(current_metric.value - stats['mean']) / stats['std']
                    self._anomaly_scores[metric_name].append(z_score)
                    
                    # Trigger alert for significant anomalies
                    if z_score > 3.0:  # 3-sigma rule
                        alert = HealthAlert(
                            severity='warning',
                            message=f"Statistical anomaly detected in {metric_name}",
                            timestamp=current_metric.timestamp,
                            metric_name=metric_name,
                            current_value=current_metric.value,
                            expected_range=(stats['mean'] - 2*stats['std'], 
                                          stats['mean'] + 2*stats['std']),
                            suggested_action="Investigate recent changes to circuit or environment"
                        )
                        self.alerts.append(alert)
                        self._trigger_alert_callbacks(alert)
    
    def _generate_predictions(self):
        """Generate predictive health assessments."""
        # Simple trend analysis
        for metric_name, history in self.metrics_history.items():
            if len(history) >= 20:  # Minimum data for prediction
                values = [m.value for m in list(history)[-20:]]  # Last 20 samples
                timestamps = [m.timestamp for m in list(history)[-20:]]
                
                # Linear trend analysis
                trend = self._calculate_trend(timestamps, values)
                
                if abs(trend) > 0.01:  # Significant trend
                    direction = "declining" if trend < 0 else "improving"
                    
                    # Predict future value
                    future_time = timestamps[-1] + 300  # 5 minutes ahead
                    predicted_value = values[-1] + trend * 300
                    
                    # Generate prediction alert if concerning
                    if ((trend < 0 and metric_name in ['fidelity', 'success_probability']) or
                        (trend > 0 and metric_name in ['error_probability'])):
                        
                        alert = HealthAlert(
                            severity='warning',
                            message=f"Predictive trend analysis: {metric_name} is {direction}",
                            timestamp=time.time(),
                            metric_name=metric_name,
                            current_value=values[-1],
                            expected_range=(None, None),
                            suggested_action=f"Monitor {metric_name} closely, predicted value: {predicted_value:.3f}"
                        )
                        self.alerts.append(alert)
    
    def _calculate_trend(self, timestamps: List[float], values: List[float]) -> float:
        """Calculate linear trend (slope) of values over time."""
        if len(timestamps) < 2:
            return 0.0
        
        n = len(timestamps)
        sum_t = sum(timestamps)
        sum_v = sum(values)
        sum_tv = sum(t * v for t, v in zip(timestamps, values))
        sum_t2 = sum(t * t for t in timestamps)
        
        # Linear regression slope
        denominator = n * sum_t2 - sum_t * sum_t
        if denominator == 0:
            return 0.0
        
        slope = (n * sum_tv - sum_t * sum_v) / denominator
        return slope
    
    def add_alert_callback(self, callback: Callable[[HealthAlert], None]):
        """Add callback function for health alerts."""
        self._callbacks.append(callback)
    
    def _trigger_alert_callbacks(self, alert: HealthAlert):
        """Trigger all registered alert callbacks."""
        for callback in self._callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Error in alert callback: {e}")
    
    def get_current_health_score(self) -> float:
        """Calculate overall health score (0-1, higher is better)."""
        if not self.current_metrics:
            return 0.5  # Unknown health
        
        critical_metrics = [
            m for m in self.current_metrics.values() 
            if m.is_critical
        ]
        
        if not critical_metrics:
            return 0.8  # Good default if no critical metrics
        
        # Weight critical metrics more heavily
        scores = []
        for metric in critical_metrics:
            if metric.name in ['fidelity', 'success_probability']:
                scores.append(metric.value)
            elif metric.name in ['error_probability']:
                scores.append(1.0 - metric.value)
        
        return np.mean(scores) if scores else 0.5
    
    def get_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report."""
        overall_score = self.get_current_health_score()
        
        recent_alerts = [
            alert for alert in self.alerts 
            if time.time() - alert.timestamp < 3600  # Last hour
        ]
        
        critical_alerts = [
            alert for alert in recent_alerts 
            if alert.severity == 'critical'
        ]
        
        return {
            'timestamp': time.time(),
            'overall_health_score': overall_score,
            'health_status': self._classify_health_status(overall_score),
            'current_metrics': {
                name: {
                    'value': metric.value,
                    'timestamp': metric.timestamp,
                    'is_critical': metric.is_critical
                }
                for name, metric in self.current_metrics.items()
            },
            'recent_alerts_count': len(recent_alerts),
            'critical_alerts_count': len(critical_alerts),
            'monitoring_active': self._monitoring_active,
            'recommendations': self._generate_health_recommendations(overall_score, recent_alerts)
        }
    
    def _classify_health_status(self, score: float) -> str:
        """Classify health status based on score."""
        if score >= 0.9:
            return 'excellent'
        elif score >= 0.8:
            return 'good'
        elif score >= 0.7:
            return 'fair'
        elif score >= 0.5:
            return 'poor'
        else:
            return 'critical'
    
    def _generate_health_recommendations(self, score: float, alerts: List[HealthAlert]) -> List[str]:
        """Generate health improvement recommendations."""
        recommendations = []
        
        if score < 0.7:
            recommendations.append("Consider reducing circuit complexity")
            recommendations.append("Implement error correction if not already present")
        
        if score < 0.5:
            recommendations.append("Circuit requires immediate attention")
            recommendations.append("Review quantum algorithm implementation")
        
        # Alert-based recommendations
        alert_types = set(alert.metric_name for alert in alerts)
        if 'fidelity' in alert_types:
            recommendations.append("Improve quantum gate fidelity")
        if 'error_probability' in alert_types:
            recommendations.append("Enhance error mitigation strategies")
        
        return list(set(recommendations))  # Remove duplicates
    
    def cleanup(self):
        """Clean up resources."""
        self.stop_monitoring()
        self._circuit_cache.clear()
        self.alerts.clear()
        self.metrics_history.clear()
        self.current_metrics.clear()