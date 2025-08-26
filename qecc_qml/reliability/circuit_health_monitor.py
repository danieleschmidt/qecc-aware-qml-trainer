#!/usr/bin/env python3
"""
Advanced Circuit Health Monitoring and Self-Healing System
Real-time quantum circuit health monitoring with autonomous recovery
"""

import asyncio
import logging
import time
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Callable
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import json

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector, fidelity
from qiskit.providers import Backend, Job
from scipy.stats import entropy
from scipy.optimize import minimize


class HealthStatus(Enum):
    """Circuit health status levels"""
    HEALTHY = "healthy"
    WARNING = "warning"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    FAILED = "failed"


class RecoveryAction(Enum):
    """Self-healing recovery actions"""
    NONE = "none"
    RECALIBRATE = "recalibrate"
    SWITCH_BACKEND = "switch_backend"
    REDUCE_COMPLEXITY = "reduce_complexity"
    EMERGENCY_STOP = "emergency_stop"


@dataclass
class HealthMetrics:
    """Comprehensive health metrics for quantum circuits"""
    timestamp: float
    fidelity_score: float = 0.0
    error_rate: float = 1.0
    coherence_score: float = 0.0
    gate_success_rate: float = 0.0
    measurement_accuracy: float = 0.0
    circuit_depth_efficiency: float = 0.0
    resource_utilization: float = 0.0
    
    # Advanced metrics
    entropy_measure: float = 0.0
    noise_floor: float = 0.0
    temporal_stability: float = 0.0
    cross_talk_level: float = 0.0
    
    def overall_health_score(self) -> float:
        """Calculate weighted overall health score"""
        weights = {
            'fidelity_score': 0.25,
            'gate_success_rate': 0.20,
            'coherence_score': 0.15,
            'measurement_accuracy': 0.15,
            'circuit_depth_efficiency': 0.10,
            'resource_utilization': 0.10,
            'temporal_stability': 0.05
        }
        
        return sum(getattr(self, metric) * weight for metric, weight in weights.items())


@dataclass
class HealthAlert:
    """Health monitoring alert"""
    timestamp: float
    severity: HealthStatus
    component: str
    message: str
    metrics: Dict[str, float]
    recommended_action: RecoveryAction
    auto_fixable: bool = False


@dataclass
class RecoveryPlan:
    """Automated recovery plan"""
    plan_id: str
    trigger_condition: str
    actions: List[RecoveryAction]
    expected_improvement: float
    execution_time_estimate: float
    success_probability: float
    rollback_plan: Optional[List[RecoveryAction]] = None


class AdvancedCircuitHealthMonitor:
    """
    Advanced quantum circuit health monitoring system with real-time
    diagnostics, predictive analytics, and automated self-healing.
    """
    
    def __init__(self, 
                 backend: Optional[Backend] = None,
                 monitoring_interval: float = 5.0,
                 history_size: int = 1000,
                 enable_auto_healing: bool = True):
        
        self.backend = backend
        self.monitoring_interval = monitoring_interval
        self.history_size = history_size
        self.enable_auto_healing = enable_auto_healing
        
        # Health monitoring state
        self.health_history: deque = deque(maxlen=history_size)
        self.current_health: Optional[HealthMetrics] = None
        self.health_alerts: List[HealthAlert] = []
        self.recovery_plans: Dict[str, RecoveryPlan] = {}
        
        # Monitoring status
        self._monitoring_active = False
        self._monitoring_thread: Optional[threading.Thread] = None
        self._health_lock = threading.Lock()
        
        # Performance baselines and thresholds
        self.health_thresholds = {
            "fidelity_min": 0.95,
            "error_rate_max": 0.05,
            "coherence_min": 0.90,
            "gate_success_min": 0.98,
            "measurement_accuracy_min": 0.97
        }
        
        # Predictive models and trend analysis
        self.trend_window = 50
        self.prediction_horizon = 10
        self._trend_models: Dict[str, Any] = {}
        
        # Auto-healing configuration
        self.recovery_strategies = {
            HealthStatus.WARNING: [RecoveryAction.RECALIBRATE],
            HealthStatus.DEGRADED: [RecoveryAction.RECALIBRATE, RecoveryAction.REDUCE_COMPLEXITY],
            HealthStatus.CRITICAL: [RecoveryAction.SWITCH_BACKEND, RecoveryAction.EMERGENCY_STOP],
            HealthStatus.FAILED: [RecoveryAction.EMERGENCY_STOP]
        }
        
        # Circuit registry for monitoring
        self.monitored_circuits: Dict[str, QuantumCircuit] = {}
        self.circuit_baselines: Dict[str, HealthMetrics] = {}
        
        self.logger = logging.getLogger(__name__)
        self._setup_recovery_plans()
    
    def add_circuit_for_monitoring(self, circuit_id: str, circuit: QuantumCircuit) -> None:
        """Add quantum circuit to health monitoring"""
        self.monitored_circuits[circuit_id] = circuit
        
        # Establish baseline health metrics
        baseline = self._establish_baseline(circuit)
        self.circuit_baselines[circuit_id] = baseline
        
        self.logger.info(f"Added circuit {circuit_id} for monitoring")
    
    def start_monitoring(self) -> None:
        """Start real-time health monitoring"""
        if self._monitoring_active:
            self.logger.warning("Health monitoring already active")
            return
        
        self._monitoring_active = True
        self._monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitoring_thread.start()
        
        self.logger.info("Started advanced circuit health monitoring")
    
    def stop_monitoring(self) -> None:
        """Stop health monitoring"""
        self._monitoring_active = False
        
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=10)
        
        self.logger.info("Stopped circuit health monitoring")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        while self._monitoring_active:
            try:
                start_time = time.time()
                
                # Collect health metrics
                health_metrics = self._collect_comprehensive_health_metrics()
                
                # Update health state
                with self._health_lock:
                    self.current_health = health_metrics
                    self.health_history.append(health_metrics)
                
                # Analyze health trends and patterns
                health_status = self._analyze_health_status(health_metrics)
                
                # Generate alerts if needed
                alerts = self._generate_health_alerts(health_metrics, health_status)
                self.health_alerts.extend(alerts)
                
                # Trigger auto-healing if enabled
                if self.enable_auto_healing and health_status in [HealthStatus.WARNING, HealthStatus.DEGRADED, HealthStatus.CRITICAL]:
                    asyncio.run(self._execute_auto_healing(health_status, health_metrics))
                
                # Predictive analysis
                self._update_trend_models()
                
                # Log health status
                execution_time = time.time() - start_time
                self.logger.debug(
                    f"Health check completed: status={health_status.value}, "
                    f"score={health_metrics.overall_health_score():.3f}, "
                    f"time={execution_time:.3f}s"
                )
                
                # Wait for next monitoring cycle
                sleep_time = max(0, self.monitoring_interval - execution_time)
                time.sleep(sleep_time)
                
            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
                time.sleep(self.monitoring_interval)
    
    def _collect_comprehensive_health_metrics(self) -> HealthMetrics:
        """Collect comprehensive health metrics from quantum system"""
        metrics = HealthMetrics(timestamp=time.time())
        
        try:
            if self.backend:
                # Collect from real backend
                properties = self.backend.properties()
                if properties:
                    # Gate success rates
                    gate_errors = []
                    for gate in properties.gates:
                        if gate.parameters:
                            error_rate = gate.parameters[0].value
                            gate_errors.append(1.0 - error_rate)  # Convert to success rate
                    
                    metrics.gate_success_rate = np.mean(gate_errors) if gate_errors else 0.95
                    metrics.error_rate = 1.0 - metrics.gate_success_rate
                    
                    # Readout fidelity
                    readout_errors = []
                    for qubit_props in properties.qubits:
                        readout_error = next(
                            (param.value for param in qubit_props if param.name == "readout_error"),
                            0.01
                        )
                        readout_errors.append(1.0 - readout_error)
                    
                    metrics.measurement_accuracy = np.mean(readout_errors) if readout_errors else 0.98
                    
                    # Coherence metrics
                    t1_times = []
                    t2_times = []
                    for qubit_props in properties.qubits:
                        t1 = next((param.value for param in qubit_props if param.name == "T1"), 50e-6)
                        t2 = next((param.value for param in qubit_props if param.name == "T2"), 70e-6)
                        t1_times.append(t1)
                        t2_times.append(t2)
                    
                    # Normalize coherence scores
                    avg_t1 = np.mean(t1_times) if t1_times else 50e-6
                    avg_t2 = np.mean(t2_times) if t2_times else 70e-6
                    metrics.coherence_score = min(1.0, (avg_t1 + avg_t2) / 120e-6)
            
            else:
                # Simulate metrics for testing
                metrics.gate_success_rate = np.random.normal(0.985, 0.01)
                metrics.error_rate = 1.0 - metrics.gate_success_rate
                metrics.measurement_accuracy = np.random.normal(0.98, 0.005)
                metrics.coherence_score = np.random.normal(0.92, 0.03)
            
            # Calculate derived metrics
            if self.monitored_circuits:
                metrics.circuit_depth_efficiency = self._calculate_depth_efficiency()
                metrics.resource_utilization = self._calculate_resource_utilization()
                metrics.fidelity_score = self._estimate_circuit_fidelity()
            
            # Advanced metrics
            metrics.entropy_measure = self._calculate_system_entropy()
            metrics.noise_floor = self._estimate_noise_floor()
            metrics.temporal_stability = self._calculate_temporal_stability()
            metrics.cross_talk_level = self._estimate_crosstalk()
            
            # Ensure metrics are in valid ranges
            metrics = self._normalize_metrics(metrics)
            
        except Exception as e:
            self.logger.warning(f"Error collecting health metrics: {e}")
            # Return default metrics
            metrics.gate_success_rate = 0.90
            metrics.error_rate = 0.10
            
        return metrics
    
    def _calculate_depth_efficiency(self) -> float:
        """Calculate circuit depth efficiency"""
        if not self.monitored_circuits:
            return 1.0
        
        efficiencies = []
        for circuit_id, circuit in self.monitored_circuits.items():
            # Simple efficiency metric: gates per depth
            actual_depth = circuit.depth()
            ideal_depth = circuit.size()  # All gates in parallel (theoretical minimum)
            
            if actual_depth > 0:
                efficiency = min(1.0, ideal_depth / actual_depth)
                efficiencies.append(efficiency)
        
        return np.mean(efficiencies) if efficiencies else 1.0
    
    def _calculate_resource_utilization(self) -> float:
        """Calculate quantum resource utilization efficiency"""
        if not self.monitored_circuits:
            return 1.0
        
        utilizations = []
        for circuit in self.monitored_circuits.values():
            # Calculate utilization as active qubits / total qubits
            active_qubits = circuit.num_qubits
            total_available = self.backend.configuration().num_qubits if self.backend else 10
            
            utilization = active_qubits / total_available
            utilizations.append(min(1.0, utilization))
        
        return np.mean(utilizations) if utilizations else 0.5
    
    def _estimate_circuit_fidelity(self) -> float:
        """Estimate overall circuit fidelity"""
        # Simplified fidelity estimation based on gate count and error rates
        if not self.current_health:
            return 0.90
        
        gate_fidelity = self.current_health.gate_success_rate
        measurement_fidelity = self.current_health.measurement_accuracy
        
        # Estimate compound fidelity
        total_gates = sum(circuit.size() for circuit in self.monitored_circuits.values())
        if total_gates == 0:
            return 1.0
        
        # Exponential decay model for fidelity
        compound_gate_fidelity = gate_fidelity ** total_gates
        overall_fidelity = compound_gate_fidelity * measurement_fidelity
        
        return min(1.0, max(0.0, overall_fidelity))
    
    def _calculate_system_entropy(self) -> float:
        """Calculate system entropy as measure of randomness/disorder"""
        if len(self.health_history) < 5:
            return 0.5
        
        # Calculate entropy of recent error rate distribution
        recent_errors = [h.error_rate for h in list(self.health_history)[-10:]]
        
        # Discretize into bins
        bins = np.histogram(recent_errors, bins=5)[0]
        
        # Calculate entropy
        probabilities = bins / np.sum(bins) if np.sum(bins) > 0 else np.ones(len(bins)) / len(bins)
        probabilities = probabilities[probabilities > 0]  # Remove zero probabilities
        
        return entropy(probabilities) if len(probabilities) > 1 else 0.5
    
    def _estimate_noise_floor(self) -> float:
        """Estimate system noise floor"""
        if len(self.health_history) < 5:
            return 0.01
        
        # Use minimum error rate over recent history as noise floor estimate
        recent_errors = [h.error_rate for h in list(self.health_history)[-20:]]
        return np.percentile(recent_errors, 10)  # 10th percentile as noise floor
    
    def _calculate_temporal_stability(self) -> float:
        """Calculate temporal stability of system performance"""
        if len(self.health_history) < 10:
            return 0.8
        
        # Calculate coefficient of variation for key metrics
        recent_scores = [h.overall_health_score() for h in list(self.health_history)[-20:]]
        
        mean_score = np.mean(recent_scores)
        std_score = np.std(recent_scores)
        
        # Stability = 1 - coefficient_of_variation
        cv = std_score / mean_score if mean_score > 0 else 1
        return max(0, 1 - cv)
    
    def _estimate_crosstalk(self) -> float:
        """Estimate crosstalk level between qubits"""
        # Simplified crosstalk estimation
        # In practice, would analyze correlation between neighboring qubit errors
        return np.random.normal(0.05, 0.01)  # Simulated crosstalk level
    
    def _normalize_metrics(self, metrics: HealthMetrics) -> HealthMetrics:
        """Normalize all metrics to valid ranges [0, 1]"""
        
        # Clamp all metrics to [0, 1] range
        for field_name in ['fidelity_score', 'gate_success_rate', 'coherence_score', 
                          'measurement_accuracy', 'circuit_depth_efficiency', 
                          'resource_utilization', 'temporal_stability']:
            value = getattr(metrics, field_name)
            setattr(metrics, field_name, max(0.0, min(1.0, value)))
        
        # Error rate should be [0, 1] where lower is better
        metrics.error_rate = max(0.0, min(1.0, metrics.error_rate))
        
        # Entropy should be normalized
        metrics.entropy_measure = max(0.0, min(1.0, metrics.entropy_measure))
        
        # Noise floor should be small positive value
        metrics.noise_floor = max(1e-6, min(1.0, metrics.noise_floor))
        
        return metrics
    
    def _analyze_health_status(self, metrics: HealthMetrics) -> HealthStatus:
        """Analyze health metrics to determine overall status"""
        
        # Calculate overall health score
        health_score = metrics.overall_health_score()
        
        # Check critical thresholds
        if metrics.error_rate > 0.20:
            return HealthStatus.FAILED
        if metrics.gate_success_rate < 0.80:
            return HealthStatus.CRITICAL
        if metrics.fidelity_score < 0.85:
            return HealthStatus.CRITICAL
        
        # Check warning thresholds
        if health_score < 0.70:
            return HealthStatus.DEGRADED
        if health_score < 0.85:
            return HealthStatus.WARNING
        
        return HealthStatus.HEALTHY
    
    def _generate_health_alerts(self, metrics: HealthMetrics, status: HealthStatus) -> List[HealthAlert]:
        """Generate health alerts based on metrics and status"""
        alerts = []
        
        # Check individual metric thresholds
        if metrics.fidelity_score < self.health_thresholds["fidelity_min"]:
            alerts.append(HealthAlert(
                timestamp=time.time(),
                severity=HealthStatus.WARNING,
                component="fidelity",
                message=f"Circuit fidelity below threshold: {metrics.fidelity_score:.3f} < {self.health_thresholds['fidelity_min']:.3f}",
                metrics={"fidelity": metrics.fidelity_score},
                recommended_action=RecoveryAction.RECALIBRATE,
                auto_fixable=True
            ))
        
        if metrics.error_rate > self.health_thresholds["error_rate_max"]:
            alerts.append(HealthAlert(
                timestamp=time.time(),
                severity=HealthStatus.WARNING,
                component="error_rate",
                message=f"Error rate above threshold: {metrics.error_rate:.3f} > {self.health_thresholds['error_rate_max']:.3f}",
                metrics={"error_rate": metrics.error_rate},
                recommended_action=RecoveryAction.REDUCE_COMPLEXITY,
                auto_fixable=True
            ))
        
        if metrics.coherence_score < self.health_thresholds["coherence_min"]:
            alerts.append(HealthAlert(
                timestamp=time.time(),
                severity=HealthStatus.DEGRADED,
                component="coherence",
                message=f"Coherence below threshold: {metrics.coherence_score:.3f} < {self.health_thresholds['coherence_min']:.3f}",
                metrics={"coherence": metrics.coherence_score},
                recommended_action=RecoveryAction.SWITCH_BACKEND,
                auto_fixable=False
            ))
        
        return alerts
    
    async def _execute_auto_healing(self, status: HealthStatus, metrics: HealthMetrics) -> bool:
        """Execute automatic healing procedures"""
        try:
            recovery_actions = self.recovery_strategies.get(status, [])
            
            if not recovery_actions:
                self.logger.info(f"No recovery actions defined for status {status.value}")
                return False
            
            self.logger.info(f"Executing auto-healing for status {status.value}")
            
            for action in recovery_actions:
                success = await self._execute_recovery_action(action, metrics)
                if success:
                    self.logger.info(f"Recovery action {action.value} completed successfully")
                    return True
                else:
                    self.logger.warning(f"Recovery action {action.value} failed")
            
            return False
            
        except Exception as e:
            self.logger.error(f"Auto-healing execution failed: {e}")
            return False
    
    async def _execute_recovery_action(self, action: RecoveryAction, metrics: HealthMetrics) -> bool:
        """Execute specific recovery action"""
        
        try:
            if action == RecoveryAction.RECALIBRATE:
                return await self._recalibrate_system()
            
            elif action == RecoveryAction.SWITCH_BACKEND:
                return await self._switch_to_backup_backend()
            
            elif action == RecoveryAction.REDUCE_COMPLEXITY:
                return await self._reduce_circuit_complexity()
            
            elif action == RecoveryAction.EMERGENCY_STOP:
                return await self._emergency_stop()
            
            else:
                self.logger.warning(f"Unknown recovery action: {action.value}")
                return False
                
        except Exception as e:
            self.logger.error(f"Recovery action {action.value} failed: {e}")
            return False
    
    async def _recalibrate_system(self) -> bool:
        """Recalibrate quantum system"""
        self.logger.info("Executing system recalibration")
        
        # Simulate recalibration process
        await asyncio.sleep(2)  # Simulate recalibration time
        
        # Reset thresholds based on current performance
        if self.current_health:
            self.health_thresholds["fidelity_min"] = max(0.90, self.current_health.fidelity_score - 0.02)
            self.health_thresholds["error_rate_max"] = min(0.10, self.current_health.error_rate + 0.01)
        
        return True
    
    async def _switch_to_backup_backend(self) -> bool:
        """Switch to backup quantum backend"""
        self.logger.info("Switching to backup backend")
        
        # Simulate backend switch
        await asyncio.sleep(1)
        
        # In practice, would actually switch to different hardware
        return True
    
    async def _reduce_circuit_complexity(self) -> bool:
        """Reduce circuit complexity to improve reliability"""
        self.logger.info("Reducing circuit complexity")
        
        # Simplify monitored circuits
        for circuit_id, circuit in self.monitored_circuits.items():
            if circuit.depth() > 10:  # Only reduce complex circuits
                # Create simplified version (placeholder logic)
                simplified_circuit = QuantumCircuit(circuit.num_qubits)
                # Add essential gates only (simplified)
                for instruction in circuit.data[:circuit.size()//2]:  # Keep first half of gates
                    simplified_circuit.append(instruction.operation, instruction.qubits)
                
                self.monitored_circuits[circuit_id] = simplified_circuit
                self.logger.info(f"Simplified circuit {circuit_id}: depth {circuit.depth()} -> {simplified_circuit.depth()}")
        
        return True
    
    async def _emergency_stop(self) -> bool:
        """Execute emergency stop procedure"""
        self.logger.critical("Executing emergency stop")
        
        # Stop all quantum operations
        self.monitored_circuits.clear()
        
        # Generate critical alert
        alert = HealthAlert(
            timestamp=time.time(),
            severity=HealthStatus.FAILED,
            component="system",
            message="Emergency stop executed due to critical system failure",
            metrics={},
            recommended_action=RecoveryAction.NONE,
            auto_fixable=False
        )
        self.health_alerts.append(alert)
        
        return True
    
    def _update_trend_models(self) -> None:
        """Update predictive trend models"""
        if len(self.health_history) < self.trend_window:
            return
        
        try:
            # Extract recent health scores for trend analysis
            recent_data = list(self.health_history)[-self.trend_window:]
            timestamps = [h.timestamp for h in recent_data]
            health_scores = [h.overall_health_score() for h in recent_data]
            
            # Simple linear trend model
            time_deltas = [(t - timestamps[0]) for t in timestamps]
            
            if len(time_deltas) > 5:
                # Fit linear trend
                coeffs = np.polyfit(time_deltas, health_scores, 1)
                self._trend_models["health_trend"] = {
                    "slope": coeffs[0],
                    "intercept": coeffs[1],
                    "last_update": time.time()
                }
                
                # Log significant trends
                if abs(coeffs[0]) > 1e-5:  # Significant slope
                    trend_direction = "improving" if coeffs[0] > 0 else "declining"
                    self.logger.info(f"Health trend detected: {trend_direction} (slope: {coeffs[0]:.6f})")
        
        except Exception as e:
            self.logger.debug(f"Trend model update failed: {e}")
    
    def predict_health_trend(self, time_horizon: float = 300.0) -> Optional[Dict[str, float]]:
        """Predict health metrics for future time horizon"""
        if "health_trend" not in self._trend_models:
            return None
        
        try:
            trend_model = self._trend_models["health_trend"]
            current_time = time.time()
            
            # Project trend into future
            predicted_score = trend_model["slope"] * time_horizon + trend_model["intercept"]
            predicted_score = max(0.0, min(1.0, predicted_score))  # Clamp to valid range
            
            return {
                "predicted_health_score": predicted_score,
                "prediction_horizon": time_horizon,
                "confidence": 0.7,  # Static confidence for simplicity
                "trend_slope": trend_model["slope"]
            }
            
        except Exception as e:
            self.logger.warning(f"Health prediction failed: {e}")
            return None
    
    def _establish_baseline(self, circuit: QuantumCircuit) -> HealthMetrics:
        """Establish baseline health metrics for circuit"""
        # Create baseline metrics (in practice, would run actual measurements)
        baseline = HealthMetrics(
            timestamp=time.time(),
            fidelity_score=0.95,
            error_rate=0.02,
            gate_success_rate=0.98,
            measurement_accuracy=0.97,
            coherence_score=0.93,
            circuit_depth_efficiency=1.0,
            resource_utilization=circuit.num_qubits / 10.0
        )
        
        return baseline
    
    def _setup_recovery_plans(self) -> None:
        """Setup predefined recovery plans"""
        
        # Basic recalibration plan
        self.recovery_plans["basic_recalibration"] = RecoveryPlan(
            plan_id="basic_recalibration",
            trigger_condition="fidelity < 0.95",
            actions=[RecoveryAction.RECALIBRATE],
            expected_improvement=0.05,
            execution_time_estimate=60.0,
            success_probability=0.8
        )
        
        # Complexity reduction plan
        self.recovery_plans["complexity_reduction"] = RecoveryPlan(
            plan_id="complexity_reduction",
            trigger_condition="error_rate > 0.10",
            actions=[RecoveryAction.REDUCE_COMPLEXITY, RecoveryAction.RECALIBRATE],
            expected_improvement=0.10,
            execution_time_estimate=30.0,
            success_probability=0.9
        )
        
        # Emergency response plan
        self.recovery_plans["emergency_response"] = RecoveryPlan(
            plan_id="emergency_response",
            trigger_condition="health_score < 0.50",
            actions=[RecoveryAction.EMERGENCY_STOP],
            expected_improvement=0.0,  # Safety measure, not improvement
            execution_time_estimate=5.0,
            success_probability=1.0
        )
    
    def get_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report"""
        current_health = self.current_health
        
        if not current_health:
            return {"status": "no_data", "message": "Health monitoring not yet initialized"}
        
        # Recent alerts
        recent_alerts = [
            {
                "timestamp": alert.timestamp,
                "severity": alert.severity.value,
                "component": alert.component,
                "message": alert.message,
                "recommended_action": alert.recommended_action.value
            }
            for alert in self.health_alerts[-10:]  # Last 10 alerts
        ]
        
        # Health trend prediction
        trend_prediction = self.predict_health_trend()
        
        return {
            "timestamp": current_health.timestamp,
            "overall_health_score": current_health.overall_health_score(),
            "status": self._analyze_health_status(current_health).value,
            "metrics": {
                "fidelity_score": current_health.fidelity_score,
                "error_rate": current_health.error_rate,
                "gate_success_rate": current_health.gate_success_rate,
                "measurement_accuracy": current_health.measurement_accuracy,
                "coherence_score": current_health.coherence_score,
                "temporal_stability": current_health.temporal_stability
            },
            "monitored_circuits": len(self.monitored_circuits),
            "recent_alerts": recent_alerts,
            "trend_prediction": trend_prediction,
            "auto_healing_enabled": self.enable_auto_healing,
            "monitoring_active": self._monitoring_active
        }


# Demonstration function
async def demo_circuit_health_monitor():
    """Demonstrate circuit health monitoring system"""
    print("🏥 Starting Circuit Health Monitor Demo")
    
    # Create health monitor
    monitor = AdvancedCircuitHealthMonitor(
        backend=None,  # Use simulation
        monitoring_interval=2.0,
        enable_auto_healing=True
    )
    
    # Add sample circuits for monitoring
    from qiskit.circuit import Parameter
    
    # Create test circuit
    test_circuit = QuantumCircuit(4)
    theta = Parameter('theta')
    
    for i in range(4):
        test_circuit.ry(theta, i)
    for i in range(3):
        test_circuit.cx(i, i+1)
    
    monitor.add_circuit_for_monitoring("test_circuit_1", test_circuit)
    
    # Start monitoring
    monitor.start_monitoring()
    
    # Let it run for demonstration
    print("📊 Monitoring circuit health for 30 seconds...")
    await asyncio.sleep(30)
    
    # Get health report
    report = monitor.get_health_report()
    
    # Stop monitoring
    monitor.stop_monitoring()
    
    # Display results
    print(f"\n🏥 Health Report:")
    print(f"Overall Health Score: {report['overall_health_score']:.3f}")
    print(f"System Status: {report['status'].upper()}")
    print(f"Fidelity: {report['metrics']['fidelity_score']:.3f}")
    print(f"Error Rate: {report['metrics']['error_rate']:.4f}")
    print(f"Gate Success Rate: {report['metrics']['gate_success_rate']:.3f}")
    
    if report['recent_alerts']:
        print(f"\nRecent Alerts: {len(report['recent_alerts'])}")
        for alert in report['recent_alerts'][-3:]:
            print(f"  - {alert['severity'].upper()}: {alert['message']}")
    
    if report['trend_prediction']:
        trend = report['trend_prediction']
        print(f"\nTrend Prediction:")
        print(f"  Predicted Health Score: {trend['predicted_health_score']:.3f}")
        print(f"  Trend Direction: {'Improving' if trend['trend_slope'] > 0 else 'Declining'}")
    
    return report


if __name__ == "__main__":
    asyncio.run(demo_circuit_health_monitor())