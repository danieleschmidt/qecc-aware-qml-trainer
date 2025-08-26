#!/usr/bin/env python3
"""
Real-Time Adaptive Quantum Error Correction System
Dynamic error correction adaptation based on live hardware feedback
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor
import threading

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector, fidelity, partial_trace
from qiskit.providers import Backend, Job
from scipy.optimize import minimize


class AdaptationStrategy(Enum):
    """Error correction adaptation strategies"""
    CONSERVATIVE = "conservative"
    AGGRESSIVE = "aggressive"
    PREDICTIVE = "predictive"
    LEARNING = "learning"


class ErrorThreshold(Enum):
    """Error threshold levels for adaptation triggers"""
    LOW = 1e-3
    MEDIUM = 1e-2
    HIGH = 1e-1
    CRITICAL = 2e-1


@dataclass
class NoiseProfile:
    """Real-time noise characterization"""
    gate_errors: Dict[str, float] = field(default_factory=dict)
    readout_errors: Dict[int, float] = field(default_factory=dict)
    coherence_times: Dict[int, Tuple[float, float]] = field(default_factory=dict)  # (T1, T2)
    crosstalk_matrix: Optional[np.ndarray] = None
    error_correlation: Optional[np.ndarray] = None
    temporal_drift: float = 0.0
    last_calibration: float = 0.0
    
    def effective_error_rate(self) -> float:
        """Calculate effective combined error rate"""
        if not self.gate_errors:
            return 0.1  # Default assumption
        
        return np.mean(list(self.gate_errors.values()))


@dataclass 
class AdaptationDecision:
    """Decision made by adaptive QECC system"""
    timestamp: float
    trigger_reason: str
    old_code: str
    new_code: str
    expected_improvement: float
    confidence: float
    adaptation_cost: float


class RealTimeAdaptiveQECC:
    """
    Real-time adaptive quantum error correction system that monitors
    hardware performance and automatically adjusts error correction
    strategies based on live feedback.
    """
    
    def __init__(self, backend: Optional[Backend] = None,
                 strategy: AdaptationStrategy = AdaptationStrategy.PREDICTIVE,
                 adaptation_threshold: ErrorThreshold = ErrorThreshold.MEDIUM,
                 monitoring_interval: float = 5.0):
        
        self.backend = backend
        self.strategy = strategy
        self.adaptation_threshold = adaptation_threshold.value
        self.monitoring_interval = monitoring_interval
        
        # State management
        self.current_noise_profile = NoiseProfile()
        self.adaptation_history: List[AdaptationDecision] = []
        self.performance_metrics: Dict[str, List[float]] = {
            "fidelity": [],
            "error_rate": [],
            "execution_time": [],
            "success_rate": []
        }
        
        # Threading and async management
        self._monitoring_active = False
        self._adaptation_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=3)
        
        # Learning system
        self._prediction_model = None
        self._adaptation_weights = {
            "fidelity_impact": 0.4,
            "execution_cost": 0.2,
            "implementation_complexity": 0.2,
            "hardware_compatibility": 0.2
        }
        
        # Available error correction codes
        self._available_codes = {
            "surface_3": {"distance": 3, "qubits": 17, "threshold": 1e-2},
            "surface_5": {"distance": 5, "qubits": 41, "threshold": 1e-2},
            "color_3": {"distance": 3, "qubits": 17, "threshold": 8e-3},
            "steane": {"distance": 3, "qubits": 7, "threshold": 5e-3},
            "repetition_3": {"distance": 3, "qubits": 3, "threshold": 2e-2}
        }
        
        self.current_code = "surface_3"
        self.logger = logging.getLogger(__name__)
    
    async def start_monitoring(self) -> None:
        """Start real-time monitoring and adaptation"""
        if self._monitoring_active:
            self.logger.warning("Monitoring already active")
            return
        
        self._monitoring_active = True
        self.logger.info("Starting real-time adaptive QECC monitoring")
        
        # Start monitoring tasks
        monitoring_task = asyncio.create_task(self._monitoring_loop())
        analysis_task = asyncio.create_task(self._analysis_loop())
        adaptation_task = asyncio.create_task(self._adaptation_loop())
        
        try:
            await asyncio.gather(monitoring_task, analysis_task, adaptation_task)
        except Exception as e:
            self.logger.error(f"Monitoring failed: {e}")
        finally:
            self._monitoring_active = False
    
    def stop_monitoring(self) -> None:
        """Stop real-time monitoring"""
        self._monitoring_active = False
        self.logger.info("Stopping real-time adaptive QECC monitoring")
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop for hardware metrics"""
        while self._monitoring_active:
            try:
                # Collect hardware metrics
                noise_profile = await self._collect_noise_profile()
                
                # Update current profile
                with self._adaptation_lock:
                    self.current_noise_profile = noise_profile
                
                # Log performance metrics
                effective_error_rate = noise_profile.effective_error_rate()
                self.performance_metrics["error_rate"].append(effective_error_rate)
                
                self.logger.debug(f"Noise profile updated: error_rate={effective_error_rate:.4f}")
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
            
            await asyncio.sleep(self.monitoring_interval)
    
    async def _analysis_loop(self) -> None:
        """Analysis loop for trend detection and prediction"""
        while self._monitoring_active:
            try:
                await self._analyze_performance_trends()
                await self._update_prediction_model()
                
            except Exception as e:
                self.logger.error(f"Analysis loop error: {e}")
            
            await asyncio.sleep(self.monitoring_interval * 2)
    
    async def _adaptation_loop(self) -> None:
        """Decision and adaptation loop"""
        while self._monitoring_active:
            try:
                # Evaluate need for adaptation
                adaptation_needed = await self._evaluate_adaptation_need()
                
                if adaptation_needed:
                    optimal_code = await self._select_optimal_code()
                    
                    if optimal_code != self.current_code:
                        await self._execute_adaptation(optimal_code)
                
            except Exception as e:
                self.logger.error(f"Adaptation loop error: {e}")
            
            await asyncio.sleep(self.monitoring_interval * 3)
    
    async def _collect_noise_profile(self) -> NoiseProfile:
        """Collect current hardware noise characteristics"""
        profile = NoiseProfile()
        profile.last_calibration = time.time()
        
        if self.backend is None:
            # Simulate noise profile for testing
            profile.gate_errors = {
                "cx": np.random.normal(5e-3, 1e-3),
                "x": np.random.normal(1e-3, 2e-4),
                "rz": np.random.normal(5e-4, 1e-4)
            }
            profile.readout_errors = {
                i: np.random.normal(2e-2, 5e-3) 
                for i in range(5)
            }
            profile.coherence_times = {
                i: (
                    np.random.normal(50e-6, 10e-6),  # T1
                    np.random.normal(70e-6, 15e-6)   # T2
                )
                for i in range(5)
            }
        else:
            # Collect from real backend
            properties = self.backend.properties()
            if properties:
                # Extract gate error rates
                for gate in properties.gates:
                    error_rate = gate.parameters[0].value if gate.parameters else 0.001
                    profile.gate_errors[gate.gate] = error_rate
                
                # Extract readout errors
                for i, qubit_props in enumerate(properties.qubits):
                    readout_error = next(
                        (param.value for param in qubit_props if param.name == "readout_error"),
                        0.01
                    )
                    profile.readout_errors[i] = readout_error
                
                # Extract coherence times  
                for i, qubit_props in enumerate(properties.qubits):
                    t1 = next((param.value for param in qubit_props if param.name == "T1"), 50e-6)
                    t2 = next((param.value for param in qubit_props if param.name == "T2"), 70e-6)
                    profile.coherence_times[i] = (t1, t2)
        
        return profile
    
    async def _analyze_performance_trends(self) -> None:
        """Analyze performance trends and detect degradation"""
        if len(self.performance_metrics["error_rate"]) < 5:
            return
        
        # Simple trend analysis using recent data
        recent_errors = self.performance_metrics["error_rate"][-10:]
        error_trend = np.polyfit(range(len(recent_errors)), recent_errors, 1)[0]
        
        # Check for degradation trend
        if error_trend > 1e-4:  # Error rate increasing
            self.logger.warning(f"Performance degradation detected: trend={error_trend:.6f}")
        
        # Calculate performance volatility
        error_volatility = np.std(recent_errors) if len(recent_errors) > 2 else 0
        
        self.logger.debug(f"Performance analysis: trend={error_trend:.6f}, volatility={error_volatility:.6f}")
    
    async def _update_prediction_model(self) -> None:
        """Update predictive model for error correction effectiveness"""
        if len(self.adaptation_history) < 3:
            return
        
        # Simple ML-based prediction model
        # In practice, this would use more sophisticated ML techniques
        
        try:
            # Extract features from adaptation history
            features = []
            targets = []
            
            for decision in self.adaptation_history[-20:]:  # Recent adaptations
                feature = [
                    decision.adaptation_cost,
                    decision.confidence,
                    decision.expected_improvement
                ]
                features.append(feature)
                
                # Target is actual improvement (would be measured in practice)
                target = decision.expected_improvement * np.random.uniform(0.8, 1.2)
                targets.append(target)
            
            if len(features) >= 5:
                # Simple linear regression for prediction
                X = np.array(features)
                y = np.array(targets)
                
                # Update prediction weights (simplified)
                correlation = np.corrcoef(X.T, y)[:-1, -1]
                self._adaptation_weights = {
                    "fidelity_impact": abs(correlation[0]) if not np.isnan(correlation[0]) else 0.4,
                    "execution_cost": abs(correlation[1]) if not np.isnan(correlation[1]) else 0.2,
                    "implementation_complexity": abs(correlation[2]) if not np.isnan(correlation[2]) else 0.2,
                    "hardware_compatibility": 0.2
                }
                
                # Normalize weights
                total_weight = sum(self._adaptation_weights.values())
                self._adaptation_weights = {
                    k: v / total_weight for k, v in self._adaptation_weights.items()
                }
                
                self.logger.debug(f"Updated adaptation weights: {self._adaptation_weights}")
        
        except Exception as e:
            self.logger.warning(f"Prediction model update failed: {e}")
    
    async def _evaluate_adaptation_need(self) -> bool:
        """Evaluate whether adaptation is needed based on current metrics"""
        if len(self.performance_metrics["error_rate"]) < 3:
            return False
        
        current_error_rate = self.performance_metrics["error_rate"][-1]
        
        # Check threshold-based triggers
        if current_error_rate > self.adaptation_threshold:
            self.logger.info(f"Adaptation triggered: error_rate={current_error_rate:.6f} > threshold={self.adaptation_threshold:.6f}")
            return True
        
        # Check trend-based triggers
        if len(self.performance_metrics["error_rate"]) >= 5:
            recent_trend = np.polyfit(
                range(5), 
                self.performance_metrics["error_rate"][-5:], 
                1
            )[0]
            
            if recent_trend > self.adaptation_threshold * 0.1:  # 10% of threshold as trend trigger
                self.logger.info(f"Adaptation triggered: trend={recent_trend:.6f}")
                return True
        
        return False
    
    async def _select_optimal_code(self) -> str:
        """Select optimal error correction code based on current conditions"""
        current_error_rate = self.current_noise_profile.effective_error_rate()
        
        # Evaluate each available code
        code_scores = {}
        
        for code_name, code_info in self._available_codes.items():
            score = await self._evaluate_code_effectiveness(code_name, code_info, current_error_rate)
            code_scores[code_name] = score
        
        # Select code with highest score
        optimal_code = max(code_scores.keys(), key=lambda k: code_scores[k])
        
        self.logger.info(f"Code evaluation scores: {code_scores}")
        self.logger.info(f"Selected optimal code: {optimal_code}")
        
        return optimal_code
    
    async def _evaluate_code_effectiveness(self, code_name: str, code_info: Dict[str, Any], 
                                         error_rate: float) -> float:
        """Evaluate effectiveness of specific error correction code"""
        
        # Base effectiveness based on threshold comparison
        threshold_score = 1.0 if error_rate < code_info["threshold"] else code_info["threshold"] / error_rate
        
        # Resource efficiency (fewer qubits better for limited hardware)
        max_qubits = 50  # Assume hardware limit
        resource_score = max(0.1, (max_qubits - code_info["qubits"]) / max_qubits)
        
        # Distance effectiveness (higher distance better for high error rates) 
        distance_score = code_info["distance"] / 5.0  # Normalize by max distance
        
        # Implementation complexity (lower is better)
        complexity_scores = {
            "repetition_3": 1.0,
            "steane": 0.8,
            "surface_3": 0.6,
            "color_3": 0.5,
            "surface_5": 0.3
        }
        complexity_score = complexity_scores.get(code_name, 0.5)
        
        # Weighted combination
        total_score = (
            threshold_score * self._adaptation_weights["fidelity_impact"] +
            resource_score * self._adaptation_weights["execution_cost"] +
            distance_score * self._adaptation_weights["hardware_compatibility"] +
            complexity_score * self._adaptation_weights["implementation_complexity"]
        )
        
        return total_score
    
    async def _execute_adaptation(self, new_code: str) -> None:
        """Execute adaptation to new error correction code"""
        old_code = self.current_code
        
        try:
            # Calculate adaptation metrics
            adaptation_cost = self._calculate_adaptation_cost(old_code, new_code)
            expected_improvement = await self._predict_improvement(old_code, new_code)
            confidence = self._calculate_confidence(new_code)
            
            # Create adaptation decision record
            decision = AdaptationDecision(
                timestamp=time.time(),
                trigger_reason="automatic_optimization",
                old_code=old_code,
                new_code=new_code,
                expected_improvement=expected_improvement,
                confidence=confidence,
                adaptation_cost=adaptation_cost
            )
            
            # Execute adaptation if confidence is sufficient
            if confidence > 0.6:  # Minimum confidence threshold
                with self._adaptation_lock:
                    self.current_code = new_code
                
                self.adaptation_history.append(decision)
                
                self.logger.info(
                    f"Executed adaptation: {old_code} -> {new_code} "
                    f"(improvement: {expected_improvement:.3f}, confidence: {confidence:.3f})"
                )
            else:
                self.logger.info(
                    f"Adaptation rejected due to low confidence: {confidence:.3f} < 0.6"
                )
                
        except Exception as e:
            self.logger.error(f"Adaptation execution failed: {e}")
    
    def _calculate_adaptation_cost(self, old_code: str, new_code: str) -> float:
        """Calculate cost of adaptation between codes"""
        old_info = self._available_codes.get(old_code, {"qubits": 10})
        new_info = self._available_codes.get(new_code, {"qubits": 10})
        
        # Cost based on qubit count difference and complexity
        qubit_cost = abs(new_info["qubits"] - old_info["qubits"]) / 50.0
        
        # Implementation change cost (simple approximation)
        complexity_cost = 0.1 if old_code != new_code else 0.0
        
        return qubit_cost + complexity_cost
    
    async def _predict_improvement(self, old_code: str, new_code: str) -> float:
        """Predict performance improvement from code change"""
        current_error_rate = self.current_noise_profile.effective_error_rate()
        
        old_threshold = self._available_codes.get(old_code, {}).get("threshold", 1e-2)
        new_threshold = self._available_codes.get(new_code, {}).get("threshold", 1e-2)
        
        # Simple improvement prediction based on threshold comparison
        if current_error_rate > old_threshold and current_error_rate < new_threshold:
            improvement = (old_threshold - new_threshold) / old_threshold
        else:
            improvement = max(0, (new_threshold - current_error_rate) / new_threshold)
        
        return min(1.0, improvement)
    
    def _calculate_confidence(self, code: str) -> float:
        """Calculate confidence in code selection"""
        # Base confidence on historical performance
        base_confidence = 0.7
        
        # Adjust based on available data
        if len(self.adaptation_history) > 5:
            # Calculate success rate of recent adaptations
            recent_adaptations = self.adaptation_history[-5:]
            success_rate = sum(
                1 for decision in recent_adaptations 
                if decision.expected_improvement > 0.1
            ) / len(recent_adaptations)
            
            base_confidence = 0.5 + 0.5 * success_rate
        
        # Adjust based on hardware compatibility
        current_error_rate = self.current_noise_profile.effective_error_rate()
        code_threshold = self._available_codes.get(code, {}).get("threshold", 1e-2)
        
        if current_error_rate < code_threshold:
            hardware_confidence = min(1.0, code_threshold / current_error_rate)
        else:
            hardware_confidence = 0.5
        
        return (base_confidence + hardware_confidence) / 2
    
    def get_adaptation_summary(self) -> Dict[str, Any]:
        """Get summary of adaptation system status"""
        return {
            "monitoring_active": self._monitoring_active,
            "current_code": self.current_code,
            "current_error_rate": self.current_noise_profile.effective_error_rate(),
            "adaptations_count": len(self.adaptation_history),
            "recent_adaptations": self.adaptation_history[-3:] if self.adaptation_history else [],
            "performance_trend": self._get_performance_trend(),
            "adaptation_weights": self._adaptation_weights
        }
    
    def _get_performance_trend(self) -> str:
        """Get current performance trend"""
        if len(self.performance_metrics["error_rate"]) < 3:
            return "insufficient_data"
        
        recent_errors = self.performance_metrics["error_rate"][-5:]
        trend = np.polyfit(range(len(recent_errors)), recent_errors, 1)[0]
        
        if abs(trend) < 1e-5:
            return "stable"
        elif trend > 0:
            return "degrading"
        else:
            return "improving"


# Utility function for testing
async def demo_adaptive_qecc():
    """Demonstration of real-time adaptive QECC system"""
    print("🔄 Starting Real-Time Adaptive QECC Demo")
    
    # Initialize system
    adaptive_qecc = RealTimeAdaptiveQECC(
        backend=None,  # Use simulation
        strategy=AdaptationStrategy.PREDICTIVE,
        adaptation_threshold=ErrorThreshold.MEDIUM,
        monitoring_interval=2.0
    )
    
    # Start monitoring for short demo
    monitoring_task = asyncio.create_task(adaptive_qecc.start_monitoring())
    
    # Let it run for 20 seconds
    await asyncio.sleep(20)
    
    # Stop monitoring
    adaptive_qecc.stop_monitoring()
    
    # Cancel monitoring task
    monitoring_task.cancel()
    
    try:
        await monitoring_task
    except asyncio.CancelledError:
        pass
    
    # Print summary
    summary = adaptive_qecc.get_adaptation_summary()
    print("\n📊 Adaptive QECC Summary:")
    print(f"Current Code: {summary['current_code']}")
    print(f"Error Rate: {summary['current_error_rate']:.6f}")
    print(f"Adaptations Made: {summary['adaptations_count']}")
    print(f"Performance Trend: {summary['performance_trend']}")
    
    return summary


if __name__ == "__main__":
    asyncio.run(demo_adaptive_qecc())