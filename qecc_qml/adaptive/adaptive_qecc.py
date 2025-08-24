"""
Adaptive quantum error correction implementation.

Dynamically selects and adjusts error correction schemes based on:
- Real-time hardware noise characteristics
- Circuit performance metrics
- Resource constraints
- Training objectives
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
from typing import Dict, List, Optional, Union, Any, Callable
from dataclasses import dataclass
from enum import Enum
import logging
from abc import ABC, abstractmethod

from ..core.error_correction import ErrorCorrectionScheme
from ..core.noise_models import NoiseModel
from ..codes.surface_code import SurfaceCode
from ..codes.color_code import ColorCode
from ..codes.steane_code import SteaneCode


class QECCSelectionStrategy(Enum):
    """Strategies for selecting error correction codes."""
    THRESHOLD_BASED = "threshold_based"
    PERFORMANCE_OPTIMIZED = "performance_optimized"
    RESOURCE_CONSTRAINED = "resource_constrained"
    HYBRID_ADAPTIVE = "hybrid_adaptive"


@dataclass
class QECCConfiguration:
    """Configuration for an error correction scheme."""
    scheme: ErrorCorrectionScheme
    distance: int
    logical_qubits: int
    physical_qubits: int
    threshold: float
    overhead: float
    performance_score: float = 0.0


class AdaptiveQECC:
    """
    Adaptive quantum error correction system.
    
    Automatically selects and adjusts error correction schemes based on
    real-time noise characteristics and performance requirements.
    """
    
    def __init__(
        self,
        base_codes: Optional[List[ErrorCorrectionScheme]] = None,
        strategy: QECCSelectionStrategy = QECCSelectionStrategy.THRESHOLD_BASED,
        adaptation_frequency: int = 10,
        performance_window: int = 50,
        min_adaptation_threshold: float = 0.05,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize adaptive QECC system.
        
        Args:
            base_codes: Available error correction codes
            strategy: Selection strategy to use
            adaptation_frequency: How often to check for adaptation (epochs)
            performance_window: Window for performance tracking
            min_adaptation_threshold: Minimum performance change to trigger adaptation
            logger: Optional logger instance
        """
        self.strategy = strategy
        self.adaptation_frequency = adaptation_frequency
        self.performance_window = performance_window
        self.min_adaptation_threshold = min_adaptation_threshold
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize available codes
        if base_codes is None:
            self.available_codes = self._initialize_default_codes()
        else:
            self.available_codes = base_codes
        
        # Current configuration
        self.current_config: Optional[QECCConfiguration] = None
        self.performance_history: List[float] = []
        self.noise_history: List[Dict[str, float]] = []
        self.adaptation_count = 0
        
        # Selection criteria
        self.selection_criteria = self._initialize_criteria()
        
        # Performance metrics
        self.metrics = {
            'fidelity': [],
            'error_rate': [],
            'overhead': [],
            'success_rate': []
        }
    
    def _initialize_default_codes(self) -> List[QECCConfiguration]:
        """Initialize default set of error correction codes."""
        codes = []
        
        # Surface codes with different distances
        for distance in [3, 5, 7]:
            surface_code = SurfaceCode(distance=distance)
            config = QECCConfiguration(
                scheme=surface_code,
                distance=distance,
                logical_qubits=1,
                physical_qubits=surface_code.num_physical_qubits,
                threshold=0.01,  # ~1% for surface codes
                overhead=surface_code.num_physical_qubits
            )
            codes.append(config)
        
        # Color codes
        for distance in [3, 5]:
            color_code = ColorCode(distance=distance)
            config = QECCConfiguration(
                scheme=color_code,
                distance=distance,
                logical_qubits=1,
                physical_qubits=color_code.num_physical_qubits,
                threshold=0.008,  # Slightly lower than surface codes
                overhead=color_code.num_physical_qubits
            )
            codes.append(config)
        
        # Steane code
        steane_code = SteaneCode()
        config = QECCConfiguration(
            scheme=steane_code,
            distance=3,
            logical_qubits=1,
            physical_qubits=7,
            threshold=0.005,  # Lower threshold
            overhead=7
        )
        codes.append(config)
        
        return codes
    
    def _initialize_criteria(self) -> Dict[str, Callable]:
        """Initialize selection criteria functions."""
        return {
            'error_rate_criterion': lambda noise, config: noise['gate_error_rate'] < config.threshold,
            'resource_criterion': lambda resources, config: resources['available_qubits'] >= config.physical_qubits,
            'performance_criterion': lambda perf, config: perf.get('target_fidelity', 0.9) <= config.performance_score,
            'latency_criterion': lambda timing, config: timing.get('max_overhead', 10.0) >= config.overhead
        }
    
    def select_optimal_code(
        self,
        noise_model: NoiseModel,
        resource_constraints: Dict[str, Any],
        performance_requirements: Dict[str, float]
    ) -> QECCConfiguration:
        """
        Select optimal error correction code based on current conditions.
        
        Args:
            noise_model: Current noise characteristics
            resource_constraints: Available resources
            performance_requirements: Required performance metrics
            
        Returns:
            Selected QECC configuration
        """
        if self.strategy == QECCSelectionStrategy.THRESHOLD_BASED:
            return self._select_threshold_based(noise_model, resource_constraints)
        elif self.strategy == QECCSelectionStrategy.PERFORMANCE_OPTIMIZED:
            return self._select_performance_optimized(noise_model, performance_requirements)
        elif self.strategy == QECCSelectionStrategy.RESOURCE_CONSTRAINED:
            return self._select_resource_constrained(resource_constraints)
        elif self.strategy == QECCSelectionStrategy.HYBRID_ADAPTIVE:
            return self._select_hybrid_adaptive(noise_model, resource_constraints, performance_requirements)
        else:
            raise ValueError(f"Unknown selection strategy: {self.strategy}")
    
    def _select_threshold_based(
        self, 
        noise_model: NoiseModel, 
        resource_constraints: Dict[str, Any]
    ) -> QECCConfiguration:
        """Select code based on error rate thresholds."""
        noise_dict = {
            'gate_error_rate': noise_model.gate_error_rate,
            'readout_error_rate': noise_model.readout_error_rate
        }
        
        viable_codes = []
        for config in self.available_codes:
            if (self.selection_criteria['error_rate_criterion'](noise_dict, config) and
                self.selection_criteria['resource_criterion'](resource_constraints, config)):
                viable_codes.append(config)
        
        if not viable_codes:
            self.logger.warning("No viable codes found, using minimal protection")
            return min(self.available_codes, key=lambda c: c.overhead)
        
        # Select code with best distance-to-overhead ratio
        return max(viable_codes, key=lambda c: c.distance / c.overhead)
    
    def _select_performance_optimized(
        self,
        noise_model: NoiseModel,
        performance_requirements: Dict[str, float]
    ) -> QECCConfiguration:
        """Select code optimized for performance."""
        target_fidelity = performance_requirements.get('target_fidelity', 0.95)
        
        # Score codes based on expected performance
        best_config = None
        best_score = -np.inf
        
        for config in self.available_codes:
            # Estimate logical error rate
            logical_error_rate = self._estimate_logical_error_rate(noise_model, config)
            estimated_fidelity = 1 - logical_error_rate
            
            # Performance score considers fidelity vs overhead tradeoff
            if estimated_fidelity >= target_fidelity:
                score = estimated_fidelity / config.overhead
                if score > best_score:
                    best_score = score
                    best_config = config
        
        if best_config is None:
            # Fallback to highest distance code
            return max(self.available_codes, key=lambda c: c.distance)
        
        return best_config
    
    def _select_resource_constrained(
        self,
        resource_constraints: Dict[str, Any]
    ) -> QECCConfiguration:
        """Select code under resource constraints."""
        max_qubits = resource_constraints.get('available_qubits', np.inf)
        max_overhead = resource_constraints.get('max_overhead', np.inf)
        
        viable_codes = [
            config for config in self.available_codes
            if config.physical_qubits <= max_qubits and config.overhead <= max_overhead
        ]
        
        if not viable_codes:
            return min(self.available_codes, key=lambda c: c.physical_qubits)
        
        # Select highest distance code within constraints
        return max(viable_codes, key=lambda c: c.distance)
    
    def _select_hybrid_adaptive(
        self,
        noise_model: NoiseModel,
        resource_constraints: Dict[str, Any],
        performance_requirements: Dict[str, float]
    ) -> QECCConfiguration:
        """Hybrid selection combining multiple criteria."""
        scores = {}
        
        for config in self.available_codes:
            score = 0.0
            
            # Performance score (40% weight)
            logical_error_rate = self._estimate_logical_error_rate(noise_model, config)
            fidelity_score = (1 - logical_error_rate) * 0.4
            
            # Efficiency score (30% weight)
            efficiency_score = (config.distance / config.overhead) * 0.3
            
            # Resource score (20% weight)
            max_qubits = resource_constraints.get('available_qubits', 100)
            resource_score = (1 - config.physical_qubits / max_qubits) * 0.2
            
            # Threshold compliance (10% weight)
            threshold_score = 0.1 if noise_model.gate_error_rate < config.threshold else 0.0
            
            total_score = fidelity_score + efficiency_score + resource_score + threshold_score
            scores[config] = total_score
        
        return max(scores.keys(), key=lambda c: scores[c])
    
    def _estimate_logical_error_rate(
        self,
        noise_model: NoiseModel,
        config: QECCConfiguration
    ) -> float:
        """Estimate logical error rate for a configuration."""
        physical_error_rate = noise_model.gate_error_rate
        
        # Simplified model: logical error rate scales as (p/p_th)^((d+1)/2) for p < p_th
        if physical_error_rate < config.threshold:
            ratio = physical_error_rate / config.threshold
            logical_rate = ratio ** ((config.distance + 1) / 2)
        else:
            # Above threshold, error correction may not help
            logical_rate = physical_error_rate * 0.5  # Some mitigation still occurs
        
        return min(logical_rate, physical_error_rate)  # Can't be worse than uncorrected
    
    def should_adapt(self, current_performance: Dict[str, float]) -> bool:
        """
        Check if adaptation should occur based on performance.
        
        Args:
            current_performance: Current performance metrics
            
        Returns:
            True if adaptation is recommended
        """
        if len(self.performance_history) < self.performance_window:
            return False
        
        # Check if we have enough epochs since last adaptation
        if len(self.performance_history) % self.adaptation_frequency != 0:
            return False
        
        # Calculate performance trend
        recent_performance = np.mean(self.performance_history[-10:])
        historical_performance = np.mean(self.performance_history[-self.performance_window:-10])
        
        performance_change = (recent_performance - historical_performance) / historical_performance
        
        # Adapt if performance is degrading significantly
        return performance_change < -self.min_adaptation_threshold
    
    def adapt(
        self,
        noise_model: NoiseModel,
        resource_constraints: Dict[str, Any],
        performance_requirements: Dict[str, float]
    ) -> bool:
        """
        Perform adaptation if needed.
        
        Args:
            noise_model: Current noise model
            resource_constraints: Available resources
            performance_requirements: Performance requirements
            
        Returns:
            True if adaptation occurred
        """
        new_config = self.select_optimal_code(
            noise_model, resource_constraints, performance_requirements
        )
        
        if (self.current_config is None or 
            new_config.scheme != self.current_config.scheme or
            new_config.distance != self.current_config.distance):
            
            old_config = self.current_config
            self.current_config = new_config
            self.adaptation_count += 1
            
            self.logger.info(
                f"QECC adaptation #{self.adaptation_count}: "
                f"{old_config.scheme.name if old_config else 'None'} -> "
                f"{new_config.scheme.name} (distance {new_config.distance})"
            )
            
            return True
        
        return False
    
    def update_performance(self, metrics: Dict[str, float]):
        """Update performance tracking."""
        self.performance_history.append(metrics.get('fidelity', 0.0))
        
        for key, value in metrics.items():
            if key in self.metrics:
                self.metrics[key].append(value)
        
        # Keep history bounded
        max_history = self.performance_window * 2
        if len(self.performance_history) > max_history:
            self.performance_history = self.performance_history[-max_history:]
            for key in self.metrics:
                if len(self.metrics[key]) > max_history:
                    self.metrics[key] = self.metrics[key][-max_history:]
    
    def update_noise_characteristics(self, noise_data: Dict[str, float]):
        """Update noise tracking."""
        self.noise_history.append(noise_data.copy())
        
        # Keep history bounded
        max_history = self.performance_window * 2
        if len(self.noise_history) > max_history:
            self.noise_history = self.noise_history[-max_history:]
    
    def get_current_scheme(self) -> Optional[ErrorCorrectionScheme]:
        """Get currently selected error correction scheme."""
        return self.current_config.scheme if self.current_config else None
    
    def get_adaptation_statistics(self) -> Dict[str, Any]:
        """Get statistics about adaptation behavior."""
        return {
            'total_adaptations': self.adaptation_count,
            'current_scheme': self.current_config.scheme.name if self.current_config else None,
            'current_distance': self.current_config.distance if self.current_config else None,
            'current_overhead': self.current_config.overhead if self.current_config else None,
            'performance_trend': self._calculate_performance_trend(),
            'adaptation_frequency': self.adaptation_frequency,
            'strategy': self.strategy.value
        }
    
    def _calculate_performance_trend(self) -> Optional[float]:
        """Calculate recent performance trend."""
        if len(self.performance_history) < 20:
            return None
        
        recent = np.mean(self.performance_history[-10:])
        older = np.mean(self.performance_history[-20:-10])
        
        return (recent - older) / older if older != 0 else 0.0
    
    def reset(self):
        """Reset adaptation state."""
        self.current_config = None
        self.performance_history.clear()
        self.noise_history.clear()
        self.adaptation_count = 0
        for key in self.metrics:
            self.metrics[key].clear()
        
        self.logger.info("Adaptive QECC system reset")