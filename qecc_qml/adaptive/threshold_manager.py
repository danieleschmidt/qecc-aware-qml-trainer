"""
Threshold management for adaptive QECC systems.

Manages dynamic thresholds for QECC adaptation decisions based on
historical performance and system constraints.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
import logging
from enum import Enum


class ThresholdType(Enum):
    """Types of thresholds for QECC adaptation."""
    PERFORMANCE = "performance"
    ERROR_RATE = "error_rate"
    FIDELITY = "fidelity"
    RESOURCE = "resource"
    LATENCY = "latency"


@dataclass
class ThresholdConfig:
    """Configuration for a single threshold."""
    name: str
    threshold_type: ThresholdType
    initial_value: float
    min_value: float
    max_value: float
    adaptation_rate: float = 0.1
    sensitivity: float = 1.0
    current_value: float = field(init=False)
    
    def __post_init__(self):
        self.current_value = self.initial_value


class AdaptiveThresholds:
    """
    Adaptive threshold management system.
    
    Automatically adjusts thresholds based on system performance,
    historical data, and environmental conditions.
    """
    
    def __init__(
        self,
        initial_thresholds: Optional[Dict[str, ThresholdConfig]] = None,
        adaptation_window: int = 50,
        stability_factor: float = 0.95,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize adaptive thresholds.
        
        Args:
            initial_thresholds: Initial threshold configurations
            adaptation_window: Window size for threshold adaptation
            stability_factor: Factor for threshold stability (0-1)
            logger: Optional logger instance
        """
        self.adaptation_window = adaptation_window
        self.stability_factor = stability_factor
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize thresholds
        if initial_thresholds is None:
            self.thresholds = self._create_default_thresholds()
        else:
            self.thresholds = initial_thresholds
        
        # Historical data
        self.performance_history: List[Dict[str, float]] = []
        self.threshold_history: List[Dict[str, float]] = []
        self.adaptation_events: List[Dict[str, Any]] = []
        
        # Statistics
        self.adaptation_count = 0
        self.violation_counts: Dict[str, int] = {name: 0 for name in self.thresholds.keys()}
        
    def _create_default_thresholds(self) -> Dict[str, ThresholdConfig]:
        """Create default threshold configurations."""
        return {
            'fidelity_min': ThresholdConfig(
                name='fidelity_min',
                threshold_type=ThresholdType.FIDELITY,
                initial_value=0.90,
                min_value=0.50,
                max_value=0.99,
                adaptation_rate=0.05,
                sensitivity=2.0
            ),
            'error_rate_max': ThresholdConfig(
                name='error_rate_max',
                threshold_type=ThresholdType.ERROR_RATE,
                initial_value=0.01,
                min_value=0.001,
                max_value=0.10,
                adaptation_rate=0.10,
                sensitivity=1.5
            ),
            'performance_degradation': ThresholdConfig(
                name='performance_degradation',
                threshold_type=ThresholdType.PERFORMANCE,
                initial_value=0.05,  # 5% degradation triggers adaptation
                min_value=0.01,
                max_value=0.20,
                adaptation_rate=0.08,
                sensitivity=1.0
            ),
            'resource_utilization': ThresholdConfig(
                name='resource_utilization',
                threshold_type=ThresholdType.RESOURCE,
                initial_value=0.80,  # 80% resource utilization
                min_value=0.50,
                max_value=0.95,
                adaptation_rate=0.05,
                sensitivity=0.8
            ),
            'latency_max': ThresholdConfig(
                name='latency_max',
                threshold_type=ThresholdType.LATENCY,
                initial_value=5.0,  # 5x overhead maximum
                min_value=2.0,
                max_value=20.0,
                adaptation_rate=0.12,
                sensitivity=1.2
            )
        }
    
    def get_threshold(self, name: str) -> Optional[float]:
        """Get current threshold value."""
        if name in self.thresholds:
            return self.thresholds[name].current_value
        return None
    
    def set_threshold(self, name: str, value: float) -> bool:
        """
        Set threshold value.
        
        Args:
            name: Threshold name
            value: New threshold value
            
        Returns:
            True if threshold was set successfully
        """
        if name not in self.thresholds:
            return False
        
        config = self.thresholds[name]
        if config.min_value <= value <= config.max_value:
            old_value = config.current_value
            config.current_value = value
            
            self.logger.info(f"Threshold {name} updated: {old_value:.4f} -> {value:.4f}")
            
            self.adaptation_events.append({
                'timestamp': self._get_timestamp(),
                'threshold': name,
                'old_value': old_value,
                'new_value': value,
                'method': 'manual'
            })
            
            return True
        else:
            self.logger.warning(
                f"Threshold value {value} out of bounds for {name} "
                f"[{config.min_value}, {config.max_value}]"
            )
            return False
    
    def check_threshold(self, name: str, value: float) -> bool:
        """
        Check if value violates threshold.
        
        Args:
            name: Threshold name
            value: Value to check
            
        Returns:
            True if threshold is violated
        """
        if name not in self.thresholds:
            return False
        
        config = self.thresholds[name]
        threshold = config.current_value
        
        # Different threshold types have different violation conditions
        if config.threshold_type in [ThresholdType.FIDELITY, ThresholdType.PERFORMANCE]:
            violated = value < threshold  # Should be above threshold
        else:
            violated = value > threshold  # Should be below threshold
        
        if violated:
            self.violation_counts[name] += 1
            self.logger.debug(f"Threshold violation: {name} = {value:.4f} vs {threshold:.4f}")
        
        return violated
    
    def update_performance(self, metrics: Dict[str, float]):
        """Update performance history for threshold adaptation."""
        self.performance_history.append(metrics.copy())
        
        # Keep history bounded
        if len(self.performance_history) > self.adaptation_window * 2:
            self.performance_history = self.performance_history[-self.adaptation_window * 2:]
    
    def adapt_thresholds(self) -> Dict[str, float]:
        """
        Adapt thresholds based on recent performance.
        
        Returns:
            Dictionary of threshold changes
        """
        if len(self.performance_history) < self.adaptation_window:
            return {}
        
        changes = {}
        recent_metrics = self.performance_history[-self.adaptation_window:]
        
        for name, config in self.thresholds.items():
            old_value = config.current_value
            new_value = self._adapt_single_threshold(config, recent_metrics)
            
            if abs(new_value - old_value) > 1e-6:  # Significant change
                config.current_value = new_value
                changes[name] = new_value - old_value
                
                self.adaptation_events.append({
                    'timestamp': self._get_timestamp(),
                    'threshold': name,
                    'old_value': old_value,
                    'new_value': new_value,
                    'method': 'adaptive',
                    'change': new_value - old_value
                })
        
        if changes:
            self.adaptation_count += 1
            self.logger.info(f"Threshold adaptation #{self.adaptation_count}: {len(changes)} thresholds updated")
        
        # Store current threshold values in history
        current_thresholds = {name: config.current_value for name, config in self.thresholds.items()}
        self.threshold_history.append(current_thresholds)
        
        return changes
    
    def _adapt_single_threshold(
        self, 
        config: ThresholdConfig, 
        recent_metrics: List[Dict[str, float]]
    ) -> float:
        """Adapt a single threshold based on recent performance."""
        current_value = config.current_value
        
        if config.threshold_type == ThresholdType.FIDELITY:
            return self._adapt_fidelity_threshold(config, recent_metrics)
        elif config.threshold_type == ThresholdType.ERROR_RATE:
            return self._adapt_error_rate_threshold(config, recent_metrics)
        elif config.threshold_type == ThresholdType.PERFORMANCE:
            return self._adapt_performance_threshold(config, recent_metrics)
        elif config.threshold_type == ThresholdType.RESOURCE:
            return self._adapt_resource_threshold(config, recent_metrics)
        elif config.threshold_type == ThresholdType.LATENCY:
            return self._adapt_latency_threshold(config, recent_metrics)
        else:
            return current_value
    
    def _adapt_fidelity_threshold(
        self, 
        config: ThresholdConfig, 
        recent_metrics: List[Dict[str, float]]
    ) -> float:
        """Adapt fidelity threshold based on achieved fidelities."""
        fidelities = [m.get('fidelity', 0.9) for m in recent_metrics]
        
        if not fidelities:
            return config.current_value
        
        mean_fidelity = np.mean(fidelities)
        std_fidelity = np.std(fidelities)
        
        # If we're consistently achieving high fidelity, we can raise the threshold
        if mean_fidelity > config.current_value + 2 * std_fidelity:
            # Gradually increase threshold
            adjustment = config.adaptation_rate * config.sensitivity * std_fidelity
            new_value = config.current_value + adjustment
        # If we're struggling to meet the threshold, lower it
        elif mean_fidelity < config.current_value - std_fidelity:
            # Gradually decrease threshold
            adjustment = config.adaptation_rate * config.sensitivity * std_fidelity
            new_value = config.current_value - adjustment
        else:
            # Keep current value
            new_value = config.current_value
        
        # Apply stability factor and bounds
        stable_value = (self.stability_factor * config.current_value + 
                       (1 - self.stability_factor) * new_value)
        
        return np.clip(stable_value, config.min_value, config.max_value)
    
    def _adapt_error_rate_threshold(
        self,
        config: ThresholdConfig,
        recent_metrics: List[Dict[str, float]]
    ) -> float:
        """Adapt error rate threshold based on observed error rates."""
        error_rates = [m.get('logical_error_rate', 0.001) for m in recent_metrics]
        
        if not error_rates:
            return config.current_value
        
        mean_error_rate = np.mean(error_rates)
        max_error_rate = np.max(error_rates)
        
        # If error rates are consistently low, we can tighten the threshold
        if max_error_rate < config.current_value * 0.5:
            adjustment = config.adaptation_rate * config.sensitivity
            new_value = config.current_value * (1 - adjustment)
        # If error rates are often high, relax the threshold
        elif mean_error_rate > config.current_value * 0.8:
            adjustment = config.adaptation_rate * config.sensitivity
            new_value = config.current_value * (1 + adjustment)
        else:
            new_value = config.current_value
        
        stable_value = (self.stability_factor * config.current_value + 
                       (1 - self.stability_factor) * new_value)
        
        return np.clip(stable_value, config.min_value, config.max_value)
    
    def _adapt_performance_threshold(
        self,
        config: ThresholdConfig,
        recent_metrics: List[Dict[str, float]]
    ) -> float:
        """Adapt performance degradation threshold."""
        accuracies = [m.get('accuracy', 0.8) for m in recent_metrics]
        
        if len(accuracies) < 2:
            return config.current_value
        
        # Calculate performance variability
        performance_std = np.std(accuracies)
        performance_range = np.max(accuracies) - np.min(accuracies)
        
        # If performance is very stable, we can use tighter thresholds
        if performance_std < 0.02 and performance_range < 0.05:
            adjustment = config.adaptation_rate * config.sensitivity
            new_value = config.current_value * (1 - adjustment)
        # If performance is highly variable, use looser thresholds
        elif performance_std > 0.05 or performance_range > 0.15:
            adjustment = config.adaptation_rate * config.sensitivity
            new_value = config.current_value * (1 + adjustment)
        else:
            new_value = config.current_value
        
        stable_value = (self.stability_factor * config.current_value + 
                       (1 - self.stability_factor) * new_value)
        
        return np.clip(stable_value, config.min_value, config.max_value)
    
    def _adapt_resource_threshold(
        self,
        config: ThresholdConfig,
        recent_metrics: List[Dict[str, float]]
    ) -> float:
        """Adapt resource utilization threshold."""
        resource_usage = [m.get('resource_utilization', 0.7) for m in recent_metrics]
        
        if not resource_usage:
            return config.current_value
        
        mean_usage = np.mean(resource_usage)
        max_usage = np.max(resource_usage)
        
        # Adjust based on actual resource usage patterns
        if max_usage < config.current_value * 0.8:
            # Resources are underutilized, can be more aggressive
            adjustment = config.adaptation_rate * config.sensitivity
            new_value = config.current_value + adjustment * 0.1
        elif mean_usage > config.current_value * 0.95:
            # Close to resource limits, be more conservative
            adjustment = config.adaptation_rate * config.sensitivity
            new_value = config.current_value - adjustment * 0.1
        else:
            new_value = config.current_value
        
        stable_value = (self.stability_factor * config.current_value + 
                       (1 - self.stability_factor) * new_value)
        
        return np.clip(stable_value, config.min_value, config.max_value)
    
    def _adapt_latency_threshold(
        self,
        config: ThresholdConfig,
        recent_metrics: List[Dict[str, float]]
    ) -> float:
        """Adapt latency threshold based on observed overheads."""
        overheads = [m.get('overhead', 3.0) for m in recent_metrics]
        
        if not overheads:
            return config.current_value
        
        mean_overhead = np.mean(overheads)
        max_overhead = np.max(overheads)
        
        # Adjust based on actual overhead patterns
        if max_overhead < config.current_value * 0.7:
            # Overheads are consistently low, can tighten threshold
            adjustment = config.adaptation_rate * config.sensitivity
            new_value = config.current_value * (1 - adjustment)
        elif mean_overhead > config.current_value * 0.8:
            # Overheads are high, need to relax threshold
            adjustment = config.adaptation_rate * config.sensitivity
            new_value = config.current_value * (1 + adjustment)
        else:
            new_value = config.current_value
        
        stable_value = (self.stability_factor * config.current_value + 
                       (1 - self.stability_factor) * new_value)
        
        return np.clip(stable_value, config.min_value, config.max_value)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get threshold management statistics."""
        return {
            'adaptation_count': self.adaptation_count,
            'violation_counts': self.violation_counts.copy(),
            'current_thresholds': {name: config.current_value for name, config in self.thresholds.items()},
            'threshold_ranges': {
                name: {
                    'min': config.min_value,
                    'max': config.max_value,
                    'current': config.current_value,
                    'initial': config.initial_value
                } for name, config in self.thresholds.items()
            },
            'recent_adaptations': self.adaptation_events[-10:] if self.adaptation_events else []
        }
    
    def reset_statistics(self):
        """Reset threshold statistics."""
        self.adaptation_count = 0
        self.violation_counts = {name: 0 for name in self.thresholds.keys()}
        self.adaptation_events.clear()
        self.threshold_history.clear()
        
        self.logger.info("Threshold statistics reset")
    
    def _get_timestamp(self) -> float:
        """Get current timestamp."""
        import time
        return time.time()


class ThresholdManager:
    """
    High-level manager for multiple adaptive threshold systems.
    
    Coordinates threshold adaptation across different components
    of the QECC-aware system.
    """
    
    def __init__(
        self,
        adaptive_thresholds: Optional[AdaptiveThresholds] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize threshold manager.
        
        Args:
            adaptive_thresholds: Adaptive threshold system
            logger: Optional logger instance
        """
        self.adaptive_thresholds = adaptive_thresholds or AdaptiveThresholds()
        self.logger = logger or logging.getLogger(__name__)
        
        # Component-specific threshold managers
        self.component_managers: Dict[str, AdaptiveThresholds] = {}
        
        # Global constraints
        self.global_constraints = {
            'max_resource_utilization': 0.95,
            'min_performance_threshold': 0.5,
            'max_latency_multiplier': 20.0
        }
    
    def add_component(self, name: str, thresholds: AdaptiveThresholds):
        """Add component-specific threshold manager."""
        self.component_managers[name] = thresholds
        self.logger.info(f"Added threshold manager for component: {name}")
    
    def update_all_thresholds(self, metrics: Dict[str, Dict[str, float]]):
        """Update all threshold systems with component metrics."""
        # Update main adaptive thresholds
        if 'global' in metrics:
            self.adaptive_thresholds.update_performance(metrics['global'])
        
        # Update component-specific thresholds
        for component_name, component_thresholds in self.component_managers.items():
            if component_name in metrics:
                component_thresholds.update_performance(metrics[component_name])
    
    def adapt_all_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Adapt all threshold systems."""
        all_changes = {}
        
        # Adapt main thresholds
        main_changes = self.adaptive_thresholds.adapt_thresholds()
        if main_changes:
            all_changes['global'] = main_changes
        
        # Adapt component thresholds
        for component_name, component_thresholds in self.component_managers.items():
            component_changes = component_thresholds.adapt_thresholds()
            if component_changes:
                all_changes[component_name] = component_changes
        
        # Enforce global constraints
        self._enforce_global_constraints()
        
        return all_changes
    
    def _enforce_global_constraints(self):
        """Enforce global constraints across all threshold systems."""
        # Check resource utilization constraints
        for thresholds in [self.adaptive_thresholds] + list(self.component_managers.values()):
            resource_threshold = thresholds.get_threshold('resource_utilization')
            if resource_threshold and resource_threshold > self.global_constraints['max_resource_utilization']:
                thresholds.set_threshold('resource_utilization', self.global_constraints['max_resource_utilization'])
                self.logger.warning("Enforced global resource utilization constraint")
        
        # Similar checks for other global constraints...
    
    def get_global_statistics(self) -> Dict[str, Any]:
        """Get statistics from all threshold systems."""
        stats = {
            'global': self.adaptive_thresholds.get_statistics()
        }
        
        for name, component_thresholds in self.component_managers.items():
            stats[name] = component_thresholds.get_statistics()
        
        return stats