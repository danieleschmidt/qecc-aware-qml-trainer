"""
Real-time fidelity tracking for quantum circuits.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import time
from collections import deque

from ..core.noise_models import NoiseModel


class FidelityTracker:
    """
    Real-time tracker for quantum circuit fidelity and error rates.
    
    Monitors fidelity degradation during training and provides
    adaptive feedback for error correction strategies.
    """
    
    def __init__(
        self,
        window_size: int = 100,
        update_frequency: float = 1.0,
        track_logical_errors: bool = True,
    ):
        """
        Initialize fidelity tracker.
        
        Args:
            window_size: Number of recent measurements to keep
            update_frequency: Update frequency in seconds
            track_logical_errors: Whether to track logical error rates
        """
        self.window_size = window_size
        self.update_frequency = update_frequency
        self.track_logical_errors = track_logical_errors
        
        # Tracking data
        self.fidelity_history = deque(maxlen=window_size)
        self.error_rate_history = deque(maxlen=window_size)
        self.logical_error_history = deque(maxlen=window_size)
        self.timestamp_history = deque(maxlen=window_size)
        
        # Statistics
        self.running_stats = {
            'mean_fidelity': 0.0,
            'std_fidelity': 0.0,
            'mean_error_rate': 0.0,
            'error_trend': 0.0,
            'last_update': 0.0,
        }
        
        # Thresholds and alerts
        self.fidelity_threshold = 0.8
        self.error_threshold = 0.1
        self.alerts = []
    
    def update(
        self,
        circuit_fidelity: float,
        physical_error_rate: float,
        logical_error_rate: Optional[float] = None,
        timestamp: Optional[float] = None
    ):
        """
        Update fidelity tracking with new measurements.
        
        Args:
            circuit_fidelity: Current circuit fidelity (0-1)
            physical_error_rate: Physical error rate
            logical_error_rate: Logical error rate (if available)
            timestamp: Measurement timestamp (defaults to current time)
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Store measurements
        self.fidelity_history.append(circuit_fidelity)
        self.error_rate_history.append(physical_error_rate)
        self.timestamp_history.append(timestamp)
        
        if logical_error_rate is not None and self.track_logical_errors:
            self.logical_error_history.append(logical_error_rate)
        
        # Update statistics
        self._update_statistics()
        
        # Check for alerts
        self._check_alerts(circuit_fidelity, physical_error_rate, logical_error_rate)
    
    def _update_statistics(self):
        """Update running statistics."""
        if len(self.fidelity_history) < 2:
            return
        
        fidelities = np.array(self.fidelity_history)
        error_rates = np.array(self.error_rate_history)
        
        # Basic statistics
        self.running_stats['mean_fidelity'] = np.mean(fidelities)
        self.running_stats['std_fidelity'] = np.std(fidelities)
        self.running_stats['mean_error_rate'] = np.mean(error_rates)
        
        # Trend analysis (simple linear fit)
        if len(fidelities) >= 10:
            x = np.arange(len(fidelities))
            trend_coeff = np.polyfit(x, fidelities, 1)[0]
            self.running_stats['error_trend'] = -trend_coeff  # Negative fidelity trend = positive error trend
        
        self.running_stats['last_update'] = time.time()
    
    def _check_alerts(
        self, 
        fidelity: float, 
        error_rate: float, 
        logical_error_rate: Optional[float]
    ):
        """Check for alert conditions."""
        current_time = time.time()
        
        # Low fidelity alert
        if fidelity < self.fidelity_threshold:
            self.alerts.append({
                'type': 'low_fidelity',
                'timestamp': current_time,
                'message': f'Circuit fidelity dropped to {fidelity:.3f}',
                'severity': 'warning' if fidelity > 0.5 else 'critical'
            })
        
        # High error rate alert
        if error_rate > self.error_threshold:
            self.alerts.append({
                'type': 'high_error_rate',
                'timestamp': current_time,
                'message': f'Physical error rate increased to {error_rate:.1e}',
                'severity': 'warning' if error_rate < 0.2 else 'critical'
            })
        
        # Degradation trend alert
        if self.running_stats['error_trend'] > 0.01:  # Significant degradation trend
            self.alerts.append({
                'type': 'degradation_trend',
                'timestamp': current_time,
                'message': f'Fidelity degradation trend detected: {self.running_stats["error_trend"]:.4f}/step',
                'severity': 'warning'
            })
        
        # Logical vs physical error comparison
        if logical_error_rate is not None and logical_error_rate > error_rate:
            self.alerts.append({
                'type': 'ineffective_qecc',
                'timestamp': current_time,
                'message': f'Logical errors ({logical_error_rate:.1e}) exceed physical errors ({error_rate:.1e})',
                'severity': 'critical'
            })
        
        # Keep only recent alerts (last hour)
        cutoff_time = current_time - 3600
        self.alerts = [alert for alert in self.alerts if alert['timestamp'] > cutoff_time]
    
    def get_current_stats(self) -> Dict[str, float]:
        """Get current fidelity statistics."""
        stats = self.running_stats.copy()
        
        if self.fidelity_history:
            stats['current_fidelity'] = self.fidelity_history[-1]
            stats['current_error_rate'] = self.error_rate_history[-1]
        
        if self.logical_error_history:
            stats['current_logical_error_rate'] = self.logical_error_history[-1]
        
        return stats
    
    def get_recent_alerts(self, severity: Optional[str] = None) -> List[Dict]:
        """
        Get recent alerts.
        
        Args:
            severity: Filter by severity ('warning', 'critical', or None for all)
            
        Returns:
            List of alert dictionaries
        """
        if severity is None:
            return self.alerts.copy()
        else:
            return [alert for alert in self.alerts if alert['severity'] == severity]
    
    def estimate_improvement_potential(self) -> Dict[str, float]:
        """
        Estimate potential improvement from better error correction.
        
        Returns:
            Dictionary with improvement estimates
        """
        if len(self.fidelity_history) < 10:
            return {}
        
        current_fidelity = self.fidelity_history[-1]
        mean_fidelity = self.running_stats['mean_fidelity']
        
        # Estimate potential improvement
        improvement_potential = min(1.0 - current_fidelity, 0.2)  # Cap at 20% improvement
        
        estimates = {
            'current_fidelity': current_fidelity,
            'mean_fidelity': mean_fidelity,
            'improvement_potential': improvement_potential,
            'estimated_optimal_fidelity': min(1.0, current_fidelity + improvement_potential),
        }
        
        # Performance degradation rate
        if len(self.fidelity_history) >= 20:
            recent_fidelity = np.mean(list(self.fidelity_history)[-10:])
            older_fidelity = np.mean(list(self.fidelity_history)[-20:-10])
            degradation_rate = (older_fidelity - recent_fidelity) / 10  # per step
            
            estimates['degradation_rate'] = degradation_rate
            estimates['stability_score'] = max(0.0, 1.0 - abs(degradation_rate) * 100)
        
        return estimates
    
    def recommend_actions(self) -> List[Dict[str, str]]:
        """
        Recommend actions based on current fidelity status.
        
        Returns:
            List of recommendation dictionaries
        """
        recommendations = []
        stats = self.get_current_stats()
        
        # Low fidelity recommendations
        if stats.get('current_fidelity', 1.0) < self.fidelity_threshold:
            recommendations.append({
                'action': 'increase_error_correction',
                'reason': 'Circuit fidelity below threshold',
                'urgency': 'high' if stats['current_fidelity'] < 0.5 else 'medium'
            })
        
        # High error rate recommendations
        if stats.get('current_error_rate', 0.0) > self.error_threshold:
            recommendations.append({
                'action': 'reduce_circuit_depth',
                'reason': 'High physical error rate detected',
                'urgency': 'medium'
            })
        
        # Degradation trend recommendations
        if stats.get('error_trend', 0.0) > 0.01:
            recommendations.append({
                'action': 'adaptive_error_correction',
                'reason': 'Fidelity degradation trend detected',
                'urgency': 'medium'
            })
        
        # QECC effectiveness recommendations
        logical_err = stats.get('current_logical_error_rate')
        physical_err = stats.get('current_error_rate')
        
        if logical_err is not None and physical_err is not None:
            if logical_err > physical_err * 0.8:  # QECC not very effective
                recommendations.append({
                    'action': 'optimize_qecc_parameters',
                    'reason': 'Error correction showing limited effectiveness',
                    'urgency': 'low'
                })
        
        return recommendations
    
    def get_fidelity_trend(self, window: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get fidelity trend data for plotting.
        
        Args:
            window: Number of recent points to return (default: all)
            
        Returns:
            Tuple of (timestamps, fidelities)
        """
        if not self.fidelity_history:
            return np.array([]), np.array([])
        
        fidelities = np.array(self.fidelity_history)
        timestamps = np.array(self.timestamp_history)
        
        if window is not None and len(fidelities) > window:
            fidelities = fidelities[-window:]
            timestamps = timestamps[-window:]
        
        return timestamps, fidelities
    
    def export_tracking_data(self, filepath: str):
        """Export tracking data to file."""
        import json
        
        export_data = {
            'fidelity_history': list(self.fidelity_history),
            'error_rate_history': list(self.error_rate_history),
            'logical_error_history': list(self.logical_error_history),
            'timestamp_history': list(self.timestamp_history),
            'running_stats': self.running_stats,
            'alerts': self.alerts,
            'config': {
                'window_size': self.window_size,
                'update_frequency': self.update_frequency,
                'fidelity_threshold': self.fidelity_threshold,
                'error_threshold': self.error_threshold,
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"Tracking data exported to {filepath}")
    
    def reset(self):
        """Reset all tracking data."""
        self.fidelity_history.clear()
        self.error_rate_history.clear()
        self.logical_error_history.clear()
        self.timestamp_history.clear()
        self.alerts.clear()
        
        self.running_stats = {
            'mean_fidelity': 0.0,
            'std_fidelity': 0.0,
            'mean_error_rate': 0.0,
            'error_trend': 0.0,
            'last_update': 0.0,
        }
    
    def __str__(self) -> str:
        stats = self.get_current_stats()
        return (f"FidelityTracker(current={stats.get('current_fidelity', 0):.3f}, "
                f"mean={stats.get('mean_fidelity', 0):.3f}, "
                f"alerts={len(self.alerts)})")
    
    def __repr__(self) -> str:
        return self.__str__()