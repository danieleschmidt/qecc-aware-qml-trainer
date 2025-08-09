"""
Real-time noise monitoring for adaptive QECC systems.

Monitors hardware noise characteristics and provides real-time feedback
for adaptive error correction decision making.
"""

import numpy as np
import time
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import threading
import logging
from queue import Queue, Empty

from ..core.noise_models import NoiseModel


@dataclass
class NoiseSnapshot:
    """Snapshot of noise characteristics at a point in time."""
    timestamp: float
    gate_error_rate: float
    readout_error_rate: float
    coherence_t1: float
    coherence_t2: float
    gate_fidelities: Dict[str, float] = field(default_factory=dict)
    cross_talk: Optional[np.ndarray] = None
    temperature: Optional[float] = None
    additional_metrics: Dict[str, Any] = field(default_factory=dict)


class NoiseMonitor(ABC):
    """Abstract base class for noise monitoring systems."""
    
    @abstractmethod
    def get_current_noise(self) -> NoiseSnapshot:
        """Get current noise characteristics."""
        pass
    
    @abstractmethod
    def start_monitoring(self):
        """Start continuous monitoring."""
        pass
    
    @abstractmethod
    def stop_monitoring(self):
        """Stop monitoring."""
        pass


class HardwareMonitor(NoiseMonitor):
    """
    Real hardware noise monitor.
    
    Interfaces with quantum hardware backends to collect real-time
    noise and calibration data.
    """
    
    def __init__(
        self,
        backend,
        update_interval: float = 30.0,  # seconds
        history_length: int = 1000,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize hardware monitor.
        
        Args:
            backend: Quantum backend to monitor
            update_interval: How often to update noise data
            history_length: Number of historical snapshots to keep
            logger: Optional logger instance
        """
        self.backend = backend
        self.update_interval = update_interval
        self.history_length = history_length
        self.logger = logger or logging.getLogger(__name__)
        
        self.noise_history: List[NoiseSnapshot] = []
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.update_queue = Queue()
        
        # Callbacks for noise changes
        self.change_callbacks: List[Callable[[NoiseSnapshot], None]] = []
        
        # Alert thresholds
        self.alert_thresholds = {
            'gate_error_rate': 0.01,  # Alert if > 1%
            'readout_error_rate': 0.05,  # Alert if > 5%
            'coherence_t1': 10e-6,  # Alert if < 10 μs
            'coherence_t2': 5e-6,   # Alert if < 5 μs
        }
    
    def get_current_noise(self) -> NoiseSnapshot:
        """Get current noise characteristics from hardware."""
        try:
            # Get backend properties
            properties = self.backend.properties()
            
            if properties is None:
                return self._create_default_snapshot()
            
            # Extract gate error rates
            gate_errors = []
            gate_fidelities = {}
            
            for gate in properties.gates:
                if hasattr(gate, 'error') and gate.error is not None:
                    gate_errors.append(gate.error)
                    gate_name = f"{gate.gate}_{gate.qubits}"
                    gate_fidelities[gate_name] = 1 - gate.error
            
            avg_gate_error = np.mean(gate_errors) if gate_errors else 0.001
            
            # Extract readout error rates
            readout_errors = []
            for qubit in properties.qubits:
                for readout_error in qubit:
                    if readout_error.name == 'readout_error':
                        readout_errors.append(readout_error.value)
            
            avg_readout_error = np.mean(readout_errors) if readout_errors else 0.01
            
            # Extract coherence times
            t1_times = []
            t2_times = []
            
            for qubit in properties.qubits:
                for param in qubit:
                    if param.name == 'T1':
                        t1_times.append(param.value)
                    elif param.name == 'T2':
                        t2_times.append(param.value)
            
            avg_t1 = np.mean(t1_times) if t1_times else 50e-6
            avg_t2 = np.mean(t2_times) if t2_times else 70e-6
            
            snapshot = NoiseSnapshot(
                timestamp=time.time(),
                gate_error_rate=avg_gate_error,
                readout_error_rate=avg_readout_error,
                coherence_t1=avg_t1,
                coherence_t2=avg_t2,
                gate_fidelities=gate_fidelities
            )
            
            return snapshot
            
        except Exception as e:
            self.logger.error(f"Failed to get hardware noise data: {e}")
            return self._create_default_snapshot()
    
    def _create_default_snapshot(self) -> NoiseSnapshot:
        """Create default noise snapshot when hardware data unavailable."""
        return NoiseSnapshot(
            timestamp=time.time(),
            gate_error_rate=0.001,
            readout_error_rate=0.01,
            coherence_t1=50e-6,
            coherence_t2=70e-6
        )
    
    def start_monitoring(self):
        """Start continuous hardware monitoring."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        self.logger.info(f"Started hardware monitoring with {self.update_interval}s interval")
    
    def stop_monitoring(self):
        """Stop hardware monitoring."""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        
        self.logger.info("Stopped hardware monitoring")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                # Get current noise snapshot
                snapshot = self.get_current_noise()
                
                # Add to history
                self.noise_history.append(snapshot)
                
                # Keep history bounded
                if len(self.noise_history) > self.history_length:
                    self.noise_history = self.noise_history[-self.history_length:]
                
                # Check for alerts
                self._check_noise_alerts(snapshot)
                
                # Notify callbacks
                for callback in self.change_callbacks:
                    try:
                        callback(snapshot)
                    except Exception as e:
                        self.logger.error(f"Noise monitoring callback failed: {e}")
                
                # Add to update queue for external polling
                try:
                    self.update_queue.put_nowait(snapshot)
                except:
                    pass  # Queue full, skip this update
                
                time.sleep(self.update_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.update_interval)
    
    def _check_noise_alerts(self, snapshot: NoiseSnapshot):
        """Check for noise threshold violations."""
        alerts = []
        
        if snapshot.gate_error_rate > self.alert_thresholds['gate_error_rate']:
            alerts.append(f"High gate error rate: {snapshot.gate_error_rate:.4f}")
        
        if snapshot.readout_error_rate > self.alert_thresholds['readout_error_rate']:
            alerts.append(f"High readout error rate: {snapshot.readout_error_rate:.4f}")
        
        if snapshot.coherence_t1 < self.alert_thresholds['coherence_t1']:
            alerts.append(f"Low T1 coherence: {snapshot.coherence_t1*1e6:.1f} μs")
        
        if snapshot.coherence_t2 < self.alert_thresholds['coherence_t2']:
            alerts.append(f"Low T2 coherence: {snapshot.coherence_t2*1e6:.1f} μs")
        
        if alerts:
            for alert in alerts:
                self.logger.warning(f"Noise alert: {alert}")
    
    def add_change_callback(self, callback: Callable[[NoiseSnapshot], None]):
        """Add callback for noise changes."""
        self.change_callbacks.append(callback)
    
    def remove_change_callback(self, callback: Callable[[NoiseSnapshot], None]):
        """Remove noise change callback."""
        if callback in self.change_callbacks:
            self.change_callbacks.remove(callback)
    
    def get_noise_history(self, duration: Optional[float] = None) -> List[NoiseSnapshot]:
        """
        Get historical noise data.
        
        Args:
            duration: Time window in seconds (None for all history)
            
        Returns:
            List of noise snapshots
        """
        if duration is None:
            return self.noise_history.copy()
        
        cutoff_time = time.time() - duration
        return [s for s in self.noise_history if s.timestamp >= cutoff_time]
    
    def get_noise_statistics(self, duration: Optional[float] = None) -> Dict[str, Any]:
        """Get noise statistics over time window."""
        history = self.get_noise_history(duration)
        
        if not history:
            return {}
        
        gate_errors = [s.gate_error_rate for s in history]
        readout_errors = [s.readout_error_rate for s in history]
        t1_times = [s.coherence_t1 for s in history]
        t2_times = [s.coherence_t2 for s in history]
        
        return {
            'gate_error_rate': {
                'mean': np.mean(gate_errors),
                'std': np.std(gate_errors),
                'min': np.min(gate_errors),
                'max': np.max(gate_errors)
            },
            'readout_error_rate': {
                'mean': np.mean(readout_errors),
                'std': np.std(readout_errors),
                'min': np.min(readout_errors),
                'max': np.max(readout_errors)
            },
            'coherence_t1': {
                'mean': np.mean(t1_times),
                'std': np.std(t1_times),
                'min': np.min(t1_times),
                'max': np.max(t1_times)
            },
            'coherence_t2': {
                'mean': np.mean(t2_times),
                'std': np.std(t2_times),
                'min': np.min(t2_times),
                'max': np.max(t2_times)
            },
            'sample_count': len(history),
            'time_span': history[-1].timestamp - history[0].timestamp if len(history) > 1 else 0
        }
    
    def set_alert_threshold(self, metric: str, threshold: float):
        """Set alert threshold for a noise metric."""
        if metric in self.alert_thresholds:
            self.alert_thresholds[metric] = threshold
            self.logger.info(f"Updated alert threshold for {metric}: {threshold}")
        else:
            self.logger.warning(f"Unknown metric for alert threshold: {metric}")
    
    def poll_update(self, timeout: float = 0.1) -> Optional[NoiseSnapshot]:
        """Poll for noise updates (non-blocking)."""
        try:
            return self.update_queue.get(timeout=timeout)
        except Empty:
            return None


class SimulatedNoiseMonitor(NoiseMonitor):
    """
    Simulated noise monitor for testing and development.
    
    Generates realistic noise variations for testing adaptive QECC systems.
    """
    
    def __init__(
        self,
        base_noise: NoiseModel,
        variation_amplitude: float = 0.1,
        trend_period: float = 300.0,  # seconds
        spike_probability: float = 0.05,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize simulated monitor.
        
        Args:
            base_noise: Base noise model
            variation_amplitude: Amplitude of noise variations
            trend_period: Period of noise trends
            spike_probability: Probability of noise spikes per update
            logger: Optional logger
        """
        self.base_noise = base_noise
        self.variation_amplitude = variation_amplitude
        self.trend_period = trend_period
        self.spike_probability = spike_probability
        self.logger = logger or logging.getLogger(__name__)
        
        self.start_time = time.time()
        self.is_monitoring = False
        
    def get_current_noise(self) -> NoiseSnapshot:
        """Generate simulated current noise."""
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        # Base noise with sinusoidal variation
        phase = 2 * np.pi * elapsed / self.trend_period
        variation_factor = 1 + self.variation_amplitude * np.sin(phase)
        
        gate_error = self.base_noise.gate_error_rate * variation_factor
        readout_error = self.base_noise.readout_error_rate * variation_factor
        
        # Add random spikes
        if np.random.random() < self.spike_probability:
            spike_factor = 1 + np.random.exponential(0.5)
            gate_error *= spike_factor
            self.logger.info(f"Simulated noise spike: {spike_factor:.2f}x increase")
        
        # Coherence times inversely related to error rates
        t1 = self.base_noise.T1 * (1 / variation_factor)
        t2 = self.base_noise.T2 * (1 / variation_factor)
        
        return NoiseSnapshot(
            timestamp=current_time,
            gate_error_rate=gate_error,
            readout_error_rate=readout_error,
            coherence_t1=t1,
            coherence_t2=t2,
            additional_metrics={
                'variation_factor': variation_factor,
                'phase': phase,
                'elapsed_time': elapsed
            }
        )
    
    def start_monitoring(self):
        """Start simulated monitoring."""
        self.is_monitoring = True
        self.start_time = time.time()
        self.logger.info("Started simulated noise monitoring")
    
    def stop_monitoring(self):
        """Stop simulated monitoring."""
        self.is_monitoring = False
        self.logger.info("Stopped simulated noise monitoring")