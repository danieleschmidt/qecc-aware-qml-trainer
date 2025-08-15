"""
Noise models for quantum error correction simulations.
"""

from typing import Dict, Optional, Union, List
import numpy as np

try:
    from qiskit_aer.noise import NoiseModel as QiskitNoiseModel
    from qiskit_aer.noise import depolarizing_error, thermal_relaxation_error, ReadoutError
    from qiskit_aer.noise import pauli_error, amplitude_damping_error, phase_damping_error
except ImportError:
    # Mock noise model classes for fallback
    class QiskitNoiseModel:
        def __init__(self):
            pass
    
    def thermal_relaxation_error(t1, t2, time, excited_state_population=0):
        return None
    
    def depolarizing_error(p, num_qubits):
        return None
    
    def ReadoutError(probabilities):
        return None
    
    def pauli_error(error_list):
        return None
    
    def amplitude_damping_error(gamma):
        return None
    
    def phase_damping_error(gamma):
        return None


class NoiseModel:
    """
    Comprehensive noise model for quantum error correction simulations.
    """
    
    def __init__(
        self,
        gate_error_rate: float = 0.001,
        readout_error_rate: float = 0.01,
        T1: float = 50e-6,  # Relaxation time (s)
        T2: float = 70e-6,  # Dephasing time (s)
        gate_time: float = 20e-9,  # Gate time (s)
        measurement_time: float = 1e-6,  # Measurement time (s)
        thermal_population: float = 0.0,
        include_crosstalk: bool = False,
        crosstalk_strength: float = 0.01,
    ):
        """
        Initialize comprehensive noise model.
        
        Args:
            gate_error_rate: Error rate for single/two-qubit gates
            readout_error_rate: Measurement error rate
            T1: Amplitude damping time constant
            T2: Phase damping time constant
            gate_time: Duration of quantum gates
            measurement_time: Duration of measurements
            thermal_population: Thermal excited state population
            include_crosstalk: Whether to include crosstalk errors
            crosstalk_strength: Strength of crosstalk coupling
        """
        self.gate_error_rate = gate_error_rate
        self.readout_error_rate = readout_error_rate
        self.T1 = T1
        self.T2 = T2
        self.gate_time = gate_time
        self.measurement_time = measurement_time
        self.thermal_population = thermal_population
        self.include_crosstalk = include_crosstalk
        self.crosstalk_strength = crosstalk_strength
        
        self._qiskit_noise_model = None
        self._build_noise_model()
    
    def _build_noise_model(self):
        """Build the Qiskit noise model."""
        noise_model = QiskitNoiseModel()
        
        # Single-qubit gate errors
        single_qubit_error = self._get_single_qubit_error()
        noise_model.add_all_qubit_quantum_error(
            single_qubit_error, 
            ['u1', 'u2', 'u3', 'rx', 'ry', 'rz', 'x', 'y', 'z', 'h', 's', 't']
        )
        
        # Two-qubit gate errors
        two_qubit_error = self._get_two_qubit_error()
        noise_model.add_all_qubit_quantum_error(
            two_qubit_error,
            ['cx', 'cy', 'cz', 'swap', 'iswap', 'rzz']
        )
        
        # Readout errors
        readout_error = self._get_readout_error()
        noise_model.add_all_qubit_readout_error(readout_error)
        
        # Idle errors (decoherence during measurement)
        idle_error = self._get_idle_error()
        noise_model.add_all_qubit_quantum_error(idle_error, ['id'])
        
        self._qiskit_noise_model = noise_model
    
    def _get_single_qubit_error(self):
        """Get single-qubit gate error model."""
        # Combine depolarizing and relaxation errors
        depol_error = depolarizing_error(self.gate_error_rate, 1)
        
        # Thermal relaxation during gate operation
        relax_error = thermal_relaxation_error(
            self.T1, self.T2, self.gate_time, self.thermal_population
        )
        
        return depol_error.compose(relax_error)
    
    def _get_two_qubit_error(self):
        """Get two-qubit gate error model."""
        # Higher error rate for two-qubit gates
        two_qubit_error_rate = self.gate_error_rate * 10
        depol_error = depolarizing_error(two_qubit_error_rate, 2)
        
        # Thermal relaxation for both qubits
        relax_error1 = thermal_relaxation_error(
            self.T1, self.T2, self.gate_time * 2, self.thermal_population
        ).expand(1)
        relax_error2 = thermal_relaxation_error(
            self.T1, self.T2, self.gate_time * 2, self.thermal_population
        ).expand(1)
        
        # Combine relaxation errors for both qubits
        combined_relax = relax_error1.tensor(relax_error2)
        
        return depol_error.compose(combined_relax)
    
    def _get_readout_error(self):
        """Get measurement readout error."""
        # Symmetric readout error
        p01 = self.readout_error_rate  # P(measure 1 | state 0)
        p10 = self.readout_error_rate  # P(measure 0 | state 1)
        
        return ReadoutError([[1 - p01, p01], [p10, 1 - p10]])
    
    def _get_idle_error(self):
        """Get idle/identity error (decoherence during measurement)."""
        return thermal_relaxation_error(
            self.T1, self.T2, self.measurement_time, self.thermal_population
        )
    
    def get_qiskit_noise_model(self) -> QiskitNoiseModel:
        """Get the corresponding Qiskit noise model."""
        return self._qiskit_noise_model
    
    def get_effective_error_rate(self, circuit_depth: int, num_two_qubit_gates: int = 0) -> float:
        """
        Estimate effective error rate for a circuit.
        
        Args:
            circuit_depth: Depth of the quantum circuit
            num_two_qubit_gates: Number of two-qubit gates
            
        Returns:
            Estimated effective error rate
        """
        single_qubit_errors = circuit_depth * self.gate_error_rate
        two_qubit_errors = num_two_qubit_gates * self.gate_error_rate * 10
        readout_errors = self.readout_error_rate
        
        # Simple additive model (more sophisticated models possible)
        return min(1.0, single_qubit_errors + two_qubit_errors + readout_errors)
    
    def is_below_threshold(self, threshold: float) -> bool:
        """Check if noise is below error correction threshold."""
        return self.gate_error_rate < threshold
    
    @classmethod
    def from_backend(cls, backend_name: str) -> "NoiseModel":
        """
        Create noise model from known backend characteristics.
        
        Args:
            backend_name: Name of the backend ('ibm_lagos', 'ibm_nairobi', etc.)
            
        Returns:
            Configured noise model
        """
        # Predefined backend characteristics
        backend_specs = {
            "ibm_lagos": {
                "gate_error_rate": 0.001,
                "readout_error_rate": 0.015,
                "T1": 55e-6,
                "T2": 75e-6,
            },
            "ibm_nairobi": {
                "gate_error_rate": 0.0008,
                "readout_error_rate": 0.012,
                "T1": 60e-6,
                "T2": 80e-6,
            },
            "google_sycamore": {
                "gate_error_rate": 0.0015,
                "readout_error_rate": 0.02,
                "T1": 40e-6,
                "T2": 50e-6,
            },
            "ionq_harmony": {
                "gate_error_rate": 0.0005,
                "readout_error_rate": 0.008,
                "T1": 100e-6,
                "T2": 120e-6,
            }
        }
        
        if backend_name.lower() not in backend_specs:
            raise ValueError(f"Unknown backend: {backend_name}")
        
        specs = backend_specs[backend_name.lower()]
        return cls(**specs)
    
    @classmethod
    def ideal(cls) -> "NoiseModel":
        """Create an ideal (noiseless) model for testing."""
        return cls(
            gate_error_rate=0.0,
            readout_error_rate=0.0,
            T1=float('inf'),
            T2=float('inf'),
        )
    
    def scale_noise(self, factor: float) -> "NoiseModel":
        """
        Create a scaled version of this noise model.
        
        Args:
            factor: Scaling factor (1.0 = same, 2.0 = double noise, 0.5 = half noise)
            
        Returns:
            New noise model with scaled parameters
        """
        return NoiseModel(
            gate_error_rate=self.gate_error_rate * factor,
            readout_error_rate=self.readout_error_rate * factor,
            T1=self.T1 / factor if factor > 0 else float('inf'),
            T2=self.T2 / factor if factor > 0 else float('inf'),
            gate_time=self.gate_time,
            measurement_time=self.measurement_time,
            thermal_population=self.thermal_population * factor,
            include_crosstalk=self.include_crosstalk,
            crosstalk_strength=self.crosstalk_strength * factor,
        )
    
    def get_fidelity_estimate(self, circuit_depth: int) -> float:
        """
        Estimate circuit fidelity under this noise model.
        
        Args:
            circuit_depth: Depth of the circuit
            
        Returns:
            Estimated fidelity (0-1)
        """
        error_rate = self.get_effective_error_rate(circuit_depth)
        return max(0.0, 1.0 - error_rate)
    
    def __str__(self) -> str:
        return (f"NoiseModel(gate_err={self.gate_error_rate:.1e}, "
                f"readout_err={self.readout_error_rate:.1e}, "
                f"T1={self.T1*1e6:.1f}μs, T2={self.T2*1e6:.1f}μs)")
    
    def __repr__(self) -> str:
        return self.__str__()