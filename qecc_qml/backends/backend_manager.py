"""
Quantum backend management for automatic selection and optimization.
"""

from typing import Optional, Dict, Any
from qiskit_aer import AerSimulator
from qiskit.providers.fake_provider import FakeLagos


class QuantumBackendManager:
    """
    Manages quantum backends with automatic selection based on requirements.
    """
    
    def __init__(self):
        """Initialize backend manager."""
        self.available_backends = {}
        self._initialize_simulators()
        
    def _initialize_simulators(self):
        """Initialize available simulator backends."""
        self.available_backends['aer_simulator'] = {
            'backend': AerSimulator(),
            'type': 'simulator',
            'noise_free': True,
            'max_qubits': 32,
        }
        
        self.available_backends['fake_lagos'] = {
            'backend': AerSimulator.from_backend(FakeLagos()),
            'type': 'simulator',
            'noise_free': False,
            'max_qubits': 7,
        }
    
    def get_backend(
        self,
        provider: str = "simulator",
        min_qubits: int = 4,
        noise_model: Optional[str] = None,
    ):
        """Get optimal backend based on requirements."""
        if provider == "simulator":
            if noise_model == "noiseless":
                return self.available_backends['aer_simulator']['backend']
            else:
                return self.available_backends['fake_lagos']['backend']
        else:
            return self.available_backends['aer_simulator']['backend']