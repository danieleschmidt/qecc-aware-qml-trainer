"""
Enhanced Quantum backend management for automatic selection and optimization.

Provides unified interface for accessing quantum hardware and simulators
from IBM Quantum, Google Quantum AI, AWS Braket, and other providers.

Author: Terragon Labs SDLC System
"""

from typing import Optional, Dict, Any, List, Union
import warnings
import time
from dataclasses import dataclass
from qiskit_aer import AerSimulator
from qiskit.providers.fake_provider import FakeLagos, FakeBoeblingen, FakeMontreal
import numpy as np

# Optional imports for cloud providers
try:
    from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Estimator
    from qiskit_ibm_provider import IBMProvider
    IBM_AVAILABLE = True
except ImportError:
    IBM_AVAILABLE = False
    warnings.warn("IBM Quantum not available. Install qiskit-ibm-runtime for cloud access.")

try:
    import cirq
    import cirq_google
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False
    warnings.warn("Google Cirq not available. Install cirq-google for cloud access.")

try:
    from braket.aws import AwsDevice
    from braket.devices import LocalSimulator
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False
    warnings.warn("AWS Braket not available. Install amazon-braket-sdk for cloud access.")


@dataclass
class BackendInfo:
    """Information about a quantum backend."""
    name: str
    backend: Any
    provider: str
    backend_type: str  # 'simulator', 'hardware'
    num_qubits: int
    noise_free: bool
    gate_error_rate: float
    readout_error_rate: float
    coherence_time: float  # T1 in microseconds
    coupling_map: Optional[List[List[int]]] = None
    queue_length: Optional[int] = None
    calibration_time: Optional[float] = None


class EnhancedQuantumBackendManager:
    """
    Advanced quantum backend manager with automatic selection and optimization.
    
    Features:
    - Multi-provider support (IBM, Google, AWS)
    - Automatic backend selection based on requirements
    - Real-time calibration monitoring
    - Queue length optimization
    - Noise model adaptation
    - Cost-aware resource allocation
    """
    
    def __init__(self, enable_cloud: bool = False, api_tokens: Optional[Dict[str, str]] = None):
        """
        Initialize backend manager.
        
        Args:
            enable_cloud: Whether to enable cloud backend access
            api_tokens: Dict of provider API tokens {provider: token}
        """
        self.enable_cloud = enable_cloud
        self.api_tokens = api_tokens or {}
        self.available_backends: Dict[str, BackendInfo] = {}
        
        # Initialize backends
        self._initialize_simulators()
        
        if enable_cloud:
            self._initialize_cloud_backends()
        
        # Caching for backend performance
        self._backend_cache = {}
        self._last_calibration_check = {}
        self.cache_duration = 300  # 5 minutes
        
    def _initialize_simulators(self) -> None:
        """Initialize available simulator backends."""
        # Noiseless simulator
        self.available_backends['aer_simulator'] = BackendInfo(
            name='aer_simulator',
            backend=AerSimulator(),
            provider='qiskit',
            backend_type='simulator',
            num_qubits=32,
            noise_free=True,
            gate_error_rate=0.0,
            readout_error_rate=0.0,
            coherence_time=float('inf')
        )
        
        # Realistic noise simulators based on real hardware
        fake_backends = [
            ('fake_lagos', FakeLagos(), 7, 0.001, 0.01, 50.0),
            ('fake_boeblingen', FakeBoeblingen(), 20, 0.0015, 0.012, 45.0),
            ('fake_montreal', FakeMontreal(), 27, 0.002, 0.015, 40.0)
        ]
        
        for name, fake_backend, qubits, gate_err, readout_err, t1 in fake_backends:
            try:
                self.available_backends[name] = BackendInfo(
                    name=name,
                    backend=AerSimulator.from_backend(fake_backend),
                    provider='qiskit',
                    backend_type='simulator',
                    num_qubits=qubits,
                    noise_free=False,
                    gate_error_rate=gate_err,
                    readout_error_rate=readout_err,
                    coherence_time=t1,
                    coupling_map=fake_backend.coupling_map
                )
            except Exception as e:
                warnings.warn(f"Failed to initialize {name}: {e}")
                
    def _initialize_cloud_backends(self) -> None:
        """Initialize cloud quantum backends."""
        if IBM_AVAILABLE and 'ibm' in self.api_tokens:
            try:
                self._initialize_ibm_backends()
            except Exception as e:
                warnings.warn(f"Failed to initialize IBM backends: {e}")
        
        if GOOGLE_AVAILABLE and 'google' in self.api_tokens:
            try:
                self._initialize_google_backends()
            except Exception as e:
                warnings.warn(f"Failed to initialize Google backends: {e}")
        
        if AWS_AVAILABLE and 'aws' in self.api_tokens:
            try:
                self._initialize_aws_backends()
            except Exception as e:
                warnings.warn(f"Failed to initialize AWS backends: {e}")
    
    def _initialize_ibm_backends(self) -> None:
        """Initialize IBM Quantum backends."""
        try:
            service = QiskitRuntimeService(token=self.api_tokens['ibm'])
            backends = service.backends(simulator=False, operational=True)
            
            for backend in backends[:5]:  # Limit to top 5 backends
                properties = backend.properties()
                config = backend.configuration()
                
                # Calculate average gate error rate
                gate_errors = [prop.value for prop in properties.gates if hasattr(prop, 'value')]
                avg_gate_error = np.mean(gate_errors) if gate_errors else 0.001
                
                # Calculate average readout error  
                readout_errors = [prop.value for prop in properties.readout_errors if hasattr(prop, 'value')]
                avg_readout_error = np.mean(readout_errors) if readout_errors else 0.01
                
                # Get T1 coherence time
                t1_times = [prop.value for prop in properties.t1s if hasattr(prop, 'value')]
                avg_t1 = np.mean(t1_times) * 1e6 if t1_times else 50.0  # Convert to microseconds
                
                self.available_backends[backend.name] = BackendInfo(
                    name=backend.name,
                    backend=backend,
                    provider='ibm',
                    backend_type='hardware',
                    num_qubits=config.num_qubits,
                    noise_free=False,
                    gate_error_rate=avg_gate_error,
                    readout_error_rate=avg_readout_error,
                    coherence_time=avg_t1,
                    coupling_map=config.coupling_map,
                    queue_length=backend.status().pending_jobs
                )
        except Exception as e:
            warnings.warn(f"IBM backend initialization failed: {e}")
    
    def _initialize_google_backends(self) -> None:
        """Initialize Google Quantum AI backends."""
        # Placeholder for Google Quantum AI integration
        # Would require proper Google Cloud credentials and API setup
        pass
    
    def _initialize_aws_backends(self) -> None:
        """Initialize AWS Braket backends."""
        try:
            # Add AWS Braket local simulator
            self.available_backends['braket_local'] = BackendInfo(
                name='braket_local',
                backend=LocalSimulator(),
                provider='aws',
                backend_type='simulator',
                num_qubits=25,
                noise_free=True,
                gate_error_rate=0.0,
                readout_error_rate=0.0,
                coherence_time=float('inf')
            )
            
            # Could add cloud devices like IonQ, Rigetti, etc.
            # device = AwsDevice("arn:aws:braket:::device/quantum-simulator/amazon/sv1")
            
        except Exception as e:
            warnings.warn(f"AWS backend initialization failed: {e}")
    
    def get_backend(
        self,
        provider: str = "auto",
        min_qubits: int = 4,
        max_queue_time: Optional[int] = None,
        noise_model: Optional[str] = None,
        calibration_threshold: float = 0.95,
        cost_optimization: bool = True
    ) -> BackendInfo:
        """
        Get optimal backend based on requirements.
        
        Args:
            provider: Provider preference ('auto', 'ibm', 'google', 'aws', 'simulator')
            min_qubits: Minimum number of qubits required
            max_queue_time: Maximum acceptable queue time in seconds
            noise_model: Noise preference ('noiseless', 'realistic', 'hardware')
            calibration_threshold: Minimum calibration quality (0-1)
            cost_optimization: Whether to optimize for cost
            
        Returns:
            BackendInfo for the selected backend
        """
        candidates = self._filter_backends(
            min_qubits=min_qubits,
            max_queue_time=max_queue_time,
            noise_model=noise_model,
            calibration_threshold=calibration_threshold,
            provider=provider
        )
        
        if not candidates:
            # Fallback to basic simulator
            return self.available_backends['aer_simulator']
        
        # Score and rank candidates
        best_backend = self._select_optimal_backend(
            candidates, cost_optimization=cost_optimization
        )
        
        return best_backend
    
    def _filter_backends(
        self,
        min_qubits: int,
        max_queue_time: Optional[int],
        noise_model: Optional[str],
        calibration_threshold: float,
        provider: str
    ) -> List[BackendInfo]:
        """Filter backends based on requirements."""
        candidates = []
        
        for backend_info in self.available_backends.values():
            # Check provider
            if provider != 'auto' and backend_info.provider != provider:
                continue
            
            # Check qubit count
            if backend_info.num_qubits < min_qubits:
                continue
            
            # Check noise model preference
            if noise_model == 'noiseless' and not backend_info.noise_free:
                continue
            elif noise_model == 'hardware' and backend_info.backend_type != 'hardware':
                continue
            
            # Check queue length
            if max_queue_time and backend_info.queue_length:
                if backend_info.queue_length > max_queue_time:
                    continue
            
            # Check calibration quality (for hardware backends)
            if backend_info.backend_type == 'hardware':
                fidelity = 1.0 - backend_info.gate_error_rate
                if fidelity < calibration_threshold:
                    continue
            
            candidates.append(backend_info)
        
        return candidates
    
    def _select_optimal_backend(
        self, 
        candidates: List[BackendInfo], 
        cost_optimization: bool = True
    ) -> BackendInfo:
        """Select optimal backend from candidates using scoring."""
        if len(candidates) == 1:
            return candidates[0]
        
        scores = []
        
        for backend in candidates:
            score = 0.0
            
            # Fidelity score (higher is better)
            fidelity = 1.0 - backend.gate_error_rate
            score += fidelity * 40  # Weight: 40%
            
            # Coherence time score
            if backend.coherence_time != float('inf'):
                # Normalize coherence time (50μs = good)
                coherence_score = min(backend.coherence_time / 50.0, 2.0)
                score += coherence_score * 20  # Weight: 20%
            else:
                score += 40  # Perfect for simulators
            
            # Queue length score (lower is better)
            if backend.queue_length is not None:
                queue_score = max(0, 20 - backend.queue_length / 10)
                score += queue_score * 20  # Weight: 20%
            else:
                score += 20  # No queue for simulators
            
            # Cost optimization
            if cost_optimization:
                if backend.backend_type == 'simulator':
                    score += 20  # Prefer simulators for cost
                else:
                    score += max(0, 20 - backend.num_qubits)  # Prefer smaller hardware
            
            scores.append(score)
        
        # Select backend with highest score
        best_idx = np.argmax(scores)
        return candidates[best_idx]
    
    def get_backend_info(self, backend_name: str) -> Optional[BackendInfo]:
        """Get detailed information about a specific backend."""
        return self.available_backends.get(backend_name)
    
    def list_backends(
        self, 
        provider: Optional[str] = None,
        backend_type: Optional[str] = None
    ) -> List[BackendInfo]:
        """List available backends with optional filtering."""
        backends = list(self.available_backends.values())
        
        if provider:
            backends = [b for b in backends if b.provider == provider]
        
        if backend_type:
            backends = [b for b in backends if b.backend_type == backend_type]
        
        return backends
    
    def refresh_backend_status(self, backend_name: str) -> bool:
        """Refresh calibration and queue status for a backend."""
        if backend_name not in self.available_backends:
            return False
        
        backend_info = self.available_backends[backend_name]
        
        # Only refresh hardware backends
        if backend_info.backend_type != 'hardware':
            return True
        
        try:
            if backend_info.provider == 'ibm':
                # Refresh IBM backend status
                status = backend_info.backend.status()
                properties = backend_info.backend.properties()
                
                backend_info.queue_length = status.pending_jobs
                
                # Update error rates
                gate_errors = [prop.value for prop in properties.gates if hasattr(prop, 'value')]
                if gate_errors:
                    backend_info.gate_error_rate = np.mean(gate_errors)
                
                backend_info.calibration_time = time.time()
            
            return True
        except Exception as e:
            warnings.warn(f"Failed to refresh backend {backend_name}: {e}")
            return False
    
    def get_recommended_backends(self, task_type: str = "qml") -> List[BackendInfo]:
        """
        Get recommended backends for specific tasks.
        
        Args:
            task_type: Type of task ('qml', 'optimization', 'simulation')
            
        Returns:
            List of recommended backends
        """
        if task_type == "qml":
            # For QML: prefer moderate qubit count with good fidelity
            return self._filter_backends(
                min_qubits=4,
                max_queue_time=3600,
                noise_model="realistic",
                calibration_threshold=0.98,
                provider="auto"
            )
        elif task_type == "optimization":
            # For optimization: prefer high qubit count
            return self._filter_backends(
                min_qubits=10,
                max_queue_time=7200,
                noise_model=None,
                calibration_threshold=0.95,
                provider="auto"
            )
        elif task_type == "simulation":
            # For simulation: prefer noiseless simulators
            return [b for b in self.available_backends.values() 
                   if b.backend_type == 'simulator']
        
        return list(self.available_backends.values())
    
    def estimate_execution_cost(
        self, 
        backend_name: str, 
        circuit_depth: int, 
        num_shots: int = 1024
    ) -> Dict[str, Union[float, str]]:
        """
        Estimate execution cost for a circuit.
        
        Args:
            backend_name: Name of backend
            circuit_depth: Depth of quantum circuit
            num_shots: Number of measurement shots
            
        Returns:
            Cost estimation dictionary
        """
        if backend_name not in self.available_backends:
            return {"error": "Backend not found"}
        
        backend_info = self.available_backends[backend_name]
        
        if backend_info.backend_type == 'simulator':
            return {
                "cost_usd": 0.0,
                "execution_time_seconds": circuit_depth * num_shots * 0.001,
                "cost_type": "free"
            }
        
        # Rough hardware cost estimation (IBM Quantum pricing model)
        base_cost_per_shot = 0.00085  # USD per shot
        complexity_factor = 1 + (circuit_depth / 100) * 0.5
        
        estimated_cost = base_cost_per_shot * num_shots * complexity_factor
        estimated_time = 60 + circuit_depth * 2  # seconds
        
        return {
            "cost_usd": round(estimated_cost, 4),
            "execution_time_seconds": estimated_time,
            "cost_type": "hardware",
            "shots": num_shots,
            "circuit_depth": circuit_depth
        }
    
    def __str__(self) -> str:
        """String representation."""
        return f"EnhancedQuantumBackendManager({len(self.available_backends)} backends)"
    
    def __repr__(self) -> str:
        """Detailed representation."""
        providers = set(b.provider for b in self.available_backends.values())
        return (f"EnhancedQuantumBackendManager(backends={len(self.available_backends)}, "
                f"providers={list(providers)}, cloud_enabled={self.enable_cloud})")