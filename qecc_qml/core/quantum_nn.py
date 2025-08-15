"""
Core quantum neural network implementation with QECC awareness.
"""

from typing import Optional, List, Union, Dict, Any
import numpy as np

# Import with fallback support
from .fallback_imports import (
    QuantumCircuit, QuantumRegister, ClassicalRegister,
    Parameter, ParameterVector, SparsePauliOp, AerSimulator
)

from .error_correction import ErrorCorrectionScheme
from .noise_models import NoiseModel


class QECCAwareQNN:
    """
    Quantum Error Correction-Aware Quantum Neural Network.
    
    A quantum neural network that integrates error correction codes
    for improved noise resilience during training and inference.
    """
    
    def __init__(
        self,
        num_qubits: int,
        num_layers: int = 3,
        entanglement: str = "circular",
        feature_map: str = "amplitude_encoding",
        rotation_gates: List[str] = None,
        insert_barriers: bool = True,
    ):
        """
        Initialize QECC-aware QNN.
        
        Args:
            num_qubits: Number of logical qubits for the QNN
            num_layers: Number of variational layers
            entanglement: Entanglement pattern ('circular', 'linear', 'full')
            feature_map: Feature encoding method
            rotation_gates: Gates to use for rotations ['rx', 'ry', 'rz']
            insert_barriers: Whether to insert barriers between layers
        """
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.entanglement = entanglement
        self.feature_map = feature_map
        self.rotation_gates = rotation_gates or ['rx', 'ry', 'rz']
        self.insert_barriers = insert_barriers
        
        # Error correction
        self.error_correction: Optional[ErrorCorrectionScheme] = None
        self.num_physical_qubits = num_qubits  # Will be updated when QECC is added
        
        # Parameters
        self.feature_params = ParameterVector('x', self.num_qubits)
        self.weight_params = ParameterVector(
            'θ', 
            self.num_layers * len(self.rotation_gates) * self.num_qubits
        )
        
        # Circuit components
        self._feature_circuit = None
        self._variational_circuit = None
        self._full_circuit = None
        
        # Initialize circuits
        self._build_feature_map()
        self._build_variational_circuit()
        self._build_full_circuit()
    
    def add_error_correction(
        self,
        scheme: ErrorCorrectionScheme,
        syndrome_extraction_frequency: int = 1,
        decoder: str = "minimum_weight_matching"
    ):
        """
        Add error correction to the quantum neural network.
        
        Args:
            scheme: Error correction scheme to use
            syndrome_extraction_frequency: How often to extract syndromes
            decoder: Decoding algorithm to use
        """
        self.error_correction = scheme
        self.num_physical_qubits = scheme.get_physical_qubits(self.num_qubits)
        
        # Rebuild circuits with error correction
        self._build_full_circuit()
    
    def _build_feature_map(self):
        """Build the feature encoding circuit."""
        qreg = QuantumRegister(self.num_qubits, 'q')
        circuit = QuantumCircuit(qreg)
        
        if self.feature_map == "amplitude_encoding":
            # Simple amplitude encoding with rotation gates
            for i in range(self.num_qubits):
                circuit.ry(self.feature_params[i], qreg[i])
        
        elif self.feature_map == "angle_encoding":
            # Angle encoding in multiple rotations
            for i in range(self.num_qubits):
                circuit.rx(self.feature_params[i], qreg[i])
                circuit.rz(self.feature_params[i], qreg[i])
        
        else:
            raise ValueError(f"Unknown feature map: {self.feature_map}")
        
        self._feature_circuit = circuit
    
    def _build_variational_circuit(self):
        """Build the variational quantum circuit."""
        qreg = QuantumRegister(self.num_qubits, 'q')
        circuit = QuantumCircuit(qreg)
        
        param_idx = 0
        
        for layer in range(self.num_layers):
            # Rotation gates
            for gate in self.rotation_gates:
                for qubit in range(self.num_qubits):
                    if gate == 'rx':
                        circuit.rx(self.weight_params[param_idx], qreg[qubit])
                    elif gate == 'ry':
                        circuit.ry(self.weight_params[param_idx], qreg[qubit])
                    elif gate == 'rz':
                        circuit.rz(self.weight_params[param_idx], qreg[qubit])
                    param_idx += 1
            
            # Entangling gates
            if self.entanglement == "circular":
                for i in range(self.num_qubits):
                    circuit.cx(qreg[i], qreg[(i + 1) % self.num_qubits])
            elif self.entanglement == "linear":
                for i in range(self.num_qubits - 1):
                    circuit.cx(qreg[i], qreg[i + 1])
            elif self.entanglement == "full":
                for i in range(self.num_qubits):
                    for j in range(i + 1, self.num_qubits):
                        circuit.cx(qreg[i], qreg[j])
            
            if self.insert_barriers and layer < self.num_layers - 1:
                circuit.barrier()
        
        self._variational_circuit = circuit
    
    def _build_full_circuit(self):
        """Build the complete QNN circuit with optional error correction."""
        if self.error_correction is None:
            # Simple concatenation without error correction
            qreg = QuantumRegister(self.num_qubits, 'q')
            creg = ClassicalRegister(self.num_qubits, 'c')
            circuit = QuantumCircuit(qreg, creg)
            
            # Add feature map
            circuit.compose(self._feature_circuit, inplace=True)
            
            if self.insert_barriers:
                circuit.barrier()
            
            # Add variational circuit
            circuit.compose(self._variational_circuit, inplace=True)
            
            # Measurement
            circuit.measure(qreg, creg)
            
        else:
            # Build circuit with error correction
            circuit = self.error_correction.build_protected_circuit(
                logical_circuit=self._get_logical_circuit(),
                num_logical_qubits=self.num_qubits
            )
        
        self._full_circuit = circuit
    
    def _get_logical_circuit(self):
        """Get the logical circuit (feature map + variational)."""
        qreg = QuantumRegister(self.num_qubits, 'q')
        circuit = QuantumCircuit(qreg)
        
        circuit.compose(self._feature_circuit, inplace=True)
        if self.insert_barriers:
            circuit.barrier()
        circuit.compose(self._variational_circuit, inplace=True)
        
        return circuit
    
    def forward(
        self,
        x: np.ndarray,
        parameters: np.ndarray,
        shots: int = 1024,
        backend=None
    ) -> np.ndarray:
        """
        Forward pass through the quantum neural network.
        
        Args:
            x: Input features (batch_size, num_features)
            parameters: Variational parameters
            shots: Number of measurement shots
            backend: Quantum backend to use
            
        Returns:
            Measurement probabilities or expectation values
        """
        if backend is None:
            backend = AerSimulator()
        
        batch_size = x.shape[0] if x.ndim > 1 else 1
        x = x.reshape(batch_size, -1)
        
        results = []
        
        for i in range(batch_size):
            # Bind parameters
            param_dict = {}
            
            # Bind feature parameters
            for j, param in enumerate(self.feature_params):
                if j < x.shape[1]:
                    param_dict[param] = x[i, j]
                else:
                    param_dict[param] = 0.0
            
            # Bind variational parameters
            for j, param in enumerate(self.weight_params):
                if j < len(parameters):
                    param_dict[param] = parameters[j]
                else:
                    param_dict[param] = 0.0
            
            # Bind circuit
            bound_circuit = self._full_circuit.bind_parameters(param_dict)
            
            # Execute
            job = backend.run(bound_circuit, shots=shots)
            result = job.result()
            counts = result.get_counts(0)
            
            # Convert to probabilities
            total_shots = sum(counts.values())
            probs = np.zeros(2**self.num_qubits)
            
            for bitstring, count in counts.items():
                idx = int(bitstring, 2)
                probs[idx] = count / total_shots
            
            results.append(probs)
        
        return np.array(results)
    
    def get_circuit(self, bind_parameters: bool = False, **kwargs) -> QuantumCircuit:
        """
        Get the quantum circuit.
        
        Args:
            bind_parameters: Whether to bind parameters
            **kwargs: Parameter values to bind
            
        Returns:
            The quantum circuit
        """
        if bind_parameters:
            return self._full_circuit.bind_parameters(kwargs)
        return self._full_circuit
    
    def get_num_parameters(self) -> int:
        """Get the number of trainable parameters."""
        return len(self.weight_params)
    
    def get_circuit_depth(self) -> int:
        """Get the circuit depth."""
        return self._full_circuit.depth()
    
    def __str__(self) -> str:
        qecc_info = f" with {self.error_correction.name}" if self.error_correction else ""
        return (f"QECCAwareQNN({self.num_qubits} qubits, {self.num_layers} layers"
                f", {self.get_num_parameters()} parameters{qecc_info})")
    
    def __repr__(self) -> str:
        return self.__str__()