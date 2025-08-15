"""
Robust circuit validation and security measures.
"""

from typing import List, Dict, Any, Optional, Tuple
import warnings
import numpy as np
try:
    from qiskit import QuantumCircuit
except ImportError:
    from qecc_qml.core.fallback_imports import QuantumCircuit
try:
    from qiskit.quantum_info import Operator, process_fidelity
except ImportError:
    from qecc_qml.core.fallback_imports import Operator, process_fidelity

from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class CircuitValidator:
    """
    Validates quantum circuits for security, correctness, and performance.
    """
    
    def __init__(self, max_qubits: int = 100, max_depth: int = 1000):
        self.max_qubits = max_qubits
        self.max_depth = max_depth
        self.security_checks_enabled = True
        
    def validate_circuit(self, circuit: QuantumCircuit) -> Dict[str, Any]:
        """
        Comprehensive circuit validation.
        
        Returns:
            Dict with validation results and recommendations
        """
        results = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "recommendations": [],
            "metrics": {}
        }
        
        try:
            # Basic structure validation
            self._validate_structure(circuit, results)
            
            # Security validation
            if self.security_checks_enabled:
                self._validate_security(circuit, results)
            
            # Performance validation
            self._validate_performance(circuit, results)
            
            # QECC compatibility validation
            self._validate_qecc_compatibility(circuit, results)
            
            # Calculate circuit metrics
            results["metrics"] = self._calculate_metrics(circuit)
            
        except Exception as e:
            logger.error(f"Circuit validation failed: {str(e)}")
            results["valid"] = False
            results["errors"].append(f"Validation error: {str(e)}")
            
        return results
    
    def _validate_structure(self, circuit: QuantumCircuit, results: Dict):
        """Validate basic circuit structure."""
        
        # Check qubit count limits
        if circuit.num_qubits > self.max_qubits:
            results["errors"].append(f"Too many qubits: {circuit.num_qubits} > {self.max_qubits}")
            results["valid"] = False
            
        # Check circuit depth limits  
        if circuit.depth() > self.max_depth:
            results["errors"].append(f"Circuit too deep: {circuit.depth()} > {self.max_depth}")
            results["valid"] = False
            
        # Check for empty circuit
        if len(circuit.data) == 0:
            results["warnings"].append("Empty circuit detected")
            
        # Check for unused qubits
        used_qubits = set()
        for instruction in circuit.data:
            for qubit in instruction.qubits:
                used_qubits.add(circuit.find_bit(qubit).index)
                
        unused = set(range(circuit.num_qubits)) - used_qubits
        if unused:
            results["warnings"].append(f"Unused qubits: {unused}")
            
    def _validate_security(self, circuit: QuantumCircuit, results: Dict):
        """Validate circuit security."""
        
        # Check for suspicious patterns
        gate_counts = {}
        for instruction in circuit.data:
            gate_name = instruction.operation.name
            gate_counts[gate_name] = gate_counts.get(gate_name, 0) + 1
            
        # Flag excessive rotation gates (potential side-channel attacks)
        rotation_gates = ['rx', 'ry', 'rz', 'u1', 'u2', 'u3']
        total_rotations = sum(gate_counts.get(gate, 0) for gate in rotation_gates)
        
        if total_rotations > circuit.num_qubits * 50:  # Heuristic threshold
            results["warnings"].append(f"Excessive rotation gates: {total_rotations}")
            
        # Check for potential information leakage patterns
        if 'measure' in gate_counts and gate_counts['measure'] > circuit.num_qubits:
            results["warnings"].append("Multiple measurements detected - check for information leakage")
            
    def _validate_performance(self, circuit: QuantumCircuit, results: Dict):
        """Validate circuit performance characteristics."""
        
        # Check circuit depth efficiency
        if circuit.depth() > circuit.num_qubits * 10:  # Heuristic
            results["recommendations"].append("Consider circuit optimization - high depth detected")
            
        # Check for redundant gates
        consecutive_gates = []
        prev_gate = None
        for instruction in circuit.data:
            if prev_gate and instruction.operation.name == prev_gate:
                consecutive_gates.append(instruction.operation.name)
            prev_gate = instruction.operation.name
            
        if consecutive_gates:
            results["recommendations"].append(f"Potential redundant gates: {set(consecutive_gates)}")
            
    def _validate_qecc_compatibility(self, circuit: QuantumCircuit, results: Dict):
        """Validate QECC compatibility."""
        
        # Check if circuit structure is compatible with common QECC schemes
        num_qubits = circuit.num_qubits
        
        # Surface code compatibility (requires rectangular grid)
        if num_qubits >= 9:  # Minimum for distance-3 surface code
            surface_compatible = self._check_surface_code_compatibility(num_qubits)
            if surface_compatible:
                results["recommendations"].append("Circuit compatible with surface code")
            else:
                results["warnings"].append("Circuit may not be optimal for surface code")
                
        # Steane code compatibility (7 qubits)
        if num_qubits == 7:
            results["recommendations"].append("Circuit compatible with Steane code")
            
        # Check for syndrome extraction compatibility
        if circuit.num_clbits == 0:
            results["warnings"].append("No classical registers - syndrome extraction may be limited")
            
    def _check_surface_code_compatibility(self, num_qubits: int) -> bool:
        """Check if qubit count is compatible with surface codes."""
        # Surface code requires (2d+1)^2 qubits for distance d
        valid_sizes = [9, 25, 49, 81]  # d=1,2,3,4
        return num_qubits in valid_sizes
        
    def _calculate_metrics(self, circuit: QuantumCircuit) -> Dict[str, Any]:
        """Calculate circuit performance metrics."""
        
        metrics = {
            "num_qubits": circuit.num_qubits,
            "num_clbits": circuit.num_clbits,
            "depth": circuit.depth(),
            "size": circuit.size(),
            "width": circuit.width(),
        }
        
        # Gate type distribution
        gate_counts = {}
        for instruction in circuit.data:
            gate_name = instruction.operation.name
            gate_counts[gate_name] = gate_counts.get(gate_name, 0) + 1
            
        metrics["gate_distribution"] = gate_counts
        
        # Estimate resource requirements
        metrics["estimated_runtime"] = self._estimate_runtime(circuit)
        metrics["estimated_fidelity_loss"] = self._estimate_fidelity_loss(circuit)
        
        return metrics
        
    def _estimate_runtime(self, circuit: QuantumCircuit) -> float:
        """Estimate circuit runtime in microseconds."""
        # Rough estimates based on typical gate times
        gate_times = {
            'x': 0.02, 'y': 0.02, 'z': 0.01,
            'rx': 0.02, 'ry': 0.02, 'rz': 0.01,
            'cx': 0.5, 'cz': 0.4, 'ccx': 1.0,
            'measure': 1.0, 'reset': 2.0
        }
        
        total_time = 0.0
        for instruction in circuit.data:
            gate_name = instruction.operation.name
            gate_time = gate_times.get(gate_name, 0.1)  # Default for unknown gates
            total_time += gate_time
            
        return total_time
        
    def _estimate_fidelity_loss(self, circuit: QuantumCircuit) -> float:
        """Estimate fidelity loss from gate errors."""
        # Typical error rates
        single_qubit_error = 1e-4
        two_qubit_error = 1e-3
        
        total_error = 0.0
        for instruction in circuit.data:
            num_qubits = len(instruction.qubits)
            if num_qubits == 1:
                total_error += single_qubit_error
            elif num_qubits == 2:
                total_error += two_qubit_error
            else:
                total_error += two_qubit_error * (num_qubits - 1)  # Approximate
                
        # Convert to fidelity (first order approximation)
        fidelity_loss = min(total_error, 0.99)  # Cap at 99% loss
        return fidelity_loss


class SecurityManager:
    """
    Manages security aspects of quantum circuit execution.
    """
    
    def __init__(self):
        self.allowed_gates = {
            # Standard gates
            'x', 'y', 'z', 'h', 's', 't', 'sdg', 'tdg',
            'rx', 'ry', 'rz', 'u1', 'u2', 'u3',
            'cx', 'cy', 'cz', 'ch', 'crx', 'cry', 'crz',
            'ccx', 'cswap', 'swap',
            # Measurements and resets
            'measure', 'reset',
            # Barriers and delays
            'barrier', 'delay'
        }
        self.security_enabled = True
        
    def sanitize_circuit(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """
        Sanitize circuit by removing potentially dangerous operations.
        """
        if not self.security_enabled:
            return circuit
            
        sanitized = QuantumCircuit(circuit.num_qubits, circuit.num_clbits)
        
        for instruction in circuit.data:
            gate_name = instruction.operation.name.lower()
            
            # Only allow whitelisted gates
            if gate_name in self.allowed_gates:
                sanitized.append(instruction.operation, instruction.qubits, instruction.clbits)
            else:
                logger.warning(f"Blocked potentially unsafe gate: {gate_name}")
                
        return sanitized
        
    def validate_parameters(self, parameters: List[float]) -> bool:
        """
        Validate parameter values for safety.
        """
        for param in parameters:
            # Check for NaN or infinite values
            if not np.isfinite(param):
                logger.error(f"Invalid parameter value: {param}")
                return False
                
            # Check for suspiciously large values
            if abs(param) > 100 * np.pi:
                logger.warning(f"Large parameter value detected: {param}")
                
        return True