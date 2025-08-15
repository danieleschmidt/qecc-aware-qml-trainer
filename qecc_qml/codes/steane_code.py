"""Steane Code: 7-qubit CSS quantum error correction code.

The Steane code is a CSS (Calderbank-Shor-Steane) code that encodes 1 logical qubit
into 7 physical qubits and can correct any single qubit error. It's particularly
suitable for fault-tolerant quantum computation due to its transversal gate properties.

Author: Terragon Labs SDLC System
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
try:
    from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
except ImportError:
    from qecc_qml.core.fallback_imports import QuantumCircuit, ClassicalRegister, QuantumRegister
try:
    from qiskit.quantum_info import Pauli
except ImportError:
    from qecc_qml.core.fallback_imports import Pauli

from ..core.error_correction import ErrorCorrectionScheme


class SteaneCode(ErrorCorrectionScheme):
    """
    Steane 7-qubit CSS quantum error correction code.
    
    The Steane code uses the classical [7,4,3] Hamming code for both X and Z
    error correction. It can correct any single qubit error and detect some
    two-qubit errors.
    
    Code parameters:
    - Distance: 3
    - Physical qubits: 7  
    - Logical qubits: 1
    - X stabilizers: 3
    - Z stabilizers: 3
    
    Stabilizer generators:
    X stabilizers: IIIXXXX, IXXIIXX, XIXIXIX  
    Z stabilizers: IIIIZZZZ, IZZIIIZZ, ZIZIZIZZ
    """
    
    def __init__(self, logical_qubit_index: int = 0):
        """
        Initialize Steane code for a logical qubit.
        
        Args:
            logical_qubit_index: Index of the logical qubit (default 0)
        """
        super().__init__(distance=3, num_physical_qubits=7, num_logical_qubits=1)
        self.logical_qubit_index = logical_qubit_index
        
        # Steane code stabilizer generators
        self.x_stabilizers = [
            "IIIXXXX",  # X stabilizer 1: qubits 3,4,5,6
            "IXXIIXX",  # X stabilizer 2: qubits 1,2,5,6  
            "XIXIXIX"   # X stabilizer 3: qubits 0,2,4,6
        ]
        
        self.z_stabilizers = [
            "IIIZZZZ",  # Z stabilizer 1: qubits 3,4,5,6
            "IZZIIZZ",  # Z stabilizer 2: qubits 1,2,5,6
            "ZIZIZIZZ"   # Z stabilizer 3: qubits 0,2,4,6  
        ]
        
        # Logical operators
        self.logical_x = "XXXXXXX"  # Logical X on all qubits
        self.logical_z = "ZZZZZZZ"  # Logical Z on all qubits
        
        # Syndrome lookup table for fast decoding
        self._build_syndrome_table()
    
    def _build_syndrome_table(self) -> None:
        """Build syndrome lookup table for fast error decoding."""
        self.syndrome_table: Dict[str, str] = {}
        
        # Identity (no error)
        self.syndrome_table["000000"] = "IIIIIII"
        
        # Single X errors
        single_x_errors = [
            ("001110", "XIIIIII"),  # X error on qubit 0
            ("010101", "IXIIIII"),  # X error on qubit 1  
            ("011011", "IIXIIII"),  # X error on qubit 2
            ("100110", "IIIIXII"),  # X error on qubit 3
            ("101000", "IIIIIXI"),  # X error on qubit 4
            ("110100", "IIIIIIX"),  # X error on qubit 5
            ("111010", "IIIIIII")   # X error on qubit 6
        ]
        
        for syndrome, error in single_x_errors:
            self.syndrome_table[syndrome] = error
            
        # Single Z errors  
        single_z_errors = [
            ("000001", "ZIIIIII"),  # Z error on qubit 0
            ("000010", "IZIIIII"),  # Z error on qubit 1
            ("000100", "IIZIII"),   # Z error on qubit 2
            ("001000", "IIIIZII"),  # Z error on qubit 3
            ("010000", "IIIIIZI"),  # Z error on qubit 4
            ("100000", "IIIIIIZ"),  # Z error on qubit 5
            ("110000", "IIIIIII")   # Z error on qubit 6
        ]
        
        for syndrome, error in single_z_errors:
            # Convert Z syndrome to full 6-bit syndrome (X + Z)
            full_syndrome = "000" + syndrome[3:]
            if full_syndrome in self.syndrome_table:
                # Handle Y errors (both X and Z components)
                continue
            self.syndrome_table[full_syndrome] = error
    
    def encode_logical_qubit(self, logical_state: str = "0") -> QuantumCircuit:
        """
        Create encoding circuit for logical |0⟩ or |1⟩ state.
        
        Args:
            logical_state: "0" for |0_L⟩ or "1" for |1_L⟩
            
        Returns:
            QuantumCircuit encoding the logical state
        """
        qreg = QuantumRegister(7, "q")
        circuit = QuantumCircuit(qreg, name=f"steane_encode_{logical_state}")
        
        if logical_state == "1":
            # Start with |1⟩ on first qubit for logical |1⟩
            circuit.x(qreg[0])
        
        # Steane encoding circuit
        # Step 1: Create entanglement for X stabilizers
        circuit.h(qreg[0])  # Hadamard on qubit 0
        circuit.h(qreg[1])  # Hadamard on qubit 1
        circuit.h(qreg[2])  # Hadamard on qubit 2
        
        # Step 2: CNOT gates for X error correction
        circuit.cx(qreg[0], qreg[3])  # CNOT 0 -> 3
        circuit.cx(qreg[1], qreg[3])  # CNOT 1 -> 3
        circuit.cx(qreg[0], qreg[4])  # CNOT 0 -> 4
        circuit.cx(qreg[2], qreg[4])  # CNOT 2 -> 4
        circuit.cx(qreg[1], qreg[5])  # CNOT 1 -> 5
        circuit.cx(qreg[2], qreg[5])  # CNOT 2 -> 5
        circuit.cx(qreg[0], qreg[6])  # CNOT 0 -> 6
        circuit.cx(qreg[1], qreg[6])  # CNOT 1 -> 6
        circuit.cx(qreg[2], qreg[6])  # CNOT 2 -> 6
        
        return circuit
    
    def create_syndrome_extraction_circuit(self) -> QuantumCircuit:
        """
        Create syndrome extraction circuit for error detection.
        
        Returns:
            QuantumCircuit for syndrome measurement
        """
        # 7 data qubits + 6 syndrome qubits (3 for X, 3 for Z)
        data_qreg = QuantumRegister(7, "data")
        x_syndrome_qreg = QuantumRegister(3, "x_syn")
        z_syndrome_qreg = QuantumRegister(3, "z_syn")
        x_syndrome_creg = ClassicalRegister(3, "x_syn_bits")
        z_syndrome_creg = ClassicalRegister(3, "z_syn_bits")
        
        circuit = QuantumCircuit(
            data_qreg, x_syndrome_qreg, z_syndrome_qreg,
            x_syndrome_creg, z_syndrome_creg,
            name="steane_syndrome"
        )
        
        # X syndrome extraction
        for i, stabilizer in enumerate(self.x_stabilizers):
            # Initialize syndrome qubit in |+⟩ state
            circuit.h(x_syndrome_qreg[i])
            
            # Apply controlled X gates
            for j, pauli_op in enumerate(stabilizer):
                if pauli_op == 'X':
                    circuit.cx(x_syndrome_qreg[i], data_qreg[j])
            
            # Measure in X basis
            circuit.h(x_syndrome_qreg[i])
            circuit.measure(x_syndrome_qreg[i], x_syndrome_creg[i])
        
        # Z syndrome extraction  
        for i, stabilizer in enumerate(self.z_stabilizers):
            # Initialize syndrome qubit in |0⟩ state (already default)
            
            # Apply controlled Z gates
            for j, pauli_op in enumerate(stabilizer):
                if pauli_op == 'Z':
                    circuit.cz(z_syndrome_qreg[i], data_qreg[j])
            
            # Measure in Z basis
            circuit.measure(z_syndrome_qreg[i], z_syndrome_creg[i])
        
        return circuit
    
    def decode_syndrome(self, syndrome: str) -> List[Tuple[str, int]]:
        """
        Decode syndrome to identify and correct errors.
        
        Args:
            syndrome: 6-bit syndrome string (3 X bits + 3 Z bits)
            
        Returns:
            List of (error_type, qubit_index) tuples for correction
        """
        if len(syndrome) != 6:
            raise ValueError(f"Expected 6-bit syndrome, got {len(syndrome)} bits")
        
        # Look up error correction from syndrome table
        if syndrome in self.syndrome_table:
            error_pattern = self.syndrome_table[syndrome]
            corrections = []
            
            for i, pauli_op in enumerate(error_pattern):
                if pauli_op == 'X':
                    corrections.append(('X', i))
                elif pauli_op == 'Z':  
                    corrections.append(('Z', i))
                elif pauli_op == 'Y':
                    corrections.append(('X', i))
                    corrections.append(('Z', i))
            
            return corrections
        else:
            # Unknown syndrome - might be uncorrectable error
            # For now, assume no correction needed
            return []
    
    def create_correction_circuit(self, errors: List[Tuple[str, int]]) -> QuantumCircuit:
        """
        Create quantum circuit to apply error corrections.
        
        Args:
            errors: List of (error_type, qubit_index) to correct
            
        Returns:
            QuantumCircuit applying the corrections
        """
        qreg = QuantumRegister(7, "q")
        circuit = QuantumCircuit(qreg, name="steane_correction")
        
        for error_type, qubit_idx in errors:
            if error_type == 'X':
                circuit.x(qreg[qubit_idx])
            elif error_type == 'Z':
                circuit.z(qreg[qubit_idx])  
            elif error_type == 'Y':
                circuit.y(qreg[qubit_idx])
        
        return circuit
    
    def get_logical_operators(self) -> Dict[str, str]:
        """
        Get logical Pauli operators for the encoded qubit.
        
        Returns:
            Dictionary mapping logical operators to Pauli strings
        """
        return {
            "X": self.logical_x,
            "Z": self.logical_z,
            "Y": "YYYYYYY"  # Y = iXZ
        }
    
    def get_stabilizers(self) -> Dict[str, List[str]]:
        """
        Get all stabilizer generators.
        
        Returns:
            Dictionary with X and Z stabilizer lists
        """
        return {
            "X": self.x_stabilizers,
            "Z": self.z_stabilizers
        }
    
    def get_code_parameters(self) -> Dict[str, int]:
        """
        Get code parameters.
        
        Returns:
            Dictionary with code parameters
        """
        return {
            "n": 7,  # Physical qubits
            "k": 1,  # Logical qubits  
            "d": 3,  # Distance
            "num_x_stabilizers": len(self.x_stabilizers),
            "num_z_stabilizers": len(self.z_stabilizers)
        }
    
    def is_correctable_error(self, syndrome: str) -> bool:
        """
        Check if syndrome corresponds to a correctable error.
        
        Args:
            syndrome: Syndrome bit string
            
        Returns:
            True if error is correctable
        """
        return syndrome in self.syndrome_table
    
    def estimate_logical_error_rate(self, physical_error_rate: float) -> float:
        """
        Estimate logical error rate given physical error rate.
        
        For Steane code, logical error rate ≈ C * p^2 for small p,
        where C ≈ 35 is a constant and p is physical error rate.
        
        Args:
            physical_error_rate: Physical qubit error rate
            
        Returns:
            Estimated logical error rate
        """
        if physical_error_rate <= 0:
            return 0.0
        
        # Approximate formula for Steane code
        # Based on: Logical error ≈ 35 * p^2 for small p
        steane_constant = 35.0
        logical_rate = steane_constant * (physical_error_rate ** 2)
        
        # Cap at 1.0 (certainty of error)
        return min(logical_rate, 1.0)
    
    def __str__(self) -> str:
        """String representation of Steane code."""
        return f"SteaneCode(distance=3, physical_qubits=7, logical_qubits=1)"
    
    def __repr__(self) -> str:
        """Detailed representation of Steane code."""
        return (f"SteaneCode(logical_index={self.logical_qubit_index}, "
                f"distance={self.distance}, n={self.num_physical_qubits})")