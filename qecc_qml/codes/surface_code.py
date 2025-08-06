"""
Surface code implementation for quantum error correction.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import networkx as nx

from ..core.error_correction import ErrorCorrectionScheme


class SurfaceCode(ErrorCorrectionScheme):
    """
    Surface code implementation for quantum error correction.
    
    The surface code is a topological quantum error correction code that can
    correct both bit-flip and phase-flip errors. It requires a 2D lattice of
    physical qubits with distance d requiring d² qubits per logical qubit.
    """
    
    def __init__(self, distance: int = 3, logical_qubits: int = 1):
        """
        Initialize surface code.
        
        Args:
            distance: Code distance (must be odd, ≥3)
            logical_qubits: Number of logical qubits to encode
        """
        if distance < 3 or distance % 2 == 0:
            raise ValueError("Distance must be odd and ≥3")
        
        super().__init__(f"Surface Code (d={distance})")
        self.distance = distance
        self.logical_qubits = logical_qubits
        
        # Calculate qubit layout
        self.data_qubits_per_logical = distance ** 2
        self.ancilla_qubits_per_logical = distance ** 2 - 1
        self.total_qubits_per_logical = self.data_qubits_per_logical + self.ancilla_qubits_per_logical
        
        # Build lattice structure
        self._build_lattice()
        self._build_stabilizers()
    
    def _build_lattice(self):
        """Build the 2D lattice structure for the surface code."""
        # Create lattice graph
        self.lattice = nx.grid_2d_graph(self.distance, self.distance)
        
        # Assign qubit types (data qubits are on vertices, ancillas on faces/edges)
        self.data_qubit_positions = list(self.lattice.nodes())
        
        # X-type stabilizers (star operators) - on faces
        self.x_stabilizer_positions = []
        for i in range(self.distance - 1):
            for j in range(self.distance - 1):
                if (i + j) % 2 == 0:  # Checkerboard pattern
                    self.x_stabilizer_positions.append((i + 0.5, j + 0.5))
        
        # Z-type stabilizers (plaquette operators) - on faces  
        self.z_stabilizer_positions = []
        for i in range(self.distance - 1):
            for j in range(self.distance - 1):
                if (i + j) % 2 == 1:  # Opposite checkerboard pattern
                    self.z_stabilizer_positions.append((i + 0.5, j + 0.5))
    
    def _build_stabilizers(self):
        """Build stabilizer generators for the surface code."""
        self.x_stabilizers = []
        self.z_stabilizers = []
        
        # X-stabilizers: X⊗X⊗X⊗X on each face (star)
        for pos in self.x_stabilizer_positions:
            i, j = int(pos[0]), int(pos[1])
            neighbors = [
                (i, j), (i + 1, j), (i, j + 1), (i + 1, j + 1)
            ]
            # Filter valid neighbors within lattice
            neighbors = [n for n in neighbors if n in self.data_qubit_positions]
            self.x_stabilizers.append(neighbors)
        
        # Z-stabilizers: Z⊗Z⊗Z⊗Z on each plaquette
        for pos in self.z_stabilizer_positions:
            i, j = int(pos[0]), int(pos[1])
            neighbors = [
                (i, j), (i + 1, j), (i, j + 1), (i + 1, j + 1)
            ]
            neighbors = [n for n in neighbors if n in self.data_qubit_positions]
            self.z_stabilizers.append(neighbors)
    
    def get_physical_qubits(self, num_logical_qubits: int) -> int:
        """Get number of physical qubits needed."""
        return num_logical_qubits * self.total_qubits_per_logical
    
    def encode_logical_qubit(self, logical_qubit_index: int) -> QuantumCircuit:
        """
        Create encoding circuit for a logical qubit.
        
        For surface code, we encode |0⟩_L as the +1 eigenstate of all stabilizers.
        """
        num_data = self.data_qubits_per_logical
        num_ancilla = self.ancilla_qubits_per_logical
        
        qreg = QuantumRegister(num_data + num_ancilla, 'q')
        circuit = QuantumCircuit(qreg)
        
        # Initialize all data qubits in |+⟩ state
        for i in range(num_data):
            circuit.h(qreg[i])
        
        # Apply stabilizer projections to create code space
        # This is a simplified encoding - real surface codes use more sophisticated procedures
        for i, stabilizer_qubits in enumerate(self.x_stabilizers):
            if i < len(stabilizer_qubits) - 1:
                # Project onto +1 eigenspace of X-stabilizers
                ancilla_idx = num_data + i
                if ancilla_idx < num_data + num_ancilla:
                    circuit.h(qreg[ancilla_idx])
                    for data_qubit_pos in stabilizer_qubits:
                        data_idx = self.data_qubit_positions.index(data_qubit_pos)
                        circuit.cx(qreg[ancilla_idx], qreg[data_idx])
                    circuit.h(qreg[ancilla_idx])
        
        return circuit
    
    def get_syndrome_extraction_circuit(self) -> QuantumCircuit:
        """Get syndrome extraction circuit for surface code."""
        num_data = self.data_qubits_per_logical
        num_x_syndromes = len(self.x_stabilizers)
        num_z_syndromes = len(self.z_stabilizers)
        total_syndromes = num_x_syndromes + num_z_syndromes
        
        # Data qubits + syndrome ancillas
        qreg = QuantumRegister(num_data, 'q')
        anc_reg = QuantumRegister(total_syndromes, 'anc')
        creg = ClassicalRegister(total_syndromes, 'syn')
        
        circuit = QuantumCircuit(qreg, anc_reg, creg)
        
        # Initialize syndrome ancillas
        for i in range(total_syndromes):
            circuit.reset(anc_reg[i])
        
        # X-syndrome extraction
        for i, stabilizer_qubits in enumerate(self.x_stabilizers):
            # Initialize ancilla in |+⟩
            circuit.h(anc_reg[i])
            
            # Apply controlled-X gates
            for data_qubit_pos in stabilizer_qubits:
                if data_qubit_pos in self.data_qubit_positions:
                    data_idx = self.data_qubit_positions.index(data_qubit_pos)
                    if data_idx < num_data:
                        circuit.cx(anc_reg[i], qreg[data_idx])
            
            # Measure in X basis
            circuit.h(anc_reg[i])
            circuit.measure(anc_reg[i], creg[i])
        
        # Z-syndrome extraction
        syndrome_offset = num_x_syndromes
        for i, stabilizer_qubits in enumerate(self.z_stabilizers):
            ancilla_idx = syndrome_offset + i
            if ancilla_idx < total_syndromes:
                # Apply controlled-Z gates (implemented as CX with H gates)
                for data_qubit_pos in stabilizer_qubits:
                    if data_qubit_pos in self.data_qubit_positions:
                        data_idx = self.data_qubit_positions.index(data_qubit_pos)
                        if data_idx < num_data:
                            circuit.h(qreg[data_idx])
                            circuit.cx(anc_reg[ancilla_idx], qreg[data_idx])
                            circuit.h(qreg[data_idx])
                
                # Measure ancilla
                circuit.measure(anc_reg[ancilla_idx], creg[ancilla_idx])
        
        return circuit
    
    def decode_syndrome(self, syndrome: str) -> List[Tuple[str, int]]:
        """
        Decode syndrome using minimum weight perfect matching.
        
        This is a simplified decoder - real surface codes use sophisticated
        decoders like Union-Find or MWPM with Blossom algorithm.
        """
        if len(syndrome) != len(self.x_stabilizers) + len(self.z_stabilizers):
            return []
        
        errors = []
        
        # Decode X-syndrome (for Z errors)
        x_syndrome = syndrome[:len(self.x_stabilizers)]
        x_violations = [i for i, bit in enumerate(x_syndrome) if bit == '1']
        
        if len(x_violations) % 2 == 1:
            # Odd number of violations - likely boundary effect or logical error
            x_violations.append(-1)  # Virtual boundary node
        
        # Simple nearest neighbor pairing for X violations
        while len(x_violations) >= 2:
            v1 = x_violations.pop(0)
            v2 = x_violations.pop(0)
            if v1 >= 0 and v2 >= 0:
                # Find path between violations and add Z errors
                path = self._find_error_path(v1, v2, 'Z')
                errors.extend(path)
        
        # Decode Z-syndrome (for X errors)
        z_syndrome = syndrome[len(self.x_stabilizers):]
        z_violations = [i for i, bit in enumerate(z_syndrome) if bit == '1']
        
        if len(z_violations) % 2 == 1:
            z_violations.append(-1)  # Virtual boundary node
        
        # Simple nearest neighbor pairing for Z violations
        while len(z_violations) >= 2:
            v1 = z_violations.pop(0)
            v2 = z_violations.pop(0)
            if v1 >= 0 and v2 >= 0:
                # Find path between violations and add X errors
                path = self._find_error_path(v1, v2, 'X')
                errors.extend(path)
        
        return errors
    
    def _find_error_path(self, v1: int, v2: int, error_type: str) -> List[Tuple[str, int]]:
        """
        Find shortest path between syndrome violations.
        
        This is a simplified implementation - real decoders use more sophisticated
        algorithms like minimum weight perfect matching.
        """
        # For simplicity, just return a single error at the midpoint
        if error_type == 'Z' and v1 < len(self.x_stabilizer_positions) and v2 < len(self.x_stabilizer_positions):
            # Z error affecting X stabilizers
            pos1 = self.x_stabilizer_positions[v1]
            pos2 = self.x_stabilizer_positions[v2]
            mid_pos = ((pos1[0] + pos2[0]) / 2, (pos1[1] + pos2[1]) / 2)
            
            # Find nearest data qubit
            nearest_data = min(
                self.data_qubit_positions,
                key=lambda p: (p[0] - mid_pos[0])**2 + (p[1] - mid_pos[1])**2
            )
            data_idx = self.data_qubit_positions.index(nearest_data)
            return [('Z', data_idx)]
        
        elif error_type == 'X' and v1 < len(self.z_stabilizer_positions) and v2 < len(self.z_stabilizer_positions):
            # X error affecting Z stabilizers
            pos1 = self.z_stabilizer_positions[v1]
            pos2 = self.z_stabilizer_positions[v2]
            mid_pos = ((pos1[0] + pos2[0]) / 2, (pos1[1] + pos2[1]) / 2)
            
            # Find nearest data qubit
            nearest_data = min(
                self.data_qubit_positions,
                key=lambda p: (p[0] - mid_pos[0])**2 + (p[1] - mid_pos[1])**2
            )
            data_idx = self.data_qubit_positions.index(nearest_data)
            return [('X', data_idx)]
        
        return []
    
    def get_recovery_circuit(self, errors: List[Tuple[str, int]]) -> QuantumCircuit:
        """Get recovery circuit for given errors."""
        qreg = QuantumRegister(self.data_qubits_per_logical, 'q')
        circuit = QuantumCircuit(qreg)
        
        for error_type, qubit_idx in errors:
            if qubit_idx < self.data_qubits_per_logical:
                if error_type == 'X':
                    circuit.x(qreg[qubit_idx])
                elif error_type == 'Z':
                    circuit.z(qreg[qubit_idx])
                elif error_type == 'Y':
                    circuit.y(qreg[qubit_idx])
        
        return circuit
    
    def _get_num_syndrome_qubits(self) -> int:
        return len(self.x_stabilizers) + len(self.z_stabilizers)
    
    def _get_physical_qubits_for_logical(self, logical_index: int) -> List[int]:
        start_idx = logical_index * self.total_qubits_per_logical
        return list(range(start_idx, start_idx + self.data_qubits_per_logical))
    
    def _get_logical_to_physical_mapping(self, num_logical: int) -> Dict[int, List[int]]:
        mapping = {}
        for i in range(num_logical):
            mapping[i] = self._get_physical_qubits_for_logical(i)
        return mapping
    
    def get_error_threshold(self) -> float:
        """Surface code threshold is approximately 1%."""
        return 0.01
    
    def get_code_distance(self) -> int:
        return self.distance
    
    def get_logical_operators(self) -> Dict[str, List[int]]:
        """
        Get logical operators for the surface code.
        
        Returns:
            Dictionary with 'X' and 'Z' logical operators
        """
        # Logical X: chain across one boundary
        logical_x = list(range(0, self.distance))
        
        # Logical Z: chain across perpendicular boundary  
        logical_z = list(range(0, self.data_qubits_per_logical, self.distance))
        
        return {'X': logical_x, 'Z': logical_z}
    
    def __str__(self) -> str:
        return (f"SurfaceCode(distance={self.distance}, "
                f"data_qubits={self.data_qubits_per_logical}, "
                f"ancilla_qubits={self.ancilla_qubits_per_logical})")
    
    def __repr__(self) -> str:
        return self.__str__()