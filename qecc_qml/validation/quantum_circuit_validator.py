"""
Comprehensive Quantum Circuit Validation Framework

This module provides robust validation for quantum circuits in QECC-aware QML systems,
ensuring correctness, safety, and optimal performance before execution.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import json
import time
import warnings

try:
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
except ImportError:
    from qecc_qml.core.fallback_imports import QuantumCircuit, QuantumRegister, ClassicalRegister

try:
    from qiskit.quantum_info import Pauli, SparsePauliOp, Operator
except ImportError:
    from qecc_qml.core.fallback_imports import Pauli, SparsePauliOp, Operator

try:
    from qiskit.circuit import Gate, Instruction
except ImportError:
    from qecc_qml.core.fallback_imports import Gate, Instruction
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False


class ValidationSeverity(Enum):
    """Validation issue severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    """Represents a validation issue."""
    severity: ValidationSeverity
    category: str
    message: str
    location: Optional[str] = None
    suggestion: Optional[str] = None
    auto_fixable: bool = False


@dataclass
class ValidationResult:
    """Comprehensive validation result."""
    is_valid: bool
    issues: List[ValidationIssue]
    performance_score: float
    estimated_runtime: float
    resource_requirements: Dict[str, Any]
    recommendations: List[str]


class QuantumCircuitValidator:
    """
    Comprehensive quantum circuit validation framework.
    
    Validates quantum circuits for:
    - Logical correctness
    - Hardware compatibility  
    - Performance optimization
    - Security considerations
    - QECC integration
    """
    
    def __init__(self, max_qubits: int = 100, max_depth: int = 1000):
        self.max_qubits = max_qubits
        self.max_depth = max_depth
        self.validation_rules = self._initialize_validation_rules()
        self.performance_thresholds = {
            'gate_count': 10000,
            'depth': 1000,
            'cx_gate_ratio': 0.5,
            'measurement_ratio': 0.1
        }
        
    def _initialize_validation_rules(self) -> Dict[str, Any]:
        """Initialize validation rules configuration."""
        return {
            'enforce_qubit_limits': True,
            'check_gate_compatibility': True,
            'validate_measurements': True,
            'check_qecc_integration': True,
            'validate_noise_resilience': True,
            'check_security': True,
            'validate_performance': True
        }
    
    def validate_circuit(self, circuit: Any, 
                        hardware_backend: Optional[str] = None,
                        noise_model: Optional[Any] = None) -> ValidationResult:
        """
        Comprehensive circuit validation.
        
        Args:
            circuit: Quantum circuit to validate
            hardware_backend: Target hardware backend
            noise_model: Noise model for validation
            
        Returns:
            Comprehensive validation result
        """
        issues = []
        performance_score = 100.0
        
        # Basic structure validation
        structure_issues = self._validate_structure(circuit)
        issues.extend(structure_issues)
        
        # Gate-level validation
        gate_issues = self._validate_gates(circuit)
        issues.extend(gate_issues)
        
        # Hardware compatibility
        if hardware_backend:
            hardware_issues = self._validate_hardware_compatibility(circuit, hardware_backend)
            issues.extend(hardware_issues)
        
        # Performance validation
        perf_issues, perf_score = self._validate_performance(circuit)
        issues.extend(perf_issues)
        performance_score = min(performance_score, perf_score)
        
        # QECC integration validation
        qecc_issues = self._validate_qecc_integration(circuit)
        issues.extend(qecc_issues)
        
        # Security validation
        security_issues = self._validate_security(circuit)
        issues.extend(security_issues)
        
        # Noise resilience validation
        if noise_model:
            noise_issues = self._validate_noise_resilience(circuit, noise_model)
            issues.extend(noise_issues)
        
        # Calculate overall validity
        critical_issues = [i for i in issues if i.severity == ValidationSeverity.CRITICAL]
        error_issues = [i for i in issues if i.severity == ValidationSeverity.ERROR]
        is_valid = len(critical_issues) == 0 and len(error_issues) == 0
        
        # Estimate runtime and resources
        estimated_runtime = self._estimate_runtime(circuit)
        resource_requirements = self._calculate_resource_requirements(circuit)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(issues, circuit)
        
        return ValidationResult(
            is_valid=is_valid,
            issues=issues,
            performance_score=performance_score,
            estimated_runtime=estimated_runtime,
            resource_requirements=resource_requirements,
            recommendations=recommendations
        )
    
    def _validate_structure(self, circuit: Any) -> List[ValidationIssue]:
        """Validate basic circuit structure."""
        issues = []
        
        if not QISKIT_AVAILABLE:
            return self._mock_structure_validation(circuit)
        
        if not isinstance(circuit, QuantumCircuit):
            issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                category="structure",
                message="Invalid circuit type. Expected QuantumCircuit.",
                suggestion="Ensure circuit is a valid Qiskit QuantumCircuit object."
            ))
            return issues
        
        # Check qubit count
        if circuit.num_qubits > self.max_qubits:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category="structure",
                message=f"Circuit exceeds maximum qubits: {circuit.num_qubits} > {self.max_qubits}",
                suggestion=f"Reduce circuit size or increase max_qubits limit."
            ))
        
        # Check circuit depth
        if circuit.depth() > self.max_depth:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="structure",
                message=f"Circuit depth is high: {circuit.depth()} > {self.max_depth}",
                suggestion="Consider circuit optimization to reduce depth."
            ))
        
        # Check for empty circuit
        if len(circuit.data) == 0:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="structure",
                message="Circuit is empty (no operations)",
                suggestion="Add quantum operations to the circuit."
            ))
        
        return issues
    
    def _mock_structure_validation(self, circuit: Any) -> List[ValidationIssue]:
        """Mock structure validation when Qiskit unavailable."""
        issues = []
        
        if isinstance(circuit, dict):
            num_qubits = circuit.get('physical_qubits', circuit.get('num_qubits', 0))
            if num_qubits > self.max_qubits:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category="structure",
                    message=f"Circuit exceeds maximum qubits: {num_qubits} > {self.max_qubits}"
                ))
        
        return issues
    
    def _validate_gates(self, circuit: Any) -> List[ValidationIssue]:
        """Validate quantum gates in the circuit."""
        issues = []
        
        if not QISKIT_AVAILABLE:
            return []
        
        if not isinstance(circuit, QuantumCircuit):
            return []
        
        gate_counts = {}
        for instruction, qargs, cargs in circuit.data:
            gate_name = instruction.name
            gate_counts[gate_name] = gate_counts.get(gate_name, 0) + 1
            
            # Check for deprecated gates
            if gate_name in ['u1', 'u2', 'u3']:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="gates",
                    message=f"Deprecated gate '{gate_name}' found",
                    suggestion=f"Replace {gate_name} with modern gate equivalents.",
                    auto_fixable=True
                ))
            
            # Check for inefficient gate patterns
            if gate_name == 'cx' and len(qargs) == 2:
                # Check for consecutive CX gates on same qubits (potential optimization)
                pass
        
        # Check gate ratios
        total_gates = sum(gate_counts.values())
        if total_gates > 0:
            cx_ratio = gate_counts.get('cx', 0) / total_gates
            if cx_ratio > self.performance_thresholds['cx_gate_ratio']:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="gates",
                    message=f"High CNOT gate ratio: {cx_ratio:.2f}",
                    suggestion="Consider gate optimization to reduce CNOT count."
                ))
        
        return issues
    
    def _validate_hardware_compatibility(self, circuit: Any, backend: str) -> List[ValidationIssue]:
        """Validate hardware compatibility."""
        issues = []
        
        # Define hardware constraints
        hardware_constraints = {
            'ibm_quantum': {
                'max_qubits': 127,
                'native_gates': {'id', 'rz', 'sx', 'x', 'cx', 'measure'},
                'connectivity': 'heavy_hex',
                'max_shots': 8192
            },
            'google_quantum': {
                'max_qubits': 70,
                'native_gates': {'id', 'rz', 'ry', 'rx', 'cz', 'measure'},
                'connectivity': 'grid',
                'max_shots': 1000000
            },
            'ionq': {
                'max_qubits': 32,
                'native_gates': {'id', 'rx', 'ry', 'rz', 'xx', 'measure'},
                'connectivity': 'all_to_all',
                'max_shots': 1024
            }
        }
        
        if backend not in hardware_constraints:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="hardware",
                message=f"Unknown hardware backend: {backend}",
                suggestion="Use a supported backend or add backend specifications."
            ))
            return issues
        
        constraints = hardware_constraints[backend]
        
        if not QISKIT_AVAILABLE or not isinstance(circuit, QuantumCircuit):
            # Mock validation for non-Qiskit circuits
            mock_qubits = getattr(circuit, 'num_qubits', 4) if hasattr(circuit, 'num_qubits') else 4
            if mock_qubits > constraints['max_qubits']:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category="hardware",
                    message=f"Circuit requires {mock_qubits} qubits, but {backend} supports max {constraints['max_qubits']}"
                ))
            return issues
        
        # Check qubit count
        if circuit.num_qubits > constraints['max_qubits']:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category="hardware",
                message=f"Circuit requires {circuit.num_qubits} qubits, but {backend} supports max {constraints['max_qubits']}",
                suggestion=f"Reduce circuit size or use a larger backend."
            ))
        
        # Check gate compatibility
        used_gates = set(instr.name for instr, _, _ in circuit.data)
        unsupported_gates = used_gates - constraints['native_gates']
        
        if unsupported_gates:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="hardware",
                message=f"Non-native gates detected: {unsupported_gates}",
                suggestion="Gates will be transpiled, which may increase circuit depth.",
                auto_fixable=True
            ))
        
        return issues
    
    def _validate_performance(self, circuit: Any) -> Tuple[List[ValidationIssue], float]:
        """Validate circuit performance characteristics."""
        issues = []
        performance_score = 100.0
        
        if not QISKIT_AVAILABLE or not isinstance(circuit, QuantumCircuit):
            return issues, performance_score
        
        # Gate count analysis
        gate_count = len(circuit.data)
        if gate_count > self.performance_thresholds['gate_count']:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="performance",
                message=f"High gate count: {gate_count}",
                suggestion="Consider circuit optimization techniques."
            ))
            performance_score -= 20
        
        # Depth analysis
        depth = circuit.depth()
        if depth > self.performance_thresholds['depth']:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="performance", 
                message=f"High circuit depth: {depth}",
                suggestion="Optimize circuit for shallower depth."
            ))
            performance_score -= 30
        
        # Measurement analysis
        measurements = sum(1 for instr, _, _ in circuit.data if instr.name == 'measure')
        total_ops = len(circuit.data)
        measurement_ratio = measurements / total_ops if total_ops > 0 else 0
        
        if measurement_ratio > self.performance_thresholds['measurement_ratio']:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.INFO,
                category="performance",
                message=f"High measurement frequency: {measurement_ratio:.2f}",
                suggestion="Consider batching measurements for efficiency."
            ))
        
        return issues, max(0, performance_score)
    
    def _validate_qecc_integration(self, circuit: Any) -> List[ValidationIssue]:
        """Validate QECC integration aspects."""
        issues = []
        
        # Check for syndrome extraction patterns
        if QISKIT_AVAILABLE and isinstance(circuit, QuantumCircuit):
            # Look for stabilizer measurement patterns
            has_stabilizer_measurements = False
            
            for instr, qargs, cargs in circuit.data:
                if instr.name == 'measure' and len(cargs) > 0:
                    has_stabilizer_measurements = True
                    break
            
            if not has_stabilizer_measurements and circuit.num_qubits > 1:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    category="qecc",
                    message="No syndrome measurements detected",
                    suggestion="Consider adding error syndrome extraction for QECC."
                ))
        
        # Check for error correction sequences
        # This would involve more sophisticated pattern recognition
        # For now, provide general guidance
        
        return issues
    
    def _validate_security(self, circuit: Any) -> List[ValidationIssue]:
        """Validate security aspects of the circuit."""
        issues = []
        
        # Check for information leakage patterns
        if QISKIT_AVAILABLE and isinstance(circuit, QuantumCircuit):
            # Check for excessive measurements that might leak information
            measurement_count = sum(1 for instr, _, _ in circuit.data if instr.name == 'measure')
            
            if measurement_count > circuit.num_qubits * 2:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="security",
                    message="Excessive measurements may leak quantum information",
                    suggestion="Review measurement strategy for information security."
                ))
        
        # Check for deterministic patterns that might be vulnerable
        # This is a simplified check - real security validation would be more comprehensive
        
        return issues
    
    def _validate_noise_resilience(self, circuit: Any, noise_model: Any) -> List[ValidationIssue]:
        """Validate circuit resilience to noise."""
        issues = []
        
        if not QISKIT_AVAILABLE or not isinstance(circuit, QuantumCircuit):
            return issues
        
        # Analyze circuit susceptibility to common noise types
        depth = circuit.depth()
        gate_count = len(circuit.data)
        
        # High depth circuits are more susceptible to decoherence
        if depth > 50:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="noise_resilience",
                message=f"Circuit depth {depth} may be susceptible to decoherence",
                suggestion="Consider error mitigation techniques or circuit optimization."
            ))
        
        # Count two-qubit gates (more error-prone)
        two_qubit_gates = sum(1 for instr, qargs, _ in circuit.data if len(qargs) == 2)
        two_qubit_ratio = two_qubit_gates / gate_count if gate_count > 0 else 0
        
        if two_qubit_ratio > 0.3:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.INFO,
                category="noise_resilience",
                message=f"High two-qubit gate ratio: {two_qubit_ratio:.2f}",
                suggestion="Two-qubit gates have higher error rates. Consider optimization."
            ))
        
        return issues
    
    def _estimate_runtime(self, circuit: Any) -> float:
        """Estimate circuit execution runtime."""
        if not QISKIT_AVAILABLE or not isinstance(circuit, QuantumCircuit):
            # Mock estimation
            return 1.0
        
        # Simple runtime estimation based on gate count and depth
        base_time = 0.001  # Base overhead
        gate_time = len(circuit.data) * 0.0001  # Time per gate
        depth_penalty = circuit.depth() * 0.00001  # Parallelization limits
        
        return base_time + gate_time + depth_penalty
    
    def _calculate_resource_requirements(self, circuit: Any) -> Dict[str, Any]:
        """Calculate resource requirements for circuit execution."""
        if not QISKIT_AVAILABLE or not isinstance(circuit, QuantumCircuit):
            return {
                'qubits': 4,
                'memory': '1MB',
                'compute_units': 10
            }
        
        return {
            'qubits': circuit.num_qubits,
            'classical_bits': circuit.num_clbits,
            'gate_count': len(circuit.data),
            'circuit_depth': circuit.depth(),
            'estimated_memory': f"{circuit.num_qubits * 2}MB",
            'compute_units': len(circuit.data) + circuit.depth()
        }
    
    def _generate_recommendations(self, issues: List[ValidationIssue], 
                                circuit: Any) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        
        # Collect suggestions from issues
        for issue in issues:
            if issue.suggestion and issue.suggestion not in recommendations:
                recommendations.append(issue.suggestion)
        
        # Add general recommendations based on circuit characteristics
        if QISKIT_AVAILABLE and isinstance(circuit, QuantumCircuit):
            if circuit.depth() > 100:
                recommendations.append("Consider circuit compilation with optimization level 3")
            
            if len(circuit.data) > 1000:
                recommendations.append("Explore circuit decomposition techniques")
        
        return recommendations
    
    def auto_fix_issues(self, circuit: Any) -> Tuple[Any, List[str]]:
        """Automatically fix fixable validation issues."""
        if not QISKIT_AVAILABLE or not isinstance(circuit, QuantumCircuit):
            return circuit, ["Auto-fix not available without Qiskit"]
        
        fixes_applied = []
        fixed_circuit = circuit.copy()
        
        # Replace deprecated gates
        for i, (instr, qargs, cargs) in enumerate(fixed_circuit.data):
            if instr.name == 'u1':
                # Replace u1 with rz
                fixed_circuit.data[i] = (fixed_circuit.rz(instr.params[0], qargs[0]).data[0])
                fixes_applied.append("Replaced deprecated u1 gate with rz")
        
        return fixed_circuit, fixes_applied


def validate_qecc_circuit_comprehensive():
    """Comprehensive validation example for QECC circuits."""
    print("🔍 Comprehensive QECC Circuit Validation")
    print("=" * 50)
    
    validator = QuantumCircuitValidator(max_qubits=50, max_depth=500)
    
    # Create test circuit
    if QISKIT_AVAILABLE:
        circuit = QuantumCircuit(5, 5)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.cx(1, 2)
        circuit.measure_all()
    else:
        # Mock circuit for testing
        circuit = {
            'num_qubits': 5,
            'physical_qubits': 5,
            'gates': ['h', 'cx', 'cx', 'measure_all']
        }
    
    # Run validation
    result = validator.validate_circuit(
        circuit, 
        hardware_backend='ibm_quantum',
        noise_model=None
    )
    
    print(f"✅ Circuit Valid: {result.is_valid}")
    print(f"📊 Performance Score: {result.performance_score:.1f}/100")
    print(f"⏱️  Estimated Runtime: {result.estimated_runtime:.4f}s")
    print(f"💾 Resource Requirements: {result.resource_requirements}")
    
    print(f"\n🔍 Validation Issues ({len(result.issues)}):")
    for issue in result.issues:
        emoji = {"info": "ℹ️", "warning": "⚠️", "error": "❌", "critical": "🚨"}
        print(f"  {emoji[issue.severity.value]} [{issue.category}] {issue.message}")
        if issue.suggestion:
            print(f"    💡 {issue.suggestion}")
    
    print(f"\n💡 Recommendations ({len(result.recommendations)}):")
    for rec in result.recommendations:
        print(f"  • {rec}")
    
    # Test auto-fix
    if QISKIT_AVAILABLE:
        fixed_circuit, fixes = validator.auto_fix_issues(circuit)
        if fixes:
            print(f"\n🔧 Auto-fixes Applied ({len(fixes)}):")
            for fix in fixes:
                print(f"  • {fix}")
    
    return result


if __name__ == "__main__":
    validate_qecc_circuit_comprehensive()