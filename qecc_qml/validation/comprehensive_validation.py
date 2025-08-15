"""
Comprehensive validation system for QECC-QML operations.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import traceback
import time

try:
    from qiskit import QuantumCircuit
except ImportError:
    from qecc_qml.core.fallback_imports import QuantumCircuit
try:
    from qiskit.quantum_info import Statevector, DensityMatrix
except ImportError:
    from qecc_qml.core.fallback_imports import Statevector, DensityMatrix

from ..utils.logging_config import get_logger
from ..core.circuit_validation import CircuitValidator, SecurityManager

logger = get_logger(__name__)


@dataclass
class ValidationResult:
    """Result of a validation check."""
    passed: bool
    message: str
    severity: str = "info"  # info, warning, error, critical
    details: Optional[Dict[str, Any]] = None
    timestamp: float = 0.0
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


class ComprehensiveValidator:
    """
    Comprehensive validation system for all QECC-QML operations.
    """
    
    def __init__(self, strict_mode: bool = False):
        self.strict_mode = strict_mode
        self.circuit_validator = CircuitValidator()
        self.security_manager = SecurityManager()
        self.validation_history = []
        
    def validate_training_inputs(self, 
                                X: np.ndarray, 
                                y: np.ndarray,
                                batch_size: Optional[int] = None,
                                num_epochs: Optional[int] = None) -> List[ValidationResult]:
        """Validate training data and parameters."""
        
        results = []
        
        # Input data validation
        results.extend(self._validate_input_data(X, y))
        
        # Training parameter validation
        if batch_size is not None:
            results.extend(self._validate_batch_size(batch_size, len(X)))
            
        if num_epochs is not None:
            results.extend(self._validate_epochs(num_epochs))
            
        return results
        
    def validate_quantum_circuit(self, circuit: QuantumCircuit) -> List[ValidationResult]:
        """Validate quantum circuit comprehensively."""
        
        results = []
        
        try:
            # Basic circuit validation
            circuit_results = self.circuit_validator.validate_circuit(circuit)
            
            # Convert to ValidationResult format
            if not circuit_results["valid"]:
                for error in circuit_results["errors"]:
                    results.append(ValidationResult(
                        passed=False,
                        message=error,
                        severity="error",
                        details=circuit_results
                    ))
                    
            for warning in circuit_results["warnings"]:
                results.append(ValidationResult(
                    passed=True,
                    message=warning,
                    severity="warning",
                    details=circuit_results
                ))
                
            # Security validation
            sanitized_circuit = self.security_manager.sanitize_circuit(circuit)
            if sanitized_circuit.size() != circuit.size():
                results.append(ValidationResult(
                    passed=False,
                    message="Circuit contains potentially unsafe operations",
                    severity="critical"
                ))
                
        except Exception as e:
            results.append(ValidationResult(
                passed=False,
                message=f"Circuit validation failed: {str(e)}",
                severity="error",
                details={"exception": traceback.format_exc()}
            ))
            
        return results
        
    def validate_error_correction_config(self, 
                                       qecc_config: Dict[str, Any]) -> List[ValidationResult]:
        """Validate QECC configuration."""
        
        results = []
        
        # Required fields
        required_fields = ["code_type", "distance", "logical_qubits"]
        for field in required_fields:
            if field not in qecc_config:
                results.append(ValidationResult(
                    passed=False,
                    message=f"Missing required QECC field: {field}",
                    severity="error"
                ))
                
        # Validate code parameters
        if "distance" in qecc_config:
            distance = qecc_config["distance"]
            if not isinstance(distance, int) or distance < 1:
                results.append(ValidationResult(
                    passed=False,
                    message=f"Invalid QECC distance: {distance}",
                    severity="error"
                ))
            elif distance > 9:
                results.append(ValidationResult(
                    passed=True,
                    message=f"Very high QECC distance ({distance}) may impact performance",
                    severity="warning"
                ))
                
        # Validate physical qubit requirements
        if "logical_qubits" in qecc_config and "distance" in qecc_config:
            logical_qubits = qecc_config["logical_qubits"]
            distance = qecc_config["distance"]
            
            # Estimate physical qubit requirements
            estimated_physical = self._estimate_physical_qubits(
                qecc_config["code_type"], logical_qubits, distance
            )
            
            if estimated_physical > 100:  # Arbitrary limit
                results.append(ValidationResult(
                    passed=True,
                    message=f"High physical qubit requirement: {estimated_physical}",
                    severity="warning",
                    details={"estimated_physical_qubits": estimated_physical}
                ))
                
        return results
        
    def validate_noise_model(self, noise_model: Any) -> List[ValidationResult]:
        """Validate noise model configuration."""
        
        results = []
        
        try:
            # Check if noise model has required attributes
            required_attributes = ["gate_error_rate", "readout_error_rate"]
            
            for attr in required_attributes:
                if not hasattr(noise_model, attr):
                    results.append(ValidationResult(
                        passed=False,
                        message=f"Noise model missing attribute: {attr}",
                        severity="error"
                    ))
                else:
                    # Validate error rates
                    error_rate = getattr(noise_model, attr)
                    if error_rate < 0 or error_rate > 1:
                        results.append(ValidationResult(
                            passed=False,
                            message=f"Invalid error rate {attr}: {error_rate}",
                            severity="error"
                        ))
                    elif error_rate > 0.1:  # 10% seems quite high
                        results.append(ValidationResult(
                            passed=True,
                            message=f"High error rate {attr}: {error_rate}",
                            severity="warning"
                        ))
                        
        except Exception as e:
            results.append(ValidationResult(
                passed=False,
                message=f"Noise model validation failed: {str(e)}",
                severity="error",
                details={"exception": traceback.format_exc()}
            ))
            
        return results
        
    def validate_hardware_backend(self, backend: Any) -> List[ValidationResult]:
        """Validate hardware backend configuration."""
        
        results = []
        
        try:
            # Check backend availability
            if hasattr(backend, 'status'):
                status = backend.status()
                if not status.operational:
                    results.append(ValidationResult(
                        passed=False,
                        message=f"Backend not operational: {status.status_msg}",
                        severity="critical"
                    ))
                    
                # Check queue length
                if hasattr(status, 'pending_jobs'):
                    if status.pending_jobs > 50:
                        results.append(ValidationResult(
                            passed=True,
                            message=f"High queue length: {status.pending_jobs}",
                            severity="warning"
                        ))
                        
            # Check backend configuration
            if hasattr(backend, 'configuration'):
                config = backend.configuration()
                
                # Check qubit count
                if hasattr(config, 'n_qubits'):
                    if config.n_qubits < 5:
                        results.append(ValidationResult(
                            passed=True,
                            message=f"Low qubit count: {config.n_qubits}",
                            severity="warning"
                        ))
                        
                # Check coupling map
                if hasattr(config, 'coupling_map') and config.coupling_map:
                    # Analyze connectivity
                    connectivity = self._analyze_connectivity(config.coupling_map)
                    if connectivity < 0.3:  # Low connectivity
                        results.append(ValidationResult(
                            passed=True,
                            message=f"Low qubit connectivity: {connectivity:.2f}",
                            severity="warning"
                        ))
                        
        except Exception as e:
            results.append(ValidationResult(
                passed=False,
                message=f"Backend validation failed: {str(e)}",
                severity="error",
                details={"exception": traceback.format_exc()}
            ))
            
        return results
        
    def validate_experiment_results(self, 
                                  results: Dict[str, Any],
                                  expected_accuracy: Optional[float] = None) -> List[ValidationResult]:
        """Validate experimental results."""
        
        validation_results = []
        
        # Check required result fields
        required_fields = ["accuracy", "loss", "fidelity"]
        for field in required_fields:
            if field not in results:
                validation_results.append(ValidationResult(
                    passed=False,
                    message=f"Missing result field: {field}",
                    severity="error"
                ))
            else:
                # Validate field values
                value = results[field]
                if field == "accuracy":
                    if value < 0 or value > 1:
                        validation_results.append(ValidationResult(
                            passed=False,
                            message=f"Invalid accuracy value: {value}",
                            severity="error"
                        ))
                    elif expected_accuracy and value < expected_accuracy * 0.8:
                        validation_results.append(ValidationResult(
                            passed=True,
                            message=f"Low accuracy: {value} < {expected_accuracy * 0.8}",
                            severity="warning"
                        ))
                        
                elif field == "fidelity":
                    if value < 0 or value > 1:
                        validation_results.append(ValidationResult(
                            passed=False,
                            message=f"Invalid fidelity value: {value}",
                            severity="error"
                        ))
                    elif value < 0.8:
                        validation_results.append(ValidationResult(
                            passed=True,
                            message=f"Low fidelity: {value}",
                            severity="warning"
                        ))
                        
        return validation_results
        
    def _validate_input_data(self, X: np.ndarray, y: np.ndarray) -> List[ValidationResult]:
        """Validate input training data."""
        
        results = []
        
        # Check data types
        if not isinstance(X, np.ndarray):
            results.append(ValidationResult(
                passed=False,
                message=f"X must be numpy array, got {type(X)}",
                severity="error"
            ))
            return results
            
        if not isinstance(y, np.ndarray):
            results.append(ValidationResult(
                passed=False,
                message=f"y must be numpy array, got {type(y)}",
                severity="error"
            ))
            return results
            
        # Check shapes
        if X.shape[0] != y.shape[0]:
            results.append(ValidationResult(
                passed=False,
                message=f"X and y shape mismatch: {X.shape[0]} vs {y.shape[0]}",
                severity="error"
            ))
            
        # Check for NaN or inf values
        if np.any(~np.isfinite(X)):
            results.append(ValidationResult(
                passed=False,
                message="X contains NaN or infinite values",
                severity="error"
            ))
            
        if np.any(~np.isfinite(y)):
            results.append(ValidationResult(
                passed=False,
                message="y contains NaN or infinite values",
                severity="error"
            ))
            
        # Check data size
        if X.shape[0] < 10:
            results.append(ValidationResult(
                passed=True,
                message=f"Very small dataset: {X.shape[0]} samples",
                severity="warning"
            ))
            
        # Check for data imbalance (for classification)
        if len(np.unique(y)) > 1:
            unique, counts = np.unique(y, return_counts=True)
            imbalance_ratio = max(counts) / min(counts)
            if imbalance_ratio > 10:
                results.append(ValidationResult(
                    passed=True,
                    message=f"High class imbalance: {imbalance_ratio:.1f}x",
                    severity="warning"
                ))
                
        return results
        
    def _validate_batch_size(self, batch_size: int, data_size: int) -> List[ValidationResult]:
        """Validate batch size parameter."""
        
        results = []
        
        if batch_size <= 0:
            results.append(ValidationResult(
                passed=False,
                message=f"Invalid batch size: {batch_size}",
                severity="error"
            ))
        elif batch_size > data_size:
            results.append(ValidationResult(
                passed=True,
                message=f"Batch size ({batch_size}) larger than data size ({data_size})",
                severity="warning"
            ))
        elif batch_size == 1:
            results.append(ValidationResult(
                passed=True,
                message="Batch size of 1 may lead to unstable training",
                severity="warning"
            ))
            
        return results
        
    def _validate_epochs(self, num_epochs: int) -> List[ValidationResult]:
        """Validate number of epochs."""
        
        results = []
        
        if num_epochs <= 0:
            results.append(ValidationResult(
                passed=False,
                message=f"Invalid number of epochs: {num_epochs}",
                severity="error"
            ))
        elif num_epochs > 1000:
            results.append(ValidationResult(
                passed=True,
                message=f"Very high epoch count: {num_epochs}",
                severity="warning"
            ))
            
        return results
        
    def _estimate_physical_qubits(self, code_type: str, logical_qubits: int, distance: int) -> int:
        """Estimate physical qubits required for QECC."""
        
        if code_type.lower() == "surface":
            # Surface code: roughly (2d+1)^2 per logical qubit
            return logical_qubits * (2 * distance + 1) ** 2
        elif code_type.lower() == "steane":
            # Steane code: 7 physical per logical
            return logical_qubits * 7
        elif code_type.lower() == "shor":
            # Shor code: 9 physical per logical
            return logical_qubits * 9
        else:
            # Conservative estimate
            return logical_qubits * 15
            
    def _analyze_connectivity(self, coupling_map: List[List[int]]) -> float:
        """Analyze connectivity of coupling map."""
        
        if not coupling_map:
            return 0.0
            
        # Count total possible connections
        num_qubits = max(max(pair) for pair in coupling_map) + 1
        max_connections = num_qubits * (num_qubits - 1) / 2
        
        # Count actual connections
        actual_connections = len(coupling_map)
        
        return actual_connections / max_connections
        
    def get_validation_summary(self, results: List[ValidationResult]) -> Dict[str, Any]:
        """Generate summary of validation results."""
        
        summary = {
            "total_checks": len(results),
            "passed": sum(1 for r in results if r.passed),
            "failed": sum(1 for r in results if not r.passed),
            "severity_counts": {
                "info": 0,
                "warning": 0, 
                "error": 0,
                "critical": 0
            },
            "critical_issues": [],
            "warnings": [],
            "overall_status": "pass"
        }
        
        for result in results:
            summary["severity_counts"][result.severity] += 1
            
            if result.severity == "critical":
                summary["critical_issues"].append(result.message)
            elif result.severity == "warning":
                summary["warnings"].append(result.message)
                
        # Determine overall status
        if summary["severity_counts"]["critical"] > 0:
            summary["overall_status"] = "critical"
        elif summary["severity_counts"]["error"] > 0:
            summary["overall_status"] = "error"
        elif summary["severity_counts"]["warning"] > 0:
            summary["overall_status"] = "warning"
            
        return summary
        
    def run_comprehensive_validation(self, **kwargs) -> Dict[str, Any]:
        """Run comprehensive validation on all provided components."""
        
        all_results = []
        validation_sections = {}
        
        # Validate different components based on provided kwargs
        if "X" in kwargs and "y" in kwargs:
            training_results = self.validate_training_inputs(
                kwargs["X"], kwargs["y"],
                kwargs.get("batch_size"),
                kwargs.get("num_epochs")
            )
            all_results.extend(training_results)
            validation_sections["training_data"] = training_results
            
        if "circuit" in kwargs:
            circuit_results = self.validate_quantum_circuit(kwargs["circuit"])
            all_results.extend(circuit_results)
            validation_sections["circuit"] = circuit_results
            
        if "qecc_config" in kwargs:
            qecc_results = self.validate_error_correction_config(kwargs["qecc_config"])
            all_results.extend(qecc_results)
            validation_sections["qecc"] = qecc_results
            
        if "noise_model" in kwargs:
            noise_results = self.validate_noise_model(kwargs["noise_model"])
            all_results.extend(noise_results)
            validation_sections["noise_model"] = noise_results
            
        if "backend" in kwargs:
            backend_results = self.validate_hardware_backend(kwargs["backend"])
            all_results.extend(backend_results)
            validation_sections["backend"] = backend_results
            
        if "results" in kwargs:
            result_validation = self.validate_experiment_results(
                kwargs["results"],
                kwargs.get("expected_accuracy")
            )
            all_results.extend(result_validation)
            validation_sections["results"] = result_validation
            
        # Generate comprehensive summary
        summary = self.get_validation_summary(all_results)
        
        return {
            "timestamp": time.time(),
            "validation_summary": summary,
            "sections": validation_sections,
            "all_results": all_results,
            "recommendations": self._generate_recommendations(all_results)
        }
        
    def _generate_recommendations(self, results: List[ValidationResult]) -> List[str]:
        """Generate actionable recommendations based on validation results."""
        
        recommendations = []
        
        # Analyze patterns in validation results
        error_patterns = {}
        for result in results:
            if not result.passed:
                category = result.message.split(":")[0] if ":" in result.message else "general"
                error_patterns[category] = error_patterns.get(category, 0) + 1
                
        # Generate specific recommendations
        if "Circuit" in error_patterns:
            recommendations.append("Review quantum circuit design for optimization opportunities")
            
        if "Invalid" in error_patterns:
            recommendations.append("Validate input parameters before training")
            
        if any("Low" in r.message for r in results):
            recommendations.append("Consider adjusting model parameters to improve performance")
            
        if any("High" in r.message for r in results):
            recommendations.append("Monitor resource usage and consider optimization")
            
        return recommendations