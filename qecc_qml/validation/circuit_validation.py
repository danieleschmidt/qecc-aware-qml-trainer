"""
Robust circuit validation and error handling for QECC-aware QML.
Generation 2: Enhanced reliability and validation.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import warnings
try:
    from qiskit import QuantumCircuit
except ImportError:
    from qecc_qml.core.fallback_imports import QuantumCircuit
try:
    from qiskit.quantum_info import Operator
except ImportError:
    from qecc_qml.core.fallback_imports import Operator
import logging

logger = logging.getLogger(__name__)


class CircuitValidator:
    """
    Comprehensive validation for quantum circuits and QECC configurations.
    """
    
    def __init__(self, strict_mode: bool = True):
        """
        Initialize circuit validator.
        
        Args:
            strict_mode: If True, raise errors for validation failures.
                        If False, issue warnings and attempt fixes.
        """
        self.strict_mode = strict_mode
        self.validation_history = []
        
    def validate_qnn_config(
        self,
        num_qubits: int,
        num_layers: int,
        entanglement: str,
        feature_map: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Validate QNN configuration parameters.
        
        Returns:
            Dictionary with validation results and recommendations
        """
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'recommendations': [],
            'corrected_params': {}
        }
        
        # Validate qubit count
        if num_qubits <= 0:
            results['valid'] = False
            results['errors'].append(f"Number of qubits must be positive, got {num_qubits}")
        elif num_qubits > 50:
            results['warnings'].append(f"Large number of qubits ({num_qubits}) may cause performance issues")
            
        # Validate layer count
        if num_layers <= 0:
            results['valid'] = False
            results['errors'].append(f"Number of layers must be positive, got {num_layers}")
        elif num_layers > 20:
            results['warnings'].append(f"Many layers ({num_layers}) may lead to barren plateaus")
            results['recommendations'].append("Consider using fewer layers with deeper circuits")
            
        # Validate entanglement pattern
        valid_entanglements = ['circular', 'linear', 'full', 'sca', 'pairwise']
        if entanglement not in valid_entanglements:
            if self.strict_mode:
                results['valid'] = False
                results['errors'].append(f"Unknown entanglement '{entanglement}'. Valid options: {valid_entanglements}")
            else:
                results['warnings'].append(f"Unknown entanglement '{entanglement}', using 'circular'")
                results['corrected_params']['entanglement'] = 'circular'
                
        # Validate feature map
        valid_feature_maps = ['amplitude_encoding', 'angle_encoding', 'pauli_encoding', 'iqp_encoding']
        if feature_map not in valid_feature_maps:
            if self.strict_mode:
                results['valid'] = False
                results['errors'].append(f"Unknown feature map '{feature_map}'. Valid options: {valid_feature_maps}")
            else:
                results['warnings'].append(f"Unknown feature map '{feature_map}', using 'amplitude_encoding'")
                results['corrected_params']['feature_map'] = 'amplitude_encoding'
        
        # Check for optimal configurations
        if num_qubits > 4 and num_layers < 3:
            results['recommendations'].append("Consider increasing layers for better expressivity with many qubits")
            
        self.validation_history.append(results)
        return results
    
    def validate_error_correction_compatibility(
        self,
        logical_qubits: int,
        error_correction_scheme: str,
        distance: int,
        available_physical_qubits: int = None
    ) -> Dict[str, Any]:
        """
        Validate error correction scheme compatibility.
        """
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'physical_qubits_required': 0,
            'threshold_error_rate': 0.0,
            'recommendations': []
        }
        
        # Calculate physical qubit requirements
        if error_correction_scheme.lower() == 'surface_code':
            results['physical_qubits_required'] = logical_qubits * (distance ** 2 + (distance - 1) ** 2)
            results['threshold_error_rate'] = 0.01  # ~1% threshold
            
        elif error_correction_scheme.lower() == 'color_code':
            results['physical_qubits_required'] = logical_qubits * (3 * distance ** 2 - 3 * distance + 1)
            results['threshold_error_rate'] = 0.008  # ~0.8% threshold
            
        elif error_correction_scheme.lower() == 'steane_code':
            if distance != 3:
                results['warnings'].append("Steane code only supports distance 3")
                results['corrected_params'] = {'distance': 3}
            results['physical_qubits_required'] = logical_qubits * 7
            results['threshold_error_rate'] = 0.005  # ~0.5% threshold
            
        else:
            results['errors'].append(f"Unknown error correction scheme: {error_correction_scheme}")
            results['valid'] = False
            
        # Check physical qubit availability
        if available_physical_qubits and results['physical_qubits_required'] > available_physical_qubits:
            results['warnings'].append(
                f"Requires {results['physical_qubits_required']} physical qubits "
                f"but only {available_physical_qubits} available"
            )
            results['recommendations'].append("Consider reducing distance or using different code")
            
        return results
    
    def validate_training_parameters(
        self,
        learning_rate: float,
        epochs: int,
        batch_size: int,
        shots: int
    ) -> Dict[str, Any]:
        """
        Validate training hyperparameters.
        """
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'corrected_params': {}
        }
        
        # Learning rate validation
        if learning_rate <= 0:
            results['errors'].append(f"Learning rate must be positive, got {learning_rate}")
            results['valid'] = False
        elif learning_rate > 1.0:
            results['warnings'].append(f"High learning rate ({learning_rate}) may cause instability")
        elif learning_rate < 1e-5:
            results['warnings'].append(f"Very low learning rate ({learning_rate}) may slow convergence")
            
        # Epochs validation
        if epochs <= 0:
            results['errors'].append(f"Epochs must be positive, got {epochs}")
            results['valid'] = False
        elif epochs > 1000:
            results['warnings'].append(f"Many epochs ({epochs}) may lead to overfitting")
            
        # Batch size validation
        if batch_size <= 0:
            results['errors'].append(f"Batch size must be positive, got {batch_size}")
            results['valid'] = False
        elif batch_size > 1024:
            results['warnings'].append(f"Large batch size ({batch_size}) may reduce gradient noise benefits")
            
        # Shots validation
        if shots <= 0:
            results['errors'].append(f"Shots must be positive, got {shots}")
            results['valid'] = False
        elif shots < 100:
            results['warnings'].append(f"Few shots ({shots}) may introduce high statistical noise")
        elif shots > 10000:
            results['warnings'].append(f"Many shots ({shots}) increase computation time")
            
        return results
    
    def validate_data_compatibility(
        self,
        X: np.ndarray,
        y: np.ndarray,
        num_qubits: int
    ) -> Dict[str, Any]:
        """
        Validate training data compatibility with quantum circuit.
        """
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'recommendations': []
        }
        
        # Basic shape validation
        if len(X.shape) != 2:
            results['errors'].append(f"X must be 2D array, got shape {X.shape}")
            results['valid'] = False
            
        if len(y.shape) != 1:
            results['errors'].append(f"y must be 1D array, got shape {y.shape}")
            results['valid'] = False
            
        if results['valid'] and X.shape[0] != y.shape[0]:
            results['errors'].append(f"X and y sample count mismatch: {X.shape[0]} vs {y.shape[0]}")
            results['valid'] = False
            
        if results['valid']:
            # Feature count validation
            if X.shape[1] != num_qubits:
                if X.shape[1] > num_qubits:
                    results['warnings'].append(
                        f"More features ({X.shape[1]}) than qubits ({num_qubits}). "
                        "Consider dimensionality reduction."
                    )
                else:
                    results['warnings'].append(
                        f"Fewer features ({X.shape[1]}) than qubits ({num_qubits}). "
                        "Will pad with zeros."
                    )
            
            # Data range validation for quantum encoding
            data_min, data_max = X.min(), X.max()
            if data_min < -1 or data_max > 1:
                results['warnings'].append(
                    f"Data range [{data_min:.3f}, {data_max:.3f}] outside [-1, 1]. "
                    "Consider normalization for quantum encoding."
                )
            
            # Class balance check
            unique_classes, counts = np.unique(y, return_counts=True)
            if len(unique_classes) > 2:
                results['warnings'].append("Multi-class classification may require different encoding")
                
            class_balance = min(counts) / max(counts)
            if class_balance < 0.3:
                results['warnings'].append(f"Imbalanced classes (ratio: {class_balance:.2f})")
                results['recommendations'].append("Consider data augmentation or class weighting")
                
        return results
    
    def validate_circuit_depth(self, circuit: QuantumCircuit, max_depth: int = 100) -> Dict[str, Any]:
        """
        Validate quantum circuit depth for NISQ compatibility.
        """
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'circuit_depth': circuit.depth(),
            'gate_count': sum(circuit.count_ops().values()),
            'recommendations': []
        }
        
        if results['circuit_depth'] > max_depth:
            results['warnings'].append(
                f"Circuit depth ({results['circuit_depth']}) exceeds recommended "
                f"maximum ({max_depth}) for NISQ devices"
            )
            results['recommendations'].append("Consider circuit optimization or decomposition")
            
        # Check for expensive gate types
        expensive_gates = ['ccx', 'mcx', 'mct']
        for gate in expensive_gates:
            if gate in circuit.count_ops():
                count = circuit.count_ops()[gate]
                results['warnings'].append(f"Contains {count} {gate} gates which are expensive to implement")
                
        return results
    
    def get_validation_report(self) -> Dict[str, Any]:
        """Get comprehensive validation report."""
        return {
            'total_validations': len(self.validation_history),
            'failed_validations': sum(1 for v in self.validation_history if not v.get('valid', True)),
            'total_errors': sum(len(v.get('errors', [])) for v in self.validation_history),
            'total_warnings': sum(len(v.get('warnings', [])) for v in self.validation_history),
            'recent_validations': self.validation_history[-5:] if self.validation_history else []
        }


class RobustErrorHandler:
    """
    Comprehensive error handling and recovery for quantum operations.
    """
    
    def __init__(self, max_retries: int = 3, backoff_factor: float = 1.5):
        """
        Initialize error handler.
        
        Args:
            max_retries: Maximum number of retry attempts
            backoff_factor: Exponential backoff multiplier
        """
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.error_counts = {}
        
    def handle_circuit_execution_error(self, error: Exception, context: Dict[str, Any]) -> Optional[str]:
        """
        Handle quantum circuit execution errors with appropriate recovery.
        
        Returns:
            Recovery strategy or None if unrecoverable
        """
        error_type = type(error).__name__
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        # Common error recovery strategies
        if 'timeout' in str(error).lower():
            return 'retry_with_backoff'
            
        elif 'queue' in str(error).lower() or 'busy' in str(error).lower():
            return 'switch_backend'
            
        elif 'calibration' in str(error).lower():
            return 'recalibrate_backend'
            
        elif 'memory' in str(error).lower():
            return 'reduce_batch_size'
            
        elif 'parameter' in str(error).lower():
            return 'validate_parameters'
            
        else:
            logger.error(f"Unhandled error type: {error_type}, message: {str(error)}")
            return None
    
    def get_error_statistics(self) -> Dict[str, int]:
        """Get error occurrence statistics."""
        return self.error_counts.copy()