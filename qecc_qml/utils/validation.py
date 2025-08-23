"""
Input validation utilities for QECC-QML framework.
"""

import numpy as np
from typing import Union, Optional, Tuple, Any, List, Dict


def validate_input_data(X: np.ndarray, y: np.ndarray) -> None:
    """
    Validate input training data.
    """
    if not isinstance(X, np.ndarray):
        raise ValueError(f"X must be numpy array, got {type(X)}")
    
    if not isinstance(y, np.ndarray):
        raise ValueError(f"y must be numpy array, got {type(y)}")
    
    if len(X) == 0:
        raise ValueError("X cannot be empty")
    
    if len(y) == 0:
        raise ValueError("y cannot be empty")
    
    if len(X) != len(y):
        raise ValueError(f"X and y must have same length: {len(X)} vs {len(y)}")
    
    if X.ndim != 2:
        raise ValueError(f"X must be 2D array, got shape {X.shape}")
    
    if y.ndim != 1:
        raise ValueError(f"y must be 1D array, got shape {y.shape}")
    
    # Check for NaN or infinite values
    if np.any(np.isnan(X)):
        raise ValueError("X contains NaN values")
    
    if np.any(np.isnan(y)):
        raise ValueError("y contains NaN values")
    
    if np.any(np.isinf(X)):
        raise ValueError("X contains infinite values")
    
    if np.any(np.isinf(y)):
        raise ValueError("y contains infinite values")


def validate_parameters(params: np.ndarray, expected_shape: Optional[Tuple[int, ...]] = None) -> None:
    """
    Validate parameter array.
    """
    if not isinstance(params, np.ndarray):
        raise ValueError(f"Parameters must be numpy array, got {type(params)}")
    
    if len(params) == 0:
        raise ValueError("Parameters cannot be empty")
    
    if expected_shape and params.shape != expected_shape:
        raise ValueError(f"Parameters shape {params.shape} does not match expected {expected_shape}")
    
    # Check for NaN or infinite values
    if np.any(np.isnan(params)):
        raise ValueError("Parameters contain NaN values")
    
    if np.any(np.isinf(params)):
        raise ValueError("Parameters contain infinite values")


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


class QuantumValidationError(ValidationError):
    """Specific validation error for quantum parameters."""
    pass


def validate_input(func):
    """
    Decorator for input validation.
    
    Adds comprehensive input validation to functions dealing with
    quantum parameters, training data, and configuration.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            # Log function call for debugging
            logger.debug(f"Calling {func.__name__} with args={len(args)}, kwargs={list(kwargs.keys())}")
            
            # Pre-validation hook
            if hasattr(func, '_pre_validate'):
                func._pre_validate(*args, **kwargs)
            
            result = func(*args, **kwargs)
            
            # Post-validation hook
            if hasattr(func, '_post_validate'):
                func._post_validate(result)
            
            logger.debug(f"Successfully completed {func.__name__}")
            return result
            
        except ValidationError as e:
            logger.error(f"Validation error in {func.__name__}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {e}")
            raise
    
    return wrapper


def validate_qnn_config(
    num_qubits: int,
    num_layers: int,
    entanglement: str = "circular",
    feature_map: str = "amplitude_encoding",
    rotation_gates: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Validate quantum neural network configuration.
    
    Args:
        num_qubits: Number of logical qubits
        num_layers: Number of variational layers
        entanglement: Entanglement pattern
        feature_map: Feature encoding method
        rotation_gates: List of rotation gates
        
    Returns:
        Validated configuration dictionary
        
    Raises:
        QuantumValidationError: If configuration is invalid
    """
    logger.debug("Validating QNN configuration")
    
    # Validate num_qubits
    if not isinstance(num_qubits, int):
        raise QuantumValidationError(f"num_qubits must be integer, got {type(num_qubits)}")
    if num_qubits < 1:
        raise QuantumValidationError(f"num_qubits must be >= 1, got {num_qubits}")
    if num_qubits > 50:
        warnings.warn(f"Large number of qubits ({num_qubits}) may cause performance issues")
    
    # Validate num_layers
    if not isinstance(num_layers, int):
        raise QuantumValidationError(f"num_layers must be integer, got {type(num_layers)}")
    if num_layers < 1:
        raise QuantumValidationError(f"num_layers must be >= 1, got {num_layers}")
    if num_layers > 20:
        warnings.warn(f"Deep circuit ({num_layers} layers) may suffer from noise")
    
    # Validate entanglement
    valid_entanglements = ["circular", "linear", "full", "none"]
    if entanglement not in valid_entanglements:
        raise QuantumValidationError(
            f"entanglement must be one of {valid_entanglements}, got {entanglement}"
        )
    
    # Validate feature_map
    valid_feature_maps = ["amplitude_encoding", "angle_encoding", "pauli_encoding"]
    if feature_map not in valid_feature_maps:
        raise QuantumValidationError(
            f"feature_map must be one of {valid_feature_maps}, got {feature_map}"
        )
    
    # Validate rotation_gates
    if rotation_gates is None:
        rotation_gates = ['rx', 'ry', 'rz']
    
    valid_gates = ['rx', 'ry', 'rz', 'u3']
    invalid_gates = [gate for gate in rotation_gates if gate not in valid_gates]
    if invalid_gates:
        raise QuantumValidationError(
            f"Invalid rotation gates {invalid_gates}. Valid gates: {valid_gates}"
        )
    
    # Check parameter count feasibility
    num_params = num_layers * len(rotation_gates) * num_qubits
    if num_params > 1000:
        warnings.warn(f"Large parameter count ({num_params}) may slow optimization")
    
    config = {
        'num_qubits': num_qubits,
        'num_layers': num_layers,
        'entanglement': entanglement,
        'feature_map': feature_map,
        'rotation_gates': rotation_gates,
        'estimated_parameters': num_params,
    }
    
    logger.info(f"Validated QNN config: {config}")
    return config


def validate_noise_model(
    gate_error_rate: float,
    readout_error_rate: float,
    T1: float,
    T2: float,
    **kwargs
) -> Dict[str, Any]:
    """
    Validate noise model parameters.
    
    Args:
        gate_error_rate: Gate error rate (0-1)
        readout_error_rate: Readout error rate (0-1)
        T1: Relaxation time (seconds)
        T2: Dephasing time (seconds)
        **kwargs: Additional parameters
        
    Returns:
        Validated noise model parameters
        
    Raises:
        ValidationError: If parameters are invalid
    """
    logger.debug("Validating noise model parameters")
    
    # Validate error rates
    for rate_name, rate_value in [("gate_error_rate", gate_error_rate), 
                                  ("readout_error_rate", readout_error_rate)]:
        if not isinstance(rate_value, (int, float)):
            raise ValidationError(f"{rate_name} must be numeric, got {type(rate_value)}")
        if not 0 <= rate_value <= 1:
            raise ValidationError(f"{rate_name} must be in [0,1], got {rate_value}")
    
    # Validate coherence times
    for time_name, time_value in [("T1", T1), ("T2", T2)]:
        if not isinstance(time_value, (int, float)):
            raise ValidationError(f"{time_name} must be numeric, got {type(time_value)}")
        if time_value <= 0:
            raise ValidationError(f"{time_name} must be positive, got {time_value}")
    
    # Physical consistency checks
    if T2 > 2 * T1:
        warnings.warn(f"T2 ({T2}) > 2*T1 ({2*T1}), may be unphysical")
    
    # Realistic range checks
    if gate_error_rate > 0.1:
        warnings.warn(f"Very high gate error rate ({gate_error_rate})")
    if T1 < 1e-6:
        warnings.warn(f"Very short T1 time ({T1}s)")
    if T2 < 1e-6:
        warnings.warn(f"Very short T2 time ({T2}s)")
    
    validated_params = {
        'gate_error_rate': float(gate_error_rate),
        'readout_error_rate': float(readout_error_rate),
        'T1': float(T1),
        'T2': float(T2),
    }
    
    # Add validated kwargs
    for key, value in kwargs.items():
        if isinstance(value, (int, float)):
            validated_params[key] = float(value)
        else:
            validated_params[key] = value
    
    logger.info(f"Validated noise model: {validated_params}")
    return validated_params


def validate_training_data(
    X: np.ndarray,
    y: np.ndarray,
    num_features: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Validate training data for quantum machine learning.
    
    Args:
        X: Feature matrix
        y: Target labels
        num_features: Expected number of features
        
    Returns:
        Validated (X, y) tuple
        
    Raises:
        ValidationError: If data is invalid
    """
    logger.debug(f"Validating training data: X.shape={X.shape}, y.shape={y.shape}")
    
    # Check types
    if not isinstance(X, np.ndarray):
        try:
            X = np.array(X)
        except Exception as e:
            raise ValidationError(f"Cannot convert X to numpy array: {e}")
    
    if not isinstance(y, np.ndarray):
        try:
            y = np.array(y)
        except Exception as e:
            raise ValidationError(f"Cannot convert y to numpy array: {e}")
    
    # Check dimensions
    if X.ndim == 1:
        X = X.reshape(1, -1)
    elif X.ndim > 2:
        raise ValidationError(f"X must be 1D or 2D, got shape {X.shape}")
    
    if y.ndim > 2:
        raise ValidationError(f"y must be 1D or 2D, got shape {y.shape}")
    
    # Check sample count consistency
    if len(X) != len(y):
        raise ValidationError(f"Sample count mismatch: X has {len(X)}, y has {len(y)}")
    
    # Check for empty data
    if len(X) == 0:
        raise ValidationError("Empty dataset provided")
    
    # Check feature count
    if num_features is not None and X.shape[1] != num_features:
        raise ValidationError(
            f"Expected {num_features} features, got {X.shape[1]}"
        )
    
    # Check for NaN/inf values
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValidationError("X contains NaN or infinite values")
    
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValidationError("y contains NaN or infinite values")
    
    # Quantum-specific validations
    # Features should typically be in [0, π] or [-π, π] for quantum encoding
    x_min, x_max = np.min(X), np.max(X)
    if x_min < -2*np.pi or x_max > 2*np.pi:
        warnings.warn(
            f"Features outside typical quantum range: [{x_min:.3f}, {x_max:.3f}]. "
            "Consider normalizing to [0, π] or [-π, π]"
        )
    
    # Check label format
    unique_labels = np.unique(y)
    if len(unique_labels) > 10:
        warnings.warn(f"Many unique labels ({len(unique_labels)}), consider regression instead")
    
    logger.info(f"Validated data: {len(X)} samples, {X.shape[1]} features, {len(unique_labels)} classes")
    
    return X, y


def validate_error_correction_config(
    scheme_name: str,
    distance: int,
    logical_qubits: int,
    **kwargs
) -> Dict[str, Any]:
    """
    Validate error correction configuration.
    
    Args:
        scheme_name: Name of error correction scheme
        distance: Code distance
        logical_qubits: Number of logical qubits
        **kwargs: Additional scheme-specific parameters
        
    Returns:
        Validated configuration
        
    Raises:
        ValidationError: If configuration is invalid
    """
    logger.debug(f"Validating error correction: {scheme_name}, d={distance}")
    
    # Validate scheme name
    valid_schemes = ["surface_code", "color_code", "repetition_code", "steane_code"]
    if scheme_name.lower() not in valid_schemes:
        raise ValidationError(f"Unknown scheme '{scheme_name}'. Valid: {valid_schemes}")
    
    # Validate distance
    if not isinstance(distance, int):
        raise ValidationError(f"Distance must be integer, got {type(distance)}")
    if distance < 3:
        raise ValidationError(f"Distance must be >= 3, got {distance}")
    if distance % 2 == 0:
        raise ValidationError(f"Distance must be odd, got {distance}")
    if distance > 15:
        warnings.warn(f"Large distance ({distance}) will require many physical qubits")
    
    # Validate logical qubits
    if not isinstance(logical_qubits, int):
        raise ValidationError(f"logical_qubits must be integer, got {type(logical_qubits)}")
    if logical_qubits < 1:
        raise ValidationError(f"logical_qubits must be >= 1, got {logical_qubits}")
    if logical_qubits > 10:
        warnings.warn(f"Many logical qubits ({logical_qubits}) will be resource-intensive")
    
    # Estimate resource requirements
    if scheme_name.lower() == "surface_code":
        physical_qubits_per_logical = distance ** 2
        total_physical_qubits = logical_qubits * physical_qubits_per_logical
        
        if total_physical_qubits > 100:
            warnings.warn(
                f"Surface code will require {total_physical_qubits} physical qubits"
            )
    
    config = {
        'scheme_name': scheme_name.lower(),
        'distance': distance,
        'logical_qubits': logical_qubits,
        'correctable_errors': (distance - 1) // 2,
    }
    
    # Add validated kwargs
    config.update(kwargs)
    
    logger.info(f"Validated error correction config: {config}")
    return config


def validate_hyperparameters(
    learning_rate: float,
    epochs: int,
    batch_size: int,
    shots: int,
    **kwargs
) -> Dict[str, Any]:
    """
    Validate training hyperparameters.
    
    Args:
        learning_rate: Learning rate for optimization
        epochs: Number of training epochs
        batch_size: Training batch size
        shots: Number of quantum measurement shots
        **kwargs: Additional hyperparameters
        
    Returns:
        Validated hyperparameters
        
    Raises:
        ValidationError: If hyperparameters are invalid
    """
    logger.debug("Validating training hyperparameters")
    
    # Validate learning rate
    if not isinstance(learning_rate, (int, float)):
        raise ValidationError(f"learning_rate must be numeric, got {type(learning_rate)}")
    if not 0 < learning_rate <= 1:
        raise ValidationError(f"learning_rate must be in (0,1], got {learning_rate}")
    if learning_rate > 0.5:
        warnings.warn(f"High learning rate ({learning_rate}) may cause instability")
    
    # Validate epochs
    if not isinstance(epochs, int):
        raise ValidationError(f"epochs must be integer, got {type(epochs)}")
    if epochs < 1:
        raise ValidationError(f"epochs must be >= 1, got {epochs}")
    if epochs > 1000:
        warnings.warn(f"Many epochs ({epochs}) will take long time")
    
    # Validate batch size
    if not isinstance(batch_size, int):
        raise ValidationError(f"batch_size must be integer, got {type(batch_size)}")
    if batch_size < 1:
        raise ValidationError(f"batch_size must be >= 1, got {batch_size}")
    if batch_size > 1000:
        warnings.warn(f"Large batch size ({batch_size}) may require lots of memory")
    
    # Validate shots
    if not isinstance(shots, int):
        raise ValidationError(f"shots must be integer, got {type(shots)}")
    if shots < 1:
        raise ValidationError(f"shots must be >= 1, got {shots}")
    if shots < 100:
        warnings.warn(f"Few shots ({shots}) may give noisy results")
    if shots > 100000:
        warnings.warn(f"Many shots ({shots}) will be slow")
    
    hyperparams = {
        'learning_rate': float(learning_rate),
        'epochs': epochs,
        'batch_size': batch_size,
        'shots': shots,
    }
    
    # Validate additional hyperparameters
    for key, value in kwargs.items():
        if isinstance(value, (int, float)):
            if value < 0:
                warnings.warn(f"Negative value for {key}: {value}")
            hyperparams[key] = float(value) if isinstance(value, float) else value
        else:
            hyperparams[key] = value
    
    logger.info(f"Validated hyperparameters: {hyperparams}")
    return hyperparams


def check_quantum_hardware_compatibility(
    num_qubits: int,
    circuit_depth: int,
    backend_name: Optional[str] = None
) -> Dict[str, bool]:
    """
    Check compatibility with quantum hardware constraints.
    
    Args:
        num_qubits: Required number of qubits
        circuit_depth: Circuit depth
        backend_name: Target backend name
        
    Returns:
        Dictionary of compatibility checks
    """
    logger.debug(f"Checking hardware compatibility: {num_qubits} qubits, depth {circuit_depth}")
    
    compatibility = {
        'qubit_count_ok': True,
        'depth_ok': True,
        'connectivity_ok': True,
        'coherence_ok': True,
    }
    
    # Generic hardware limits
    if num_qubits > 127:  # Current largest processors
        compatibility['qubit_count_ok'] = False
        warnings.warn(f"Requested {num_qubits} qubits exceeds current hardware limits")
    
    if circuit_depth > 100:
        compatibility['depth_ok'] = False
        warnings.warn(f"Circuit depth {circuit_depth} may exceed coherence time")
    
    # Backend-specific checks
    if backend_name:
        backend_limits = {
            'ibm_lagos': {'qubits': 7, 'max_depth': 50},
            'ibm_nairobi': {'qubits': 7, 'max_depth': 50},
            'ibm_washington': {'qubits': 127, 'max_depth': 30},
            'google_sycamore': {'qubits': 70, 'max_depth': 20},
            'ionq_harmony': {'qubits': 32, 'max_depth': 100},
        }
        
        if backend_name.lower() in backend_limits:
            limits = backend_limits[backend_name.lower()]
            
            if num_qubits > limits['qubits']:
                compatibility['qubit_count_ok'] = False
            
            if circuit_depth > limits['max_depth']:
                compatibility['depth_ok'] = False
    
    logger.info(f"Hardware compatibility: {compatibility}")
    return compatibility