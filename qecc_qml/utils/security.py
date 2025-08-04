"""
Security utilities for QECC-aware QML library.

Provides input sanitization, access control, and security validation
to prevent malicious inputs and ensure safe operation.
"""

import os
import re
import hashlib
import hmac
import secrets
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
import numpy as np
import pickle
import ast

from .logging_config import get_logger

logger = get_logger(__name__)


class SecurityError(Exception):
    """Exception raised for security-related issues."""
    pass


class InputSanitizer:
    """
    Comprehensive input sanitization for quantum ML parameters.
    """
    
    # Dangerous patterns to reject
    DANGEROUS_PATTERNS = [
        r'__.*__',  # Python dunder methods
        r'exec\s*\(',  # Code execution
        r'eval\s*\(',  # Code evaluation  
        r'import\s+',  # Module imports
        r'\.system\s*\(',  # System calls
        r'subprocess',  # Subprocess module
        r'os\.',  # OS module
        r'open\s*\(',  # File operations
        r'file\s*\(',  # File operations
        r'compile\s*\(',  # Code compilation
    ]
    
    def __init__(self):
        self.patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.DANGEROUS_PATTERNS]
    
    def sanitize_string(self, value: str, max_length: int = 1000) -> str:
        """
        Sanitize string input.
        
        Args:
            value: String to sanitize
            max_length: Maximum allowed length
            
        Returns:
            Sanitized string
            
        Raises:
            SecurityError: If input contains dangerous patterns
        """
        if not isinstance(value, str):
            raise SecurityError(f"Expected string, got {type(value)}")
        
        if len(value) > max_length:
            raise SecurityError(f"String too long: {len(value)} > {max_length}")
        
        # Check for dangerous patterns
        for pattern in self.patterns:
            if pattern.search(value):
                logger.warning(f"Dangerous pattern detected in input: {pattern.pattern}")
                raise SecurityError(f"Potentially dangerous input detected")
        
        # Remove null bytes and control characters
        sanitized = ''.join(char for char in value if ord(char) >= 32 or char in '\n\t')
        
        return sanitized
    
    def sanitize_numeric(
        self, 
        value: Union[int, float], 
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
        allow_inf: bool = False,
        allow_nan: bool = False
    ) -> Union[int, float]:
        """
        Sanitize numeric input.
        
        Args:
            value: Numeric value to sanitize
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            allow_inf: Whether to allow infinite values
            allow_nan: Whether to allow NaN values
            
        Returns:
            Sanitized numeric value
            
        Raises:
            SecurityError: If value is out of bounds or invalid
        """
        if not isinstance(value, (int, float, np.integer, np.floating)):
            raise SecurityError(f"Expected numeric type, got {type(value)}")
        
        # Convert to Python native types
        if isinstance(value, (np.integer, np.floating)):
            value = value.item()
        
        # Check for NaN and infinity
        if np.isnan(value) and not allow_nan:
            raise SecurityError("NaN values not allowed")
        
        if np.isinf(value) and not allow_inf:
            raise SecurityError("Infinite values not allowed")
        
        # Check bounds
        if min_val is not None and value < min_val:
            raise SecurityError(f"Value {value} below minimum {min_val}")
        
        if max_val is not None and value > max_val:
            raise SecurityError(f"Value {value} above maximum {max_val}")
        
        return value
    
    def sanitize_array(
        self, 
        value: np.ndarray,
        max_size: int = 1000000,  # 1M elements
        allowed_dtypes: Optional[List[np.dtype]] = None,
        check_finite: bool = True
    ) -> np.ndarray:
        """
        Sanitize numpy array input.
        
        Args:
            value: Array to sanitize
            max_size: Maximum array size
            allowed_dtypes: List of allowed data types
            check_finite: Whether to check for finite values
            
        Returns:
            Sanitized array
            
        Raises:
            SecurityError: If array is invalid or dangerous
        """
        if not isinstance(value, np.ndarray):
            try:
                value = np.array(value)
            except Exception as e:
                raise SecurityError(f"Cannot convert to numpy array: {e}")
        
        # Check array size
        if value.size > max_size:
            raise SecurityError(f"Array too large: {value.size} > {max_size}")
        
        # Check data type
        if allowed_dtypes and value.dtype not in allowed_dtypes:
            raise SecurityError(f"Disallowed dtype: {value.dtype}")
        
        # Check for finite values
        if check_finite and not np.all(np.isfinite(value)):
            raise SecurityError("Array contains non-finite values")
        
        # Check for reasonable memory usage
        memory_usage = value.nbytes
        if memory_usage > 100 * 1024 * 1024:  # 100MB
            logger.warning(f"Large array: {memory_usage / 1024 / 1024:.1f} MB")
        
        return value


def sanitize_input(value: Any, input_type: str = 'auto', **kwargs) -> Any:
    """
    Sanitize input based on type.
    
    Args:
        value: Value to sanitize
        input_type: Type of input ('string', 'numeric', 'array', 'auto')
        **kwargs: Additional sanitization parameters
        
    Returns:
        Sanitized value
    """
    sanitizer = InputSanitizer()
    
    if input_type == 'auto':
        if isinstance(value, str):
            input_type = 'string'
        elif isinstance(value, (int, float, np.number)):
            input_type = 'numeric'
        elif isinstance(value, np.ndarray):
            input_type = 'array'
        else:
            logger.warning(f"Unknown input type for sanitization: {type(value)}")
            return value
    
    if input_type == 'string':
        return sanitizer.sanitize_string(value, **kwargs)
    elif input_type == 'numeric':
        return sanitizer.sanitize_numeric(value, **kwargs)
    elif input_type == 'array':
        return sanitizer.sanitize_array(value, **kwargs)
    else:
        raise SecurityError(f"Unknown sanitization type: {input_type}")


def validate_file_path(
    file_path: Union[str, Path],
    allowed_extensions: Optional[List[str]] = None,
    max_path_length: int = 1000,
    allow_absolute: bool = True,
    base_directory: Optional[str] = None
) -> Path:
    """
    Validate and sanitize file paths.
    
    Args:
        file_path: File path to validate
        allowed_extensions: List of allowed file extensions
        max_path_length: Maximum path length
        allow_absolute: Whether to allow absolute paths
        base_directory: Base directory to restrict access to
        
    Returns:
        Validated Path object
        
    Raises:
        SecurityError: If path is invalid or dangerous
    """
    if isinstance(file_path, str):
        file_path = Path(file_path)
    
    path_str = str(file_path)
    
    # Check path length
    if len(path_str) > max_path_length:
        raise SecurityError(f"Path too long: {len(path_str)} > {max_path_length}")
    
    # Check for dangerous patterns
    dangerous_patterns = ['..', '~', '$', '`', '|', ';', '&', '\\x']
    for pattern in dangerous_patterns:
        if pattern in path_str:
            raise SecurityError(f"Dangerous pattern '{pattern}' in path")
    
    # Check absolute path permission
    if file_path.is_absolute() and not allow_absolute:
        raise SecurityError("Absolute paths not allowed")
    
    # Resolve path and check for directory traversal
    try:
        resolved_path = file_path.resolve()
    except Exception as e:
        raise SecurityError(f"Cannot resolve path: {e}")
    
    # Check base directory restriction
    if base_directory:
        base_path = Path(base_directory).resolve()
        try:
            resolved_path.relative_to(base_path)
        except ValueError:
            raise SecurityError(f"Path outside allowed directory: {base_path}")
    
    # Check file extension
    if allowed_extensions:
        extension = resolved_path.suffix.lower()
        if extension not in [ext.lower() for ext in allowed_extensions]:
            raise SecurityError(f"File extension '{extension}' not allowed")
    
    logger.debug(f"Validated file path: {resolved_path}")
    return resolved_path


def check_permissions(
    file_path: Union[str, Path],
    required_permissions: str = 'r'
) -> bool:
    """
    Check file/directory permissions.
    
    Args:
        file_path: Path to check
        required_permissions: Required permissions ('r', 'w', 'x', combinations)
        
    Returns:
        True if permissions are sufficient
        
    Raises:
        SecurityError: If permissions are insufficient
    """
    path = Path(file_path)
    
    if not path.exists():
        raise SecurityError(f"Path does not exist: {path}")
    
    # Check read permission
    if 'r' in required_permissions and not os.access(path, os.R_OK):
        raise SecurityError(f"Read permission denied: {path}")
    
    # Check write permission
    if 'w' in required_permissions and not os.access(path, os.W_OK):
        raise SecurityError(f"Write permission denied: {path}")
    
    # Check execute permission
    if 'x' in required_permissions and not os.access(path, os.X_OK):
        raise SecurityError(f"Execute permission denied: {path}")
    
    return True


def secure_pickle_load(
    file_path: Union[str, Path],
    allowed_modules: Optional[List[str]] = None,
    max_file_size: int = 100 * 1024 * 1024  # 100MB
) -> Any:
    """
    Securely load pickle files with restrictions.
    
    Args:
        file_path: Path to pickle file
        allowed_modules: List of allowed module names
        max_file_size: Maximum file size in bytes
        
    Returns:
        Unpickled object
        
    Raises:
        SecurityError: If file is unsafe to load
    """
    path = validate_file_path(file_path, allowed_extensions=['.pkl', '.pickle'])
    
    # Check file size
    file_size = path.stat().st_size
    if file_size > max_file_size:
        raise SecurityError(f"File too large: {file_size} > {max_file_size}")
    
    # Create restricted unpickler
    class RestrictedUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if allowed_modules and module not in allowed_modules:
                raise SecurityError(f"Module '{module}' not allowed")
            
            # Block dangerous classes
            dangerous_classes = [
                'subprocess', 'os', 'sys', 'builtins', '__builtin__',
                'importlib', 'imp', 'marshal', 'types'
            ]
            
            if module in dangerous_classes:
                raise SecurityError(f"Dangerous module '{module}' blocked")
            
            return super().find_class(module, name)
    
    # Load with restrictions
    try:
        with open(path, 'rb') as f:
            unpickler = RestrictedUnpickler(f)
            obj = unpickler.load()
        
        logger.info(f"Securely loaded pickle file: {path}")
        return obj
        
    except Exception as e:
        logger.error(f"Failed to load pickle file {path}: {e}")
        raise SecurityError(f"Cannot load pickle file: {e}")


def compute_file_hash(file_path: Union[str, Path], algorithm: str = 'sha256') -> str:
    """
    Compute hash of a file for integrity checking.
    
    Args:
        file_path: Path to file
        algorithm: Hash algorithm ('md5', 'sha1', 'sha256', 'sha512')
        
    Returns:
        Hex digest of file hash
    """
    path = validate_file_path(file_path)
    
    try:
        hasher = hashlib.new(algorithm)
        
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                hasher.update(chunk)
        
        file_hash = hasher.hexdigest()
        logger.debug(f"Computed {algorithm} hash for {path}: {file_hash[:16]}...")
        
        return file_hash
        
    except Exception as e:
        raise SecurityError(f"Cannot compute hash for {path}: {e}")


def verify_file_integrity(
    file_path: Union[str, Path],
    expected_hash: str,
    algorithm: str = 'sha256'
) -> bool:
    """
    Verify file integrity using hash comparison.
    
    Args:
        file_path: Path to file
        expected_hash: Expected hash value
        algorithm: Hash algorithm
        
    Returns:
        True if file integrity is verified
        
    Raises:
        SecurityError: If integrity check fails
    """
    actual_hash = compute_file_hash(file_path, algorithm)
    
    if not hmac.compare_digest(actual_hash, expected_hash.lower()):
        raise SecurityError(
            f"File integrity check failed for {file_path}. "
            f"Expected: {expected_hash}, Got: {actual_hash}"
        )
    
    logger.info(f"File integrity verified: {file_path}")
    return True


def generate_secure_token(length: int = 32) -> str:
    """
    Generate cryptographically secure random token.
    
    Args:
        length: Token length in bytes
        
    Returns:
        Hex-encoded secure token
    """
    token = secrets.token_hex(length)
    logger.debug(f"Generated secure token of length {len(token)}")
    return token


def sanitize_environment_variables() -> Dict[str, str]:
    """
    Sanitize environment variables for safe use.
    
    Returns:
        Dictionary of sanitized environment variables
    """
    sanitizer = InputSanitizer()
    safe_env = {}
    
    # List of safe environment variables to preserve
    safe_vars = [
        'PATH', 'HOME', 'USER', 'SHELL', 'LANG', 'LC_ALL',
        'PYTHONPATH', 'QISKIT_IBM_TOKEN', 'AWS_REGION',
        'CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS'
    ]
    
    for var_name in safe_vars:
        if var_name in os.environ:
            try:
                safe_value = sanitizer.sanitize_string(
                    os.environ[var_name], max_length=2000
                )
                safe_env[var_name] = safe_value
            except SecurityError as e:
                logger.warning(f"Skipping unsafe environment variable {var_name}: {e}")
    
    logger.debug(f"Sanitized {len(safe_env)} environment variables")
    return safe_env


def validate_quantum_parameters(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate quantum circuit parameters for security.
    
    Args:
        params: Dictionary of quantum parameters
        
    Returns:
        Validated parameters
        
    Raises:
        SecurityError: If parameters are unsafe
    """
    sanitizer = InputSanitizer()
    validated = {}
    
    for key, value in params.items():
        # Sanitize key
        safe_key = sanitizer.sanitize_string(key, max_length=100)
        
        # Validate value based on common quantum parameter types
        if isinstance(value, str):
            validated[safe_key] = sanitizer.sanitize_string(value, max_length=500)
        elif isinstance(value, (int, float)):
            validated[safe_key] = sanitizer.sanitize_numeric(
                value, min_val=-100*np.pi, max_val=100*np.pi
            )
        elif isinstance(value, np.ndarray):
            validated[safe_key] = sanitizer.sanitize_array(
                value, max_size=10000, check_finite=True
            )
        elif isinstance(value, (list, tuple)) and len(value) < 1000:
            # Small lists/tuples are generally safe
            validated[safe_key] = value
        else:
            logger.warning(f"Skipping unknown parameter type {key}: {type(value)}")
    
    logger.debug(f"Validated {len(validated)} quantum parameters")
    return validated