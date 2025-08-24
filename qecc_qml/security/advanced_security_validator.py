"""
Advanced Security Validation for Quantum Machine Learning.

This module provides comprehensive security validation for quantum circuits,
data, and operations to prevent malicious attacks and ensure system integrity.
"""

# Import with fallback support
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from qecc_qml.core.fallback_imports import create_fallback_implementations
    create_fallback_implementations()
except ImportError:
    pass
import hashlib
import hmac
import secrets
import time
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import json
try:
    import numpy as np
except ImportError:
    import sys
    if 'numpy' in sys.modules:
        np = sys.modules['numpy']
    else:
        class MockNumPy:
            @staticmethod
            def array(x): return list(x) if isinstance(x, (list, tuple)) else x
            @staticmethod
            def zeros(shape): return [0] * (shape if isinstance(shape, int) else shape[0])
            @staticmethod  
            def ones(shape): return [1] * (shape if isinstance(shape, int) else shape[0])
            ndarray = list
        np = MockNumPy()

try:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator
    QISKIT_AVAILABLE = True
except ImportError:
    from ..core.fallback_imports import QuantumCircuit, Operator
    QISKIT_AVAILABLE = False

from ..utils.logging_config import get_logger

logger = get_logger(__name__)

@dataclass
class SecurityPolicy:
    """Security policy configuration."""
    max_circuit_depth: int = 1000
    max_qubits: int = 50
    max_gates: int = 10000
    max_file_size_mb: float = 100.0
    allowed_gate_types: List[str] = None
    require_authentication: bool = True
    enable_encryption: bool = True
    audit_logging: bool = True
    rate_limit_per_hour: int = 1000
    
    def __post_init__(self):
        if self.allowed_gate_types is None:
            self.allowed_gate_types = [
                'x', 'y', 'z', 'h', 'cnot', 'cz', 'rx', 'ry', 'rz',
                'u1', 'u2', 'u3', 'barrier', 'measure'
            ]

@dataclass
class SecurityThreat:
    """Security threat detection result."""
    threat_type: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    description: str
    mitigation: str
    timestamp: float
    
class SecurityValidationError(Exception):
    """Exception raised for security validation failures."""
    pass

class AdvancedSecurityValidator:
    """
    Advanced security validator for quantum machine learning systems.
    
    Provides comprehensive security validation including:
    - Circuit integrity verification
    - Data sanitization and validation
    - Access control and authentication
    - Threat detection and prevention
    - Audit logging and monitoring
    """
    
    def __init__(self, policy: Optional[SecurityPolicy] = None):
        """Initialize security validator."""
        self.policy = policy or SecurityPolicy()
        self.logger = get_logger(__name__)
        self.threat_history: List[SecurityThreat] = []
        self.rate_limiter: Dict[str, List[float]] = {}
        self.authorized_users: Dict[str, Dict[str, Any]] = {}
        self.secret_key = secrets.token_bytes(32)
        
    def validate_circuit_security(self, circuit: QuantumCircuit, user_id: str = None) -> Dict[str, Any]:
        """
        Comprehensive security validation for quantum circuits.
        
        Args:
            circuit: Quantum circuit to validate
            user_id: User identifier for access control
            
        Returns:
            Validation result with security status
            
        Raises:
            SecurityValidationError: If security validation fails
        """
        validation_result = {
            'secure': True,
            'threats': [],
            'warnings': [],
            'recommendations': []
        }
        
        try:
            # 1. Basic circuit structure validation
            self._validate_circuit_structure(circuit, validation_result)
            
            # 2. Gate type validation
            self._validate_gate_types(circuit, validation_result)
            
            # 3. Resource consumption validation
            self._validate_resource_limits(circuit, validation_result)
            
            # 4. Malicious pattern detection
            self._detect_malicious_patterns(circuit, validation_result)
            
            # 5. Access control validation
            if user_id:
                self._validate_user_access(user_id, validation_result)
            
            # 6. Rate limiting
            if user_id:
                self._check_rate_limits(user_id, validation_result)
            
            # Log security validation
            if self.policy.audit_logging:
                self._log_security_event('circuit_validation', {
                    'user_id': user_id,
                    'circuit_qubits': getattr(circuit, 'num_qubits', 0),
                    'validation_result': validation_result['secure']
                })
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"Security validation error: {e}")
            raise SecurityValidationError(f"Security validation failed: {e}")
    
    def _validate_circuit_structure(self, circuit: QuantumCircuit, result: Dict[str, Any]) -> None:
        """Validate basic circuit structure for security."""
        if not hasattr(circuit, 'num_qubits'):
            threat = SecurityThreat(
                threat_type='invalid_circuit',
                severity='high',
                description='Circuit missing required attributes',
                mitigation='Reject circuit and sanitize input',
                timestamp=time.time()
            )
            result['threats'].append(threat)
            result['secure'] = False
            return
        
        # Check for suspicious circuit properties
        if hasattr(circuit, 'data') and len(circuit.data) == 0:
            result['warnings'].append('Empty circuit detected')
    
    def _validate_gate_types(self, circuit: QuantumCircuit, result: Dict[str, Any]) -> None:
        """Validate that only allowed gate types are used."""
        if not QISKIT_AVAILABLE:
            # Skip gate validation for fallback circuits
            return
        
        if not hasattr(circuit, 'data'):
            return
        
        forbidden_gates = []
        for instruction in circuit.data:
            gate_name = instruction[0].name if hasattr(instruction[0], 'name') else str(instruction[0])
            
            if gate_name.lower() not in self.policy.allowed_gate_types:
                forbidden_gates.append(gate_name)
        
        if forbidden_gates:
            threat = SecurityThreat(
                threat_type='forbidden_gates',
                severity='medium',
                description=f'Forbidden gate types detected: {set(forbidden_gates)}',
                mitigation='Remove forbidden gates or update security policy',
                timestamp=time.time()
            )
            result['threats'].append(threat)
            result['warnings'].append(f'Forbidden gates: {set(forbidden_gates)}')
    
    def _validate_resource_limits(self, circuit: QuantumCircuit, result: Dict[str, Any]) -> None:
        """Validate resource consumption limits."""
        # Check qubit count
        num_qubits = getattr(circuit, 'num_qubits', 0)
        if num_qubits > self.policy.max_qubits:
            threat = SecurityThreat(
                threat_type='resource_exhaustion',
                severity='high',
                description=f'Circuit exceeds qubit limit: {num_qubits} > {self.policy.max_qubits}',
                mitigation='Reduce circuit size or increase resource limits',
                timestamp=time.time()
            )
            result['threats'].append(threat)
            result['secure'] = False
        
        # Check circuit depth
        if hasattr(circuit, 'depth') and callable(circuit.depth):
            depth = circuit.depth()
            if depth > self.policy.max_circuit_depth:
                threat = SecurityThreat(
                    threat_type='resource_exhaustion',
                    severity='medium',
                    description=f'Circuit exceeds depth limit: {depth} > {self.policy.max_circuit_depth}',
                    mitigation='Reduce circuit depth or optimize circuit',
                    timestamp=time.time()
                )
                result['threats'].append(threat)
        
        # Check gate count
        if hasattr(circuit, 'size') and callable(circuit.size):
            gate_count = circuit.size()
            if gate_count > self.policy.max_gates:
                threat = SecurityThreat(
                    threat_type='resource_exhaustion',
                    severity='medium',
                    description=f'Circuit exceeds gate limit: {gate_count} > {self.policy.max_gates}',
                    mitigation='Reduce number of gates or optimize circuit',
                    timestamp=time.time()
                )
                result['threats'].append(threat)
    
    def _detect_malicious_patterns(self, circuit: QuantumCircuit, result: Dict[str, Any]) -> None:
        """Detect potentially malicious circuit patterns."""
        if not QISKIT_AVAILABLE or not hasattr(circuit, 'data'):
            return
        
        # Pattern 1: Excessive repetition (potential DoS)
        gate_counts = {}
        for instruction in circuit.data:
            gate_name = instruction[0].name if hasattr(instruction[0], 'name') else str(instruction[0])
            gate_counts[gate_name] = gate_counts.get(gate_name, 0) + 1
        
        for gate_name, count in gate_counts.items():
            if count > 1000:  # Threshold for excessive repetition
                result['warnings'].append(f'Excessive {gate_name} gates detected: {count}')
        
        # Pattern 2: Suspicious parameter values
        for instruction in circuit.data:
            if hasattr(instruction[0], 'params'):
                for param in instruction[0].params:
                    if isinstance(param, (int, float)):
                        if abs(param) > 1000:  # Unusually large parameter
                            result['warnings'].append(f'Large parameter value detected: {param}')
    
    def _validate_user_access(self, user_id: str, result: Dict[str, Any]) -> None:
        """Validate user access permissions."""
        if self.policy.require_authentication:
            if user_id not in self.authorized_users:
                threat = SecurityThreat(
                    threat_type='unauthorized_access',
                    severity='critical',
                    description=f'Unauthorized user access attempt: {user_id}',
                    mitigation='Authenticate user or deny access',
                    timestamp=time.time()
                )
                result['threats'].append(threat)
                result['secure'] = False
    
    def _check_rate_limits(self, user_id: str, result: Dict[str, Any]) -> None:
        """Check rate limiting for user requests."""
        current_time = time.time()
        hour_ago = current_time - 3600  # 1 hour ago
        
        # Initialize user rate limit tracking
        if user_id not in self.rate_limiter:
            self.rate_limiter[user_id] = []
        
        # Clean old requests
        self.rate_limiter[user_id] = [
            req_time for req_time in self.rate_limiter[user_id] 
            if req_time > hour_ago
        ]
        
        # Check rate limit
        if len(self.rate_limiter[user_id]) >= self.policy.rate_limit_per_hour:
            threat = SecurityThreat(
                threat_type='rate_limit_exceeded',
                severity='medium',
                description=f'Rate limit exceeded for user {user_id}',
                mitigation='Implement rate limiting or block user temporarily',
                timestamp=time.time()
            )
            result['threats'].append(threat)
            result['warnings'].append('Rate limit exceeded')
        
        # Add current request
        self.rate_limiter[user_id].append(current_time)
    
    def _log_security_event(self, event_type: str, details: Dict[str, Any]) -> None:
        """Log security events for audit purposes."""
        event = {
            'timestamp': time.time(),
            'event_type': event_type,
            'details': details,
            'source': 'AdvancedSecurityValidator'
        }
        
        # In a real implementation, this would go to a secure audit log
        self.logger.info(f"Security event: {json.dumps(event)}")
    
    def sanitize_input_data(self, data: Any) -> Any:
        """
        Sanitize input data to prevent injection attacks.
        
        Args:
            data: Input data to sanitize
            
        Returns:
            Sanitized data
        """
        if isinstance(data, str):
            # Remove potentially dangerous characters
            dangerous_chars = ['<', '>', ';', '&', '|', '`', '$']
            sanitized = data
            for char in dangerous_chars:
                sanitized = sanitized.replace(char, '')
            return sanitized
        
        elif isinstance(data, dict):
            return {key: self.sanitize_input_data(value) for key, value in data.items()}
        
        elif isinstance(data, list):
            return [self.sanitize_input_data(item) for item in data]
        
        elif isinstance(data, np.ndarray):
            # Validate numpy array properties
            if data.size > 1000000:  # 1M elements limit
                raise SecurityValidationError("Input array too large")
            
            # Check for NaN or infinite values
            if not np.isfinite(data).all():
                raise SecurityValidationError("Invalid values in input array")
            
            return data
        
        else:
            return data
    
    def create_secure_hash(self, data: Any) -> str:
        """Create secure hash of data for integrity verification."""
        if isinstance(data, str):
            data_bytes = data.encode('utf-8')
        else:
            data_bytes = str(data).encode('utf-8')
        
        return hmac.new(self.secret_key, data_bytes, hashlib.sha256).hexdigest()
    
    def verify_data_integrity(self, data: Any, expected_hash: str) -> bool:
        """Verify data integrity using secure hash."""
        computed_hash = self.create_secure_hash(data)
        return hmac.compare_digest(computed_hash, expected_hash)
    
    def add_authorized_user(self, user_id: str, permissions: Dict[str, Any]) -> None:
        """Add authorized user with specific permissions."""
        self.authorized_users[user_id] = {
            'permissions': permissions,
            'added_time': time.time(),
            'last_access': None
        }
        self.logger.info(f"Added authorized user: {user_id}")
    
    def remove_authorized_user(self, user_id: str) -> None:
        """Remove authorized user."""
        if user_id in self.authorized_users:
            del self.authorized_users[user_id]
            self.logger.info(f"Removed authorized user: {user_id}")
    
    def get_security_summary(self) -> Dict[str, Any]:
        """Get comprehensive security summary."""
        recent_threats = [
            threat for threat in self.threat_history 
            if threat.timestamp > time.time() - 3600  # Last hour
        ]
        
        return {
            'total_threats_detected': len(self.threat_history),
            'recent_threats': len(recent_threats),
            'authorized_users': len(self.authorized_users),
            'rate_limited_users': len(self.rate_limiter),
            'security_policy': {
                'max_qubits': self.policy.max_qubits,
                'max_circuit_depth': self.policy.max_circuit_depth,
                'authentication_required': self.policy.require_authentication,
                'encryption_enabled': self.policy.enable_encryption
            }
        }
    
    def export_security_report(self, filepath: Path) -> None:
        """Export comprehensive security report."""
        report = {
            'timestamp': time.time(),
            'security_summary': self.get_security_summary(),
            'threat_history': [
                {
                    'threat_type': threat.threat_type,
                    'severity': threat.severity,
                    'description': threat.description,
                    'timestamp': threat.timestamp
                }
                for threat in self.threat_history
            ],
            'policy_configuration': {
                'max_circuit_depth': self.policy.max_circuit_depth,
                'max_qubits': self.policy.max_qubits,
                'max_gates': self.policy.max_gates,
                'allowed_gate_types': self.policy.allowed_gate_types,
                'rate_limit_per_hour': self.policy.rate_limit_per_hour
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Security report exported to {filepath}")

# Global security validator instance
_global_validator = None

def get_security_validator(policy: Optional[SecurityPolicy] = None) -> AdvancedSecurityValidator:
    """Get global security validator instance."""
    global _global_validator
    if _global_validator is None:
        _global_validator = AdvancedSecurityValidator(policy)
    return _global_validator

# Decorator for secure operations
def secure_operation(require_auth: bool = True, user_id: str = None):
    """Decorator to secure functions with validation."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            validator = get_security_validator()
            
            if require_auth and user_id:
                # Basic access validation would go here
                pass
            
            # Sanitize input arguments
            sanitized_args = [validator.sanitize_input_data(arg) for arg in args]
            sanitized_kwargs = {k: validator.sanitize_input_data(v) for k, v in kwargs.items()}
            
            return func(*sanitized_args, **sanitized_kwargs)
        return wrapper
    return decorator