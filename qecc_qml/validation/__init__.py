"""
Comprehensive validation and error handling module.

Provides robust validation, error recovery, and system resilience
for QECC-aware QML systems.
"""

from .input_validation import InputValidator, ValidationError, ValidationRule
from .error_handling import ErrorHandler, ErrorRecovery, RetryPolicy
from .system_health import HealthChecker, SystemMonitor, HealthStatus
from .comprehensive_validation import ComprehensiveValidator, ValidationResult

# Add circuit validation if it exists
try:
    from .circuit_validation import QuantumCircuitValidator, CircuitValidationError
except ImportError:
    pass

__all__ = [
    "InputValidator",
    "ValidationError", 
    "ValidationRule",
    "ErrorHandler",
    "ErrorRecovery",
    "RetryPolicy",
    "HealthChecker",
    "SystemMonitor",
    "HealthStatus",
    "ComprehensiveValidator",
    "ValidationResult",
]

# Add circuit validation classes if available
try:
    from .circuit_validation import QuantumCircuitValidator, CircuitValidationError
    __all__.extend(["QuantumCircuitValidator", "CircuitValidationError"])
except ImportError:
    pass