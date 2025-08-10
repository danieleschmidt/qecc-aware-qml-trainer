"""
Robust error recovery and fault tolerance for QECC-QML operations.
"""

import time
import random
import functools
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import traceback
import threading

from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ErrorRecord:
    """Record of an error occurrence."""
    timestamp: float
    error_type: str
    severity: ErrorSeverity
    message: str
    function_name: str
    retry_count: int = 0
    resolved: bool = False
    recovery_action: Optional[str] = None


class ErrorRecoveryManager:
    """
    Manages error recovery and fault tolerance for QECC operations.
    """
    
    def __init__(self, 
                 max_retries: int = 3,
                 base_delay: float = 1.0,
                 max_delay: float = 60.0,
                 enable_circuit_breaker: bool = True):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.enable_circuit_breaker = enable_circuit_breaker
        
        # Error tracking
        self.error_history: List[ErrorRecord] = []
        self.function_failures: Dict[str, int] = {}
        self.circuit_breakers: Dict[str, 'CircuitBreaker'] = {}
        
        # Recovery strategies
        self.recovery_strategies = self._initialize_recovery_strategies()
        
        # Thread safety
        self._lock = threading.Lock()
        
    def _initialize_recovery_strategies(self) -> Dict[str, Callable]:
        """Initialize error-specific recovery strategies."""
        return {
            "ConnectionError": self._recover_connection_error,
            "TimeoutError": self._recover_timeout_error,
            "MemoryError": self._recover_memory_error,
            "ValueError": self._recover_value_error,
            "RuntimeError": self._recover_runtime_error,
            "QuantumError": self._recover_quantum_error,
            "BackendError": self._recover_backend_error,
        }
        
    def retry_with_backoff(self, 
                          max_retries: Optional[int] = None,
                          backoff_factor: float = 2.0,
                          exceptions: Tuple = (Exception,),
                          on_retry: Optional[Callable] = None):
        """
        Decorator for automatic retry with exponential backoff.
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                retries = max_retries or self.max_retries
                delay = self.base_delay
                
                last_exception = None
                
                for attempt in range(retries + 1):
                    try:
                        # Check circuit breaker
                        if self.enable_circuit_breaker:
                            breaker = self._get_circuit_breaker(func.__name__)
                            if breaker.is_open():
                                raise CircuitBreakerOpenError(f"Circuit breaker open for {func.__name__}")
                        
                        result = func(*args, **kwargs)
                        
                        # Reset failure count on success
                        with self._lock:
                            if func.__name__ in self.function_failures:
                                del self.function_failures[func.__name__]
                        
                        return result
                        
                    except exceptions as e:
                        last_exception = e
                        
                        # Record error
                        self._record_error(func.__name__, e, attempt)
                        
                        if attempt < retries:
                            # Apply recovery strategy if available
                            recovery_applied = self._apply_recovery_strategy(e)
                            
                            # Call retry callback
                            if on_retry:
                                on_retry(attempt, e, delay)
                                
                            # Wait before retry
                            time.sleep(delay + random.uniform(0, delay * 0.1))  # Add jitter
                            delay = min(delay * backoff_factor, self.max_delay)
                            
                            logger.warning(f"Retry {attempt + 1}/{retries} for {func.__name__}: {str(e)}")
                        
                # All retries exhausted
                self._handle_final_failure(func.__name__, last_exception)
                raise last_exception
                
            return wrapper
        return decorator
        
    def _get_circuit_breaker(self, function_name: str) -> 'CircuitBreaker':
        """Get or create circuit breaker for function."""
        if function_name not in self.circuit_breakers:
            self.circuit_breakers[function_name] = CircuitBreaker(
                failure_threshold=5,
                recovery_timeout=30.0,
                expected_exception=Exception
            )
        return self.circuit_breakers[function_name]
        
    def _record_error(self, function_name: str, error: Exception, retry_count: int):
        """Record error occurrence."""
        with self._lock:
            error_record = ErrorRecord(
                timestamp=time.time(),
                error_type=type(error).__name__,
                severity=self._classify_error_severity(error),
                message=str(error),
                function_name=function_name,
                retry_count=retry_count
            )
            
            self.error_history.append(error_record)
            self.function_failures[function_name] = self.function_failures.get(function_name, 0) + 1
            
            # Update circuit breaker
            if self.enable_circuit_breaker:
                breaker = self._get_circuit_breaker(function_name)
                breaker.record_failure()
                
    def _classify_error_severity(self, error: Exception) -> ErrorSeverity:
        """Classify error severity."""
        error_type = type(error).__name__
        
        critical_errors = ["MemoryError", "SystemError", "KeyboardInterrupt"]
        high_errors = ["ConnectionError", "TimeoutError", "RuntimeError"]
        medium_errors = ["ValueError", "TypeError", "AttributeError"]
        
        if error_type in critical_errors:
            return ErrorSeverity.CRITICAL
        elif error_type in high_errors:
            return ErrorSeverity.HIGH
        elif error_type in medium_errors:
            return ErrorSeverity.MEDIUM
        else:
            return ErrorSeverity.LOW
            
    def _apply_recovery_strategy(self, error: Exception) -> bool:
        """Apply appropriate recovery strategy."""
        error_type = type(error).__name__
        
        if error_type in self.recovery_strategies:
            try:
                recovery_func = self.recovery_strategies[error_type]
                recovery_func(error)
                logger.info(f"Applied recovery strategy for {error_type}")
                return True
            except Exception as recovery_error:
                logger.error(f"Recovery strategy failed for {error_type}: {recovery_error}")
                
        return False
        
    def _handle_final_failure(self, function_name: str, error: Exception):
        """Handle final failure after all retries."""
        logger.error(f"Final failure in {function_name}: {str(error)}")
        
        # Update circuit breaker
        if self.enable_circuit_breaker:
            breaker = self._get_circuit_breaker(function_name)
            breaker.record_failure()
            
        # Mark as unresolved critical error
        with self._lock:
            if self.error_history:
                self.error_history[-1].resolved = False
                
    # Recovery strategy implementations
    def _recover_connection_error(self, error: Exception):
        """Recover from connection errors."""
        logger.info("Attempting connection recovery...")
        time.sleep(2.0)  # Give time for connection to stabilize
        
    def _recover_timeout_error(self, error: Exception):
        """Recover from timeout errors."""
        logger.info("Attempting timeout recovery...")
        # Could implement queue management, resource cleanup, etc.
        
    def _recover_memory_error(self, error: Exception):
        """Recover from memory errors."""
        logger.info("Attempting memory recovery...")
        import gc
        gc.collect()  # Force garbage collection
        
    def _recover_value_error(self, error: Exception):
        """Recover from value errors."""
        logger.info("Attempting value error recovery...")
        # Could implement parameter sanitization, default fallbacks, etc.
        
    def _recover_runtime_error(self, error: Exception):
        """Recover from runtime errors."""
        logger.info("Attempting runtime error recovery...")
        # Could implement state reset, resource reinitialization, etc.
        
    def _recover_quantum_error(self, error: Exception):
        """Recover from quantum-specific errors."""
        logger.info("Attempting quantum error recovery...")
        # Could implement circuit simplification, backend switching, etc.
        
    def _recover_backend_error(self, error: Exception):
        """Recover from backend errors."""
        logger.info("Attempting backend recovery...")
        # Could implement backend switching, configuration reset, etc.
        
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics and health metrics."""
        with self._lock:
            if not self.error_history:
                return {"total_errors": 0, "functions_affected": 0}
                
            stats = {
                "total_errors": len(self.error_history),
                "functions_affected": len(self.function_failures),
                "error_rate_per_hour": self._calculate_error_rate(),
                "severity_distribution": self._get_severity_distribution(),
                "most_problematic_functions": self._get_problematic_functions(),
                "recent_errors": self._get_recent_errors(10),
                "circuit_breaker_status": self._get_circuit_breaker_status()
            }
            
        return stats
        
    def _calculate_error_rate(self) -> float:
        """Calculate error rate per hour."""
        if not self.error_history:
            return 0.0
            
        current_time = time.time()
        hour_ago = current_time - 3600
        
        recent_errors = sum(1 for error in self.error_history if error.timestamp > hour_ago)
        return recent_errors
        
    def _get_severity_distribution(self) -> Dict[str, int]:
        """Get distribution of error severities."""
        distribution = {severity.value: 0 for severity in ErrorSeverity}
        
        for error in self.error_history:
            distribution[error.severity.value] += 1
            
        return distribution
        
    def _get_problematic_functions(self, top_n: int = 5) -> List[Tuple[str, int]]:
        """Get most problematic functions."""
        return sorted(self.function_failures.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
    def _get_recent_errors(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent error records."""
        recent = self.error_history[-count:] if self.error_history else []
        
        return [{
            "timestamp": error.timestamp,
            "function": error.function_name,
            "type": error.error_type,
            "severity": error.severity.value,
            "message": error.message,
            "resolved": error.resolved
        } for error in recent]
        
    def _get_circuit_breaker_status(self) -> Dict[str, str]:
        """Get status of all circuit breakers."""
        return {name: breaker.get_state() for name, breaker in self.circuit_breakers.items()}
        
    def reset_error_history(self):
        """Reset error history and statistics."""
        with self._lock:
            self.error_history.clear()
            self.function_failures.clear()
            
        # Reset circuit breakers
        for breaker in self.circuit_breakers.values():
            breaker.reset()
            
        logger.info("Error history reset")


class CircuitBreaker:
    """
    Circuit breaker implementation for preventing cascading failures.
    """
    
    def __init__(self, 
                 failure_threshold: int = 5,
                 recovery_timeout: float = 30.0,
                 expected_exception: type = Exception):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        
        self._lock = threading.Lock()
        
    def call(self, func: Callable, *args, **kwargs):
        """Call function through circuit breaker."""
        with self._lock:
            if self.is_open():
                if self._should_attempt_reset():
                    self.state = "HALF_OPEN"
                else:
                    raise CircuitBreakerOpenError("Circuit breaker is OPEN")
                    
            try:
                result = func(*args, **kwargs)
                self.record_success()
                return result
                
            except self.expected_exception as e:
                self.record_failure()
                raise e
                
    def record_success(self):
        """Record successful call."""
        with self._lock:
            self.failure_count = 0
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                
    def record_failure(self):
        """Record failed call."""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
                
    def is_open(self) -> bool:
        """Check if circuit breaker is open."""
        return self.state == "OPEN"
        
    def _should_attempt_reset(self) -> bool:
        """Check if should attempt to reset circuit breaker."""
        if not self.last_failure_time:
            return False
            
        return (time.time() - self.last_failure_time) >= self.recovery_timeout
        
    def reset(self):
        """Reset circuit breaker to closed state."""
        with self._lock:
            self.state = "CLOSED"
            self.failure_count = 0
            self.last_failure_time = None
            
    def get_state(self) -> str:
        """Get current circuit breaker state."""
        return self.state


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""
    pass


class FaultTolerantWrapper:
    """
    Wrapper for making any function fault-tolerant.
    """
    
    def __init__(self, recovery_manager: ErrorRecoveryManager):
        self.recovery_manager = recovery_manager
        
    def make_fault_tolerant(self, 
                           func: Callable,
                           max_retries: int = 3,
                           exceptions: Tuple = (Exception,),
                           fallback_func: Optional[Callable] = None) -> Callable:
        """Make a function fault-tolerant with automatic recovery."""
        
        @self.recovery_manager.retry_with_backoff(
            max_retries=max_retries,
            exceptions=exceptions
        )
        @functools.wraps(func)
        def fault_tolerant_func(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                if fallback_func:
                    logger.warning(f"Using fallback for {func.__name__}: {str(e)}")
                    return fallback_func(*args, **kwargs)
                raise e
                
        return fault_tolerant_func
        
    def create_safe_context(self, context_name: str = "safe_operation"):
        """Create a safe execution context."""
        return SafeExecutionContext(context_name, self.recovery_manager)


class SafeExecutionContext:
    """
    Context manager for safe execution with automatic error handling.
    """
    
    def __init__(self, name: str, recovery_manager: ErrorRecoveryManager):
        self.name = name
        self.recovery_manager = recovery_manager
        self.start_time = None
        self.errors = []
        
    def __enter__(self):
        self.start_time = time.time()
        logger.info(f"Entering safe execution context: {self.name}")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        
        if exc_type is not None:
            error_record = ErrorRecord(
                timestamp=time.time(),
                error_type=exc_type.__name__,
                severity=self.recovery_manager._classify_error_severity(exc_val),
                message=str(exc_val),
                function_name=self.name
            )
            self.errors.append(error_record)
            
            logger.error(f"Error in {self.name}: {str(exc_val)}")
            
            # Try to apply recovery
            self.recovery_manager._apply_recovery_strategy(exc_val)
            
        logger.info(f"Exiting safe execution context: {self.name} (duration: {duration:.2f}s)")
        
        # Don't suppress exceptions
        return False
        
    def get_errors(self) -> List[ErrorRecord]:
        """Get errors that occurred in this context."""
        return self.errors.copy()