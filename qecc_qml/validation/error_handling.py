"""
Comprehensive error handling and recovery system.

Provides robust error handling, automatic recovery mechanisms,
and resilient system behavior for QECC-aware QML applications.
"""

import time
import logging
import traceback
import functools
from typing import Any, Dict, List, Optional, Callable, Type, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import threading
import random


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryStrategy(Enum):
    """Error recovery strategies."""
    RETRY = "retry"
    FALLBACK = "fallback"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    CIRCUIT_BREAKER = "circuit_breaker"
    ADAPTIVE = "adaptive"
    FAIL_FAST = "fail_fast"


@dataclass
class ErrorContext:
    """Context information for error handling."""
    timestamp: float
    function_name: str
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    attempt_number: int = 1
    previous_errors: List[Exception] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetryPolicy:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    backoff_strategy: str = "exponential"  # "linear", "exponential", "constant"
    retriable_exceptions: tuple = (Exception,)
    non_retriable_exceptions: tuple = ()


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker pattern."""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    success_threshold: int = 3
    half_open_max_calls: int = 2


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"       # Normal operation
    OPEN = "open"          # Failing, calls rejected
    HALF_OPEN = "half_open" # Testing if service recovered


@dataclass
class ErrorRecord:
    """Record of an error occurrence."""
    timestamp: float
    exception: Exception
    severity: ErrorSeverity
    context: ErrorContext
    resolved: bool = False
    resolution_method: Optional[str] = None
    recovery_time: Optional[float] = None


class CircuitBreaker:
    """Circuit breaker implementation for fault tolerance."""
    
    def __init__(self, config: CircuitBreakerConfig, name: str = "default"):
        self.config = config
        self.name = name
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.half_open_calls = 0
        self.lock = threading.RLock()
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator to apply circuit breaker to a function."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        return wrapper
    
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        with self.lock:
            if self.state == CircuitBreakerState.OPEN:
                if time.time() - self.last_failure_time >= self.config.recovery_timeout:
                    self.state = CircuitBreakerState.HALF_OPEN
                    self.half_open_calls = 0
                else:
                    raise CircuitBreakerOpenError(
                        f"Circuit breaker '{self.name}' is OPEN. "
                        f"Next attempt allowed in {self.config.recovery_timeout - (time.time() - self.last_failure_time):.1f}s"
                    )
            
            elif self.state == CircuitBreakerState.HALF_OPEN:
                if self.half_open_calls >= self.config.half_open_max_calls:
                    raise CircuitBreakerOpenError(
                        f"Circuit breaker '{self.name}' is HALF_OPEN and max calls exceeded"
                    )
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
            
        except Exception as e:
            self._on_failure(e)
            raise
    
    def _on_success(self):
        """Handle successful call."""
        with self.lock:
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self.state = CircuitBreakerState.CLOSED
                    self.failure_count = 0
                    self.success_count = 0
            else:
                self.failure_count = max(0, self.failure_count - 1)
    
    def _on_failure(self, exception: Exception):
        """Handle failed call."""
        with self.lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.state = CircuitBreakerState.OPEN
                self.half_open_calls = 0
            elif self.failure_count >= self.config.failure_threshold:
                self.state = CircuitBreakerState.OPEN
    
    def reset(self):
        """Reset circuit breaker to closed state."""
        with self.lock:
            self.state = CircuitBreakerState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            self.half_open_calls = 0
    
    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state."""
        with self.lock:
            return {
                'name': self.name,
                'state': self.state.value,
                'failure_count': self.failure_count,
                'success_count': self.success_count,
                'last_failure_time': self.last_failure_time,
                'half_open_calls': self.half_open_calls
            }


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""
    pass


class ErrorRecovery:
    """Error recovery mechanism with multiple strategies."""
    
    def __init__(self, strategy: RecoveryStrategy = RecoveryStrategy.RETRY):
        self.strategy = strategy
        self.fallback_functions: Dict[str, Callable] = {}
        self.degraded_functions: Dict[str, Callable] = {}
    
    def add_fallback(self, function_name: str, fallback_func: Callable):
        """Add fallback function for a specific function."""
        self.fallback_functions[function_name] = fallback_func
    
    def add_degraded_mode(self, function_name: str, degraded_func: Callable):
        """Add degraded mode function for graceful degradation."""
        self.degraded_functions[function_name] = degraded_func
    
    def recover(self, context: ErrorContext, exception: Exception) -> Any:
        """Execute recovery strategy."""
        if self.strategy == RecoveryStrategy.FALLBACK:
            return self._fallback_recovery(context, exception)
        elif self.strategy == RecoveryStrategy.GRACEFUL_DEGRADATION:
            return self._graceful_degradation(context, exception)
        elif self.strategy == RecoveryStrategy.ADAPTIVE:
            return self._adaptive_recovery(context, exception)
        else:
            raise exception
    
    def _fallback_recovery(self, context: ErrorContext, exception: Exception) -> Any:
        """Execute fallback function."""
        fallback_func = self.fallback_functions.get(context.function_name)
        if fallback_func:
            return fallback_func(*context.args, **context.kwargs)
        raise exception
    
    def _graceful_degradation(self, context: ErrorContext, exception: Exception) -> Any:
        """Execute degraded mode function."""
        degraded_func = self.degraded_functions.get(context.function_name)
        if degraded_func:
            return degraded_func(*context.args, **context.kwargs)
        raise exception
    
    def _adaptive_recovery(self, context: ErrorContext, exception: Exception) -> Any:
        """Adaptive recovery based on error history."""
        # Choose strategy based on error frequency and type
        if len(context.previous_errors) > 3:
            # Too many errors, try degraded mode
            return self._graceful_degradation(context, exception)
        else:
            # Try fallback first
            try:
                return self._fallback_recovery(context, exception)
            except:
                return self._graceful_degradation(context, exception)


class ErrorHandler:
    """
    Comprehensive error handling system.
    
    Provides automatic retry, fallback mechanisms, circuit breakers,
    and error monitoring with adaptive recovery strategies.
    """
    
    def __init__(
        self,
        default_retry_policy: Optional[RetryPolicy] = None,
        default_recovery: Optional[ErrorRecovery] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize error handler.
        
        Args:
            default_retry_policy: Default retry configuration
            default_recovery: Default recovery mechanism
            logger: Optional logger instance
        """
        self.default_retry_policy = default_retry_policy or RetryPolicy()
        self.default_recovery = default_recovery or ErrorRecovery()
        self.logger = logger or logging.getLogger(__name__)
        
        # Error tracking
        self.error_history: deque = deque(maxlen=1000)
        self.function_policies: Dict[str, RetryPolicy] = {}
        self.function_recoveries: Dict[str, ErrorRecovery] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # Statistics
        self.error_counts: Dict[str, int] = {}
        self.recovery_counts: Dict[str, int] = {}
        self.total_errors = 0
        self.total_recoveries = 0
        
        # Thread safety
        self.lock = threading.RLock()
    
    def configure_function(
        self,
        function_name: str,
        retry_policy: Optional[RetryPolicy] = None,
        recovery: Optional[ErrorRecovery] = None,
        circuit_breaker_config: Optional[CircuitBreakerConfig] = None
    ):
        """Configure error handling for a specific function."""
        if retry_policy:
            self.function_policies[function_name] = retry_policy
        
        if recovery:
            self.function_recoveries[function_name] = recovery
        
        if circuit_breaker_config:
            self.circuit_breakers[function_name] = CircuitBreaker(
                circuit_breaker_config, function_name
            )
    
    def with_error_handling(
        self,
        retry_policy: Optional[RetryPolicy] = None,
        recovery: Optional[ErrorRecovery] = None,
        circuit_breaker_config: Optional[CircuitBreakerConfig] = None
    ):
        """Decorator to add error handling to a function."""
        def decorator(func: Callable) -> Callable:
            func_name = func.__name__
            
            # Configure function-specific policies
            if retry_policy:
                self.function_policies[func_name] = retry_policy
            if recovery:
                self.function_recoveries[func_name] = recovery
            if circuit_breaker_config:
                self.circuit_breakers[func_name] = CircuitBreaker(
                    circuit_breaker_config, func_name
                )
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return self.execute_with_handling(func, *args, **kwargs)
            
            return wrapper
        
        return decorator
    
    def execute_with_handling(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with comprehensive error handling."""
        func_name = func.__name__
        
        # Get policies for this function
        retry_policy = self.function_policies.get(func_name, self.default_retry_policy)
        recovery = self.function_recoveries.get(func_name, self.default_recovery)
        circuit_breaker = self.circuit_breakers.get(func_name)
        
        # Create error context
        context = ErrorContext(
            timestamp=time.time(),
            function_name=func_name,
            args=args,
            kwargs=kwargs
        )
        
        # Apply circuit breaker if configured
        if circuit_breaker:
            try:
                return circuit_breaker.call(func, *args, **kwargs)
            except CircuitBreakerOpenError:
                # Try recovery if circuit breaker is open
                return self._attempt_recovery(context, None, recovery)
        
        # Execute with retry logic
        for attempt in range(retry_policy.max_attempts):
            context.attempt_number = attempt + 1
            
            try:
                result = func(*args, **kwargs)
                
                # Log successful recovery if this wasn't the first attempt
                if attempt > 0:
                    self._log_recovery(context, attempt)
                
                return result
                
            except Exception as e:
                context.previous_errors.append(e)
                
                # Check if this exception should not be retried
                if retry_policy.non_retriable_exceptions and isinstance(e, retry_policy.non_retriable_exceptions):
                    self._record_error(e, ErrorSeverity.HIGH, context)
                    return self._attempt_recovery(context, e, recovery)
                
                # Check if this exception is retriable
                if not isinstance(e, retry_policy.retriable_exceptions):
                    self._record_error(e, ErrorSeverity.HIGH, context)
                    return self._attempt_recovery(context, e, recovery)
                
                # Log the error
                severity = self._determine_severity(e, attempt, retry_policy.max_attempts)
                self._record_error(e, severity, context)
                
                # If this was the last attempt, try recovery or re-raise
                if attempt == retry_policy.max_attempts - 1:
                    return self._attempt_recovery(context, e, recovery)
                
                # Calculate delay for next attempt
                delay = self._calculate_delay(attempt, retry_policy)
                
                self.logger.warning(
                    f"Attempt {attempt + 1}/{retry_policy.max_attempts} failed for {func_name}: {str(e)}. "
                    f"Retrying in {delay:.1f}s"
                )
                
                time.sleep(delay)
        
        # This should never be reached, but just in case
        raise RuntimeError(f"Unexpected error in retry logic for {func_name}")
    
    def _calculate_delay(self, attempt: int, policy: RetryPolicy) -> float:
        """Calculate delay for next retry attempt."""
        if policy.backoff_strategy == "constant":
            delay = policy.base_delay
        elif policy.backoff_strategy == "linear":
            delay = policy.base_delay * (attempt + 1)
        elif policy.backoff_strategy == "exponential":
            delay = policy.base_delay * (policy.exponential_base ** attempt)
        else:
            delay = policy.base_delay
        
        # Apply maximum delay limit
        delay = min(delay, policy.max_delay)
        
        # Add jitter if enabled
        if policy.jitter:
            delay += random.uniform(0, delay * 0.1)
        
        return delay
    
    def _determine_severity(self, exception: Exception, attempt: int, max_attempts: int) -> ErrorSeverity:
        """Determine error severity based on exception type and attempt number."""
        if isinstance(exception, (KeyboardInterrupt, SystemExit)):
            return ErrorSeverity.CRITICAL
        elif isinstance(exception, (MemoryError, RecursionError)):
            return ErrorSeverity.HIGH
        elif isinstance(exception, (ValueError, TypeError)):
            return ErrorSeverity.MEDIUM
        elif attempt == max_attempts - 1:
            return ErrorSeverity.HIGH  # Final attempt
        else:
            return ErrorSeverity.LOW
    
    def _attempt_recovery(self, context: ErrorContext, exception: Exception, recovery: ErrorRecovery) -> Any:
        """Attempt error recovery."""
        try:
            result = recovery.recover(context, exception)
            self._log_recovery(context, context.attempt_number)
            return result
        except Exception as recovery_error:
            self.logger.error(
                f"Recovery failed for {context.function_name}: {str(recovery_error)}"
            )
            
            # If recovery fails, raise the original exception
            if exception:
                raise exception
            else:
                raise recovery_error
    
    def _record_error(self, exception: Exception, severity: ErrorSeverity, context: ErrorContext):
        """Record error in history and update statistics."""
        with self.lock:
            error_record = ErrorRecord(
                timestamp=time.time(),
                exception=exception,
                severity=severity,
                context=context
            )
            
            self.error_history.append(error_record)
            
            # Update statistics
            error_type = type(exception).__name__
            self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
            self.total_errors += 1
            
            # Log based on severity
            log_message = f"Error in {context.function_name} (attempt {context.attempt_number}): {str(exception)}"
            
            if severity == ErrorSeverity.CRITICAL:
                self.logger.critical(log_message)
            elif severity == ErrorSeverity.HIGH:
                self.logger.error(log_message)
            elif severity == ErrorSeverity.MEDIUM:
                self.logger.warning(log_message)
            else:
                self.logger.info(log_message)
    
    def _log_recovery(self, context: ErrorContext, attempts_used: int):
        """Log successful recovery."""
        with self.lock:
            recovery_method = f"retry_after_{attempts_used}_attempts"
            self.recovery_counts[recovery_method] = self.recovery_counts.get(recovery_method, 0) + 1
            self.total_recoveries += 1
            
            self.logger.info(
                f"Successfully recovered {context.function_name} after {attempts_used} attempts"
            )
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics."""
        with self.lock:
            # Recent error trends
            recent_errors = [
                record for record in self.error_history
                if time.time() - record.timestamp < 3600  # Last hour
            ]
            
            # Circuit breaker states
            circuit_breaker_states = {
                name: breaker.get_state()
                for name, breaker in self.circuit_breakers.items()
            }
            
            return {
                'total_errors': self.total_errors,
                'total_recoveries': self.total_recoveries,
                'error_types': dict(self.error_counts),
                'recovery_types': dict(self.recovery_counts),
                'recent_errors': len(recent_errors),
                'recovery_rate': self.total_recoveries / self.total_errors if self.total_errors > 0 else 0,
                'circuit_breakers': circuit_breaker_states,
                'error_history_size': len(self.error_history)
            }
    
    def get_function_health(self, function_name: str) -> Dict[str, Any]:
        """Get health metrics for a specific function."""
        with self.lock:
            # Filter errors for this function
            function_errors = [
                record for record in self.error_history
                if record.context.function_name == function_name
            ]
            
            recent_errors = [
                record for record in function_errors
                if time.time() - record.timestamp < 3600
            ]
            
            circuit_breaker_state = None
            if function_name in self.circuit_breakers:
                circuit_breaker_state = self.circuit_breakers[function_name].get_state()
            
            return {
                'function_name': function_name,
                'total_errors': len(function_errors),
                'recent_errors': len(recent_errors),
                'has_retry_policy': function_name in self.function_policies,
                'has_recovery': function_name in self.function_recoveries,
                'circuit_breaker': circuit_breaker_state,
                'error_rate': len(recent_errors) / 60 if recent_errors else 0  # errors per minute
            }
    
    def reset_statistics(self):
        """Reset all error statistics."""
        with self.lock:
            self.error_history.clear()
            self.error_counts.clear()
            self.recovery_counts.clear()
            self.total_errors = 0
            self.total_recoveries = 0
            
            # Reset circuit breakers
            for breaker in self.circuit_breakers.values():
                breaker.reset()
            
            self.logger.info("Reset all error handling statistics")
    
    def export_error_report(self, filename: Optional[str] = None) -> str:
        """Export comprehensive error report."""
        import json
        from datetime import datetime
        
        if filename is None:
            filename = f"error_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with self.lock:
            report = {
                'generated_at': datetime.now().isoformat(),
                'statistics': self.get_error_statistics(),
                'function_health': {
                    func_name: self.get_function_health(func_name)
                    for func_name in set(
                        list(self.function_policies.keys()) +
                        list(self.function_recoveries.keys()) +
                        list(self.circuit_breakers.keys())
                    )
                },
                'recent_errors': [
                    {
                        'timestamp': record.timestamp,
                        'exception_type': type(record.exception).__name__,
                        'exception_message': str(record.exception),
                        'severity': record.severity.value,
                        'function_name': record.context.function_name,
                        'attempt_number': record.context.attempt_number
                    }
                    for record in list(self.error_history)[-50:]  # Last 50 errors
                ]
            }
            
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            self.logger.info(f"Exported error report to {filename}")
            return filename


# Convenience decorators and functions

def with_retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    backoff_strategy: str = "exponential"
):
    """Simple retry decorator."""
    policy = RetryPolicy(
        max_attempts=max_attempts,
        base_delay=base_delay,
        backoff_strategy=backoff_strategy
    )
    
    handler = ErrorHandler(default_retry_policy=policy)
    return handler.with_error_handling(retry_policy=policy)


def with_circuit_breaker(
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0
):
    """Simple circuit breaker decorator."""
    config = CircuitBreakerConfig(
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout
    )
    
    handler = ErrorHandler()
    return handler.with_error_handling(circuit_breaker_config=config)


def with_fallback(fallback_func: Callable):
    """Simple fallback decorator."""
    def decorator(func: Callable) -> Callable:
        recovery = ErrorRecovery(RecoveryStrategy.FALLBACK)
        recovery.add_fallback(func.__name__, fallback_func)
        
        handler = ErrorHandler(default_recovery=recovery)
        return handler.with_error_handling(recovery=recovery)(func)
    
    return decorator