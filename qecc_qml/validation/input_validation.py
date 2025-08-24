"""
Comprehensive input validation for QECC-aware QML systems.

Provides robust validation of user inputs, configuration parameters,
and data integrity checks with detailed error reporting.
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
from typing import Any, Dict, List, Optional, Union, Callable, Type
from dataclasses import dataclass, field
from enum import Enum
import re
import logging
from abc import ABC, abstractmethod


class ValidationSeverity(Enum):
    """Severity levels for validation errors."""
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    """Result of a validation check."""
    is_valid: bool
    severity: ValidationSeverity
    message: str
    field_name: Optional[str] = None
    suggested_fix: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ValidationError(Exception):
    """Exception raised for validation failures."""
    
    def __init__(self, message: str, validation_results: List[ValidationResult] = None):
        self.message = message
        self.validation_results = validation_results or []
        super().__init__(message)
    
    def get_error_summary(self) -> str:
        """Get formatted error summary."""
        if not self.validation_results:
            return self.message
        
        summary = [self.message, ""]
        
        for result in self.validation_results:
            severity_symbol = {
                ValidationSeverity.WARNING: "⚠️",
                ValidationSeverity.ERROR: "❌",
                ValidationSeverity.CRITICAL: "🚨"
            }.get(result.severity, "❓")
            
            line = f"{severity_symbol} {result.message}"
            if result.field_name:
                line = f"{severity_symbol} [{result.field_name}] {result.message}"
            
            summary.append(line)
            
            if result.suggested_fix:
                summary.append(f"   💡 Suggestion: {result.suggested_fix}")
        
        return "\n".join(summary)


class ValidationRule(ABC):
    """Abstract base class for validation rules."""
    
    def __init__(self, name: str, severity: ValidationSeverity = ValidationSeverity.ERROR):
        self.name = name
        self.severity = severity
    
    @abstractmethod
    def validate(self, value: Any, context: Dict[str, Any] = None) -> ValidationResult:
        """Validate a value and return result."""
        pass


class RangeValidationRule(ValidationRule):
    """Validation rule for numeric ranges."""
    
    def __init__(
        self,
        name: str,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        inclusive: bool = True,
        severity: ValidationSeverity = ValidationSeverity.ERROR
    ):
        super().__init__(name, severity)
        self.min_value = min_value
        self.max_value = max_value
        self.inclusive = inclusive
    
    def validate(self, value: Any, context: Dict[str, Any] = None) -> ValidationResult:
        """Validate numeric range."""
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return ValidationResult(
                is_valid=False,
                severity=self.severity,
                message=f"Value must be numeric, got {type(value).__name__}",
                suggested_fix="Provide a numeric value"
            )
        
        # Check minimum
        if self.min_value is not None:
            if self.inclusive and numeric_value < self.min_value:
                return ValidationResult(
                    is_valid=False,
                    severity=self.severity,
                    message=f"Value {numeric_value} is below minimum {self.min_value}",
                    suggested_fix=f"Use value >= {self.min_value}"
                )
            elif not self.inclusive and numeric_value <= self.min_value:
                return ValidationResult(
                    is_valid=False,
                    severity=self.severity,
                    message=f"Value {numeric_value} must be greater than {self.min_value}",
                    suggested_fix=f"Use value > {self.min_value}"
                )
        
        # Check maximum
        if self.max_value is not None:
            if self.inclusive and numeric_value > self.max_value:
                return ValidationResult(
                    is_valid=False,
                    severity=self.severity,
                    message=f"Value {numeric_value} exceeds maximum {self.max_value}",
                    suggested_fix=f"Use value <= {self.max_value}"
                )
            elif not self.inclusive and numeric_value >= self.max_value:
                return ValidationResult(
                    is_valid=False,
                    severity=self.severity,
                    message=f"Value {numeric_value} must be less than {self.max_value}",
                    suggested_fix=f"Use value < {self.max_value}"
                )
        
        return ValidationResult(
            is_valid=True,
            severity=self.severity,
            message="Value is within valid range"
        )


class TypeValidationRule(ValidationRule):
    """Validation rule for type checking."""
    
    def __init__(
        self,
        name: str,
        expected_type: Union[Type, tuple],
        allow_none: bool = False,
        severity: ValidationSeverity = ValidationSeverity.ERROR
    ):
        super().__init__(name, severity)
        self.expected_type = expected_type
        self.allow_none = allow_none
    
    def validate(self, value: Any, context: Dict[str, Any] = None) -> ValidationResult:
        """Validate type."""
        if value is None and self.allow_none:
            return ValidationResult(
                is_valid=True,
                severity=self.severity,
                message="None value is allowed"
            )
        
        if isinstance(self.expected_type, tuple):
            type_names = [t.__name__ for t in self.expected_type]
            type_description = " or ".join(type_names)
        else:
            type_description = self.expected_type.__name__
        
        if not isinstance(value, self.expected_type):
            return ValidationResult(
                is_valid=False,
                severity=self.severity,
                message=f"Expected {type_description}, got {type(value).__name__}",
                suggested_fix=f"Convert value to {type_description}"
            )
        
        return ValidationResult(
            is_valid=True,
            severity=self.severity,
            message=f"Value is of correct type ({type(value).__name__})"
        )


class ArrayShapeValidationRule(ValidationRule):
    """Validation rule for numpy array shapes."""
    
    def __init__(
        self,
        name: str,
        expected_shape: Optional[tuple] = None,
        min_dims: Optional[int] = None,
        max_dims: Optional[int] = None,
        severity: ValidationSeverity = ValidationSeverity.ERROR
    ):
        super().__init__(name, severity)
        self.expected_shape = expected_shape
        self.min_dims = min_dims
        self.max_dims = max_dims
    
    def validate(self, value: Any, context: Dict[str, Any] = None) -> ValidationResult:
        """Validate array shape."""
        if not isinstance(value, np.ndarray):
            return ValidationResult(
                is_valid=False,
                severity=self.severity,
                message=f"Expected numpy array, got {type(value).__name__}",
                suggested_fix="Convert to numpy array"
            )
        
        shape = value.shape
        
        # Check exact shape
        if self.expected_shape is not None:
            if shape != self.expected_shape:
                return ValidationResult(
                    is_valid=False,
                    severity=self.severity,
                    message=f"Expected shape {self.expected_shape}, got {shape}",
                    suggested_fix=f"Reshape array to {self.expected_shape}"
                )
        
        # Check dimensions
        num_dims = len(shape)
        
        if self.min_dims is not None and num_dims < self.min_dims:
            return ValidationResult(
                is_valid=False,
                severity=self.severity,
                message=f"Array has {num_dims} dimensions, minimum {self.min_dims} required",
                suggested_fix=f"Use array with at least {self.min_dims} dimensions"
            )
        
        if self.max_dims is not None and num_dims > self.max_dims:
            return ValidationResult(
                is_valid=False,
                severity=self.severity,
                message=f"Array has {num_dims} dimensions, maximum {self.max_dims} allowed",
                suggested_fix=f"Use array with at most {self.max_dims} dimensions"
            )
        
        return ValidationResult(
            is_valid=True,
            severity=self.severity,
            message=f"Array shape {shape} is valid"
        )


class ChoiceValidationRule(ValidationRule):
    """Validation rule for choice from predefined options."""
    
    def __init__(
        self,
        name: str,
        valid_choices: List[Any],
        case_sensitive: bool = True,
        severity: ValidationSeverity = ValidationSeverity.ERROR
    ):
        super().__init__(name, severity)
        self.valid_choices = valid_choices
        self.case_sensitive = case_sensitive
    
    def validate(self, value: Any, context: Dict[str, Any] = None) -> ValidationResult:
        """Validate choice."""
        choices_to_check = self.valid_choices
        value_to_check = value
        
        if not self.case_sensitive and isinstance(value, str):
            choices_to_check = [str(c).lower() for c in self.valid_choices]
            value_to_check = value.lower()
        
        if value_to_check not in choices_to_check:
            return ValidationResult(
                is_valid=False,
                severity=self.severity,
                message=f"Invalid choice '{value}'. Valid options: {self.valid_choices}",
                suggested_fix=f"Use one of: {self.valid_choices}"
            )
        
        return ValidationResult(
            is_valid=True,
            severity=self.severity,
            message=f"Value '{value}' is a valid choice"
        )


class RegexValidationRule(ValidationRule):
    """Validation rule using regular expressions."""
    
    def __init__(
        self,
        name: str,
        pattern: str,
        flags: int = 0,
        severity: ValidationSeverity = ValidationSeverity.ERROR
    ):
        super().__init__(name, severity)
        self.pattern = pattern
        self.regex = re.compile(pattern, flags)
    
    def validate(self, value: Any, context: Dict[str, Any] = None) -> ValidationResult:
        """Validate using regex pattern."""
        if not isinstance(value, str):
            return ValidationResult(
                is_valid=False,
                severity=self.severity,
                message=f"Expected string for regex validation, got {type(value).__name__}",
                suggested_fix="Convert to string"
            )
        
        if not self.regex.match(value):
            return ValidationResult(
                is_valid=False,
                severity=self.severity,
                message=f"String '{value}' does not match pattern '{self.pattern}'",
                suggested_fix=f"Use string matching pattern: {self.pattern}"
            )
        
        return ValidationResult(
            is_valid=True,
            severity=self.severity,
            message=f"String matches pattern '{self.pattern}'"
        )


class CustomValidationRule(ValidationRule):
    """Custom validation rule using a function."""
    
    def __init__(
        self,
        name: str,
        validator_func: Callable[[Any, Dict[str, Any]], bool],
        error_message: str = "Custom validation failed",
        success_message: str = "Custom validation passed",
        severity: ValidationSeverity = ValidationSeverity.ERROR
    ):
        super().__init__(name, severity)
        self.validator_func = validator_func
        self.error_message = error_message
        self.success_message = success_message
    
    def validate(self, value: Any, context: Dict[str, Any] = None) -> ValidationResult:
        """Validate using custom function."""
        try:
            is_valid = self.validator_func(value, context or {})
            
            return ValidationResult(
                is_valid=is_valid,
                severity=self.severity,
                message=self.success_message if is_valid else self.error_message
            )
            
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.CRITICAL,
                message=f"Validation function error: {str(e)}",
                suggested_fix="Check validation function implementation"
            )


class InputValidator:
    """
    Comprehensive input validation system.
    
    Provides configurable validation with detailed error reporting,
    automatic fixes, and context-aware validation rules.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize input validator.
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        
        # Validation rules by field name
        self.field_rules: Dict[str, List[ValidationRule]] = {}
        
        # Global validation rules
        self.global_rules: List[ValidationRule] = []
        
        # Configuration
        self.stop_on_first_error = False
        self.auto_fix_enabled = False
        
        # Initialize common rules
        self._initialize_qecc_rules()
    
    def _initialize_qecc_rules(self):
        """Initialize common validation rules for QECC systems."""
        # Quantum system parameters
        self.add_field_rule("num_qubits", TypeValidationRule("qubit_type", int))
        self.add_field_rule("num_qubits", RangeValidationRule("qubit_range", min_value=1, max_value=100))
        
        self.add_field_rule("num_layers", TypeValidationRule("layer_type", int))
        self.add_field_rule("num_layers", RangeValidationRule("layer_range", min_value=1, max_value=50))
        
        self.add_field_rule("code_distance", TypeValidationRule("distance_type", int))
        self.add_field_rule("code_distance", RangeValidationRule("distance_range", min_value=3, max_value=15))
        self.add_field_rule("code_distance", CustomValidationRule(
            "distance_odd", 
            lambda x, _: x % 2 == 1,
            "Code distance must be odd",
            "Code distance is odd"
        ))
        
        # Training parameters
        self.add_field_rule("learning_rate", TypeValidationRule("lr_type", (float, int)))
        self.add_field_rule("learning_rate", RangeValidationRule("lr_range", min_value=1e-6, max_value=1.0))
        
        self.add_field_rule("epochs", TypeValidationRule("epochs_type", int))
        self.add_field_rule("epochs", RangeValidationRule("epochs_range", min_value=1, max_value=10000))
        
        self.add_field_rule("batch_size", TypeValidationRule("batch_type", int))
        self.add_field_rule("batch_size", RangeValidationRule("batch_range", min_value=1, max_value=1024))
        
        self.add_field_rule("shots", TypeValidationRule("shots_type", int))
        self.add_field_rule("shots", RangeValidationRule("shots_range", min_value=1, max_value=100000))
        
        # Noise model parameters
        self.add_field_rule("gate_error_rate", TypeValidationRule("gate_error_type", (float, int)))
        self.add_field_rule("gate_error_rate", RangeValidationRule("gate_error_range", min_value=0.0, max_value=0.5))
        
        self.add_field_rule("readout_error_rate", TypeValidationRule("readout_error_type", (float, int)))
        self.add_field_rule("readout_error_rate", RangeValidationRule("readout_error_range", min_value=0.0, max_value=0.5))
        
        # Entanglement patterns
        self.add_field_rule("entanglement", TypeValidationRule("entanglement_type", str))
        self.add_field_rule("entanglement", ChoiceValidationRule(
            "entanglement_choice", 
            ["circular", "linear", "full", "all-to-all"],
            case_sensitive=False
        ))
        
        # Feature maps
        self.add_field_rule("feature_map", TypeValidationRule("feature_map_type", str))
        self.add_field_rule("feature_map", ChoiceValidationRule(
            "feature_map_choice",
            ["amplitude_encoding", "angle_encoding", "iqp", "pauli_feature_map"],
            case_sensitive=False
        ))
    
    def add_field_rule(self, field_name: str, rule: ValidationRule):
        """Add validation rule for a specific field."""
        if field_name not in self.field_rules:
            self.field_rules[field_name] = []
        
        self.field_rules[field_name].append(rule)
        self.logger.debug(f"Added validation rule '{rule.name}' for field '{field_name}'")
    
    def add_global_rule(self, rule: ValidationRule):
        """Add global validation rule (applies to all fields)."""
        self.global_rules.append(rule)
        self.logger.debug(f"Added global validation rule '{rule.name}'")
    
    def validate_field(self, field_name: str, value: Any, context: Dict[str, Any] = None) -> List[ValidationResult]:
        """
        Validate a single field.
        
        Args:
            field_name: Name of field to validate
            value: Value to validate
            context: Additional context for validation
            
        Returns:
            List of validation results
        """
        results = []
        context = context or {}
        
        # Apply field-specific rules
        field_rules = self.field_rules.get(field_name, [])
        for rule in field_rules:
            try:
                result = rule.validate(value, context)
                result.field_name = field_name
                results.append(result)
                
                # Stop on first error if configured
                if self.stop_on_first_error and not result.is_valid:
                    break
                    
            except Exception as e:
                self.logger.error(f"Error in validation rule '{rule.name}' for field '{field_name}': {e}")
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.CRITICAL,
                    message=f"Validation rule error: {str(e)}",
                    field_name=field_name
                ))
        
        # Apply global rules
        for rule in self.global_rules:
            try:
                result = rule.validate(value, context)
                result.field_name = field_name
                results.append(result)
                
                if self.stop_on_first_error and not result.is_valid:
                    break
                    
            except Exception as e:
                self.logger.error(f"Error in global validation rule '{rule.name}': {e}")
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.CRITICAL,
                    message=f"Global validation rule error: {str(e)}",
                    field_name=field_name
                ))
        
        return results
    
    def validate_dict(self, data: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, List[ValidationResult]]:
        """
        Validate a dictionary of field-value pairs.
        
        Args:
            data: Dictionary to validate
            context: Additional context for validation
            
        Returns:
            Dictionary of field name to validation results
        """
        all_results = {}
        context = context or {}
        
        for field_name, value in data.items():
            results = self.validate_field(field_name, value, context)
            if results:  # Only include fields with validation results
                all_results[field_name] = results
        
        return all_results
    
    def validate_and_raise(self, data: Dict[str, Any], context: Dict[str, Any] = None):
        """
        Validate data and raise ValidationError if any errors found.
        
        Args:
            data: Dictionary to validate
            context: Additional context for validation
            
        Raises:
            ValidationError: If validation fails
        """
        all_results = self.validate_dict(data, context)
        
        # Collect all error results
        error_results = []
        warning_results = []
        
        for field_results in all_results.values():
            for result in field_results:
                if not result.is_valid:
                    if result.severity == ValidationSeverity.WARNING:
                        warning_results.append(result)
                    else:
                        error_results.append(result)
        
        # Log warnings
        for warning in warning_results:
            self.logger.warning(f"Validation warning: {warning.message}")
        
        # Raise error if any critical issues found
        if error_results:
            error_count = len(error_results)
            critical_count = len([r for r in error_results if r.severity == ValidationSeverity.CRITICAL])
            
            message = f"Validation failed with {error_count} error(s)"
            if critical_count > 0:
                message += f" ({critical_count} critical)"
            
            raise ValidationError(message, error_results)
    
    def get_validation_summary(self, data: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Get summary of validation results.
        
        Args:
            data: Dictionary to validate
            context: Additional context for validation
            
        Returns:
            Validation summary
        """
        all_results = self.validate_dict(data, context)
        
        total_results = 0
        valid_results = 0
        warning_count = 0
        error_count = 0
        critical_count = 0
        
        for field_results in all_results.values():
            for result in field_results:
                total_results += 1
                if result.is_valid:
                    valid_results += 1
                else:
                    if result.severity == ValidationSeverity.WARNING:
                        warning_count += 1
                    elif result.severity == ValidationSeverity.ERROR:
                        error_count += 1
                    elif result.severity == ValidationSeverity.CRITICAL:
                        critical_count += 1
        
        return {
            'total_checks': total_results,
            'valid_checks': valid_results,
            'warning_count': warning_count,
            'error_count': error_count,
            'critical_count': critical_count,
            'success_rate': valid_results / total_results if total_results > 0 else 1.0,
            'overall_valid': error_count == 0 and critical_count == 0,
            'fields_checked': list(all_results.keys())
        }
    
    def suggest_fixes(self, data: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, str]:
        """
        Suggest fixes for validation errors.
        
        Args:
            data: Dictionary to validate
            context: Additional context for validation
            
        Returns:
            Dictionary of field name to suggested fix
        """
        all_results = self.validate_dict(data, context)
        fixes = {}
        
        for field_name, field_results in all_results.items():
            for result in field_results:
                if not result.is_valid and result.suggested_fix:
                    if field_name not in fixes:
                        fixes[field_name] = []
                    fixes[field_name].append(result.suggested_fix)
        
        # Combine multiple fixes per field
        combined_fixes = {}
        for field_name, fix_list in fixes.items():
            if len(fix_list) == 1:
                combined_fixes[field_name] = fix_list[0]
            else:
                combined_fixes[field_name] = "; ".join(fix_list)
        
        return combined_fixes
    
    def auto_fix(self, data: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Attempt automatic fixes for common validation errors.
        
        Args:
            data: Dictionary to validate and fix
            context: Additional context
            
        Returns:
            Fixed data dictionary
        """
        if not self.auto_fix_enabled:
            self.logger.warning("Auto-fix is not enabled")
            return data.copy()
        
        fixed_data = data.copy()
        
        # Common auto-fixes
        for field_name, value in fixed_data.items():
            
            # Fix numeric strings
            if field_name in ['num_qubits', 'num_layers', 'epochs', 'batch_size', 'shots']:
                if isinstance(value, str) and value.isdigit():
                    fixed_data[field_name] = int(value)
                    self.logger.info(f"Auto-fixed {field_name}: converted string '{value}' to int")
            
            # Fix float strings
            if field_name in ['learning_rate', 'gate_error_rate', 'readout_error_rate']:
                if isinstance(value, str):
                    try:
                        fixed_data[field_name] = float(value)
                        self.logger.info(f"Auto-fixed {field_name}: converted string '{value}' to float")
                    except ValueError:
                        pass
            
            # Fix case issues
            if field_name in ['entanglement', 'feature_map']:
                if isinstance(value, str):
                    fixed_data[field_name] = value.lower()
            
            # Fix boolean strings
            if isinstance(value, str) and value.lower() in ['true', 'false']:
                fixed_data[field_name] = value.lower() == 'true'
                self.logger.info(f"Auto-fixed {field_name}: converted string '{value}' to bool")
            
            # Clamp values to valid ranges
            if field_name == 'learning_rate' and isinstance(value, (int, float)):
                if value > 1.0:
                    fixed_data[field_name] = 1.0
                    self.logger.info(f"Auto-fixed {field_name}: clamped {value} to 1.0")
                elif value <= 0:
                    fixed_data[field_name] = 0.001
                    self.logger.info(f"Auto-fixed {field_name}: clamped {value} to 0.001")
        
        return fixed_data
    
    def enable_auto_fix(self, enabled: bool = True):
        """Enable or disable automatic fixes."""
        self.auto_fix_enabled = enabled
        self.logger.info(f"Auto-fix {'enabled' if enabled else 'disabled'}")
    
    def configure(self, stop_on_first_error: bool = None, auto_fix: bool = None):
        """Configure validator behavior."""
        if stop_on_first_error is not None:
            self.stop_on_first_error = stop_on_first_error
            
        if auto_fix is not None:
            self.enable_auto_fix(auto_fix)
    
    def clear_rules(self):
        """Clear all validation rules."""
        self.field_rules.clear()
        self.global_rules.clear()
        self.logger.info("Cleared all validation rules")
    
    def get_rule_summary(self) -> Dict[str, Any]:
        """Get summary of configured validation rules."""
        field_rule_count = {
            field: len(rules) for field, rules in self.field_rules.items()
        }
        
        return {
            'field_rules': field_rule_count,
            'global_rules': len(self.global_rules),
            'total_field_rules': sum(field_rule_count.values()),
            'total_rules': sum(field_rule_count.values()) + len(self.global_rules),
            'configured_fields': list(self.field_rules.keys()),
            'auto_fix_enabled': self.auto_fix_enabled,
            'stop_on_first_error': self.stop_on_first_error
        }


# Convenience functions for common validations

def validate_qnn_config(config: Dict[str, Any]) -> None:
    """Validate quantum neural network configuration."""
    validator = InputValidator()
    validator.validate_and_raise(config)


def validate_training_config(config: Dict[str, Any]) -> None:
    """Validate training configuration."""
    validator = InputValidator()
    validator.validate_and_raise(config)


def validate_noise_model_config(config: Dict[str, Any]) -> None:
    """Validate noise model configuration."""
    validator = InputValidator()
    validator.validate_and_raise(config)