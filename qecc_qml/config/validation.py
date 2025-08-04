"""
Configuration validation utilities.
"""

from typing import Dict, Any, List, Tuple, Optional
import re
from pathlib import Path

from ..utils.validation import ValidationError
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class ConfigValidator:
    """
    Comprehensive configuration validator for QECC-aware QML settings.
    """
    
    def __init__(self):
        self.errors = []
        self.warnings = []
    
    def validate_config(self, config: Dict[str, Any]) -> Tuple[bool, List[str], List[str]]:
        """
        Validate complete configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Tuple of (is_valid, errors, warnings)
        """
        self.errors.clear()
        self.warnings.clear()
        
        logger.debug("Starting comprehensive configuration validation")
        
        # Validate each section
        self._validate_quantum_settings(config.get('quantum', {}))
        self._validate_noise_settings(config.get('noise', {}))
        self._validate_training_settings(config.get('training', {}))
        self._validate_error_correction_settings(config.get('error_correction', {}))
        self._validate_logging_settings(config.get('logging', {}))
        self._validate_security_settings(config.get('security', {}))
        self._validate_performance_settings(config.get('performance', {}))
        
        # Cross-validation checks
        self._validate_cross_dependencies(config)
        
        is_valid = len(self.errors) == 0
        
        logger.info(f"Configuration validation completed: {len(self.errors)} errors, {len(self.warnings)} warnings")
        
        return is_valid, self.errors.copy(), self.warnings.copy()
    
    def _validate_quantum_settings(self, quantum_config: Dict[str, Any]):
        """Validate quantum-specific settings."""
        # Default shots validation
        shots = quantum_config.get('default_shots', 1024)
        if not isinstance(shots, int) or shots < 1:
            self.errors.append("quantum.default_shots must be a positive integer")
        elif shots < 100:
            self.warnings.append("quantum.default_shots < 100 may give noisy results")
        elif shots > 100000:
            self.warnings.append("quantum.default_shots > 100000 will be very slow")
        
        # Max qubits validation
        max_qubits = quantum_config.get('max_qubits', 50)
        if not isinstance(max_qubits, int) or max_qubits < 1:
            self.errors.append("quantum.max_qubits must be a positive integer")
        elif max_qubits > 100:
            self.warnings.append("quantum.max_qubits > 100 may exceed hardware capabilities")
        
        # Circuit depth validation
        max_depth = quantum_config.get('max_circuit_depth', 100)
        if not isinstance(max_depth, int) or max_depth < 1:
            self.errors.append("quantum.max_circuit_depth must be a positive integer")
        elif max_depth > 200:
            self.warnings.append("quantum.max_circuit_depth > 200 may suffer from decoherence")
        
        # Backend validation
        backend = quantum_config.get('default_backend', 'aer_simulator')
        valid_backends = [
            'aer_simulator', 'statevector_simulator', 'qasm_simulator',
            'ibm_lagos', 'ibm_nairobi', 'ibm_washington',
            'google_sycamore', 'ionq_harmony'
        ]
        if backend not in valid_backends:
            self.warnings.append(f"quantum.default_backend '{backend}' may not be supported")
        
        # Optimization level validation
        opt_level = quantum_config.get('optimization_level', 1)
        if not isinstance(opt_level, int) or not 0 <= opt_level <= 3:
            self.errors.append("quantum.optimization_level must be integer in range [0,3]")
    
    def _validate_noise_settings(self, noise_config: Dict[str, Any]):
        """Validate noise model settings."""
        # Gate error rate validation
        gate_error = noise_config.get('default_gate_error_rate', 0.001)
        if not isinstance(gate_error, (int, float)) or not 0 <= gate_error <= 1:
            self.errors.append("noise.default_gate_error_rate must be in range [0,1]")
        elif gate_error > 0.1:
            self.warnings.append("noise.default_gate_error_rate > 0.1 is very high")
        
        # Readout error rate validation
        readout_error = noise_config.get('default_readout_error_rate', 0.01)
        if not isinstance(readout_error, (int, float)) or not 0 <= readout_error <= 1:
            self.errors.append("noise.default_readout_error_rate must be in range [0,1]")
        
        # Coherence times validation
        T1 = noise_config.get('default_T1', 50e-6)
        T2 = noise_config.get('default_T2', 70e-6)
        
        if not isinstance(T1, (int, float)) or T1 <= 0:
            self.errors.append("noise.default_T1 must be positive")
        elif T1 < 1e-6:
            self.warnings.append("noise.default_T1 < 1μs is very short")
        
        if not isinstance(T2, (int, float)) or T2 <= 0:
            self.errors.append("noise.default_T2 must be positive")
        elif T2 < 1e-6:
            self.warnings.append("noise.default_T2 < 1μs is very short")
        
        # Physical consistency check
        if isinstance(T1, (int, float)) and isinstance(T2, (int, float)):
            if T2 > 2 * T1:
                self.warnings.append("noise.default_T2 > 2*T1 may be unphysical")
    
    def _validate_training_settings(self, training_config: Dict[str, Any]):
        """Validate training configuration."""
        # Learning rate validation
        lr = training_config.get('default_learning_rate', 0.01)
        if not isinstance(lr, (int, float)) or lr <= 0:
            self.errors.append("training.default_learning_rate must be positive")
        elif lr > 1.0:
            self.warnings.append("training.default_learning_rate > 1.0 may cause instability")
        elif lr < 1e-6:
            self.warnings.append("training.default_learning_rate < 1e-6 may be too small")
        
        # Epochs validation
        epochs = training_config.get('default_epochs', 50)
        if not isinstance(epochs, int) or epochs < 1:
            self.errors.append("training.default_epochs must be a positive integer")
        elif epochs > 1000:
            self.warnings.append("training.default_epochs > 1000 will take very long")
        
        # Batch size validation
        batch_size = training_config.get('default_batch_size', 32)
        if not isinstance(batch_size, int) or batch_size < 1:
            self.errors.append("training.default_batch_size must be a positive integer")
        elif batch_size > 1000:
            self.warnings.append("training.default_batch_size > 1000 may use excessive memory")
        
        # Validation split validation
        val_split = training_config.get('validation_split', 0.2)
        if not isinstance(val_split, (int, float)) or not 0 <= val_split <= 1:
            self.errors.append("training.validation_split must be in range [0,1]")
        
        # Training timeout validation
        timeout = training_config.get('max_training_time', 3600)
        if not isinstance(timeout, int) or timeout < 1:
            self.errors.append("training.max_training_time must be a positive integer")
        elif timeout < 60:
            self.warnings.append("training.max_training_time < 60s may be too short")
        
        # Patience validation
        patience = training_config.get('patience', 10)
        if not isinstance(patience, int) or patience < 1:
            self.errors.append("training.patience must be a positive integer")
    
    def _validate_error_correction_settings(self, ec_config: Dict[str, Any]):
        """Validate error correction settings."""
        # Scheme validation
        scheme = ec_config.get('default_scheme', 'surface_code')
        valid_schemes = ['surface_code', 'color_code', 'repetition_code', 'steane_code']
        if scheme not in valid_schemes:
            self.errors.append(f"error_correction.default_scheme must be one of {valid_schemes}")
        
        # Distance validation
        distance = ec_config.get('default_distance', 3)
        if not isinstance(distance, int):
            self.errors.append("error_correction.default_distance must be an integer")
        elif distance < 3:
            self.errors.append("error_correction.default_distance must be >= 3")
        elif distance % 2 == 0:
            self.errors.append("error_correction.default_distance must be odd")
        elif distance > 15:
            self.warnings.append("error_correction.default_distance > 15 will require many qubits")
        
        # Syndrome extraction frequency validation
        freq = ec_config.get('syndrome_extraction_frequency', 1)
        if not isinstance(freq, int) or freq < 1:
            self.errors.append("error_correction.syndrome_extraction_frequency must be a positive integer")
        elif freq > 10:
            self.warnings.append("error_correction.syndrome_extraction_frequency > 10 may add significant overhead")
        
        # Decoder timeout validation
        timeout = ec_config.get('decoder_timeout', 1.0)
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            self.errors.append("error_correction.decoder_timeout must be positive")
        elif timeout > 10.0:
            self.warnings.append("error_correction.decoder_timeout > 10s is very long")
    
    def _validate_logging_settings(self, logging_config: Dict[str, Any]):
        """Validate logging configuration."""
        # Log level validation
        level = logging_config.get('level', 'INFO')
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if level not in valid_levels:
            self.errors.append(f"logging.level must be one of {valid_levels}")
        
        # Log directory validation
        log_dir = logging_config.get('log_directory', 'logs')
        if not isinstance(log_dir, str):
            self.errors.append("logging.log_directory must be a string")
        else:
            # Check if directory is writable (if it exists)
            log_path = Path(log_dir)
            if log_path.exists() and not log_path.is_dir():
                self.errors.append(f"logging.log_directory '{log_dir}' exists but is not a directory")
        
        # File size validation
        max_size = logging_config.get('max_file_size', 10 * 1024 * 1024)
        if not isinstance(max_size, int) or max_size < 1024:
            self.errors.append("logging.max_file_size must be an integer >= 1024")
        elif max_size > 1024 * 1024 * 1024:  # 1GB
            self.warnings.append("logging.max_file_size > 1GB may use excessive disk space")
        
        # Backup count validation
        backup_count = logging_config.get('backup_count', 5)
        if not isinstance(backup_count, int) or backup_count < 0:
            self.errors.append("logging.backup_count must be a non-negative integer")
        elif backup_count > 100:
            self.warnings.append("logging.backup_count > 100 may use excessive disk space")
    
    def _validate_security_settings(self, security_config: Dict[str, Any]):
        """Validate security configuration."""
        # Max input size validation
        max_size = security_config.get('max_input_size', 1000000)
        if not isinstance(max_size, int) or max_size < 1024:
            self.errors.append("security.max_input_size must be an integer >= 1024")
        elif max_size > 1024 * 1024 * 1024:  # 1GB
            self.warnings.append("security.max_input_size > 1GB may allow DoS attacks")
        
        # File extensions validation
        extensions = security_config.get('allowed_file_extensions', [])
        if not isinstance(extensions, list):
            self.errors.append("security.allowed_file_extensions must be a list")
        else:
            for ext in extensions:
                if not isinstance(ext, str) or not ext.startswith('.'):
                    self.errors.append(f"File extension '{ext}' must be string starting with '.'")
        
        # Base directory validation
        base_dir = security_config.get('base_directory', '.')
        if not isinstance(base_dir, str):
            self.errors.append("security.base_directory must be a string")
        else:
            base_path = Path(base_dir)
            if not base_path.exists():
                self.warnings.append(f"security.base_directory '{base_dir}' does not exist")
            elif not base_path.is_dir():
                self.errors.append(f"security.base_directory '{base_dir}' is not a directory")
    
    def _validate_performance_settings(self, perf_config: Dict[str, Any]):
        """Validate performance configuration."""
        # Max workers validation
        max_workers = perf_config.get('max_workers', 4)
        if not isinstance(max_workers, int) or max_workers < 1:
            self.errors.append("performance.max_workers must be a positive integer")
        elif max_workers > 64:
            self.warnings.append("performance.max_workers > 64 may cause overhead")
        
        # Memory limit validation
        memory_limit = perf_config.get('memory_limit_gb')
        if memory_limit is not None:
            if not isinstance(memory_limit, (int, float)) or memory_limit <= 0:
                self.errors.append("performance.memory_limit_gb must be positive if specified")
            elif memory_limit < 1:
                self.warnings.append("performance.memory_limit_gb < 1GB may be too restrictive")
        
        # Cache size validation
        cache_size = perf_config.get('cache_size', 1000)
        if not isinstance(cache_size, int) or cache_size < 0:
            self.errors.append("performance.cache_size must be a non-negative integer")
        elif cache_size > 100000:
            self.warnings.append("performance.cache_size > 100000 may use excessive memory")
    
    def _validate_cross_dependencies(self, config: Dict[str, Any]):
        """Validate cross-dependencies between different configuration sections."""
        quantum_config = config.get('quantum', {})
        noise_config = config.get('noise', {})
        training_config = config.get('training', {})
        ec_config = config.get('error_correction', {})
        
        # Check if error correction distance is reasonable for max qubits
        max_qubits = quantum_config.get('max_qubits', 50)
        distance = ec_config.get('default_distance', 3)
        scheme = ec_config.get('default_scheme', 'surface_code')
        
        if scheme == 'surface_code':
            required_physical_qubits = distance ** 2
            if required_physical_qubits > max_qubits:
                self.warnings.append(
                    f"Surface code with distance {distance} requires {required_physical_qubits} "
                    f"physical qubits but max_qubits is {max_qubits}"
                )
        
        # Check if shots and batch size are compatible
        shots = quantum_config.get('default_shots', 1024)
        batch_size = training_config.get('default_batch_size', 32)
        
        if shots * batch_size > 100000:
            self.warnings.append(
                f"shots ({shots}) × batch_size ({batch_size}) = {shots * batch_size} "
                "may be very slow per batch"
            )
        
        # Check noise levels vs error correction capability
        gate_error = noise_config.get('default_gate_error_rate', 0.001)
        
        if scheme == 'surface_code' and gate_error > 0.01:
            self.warnings.append(
                f"Gate error rate {gate_error} exceeds surface code threshold (~0.01)"
            )
        elif scheme == 'repetition_code' and gate_error > 0.1:
            self.warnings.append(
                f"Gate error rate {gate_error} exceeds repetition code threshold (~0.1)"
            )
    
    def get_validation_report(self) -> str:
        """Generate a human-readable validation report."""
        report = ["Configuration Validation Report", "=" * 35, ""]
        
        if not self.errors and not self.warnings:
            report.append("✅ Configuration is valid with no issues.")
        else:
            if self.errors:
                report.extend([
                    f"❌ Errors ({len(self.errors)}):",
                    "-" * 20
                ])
                for i, error in enumerate(self.errors, 1):
                    report.append(f"{i:2d}. {error}")
                report.append("")
            
            if self.warnings:
                report.extend([
                    f"⚠️  Warnings ({len(self.warnings)}):",
                    "-" * 22
                ])
                for i, warning in enumerate(self.warnings, 1):
                    report.append(f"{i:2d}. {warning}")
                report.append("")
        
        return "\n".join(report)
    
    def validate_config_file(self, config_path: str) -> Tuple[bool, str]:
        """
        Validate configuration file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Tuple of (is_valid, report)
        """
        try:
            from .settings import load_config
            
            # Load and validate config
            settings = load_config(config_path, merge_env=False, create_if_missing=False)
            config_dict = settings.to_dict()
            
            is_valid, errors, warnings = self.validate_config(config_dict)
            
            report = self.get_validation_report()
            
            if is_valid:
                report += f"\n✅ Configuration file '{config_path}' is valid."
            else:
                report += f"\n❌ Configuration file '{config_path}' has errors."
            
            return is_valid, report
            
        except Exception as e:
            error_report = f"❌ Failed to validate '{config_path}': {e}"
            return False, error_report