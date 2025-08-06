"""
Configuration management for QECC-aware QML library.
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict, field
import logging

from ..utils.validation import ValidationError
from ..utils.security import validate_file_path, sanitize_input
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class QuantumSettings:
    """Quantum-specific settings."""
    default_shots: int = 1024
    max_qubits: int = 50
    max_circuit_depth: int = 100
    default_backend: str = "aer_simulator"
    enable_optimization: bool = True
    optimization_level: int = 1


@dataclass
class NoiseSettings:
    """Noise model settings."""
    default_gate_error_rate: float = 0.001
    default_readout_error_rate: float = 0.01
    default_T1: float = 50e-6
    default_T2: float = 70e-6
    enable_thermal_noise: bool = True
    enable_crosstalk: bool = False


@dataclass
class TrainingSettings:
    """Training configuration settings."""
    default_learning_rate: float = 0.01
    default_epochs: int = 50
    default_batch_size: int = 32
    max_training_time: int = 3600  # seconds
    enable_early_stopping: bool = True
    patience: int = 10
    validation_split: float = 0.2


@dataclass
class ErrorCorrectionSettings:
    """Error correction settings."""
    default_scheme: str = "surface_code"
    default_distance: int = 3
    syndrome_extraction_frequency: int = 1
    enable_adaptive_correction: bool = False
    decoder_timeout: float = 1.0


@dataclass
class LoggingSettings:
    """Logging configuration settings."""
    level: str = "INFO"
    enable_file_logging: bool = True
    log_directory: str = "logs"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    enable_structured_logging: bool = False
    enable_performance_logging: bool = True


@dataclass
class SecuritySettings:
    """Security configuration settings."""
    enable_input_sanitization: bool = True
    max_input_size: int = 1000000  # 1MB
    allowed_file_extensions: list = field(default_factory=lambda: ['.pkl', '.json', '.yaml', '.csv'])
    enable_path_validation: bool = True
    restrict_to_base_directory: bool = True
    base_directory: str = "."


@dataclass
class PerformanceSettings:
    """Performance optimization settings."""
    enable_parallel_execution: bool = True
    max_workers: int = 4
    enable_gpu: bool = True
    memory_limit_gb: Optional[float] = None
    cache_size: int = 1000
    enable_circuit_caching: bool = True


@dataclass
class Settings:
    """Main configuration settings for QECC-aware QML."""
    
    quantum: QuantumSettings = field(default_factory=QuantumSettings)
    noise: NoiseSettings = field(default_factory=NoiseSettings)
    training: TrainingSettings = field(default_factory=TrainingSettings)
    error_correction: ErrorCorrectionSettings = field(default_factory=ErrorCorrectionSettings)
    logging: LoggingSettings = field(default_factory=LoggingSettings)
    security: SecuritySettings = field(default_factory=SecuritySettings)
    performance: PerformanceSettings = field(default_factory=PerformanceSettings)
    
    # Global settings
    debug_mode: bool = False
    version: str = "0.1.0"
    
    def __post_init__(self):
        """Validate settings after initialization."""
        self._validate_settings()
    
    def _validate_settings(self):
        """Validate all settings for consistency and safety."""
        logger.debug("Validating configuration settings")
        
        # Quantum settings validation
        if self.quantum.default_shots < 1:
            raise ValidationError("default_shots must be >= 1")
        if self.quantum.max_qubits < 1:
            raise ValidationError("max_qubits must be >= 1")
        if self.quantum.max_circuit_depth < 1:
            raise ValidationError("max_circuit_depth must be >= 1")
        
        # Noise settings validation
        if not 0 <= self.noise.default_gate_error_rate <= 1:
            raise ValidationError("default_gate_error_rate must be in [0,1]")
        if not 0 <= self.noise.default_readout_error_rate <= 1:
            raise ValidationError("default_readout_error_rate must be in [0,1]")
        if self.noise.default_T1 <= 0:
            raise ValidationError("default_T1 must be positive")
        if self.noise.default_T2 <= 0:
            raise ValidationError("default_T2 must be positive")
        
        # Training settings validation
        if self.training.default_learning_rate <= 0:
            raise ValidationError("default_learning_rate must be positive")
        if self.training.default_epochs < 1:
            raise ValidationError("default_epochs must be >= 1")
        if self.training.default_batch_size < 1:
            raise ValidationError("default_batch_size must be >= 1")
        if not 0 <= self.training.validation_split <= 1:
            raise ValidationError("validation_split must be in [0,1]")
        
        # Error correction validation
        if self.error_correction.default_distance < 3:
            raise ValidationError("default_distance must be >= 3")
        if self.error_correction.default_distance % 2 == 0:
            raise ValidationError("default_distance must be odd")
        
        # Performance settings validation
        if self.performance.max_workers < 1:
            raise ValidationError("max_workers must be >= 1")
        if (self.performance.memory_limit_gb is not None and 
            self.performance.memory_limit_gb <= 0):
            raise ValidationError("memory_limit_gb must be positive if specified")
        
        logger.debug("Configuration validation completed successfully")
    
    def update(self, **kwargs):
        """Update settings with new values."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                logger.warning(f"Unknown setting: {key}")
        
        # Re-validate after updates
        self._validate_settings()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary."""
        return asdict(self)
    
    def to_json(self, indent: int = 2) -> str:
        """Convert settings to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)
    
    def to_yaml(self) -> str:
        """Convert settings to YAML string."""
        return yaml.dump(self.to_dict(), default_flow_style=False)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Settings':
        """Create settings from dictionary."""
        # Create nested dataclass instances
        settings_data = {}
        
        for field_name, field_value in data.items():
            if field_name == 'quantum' and isinstance(field_value, dict):
                settings_data[field_name] = QuantumSettings(**field_value)
            elif field_name == 'noise' and isinstance(field_value, dict):
                settings_data[field_name] = NoiseSettings(**field_value)
            elif field_name == 'training' and isinstance(field_value, dict):
                settings_data[field_name] = TrainingSettings(**field_value)
            elif field_name == 'error_correction' and isinstance(field_value, dict):
                settings_data[field_name] = ErrorCorrectionSettings(**field_value)
            elif field_name == 'logging' and isinstance(field_value, dict):
                settings_data[field_name] = LoggingSettings(**field_value)
            elif field_name == 'security' and isinstance(field_value, dict):
                settings_data[field_name] = SecuritySettings(**field_value)
            elif field_name == 'performance' and isinstance(field_value, dict):
                settings_data[field_name] = PerformanceSettings(**field_value)
            else:
                settings_data[field_name] = field_value
        
        return cls(**settings_data)
    
    def merge_environment_variables(self):
        """Merge environment variables into settings."""
        env_mappings = {
            'QECC_QML_DEBUG': ('debug_mode', bool),
            'QECC_QML_SHOTS': ('quantum.default_shots', int),
            'QECC_QML_BACKEND': ('quantum.default_backend', str),
            'QECC_QML_LEARNING_RATE': ('training.default_learning_rate', float),
            'QECC_QML_LOG_LEVEL': ('logging.level', str),
            'QECC_QML_MAX_WORKERS': ('performance.max_workers', int),
        }
        
        for env_var, (setting_path, setting_type) in env_mappings.items():
            if env_var in os.environ:
                try:
                    value = os.environ[env_var]
                    
                    # Convert to appropriate type
                    if setting_type == bool:
                        value = value.lower() in ('true', '1', 'yes', 'on')
                    elif setting_type == int:
                        value = int(value)
                    elif setting_type == float:
                        value = float(value)
                    
                    # Set nested attribute
                    obj = self
                    parts = setting_path.split('.')
                    for part in parts[:-1]:
                        obj = getattr(obj, part)
                    setattr(obj, parts[-1], value)
                    
                    logger.info(f"Set {setting_path} = {value} from environment")
                    
                except (ValueError, AttributeError) as e:
                    logger.warning(f"Failed to set {setting_path} from {env_var}: {e}")
        
        # Re-validate after environment updates
        self._validate_settings()


def load_config(
    config_path: Optional[Union[str, Path]] = None,
    merge_env: bool = True,
    create_if_missing: bool = True
) -> Settings:
    """
    Load configuration from file.
    
    Args:
        config_path: Path to configuration file
        merge_env: Whether to merge environment variables
        create_if_missing: Whether to create default config if file missing
        
    Returns:
        Settings instance
    """
    # Default config paths to try
    default_paths = [
        'qecc_qml_config.yaml',
        'qecc_qml_config.json',
        os.path.expanduser('~/.qecc_qml/config.yaml'),
        '/etc/qecc_qml/config.yaml',
    ]
    
    if config_path:
        config_paths = [config_path]
    else:
        config_paths = default_paths
    
    # Try to load from each path
    for path in config_paths:
        try:
            config_file = validate_file_path(
                path, 
                allowed_extensions=['.yaml', '.yml', '.json'],
                allow_absolute=True
            )
            
            if config_file.exists():
                logger.info(f"Loading configuration from {config_file}")
                
                with open(config_file, 'r') as f:
                    if config_file.suffix.lower() in ['.yaml', '.yml']:
                        data = yaml.safe_load(f)
                    else:
                        data = json.load(f)
                
                settings = Settings.from_dict(data)
                
                if merge_env:
                    settings.merge_environment_variables()
                
                return settings
                
        except Exception as e:
            logger.debug(f"Could not load config from {path}: {e}")
            continue
    
    # No config file found
    if create_if_missing:
        logger.info("No configuration file found, using defaults")
        settings = Settings()
        
        if merge_env:
            settings.merge_environment_variables()
        
        # Save default config for future use
        try:
            default_config_path = Path('qecc_qml_config.yaml')
            save_config(settings, default_config_path)
            logger.info(f"Saved default configuration to {default_config_path}")
        except Exception as e:
            logger.warning(f"Could not save default configuration: {e}")
        
        return settings
    else:
        raise FileNotFoundError("No configuration file found and create_if_missing=False")


def save_config(settings: Settings, config_path: Union[str, Path]):
    """
    Save configuration to file.
    
    Args:
        settings: Settings instance to save
        config_path: Path to save configuration file
    """
    config_file = validate_file_path(
        config_path,
        allowed_extensions=['.yaml', '.yml', '.json'],
        allow_absolute=True
    )
    
    # Create parent directory if needed
    config_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Save in appropriate format
    with open(config_file, 'w') as f:
        if config_file.suffix.lower() in ['.yaml', '.yml']:
            yaml.dump(settings.to_dict(), f, default_flow_style=False, indent=2)
        else:
            json.dump(settings.to_dict(), f, indent=2, default=str)
    
    logger.info(f"Saved configuration to {config_file}")


def get_default_config_path() -> Path:
    """Get the default configuration file path for the current user."""
    config_dir = Path.home() / '.qecc_qml'
    config_dir.mkdir(exist_ok=True)
    return config_dir / 'config.yaml'


# Global settings instance
_global_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get the global settings instance."""
    global _global_settings
    
    if _global_settings is None:
        _global_settings = load_config()
    
    return _global_settings


def set_settings(settings: Settings):
    """Set the global settings instance."""
    global _global_settings
    _global_settings = settings


def reset_settings():
    """Reset settings to defaults."""
    global _global_settings
    _global_settings = None