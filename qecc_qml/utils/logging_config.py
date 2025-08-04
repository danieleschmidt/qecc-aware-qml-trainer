"""
Comprehensive logging configuration for QECC-aware QML.
"""

import logging
import logging.handlers
import sys
import os
from pathlib import Path
from typing import Dict, Optional, Union
import json
import time
from datetime import datetime


class QuantumMLFormatter(logging.Formatter):
    """Custom formatter for quantum ML logging with structured output."""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def __init__(self, use_colors: bool = True, structured: bool = False):
        self.use_colors = use_colors and sys.stderr.isatty()
        self.structured = structured
        
        if structured:
            super().__init__()
        else:
            fmt = '%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s'
            super().__init__(fmt, datefmt='%Y-%m-%d %H:%M:%S')
    
    def format(self, record):
        if self.structured:
            return self._format_structured(record)
        else:
            return self._format_colored(record)
    
    def _format_structured(self, record):
        """Format as structured JSON for machine parsing."""
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add extra fields if present
        if hasattr(record, 'quantum_context'):
            log_entry['quantum_context'] = record.quantum_context
        
        if hasattr(record, 'performance_metrics'):
            log_entry['performance_metrics'] = record.performance_metrics
        
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry, default=str)
    
    def _format_colored(self, record):
        """Format with colors for human reading."""
        formatted = super().format(record)
        
        if self.use_colors:
            color = self.COLORS.get(record.levelname, '')
            reset = self.COLORS['RESET']
            formatted = f"{color}{formatted}{reset}"
        
        return formatted


class PerformanceFilter(logging.Filter):
    """Filter to add performance context to log records."""
    
    def __init__(self):
        super().__init__()
        self.start_time = time.time()
    
    def filter(self, record):
        # Add timing information
        record.elapsed_time = time.time() - self.start_time
        
        # Add quantum-specific context if available
        if hasattr(record, 'qubits'):
            if not hasattr(record, 'quantum_context'):
                record.quantum_context = {}
            record.quantum_context['qubits'] = record.qubits
        
        return True


def setup_logging(
    level: Union[str, int] = logging.INFO,
    log_file: Optional[str] = None,
    log_dir: Optional[str] = None,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    structured_logging: bool = False,
    enable_performance_logging: bool = True,
    quantum_debug: bool = False
) -> logging.Logger:
    """
    Set up comprehensive logging for QECC-aware QML library.
    
    Args:
        level: Logging level
        log_file: Specific log file path
        log_dir: Log directory (creates timestamped file if log_file not specified)
        max_bytes: Maximum bytes per log file before rotation
        backup_count: Number of backup files to keep
        structured_logging: Whether to use structured JSON logging
        enable_performance_logging: Whether to add performance metrics
        quantum_debug: Whether to enable detailed quantum operation logging
        
    Returns:
        Configured root logger
    """
    # Convert string level to int if needed
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    
    # Create root logger
    root_logger = logging.getLogger('qecc_qml')
    root_logger.setLevel(level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = QuantumMLFormatter(
        use_colors=True, 
        structured=structured_logging
    )
    console_handler.setFormatter(console_formatter)
    
    if enable_performance_logging:
        console_handler.addFilter(PerformanceFilter())
    
    root_logger.addHandler(console_handler)
    
    # File handler
    if log_file or log_dir:
        if log_file:
            log_path = Path(log_file)
        else:
            log_dir_path = Path(log_dir or 'logs')
            log_dir_path.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_path = log_dir_path / f'qecc_qml_{timestamp}.log'
        
        # Ensure parent directory exists
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)  # Always debug level for files
        
        file_formatter = QuantumMLFormatter(
            use_colors=False,
            structured=structured_logging
        )
        file_handler.setFormatter(file_formatter)
        
        if enable_performance_logging:
            file_handler.addFilter(PerformanceFilter())
        
        root_logger.addHandler(file_handler)
    
    # Quantum debug handler (separate file for detailed quantum operations)
    if quantum_debug:
        if log_dir:
            log_dir_path = Path(log_dir)
        else:
            log_dir_path = Path('logs')
        
        log_dir_path.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        quantum_log_path = log_dir_path / f'quantum_debug_{timestamp}.log'
        
        quantum_handler = logging.handlers.RotatingFileHandler(
            quantum_log_path,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        quantum_handler.setLevel(logging.DEBUG)
        quantum_handler.addFilter(QuantumDebugFilter())
        quantum_handler.setFormatter(QuantumMLFormatter(use_colors=False))
        
        # Create quantum debug logger
        quantum_logger = logging.getLogger('qecc_qml.quantum')
        quantum_logger.addHandler(quantum_handler)
        quantum_logger.setLevel(logging.DEBUG)
    
    # Configure third-party loggers
    logging.getLogger('qiskit').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    
    root_logger.info("Logging system initialized")
    if log_file or log_dir:
        root_logger.info(f"Log files: {log_path}")
    
    return root_logger


class QuantumDebugFilter(logging.Filter):
    """Filter for quantum-specific debug information."""
    
    def filter(self, record):
        # Only pass quantum-related log records
        quantum_keywords = [
            'circuit', 'qubit', 'gate', 'measurement', 'fidelity',
            'syndrome', 'correction', 'noise', 'error'
        ]
        
        message = record.getMessage().lower()
        return any(keyword in message for keyword in quantum_keywords)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with quantum ML context.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(f'qecc_qml.{name}')


class QuantumMetricsLogger:
    """Specialized logger for quantum performance metrics."""
    
    def __init__(self, logger_name: str = 'qecc_qml.metrics'):
        self.logger = logging.getLogger(logger_name)
        self.metrics_buffer = []
        self.flush_interval = 10  # seconds
        self.last_flush = time.time()
    
    def log_quantum_metrics(
        self,
        operation: str,
        qubits: int,
        depth: int,
        fidelity: Optional[float] = None,
        error_rate: Optional[float] = None,
        execution_time: Optional[float] = None,
        **kwargs
    ):
        """
        Log quantum operation metrics.
        
        Args:
            operation: Type of quantum operation
            qubits: Number of qubits involved
            depth: Circuit depth
            fidelity: Circuit fidelity (if available)
            error_rate: Error rate (if available)
            execution_time: Execution time in seconds
            **kwargs: Additional metrics
        """
        metrics = {
            'operation': operation,
            'qubits': qubits,
            'depth': depth,
            'timestamp': time.time(),
        }
        
        if fidelity is not None:
            metrics['fidelity'] = fidelity
        if error_rate is not None:
            metrics['error_rate'] = error_rate
        if execution_time is not None:
            metrics['execution_time'] = execution_time
        
        metrics.update(kwargs)
        self.metrics_buffer.append(metrics)
        
        # Auto-flush if needed
        if time.time() - self.last_flush > self.flush_interval:
            self.flush()
    
    def log_training_metrics(
        self,
        epoch: int,
        loss: float,
        accuracy: float,
        fidelity: Optional[float] = None,
        logical_error_rate: Optional[float] = None,
        **kwargs
    ):
        """Log training metrics."""
        self.log_quantum_metrics(
            operation='training',
            qubits=kwargs.get('qubits', 0),
            depth=kwargs.get('depth', 0),
            epoch=epoch,
            loss=loss,
            accuracy=accuracy,
            fidelity=fidelity,
            logical_error_rate=logical_error_rate,
            **{k: v for k, v in kwargs.items() if k not in ['qubits', 'depth']}
        )
    
    def log_benchmark_metrics(
        self,
        benchmark_type: str,
        noise_level: float,
        performance_score: float,
        **kwargs
    ):
        """Log benchmark metrics."""
        self.log_quantum_metrics(
            operation='benchmark',
            qubits=kwargs.get('qubits', 0),
            depth=kwargs.get('depth', 0),
            benchmark_type=benchmark_type,
            noise_level=noise_level,
            performance_score=performance_score,
            **{k: v for k, v in kwargs.items() if k not in ['qubits', 'depth']}
        )
    
    def flush(self):
        """Flush buffered metrics to log."""
        if not self.metrics_buffer:
            return
        
        # Create summary log entry
        summary = {
            'total_operations': len(self.metrics_buffer),
            'time_window': time.time() - self.last_flush,
            'operations_by_type': {},
            'average_metrics': {},
        }
        
        # Aggregate metrics
        for metrics in self.metrics_buffer:
            op_type = metrics.get('operation', 'unknown')
            summary['operations_by_type'][op_type] = summary['operations_by_type'].get(op_type, 0) + 1
        
        # Calculate averages
        numeric_metrics = ['fidelity', 'error_rate', 'execution_time', 'qubits', 'depth']
        for metric in numeric_metrics:
            values = [m.get(metric) for m in self.metrics_buffer if m.get(metric) is not None]
            if values:
                summary['average_metrics'][metric] = sum(values) / len(values)
        
        # Log individual metrics and summary
        for metrics in self.metrics_buffer:
            extra = {'performance_metrics': metrics}
            self.logger.info(f"Quantum operation: {metrics['operation']}", extra=extra)
        
        extra = {'performance_metrics': summary}
        self.logger.info(f"Metrics summary: {summary['total_operations']} operations", extra=extra)
        
        # Clear buffer
        self.metrics_buffer.clear()
        self.last_flush = time.time()
    
    def __del__(self):
        """Ensure metrics are flushed on destruction."""
        try:
            self.flush()
        except:
            pass  # Ignore errors during cleanup


# Global metrics logger instance
_metrics_logger = None

def get_metrics_logger() -> QuantumMetricsLogger:
    """Get the global metrics logger instance."""
    global _metrics_logger
    if _metrics_logger is None:
        _metrics_logger = QuantumMetricsLogger()
    return _metrics_logger


def log_quantum_operation(operation: str, qubits: int, depth: int, **kwargs):
    """Convenience function to log quantum operations."""
    metrics_logger = get_metrics_logger()
    metrics_logger.log_quantum_metrics(operation, qubits, depth, **kwargs)