"""
Health checks and system diagnostics for QECC-aware QML.
"""

import sys
import os
import psutil
import platform
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import subprocess
import importlib
import warnings
from collections import defaultdict

import numpy as np

from .logging_config import get_logger

logger = get_logger(__name__)


class HealthCheckResult:
    """Result of a health check."""
    
    def __init__(self, name: str, status: str, message: str, details: Optional[Dict] = None):
        self.name = name
        self.status = status  # 'pass', 'warning', 'fail'
        self.message = message
        self.details = details or {}
        self.timestamp = time.time()
    
    def __str__(self):
        status_symbols = {'pass': '✅', 'warning': '⚠️', 'fail': '❌'}
        symbol = status_symbols.get(self.status, '❓')
        return f"{symbol} {self.name}: {self.message}"
    
    def to_dict(self):
        return {
            'name': self.name,
            'status': self.status,
            'message': self.message,
            'details': self.details,
            'timestamp': self.timestamp
        }


class SystemDiagnostics:
    """
    Comprehensive system diagnostics for quantum ML workloads.
    """
    
    def __init__(self):
        self.results = []
        self.start_time = time.time()
    
    def run_all_checks(self) -> List[HealthCheckResult]:
        """Run all diagnostic checks."""
        logger.info("Running comprehensive system diagnostics")
        
        checks = [
            self.check_python_environment,
            self.check_quantum_libraries,
            self.check_system_resources,
            self.check_gpu_availability,
            self.check_network_connectivity,
            self.check_file_system,
            self.check_security_settings,
        ]
        
        for check in checks:
            try:
                result = check()
                self.results.extend(result if isinstance(result, list) else [result])
            except Exception as e:
                error_result = HealthCheckResult(
                    check.__name__,
                    'fail',
                    f"Check failed with error: {e}"
                )
                self.results.append(error_result)
                logger.error(f"Health check {check.__name__} failed: {e}")
        
        self._generate_summary()
        return self.results
    
    def check_python_environment(self) -> List[HealthCheckResult]:
        """Check Python environment and dependencies."""
        results = []
        
        # Python version
        py_version = sys.version_info
        if py_version >= (3, 9):
            results.append(HealthCheckResult(
                'python_version',
                'pass',
                f"Python {py_version.major}.{py_version.minor}.{py_version.micro}",
                {'version': str(py_version)}
            ))
        elif py_version >= (3, 8):
            results.append(HealthCheckResult(
                'python_version',
                'warning',
                f"Python {py_version.major}.{py_version.minor} is supported but 3.9+ recommended"
            ))
        else:
            results.append(HealthCheckResult(
                'python_version',
                'fail',
                f"Python {py_version.major}.{py_version.minor} is too old (3.8+ required)"
            ))
        
        # Check virtual environment
        in_venv = hasattr(sys, 'real_prefix') or (
            hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
        )
        
        if in_venv:
            results.append(HealthCheckResult(
                'virtual_environment',
                'pass',
                "Running in virtual environment",
                {'venv_path': sys.prefix}
            ))
        else:
            results.append(HealthCheckResult(
                'virtual_environment',
                'warning',
                "Not running in virtual environment (recommended for isolation)"
            ))
        
        # Check memory limits
        try:
            import resource
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            if soft == resource.RLIM_INFINITY:
                results.append(HealthCheckResult(
                    'memory_limits',
                    'pass',
                    "No memory limits set"
                ))
            else:
                results.append(HealthCheckResult(
                    'memory_limits',
                    'warning',
                    f"Memory limit set: {soft // 1024 // 1024} MB",
                    {'soft_limit': soft, 'hard_limit': hard}
                ))
        except ImportError:
            results.append(HealthCheckResult(
                'memory_limits',
                'warning',
                "Cannot check memory limits (not Unix-like system)"
            ))
        
        return results
    
    def check_quantum_libraries(self) -> List[HealthCheckResult]:
        """Check quantum computing library installations."""
        results = []
        
        # Required libraries
        required_libs = {
            'qiskit': '1.0.0',
            'qiskit_aer': '0.13.0',
            'numpy': '1.20.0',
            'scipy': '1.7.0',
            'matplotlib': '3.3.0',
        }
        
        # Optional libraries
        optional_libs = {
            'cirq': '1.0.0',
            'torch': '1.8.0',
            'jax': '0.3.0',
            'cupy': '10.0.0',
        }
        
        for lib_name, min_version in required_libs.items():
            try:
                lib = importlib.import_module(lib_name)
                version = getattr(lib, '__version__', 'unknown')
                
                if self._version_compare(version, min_version) >= 0:
                    results.append(HealthCheckResult(
                        f'library_{lib_name}',
                        'pass',
                        f"{lib_name} {version} installed",
                        {'version': version, 'required': min_version}
                    ))
                else:
                    results.append(HealthCheckResult(
                        f'library_{lib_name}',
                        'warning',
                        f"{lib_name} {version} is below recommended {min_version}"
                    ))
                    
            except ImportError:
                results.append(HealthCheckResult(
                    f'library_{lib_name}',
                    'fail',
                    f"Required library {lib_name} not installed"
                ))
        
        for lib_name, min_version in optional_libs.items():
            try:
                lib = importlib.import_module(lib_name)
                version = getattr(lib, '__version__', 'unknown')
                results.append(HealthCheckResult(
                    f'optional_{lib_name}',
                    'pass',
                    f"Optional library {lib_name} {version} available"
                ))
            except ImportError:
                results.append(HealthCheckResult(
                    f'optional_{lib_name}',
                    'warning',
                    f"Optional library {lib_name} not installed"
                ))
        
        return results
    
    def check_system_resources(self) -> List[HealthCheckResult]:
        """Check system resource availability."""
        results = []
        
        # Memory check
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        
        if memory_gb >= 8:
            results.append(HealthCheckResult(
                'system_memory',
                'pass',
                f"{memory_gb:.1f} GB RAM available",
                {'total_gb': memory_gb, 'available_gb': memory.available / (1024**3)}
            ))
        elif memory_gb >= 4:
            results.append(HealthCheckResult(
                'system_memory',
                'warning',
                f"Only {memory_gb:.1f} GB RAM available (8+ GB recommended)"
            ))
        else:
            results.append(HealthCheckResult(
                'system_memory',
                'fail',
                f"Insufficient RAM: {memory_gb:.1f} GB (4+ GB minimum)"
            ))
        
        # CPU check
        cpu_count = psutil.cpu_count()
        cpu_freq = psutil.cpu_freq()
        
        if cpu_count >= 4:
            freq_info = f" @ {cpu_freq.current:.0f} MHz" if cpu_freq else ""
            results.append(HealthCheckResult(
                'cpu_cores',
                'pass',
                f"{cpu_count} CPU cores{freq_info}",
                {'cores': cpu_count, 'frequency': cpu_freq.current if cpu_freq else None}
            ))
        else:
            results.append(HealthCheckResult(
                'cpu_cores',
                'warning',
                f"Only {cpu_count} CPU cores (4+ recommended for quantum simulation)"
            ))
        
        # Disk space check
        disk = psutil.disk_usage('/')
        disk_free_gb = disk.free / (1024**3)
        
        if disk_free_gb >= 10:
            results.append(HealthCheckResult(
                'disk_space',
                'pass',
                f"{disk_free_gb:.1f} GB free disk space"
            ))
        elif disk_free_gb >= 2:
            results.append(HealthCheckResult(
                'disk_space',
                'warning',
                f"Low disk space: {disk_free_gb:.1f} GB free"
            ))
        else:
            results.append(HealthCheckResult(
                'disk_space',
                'fail',
                f"Very low disk space: {disk_free_gb:.1f} GB free"
            ))
        
        return results
    
    def check_gpu_availability(self) -> List[HealthCheckResult]:
        """Check GPU availability for accelerated computing."""
        results = []
        
        # Check CUDA availability
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0)
                memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                
                results.append(HealthCheckResult(
                    'cuda_gpu',
                    'pass',
                    f"CUDA GPU available: {gpu_name} ({memory_gb:.1f} GB)",
                    {'count': gpu_count, 'name': gpu_name, 'memory_gb': memory_gb}
                ))
            else:
                results.append(HealthCheckResult(
                    'cuda_gpu',
                    'warning',
                    "CUDA not available (GPU acceleration disabled)"
                ))
        except ImportError:
            results.append(HealthCheckResult(
                'cuda_gpu',
                'warning',
                "PyTorch not available for GPU check"
            ))
        
        # Check for other GPU libraries
        try:
            import cupy
            results.append(HealthCheckResult(
                'cupy',
                'pass',
                f"CuPy available for GPU arrays"
            ))
        except ImportError:
            results.append(HealthCheckResult(
                'cupy',
                'warning',
                "CuPy not available"
            ))
        
        return results
    
    def check_network_connectivity(self) -> List[HealthCheckResult]:
        """Check network connectivity for quantum backends."""
        results = []
        
        # Check internet connectivity
        try:
            import urllib.request
            urllib.request.urlopen('https://www.google.com', timeout=5)
            results.append(HealthCheckResult(
                'internet_connectivity',
                'pass',
                "Internet connectivity available"
            ))
        except Exception as e:
            results.append(HealthCheckResult(
                'internet_connectivity',
                'warning',
                f"Internet connectivity issue: {e}"
            ))
        
        # Check quantum cloud service endpoints
        endpoints = {
            'ibm_quantum': 'https://quantum-computing.ibm.com',
            'google_quantum': 'https://quantumai.google',
            'aws_braket': 'https://braket.aws.amazon.com',
        }
        
        for service, url in endpoints.items():
            try:
                import urllib.request
                urllib.request.urlopen(url, timeout=10)
                results.append(HealthCheckResult(
                    f'endpoint_{service}',
                    'pass',
                    f"{service} endpoint reachable"
                ))
            except Exception as e:
                results.append(HealthCheckResult(
                    f'endpoint_{service}',
                    'warning',
                    f"{service} endpoint unreachable: {e}"
                ))
        
        return results
    
    def check_file_system(self) -> List[HealthCheckResult]:
        """Check file system permissions and structure."""
        results = []
        
        # Check write permissions in current directory
        try:
            test_file = Path('_qecc_test_write.tmp')
            test_file.write_text('test')
            test_file.unlink()
            
            results.append(HealthCheckResult(
                'filesystem_write',
                'pass',
                "Write permissions in current directory"
            ))
        except Exception as e:
            results.append(HealthCheckResult(
                'filesystem_write',
                'fail',
                f"Cannot write to current directory: {e}"
            ))
        
        # Check for logs directory
        logs_dir = Path('logs')
        if logs_dir.exists() and logs_dir.is_dir():
            results.append(HealthCheckResult(
                'logs_directory',
                'pass',
                "Logs directory exists"
            ))
        else:
            try:
                logs_dir.mkdir(exist_ok=True)
                results.append(HealthCheckResult(
                    'logs_directory',
                    'pass',
                    "Created logs directory"
                ))
            except Exception as e:
                results.append(HealthCheckResult(
                    'logs_directory',
                    'warning',
                    f"Cannot create logs directory: {e}"
                ))
        
        return results
    
    def check_security_settings(self) -> List[HealthCheckResult]:
        """Check security-related settings."""
        results = []
        
        # Check Python bytecode optimization
        if sys.flags.optimize:
            results.append(HealthCheckResult(
                'python_optimization',
                'warning',
                "Python optimization enabled (may affect debugging)"
            ))
        else:
            results.append(HealthCheckResult(
                'python_optimization',
                'pass',
                "Python optimization disabled (good for development)"
            ))
        
        # Check for development vs production indicators
        debug_indicators = [
            'DEBUG' in os.environ,
            'DEVELOPMENT' in os.environ,
            any('dev' in arg.lower() for arg in sys.argv),
        ]
        
        if any(debug_indicators):
            results.append(HealthCheckResult(
                'development_mode',
                'warning',
                "Development mode detected (ensure secure settings for production)"
            ))
        else:
            results.append(HealthCheckResult(
                'development_mode',
                'pass',
                "No development mode indicators found"
            ))
        
        return results
    
    def _version_compare(self, version1: str, version2: str) -> int:
        """Compare two version strings. Returns -1, 0, or 1."""
        try:
            def parse_version(v):
                return tuple(map(int, v.split('.')[:3]))  # Take first 3 components
            
            v1 = parse_version(version1)
            v2 = parse_version(version2)
            
            return (v1 > v2) - (v1 < v2)
        except:
            return 0  # Can't compare, assume equal
    
    def _generate_summary(self):
        """Generate summary of all health checks."""
        status_counts = defaultdict(int)
        for result in self.results:
            status_counts[result.status] += 1
        
        total_time = time.time() - self.start_time
        
        summary = HealthCheckResult(
            'summary',
            'pass' if status_counts['fail'] == 0 else 'fail',
            f"Completed {len(self.results)} checks in {total_time:.1f}s: " +
            f"{status_counts['pass']} pass, {status_counts['warning']} warning, {status_counts['fail']} fail",
            {
                'total_checks': len(self.results),
                'pass_count': status_counts['pass'],
                'warning_count': status_counts['warning'],
                'fail_count': status_counts['fail'],
                'execution_time': total_time
            }
        )
        
        self.results.insert(0, summary)  # Put summary first
    
    def get_failed_checks(self) -> List[HealthCheckResult]:
        """Get list of failed checks."""
        return [r for r in self.results if r.status == 'fail']
    
    def get_warnings(self) -> List[HealthCheckResult]:
        """Get list of warnings."""
        return [r for r in self.results if r.status == 'warning']
    
    def print_report(self, show_details: bool = False):
        """Print human-readable diagnostic report."""
        print("\n" + "="*60)
        print("🔍 QECC-Aware QML System Diagnostics Report")
        print("="*60)
        
        for result in self.results:
            print(f"\n{result}")
            if show_details and result.details:
                for key, value in result.details.items():
                    print(f"   {key}: {value}")
        
        # Print recommendations
        failed = self.get_failed_checks()
        warnings_list = self.get_warnings()
        
        if failed:
            print(f"\n❌ Critical Issues ({len(failed)}):")
            for result in failed:
                print(f"   • {result.name}: {result.message}")
        
        if warnings_list:
            print(f"\n⚠️  Warnings ({len(warnings_list)}):")
            for result in warnings_list:
                print(f"   • {result.name}: {result.message}")
        
        if not failed and not warnings_list:
            print(f"\n🎉 All checks passed! System is ready for QECC-aware QML.")
        
        print("\n" + "="*60)


class HealthChecker:
    """
    Lightweight health checker for runtime monitoring.
    """
    
    def __init__(self, check_interval: float = 60.0):
        self.check_interval = check_interval
        self.last_check = 0.0
        self.cached_results = {}
    
    def quick_health_check(self) -> Dict[str, Any]:
        """Perform quick health check for runtime monitoring."""
        now = time.time()
        
        if now - self.last_check < self.check_interval:
            return self.cached_results
        
        health = {}
        
        # Memory usage
        memory = psutil.virtual_memory()
        health['memory_usage_percent'] = memory.percent
        health['memory_available_gb'] = memory.available / (1024**3)
        
        # CPU usage
        health['cpu_usage_percent'] = psutil.cpu_percent(interval=0.1)
        
        # Disk usage
        disk = psutil.disk_usage('/')
        health['disk_usage_percent'] = (disk.used / disk.total) * 100
        
        # Process info
        process = psutil.Process()
        health['process_memory_mb'] = process.memory_info().rss / (1024**2)
        health['process_cpu_percent'] = process.cpu_percent()
        
        # Overall health status
        health['status'] = 'healthy'
        
        if health['memory_usage_percent'] > 90:
            health['status'] = 'warning'
            health['issues'] = health.get('issues', []) + ['High memory usage']
        
        if health['cpu_usage_percent'] > 95:
            health['status'] = 'warning'
            health['issues'] = health.get('issues', []) + ['High CPU usage']
        
        if health['disk_usage_percent'] > 95:
            health['status'] = 'critical'
            health['issues'] = health.get('issues', []) + ['Disk space critical']
        
        health['timestamp'] = now
        self.cached_results = health
        self.last_check = now
        
        return health
    
    def is_healthy(self) -> bool:
        """Check if system is in healthy state."""
        health = self.quick_health_check()
        return health['status'] == 'healthy'
    
    def get_resource_usage(self) -> Dict[str, float]:
        """Get current resource usage metrics."""
        health = self.quick_health_check()
        return {
            'memory_percent': health['memory_usage_percent'],
            'cpu_percent': health['cpu_usage_percent'],
            'disk_percent': health['disk_usage_percent'],
            'process_memory_mb': health['process_memory_mb'],
        }