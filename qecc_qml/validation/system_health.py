"""
System health monitoring and diagnostics.

Provides comprehensive health checks, system monitoring,
and automated diagnostics for QECC-aware QML systems.
"""

import time
import psutil
import platform
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
from pathlib import Path
import subprocess


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Individual health check configuration."""
    name: str
    check_function: Callable[[], Dict[str, Any]]
    interval_seconds: float = 60.0
    timeout_seconds: float = 10.0
    enabled: bool = True
    critical: bool = False
    last_result: Optional[Dict[str, Any]] = None
    last_check_time: float = 0.0


@dataclass
class SystemMetrics:
    """System performance metrics."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_available_gb: float
    disk_usage_percent: float
    disk_free_gb: float
    network_sent_mb: float
    network_recv_mb: float
    load_average: List[float]
    process_count: int
    quantum_backend_status: Dict[str, str] = field(default_factory=dict)
    gpu_metrics: Optional[Dict[str, Any]] = None


@dataclass
class HealthReport:
    """Comprehensive health report."""
    timestamp: float
    overall_status: HealthStatus
    system_metrics: SystemMetrics
    check_results: Dict[str, Dict[str, Any]]
    warnings: List[str] = field(default_factory=list)
    critical_issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class HealthChecker:
    """
    Comprehensive health checking system.
    
    Monitors system resources, quantum backends, and application
    components with automated diagnostics and recommendations.
    """
    
    def __init__(
        self,
        check_interval: float = 60.0,
        auto_start: bool = True,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize health checker.
        
        Args:
            check_interval: Default interval between checks (seconds)
            auto_start: Whether to start monitoring automatically
            logger: Optional logger instance
        """
        self.check_interval = check_interval
        self.logger = logger or logging.getLogger(__name__)
        
        # Health checks registry
        self.health_checks: Dict[str, HealthCheck] = {}
        
        # Monitoring state
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # Health history
        self.health_history: List[HealthReport] = []
        self.max_history_size = 1000
        
        # Thresholds and configuration
        self.thresholds = {
            'cpu_warning': 80.0,
            'cpu_critical': 95.0,
            'memory_warning': 80.0,
            'memory_critical': 95.0,
            'disk_warning': 85.0,
            'disk_critical': 95.0,
            'network_warning': 1000.0,  # MB/s
            'network_critical': 2000.0,
        }
        
        # Initialize default health checks
        self._register_default_checks()
        
        if auto_start:
            self.start_monitoring()
    
    def _register_default_checks(self):
        """Register default system health checks."""
        
        # System resources check
        self.register_check(
            name="system_resources",
            check_function=self._check_system_resources,
            interval_seconds=30.0,
            critical=True
        )
        
        # Quantum backend availability
        self.register_check(
            name="quantum_backends",
            check_function=self._check_quantum_backends,
            interval_seconds=120.0,
            critical=False
        )
        
        # Python environment
        self.register_check(
            name="python_environment",
            check_function=self._check_python_environment,
            interval_seconds=300.0,
            critical=False
        )
        
        # Disk space
        self.register_check(
            name="disk_space",
            check_function=self._check_disk_space,
            interval_seconds=60.0,
            critical=True
        )
        
        # Network connectivity
        self.register_check(
            name="network_connectivity",
            check_function=self._check_network_connectivity,
            interval_seconds=120.0,
            critical=False
        )
        
        # GPU availability (if applicable)
        self.register_check(
            name="gpu_status",
            check_function=self._check_gpu_status,
            interval_seconds=60.0,
            critical=False
        )
    
    def register_check(
        self,
        name: str,
        check_function: Callable[[], Dict[str, Any]],
        interval_seconds: float = 60.0,
        timeout_seconds: float = 10.0,
        critical: bool = False,
        enabled: bool = True
    ):
        """Register a new health check."""
        self.health_checks[name] = HealthCheck(
            name=name,
            check_function=check_function,
            interval_seconds=interval_seconds,
            timeout_seconds=timeout_seconds,
            critical=critical,
            enabled=enabled
        )
        
        self.logger.info(f"Registered health check: {name}")
    
    def unregister_check(self, name: str) -> bool:
        """Unregister a health check."""
        if name in self.health_checks:
            del self.health_checks[name]
            self.logger.info(f"Unregistered health check: {name}")
            return True
        return False
    
    def enable_check(self, name: str) -> bool:
        """Enable a health check."""
        if name in self.health_checks:
            self.health_checks[name].enabled = True
            return True
        return False
    
    def disable_check(self, name: str) -> bool:
        """Disable a health check."""
        if name in self.health_checks:
            self.health_checks[name].enabled = False
            return True
        return False
    
    def start_monitoring(self):
        """Start continuous health monitoring."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        self.logger.info("Started health monitoring")
    
    def stop_monitoring(self):
        """Stop health monitoring."""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        
        self.logger.info("Stopped health monitoring")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                # Run health checks that are due
                current_time = time.time()
                
                for check_name, check in self.health_checks.items():
                    if not check.enabled:
                        continue
                    
                    # Check if it's time for this health check
                    time_since_last = current_time - check.last_check_time
                    if time_since_last >= check.interval_seconds:
                        self._run_single_check(check)
                
                # Generate and store health report
                report = self.generate_health_report()
                self._store_health_report(report)
                
                # Log any critical issues
                if report.critical_issues:
                    for issue in report.critical_issues:
                        self.logger.critical(f"Health critical: {issue}")
                
                # Sleep until next check cycle
                time.sleep(min(30.0, self.check_interval))
                
            except Exception as e:
                self.logger.error(f"Error in health monitoring loop: {e}")
                time.sleep(10.0)
    
    def _run_single_check(self, check: HealthCheck):
        """Run a single health check."""
        try:
            start_time = time.time()
            
            # Run check with timeout
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError(f"Health check '{check.name}' timed out")
            
            # Set timeout (Unix-like systems only)
            if hasattr(signal, 'SIGALRM'):
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(int(check.timeout_seconds))
            
            try:
                result = check.check_function()
                check.last_result = result
                check.last_check_time = time.time()
                
                execution_time = time.time() - start_time
                
                if execution_time > check.timeout_seconds * 0.8:
                    self.logger.warning(
                        f"Health check '{check.name}' took {execution_time:.2f}s "
                        f"(close to timeout {check.timeout_seconds}s)"
                    )
                    
            finally:
                if hasattr(signal, 'SIGALRM'):
                    signal.alarm(0)
            
        except TimeoutError as e:
            self.logger.error(str(e))
            check.last_result = {
                'status': HealthStatus.CRITICAL.value,
                'message': f"Check timed out after {check.timeout_seconds}s",
                'timestamp': time.time()
            }
            check.last_check_time = time.time()
            
        except Exception as e:
            self.logger.error(f"Health check '{check.name}' failed: {e}")
            check.last_result = {
                'status': HealthStatus.CRITICAL.value,
                'message': f"Check failed: {str(e)}",
                'timestamp': time.time()
            }
            check.last_check_time = time.time()
    
    def run_check(self, check_name: str) -> Dict[str, Any]:
        """Run a specific health check immediately."""
        if check_name not in self.health_checks:
            raise ValueError(f"Unknown health check: {check_name}")
        
        check = self.health_checks[check_name]
        self._run_single_check(check)
        
        return check.last_result or {}
    
    def run_all_checks(self) -> Dict[str, Dict[str, Any]]:
        """Run all enabled health checks immediately."""
        results = {}
        
        for check_name, check in self.health_checks.items():
            if check.enabled:
                self._run_single_check(check)
                results[check_name] = check.last_result or {}
        
        return results
    
    def generate_health_report(self) -> HealthReport:
        """Generate comprehensive health report."""
        current_time = time.time()
        
        # Collect system metrics
        system_metrics = self._collect_system_metrics()
        
        # Collect check results
        check_results = {}
        warnings = []
        critical_issues = []
        
        overall_status = HealthStatus.HEALTHY
        
        for check_name, check in self.health_checks.items():
            if not check.enabled or check.last_result is None:
                continue
            
            check_results[check_name] = check.last_result
            
            result_status = check.last_result.get('status', HealthStatus.UNKNOWN.value)
            
            if result_status == HealthStatus.CRITICAL.value:
                critical_issues.append(f"{check_name}: {check.last_result.get('message', 'Unknown issue')}")
                overall_status = HealthStatus.CRITICAL
                
            elif result_status == HealthStatus.WARNING.value:
                warnings.append(f"{check_name}: {check.last_result.get('message', 'Unknown warning')}")
                if overall_status == HealthStatus.HEALTHY:
                    overall_status = HealthStatus.WARNING
        
        # Generate recommendations
        recommendations = self._generate_recommendations(system_metrics, check_results)
        
        return HealthReport(
            timestamp=current_time,
            overall_status=overall_status,
            system_metrics=system_metrics,
            check_results=check_results,
            warnings=warnings,
            critical_issues=critical_issues,
            recommendations=recommendations
        )
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        try:
            # CPU and memory
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            # Disk usage (for root filesystem)
            disk = psutil.disk_usage('/')
            
            # Network I/O
            network = psutil.net_io_counters()
            network_sent_mb = network.bytes_sent / (1024 * 1024)
            network_recv_mb = network.bytes_recv / (1024 * 1024)
            
            # System load
            try:
                load_avg = list(psutil.getloadavg())
            except AttributeError:
                # Windows doesn't have getloadavg
                load_avg = [0.0, 0.0, 0.0]
            
            # Process count
            process_count = len(psutil.pids())
            
            # GPU metrics (if available)
            gpu_metrics = self._get_gpu_metrics()
            
            return SystemMetrics(
                timestamp=time.time(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_available_gb=memory.available / (1024**3),
                disk_usage_percent=disk.percent,
                disk_free_gb=disk.free / (1024**3),
                network_sent_mb=network_sent_mb,
                network_recv_mb=network_recv_mb,
                load_average=load_avg,
                process_count=process_count,
                gpu_metrics=gpu_metrics
            )
            
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
            
            # Return minimal metrics on error
            return SystemMetrics(
                timestamp=time.time(),
                cpu_percent=0.0,
                memory_percent=0.0,
                memory_available_gb=0.0,
                disk_usage_percent=0.0,
                disk_free_gb=0.0,
                network_sent_mb=0.0,
                network_recv_mb=0.0,
                load_average=[0.0, 0.0, 0.0],
                process_count=0
            )
    
    def _get_gpu_metrics(self) -> Optional[Dict[str, Any]]:
        """Get GPU metrics if available."""
        try:
            import nvidia_ml_py3 as nvml
            
            nvml.nvmlInit()
            device_count = nvml.nvmlDeviceGetCount()
            
            gpu_metrics = {
                'device_count': device_count,
                'devices': []
            }
            
            for i in range(device_count):
                handle = nvml.nvmlDeviceGetHandleByIndex(i)
                name = nvml.nvmlDeviceGetName(handle).decode()
                memory_info = nvml.nvmlDeviceGetMemoryInfo(handle)
                utilization = nvml.nvmlDeviceGetUtilizationRates(handle)
                temperature = nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)
                
                gpu_metrics['devices'].append({
                    'index': i,
                    'name': name,
                    'memory_total_gb': memory_info.total / (1024**3),
                    'memory_used_gb': memory_info.used / (1024**3),
                    'memory_percent': (memory_info.used / memory_info.total) * 100,
                    'gpu_utilization': utilization.gpu,
                    'memory_utilization': utilization.memory,
                    'temperature': temperature
                })
            
            return gpu_metrics
            
        except ImportError:
            return None
        except Exception as e:
            self.logger.debug(f"Could not get GPU metrics: {e}")
            return None
    
    def _generate_recommendations(
        self,
        system_metrics: SystemMetrics,
        check_results: Dict[str, Dict[str, Any]]
    ) -> List[str]:
        """Generate system recommendations based on metrics and check results."""
        recommendations = []
        
        # CPU recommendations
        if system_metrics.cpu_percent > self.thresholds['cpu_critical']:
            recommendations.append("Critical CPU usage detected. Consider reducing workload or scaling resources.")
        elif system_metrics.cpu_percent > self.thresholds['cpu_warning']:
            recommendations.append("High CPU usage. Monitor for sustained load.")
        
        # Memory recommendations
        if system_metrics.memory_percent > self.thresholds['memory_critical']:
            recommendations.append("Critical memory usage. Consider adding more RAM or optimizing memory usage.")
        elif system_metrics.memory_percent > self.thresholds['memory_warning']:
            recommendations.append("High memory usage. Monitor for memory leaks.")
        
        # Disk recommendations
        if system_metrics.disk_usage_percent > self.thresholds['disk_critical']:
            recommendations.append("Critical disk usage. Free up disk space immediately.")
        elif system_metrics.disk_usage_percent > self.thresholds['disk_warning']:
            recommendations.append("High disk usage. Consider cleanup or expanding storage.")
        
        # Process count recommendations
        if system_metrics.process_count > 500:
            recommendations.append("High process count. Check for resource leaks or runaway processes.")
        
        # GPU recommendations
        if system_metrics.gpu_metrics:
            for device in system_metrics.gpu_metrics.get('devices', []):
                if device['memory_percent'] > 90:
                    recommendations.append(f"GPU {device['index']} ({device['name']}) memory usage is high.")
                if device['temperature'] > 80:
                    recommendations.append(f"GPU {device['index']} temperature is high: {device['temperature']}°C")
        
        # Quantum backend recommendations
        quantum_check = check_results.get('quantum_backends', {})
        if quantum_check.get('status') == HealthStatus.WARNING.value:
            recommendations.append("Some quantum backends are unavailable. Consider using alternative backends.")
        
        return recommendations
    
    def _store_health_report(self, report: HealthReport):
        """Store health report in history."""
        self.health_history.append(report)
        
        # Keep history size bounded
        if len(self.health_history) > self.max_history_size:
            self.health_history = self.health_history[-self.max_history_size:]
    
    def get_latest_report(self) -> Optional[HealthReport]:
        """Get the most recent health report."""
        if self.health_history:
            return self.health_history[-1]
        return None
    
    def get_health_trend(self, hours: int = 24) -> Dict[str, Any]:
        """Get health trend over specified time period."""
        cutoff_time = time.time() - (hours * 3600)
        
        recent_reports = [
            report for report in self.health_history
            if report.timestamp >= cutoff_time
        ]
        
        if not recent_reports:
            return {'message': 'No recent health data available'}
        
        # Calculate trends
        cpu_values = [report.system_metrics.cpu_percent for report in recent_reports]
        memory_values = [report.system_metrics.memory_percent for report in recent_reports]
        
        return {
            'period_hours': hours,
            'report_count': len(recent_reports),
            'cpu_trend': {
                'average': sum(cpu_values) / len(cpu_values),
                'min': min(cpu_values),
                'max': max(cpu_values),
                'current': cpu_values[-1] if cpu_values else 0
            },
            'memory_trend': {
                'average': sum(memory_values) / len(memory_values),
                'min': min(memory_values),
                'max': max(memory_values),
                'current': memory_values[-1] if memory_values else 0
            },
            'status_distribution': {
                status.value: sum(1 for report in recent_reports if report.overall_status == status)
                for status in HealthStatus
            }
        }
    
    def export_health_report(self, filename: Optional[str] = None) -> str:
        """Export comprehensive health report."""
        from datetime import datetime
        
        if filename is None:
            filename = f"health_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        latest_report = self.get_latest_report()
        if not latest_report:
            raise RuntimeError("No health data available")
        
        report_data = {
            'generated_at': datetime.now().isoformat(),
            'system_info': {
                'platform': platform.platform(),
                'python_version': platform.python_version(),
                'architecture': platform.architecture(),
                'processor': platform.processor(),
                'hostname': platform.node()
            },
            'current_health': {
                'overall_status': latest_report.overall_status.value,
                'timestamp': latest_report.timestamp,
                'system_metrics': {
                    'cpu_percent': latest_report.system_metrics.cpu_percent,
                    'memory_percent': latest_report.system_metrics.memory_percent,
                    'memory_available_gb': latest_report.system_metrics.memory_available_gb,
                    'disk_usage_percent': latest_report.system_metrics.disk_usage_percent,
                    'disk_free_gb': latest_report.system_metrics.disk_free_gb,
                    'network_sent_mb': latest_report.system_metrics.network_sent_mb,
                    'network_recv_mb': latest_report.system_metrics.network_recv_mb,
                    'load_average': latest_report.system_metrics.load_average,
                    'process_count': latest_report.system_metrics.process_count,
                    'gpu_metrics': latest_report.system_metrics.gpu_metrics
                },
                'check_results': latest_report.check_results,
                'warnings': latest_report.warnings,
                'critical_issues': latest_report.critical_issues,
                'recommendations': latest_report.recommendations
            },
            'health_trend': self.get_health_trend(24),
            'configuration': {
                'thresholds': self.thresholds,
                'registered_checks': [
                    {
                        'name': check.name,
                        'enabled': check.enabled,
                        'critical': check.critical,
                        'interval_seconds': check.interval_seconds,
                        'timeout_seconds': check.timeout_seconds
                    }
                    for check in self.health_checks.values()
                ]
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        self.logger.info(f"Exported health report to {filename}")
        return filename
    
    # Default health check implementations
    
    def _check_system_resources(self) -> Dict[str, Any]:
        """Check system resources (CPU, memory, disk)."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            status = HealthStatus.HEALTHY
            messages = []
            
            # Check CPU
            if cpu_percent >= self.thresholds['cpu_critical']:
                status = HealthStatus.CRITICAL
                messages.append(f"Critical CPU usage: {cpu_percent:.1f}%")
            elif cpu_percent >= self.thresholds['cpu_warning']:
                status = HealthStatus.WARNING
                messages.append(f"High CPU usage: {cpu_percent:.1f}%")
            
            # Check memory
            if memory.percent >= self.thresholds['memory_critical']:
                status = HealthStatus.CRITICAL
                messages.append(f"Critical memory usage: {memory.percent:.1f}%")
            elif memory.percent >= self.thresholds['memory_warning']:
                if status == HealthStatus.HEALTHY:
                    status = HealthStatus.WARNING
                messages.append(f"High memory usage: {memory.percent:.1f}%")
            
            # Check disk
            if disk.percent >= self.thresholds['disk_critical']:
                status = HealthStatus.CRITICAL
                messages.append(f"Critical disk usage: {disk.percent:.1f}%")
            elif disk.percent >= self.thresholds['disk_warning']:
                if status == HealthStatus.HEALTHY:
                    status = HealthStatus.WARNING
                messages.append(f"High disk usage: {disk.percent:.1f}%")
            
            return {
                'status': status.value,
                'message': '; '.join(messages) if messages else 'System resources are normal',
                'details': {
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'memory_available_gb': memory.available / (1024**3),
                    'disk_percent': disk.percent,
                    'disk_free_gb': disk.free / (1024**3)
                },
                'timestamp': time.time()
            }
            
        except Exception as e:
            return {
                'status': HealthStatus.CRITICAL.value,
                'message': f'Failed to check system resources: {str(e)}',
                'timestamp': time.time()
            }
    
    def _check_quantum_backends(self) -> Dict[str, Any]:
        """Check quantum backend availability."""
        backend_status = {}
        overall_status = HealthStatus.HEALTHY
        messages = []
        
        # Check Qiskit
        try:
            import qiskit
            try:
                from qiskit_aer import AerSimulator
            except ImportError:
                from qecc_qml.core.fallback_imports import AerSimulator
            
            # Test simulator
            simulator = AerSimulator()
            backend_status['qiskit_aer'] = 'available'
            
            # Test IBM backends (if configured)
            try:
                from qiskit_ibm_runtime import QiskitRuntimeService
                service = QiskitRuntimeService()
                backend_status['ibm_quantum'] = 'available'
            except Exception:
                backend_status['ibm_quantum'] = 'unavailable'
            
        except ImportError:
            backend_status['qiskit'] = 'not_installed'
            overall_status = HealthStatus.WARNING
            messages.append('Qiskit not available')
        
        # Check Cirq
        try:
            import cirq
            backend_status['cirq'] = 'available'
        except ImportError:
            backend_status['cirq'] = 'not_installed'
        
        # Check Amazon Braket
        try:
            import braket
            backend_status['braket'] = 'available'
        except ImportError:
            backend_status['braket'] = 'not_installed'
        
        return {
            'status': overall_status.value,
            'message': '; '.join(messages) if messages else 'Quantum backends are available',
            'details': backend_status,
            'timestamp': time.time()
        }
    
    def _check_python_environment(self) -> Dict[str, Any]:
        """Check Python environment and packages."""
        try:
            import pkg_resources
            
            # Check critical packages
            critical_packages = [
                'numpy', 'scipy', 'matplotlib', 'pandas',
                'scikit-learn', 'torch'
            ]
            
            package_status = {}
            missing_packages = []
            
            for package_name in critical_packages:
                try:
                    package = pkg_resources.get_distribution(package_name)
                    package_status[package_name] = package.version
                except pkg_resources.DistributionNotFound:
                    package_status[package_name] = 'not_installed'
                    missing_packages.append(package_name)
            
            status = HealthStatus.WARNING if missing_packages else HealthStatus.HEALTHY
            message = f'Missing packages: {missing_packages}' if missing_packages else 'Python environment is healthy'
            
            return {
                'status': status.value,
                'message': message,
                'details': {
                    'python_version': platform.python_version(),
                    'packages': package_status
                },
                'timestamp': time.time()
            }
            
        except Exception as e:
            return {
                'status': HealthStatus.WARNING.value,
                'message': f'Could not check Python environment: {str(e)}',
                'timestamp': time.time()
            }
    
    def _check_disk_space(self) -> Dict[str, Any]:
        """Check disk space availability."""
        try:
            disk = psutil.disk_usage('/')
            
            status = HealthStatus.HEALTHY
            if disk.percent >= self.thresholds['disk_critical']:
                status = HealthStatus.CRITICAL
                message = f'Critical disk usage: {disk.percent:.1f}% ({disk.free / (1024**3):.1f} GB free)'
            elif disk.percent >= self.thresholds['disk_warning']:
                status = HealthStatus.WARNING
                message = f'High disk usage: {disk.percent:.1f}% ({disk.free / (1024**3):.1f} GB free)'
            else:
                message = f'Disk usage normal: {disk.percent:.1f}% ({disk.free / (1024**3):.1f} GB free)'
            
            return {
                'status': status.value,
                'message': message,
                'details': {
                    'total_gb': disk.total / (1024**3),
                    'used_gb': disk.used / (1024**3),
                    'free_gb': disk.free / (1024**3),
                    'percent_used': disk.percent
                },
                'timestamp': time.time()
            }
            
        except Exception as e:
            return {
                'status': HealthStatus.CRITICAL.value,
                'message': f'Failed to check disk space: {str(e)}',
                'timestamp': time.time()
            }
    
    def _check_network_connectivity(self) -> Dict[str, Any]:
        """Check network connectivity."""
        try:
            # Test basic connectivity
            test_hosts = ['8.8.8.8', 'google.com']
            connectivity_results = {}
            
            for host in test_hosts:
                try:
                    if platform.system().lower() == 'windows':
                        cmd = ['ping', '-n', '1', host]
                    else:
                        cmd = ['ping', '-c', '1', host]
                    
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        timeout=5
                    )
                    connectivity_results[host] = result.returncode == 0
                except subprocess.TimeoutExpired:
                    connectivity_results[host] = False
                except Exception:
                    connectivity_results[host] = False
            
            successful_tests = sum(connectivity_results.values())
            total_tests = len(connectivity_results)
            
            if successful_tests == 0:
                status = HealthStatus.CRITICAL
                message = 'No network connectivity detected'
            elif successful_tests < total_tests:
                status = HealthStatus.WARNING
                message = f'Partial network connectivity ({successful_tests}/{total_tests} tests passed)'
            else:
                status = HealthStatus.HEALTHY
                message = 'Network connectivity is normal'
            
            return {
                'status': status.value,
                'message': message,
                'details': {
                    'connectivity_tests': connectivity_results,
                    'success_rate': successful_tests / total_tests
                },
                'timestamp': time.time()
            }
            
        except Exception as e:
            return {
                'status': HealthStatus.WARNING.value,
                'message': f'Could not check network connectivity: {str(e)}',
                'timestamp': time.time()
            }
    
    def _check_gpu_status(self) -> Dict[str, Any]:
        """Check GPU status and availability."""
        gpu_metrics = self._get_gpu_metrics()
        
        if gpu_metrics is None:
            return {
                'status': HealthStatus.HEALTHY.value,
                'message': 'No GPU detected or monitoring not available',
                'timestamp': time.time()
            }
        
        try:
            status = HealthStatus.HEALTHY
            messages = []
            
            for device in gpu_metrics['devices']:
                if device['memory_percent'] > 90:
                    status = HealthStatus.WARNING
                    messages.append(f"GPU {device['index']} memory usage high: {device['memory_percent']:.1f}%")
                
                if device['temperature'] > 85:
                    status = HealthStatus.CRITICAL if device['temperature'] > 90 else HealthStatus.WARNING
                    messages.append(f"GPU {device['index']} temperature high: {device['temperature']}°C")
            
            message = '; '.join(messages) if messages else f"{gpu_metrics['device_count']} GPU(s) operating normally"
            
            return {
                'status': status.value,
                'message': message,
                'details': gpu_metrics,
                'timestamp': time.time()
            }
            
        except Exception as e:
            return {
                'status': HealthStatus.WARNING.value,
                'message': f'Error checking GPU status: {str(e)}',
                'timestamp': time.time()
            }


class SystemMonitor:
    """
    High-level system monitoring coordinator.
    
    Combines health checking with performance monitoring
    and automated alerting.
    """
    
    def __init__(
        self,
        health_checker: Optional[HealthChecker] = None,
        alert_callback: Optional[Callable[[HealthReport], None]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize system monitor.
        
        Args:
            health_checker: Health checker instance
            alert_callback: Function to call for alerts
            logger: Optional logger instance
        """
        self.health_checker = health_checker or HealthChecker()
        self.alert_callback = alert_callback
        self.logger = logger or logging.getLogger(__name__)
        
        # Alert state tracking
        self.last_alert_time = 0
        self.alert_cooldown = 300  # 5 minutes
        self.consecutive_critical_reports = 0
        self.alert_threshold = 3  # Send alert after 3 consecutive critical reports
    
    def start(self):
        """Start system monitoring."""
        self.health_checker.start_monitoring()
        self.logger.info("System monitor started")
    
    def stop(self):
        """Stop system monitoring."""
        self.health_checker.stop_monitoring()
        self.logger.info("System monitor stopped")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        latest_report = self.health_checker.get_latest_report()
        
        if not latest_report:
            return {'status': 'unknown', 'message': 'No health data available'}
        
        return {
            'overall_status': latest_report.overall_status.value,
            'timestamp': latest_report.timestamp,
            'warnings_count': len(latest_report.warnings),
            'critical_issues_count': len(latest_report.critical_issues),
            'system_metrics': {
                'cpu_percent': latest_report.system_metrics.cpu_percent,
                'memory_percent': latest_report.system_metrics.memory_percent,
                'disk_usage_percent': latest_report.system_metrics.disk_usage_percent,
            },
            'recommendations_count': len(latest_report.recommendations),
            'health_trend': self.health_checker.get_health_trend(1)  # Last hour
        }
    
    def check_alerts(self):
        """Check if alerts should be sent."""
        latest_report = self.health_checker.get_latest_report()
        
        if not latest_report or not self.alert_callback:
            return
        
        current_time = time.time()
        
        # Check for critical status
        if latest_report.overall_status == HealthStatus.CRITICAL:
            self.consecutive_critical_reports += 1
        else:
            self.consecutive_critical_reports = 0
        
        # Send alert if conditions are met
        should_alert = (
            self.consecutive_critical_reports >= self.alert_threshold and
            current_time - self.last_alert_time > self.alert_cooldown
        )
        
        if should_alert:
            try:
                self.alert_callback(latest_report)
                self.last_alert_time = current_time
                self.consecutive_critical_reports = 0
                self.logger.info("System alert sent")
            except Exception as e:
                self.logger.error(f"Failed to send alert: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary over time."""
        trend_data = self.health_checker.get_health_trend(24)
        latest_report = self.health_checker.get_latest_report()
        
        if not latest_report:
            return {'message': 'No performance data available'}
        
        return {
            'current_performance': {
                'cpu_percent': latest_report.system_metrics.cpu_percent,
                'memory_percent': latest_report.system_metrics.memory_percent,
                'disk_usage_percent': latest_report.system_metrics.disk_usage_percent,
                'process_count': latest_report.system_metrics.process_count
            },
            '24h_trends': trend_data,
            'health_checks': {
                name: check.last_result.get('status', 'unknown') if check.last_result else 'unknown'
                for name, check in self.health_checker.health_checks.items()
                if check.enabled
            }
        }