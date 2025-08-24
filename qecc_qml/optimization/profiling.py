"""
Performance profiling and optimization for quantum ML workloads.
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
import time
import psutil
import threading
import statistics
from typing import Dict, List, Optional, Any, Callable
from collections import defaultdict, deque
from dataclasses import dataclass, field
from contextlib import contextmanager
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
import gc

from ..utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ProfileMetric:
    """Individual performance metric."""
    name: str
    value: float
    unit: str
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProfileSession:
    """Performance profiling session."""
    session_id: str
    start_time: float
    end_time: Optional[float] = None
    metrics: List[ProfileMetric] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def duration(self) -> Optional[float]:
        """Get session duration."""
        if self.end_time is not None:
            return self.end_time - self.start_time
        return None
    
    def add_metric(self, name: str, value: float, unit: str = "", **metadata):
        """Add metric to session."""
        metric = ProfileMetric(
            name=name,
            value=value,
            unit=unit,
            timestamp=time.time(),
            metadata=metadata
        )
        self.metrics.append(metric)


class QuantumProfiler:
    """
    Comprehensive profiler for quantum machine learning operations.
    """
    
    def __init__(
        self,
        enable_memory_tracking: bool = True,
        enable_cpu_tracking: bool = True,
        enable_quantum_tracking: bool = True,
        sampling_interval: float = 0.1
    ):
        """
        Initialize quantum profiler.
        
        Args:
            enable_memory_tracking: Track memory usage
            enable_cpu_tracking: Track CPU usage
            enable_quantum_tracking: Track quantum-specific metrics
            sampling_interval: Sampling interval for continuous metrics
        """
        self.enable_memory_tracking = enable_memory_tracking
        self.enable_cpu_tracking = enable_cpu_tracking
        self.enable_quantum_tracking = enable_quantum_tracking
        self.sampling_interval = sampling_interval
        
        # Profiling state
        self.active_sessions = {}
        self.completed_sessions = []
        self.global_metrics = defaultdict(list)
        
        # Resource monitoring
        self.process = psutil.Process()
        self._monitoring = False
        self._monitor_thread = None
        self._resource_history = deque(maxlen=1000)
        
        # Quantum-specific tracking
        self.circuit_metrics = defaultdict(list)
        self.training_metrics = defaultdict(list)
        
        logger.info("Quantum profiler initialized")
    
    def start_session(self, session_id: str, **metadata) -> ProfileSession:
        """Start a new profiling session."""
        if session_id in self.active_sessions:
            logger.warning(f"Session {session_id} already active, ending previous session")
            self.end_session(session_id)
        
        session = ProfileSession(
            session_id=session_id,
            start_time=time.time(),
            metadata=metadata
        )
        
        self.active_sessions[session_id] = session
        
        # Start resource monitoring if not already running
        if not self._monitoring:
            self.start_monitoring()
        
        logger.debug(f"Started profiling session: {session_id}")
        return session
    
    def end_session(self, session_id: str) -> Optional[ProfileSession]:
        """End a profiling session."""
        if session_id not in self.active_sessions:
            logger.warning(f"Session {session_id} not found")
            return None
        
        session = self.active_sessions[session_id]
        session.end_time = time.time()
        
        # Add final resource metrics
        self._add_resource_metrics(session)
        
        # Move to completed sessions
        del self.active_sessions[session_id]
        self.completed_sessions.append(session)
        
        # Stop monitoring if no active sessions
        if not self.active_sessions and self._monitoring:
            self.stop_monitoring()
        
        logger.debug(f"Ended profiling session: {session_id} (duration: {session.duration():.2f}s)")
        return session
    
    def add_metric(self, session_id: str, name: str, value: float, unit: str = "", **metadata):
        """Add metric to specific session."""
        if session_id in self.active_sessions:
            self.active_sessions[session_id].add_metric(name, value, unit, **metadata)
        else:
            logger.warning(f"Session {session_id} not active, cannot add metric")
    
    def track_circuit_execution(
        self, 
        session_id: str, 
        circuit_hash: str,
        num_qubits: int,
        depth: int,
        shots: int,
        execution_time: float,
        backend_name: str = ""
    ):
        """Track quantum circuit execution metrics."""
        if not self.enable_quantum_tracking:
            return
        
        metrics = {
            'circuit_hash': circuit_hash,
            'num_qubits': num_qubits,
            'depth': depth,
            'shots': shots,
            'execution_time': execution_time,
            'backend_name': backend_name,
            'shots_per_second': shots / execution_time if execution_time > 0 else 0
        }
        
        # Add to session
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            session.add_metric("circuit_execution", execution_time, "seconds", **metrics)
        
        # Add to global circuit metrics
        self.circuit_metrics[circuit_hash].append(metrics)
        
        logger.debug(f"Tracked circuit execution: {num_qubits}q, {depth}d, {shots}s in {execution_time:.3f}s")
    
    def track_training_step(
        self,
        session_id: str,
        epoch: int,
        step: int,
        loss: float,
        accuracy: float,
        step_time: float,
        gradient_norm: Optional[float] = None,
        **kwargs
    ):
        """Track training step metrics."""
        metrics = {
            'epoch': epoch,
            'step': step,
            'loss': loss,
            'accuracy': accuracy,
            'step_time': step_time,
            'gradient_norm': gradient_norm,
            **kwargs
        }
        
        # Add to session
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            session.add_metric("training_step", step_time, "seconds", **metrics)
        
        # Add to global training metrics
        self.training_metrics[f"epoch_{epoch}"].append(metrics)
        
        logger.debug(f"Tracked training step {epoch}.{step}: loss={loss:.4f}, acc={accuracy:.4f}")
    
    def start_monitoring(self):
        """Start continuous resource monitoring."""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_resources, daemon=True)
        self._monitor_thread.start()
        
        logger.debug("Started resource monitoring")
    
    def stop_monitoring(self):
        """Stop continuous resource monitoring."""
        self._monitoring = False
        
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
        
        logger.debug("Stopped resource monitoring")
    
    def _monitor_resources(self):
        """Monitor system resources continuously."""
        while self._monitoring:
            try:
                timestamp = time.time()
                
                # Memory metrics
                if self.enable_memory_tracking:
                    memory_info = self.process.memory_info()
                    memory_percent = self.process.memory_percent()
                    
                    memory_metrics = {
                        'rss_mb': memory_info.rss / 1024 / 1024,
                        'vms_mb': memory_info.vms / 1024 / 1024,
                        'percent': memory_percent,
                        'timestamp': timestamp
                    }
                    
                    self._resource_history.append(('memory', memory_metrics))
                
                # CPU metrics
                if self.enable_cpu_tracking:
                    cpu_percent = self.process.cpu_percent()
                    cpu_times = self.process.cpu_times()
                    
                    cpu_metrics = {
                        'percent': cpu_percent,
                        'user_time': cpu_times.user,
                        'system_time': cpu_times.system,
                        'timestamp': timestamp
                    }
                    
                    self._resource_history.append(('cpu', cpu_metrics))
                
                time.sleep(self.sampling_interval)
                
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
                time.sleep(1.0)
    
    def _add_resource_metrics(self, session: ProfileSession):
        """Add resource metrics to session."""
        if not self._resource_history:
            return
        
        # Filter metrics to session timeframe
        session_metrics = [
            (metric_type, data) for metric_type, data in self._resource_history
            if session.start_time <= data['timestamp'] <= (session.end_time or time.time())
        ]
        
        if not session_metrics:
            return
        
        # Aggregate memory metrics
        memory_data = [data for metric_type, data in session_metrics if metric_type == 'memory']
        if memory_data:
            max_memory = max(m['rss_mb'] for m in memory_data)
            avg_memory = statistics.mean(m['rss_mb'] for m in memory_data)
            
            session.add_metric("max_memory", max_memory, "MB")
            session.add_metric("avg_memory", avg_memory, "MB")
        
        # Aggregate CPU metrics
        cpu_data = [data for metric_type, data in session_metrics if metric_type == 'cpu']
        if cpu_data:
            max_cpu = max(c['percent'] for c in cpu_data)
            avg_cpu = statistics.mean(c['percent'] for c in cpu_data)
            
            session.add_metric("max_cpu", max_cpu, "%")
            session.add_metric("avg_cpu", avg_cpu, "%")
    
    @contextmanager
    def profile(self, operation_name: str, **metadata):
        """Context manager for profiling operations."""
        session_id = f"{operation_name}_{int(time.time() * 1000000)}"
        
        session = self.start_session(session_id, operation=operation_name, **metadata)
        
        try:
            yield session
        finally:
            self.end_session(session_id)
    
    def get_session_report(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed report for a session."""
        # Look in active sessions first
        session = self.active_sessions.get(session_id)
        
        # Then in completed sessions
        if session is None:
            session = next((s for s in self.completed_sessions if s.session_id == session_id), None)
        
        if session is None:
            return None
        
        # Generate report
        report = {
            'session_id': session.session_id,
            'start_time': session.start_time,
            'end_time': session.end_time,
            'duration': session.duration(),
            'metadata': session.metadata,
            'metric_summary': self._summarize_metrics(session.metrics),
            'detailed_metrics': [
                {
                    'name': m.name,
                    'value': m.value,
                    'unit': m.unit,
                    'timestamp': m.timestamp,
                    'metadata': m.metadata
                }
                for m in session.metrics
            ]
        }
        
        return report
    
    def _summarize_metrics(self, metrics: List[ProfileMetric]) -> Dict[str, Any]:
        """Summarize metrics for a session."""
        summary = {}
        
        # Group metrics by name
        metric_groups = defaultdict(list)
        for metric in metrics:
            metric_groups[metric.name].append(metric.value)
        
        # Generate summary statistics
        for name, values in metric_groups.items():
            if len(values) == 1:
                summary[name] = {
                    'value': values[0],
                    'count': 1
                }
            else:
                summary[name] = {
                    'count': len(values),
                    'min': min(values),
                    'max': max(values),
                    'mean': statistics.mean(values),
                    'median': statistics.median(values),
                    'std': statistics.stdev(values) if len(values) > 1 else 0
                }
        
        return summary
    
    def get_global_stats(self) -> Dict[str, Any]:
        """Get global profiling statistics."""
        return {
            'active_sessions': len(self.active_sessions),
            'completed_sessions': len(self.completed_sessions),
            'monitoring_active': self._monitoring,
            'resource_history_size': len(self._resource_history),
            'circuit_metrics_count': sum(len(metrics) for metrics in self.circuit_metrics.values()),
            'training_metrics_count': sum(len(metrics) for metrics in self.training_metrics.values()),
        }
    
    def get_performance_insights(self) -> Dict[str, Any]:
        """Get actionable performance insights."""
        insights = {
            'recommendations': [],
            'bottlenecks': [],
            'optimization_opportunities': []
        }
        
        # Analyze circuit execution patterns
        if self.circuit_metrics:
            circuit_times = []
            for circuit_hash, executions in self.circuit_metrics.items():
                times = [e['execution_time'] for e in executions]
                if times:
                    avg_time = statistics.mean(times)
                    circuit_times.append((circuit_hash, avg_time, len(times)))
            
            # Find slow circuits
            if circuit_times:
                circuit_times.sort(key=lambda x: x[1], reverse=True)
                slowest_circuit = circuit_times[0]
                
                if slowest_circuit[1] > 1.0:  # > 1 second
                    insights['bottlenecks'].append({
                        'type': 'slow_circuit',
                        'circuit_hash': slowest_circuit[0],
                        'avg_time': slowest_circuit[1],
                        'executions': slowest_circuit[2]
                    })
                    
                    insights['recommendations'].append(
                        "Consider optimizing or caching slow quantum circuits"
                    )
        
        # Analyze memory usage
        if self._resource_history:
            memory_data = [data for metric_type, data in self._resource_history if metric_type == 'memory']
            if memory_data:
                max_memory = max(m['rss_mb'] for m in memory_data)
                
                if max_memory > 1000:  # > 1GB
                    insights['recommendations'].append(
                        "High memory usage detected, consider reducing batch sizes or enabling memory optimization"
                    )
        
        # Analyze training patterns
        if self.training_metrics:
            for epoch_key, steps in self.training_metrics.items():
                step_times = [s['step_time'] for s in steps]
                if step_times and statistics.mean(step_times) > 10.0:  # > 10 seconds per step
                    insights['bottlenecks'].append({
                        'type': 'slow_training',
                        'epoch': epoch_key,
                        'avg_step_time': statistics.mean(step_times)
                    })
                    
                    insights['recommendations'].append(
                        "Training steps are slow, consider parallel execution or smaller batch sizes"
                    )
        
        return insights
    
    def clear_history(self, keep_sessions: int = 10):
        """Clear old profiling data."""
        # Keep only recent completed sessions
        if len(self.completed_sessions) > keep_sessions:
            self.completed_sessions = self.completed_sessions[-keep_sessions:]
        
        # Clear old global metrics
        self.global_metrics.clear()
        
        # Clear resource history
        self._resource_history.clear()
        
        # Clear old circuit metrics (keep only recent)
        for circuit_hash in list(self.circuit_metrics.keys()):
            metrics = self.circuit_metrics[circuit_hash]
            if len(metrics) > 100:  # Keep last 100 executions
                self.circuit_metrics[circuit_hash] = metrics[-100:]
        
        # Clear old training metrics
        if len(self.training_metrics) > 20:  # Keep last 20 epochs
            epoch_keys = sorted(self.training_metrics.keys())
            for key in epoch_keys[:-20]:
                del self.training_metrics[key]
        
        # Force garbage collection
        gc.collect()
        
        logger.info("Profiling history cleared")


class PerformanceOptimizer:
    """
    Automatic performance optimizer based on profiling data.
    """
    
    def __init__(self, profiler: QuantumProfiler):
        """Initialize performance optimizer."""
        self.profiler = profiler
        self.optimization_rules = []
        self._setup_default_rules()
        
        logger.info("Performance optimizer initialized")
    
    def _setup_default_rules(self):
        """Setup default optimization rules."""
        
        def memory_optimization_rule(insights: Dict[str, Any]) -> List[str]:
            """Rule for memory optimization."""
            recommendations = []
            
            for bottleneck in insights.get('bottlenecks', []):
                if bottleneck.get('type') == 'high_memory':
                    recommendations.extend([
                        "Enable circuit caching to reduce memory usage",
                        "Reduce batch size for training",
                        "Use gradient checkpointing if available"
                    ])
            
            return recommendations
        
        def circuit_optimization_rule(insights: Dict[str, Any]) -> List[str]:
            """Rule for circuit optimization."""
            recommendations = []
            
            for bottleneck in insights.get('bottlenecks', []):
                if bottleneck.get('type') == 'slow_circuit':
                    recommendations.extend([
                        "Enable circuit compilation optimization",
                        "Consider circuit depth reduction",
                        "Use hardware-efficient ansatz"
                    ])
            
            return recommendations
        
        def parallel_execution_rule(insights: Dict[str, Any]) -> List[str]:
            """Rule for parallel execution optimization."""
            recommendations = []
            
            for bottleneck in insights.get('bottlenecks', []):
                if bottleneck.get('type') == 'slow_training':
                    recommendations.extend([
                        "Enable parallel batch processing",
                        "Increase number of worker threads",
                        "Use asynchronous execution where possible"
                    ])
            
            return recommendations
        
        # Add rules
        self.optimization_rules.extend([
            memory_optimization_rule,
            circuit_optimization_rule,
            parallel_execution_rule
        ])
    
    def add_optimization_rule(self, rule_func: Callable[[Dict[str, Any]], List[str]]):
        """Add custom optimization rule."""
        self.optimization_rules.append(rule_func)
    
    def analyze_and_optimize(self) -> Dict[str, Any]:
        """Analyze performance and provide optimization recommendations."""
        insights = self.profiler.get_performance_insights()
        
        # Apply optimization rules
        all_recommendations = set()
        
        for rule in self.optimization_rules:
            try:
                recommendations = rule(insights)
                all_recommendations.update(recommendations)
            except Exception as e:
                logger.error(f"Optimization rule failed: {e}")
        
        optimization_report = {
            'insights': insights,
            'recommendations': list(all_recommendations),
            'timestamp': time.time(),
            'profiler_stats': self.profiler.get_global_stats()
        }
        
        logger.info(f"Performance analysis complete: {len(all_recommendations)} recommendations")
        
        return optimization_report
    
    def auto_optimize_config(self, current_config: Dict[str, Any]) -> Dict[str, Any]:
        """Automatically optimize configuration based on profiling data."""
        optimized_config = current_config.copy()
        
        insights = self.profiler.get_performance_insights()
        
        # Apply automatic optimizations
        for bottleneck in insights.get('bottlenecks', []):
            if bottleneck.get('type') == 'slow_circuit':
                # Increase optimization level
                optimized_config['quantum'] = optimized_config.get('quantum', {})
                optimized_config['quantum']['optimization_level'] = 3
                
                # Enable circuit caching
                optimized_config['performance'] = optimized_config.get('performance', {})
                optimized_config['performance']['enable_circuit_caching'] = True
            
            elif bottleneck.get('type') == 'slow_training':
                # Increase parallel workers
                optimized_config['performance'] = optimized_config.get('performance', {})
                current_workers = optimized_config['performance'].get('max_workers', 4)
                optimized_config['performance']['max_workers'] = min(current_workers * 2, 16)
                
                # Enable batch processing
                optimized_config['performance']['enable_parallel_execution'] = True
        
        # Memory optimizations
        memory_data = [
            data for metric_type, data in self.profiler._resource_history 
            if metric_type == 'memory'
        ]
        
        if memory_data:
            max_memory = max(m['rss_mb'] for m in memory_data)
            
            if max_memory > 2000:  # > 2GB
                # Reduce cache sizes
                optimized_config['caching'] = optimized_config.get('caching', {})
                optimized_config['caching']['circuit_cache_size'] = 250  # Reduce from default 500
                optimized_config['caching']['result_cache_size'] = 1000  # Reduce from default 2000
        
        logger.info("Configuration auto-optimized based on profiling data")
        
        return optimized_config