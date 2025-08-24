"""
Quantum Performance Enhancement Engine.

This module provides advanced performance optimization for quantum machine learning
operations including circuit optimization, caching, parallel execution, and
adaptive resource management.
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
import time
import threading
import multiprocessing as mp
import concurrent.futures
import hashlib
import pickle
import logging
from typing import Dict, List, Optional, Callable, Any, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
from functools import wraps, lru_cache
import warnings
from pathlib import Path
import json
import gc

try:
    from qiskit import QuantumCircuit, transpile
    from qiskit.quantum_info import Operator
    QISKIT_AVAILABLE = True
except ImportError:
    from ..core.fallback_imports import QuantumCircuit, Operator
    QISKIT_AVAILABLE = False

from ..utils.logging_config import get_logger

logger = get_logger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance optimization metrics."""
    circuit_optimization_time: float = 0.0
    execution_time: float = 0.0
    memory_usage_mb: float = 0.0
    cache_hit_rate: float = 0.0
    parallelization_efficiency: float = 0.0
    optimization_level: int = 0
    gate_reduction_percent: float = 0.0
    depth_reduction_percent: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'circuit_optimization_time': self.circuit_optimization_time,
            'execution_time': self.execution_time,
            'memory_usage_mb': self.memory_usage_mb,
            'cache_hit_rate': self.cache_hit_rate,
            'parallelization_efficiency': self.parallelization_efficiency,
            'optimization_level': self.optimization_level,
            'gate_reduction_percent': self.gate_reduction_percent,
            'depth_reduction_percent': self.depth_reduction_percent
        }

@dataclass
class OptimizationConfig:
    """Configuration for performance optimization."""
    enable_circuit_optimization: bool = True
    enable_caching: bool = True
    enable_parallel_execution: bool = True
    enable_adaptive_scaling: bool = True
    cache_size_mb: float = 256.0
    max_parallel_workers: int = None
    optimization_level: int = 2
    memory_limit_mb: float = 2048.0
    circuit_cache_ttl_hours: float = 24.0
    
    def __post_init__(self):
        if self.max_parallel_workers is None:
            self.max_parallel_workers = min(mp.cpu_count(), 8)

class CircuitCache:
    """Advanced caching system for quantum circuits."""
    
    def __init__(self, max_size_mb: float = 256.0, ttl_hours: float = 24.0):
        self.max_size_mb = max_size_mb
        self.ttl_seconds = ttl_hours * 3600
        self.cache: Dict[str, Tuple[Any, float, int]] = {}  # hash -> (data, timestamp, size_bytes)
        self.access_times: Dict[str, float] = {}
        self.current_size_bytes = 0
        self.lock = threading.RLock()
        
        # Statistics
        self.hits = 0
        self.misses = 0
        
    def _generate_key(self, circuit: QuantumCircuit, *args, **kwargs) -> str:
        """Generate cache key for circuit and parameters."""
        # Create a hashable representation
        key_data = {
            'num_qubits': getattr(circuit, 'num_qubits', 0),
            'circuit_str': str(circuit)[:1000],  # Limit size
            'args': str(args)[:500],
            'kwargs': str(sorted(kwargs.items()))[:500]
        }
        
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()
    
    def get(self, circuit: QuantumCircuit, *args, **kwargs) -> Optional[Any]:
        """Get cached result."""
        key = self._generate_key(circuit, *args, **kwargs)
        
        with self.lock:
            if key in self.cache:
                data, timestamp, size_bytes = self.cache[key]
                
                # Check TTL
                if time.time() - timestamp > self.ttl_seconds:
                    self._remove_key(key)
                    self.misses += 1
                    return None
                
                # Update access time
                self.access_times[key] = time.time()
                self.hits += 1
                return data
            
            self.misses += 1
            return None
    
    def put(self, circuit: QuantumCircuit, result: Any, *args, **kwargs) -> None:
        """Store result in cache."""
        key = self._generate_key(circuit, *args, **kwargs)
        
        try:
            # Estimate size
            result_bytes = len(pickle.dumps(result))
            
            with self.lock:
                # Check if we need to evict
                while (self.current_size_bytes + result_bytes) > (self.max_size_mb * 1024 * 1024):
                    if not self._evict_lru():
                        # Can't evict more, skip caching
                        return
                
                # Store
                self.cache[key] = (result, time.time(), result_bytes)
                self.access_times[key] = time.time()
                self.current_size_bytes += result_bytes
                
        except Exception as e:
            logger.warning(f"Cache storage error: {e}")
    
    def _remove_key(self, key: str) -> None:
        """Remove key from cache."""
        if key in self.cache:
            _, _, size_bytes = self.cache[key]
            del self.cache[key]
            self.current_size_bytes -= size_bytes
        
        if key in self.access_times:
            del self.access_times[key]
    
    def _evict_lru(self) -> bool:
        """Evict least recently used item."""
        if not self.access_times:
            return False
        
        # Find LRU key
        lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        self._remove_key(lru_key)
        return True
    
    def get_hit_rate(self) -> float:
        """Get cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    def clear(self) -> None:
        """Clear cache."""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
            self.current_size_bytes = 0

class AdaptiveResourceManager:
    """Adaptive resource management for optimal performance."""
    
    def __init__(self, max_memory_mb: float = 2048.0):
        self.max_memory_mb = max_memory_mb
        self.resource_history: deque = deque(maxlen=100)
        self.current_load = 0.0
        self.optimal_workers = mp.cpu_count()
        
    def get_optimal_workers(self, task_complexity: float = 1.0) -> int:
        """Get optimal number of workers based on current load."""
        base_workers = min(self.optimal_workers, mp.cpu_count())
        
        # Adjust based on task complexity and system load
        if task_complexity > 2.0:
            workers = max(1, int(base_workers * 0.8))  # Reduce for complex tasks
        elif task_complexity < 0.5:
            workers = min(base_workers * 2, mp.cpu_count())  # Increase for simple tasks
        else:
            workers = base_workers
        
        # Adjust based on memory constraints
        estimated_memory_per_worker = task_complexity * 100  # MB per worker
        max_workers_by_memory = int(self.max_memory_mb / estimated_memory_per_worker)
        
        return min(workers, max_workers_by_memory, mp.cpu_count())
    
    def update_performance_metrics(self, workers: int, execution_time: float, success: bool) -> None:
        """Update performance metrics for adaptive learning."""
        self.resource_history.append({
            'workers': workers,
            'execution_time': execution_time,
            'success': success,
            'timestamp': time.time()
        })
        
        # Update optimal workers based on performance
        if len(self.resource_history) >= 10:
            self._optimize_worker_count()
    
    def _optimize_worker_count(self) -> None:
        """Optimize worker count based on historical performance."""
        recent_data = list(self.resource_history)[-10:]
        
        # Group by worker count and calculate average performance
        worker_performance = defaultdict(list)
        for data in recent_data:
            if data['success']:
                worker_performance[data['workers']].append(data['execution_time'])
        
        if worker_performance:
            # Find optimal worker count (minimum average execution time)
            avg_times = {
                workers: np.mean(times) 
                for workers, times in worker_performance.items()
            }
            
            self.optimal_workers = min(avg_times.keys(), key=lambda w: avg_times[w])

class QuantumPerformanceEnhancer:
    """
    Advanced quantum performance enhancement engine.
    
    Provides comprehensive performance optimization including:
    - Circuit optimization and compilation
    - Intelligent caching with TTL
    - Parallel and distributed execution
    - Adaptive resource management
    - Memory optimization and garbage collection
    """
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        """Initialize performance enhancer."""
        self.config = config or OptimizationConfig()
        self.logger = get_logger(__name__)
        
        # Performance components
        self.cache = CircuitCache(
            max_size_mb=self.config.cache_size_mb,
            ttl_hours=self.config.circuit_cache_ttl_hours
        )
        self.resource_manager = AdaptiveResourceManager(
            max_memory_mb=self.config.memory_limit_mb
        )
        
        # Thread pools for parallel execution
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.config.max_parallel_workers
        )
        
        # Performance tracking
        self.performance_history: List[PerformanceMetrics] = []
        
        # Circuit optimization cache
        self.optimized_circuits: Dict[str, QuantumCircuit] = {}
        
        self.logger.info("Quantum performance enhancer initialized")
    
    def optimize_circuit(self, circuit: QuantumCircuit, optimization_level: int = None) -> QuantumCircuit:
        """
        Optimize quantum circuit for better performance.
        
        Args:
            circuit: Input quantum circuit
            optimization_level: Optimization level (0-3)
            
        Returns:
            Optimized quantum circuit
        """
        start_time = time.time()
        opt_level = optimization_level or self.config.optimization_level
        
        # Generate circuit hash for caching
        circuit_hash = self._hash_circuit(circuit)
        cache_key = f"{circuit_hash}_{opt_level}"
        
        # Check cache first
        if cache_key in self.optimized_circuits:
            self.logger.debug(f"Using cached optimized circuit: {cache_key}")
            return self.optimized_circuits[cache_key]
        
        try:
            optimized_circuit = self._perform_circuit_optimization(circuit, opt_level)
            
            # Cache the result
            self.optimized_circuits[cache_key] = optimized_circuit
            
            # Calculate optimization metrics
            original_depth = getattr(circuit, 'depth', lambda: 0)()
            optimized_depth = getattr(optimized_circuit, 'depth', lambda: 0)()
            
            depth_reduction = 0.0 if original_depth == 0 else ((original_depth - optimized_depth) / original_depth) * 100
            
            optimization_time = time.time() - start_time
            
            self.logger.info(f"Circuit optimized in {optimization_time:.3f}s, depth reduction: {depth_reduction:.1f}%")
            
            return optimized_circuit
            
        except Exception as e:
            self.logger.warning(f"Circuit optimization failed: {e}, using original circuit")
            return circuit
    
    def _perform_circuit_optimization(self, circuit: QuantumCircuit, optimization_level: int) -> QuantumCircuit:
        """Perform actual circuit optimization."""
        if not QISKIT_AVAILABLE:
            # For fallback circuits, return as-is
            return circuit
        
        try:
            # Use Qiskit transpiler for optimization
            if hasattr(circuit, 'data') and optimization_level > 0:
                optimized = transpile(
                    circuit,
                    optimization_level=min(optimization_level, 3),
                    seed_transpiler=42  # For reproducible results
                )
                return optimized
            else:
                return circuit
                
        except Exception as e:
            self.logger.warning(f"Qiskit optimization failed: {e}")
            return circuit
    
    def _hash_circuit(self, circuit: QuantumCircuit) -> str:
        """Generate hash for circuit identification."""
        circuit_data = {
            'num_qubits': getattr(circuit, 'num_qubits', 0),
            'circuit_str': str(circuit)[:500]  # Limit size for hashing
        }
        return hashlib.md5(str(circuit_data).encode()).hexdigest()
    
    def execute_with_caching(
        self,
        operation: Callable,
        circuit: QuantumCircuit,
        *args,
        cache_key_suffix: str = "",
        **kwargs
    ) -> Any:
        """Execute operation with intelligent caching."""
        if not self.config.enable_caching:
            return operation(circuit, *args, **kwargs)
        
        # Check cache
        cached_result = self.cache.get(circuit, *args, cache_key_suffix=cache_key_suffix, **kwargs)
        if cached_result is not None:
            return cached_result
        
        # Execute operation
        result = operation(circuit, *args, **kwargs)
        
        # Store in cache
        self.cache.put(circuit, result, *args, cache_key_suffix=cache_key_suffix, **kwargs)
        
        return result
    
    def execute_parallel_batch(
        self,
        operation: Callable,
        circuits: List[QuantumCircuit],
        *args,
        max_workers: int = None,
        **kwargs
    ) -> List[Any]:
        """Execute batch of operations in parallel."""
        if not self.config.enable_parallel_execution or len(circuits) == 1:
            # Sequential execution
            return [operation(circuit, *args, **kwargs) for circuit in circuits]
        
        # Determine optimal worker count
        task_complexity = self._estimate_task_complexity(circuits[0] if circuits else None)
        optimal_workers = self.resource_manager.get_optimal_workers(task_complexity)
        workers = min(max_workers or optimal_workers, len(circuits))
        
        start_time = time.time()
        
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                # Submit all tasks
                futures = [
                    executor.submit(operation, circuit, *args, **kwargs)
                    for circuit in circuits
                ]
                
                # Collect results
                results = []
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result(timeout=30.0)  # 30 second timeout per task
                        results.append(result)
                    except Exception as e:
                        self.logger.error(f"Parallel execution error: {e}")
                        results.append(None)
                
                execution_time = time.time() - start_time
                
                # Update resource manager
                self.resource_manager.update_performance_metrics(
                    workers=workers,
                    execution_time=execution_time,
                    success=all(r is not None for r in results)
                )
                
                # Calculate parallelization efficiency
                estimated_sequential_time = execution_time * workers
                efficiency = estimated_sequential_time / (execution_time * len(circuits)) if execution_time > 0 else 0.0
                
                self.logger.info(f"Parallel execution completed: {len(circuits)} circuits, {workers} workers, efficiency: {efficiency:.2f}")
                
                return results
                
        except Exception as e:
            self.logger.error(f"Parallel execution failed: {e}")
            # Fallback to sequential execution
            return [operation(circuit, *args, **kwargs) for circuit in circuits]
    
    def _estimate_task_complexity(self, circuit: QuantumCircuit) -> float:
        """Estimate computational complexity of a task."""
        if circuit is None:
            return 1.0
        
        num_qubits = getattr(circuit, 'num_qubits', 4)
        circuit_depth = getattr(circuit, 'depth', lambda: 10)()
        
        # Simple complexity estimation based on circuit properties
        qubit_factor = np.log2(num_qubits + 1)
        depth_factor = np.log10(circuit_depth + 1)
        
        complexity = qubit_factor * depth_factor
        return max(0.1, min(complexity, 10.0))  # Clamp between 0.1 and 10.0
    
    def optimize_memory_usage(self) -> None:
        """Optimize memory usage through garbage collection and cache management."""
        # Clear old cached circuits
        if len(self.optimized_circuits) > 100:
            # Keep only the 50 most recently used
            sorted_items = sorted(
                self.optimized_circuits.items(),
                key=lambda x: getattr(x[1], '_last_access', 0),
                reverse=True
            )
            self.optimized_circuits = dict(sorted_items[:50])
        
        # Trigger garbage collection
        collected = gc.collect()
        
        # Clear cache if memory pressure is high
        try:
            import psutil
            memory_percent = psutil.virtual_memory().percent
            
            if memory_percent > 90:
                self.cache.clear()
                self.logger.warning(f"High memory usage ({memory_percent:.1f}%), cleared cache")
                
        except ImportError:
            pass  # psutil not available
        
        self.logger.debug(f"Memory optimization completed, collected {collected} objects")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        cache_hit_rate = self.cache.get_hit_rate()
        
        recent_metrics = self.performance_history[-10:] if self.performance_history else []
        avg_execution_time = np.mean([m.execution_time for m in recent_metrics]) if recent_metrics else 0.0
        avg_optimization_time = np.mean([m.circuit_optimization_time for m in recent_metrics]) if recent_metrics else 0.0
        
        return {
            'cache_hit_rate': cache_hit_rate,
            'cache_size_mb': self.cache.current_size_bytes / (1024 * 1024),
            'cached_circuits': len(self.optimized_circuits),
            'average_execution_time': avg_execution_time,
            'average_optimization_time': avg_optimization_time,
            'optimal_workers': self.resource_manager.optimal_workers,
            'performance_samples': len(self.performance_history),
            'config': {
                'circuit_optimization': self.config.enable_circuit_optimization,
                'caching': self.config.enable_caching,
                'parallel_execution': self.config.enable_parallel_execution,
                'optimization_level': self.config.optimization_level
            }
        }
    
    def benchmark_performance(self, test_circuits: List[QuantumCircuit], iterations: int = 10) -> Dict[str, Any]:
        """Benchmark performance with and without optimizations."""
        results = {
            'optimized': {'times': [], 'success_rate': 0.0},
            'unoptimized': {'times': [], 'success_rate': 0.0}
        }
        
        def dummy_operation(circuit, optimized=False):
            """Dummy operation for benchmarking."""
            time.sleep(0.001 * getattr(circuit, 'num_qubits', 4))  # Simulate work
            return {'success': True, 'qubits': getattr(circuit, 'num_qubits', 4)}
        
        # Test optimized performance
        for _ in range(iterations):
            start_time = time.time()
            try:
                if self.config.enable_parallel_execution:
                    batch_results = self.execute_parallel_batch(dummy_operation, test_circuits, optimized=True)
                    success = all(r and r.get('success', False) for r in batch_results if r)
                else:
                    for circuit in test_circuits:
                        dummy_operation(circuit, optimized=True)
                    success = True
                    
                execution_time = time.time() - start_time
                results['optimized']['times'].append(execution_time)
                if success:
                    results['optimized']['success_rate'] += 1.0
                    
            except Exception as e:
                self.logger.error(f"Optimized benchmark error: {e}")
        
        # Test unoptimized performance
        original_config = self.config
        self.config = OptimizationConfig(
            enable_circuit_optimization=False,
            enable_caching=False,
            enable_parallel_execution=False
        )
        
        for _ in range(iterations):
            start_time = time.time()
            try:
                for circuit in test_circuits:
                    dummy_operation(circuit, optimized=False)
                    
                execution_time = time.time() - start_time
                results['unoptimized']['times'].append(execution_time)
                results['unoptimized']['success_rate'] += 1.0
                
            except Exception as e:
                self.logger.error(f"Unoptimized benchmark error: {e}")
        
        # Restore original config
        self.config = original_config
        
        # Calculate statistics
        for key in ['optimized', 'unoptimized']:
            if results[key]['times']:
                results[key]['avg_time'] = np.mean(results[key]['times'])
                results[key]['std_time'] = np.std(results[key]['times'])
                results[key]['min_time'] = np.min(results[key]['times'])
                results[key]['max_time'] = np.max(results[key]['times'])
            results[key]['success_rate'] /= iterations
        
        # Calculate improvement
        if results['optimized']['times'] and results['unoptimized']['times']:
            speedup = results['unoptimized']['avg_time'] / results['optimized']['avg_time']
            results['performance_improvement'] = {
                'speedup': speedup,
                'time_reduction_percent': ((results['unoptimized']['avg_time'] - results['optimized']['avg_time']) / results['unoptimized']['avg_time']) * 100
            }
        
        return results
    
    def shutdown(self) -> None:
        """Graceful shutdown of performance enhancer."""
        self.logger.info("Shutting down quantum performance enhancer")
        
        # Clear caches
        self.cache.clear()
        self.optimized_circuits.clear()
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        # Final memory optimization
        self.optimize_memory_usage()
        
        self.logger.info("Performance enhancer shutdown complete")

# Global performance enhancer instance
_global_enhancer = None

def get_performance_enhancer(config: Optional[OptimizationConfig] = None) -> QuantumPerformanceEnhancer:
    """Get global performance enhancer instance."""
    global _global_enhancer
    if _global_enhancer is None:
        _global_enhancer = QuantumPerformanceEnhancer(config)
    return _global_enhancer

# Performance optimization decorators
def optimize_performance(optimization_level: int = 2, enable_caching: bool = True):
    """Decorator for automatic performance optimization."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            enhancer = get_performance_enhancer()
            
            # Apply optimizations if first argument is a circuit
            if args and hasattr(args[0], 'num_qubits'):
                circuit = args[0]
                if enhancer.config.enable_circuit_optimization:
                    optimized_circuit = enhancer.optimize_circuit(circuit, optimization_level)
                    args = (optimized_circuit,) + args[1:]
            
            # Execute with caching if enabled
            if enable_caching and enhancer.config.enable_caching:
                return enhancer.execute_with_caching(func, *args, **kwargs)
            else:
                return func(*args, **kwargs)
                
        return wrapper
    return decorator

def parallel_batch(max_workers: int = None):
    """Decorator for parallel batch execution."""
    def decorator(func):
        @wraps(func)
        def wrapper(circuits, *args, **kwargs):
            if isinstance(circuits, list) and len(circuits) > 1:
                enhancer = get_performance_enhancer()
                return enhancer.execute_parallel_batch(func, circuits, *args, max_workers=max_workers, **kwargs)
            else:
                # Single circuit or not a list
                circuit = circuits[0] if isinstance(circuits, list) else circuits
                return func(circuit, *args, **kwargs)
        return wrapper
    return decorator