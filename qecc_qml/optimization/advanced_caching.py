"""
Advanced caching and performance optimization system.

Provides intelligent caching strategies, circuit optimization,
and performance acceleration for QECC-aware QML systems.
"""

import time
import hashlib
import pickle
import json
import threading
from typing import Any, Dict, List, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import OrderedDict, defaultdict
import logging
from pathlib import Path
import numpy as np
from functools import wraps, lru_cache


class CacheStrategy(Enum):
    """Cache replacement strategies."""
    LRU = "lru"           # Least Recently Used
    LFU = "lfu"           # Least Frequently Used
    FIFO = "fifo"         # First In, First Out
    TTL = "ttl"           # Time To Live
    ADAPTIVE = "adaptive"  # Adaptive based on access patterns


class CacheLevel(Enum):
    """Cache hierarchy levels."""
    L1_MEMORY = "l1_memory"     # In-memory cache
    L2_DISK = "l2_disk"         # Disk-based cache
    L3_DISTRIBUTED = "l3_distributed"  # Distributed cache


@dataclass
class CacheEntry:
    """Individual cache entry with metadata."""
    key: str
    value: Any
    timestamp: float
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    size_bytes: int = 0
    ttl: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if self.ttl is None:
            return False
        return (time.time() - self.timestamp) > self.ttl
    
    def touch(self):
        """Update access information."""
        self.access_count += 1
        self.last_accessed = time.time()


@dataclass
class CacheStats:
    """Cache performance statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    writes: int = 0
    size_bytes: int = 0
    entry_count: int = 0
    hit_rate: float = 0.0
    
    def update_hit_rate(self):
        """Update hit rate calculation."""
        total_requests = self.hits + self.misses
        self.hit_rate = self.hits / total_requests if total_requests > 0 else 0.0


class QuantumCircuitCache:
    """
    High-performance cache for quantum circuits and computations.
    
    Provides multi-level caching with intelligent eviction policies
    and performance optimization for quantum circuit operations.
    """
    
    def __init__(
        self,
        max_memory_size: int = 1024 * 1024 * 1024,  # 1GB
        max_entries: int = 10000,
        strategy: CacheStrategy = CacheStrategy.ADAPTIVE,
        enable_disk_cache: bool = True,
        disk_cache_path: Optional[str] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize quantum circuit cache.
        
        Args:
            max_memory_size: Maximum memory usage in bytes
            max_entries: Maximum number of cache entries
            strategy: Cache replacement strategy
            enable_disk_cache: Whether to enable disk caching
            disk_cache_path: Path for disk cache storage
            logger: Optional logger instance
        """
        self.max_memory_size = max_memory_size
        self.max_entries = max_entries
        self.strategy = strategy
        self.enable_disk_cache = enable_disk_cache
        self.logger = logger or logging.getLogger(__name__)
        
        # Multi-level cache storage
        self.l1_cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.l2_disk_cache_path = Path(disk_cache_path or "./cache/quantum_circuits")
        
        # Access tracking for adaptive strategies
        self.access_patterns: Dict[str, List[float]] = defaultdict(list)
        self.frequency_counter: Dict[str, int] = defaultdict(int)
        
        # Performance statistics
        self.stats = CacheStats()
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Circuit optimization cache
        self.optimized_circuits: Dict[str, Any] = {}
        self.compilation_cache: Dict[str, Any] = {}
        
        # Setup disk cache
        if self.enable_disk_cache:
            self.l2_disk_cache_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Initialized QuantumCircuitCache with {strategy.value} strategy")
    
    def _generate_key(self, circuit, parameters: Optional[np.ndarray] = None, **kwargs) -> str:
        """Generate unique cache key for circuit and parameters."""
        key_components = []
        
        # Circuit information
        if hasattr(circuit, 'qasm'):
            # For Qiskit circuits
            circuit_str = circuit.qasm()
        elif hasattr(circuit, '__str__'):
            circuit_str = str(circuit)
        else:
            circuit_str = repr(circuit)
        
        key_components.append(circuit_str)
        
        # Parameters
        if parameters is not None:
            param_hash = hashlib.md5(parameters.tobytes()).hexdigest()
            key_components.append(param_hash)
        
        # Additional kwargs
        for key, value in sorted(kwargs.items()):
            key_components.append(f"{key}={value}")
        
        # Generate final key
        combined = "|".join(key_components)
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self.lock:
            # Check L1 memory cache
            if key in self.l1_cache:
                entry = self.l1_cache[key]
                
                # Check expiration
                if entry.is_expired():
                    self._evict_entry(key)
                    self.stats.misses += 1
                    return None
                
                # Update access information
                entry.touch()
                
                # Move to end for LRU
                if self.strategy == CacheStrategy.LRU:
                    self.l1_cache.move_to_end(key)
                
                self.stats.hits += 1
                self.stats.update_hit_rate()
                
                return entry.value
            
            # Check L2 disk cache
            if self.enable_disk_cache:
                disk_value = self._load_from_disk(key)
                if disk_value is not None:
                    # Promote to L1 cache
                    self._set_l1(key, disk_value)
                    self.stats.hits += 1
                    self.stats.update_hit_rate()
                    return disk_value
            
            self.stats.misses += 1
            self.stats.update_hit_rate()
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None, metadata: Optional[Dict] = None):
        """Set value in cache."""
        with self.lock:
            # Calculate size
            try:
                size_bytes = len(pickle.dumps(value))
            except:
                size_bytes = 1024  # Default estimate
            
            # Create entry
            entry = CacheEntry(
                key=key,
                value=value,
                timestamp=time.time(),
                size_bytes=size_bytes,
                ttl=ttl,
                metadata=metadata or {}
            )
            
            # Set in L1 cache
            self._set_l1(key, value, entry)
            
            # Also save to disk cache if enabled
            if self.enable_disk_cache:
                self._save_to_disk(key, value, metadata)
            
            self.stats.writes += 1
    
    def _set_l1(self, key: str, value: Any, entry: Optional[CacheEntry] = None):
        """Set value in L1 memory cache with eviction."""
        if entry is None:
            try:
                size_bytes = len(pickle.dumps(value))
            except:
                size_bytes = 1024
            
            entry = CacheEntry(
                key=key,
                value=value,
                timestamp=time.time(),
                size_bytes=size_bytes
            )
        
        # Remove existing entry if present
        if key in self.l1_cache:
            old_entry = self.l1_cache[key]
            self.stats.size_bytes -= old_entry.size_bytes
            self.stats.entry_count -= 1
        
        # Check if we need to evict entries
        self._ensure_cache_capacity(entry.size_bytes)
        
        # Add new entry
        self.l1_cache[key] = entry
        self.stats.size_bytes += entry.size_bytes
        self.stats.entry_count += 1
        
        # Update access patterns for adaptive strategy
        if self.strategy == CacheStrategy.ADAPTIVE:
            self.access_patterns[key].append(time.time())
            self.frequency_counter[key] += 1
    
    def _ensure_cache_capacity(self, new_entry_size: int):
        """Ensure cache has capacity for new entry."""
        # Check memory limit
        while (self.stats.size_bytes + new_entry_size > self.max_memory_size or 
               len(self.l1_cache) >= self.max_entries):
            
            if not self.l1_cache:
                break
            
            evict_key = self._select_eviction_key()
            if evict_key:
                self._evict_entry(evict_key)
            else:
                break
    
    def _select_eviction_key(self) -> Optional[str]:
        """Select key for eviction based on strategy."""
        if not self.l1_cache:
            return None
        
        if self.strategy == CacheStrategy.LRU:
            # Least recently used (first in OrderedDict)
            return next(iter(self.l1_cache))
        
        elif self.strategy == CacheStrategy.LFU:
            # Least frequently used
            return min(self.l1_cache.keys(), 
                      key=lambda k: self.l1_cache[k].access_count)
        
        elif self.strategy == CacheStrategy.FIFO:
            # First in, first out (oldest timestamp)
            return min(self.l1_cache.keys(),
                      key=lambda k: self.l1_cache[k].timestamp)
        
        elif self.strategy == CacheStrategy.TTL:
            # Expired entries first, then oldest
            expired_keys = [k for k, v in self.l1_cache.items() if v.is_expired()]
            if expired_keys:
                return expired_keys[0]
            return min(self.l1_cache.keys(),
                      key=lambda k: self.l1_cache[k].timestamp)
        
        elif self.strategy == CacheStrategy.ADAPTIVE:
            return self._adaptive_eviction_key()
        
        else:
            return next(iter(self.l1_cache))
    
    def _adaptive_eviction_key(self) -> Optional[str]:
        """Select eviction key using adaptive strategy."""
        if not self.l1_cache:
            return None
        
        current_time = time.time()
        scores = {}
        
        for key, entry in self.l1_cache.items():
            # Calculate adaptive score based on multiple factors
            recency_score = 1.0 / (current_time - entry.last_accessed + 1)
            frequency_score = entry.access_count / (current_time - entry.timestamp + 1)
            size_penalty = entry.size_bytes / (1024 * 1024)  # Size in MB
            
            # Access pattern analysis
            pattern_score = 0.0
            if key in self.access_patterns:
                accesses = self.access_patterns[key]
                if len(accesses) > 1:
                    # Calculate access regularity
                    intervals = [accesses[i] - accesses[i-1] for i in range(1, len(accesses))]
                    if intervals:
                        regularity = 1.0 / (np.std(intervals) + 1)
                        pattern_score = regularity
            
            # Combine scores (higher is better)
            total_score = (recency_score * 0.3 + 
                          frequency_score * 0.3 + 
                          pattern_score * 0.2 - 
                          size_penalty * 0.2)
            
            scores[key] = total_score
        
        # Return key with lowest score for eviction
        return min(scores.keys(), key=lambda k: scores[k])
    
    def _evict_entry(self, key: str):
        """Evict entry from L1 cache."""
        if key in self.l1_cache:
            entry = self.l1_cache.pop(key)
            self.stats.size_bytes -= entry.size_bytes
            self.stats.entry_count -= 1
            self.stats.evictions += 1
            
            self.logger.debug(f"Evicted cache entry: {key}")
    
    def _save_to_disk(self, key: str, value: Any, metadata: Optional[Dict] = None):
        """Save entry to disk cache."""
        try:
            disk_file = self.l2_disk_cache_path / f"{key}.pkl"
            
            cache_data = {
                'value': value,
                'timestamp': time.time(),
                'metadata': metadata or {}
            }
            
            with open(disk_file, 'wb') as f:
                pickle.dump(cache_data, f)
                
        except Exception as e:
            self.logger.warning(f"Failed to save to disk cache: {e}")
    
    def _load_from_disk(self, key: str) -> Optional[Any]:
        """Load entry from disk cache."""
        try:
            disk_file = self.l2_disk_cache_path / f"{key}.pkl"
            
            if not disk_file.exists():
                return None
            
            with open(disk_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            return cache_data['value']
            
        except Exception as e:
            self.logger.warning(f"Failed to load from disk cache: {e}")
            return None
    
    def invalidate(self, key: str) -> bool:
        """Invalidate cache entry."""
        with self.lock:
            removed = False
            
            # Remove from L1 cache
            if key in self.l1_cache:
                self._evict_entry(key)
                removed = True
            
            # Remove from disk cache
            if self.enable_disk_cache:
                disk_file = self.l2_disk_cache_path / f"{key}.pkl"
                if disk_file.exists():
                    disk_file.unlink()
                    removed = True
            
            return removed
    
    def clear(self):
        """Clear all cache entries."""
        with self.lock:
            self.l1_cache.clear()
            self.access_patterns.clear()
            self.frequency_counter.clear()
            self.optimized_circuits.clear()
            self.compilation_cache.clear()
            
            # Reset statistics
            self.stats = CacheStats()
            
            # Clear disk cache
            if self.enable_disk_cache:
                for cache_file in self.l2_disk_cache_path.glob("*.pkl"):
                    cache_file.unlink()
            
            self.logger.info("Cache cleared")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        with self.lock:
            return {
                'hits': self.stats.hits,
                'misses': self.stats.misses,
                'hit_rate': self.stats.hit_rate,
                'evictions': self.stats.evictions,
                'writes': self.stats.writes,
                'entries': self.stats.entry_count,
                'size_mb': self.stats.size_bytes / (1024 * 1024),
                'max_size_mb': self.max_memory_size / (1024 * 1024),
                'utilization': self.stats.size_bytes / self.max_memory_size,
                'strategy': self.strategy.value
            }
    
    def optimize_cache_strategy(self):
        """Automatically optimize cache strategy based on access patterns."""
        if self.strategy != CacheStrategy.ADAPTIVE:
            return
        
        with self.lock:
            current_time = time.time()
            
            # Analyze access patterns
            total_accesses = sum(self.frequency_counter.values())
            if total_accesses < 100:
                return  # Not enough data
            
            # Calculate pattern metrics
            temporal_locality = 0.0
            spatial_locality = 0.0
            frequency_variance = 0.0
            
            if self.access_patterns:
                # Temporal locality: how often items are re-accessed quickly
                recent_reaccesses = 0
                total_intervals = 0
                
                for key, accesses in self.access_patterns.items():
                    if len(accesses) > 1:
                        intervals = [accesses[i] - accesses[i-1] for i in range(1, len(accesses))]
                        short_intervals = [i for i in intervals if i < 300]  # 5 minutes
                        recent_reaccesses += len(short_intervals)
                        total_intervals += len(intervals)
                
                if total_intervals > 0:
                    temporal_locality = recent_reaccesses / total_intervals
                
                # Frequency variance
                frequencies = list(self.frequency_counter.values())
                if len(frequencies) > 1:
                    frequency_variance = np.var(frequencies) / np.mean(frequencies)
            
            # Adjust strategy based on patterns
            if temporal_locality > 0.7:
                # High temporal locality - LRU works well
                self.logger.info("High temporal locality detected, LRU would be optimal")
            elif frequency_variance > 2.0:
                # High frequency variance - LFU might be better
                self.logger.info("High frequency variance detected, LFU might be optimal")
            else:
                # Balanced patterns - continue with adaptive
                self.logger.info("Balanced access patterns, continuing with adaptive strategy")
    
    def cache_circuit_execution(self, circuit, parameters: Optional[np.ndarray] = None, **execution_kwargs):
        """Decorator for caching circuit execution results."""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Generate cache key
                cache_key = self._generate_key(circuit, parameters, **execution_kwargs)
                
                # Try to get from cache
                cached_result = self.get(cache_key)
                if cached_result is not None:
                    return cached_result
                
                # Execute function and cache result
                result = func(*args, **kwargs)
                self.set(cache_key, result, ttl=3600)  # Cache for 1 hour
                
                return result
            
            return wrapper
        return decorator
    
    def get_cache_efficiency_report(self) -> Dict[str, Any]:
        """Generate detailed cache efficiency report."""
        with self.lock:
            stats = self.get_statistics()
            
            # Calculate efficiency metrics
            memory_efficiency = stats['entries'] / self.max_entries if self.max_entries > 0 else 0
            size_efficiency = stats['utilization']
            
            # Analyze access patterns
            hot_keys = []
            cold_keys = []
            
            for key, entry in self.l1_cache.items():
                if entry.access_count > 5:
                    hot_keys.append({
                        'key': key[:32] + '...' if len(key) > 32 else key,
                        'access_count': entry.access_count,
                        'size_kb': entry.size_bytes / 1024
                    })
                elif entry.access_count == 1:
                    cold_keys.append({
                        'key': key[:32] + '...' if len(key) > 32 else key,
                        'age_minutes': (time.time() - entry.timestamp) / 60
                    })
            
            # Sort by access frequency
            hot_keys.sort(key=lambda x: x['access_count'], reverse=True)
            cold_keys.sort(key=lambda x: x['age_minutes'], reverse=True)
            
            return {
                'performance': stats,
                'efficiency': {
                    'memory_efficiency': memory_efficiency,
                    'size_efficiency': size_efficiency,
                    'overall_efficiency': (stats['hit_rate'] + memory_efficiency + size_efficiency) / 3
                },
                'hot_items': hot_keys[:10],  # Top 10 most accessed
                'cold_items': cold_keys[:10],  # Top 10 least accessed
                'recommendations': self._generate_cache_recommendations(stats)
            }
    
    def _generate_cache_recommendations(self, stats: Dict[str, Any]) -> List[str]:
        """Generate cache optimization recommendations."""
        recommendations = []
        
        if stats['hit_rate'] < 0.5:
            recommendations.append("Low hit rate. Consider increasing cache size or adjusting TTL values.")
        
        if stats['utilization'] > 0.9:
            recommendations.append("High memory utilization. Consider increasing max cache size.")
        
        if stats['evictions'] > stats['writes'] * 0.3:
            recommendations.append("High eviction rate. Cache may be too small for workload.")
        
        if len(self.access_patterns) > 0:
            avg_pattern_length = np.mean([len(pattern) for pattern in self.access_patterns.values()])
            if avg_pattern_length < 2:
                recommendations.append("Low access pattern complexity. Consider simpler cache strategy.")
        
        return recommendations


class PerformanceOptimizer:
    """
    Quantum circuit performance optimization system.
    
    Provides circuit compilation, optimization passes,
    and performance acceleration techniques.
    """
    
    def __init__(
        self,
        cache: Optional[QuantumCircuitCache] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize performance optimizer.
        
        Args:
            cache: Circuit cache instance
            logger: Optional logger instance
        """
        self.cache = cache or QuantumCircuitCache()
        self.logger = logger or logging.getLogger(__name__)
        
        # Optimization statistics
        self.optimization_stats = {
            'circuits_optimized': 0,
            'total_time_saved': 0.0,
            'average_depth_reduction': 0.0,
            'average_gate_reduction': 0.0
        }
        
        # Compilation cache
        self.compiled_circuits: Dict[str, Any] = {}
        
        self.logger.info("Initialized PerformanceOptimizer")
    
    def optimize_circuit(self, circuit, optimization_level: int = 3) -> Any:
        """
        Optimize quantum circuit for performance.
        
        Args:
            circuit: Quantum circuit to optimize
            optimization_level: Optimization level (0-3)
            
        Returns:
            Optimized circuit
        """
        # Generate cache key
        cache_key = self.cache._generate_key(circuit, optimization_level=optimization_level)
        
        # Check cache first
        cached_circuit = self.cache.get(cache_key)
        if cached_circuit is not None:
            return cached_circuit
        
        start_time = time.time()
        
        # Perform optimization based on level
        if optimization_level == 0:
            optimized_circuit = circuit  # No optimization
        elif optimization_level == 1:
            optimized_circuit = self._basic_optimization(circuit)
        elif optimization_level == 2:
            optimized_circuit = self._advanced_optimization(circuit)
        else:  # optimization_level >= 3
            optimized_circuit = self._aggressive_optimization(circuit)
        
        optimization_time = time.time() - start_time
        
        # Update statistics
        self._update_optimization_stats(circuit, optimized_circuit, optimization_time)
        
        # Cache optimized circuit
        self.cache.set(cache_key, optimized_circuit, ttl=7200)  # Cache for 2 hours
        
        return optimized_circuit
    
    def _basic_optimization(self, circuit):
        """Basic circuit optimization."""
        try:
            # For Qiskit circuits
            if hasattr(circuit, 'decompose'):
                # Basic gate decomposition and cancellation
                try:
    from qiskit import transpile
except ImportError:
    from qecc_qml.core.fallback_imports import transpile
                from qiskit.transpiler import PassManager
                from qiskit.transpiler.passes import Unroller, CXCancellation, CommutationAnalysis
                
                pass_manager = PassManager([
                    Unroller(['u1', 'u2', 'u3', 'cx']),
                    CXCancellation(),
                    CommutationAnalysis()
                ])
                
                return pass_manager.run(circuit)
            
        except ImportError:
            self.logger.warning("Qiskit not available for circuit optimization")
        
        return circuit
    
    def _advanced_optimization(self, circuit):
        """Advanced circuit optimization."""
        try:
            if hasattr(circuit, 'decompose'):
                try:
    from qiskit import transpile
except ImportError:
    from qecc_qml.core.fallback_imports import transpile
                from qiskit.transpiler import PassManager
                from qiskit.transpiler.passes import (
                    Unroller, CXCancellation, CommutationAnalysis, CommutativeCancellation,
                    Optimize1qGates, DenseLayout, Optimize1qGatesDecomposition
                )
                
                pass_manager = PassManager([
                    # Unroll to basic gates
                    Unroller(['u1', 'u2', 'u3', 'cx']),
                    
                    # Optimization passes
                    Optimize1qGates(),
                    CXCancellation(),
                    CommutationAnalysis(),
                    CommutativeCancellation(),
                    
                    # Layout optimization
                    DenseLayout(),
                    
                    # Further optimization
                    Optimize1qGatesDecomposition(),
                ])
                
                return pass_manager.run(circuit)
        
        except ImportError:
            pass
        
        return self._basic_optimization(circuit)
    
    def _aggressive_optimization(self, circuit):
        """Aggressive circuit optimization with all available passes."""
        try:
            if hasattr(circuit, 'decompose'):
                try:
    from qiskit import transpile
except ImportError:
    from qecc_qml.core.fallback_imports import transpile
                from qiskit.transpiler import PassManager
                from qiskit.transpiler.passes import (
                    Unroller, CXCancellation, CommutationAnalysis, CommutativeCancellation,
                    Optimize1qGates, DenseLayout, Optimize1qGatesDecomposition,
                    RemoveResetInZeroState, RemoveBarriers, ConsolidateBlocks,
                    RemoveEmptyBlocks, Depth
                )
                
                pass_manager = PassManager([
                    # Initial optimization
                    RemoveResetInZeroState(),
                    RemoveBarriers(),
                    
                    # Unroll and optimize
                    Unroller(['u1', 'u2', 'u3', 'cx']),
                    Optimize1qGates(),
                    
                    # Commutation and cancellation
                    CommutationAnalysis(),
                    CommutativeCancellation(),
                    CXCancellation(),
                    
                    # Advanced optimizations
                    ConsolidateBlocks(),
                    Optimize1qGatesDecomposition(),
                    
                    # Layout optimization
                    DenseLayout(),
                    
                    # Final cleanup
                    RemoveEmptyBlocks(),
                    Depth(),  # Minimize circuit depth
                ])
                
                return pass_manager.run(circuit)
        
        except ImportError:
            pass
        
        return self._advanced_optimization(circuit)
    
    def _update_optimization_stats(self, original_circuit, optimized_circuit, optimization_time: float):
        """Update optimization statistics."""
        self.optimization_stats['circuits_optimized'] += 1
        
        try:
            # Calculate improvements
            if hasattr(original_circuit, 'depth') and hasattr(optimized_circuit, 'depth'):
                original_depth = original_circuit.depth()
                optimized_depth = optimized_circuit.depth()
                
                if original_depth > 0:
                    depth_reduction = (original_depth - optimized_depth) / original_depth
                    
                    # Update running average
                    n = self.optimization_stats['circuits_optimized']
                    current_avg = self.optimization_stats['average_depth_reduction']
                    self.optimization_stats['average_depth_reduction'] = (
                        (current_avg * (n - 1) + depth_reduction) / n
                    )
            
            # Similar for gate count if available
            if (hasattr(original_circuit, 'count_ops') and 
                hasattr(optimized_circuit, 'count_ops')):
                
                original_gates = sum(original_circuit.count_ops().values())
                optimized_gates = sum(optimized_circuit.count_ops().values())
                
                if original_gates > 0:
                    gate_reduction = (original_gates - optimized_gates) / original_gates
                    
                    n = self.optimization_stats['circuits_optimized']
                    current_avg = self.optimization_stats['average_gate_reduction']
                    self.optimization_stats['average_gate_reduction'] = (
                        (current_avg * (n - 1) + gate_reduction) / n
                    )
        
        except Exception as e:
            self.logger.debug(f"Error calculating optimization statistics: {e}")
        
        # Estimate time savings (rough heuristic)
        estimated_time_saved = optimization_time * 10  # Assume 10x speedup from optimization
        self.optimization_stats['total_time_saved'] += estimated_time_saved
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Get optimization performance report."""
        return {
            'statistics': self.optimization_stats.copy(),
            'cache_performance': self.cache.get_statistics(),
            'recommendations': self._generate_optimization_recommendations()
        }
    
    def _generate_optimization_recommendations(self) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        
        stats = self.optimization_stats
        cache_stats = self.cache.get_statistics()
        
        if stats['circuits_optimized'] > 0:
            if stats['average_depth_reduction'] < 0.1:
                recommendations.append("Low circuit depth reduction. Circuits may already be well optimized.")
            
            if stats['average_gate_reduction'] < 0.05:
                recommendations.append("Low gate count reduction. Consider more aggressive optimization levels.")
        
        if cache_stats['hit_rate'] < 0.3:
            recommendations.append("Low cache hit rate for optimized circuits. Consider increasing cache size.")
        
        if stats['circuits_optimized'] < 10:
            recommendations.append("Limited optimization data. Performance will improve with more circuit optimizations.")
        
        return recommendations
    
    @lru_cache(maxsize=128)
    def get_optimal_backend(self, circuit_properties: Tuple[int, int, str]) -> str:
        """Get optimal backend for circuit execution (cached)."""
        num_qubits, circuit_depth, gate_types = circuit_properties
        
        # Simple backend selection heuristic
        if num_qubits <= 5 and circuit_depth <= 10:
            return "qasm_simulator"
        elif num_qubits <= 15:
            return "aer_simulator"
        else:
            return "statevector_simulator"
    
    def benchmark_optimization_levels(self, circuit, repetitions: int = 3) -> Dict[str, Any]:
        """Benchmark different optimization levels for a circuit."""
        results = {}
        
        for level in range(4):
            times = []
            
            for _ in range(repetitions):
                # Clear cache for fair comparison
                cache_key = self.cache._generate_key(circuit, optimization_level=level)
                self.cache.invalidate(cache_key)
                
                start_time = time.time()
                optimized_circuit = self.optimize_circuit(circuit, optimization_level=level)
                end_time = time.time()
                
                times.append(end_time - start_time)
            
            avg_time = np.mean(times)
            std_time = np.std(times)
            
            # Get circuit properties
            try:
                if hasattr(optimized_circuit, 'depth'):
                    depth = optimized_circuit.depth()
                    gate_count = sum(optimized_circuit.count_ops().values()) if hasattr(optimized_circuit, 'count_ops') else 0
                else:
                    depth = 0
                    gate_count = 0
                    
            except:
                depth = 0
                gate_count = 0
            
            results[f'level_{level}'] = {
                'avg_time': avg_time,
                'std_time': std_time,
                'depth': depth,
                'gate_count': gate_count,
                'efficiency_score': depth * gate_count / (avg_time + 0.001)  # Higher is better
            }
        
        return results