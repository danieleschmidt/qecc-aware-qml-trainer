"""
Advanced caching systems for quantum circuit optimization.
"""

import hashlib
import pickle
import threading
import time
from typing import Any, Dict, Optional, Tuple, List, Union
from collections import OrderedDict
import numpy as np
from pathlib import Path
import weakref
import gc

from ..utils.logging_config import get_logger
from ..utils.security import sanitize_input

logger = get_logger(__name__)


class QECCCache:
    """
    High-performance cache for QECC quantum computations.
    """
    
    def __init__(self, max_size: int = 1000):
        """Initialize QECC cache."""
        self.lru_cache = LRUCache(max_size=max_size)
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        return self.lru_cache.get(key)
    
    def set(self, key: str, value: Any) -> None:
        """Set item in cache."""
        self.lru_cache.put(key, value)
    
    def clear(self) -> None:
        """Clear the cache."""
        self.lru_cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.lru_cache.stats()


class LRUCache:
    """
    Thread-safe Least Recently Used cache with size and TTL limits.
    """
    
    def __init__(self, max_size: int = 1000, ttl: Optional[float] = 3600):
        """
        Initialize LRU cache.
        
        Args:
            max_size: Maximum number of items to cache
            ttl: Time-to-live in seconds (None for no expiration)
        """
        self.max_size = max_size
        self.ttl = ttl
        self._cache = OrderedDict()
        self._timestamps = {}
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None
            
            # Check TTL
            if self.ttl and time.time() - self._timestamps[key] > self.ttl:
                del self._cache[key]
                del self._timestamps[key]
                self._misses += 1
                return None
            
            # Move to end (most recently used)
            value = self._cache.pop(key)
            self._cache[key] = value
            self._hits += 1
            
            return value
    
    def put(self, key: str, value: Any):
        """Put item in cache."""
        with self._lock:
            if key in self._cache:
                # Update existing item
                self._cache.pop(key)
            elif len(self._cache) >= self.max_size:
                # Remove least recently used item
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                if oldest_key in self._timestamps:
                    del self._timestamps[oldest_key]
            
            self._cache[key] = value
            self._timestamps[key] = time.time()
    
    def invalidate(self, key: str) -> bool:
        """Remove item from cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                if key in self._timestamps:
                    del self._timestamps[key]
                return True
            return False
    
    def clear(self):
        """Clear all cached items."""
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()
            self._hits = 0
            self._misses = 0
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0
            
            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'hits': self._hits,
                'misses': self._misses,
                'hit_rate': hit_rate,
                'ttl': self.ttl
            }


class CircuitCache:
    """
    Specialized cache for quantum circuits and their compiled forms.
    """
    
    def __init__(self, max_size: int = 500, persistent_path: Optional[str] = None):
        """
        Initialize circuit cache.
        
        Args:
            max_size: Maximum number of circuits to cache
            persistent_path: Path for persistent caching (optional)
        """
        self.cache = LRUCache(max_size=max_size, ttl=1800)  # 30 minutes TTL
        self.persistent_path = Path(persistent_path) if persistent_path else None
        self._circuit_hashes = {}
        
        if self.persistent_path:
            self.persistent_path.mkdir(parents=True, exist_ok=True)
            self._load_persistent_cache()
    
    def _compute_circuit_hash(self, circuit_data: Dict[str, Any]) -> str:
        """Compute hash for circuit configuration."""
        # Create deterministic hash from circuit parameters
        hash_data = {
            'num_qubits': circuit_data.get('num_qubits', 0),
            'num_layers': circuit_data.get('num_layers', 0),
            'entanglement': circuit_data.get('entanglement', ''),
            'feature_map': circuit_data.get('feature_map', ''),
            'rotation_gates': str(sorted(circuit_data.get('rotation_gates', []))),
            'error_correction': str(circuit_data.get('error_correction', '')),
        }
        
        # Create hash
        hash_string = str(sorted(hash_data.items()))
        return hashlib.sha256(hash_string.encode()).hexdigest()[:16]
    
    def get_circuit(self, circuit_config: Dict[str, Any]) -> Optional[Any]:
        """Get cached circuit."""
        circuit_hash = self._compute_circuit_hash(circuit_config)
        
        # Try memory cache first
        cached = self.cache.get(circuit_hash)
        if cached is not None:
            logger.debug(f"Circuit cache hit: {circuit_hash}")
            return cached
        
        # Try persistent cache
        if self.persistent_path:
            cached = self._load_from_persistent(circuit_hash)
            if cached is not None:
                # Store in memory cache too
                self.cache.put(circuit_hash, cached)
                logger.debug(f"Circuit loaded from persistent cache: {circuit_hash}")
                return cached
        
        logger.debug(f"Circuit cache miss: {circuit_hash}")
        return None
    
    def store_circuit(self, circuit_config: Dict[str, Any], circuit: Any):
        """Store circuit in cache."""
        circuit_hash = self._compute_circuit_hash(circuit_config)
        
        # Store in memory cache
        self.cache.put(circuit_hash, circuit)
        
        # Store in persistent cache
        if self.persistent_path:
            self._save_to_persistent(circuit_hash, circuit)
        
        logger.debug(f"Circuit cached: {circuit_hash}")
    
    def _load_from_persistent(self, circuit_hash: str) -> Optional[Any]:
        """Load circuit from persistent cache."""
        try:
            cache_file = self.persistent_path / f"{circuit_hash}.pkl"
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load persistent circuit {circuit_hash}: {e}")
        
        return None
    
    def _save_to_persistent(self, circuit_hash: str, circuit: Any):
        """Save circuit to persistent cache."""
        try:
            cache_file = self.persistent_path / f"{circuit_hash}.pkl"
            with open(cache_file, 'wb') as f:
                pickle.dump(circuit, f)
        except Exception as e:
            logger.warning(f"Failed to save persistent circuit {circuit_hash}: {e}")
    
    def _load_persistent_cache(self):
        """Load all persistent cached circuits."""
        if not self.persistent_path.exists():
            return
        
        loaded_count = 0
        for cache_file in self.persistent_path.glob("*.pkl"):
            try:
                circuit_hash = cache_file.stem
                with open(cache_file, 'rb') as f:
                    circuit = pickle.load(f)
                self.cache.put(circuit_hash, circuit)
                loaded_count += 1
            except Exception as e:
                logger.warning(f"Failed to load {cache_file}: {e}")
        
        if loaded_count > 0:
            logger.info(f"Loaded {loaded_count} circuits from persistent cache")
    
    def cleanup_persistent(self, max_age_days: int = 7):
        """Clean up old persistent cache files."""
        if not self.persistent_path:
            return
        
        cutoff_time = time.time() - (max_age_days * 24 * 3600)
        removed_count = 0
        
        for cache_file in self.persistent_path.glob("*.pkl"):
            if cache_file.stat().st_mtime < cutoff_time:
                try:
                    cache_file.unlink()
                    removed_count += 1
                except Exception as e:
                    logger.warning(f"Failed to remove old cache file {cache_file}: {e}")
        
        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} old cache files")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = self.cache.stats()
        
        if self.persistent_path and self.persistent_path.exists():
            persistent_files = len(list(self.persistent_path.glob("*.pkl")))
            stats['persistent_files'] = persistent_files
        
        return stats


class ParameterCache:
    """
    Cache for parameter optimization trajectories and gradients.
    """
    
    def __init__(self, max_size: int = 1000):
        """Initialize parameter cache."""
        self.gradient_cache = LRUCache(max_size=max_size, ttl=1800)
        self.trajectory_cache = LRUCache(max_size=max_size // 10, ttl=3600)
        self._parameter_tolerance = 1e-6
    
    def _parameter_hash(self, parameters: np.ndarray) -> str:
        """Create hash for parameter vector."""
        # Round parameters to avoid floating point precision issues
        rounded_params = np.round(parameters, decimals=6)
        return hashlib.sha256(rounded_params.tobytes()).hexdigest()[:16]
    
    def get_gradient(
        self, 
        parameters: np.ndarray, 
        circuit_hash: str, 
        loss_function: str
    ) -> Optional[np.ndarray]:
        """Get cached gradient."""
        cache_key = f"{self._parameter_hash(parameters)}_{circuit_hash}_{loss_function}"
        return self.gradient_cache.get(cache_key)
    
    def store_gradient(
        self, 
        parameters: np.ndarray, 
        gradient: np.ndarray,
        circuit_hash: str,
        loss_function: str
    ):
        """Store gradient in cache."""
        cache_key = f"{self._parameter_hash(parameters)}_{circuit_hash}_{loss_function}"
        self.gradient_cache.put(cache_key, gradient.copy())
    
    def get_trajectory(self, trajectory_id: str) -> Optional[List[np.ndarray]]:
        """Get cached parameter trajectory."""
        return self.trajectory_cache.get(trajectory_id)
    
    def store_trajectory(self, trajectory_id: str, trajectory: List[np.ndarray]):
        """Store parameter trajectory."""
        # Store copy to avoid references
        trajectory_copy = [params.copy() for params in trajectory]
        self.trajectory_cache.put(trajectory_id, trajectory_copy)
    
    def find_similar_parameters(
        self, 
        target_parameters: np.ndarray, 
        tolerance: float = None
    ) -> List[Tuple[str, np.ndarray, float]]:
        """
        Find parameters similar to target within tolerance.
        
        Returns:
            List of (cache_key, parameters, distance) tuples
        """
        if tolerance is None:
            tolerance = self._parameter_tolerance
        
        similar = []
        target_hash = self._parameter_hash(target_parameters)
        
        # This is a simplified similarity search
        # In practice, you might use more sophisticated methods
        with self.gradient_cache._lock:
            for key in self.gradient_cache._cache:
                if key.startswith(target_hash):
                    # Found exact match
                    similar.append((key, target_parameters, 0.0))
        
        return similar
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'gradient_cache': self.gradient_cache.stats(),
            'trajectory_cache': self.trajectory_cache.stats(),
        }


class ResultCache:
    """
    Cache for quantum circuit execution results and measurements.
    """
    
    def __init__(self, max_size: int = 2000):
        """Initialize result cache."""
        self.execution_cache = LRUCache(max_size=max_size, ttl=600)  # 10 minutes
        self.measurement_cache = LRUCache(max_size=max_size * 2, ttl=300)  # 5 minutes
        self._shot_noise_tolerance = 0.1  # Allow 10% shot noise variation
    
    def _execution_hash(
        self, 
        circuit_hash: str, 
        parameters: np.ndarray, 
        shots: int,
        backend_name: str
    ) -> str:
        """Create hash for circuit execution."""
        param_hash = hashlib.sha256(parameters.tobytes()).hexdigest()[:8]
        execution_data = f"{circuit_hash}_{param_hash}_{shots}_{backend_name}"
        return hashlib.sha256(execution_data.encode()).hexdigest()[:16]
    
    def get_execution_result(
        self,
        circuit_hash: str,
        parameters: np.ndarray,
        shots: int,
        backend_name: str,
        allow_shot_noise: bool = True
    ) -> Optional[Any]:
        """Get cached execution result."""
        execution_hash = self._execution_hash(circuit_hash, parameters, shots, backend_name)
        
        # Try exact match first
        result = self.execution_cache.get(execution_hash)
        if result is not None:
            return result
        
        # If allowing shot noise, try to find similar results with different shot counts
        if allow_shot_noise:
            base_hash = f"{circuit_hash}_{hashlib.sha256(parameters.tobytes()).hexdigest()[:8]}"
            
            with self.execution_cache._lock:
                for key in list(self.execution_cache._cache.keys()):
                    if key.startswith(base_hash):
                        cached_result = self.execution_cache._cache[key]
                        # Check if shot counts are similar enough
                        if hasattr(cached_result, 'shots'):
                            shot_ratio = min(shots, cached_result.shots) / max(shots, cached_result.shots)
                            if shot_ratio > (1 - self._shot_noise_tolerance):
                                logger.debug(f"Using similar result with {cached_result.shots} shots instead of {shots}")
                                return cached_result
        
        return None
    
    def store_execution_result(
        self,
        circuit_hash: str,
        parameters: np.ndarray,
        shots: int,
        backend_name: str,
        result: Any
    ):
        """Store execution result."""
        execution_hash = self._execution_hash(circuit_hash, parameters, shots, backend_name)
        
        # Add shot count to result for shot noise matching
        if hasattr(result, '__dict__'):
            result.shots = shots
        
        self.execution_cache.put(execution_hash, result)
    
    def get_measurement_counts(
        self,
        circuit_hash: str,
        parameters: np.ndarray,
        shots: int
    ) -> Optional[Dict[str, int]]:
        """Get cached measurement counts."""
        measurement_hash = f"{circuit_hash}_{hashlib.sha256(parameters.tobytes()).hexdigest()[:8]}_{shots}"
        return self.measurement_cache.get(measurement_hash)
    
    def store_measurement_counts(
        self,
        circuit_hash: str,
        parameters: np.ndarray,
        shots: int,
        counts: Dict[str, int]
    ):
        """Store measurement counts."""
        measurement_hash = f"{circuit_hash}_{hashlib.sha256(parameters.tobytes()).hexdigest()[:8]}_{shots}"
        self.measurement_cache.put(measurement_hash, counts.copy())
    
    def estimate_cache_effectiveness(self) -> Dict[str, float]:
        """Estimate cache effectiveness metrics."""
        exec_stats = self.execution_cache.stats()
        meas_stats = self.measurement_cache.stats()
        
        return {
            'execution_hit_rate': exec_stats['hit_rate'],
            'measurement_hit_rate': meas_stats['hit_rate'],
            'combined_hit_rate': (exec_stats['hit_rate'] + meas_stats['hit_rate']) / 2,
            'cache_utilization': (exec_stats['size'] / exec_stats['max_size'] + 
                                meas_stats['size'] / meas_stats['max_size']) / 2
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'execution_cache': self.execution_cache.stats(),
            'measurement_cache': self.measurement_cache.stats(),
            'effectiveness': self.estimate_cache_effectiveness()
        }


class SmartCache:
    """
    Intelligent cache manager that coordinates multiple cache types.
    """
    
    def __init__(
        self, 
        circuit_cache_size: int = 500,
        parameter_cache_size: int = 1000,
        result_cache_size: int = 2000,
        persistent_cache_path: Optional[str] = None
    ):
        """Initialize smart cache manager."""
        self.circuit_cache = CircuitCache(
            max_size=circuit_cache_size,
            persistent_path=persistent_cache_path
        )
        self.parameter_cache = ParameterCache(max_size=parameter_cache_size)
        self.result_cache = ResultCache(max_size=result_cache_size)
        
        # Cache coordination
        self._cache_hits = 0
        self._cache_misses = 0
        self._last_cleanup = time.time()
        self._cleanup_interval = 3600  # 1 hour
    
    def get_or_build_circuit(self, circuit_config: Dict[str, Any], builder_func):
        """Get circuit from cache or build using provided function."""
        circuit = self.circuit_cache.get_circuit(circuit_config)
        
        if circuit is None:
            # Build circuit
            circuit = builder_func(circuit_config)
            self.circuit_cache.store_circuit(circuit_config, circuit)
            self._cache_misses += 1
        else:
            self._cache_hits += 1
        
        return circuit
    
    def get_or_compute_gradient(
        self, 
        parameters: np.ndarray,
        circuit_hash: str,
        loss_function: str,
        gradient_func
    ) -> np.ndarray:
        """Get gradient from cache or compute using provided function."""
        gradient = self.parameter_cache.get_gradient(parameters, circuit_hash, loss_function)
        
        if gradient is None:
            gradient = gradient_func(parameters)
            self.parameter_cache.store_gradient(parameters, gradient, circuit_hash, loss_function)
            self._cache_misses += 1
        else:
            self._cache_hits += 1
        
        return gradient
    
    def get_or_execute_circuit(
        self,
        circuit_hash: str,
        parameters: np.ndarray,
        shots: int,
        backend_name: str,
        execution_func
    ) -> Any:
        """Get execution result from cache or execute using provided function."""
        result = self.result_cache.get_execution_result(
            circuit_hash, parameters, shots, backend_name
        )
        
        if result is None:
            result = execution_func()
            self.result_cache.store_execution_result(
                circuit_hash, parameters, shots, backend_name, result
            )
            self._cache_misses += 1
        else:
            self._cache_hits += 1
        
        return result
    
    def periodic_cleanup(self):
        """Perform periodic cache cleanup and optimization."""
        current_time = time.time()
        
        if current_time - self._last_cleanup > self._cleanup_interval:
            logger.info("Performing cache cleanup")
            
            # Clean up persistent cache
            self.circuit_cache.cleanup_persistent()
            
            # Force garbage collection
            gc.collect()
            
            self._last_cleanup = current_time
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        total_requests = self._cache_hits + self._cache_misses
        overall_hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            'overall_hit_rate': overall_hit_rate,
            'total_hits': self._cache_hits,
            'total_misses': self._cache_misses,
            'circuit_cache': self.circuit_cache.get_stats(),
            'parameter_cache': self.parameter_cache.get_stats(),
            'result_cache': self.result_cache.get_stats(),
        }
    
    def clear_all_caches(self):
        """Clear all caches."""
        self.circuit_cache.cache.clear()
        self.parameter_cache.gradient_cache.clear()
        self.parameter_cache.trajectory_cache.clear()
        self.result_cache.execution_cache.clear()
        self.result_cache.measurement_cache.clear()
        
        self._cache_hits = 0
        self._cache_misses = 0
        
        logger.info("All caches cleared")
    
    def optimize_cache_sizes(self, target_hit_rate: float = 0.8):
        """
        Automatically optimize cache sizes based on usage patterns.
        
        Args:
            target_hit_rate: Target cache hit rate
        """
        stats = self.get_comprehensive_stats()
        
        # Simple heuristic: increase cache size if hit rate is below target
        if stats['overall_hit_rate'] < target_hit_rate:
            # Increase cache sizes by 20%
            current_circuit_size = self.circuit_cache.cache.max_size
            current_param_size = self.parameter_cache.gradient_cache.max_size
            current_result_size = self.result_cache.execution_cache.max_size
            
            new_circuit_size = int(current_circuit_size * 1.2)
            new_param_size = int(current_param_size * 1.2)
            new_result_size = int(current_result_size * 1.2)
            
            logger.info(f"Optimizing cache sizes: circuit {current_circuit_size}→{new_circuit_size}, "
                       f"parameter {current_param_size}→{new_param_size}, "
                       f"result {current_result_size}→{new_result_size}")
            
            # Note: In a full implementation, you'd need to recreate the caches
            # This is a simplified version showing the concept