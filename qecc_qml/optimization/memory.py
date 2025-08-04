"""
Memory management and resource pooling for quantum ML workloads.
"""

import gc
import sys
import threading
import time
import weakref
from typing import Any, Dict, List, Optional, Set, Tuple, Callable
from collections import defaultdict, deque
import numpy as np
import psutil

from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class MemoryPool:
    """
    Memory pool for reusing numpy arrays and quantum circuit objects.
    """
    
    def __init__(self, max_pool_size: int = 100):
        """
        Initialize memory pool.
        
        Args:
            max_pool_size: Maximum number of objects to pool
        """
        self.max_pool_size = max_pool_size
        self._pools = defaultdict(deque)  # Type -> deque of objects
        self._pool_sizes = defaultdict(int)
        self._lock = threading.RLock()
        self._stats = {
            'allocations': 0,
            'deallocations': 0,
            'pool_hits': 0,
            'pool_misses': 0
        }
        
        logger.debug(f"Memory pool initialized with max size {max_pool_size}")
    
    def get_array(self, shape: Tuple[int, ...], dtype: np.dtype = np.float64) -> np.ndarray:
        """
        Get numpy array from pool or create new one.
        
        Args:
            shape: Array shape
            dtype: Array data type
            
        Returns:
            Numpy array
        """
        pool_key = (shape, dtype)
        
        with self._lock:
            pool = self._pools[pool_key]
            
            if pool:
                array = pool.popleft()
                self._pool_sizes[pool_key] -= 1
                self._stats['pool_hits'] += 1
                
                # Clear array contents
                array.fill(0)
                
                logger.debug(f"Reused array from pool: {shape}, {dtype}")
                return array
            else:
                # Create new array
                array = np.zeros(shape, dtype=dtype)
                self._stats['pool_misses'] += 1
                self._stats['allocations'] += 1
                
                logger.debug(f"Created new array: {shape}, {dtype}")
                return array
    
    def return_array(self, array: np.ndarray):
        """
        Return numpy array to pool.
        
        Args:
            array: Array to return to pool
        """
        if not isinstance(array, np.ndarray):
            return
        
        pool_key = (array.shape, array.dtype)
        
        with self._lock:
            if self._pool_sizes[pool_key] < self.max_pool_size:
                self._pools[pool_key].append(array)
                self._pool_sizes[pool_key] += 1
                self._stats['deallocations'] += 1
                
                logger.debug(f"Returned array to pool: {array.shape}, {array.dtype}")
            else:
                # Pool is full, let array be garbage collected
                logger.debug(f"Pool full, discarding array: {array.shape}, {array.dtype}")
    
    def clear_pool(self, pool_key: Optional[Tuple] = None):
        """Clear specific pool or all pools."""
        with self._lock:
            if pool_key:
                if pool_key in self._pools:
                    self._pools[pool_key].clear()
                    self._pool_sizes[pool_key] = 0
            else:
                self._pools.clear()
                self._pool_sizes.clear()
        
        logger.info("Memory pool cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory pool statistics."""
        with self._lock:
            total_pooled = sum(self._pool_sizes.values())
            pool_types = len(self._pools)
            
            return {
                **self._stats,
                'total_pooled_objects': total_pooled,
                'pool_types': pool_types,
                'hit_rate': self._stats['pool_hits'] / (self._stats['pool_hits'] + self._stats['pool_misses']) if (self._stats['pool_hits'] + self._stats['pool_misses']) > 0 else 0
            }


class ObjectTracker:
    """
    Track and manage object lifecycles for memory optimization.
    """
    
    def __init__(self):
        """Initialize object tracker."""
        self._tracked_objects = weakref.WeakSet()
        self._object_stats = defaultdict(int)
        self._creation_times = weakref.WeakKeyDictionary()
        self._lock = threading.RLock()
        
        logger.debug("Object tracker initialized")
    
    def track_object(self, obj: Any, obj_type: str = None):
        """
        Track an object for lifecycle management.
        
        Args:
            obj: Object to track
            obj_type: Optional type identifier
        """
        if obj_type is None:
            obj_type = type(obj).__name__
        
        with self._lock:
            self._tracked_objects.add(obj)
            self._object_stats[obj_type] += 1
            self._creation_times[obj] = time.time()
        
        logger.debug(f"Tracking object: {obj_type}")
    
    def untrack_object(self, obj: Any):
        """
        Untrack an object.
        
        Args:
            obj: Object to untrack
        """
        with self._lock:
            if obj in self._tracked_objects:
                self._tracked_objects.remove(obj)
            
            if obj in self._creation_times:
                del self._creation_times[obj]
    
    def get_tracked_count(self) -> int:
        """Get number of currently tracked objects."""
        return len(self._tracked_objects)
    
    def get_object_stats(self) -> Dict[str, int]:
        """Get object creation statistics by type."""
        with self._lock:
            return dict(self._object_stats)
    
    def find_long_lived_objects(self, min_age_seconds: float = 3600) -> List[Any]:
        """
        Find objects that have been alive for a long time.
        
        Args:
            min_age_seconds: Minimum age in seconds
            
        Returns:
            List of long-lived objects
        """
        current_time = time.time()
        long_lived = []
        
        with self._lock:
            for obj in self._tracked_objects:
                creation_time = self._creation_times.get(obj)
                if creation_time and (current_time - creation_time) > min_age_seconds:
                    long_lived.append(obj)
        
        return long_lived
    
    def cleanup_stale_references(self):
        """Clean up stale weak references."""
        with self._lock:
            # WeakSet automatically cleans up, but we can force it
            initial_count = len(self._tracked_objects)
            
            # Access all objects to trigger cleanup
            list(self._tracked_objects)
            
            final_count = len(self._tracked_objects)
            cleaned = initial_count - final_count
            
            if cleaned > 0:
                logger.debug(f"Cleaned up {cleaned} stale object references")


class MemoryManager:
    """
    Comprehensive memory management system.
    """
    
    def __init__(
        self,
        memory_limit_gb: Optional[float] = None,
        enable_gc_optimization: bool = True,
        gc_threshold_multiplier: float = 2.0,
        enable_object_tracking: bool = True
    ):
        """
        Initialize memory manager.
        
        Args:
            memory_limit_gb: Memory limit in GB (None for no limit)
            enable_gc_optimization: Enable garbage collection optimization
            gc_threshold_multiplier: Multiplier for GC thresholds
            enable_object_tracking: Enable object lifecycle tracking
        """
        self.memory_limit_gb = memory_limit_gb
        self.enable_gc_optimization = enable_gc_optimization
        self.enable_object_tracking = enable_object_tracking
        
        # Components
        self.memory_pool = MemoryPool()
        self.object_tracker = ObjectTracker() if enable_object_tracking else None
        
        # Memory monitoring
        self.process = psutil.Process()
        self._memory_history = deque(maxlen=100)
        self._peak_memory_mb = 0
        
        # GC optimization
        if enable_gc_optimization:
            self._optimize_garbage_collection(gc_threshold_multiplier)
        
        # Memory pressure handling
        self._memory_callbacks = []
        self._monitoring = False
        self._monitor_thread = None
        
        logger.info(f"Memory manager initialized (limit: {memory_limit_gb}GB)")
    
    def _optimize_garbage_collection(self, multiplier: float):
        """Optimize garbage collection settings."""
        # Get current thresholds
        thresholds = gc.get_threshold()
        
        # Increase thresholds to reduce GC frequency
        new_thresholds = tuple(int(t * multiplier) for t in thresholds)
        gc.set_threshold(*new_thresholds)
        
        logger.debug(f"GC thresholds adjusted: {thresholds} -> {new_thresholds}")
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage."""
        memory_info = self.process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,
            'vms_mb': memory_info.vms / 1024 / 1024,
            'percent': self.process.memory_percent(),
            'available_mb': psutil.virtual_memory().available / 1024 / 1024
        }
    
    def check_memory_pressure(self) -> bool:
        """Check if system is under memory pressure."""
        usage = self.get_memory_usage()
        
        # Check against limit
        if self.memory_limit_gb:
            if usage['rss_mb'] > self.memory_limit_gb * 1024:
                return True
        
        # Check system memory
        if usage['percent'] > 90:  # Over 90% of system memory
            return True
        
        # Check available memory
        if usage['available_mb'] < 500:  # Less than 500MB available
            return True
        
        return False
    
    def handle_memory_pressure(self):
        """Handle memory pressure by freeing resources."""
        logger.warning("Memory pressure detected, attempting to free resources")
        
        # Clear memory pools
        self.memory_pool.clear_pool()
        
        # Force garbage collection
        collected = gc.collect()
        logger.debug(f"Garbage collection freed {collected} objects")
        
        # Call registered callbacks
        for callback in self._memory_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Memory callback failed: {e}")
        
        # Clean up object tracker
        if self.object_tracker:
            self.object_tracker.cleanup_stale_references()
        
        # Log final usage
        usage = self.get_memory_usage()
        logger.info(f"After cleanup: {usage['rss_mb']:.1f}MB RSS, {usage['percent']:.1f}% system")
    
    def add_memory_callback(self, callback: Callable[[], None]):
        """
        Add callback to be called during memory pressure.
        
        Args:
            callback: Function to call when memory pressure is detected
        """
        self._memory_callbacks.append(callback)
    
    def start_monitoring(self, check_interval: float = 5.0):
        """
        Start memory monitoring.
        
        Args:
            check_interval: How often to check memory usage (seconds)
        """
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_memory,
            args=(check_interval,),
            daemon=True
        )
        self._monitor_thread.start()
        
        logger.info("Memory monitoring started")
    
    def stop_monitoring(self):
        """Stop memory monitoring."""
        self._monitoring = False
        
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
        
        logger.info("Memory monitoring stopped")
    
    def _monitor_memory(self, check_interval: float):
        """Monitor memory usage continuously."""
        while self._monitoring:
            try:
                usage = self.get_memory_usage()
                
                # Track peak memory
                if usage['rss_mb'] > self._peak_memory_mb:
                    self._peak_memory_mb = usage['rss_mb']
                
                # Add to history
                self._memory_history.append({
                    'timestamp': time.time(),
                    'rss_mb': usage['rss_mb'],
                    'percent': usage['percent']
                })
                
                # Check for memory pressure
                if self.check_memory_pressure():
                    self.handle_memory_pressure()
                
                time.sleep(check_interval)
                
            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
                time.sleep(1.0)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        current_usage = self.get_memory_usage()
        
        stats = {
            'current_usage': current_usage,
            'peak_memory_mb': self._peak_memory_mb,
            'memory_pool': self.memory_pool.get_stats(),
            'gc_stats': {
                'counts': gc.get_count(),
                'threshold': gc.get_threshold(),
                'stats': gc.get_stats() if hasattr(gc, 'get_stats') else None
            }
        }
        
        if self.object_tracker:
            stats['object_tracking'] = {
                'tracked_objects': self.object_tracker.get_tracked_count(),
                'object_stats': self.object_tracker.get_object_stats()
            }
        
        if self._memory_history:
            history_mb = [h['rss_mb'] for h in self._memory_history]
            stats['history'] = {
                'samples': len(self._memory_history),
                'min_mb': min(history_mb),
                'max_mb': max(history_mb),
                'avg_mb': sum(history_mb) / len(history_mb)
            }
        
        return stats
    
    def optimize_memory_usage(self):
        """Optimize memory usage proactively."""
        logger.info("Optimizing memory usage")
        
        # Force garbage collection
        before_usage = self.get_memory_usage()['rss_mb']
        
        # Multiple GC passes for thorough cleanup
        for generation in range(3):
            collected = gc.collect(generation)
            if collected > 0:
                logger.debug(f"GC generation {generation}: collected {collected} objects")
        
        after_usage = self.get_memory_usage()['rss_mb']
        freed_mb = before_usage - after_usage
        
        if freed_mb > 0:
            logger.info(f"Memory optimization freed {freed_mb:.1f}MB")
        
        # Clean up object tracker
        if self.object_tracker:
            self.object_tracker.cleanup_stale_references()
        
        # Clear old memory history
        if len(self._memory_history) > 50:
            self._memory_history = deque(list(self._memory_history)[-25:], maxlen=100)
    
    def create_memory_report(self) -> str:
        """Create detailed memory usage report."""
        stats = self.get_memory_stats()
        
        report_lines = [
            "Memory Usage Report",
            "=" * 50,
            f"Current RSS: {stats['current_usage']['rss_mb']:.1f} MB",
            f"Current VMS: {stats['current_usage']['vms_mb']:.1f} MB",
            f"System Usage: {stats['current_usage']['percent']:.1f}%",
            f"Peak Memory: {stats['peak_memory_mb']:.1f} MB",
            "",
            "Memory Pool:",
            f"  Pool hits: {stats['memory_pool']['pool_hits']}",
            f"  Pool misses: {stats['memory_pool']['pool_misses']}",
            f"  Hit rate: {stats['memory_pool']['hit_rate']:.2%}",
            f"  Pooled objects: {stats['memory_pool']['total_pooled_objects']}",
            "",
            "Garbage Collection:",
            f"  Current counts: {stats['gc_stats']['counts']}",
            f"  Thresholds: {stats['gc_stats']['threshold']}",
        ]
        
        if 'object_tracking' in stats:
            report_lines.extend([
                "",
                "Object Tracking:",
                f"  Tracked objects: {stats['object_tracking']['tracked_objects']}",
                "  Object types:",
            ])
            
            for obj_type, count in stats['object_tracking']['object_stats'].items():
                report_lines.append(f"    {obj_type}: {count}")
        
        if 'history' in stats:
            report_lines.extend([
                "",
                "Memory History:",
                f"  Samples: {stats['history']['samples']}",
                f"  Min: {stats['history']['min_mb']:.1f} MB",
                f"  Max: {stats['history']['max_mb']:.1f} MB",
                f"  Avg: {stats['history']['avg_mb']:.1f} MB",
            ])
        
        return "\n".join(report_lines)


class ResourcePoolManager:
    """
    Manager for pooling and reusing expensive quantum computing resources.
    """
    
    def __init__(self):
        """Initialize resource pool manager."""
        self.pools = {}
        self.memory_manager = MemoryManager()
        self._resource_factories = {}
        self._lock = threading.RLock()
        
        logger.info("Resource pool manager initialized")
    
    def register_resource_type(
        self, 
        resource_type: str, 
        factory_func: Callable[..., Any],
        max_pool_size: int = 10
    ):
        """
        Register a resource type for pooling.
        
        Args:
            resource_type: Name of resource type
            factory_func: Function to create new instances
            max_pool_size: Maximum pool size for this resource type
        """
        with self._lock:
            self.pools[resource_type] = deque(maxlen=max_pool_size)
            self._resource_factories[resource_type] = factory_func
        
        logger.debug(f"Registered resource type: {resource_type}")
    
    def get_resource(self, resource_type: str, *args, **kwargs) -> Any:
        """
        Get resource from pool or create new one.
        
        Args:
            resource_type: Type of resource to get
            *args, **kwargs: Arguments for resource creation
            
        Returns:
            Resource instance
        """
        if resource_type not in self.pools:
            raise ValueError(f"Unknown resource type: {resource_type}")
        
        with self._lock:
            pool = self.pools[resource_type]
            
            if pool:
                resource = pool.popleft()
                logger.debug(f"Reused resource from pool: {resource_type}")
                return resource
            else:
                # Create new resource
                factory = self._resource_factories[resource_type]
                resource = factory(*args, **kwargs)
                logger.debug(f"Created new resource: {resource_type}")
                return resource
    
    def return_resource(self, resource_type: str, resource: Any):
        """
        Return resource to pool.
        
        Args:
            resource_type: Type of resource
            resource: Resource instance to return
        """
        if resource_type not in self.pools:
            return
        
        with self._lock:
            pool = self.pools[resource_type]
            
            # Check if pool has space
            if len(pool) < pool.maxlen:
                # Reset resource if it has a reset method
                if hasattr(resource, 'reset'):
                    try:
                        resource.reset()
                    except Exception as e:
                        logger.warning(f"Failed to reset resource: {e}")
                        return  # Don't pool resource that failed to reset
                
                pool.append(resource)
                logger.debug(f"Returned resource to pool: {resource_type}")
    
    def clear_pools(self):
        """Clear all resource pools."""
        with self._lock:
            for pool in self.pools.values():
                pool.clear()
        
        logger.info("All resource pools cleared")
    
    def get_pool_stats(self) -> Dict[str, Dict[str, int]]:
        """Get statistics for all resource pools."""
        with self._lock:
            stats = {}
            
            for resource_type, pool in self.pools.items():
                stats[resource_type] = {
                    'current_size': len(pool),
                    'max_size': pool.maxlen
                }
        
        return stats
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.clear_pools()
        self.memory_manager.stop_monitoring()