"""
Parallel execution and batch processing for quantum circuits.
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
import asyncio
import concurrent.futures
import multiprocessing as mp
import threading
import time
import queue
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
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
from functools import partial
import psutil

from ..utils.logging_config import get_logger
from ..utils.diagnostics import HealthChecker

logger = get_logger(__name__)


# Import simple parallel processor to avoid complexity
from .parallel_simple import ParallelProcessor


class WorkerPool:
    """
    Managed pool of workers for parallel quantum circuit execution.
    """
    
    def __init__(
        self, 
        max_workers: Optional[int] = None,
        worker_type: str = "thread",  # "thread", "process", "async"
        resource_limits: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize worker pool.
        
        Args:
            max_workers: Maximum number of workers (auto-detect if None)
            worker_type: Type of workers to use
            resource_limits: Resource limits for workers
        """
        if max_workers is None:
            max_workers = min(8, (mp.cpu_count() or 1) + 4)
        
        self.max_workers = max_workers
        self.worker_type = worker_type
        self.resource_limits = resource_limits or {}
        
        # Initialize appropriate executor
        if worker_type == "thread":
            self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        elif worker_type == "process":
            self.executor = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)
        else:
            self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        
        # Initialize metrics
        self._task_count = 0
        self._completed_tasks = 0
        self._failed_tasks = 0
    
    def map(self, func: Callable, *iterables) -> List[Any]:
        """Map function over iterables in parallel."""
        futures = []
        results = []
        
        try:
            # Submit all tasks
            for args in zip(*iterables):
                future = self.executor.submit(func, *args)
                futures.append(future)
                self._task_count += 1
            
            # Collect results
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                    self._completed_tasks += 1
                except Exception as e:
                    logger.error(f"Task failed: {e}")
                    results.append(None)
                    self._failed_tasks += 1
            
            return results
            
        except Exception as e:
            logger.error(f"Error in parallel map: {e}")
            return []
    
    def shutdown(self, wait: bool = True):
        """Shutdown the worker pool."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=wait)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get worker pool statistics."""
        return {
            'max_workers': self.max_workers,
            'worker_type': self.worker_type,
            'total_tasks': self._task_count,
            'completed_tasks': self._completed_tasks,
            'failed_tasks': self._failed_tasks,
            'success_rate': self._completed_tasks / max(1, self._task_count)
        }
        
        # Resource monitoring
        self.health_checker = HealthChecker()
        self._active_tasks = 0
        self._completed_tasks = 0
        self._failed_tasks = 0
        self._start_time = time.time()
        
        logger.info(f"Initialized {worker_type} pool with {max_workers} workers")
    
    def submit(self, func: Callable, *args, **kwargs) -> concurrent.futures.Future:
        """Submit task to worker pool."""
        self._active_tasks += 1
        
        # Wrap function to track completion
        def wrapped_func(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                self._completed_tasks += 1
                return result
            except Exception as e:
                self._failed_tasks += 1
                logger.error(f"Task failed: {e}")
                raise
            finally:
                self._active_tasks -= 1
        
        return self.executor.submit(wrapped_func, *args, **kwargs)
    
    def map(self, func: Callable, iterable, timeout: Optional[float] = None):
        """Map function over iterable using worker pool."""
        return self.executor.map(func, iterable, timeout=timeout)
    
    def shutdown(self, wait: bool = True):
        """Shutdown worker pool."""
        logger.info(f"Shutting down worker pool ({self._completed_tasks} completed, {self._failed_tasks} failed)")
        self.executor.shutdown(wait=wait)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get worker pool statistics."""
        uptime = time.time() - self._start_time
        
        return {
            'max_workers': self.max_workers,
            'worker_type': self.worker_type,
            'active_tasks': self._active_tasks,
            'completed_tasks': self._completed_tasks,
            'failed_tasks': self._failed_tasks,
            'success_rate': self._completed_tasks / (self._completed_tasks + self._failed_tasks) if (self._completed_tasks + self._failed_tasks) > 0 else 1.0,
            'uptime': uptime,
            'tasks_per_second': (self._completed_tasks + self._failed_tasks) / uptime if uptime > 0 else 0
        }
    
    def is_healthy(self) -> bool:
        """Check if worker pool is healthy."""
        resource_usage = self.health_checker.get_resource_usage()
        
        # Check resource limits
        if self.resource_limits.get('max_memory_percent', 90) < resource_usage['memory_percent']:
            return False
        
        if self.resource_limits.get('max_cpu_percent', 95) < resource_usage['cpu_percent']:
            return False
        
        return True
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()


class ParallelExecutor:
    """
    High-level interface for parallel quantum circuit execution.
    """
    
    def __init__(
        self,
        max_workers: Optional[int] = None,
        enable_batching: bool = True,
        batch_size: int = 32,
        adaptive_batching: bool = True
    ):
        """
        Initialize parallel executor.
        
        Args:
            max_workers: Maximum number of parallel workers
            enable_batching: Whether to enable automatic batching
            batch_size: Default batch size
            adaptive_batching: Whether to adapt batch size based on performance
        """
        self.max_workers = max_workers or min(8, (mp.cpu_count() or 1) + 4)
        self.enable_batching = enable_batching
        self.batch_size = batch_size
        self.adaptive_batching = adaptive_batching
        
        # Multiple worker pools for different task types
        self.cpu_pool = WorkerPool(max_workers=self.max_workers, worker_type="thread")
        self.io_pool = WorkerPool(max_workers=self.max_workers * 2, worker_type="thread")
        
        # Adaptive batching state
        self._batch_performance_history = []
        self._optimal_batch_size = batch_size
        
        logger.info(f"Parallel executor initialized with {self.max_workers} workers")
    
    def execute_parallel(
        self,
        func: Callable,
        args_list: List[Tuple],
        use_batching: Optional[bool] = None,
        timeout: Optional[float] = None,
        task_type: str = "cpu"  # "cpu" or "io"
    ) -> List[Any]:
        """
        Execute function in parallel with multiple argument sets.
        
        Args:
            func: Function to execute
            args_list: List of argument tuples
            use_batching: Override batching setting
            timeout: Timeout for all tasks
            task_type: Type of task for worker pool selection
            
        Returns:
            List of results in order
        """
        if use_batching is None:
            use_batching = self.enable_batching
        
        # Select appropriate worker pool
        pool = self.io_pool if task_type == "io" else self.cpu_pool
        
        if not use_batching or len(args_list) <= self.batch_size:
            # Execute directly without batching
            return self._execute_direct(func, args_list, pool, timeout)
        else:
            # Execute with batching
            return self._execute_batched(func, args_list, pool, timeout)
    
    def _execute_direct(
        self, 
        func: Callable, 
        args_list: List[Tuple], 
        pool: WorkerPool,
        timeout: Optional[float]
    ) -> List[Any]:
        """Execute tasks directly without batching."""
        start_time = time.time()
        
        # Submit all tasks
        futures = []
        for args in args_list:
            future = pool.submit(func, *args)
            futures.append(future)
        
        # Collect results
        results = []
        for future in concurrent.futures.as_completed(futures, timeout=timeout):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Task execution failed: {e}")
                results.append(None)
        
        execution_time = time.time() - start_time
        logger.debug(f"Direct execution of {len(args_list)} tasks completed in {execution_time:.2f}s")
        
        return results
    
    def _execute_batched(
        self, 
        func: Callable, 
        args_list: List[Tuple], 
        pool: WorkerPool,
        timeout: Optional[float]
    ) -> List[Any]:
        """Execute tasks with batching."""
        batch_size = self._get_optimal_batch_size(len(args_list))
        batches = [args_list[i:i + batch_size] for i in range(0, len(args_list), batch_size)]
        
        logger.debug(f"Executing {len(args_list)} tasks in {len(batches)} batches of size {batch_size}")
        
        start_time = time.time()
        
        # Execute batches in parallel
        batch_func = partial(self._execute_batch, func)
        batch_futures = []
        
        for batch in batches:
            future = pool.submit(batch_func, batch)
            batch_futures.append(future)
        
        # Collect batch results
        all_results = []
        for future in concurrent.futures.as_completed(batch_futures, timeout=timeout):
            try:
                batch_results = future.result()
                all_results.extend(batch_results)
            except Exception as e:
                logger.error(f"Batch execution failed: {e}")
                # Add None results for failed batch
                all_results.extend([None] * batch_size)
        
        execution_time = time.time() - start_time
        
        # Update batch performance history for adaptive batching
        if self.adaptive_batching:
            self._update_batch_performance(batch_size, len(args_list), execution_time)
        
        logger.debug(f"Batched execution completed in {execution_time:.2f}s")
        
        return all_results
    
    def _execute_batch(self, func: Callable, batch_args: List[Tuple]) -> List[Any]:
        """Execute a single batch of tasks."""
        batch_results = []
        
        for args in batch_args:
            try:
                result = func(*args)
                batch_results.append(result)
            except Exception as e:
                logger.error(f"Task in batch failed: {e}")
                batch_results.append(None)
        
        return batch_results
    
    def _get_optimal_batch_size(self, total_tasks: int) -> int:
        """Get optimal batch size based on performance history."""
        if not self.adaptive_batching or not self._batch_performance_history:
            return min(self.batch_size, total_tasks)
        
        # Simple adaptive strategy: use best performing batch size from history
        return min(self._optimal_batch_size, total_tasks)
    
    def _update_batch_performance(self, batch_size: int, total_tasks: int, execution_time: float):
        """Update batch performance history."""
        performance = {
            'batch_size': batch_size,
            'total_tasks': total_tasks,
            'execution_time': execution_time,
            'tasks_per_second': total_tasks / execution_time if execution_time > 0 else 0
        }
        
        self._batch_performance_history.append(performance)
        
        # Keep only recent history
        if len(self._batch_performance_history) > 20:
            self._batch_performance_history = self._batch_performance_history[-20:]
        
        # Update optimal batch size
        if len(self._batch_performance_history) >= 3:
            best_performance = max(self._batch_performance_history, key=lambda x: x['tasks_per_second'])
            self._optimal_batch_size = best_performance['batch_size']
    
    def execute_async(
        self, 
        func: Callable, 
        args_list: List[Tuple],
        callback: Optional[Callable] = None
    ) -> List[concurrent.futures.Future]:
        """
        Execute function asynchronously with callback.
        
        Args:
            func: Function to execute
            args_list: List of argument tuples
            callback: Callback function for results
            
        Returns:
            List of futures
        """
        futures = []
        
        for args in args_list:
            future = self.cpu_pool.submit(func, *args)
            
            if callback:
                future.add_done_callback(lambda f: callback(f.result()) if not f.exception() else None)
            
            futures.append(future)
        
        return futures
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        cpu_stats = self.cpu_pool.get_stats()
        io_stats = self.io_pool.get_stats()
        
        return {
            'cpu_pool': cpu_stats,
            'io_pool': io_stats,
            'optimal_batch_size': self._optimal_batch_size,
            'batch_performance_history': self._batch_performance_history[-5:],  # Recent history
        }
    
    def shutdown(self):
        """Shutdown all worker pools."""
        logger.info("Shutting down parallel executor")
        self.cpu_pool.shutdown()
        self.io_pool.shutdown()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()


class BatchProcessor:
    """
    Intelligent batch processor for quantum circuit execution.
    """
    
    def __init__(
        self,
        batch_size: int = 32,
        max_queue_size: int = 1000,
        processing_timeout: float = 60.0,
        enable_prioritization: bool = True
    ):
        """
        Initialize batch processor.
        
        Args:
            batch_size: Default batch size
            max_queue_size: Maximum queue size
            processing_timeout: Timeout for batch processing
            enable_prioritization: Whether to enable task prioritization
        """
        self.batch_size = batch_size
        self.max_queue_size = max_queue_size
        self.processing_timeout = processing_timeout
        self.enable_prioritization = enable_prioritization
        
        # Processing queues
        if enable_prioritization:
            self.task_queue = queue.PriorityQueue(maxsize=max_queue_size)
        else:
            self.task_queue = queue.Queue(maxsize=max_queue_size)
        
        self.result_queue = queue.Queue()
        
        # Processing state
        self.executor = ParallelExecutor()
        self._processing = False
        self._worker_thread = None
        self._processed_batches = 0
        self._total_tasks = 0
        
        logger.info(f"Batch processor initialized with batch size {batch_size}")
    
    def submit_task(
        self, 
        func: Callable, 
        args: Tuple, 
        priority: int = 0,
        task_id: Optional[str] = None
    ) -> str:
        """
        Submit task for batch processing.
        
        Args:
            func: Function to execute
            args: Function arguments
            priority: Task priority (lower = higher priority)
            task_id: Optional task identifier
            
        Returns:
            Task ID
        """
        if task_id is None:
            task_id = f"task_{int(time.time() * 1000000)}"
        
        task = {
            'id': task_id,
            'func': func,
            'args': args,
            'priority': priority,
            'submitted_at': time.time()
        }
        
        if self.enable_prioritization:
            self.task_queue.put((priority, task))
        else:
            self.task_queue.put(task)
        
        logger.debug(f"Task {task_id} submitted with priority {priority}")
        
        return task_id
    
    def start_processing(self):
        """Start batch processing."""
        if self._processing:
            logger.warning("Batch processing already started")
            return
        
        self._processing = True
        self._worker_thread = threading.Thread(target=self._process_batches, daemon=True)
        self._worker_thread.start()
        
        logger.info("Batch processing started")
    
    def stop_processing(self):
        """Stop batch processing."""
        self._processing = False
        
        if self._worker_thread:
            self._worker_thread.join(timeout=5.0)
        
        logger.info("Batch processing stopped")
    
    def _process_batches(self):
        """Main batch processing loop."""
        while self._processing:
            try:
                batch = self._collect_batch()
                
                if batch:
                    self._execute_batch(batch)
                else:
                    # No tasks available, sleep briefly
                    time.sleep(0.1)
                    
            except Exception as e:
                logger.error(f"Batch processing error: {e}")
                time.sleep(1.0)  # Back off on error
    
    def _collect_batch(self) -> List[Dict[str, Any]]:
        """Collect tasks for batch processing."""
        batch = []
        
        # Try to collect full batch
        for _ in range(self.batch_size):
            try:
                if self.enable_prioritization:
                    priority, task = self.task_queue.get(timeout=0.1)
                else:
                    task = self.task_queue.get(timeout=0.1)
                
                batch.append(task)
                
            except queue.Empty:
                break
        
        # If we have some tasks but not a full batch, wait a bit more
        if batch and len(batch) < self.batch_size:
            time.sleep(0.05)  # Small delay to potentially collect more tasks
            
            for _ in range(self.batch_size - len(batch)):
                try:
                    if self.enable_prioritization:
                        priority, task = self.task_queue.get_nowait()
                    else:
                        task = self.task_queue.get_nowait()
                    
                    batch.append(task)
                    
                except queue.Empty:
                    break
        
        return batch
    
    def _execute_batch(self, batch: List[Dict[str, Any]]):
        """Execute a batch of tasks."""
        if not batch:
            return
        
        logger.debug(f"Processing batch of {len(batch)} tasks")
        
        start_time = time.time()
        
        # Group tasks by function for more efficient execution
        func_groups = {}
        for task in batch:
            func = task['func']
            if func not in func_groups:
                func_groups[func] = []
            func_groups[func].append(task)
        
        # Execute each function group
        for func, tasks in func_groups.items():
            try:
                args_list = [task['args'] for task in tasks]
                results = self.executor.execute_parallel(func, args_list)
                
                # Store results
                for task, result in zip(tasks, results):
                    result_item = {
                        'task_id': task['id'],
                        'result': result,
                        'completed_at': time.time(),
                        'processing_time': time.time() - task['submitted_at']
                    }
                    
                    self.result_queue.put(result_item)
                
            except Exception as e:
                logger.error(f"Batch execution failed for {func.__name__}: {e}")
                
                # Store error results
                for task in tasks:
                    error_result = {
                        'task_id': task['id'],
                        'result': None,
                        'error': str(e),
                        'completed_at': time.time(),
                        'processing_time': time.time() - task['submitted_at']
                    }
                    
                    self.result_queue.put(error_result)
        
        processing_time = time.time() - start_time
        self._processed_batches += 1
        self._total_tasks += len(batch)
        
        logger.debug(f"Batch processed in {processing_time:.2f}s")
    
    def get_result(self, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Get next available result."""
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_all_results(self, timeout: Optional[float] = None) -> List[Dict[str, Any]]:
        """Get all available results."""
        results = []
        
        start_time = time.time()
        
        while True:
            remaining_time = None
            if timeout is not None:
                elapsed = time.time() - start_time
                remaining_time = max(0, timeout - elapsed)
                
                if remaining_time <= 0:
                    break
            
            result = self.get_result(timeout=remaining_time)
            
            if result is None:
                break
            
            results.append(result)
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get batch processor statistics."""
        return {
            'batch_size': self.batch_size,
            'queue_size': self.task_queue.qsize(),
            'result_queue_size': self.result_queue.qsize(),
            'processed_batches': self._processed_batches,
            'total_tasks': self._total_tasks,
            'is_processing': self._processing,
            'executor_stats': self.executor.get_performance_stats()
        }
    
    def __enter__(self):
        self.start_processing()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_processing()
        self.executor.shutdown()