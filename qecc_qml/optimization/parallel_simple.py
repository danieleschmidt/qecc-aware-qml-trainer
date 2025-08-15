"""
Simple parallel processing for quantum circuits.
"""

import concurrent.futures
import multiprocessing as mp
from typing import Any, Callable, List, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class ParallelProcessor:
    """Simple parallel processor for quantum computations."""
    
    def __init__(self, max_workers: Optional[int] = None):
        """Initialize parallel processor."""
        if max_workers is None:
            max_workers = min(8, (mp.cpu_count() or 1) + 4)
        
        self.max_workers = max_workers
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
    
    def process_batch(self, func: Callable, tasks: List[Any]) -> List[Any]:
        """Process a batch of tasks in parallel."""
        futures = []
        results = []
        
        try:
            # Submit all tasks
            for task in tasks:
                if isinstance(task, (list, tuple)):
                    future = self.executor.submit(func, *task)
                else:
                    future = self.executor.submit(func, task)
                futures.append(future)
            
            # Collect results
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Task failed: {e}")
                    results.append(None)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in parallel processing: {e}")
            return []
    
    def shutdown(self):
        """Shutdown the parallel processor."""
        self.executor.shutdown(wait=True)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()


class WorkerPool:
    """Simple worker pool wrapper."""
    
    def __init__(self, max_workers: Optional[int] = None, worker_type: str = "thread"):
        """Initialize worker pool."""
        if max_workers is None:
            max_workers = min(8, (mp.cpu_count() or 1) + 4)
        
        self.max_workers = max_workers
        self.worker_type = worker_type
        
        if worker_type == "thread":
            self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        elif worker_type == "process":
            self.executor = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)
        else:
            self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
    
    def map(self, func: Callable, iterable) -> List[Any]:
        """Map function over iterable using worker pool."""
        return list(self.executor.map(func, iterable))
    
    def shutdown(self, wait: bool = True):
        """Shutdown worker pool."""
        self.executor.shutdown(wait=wait)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get worker pool statistics."""
        return {
            'max_workers': self.max_workers,
            'worker_type': self.worker_type
        }