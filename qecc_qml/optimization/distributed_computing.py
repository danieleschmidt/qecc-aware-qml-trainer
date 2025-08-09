"""
Distributed computing and parallelization for QECC-aware QML systems.

Provides distributed training, parallel circuit execution,
and scalable computation across multiple workers.
"""

import time
import threading
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Any, Callable, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import queue
import socket
import pickle
import numpy as np
from pathlib import Path
import json


class ExecutionBackend(Enum):
    """Execution backend types."""
    THREAD = "thread"
    PROCESS = "process"
    DISTRIBUTED = "distributed"
    GPU = "gpu"
    HYBRID = "hybrid"


class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class ComputationTask:
    """Individual computation task."""
    task_id: str
    function: Callable
    args: tuple = ()
    kwargs: dict = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    timeout: Optional[float] = None
    retries: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    dependencies: List[str] = field(default_factory=list)


@dataclass
class TaskResult:
    """Task execution result."""
    task_id: str
    success: bool
    result: Any = None
    error: Optional[Exception] = None
    execution_time: float = 0.0
    worker_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    completed_at: float = field(default_factory=time.time)


class WorkerPool:
    """
    Intelligent worker pool for distributed quantum computations.
    
    Provides adaptive worker management, load balancing,
    and fault tolerance for quantum circuit execution.
    """
    
    def __init__(
        self,
        max_workers: Optional[int] = None,
        backend: ExecutionBackend = ExecutionBackend.THREAD,
        enable_monitoring: bool = True,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize worker pool.
        
        Args:
            max_workers: Maximum number of workers
            backend: Execution backend type
            enable_monitoring: Whether to enable worker monitoring
            logger: Optional logger instance
        """
        self.backend = backend
        self.enable_monitoring = enable_monitoring
        self.logger = logger or logging.getLogger(__name__)
        
        # Determine optimal worker count
        if max_workers is None:
            if backend == ExecutionBackend.THREAD:
                max_workers = min(32, (mp.cpu_count() or 1) + 4)
            elif backend == ExecutionBackend.PROCESS:
                max_workers = mp.cpu_count() or 1
            else:
                max_workers = 8
        
        self.max_workers = max_workers
        
        # Worker management
        self.workers: Dict[str, Any] = {}
        self.worker_stats: Dict[str, Dict[str, Any]] = {}
        self.active_tasks: Dict[str, ComputationTask] = {}
        
        # Task queues by priority
        self.task_queues = {
            priority: queue.PriorityQueue() for priority in TaskPriority
        }
        self.result_queue = queue.Queue()
        self.completed_tasks: Dict[str, TaskResult] = {}
        
        # Synchronization
        self.lock = threading.RLock()
        self.shutdown_event = threading.Event()
        
        # Performance metrics
        self.total_tasks_completed = 0
        self.total_execution_time = 0.0
        self.failed_tasks = 0
        
        # Initialize executor
        self._initialize_executor()
        
        # Start monitoring thread
        if self.enable_monitoring:
            self.monitor_thread = threading.Thread(target=self._monitoring_loop)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
    
    def _initialize_executor(self):
        """Initialize the appropriate executor."""
        if self.backend == ExecutionBackend.THREAD:
            self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        elif self.backend == ExecutionBackend.PROCESS:
            self.executor = ProcessPoolExecutor(max_workers=self.max_workers)
        else:
            # For distributed and other backends, use thread pool as base
            self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        self.logger.info(f"Initialized worker pool with {self.max_workers} {self.backend.value} workers")
    
    def submit_task(self, task: ComputationTask) -> str:
        """Submit a task for execution."""
        with self.lock:
            # Add to appropriate priority queue
            priority_value = -task.priority.value  # Negative for max priority queue
            self.task_queues[task.priority].put((priority_value, task.created_at, task))
            
            self.logger.debug(f"Submitted task {task.task_id} with priority {task.priority.name}")
            
            # Process tasks immediately
            self._process_pending_tasks()
            
            return task.task_id
    
    def _process_pending_tasks(self):
        """Process pending tasks from priority queues."""
        tasks_to_execute = []
        
        # Collect tasks from all priority queues (highest priority first)
        for priority in reversed(list(TaskPriority)):
            while not self.task_queues[priority].empty():
                try:
                    _, _, task = self.task_queues[priority].get_nowait()
                    
                    # Check dependencies
                    if self._dependencies_satisfied(task):
                        tasks_to_execute.append(task)
                    else:
                        # Put back in queue if dependencies not satisfied
                        priority_value = -task.priority.value
                        self.task_queues[priority].put((priority_value, task.created_at, task))
                        break
                        
                except queue.Empty:
                    break
        
        # Execute collected tasks
        for task in tasks_to_execute:
            self._execute_task(task)
    
    def _dependencies_satisfied(self, task: ComputationTask) -> bool:
        """Check if task dependencies are satisfied."""
        for dep_id in task.dependencies:
            if (dep_id not in self.completed_tasks or 
                not self.completed_tasks[dep_id].success):
                return False
        return True
    
    def _execute_task(self, task: ComputationTask):
        """Execute a single task."""
        with self.lock:
            self.active_tasks[task.task_id] = task
        
        # Submit to executor
        future = self.executor.submit(self._run_task, task)
        
        # Store future for monitoring
        self.workers[task.task_id] = {
            'future': future,
            'task': task,
            'start_time': time.time()
        }
    
    def _run_task(self, task: ComputationTask) -> TaskResult:
        """Run a task and return result."""
        start_time = time.time()
        worker_id = threading.current_thread().name
        
        try:
            # Execute task function
            result = task.function(*task.args, **task.kwargs)
            
            execution_time = time.time() - start_time
            
            task_result = TaskResult(
                task_id=task.task_id,
                success=True,
                result=result,
                execution_time=execution_time,
                worker_id=worker_id
            )
            
            self._handle_task_completion(task_result)
            return task_result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            task_result = TaskResult(
                task_id=task.task_id,
                success=False,
                error=e,
                execution_time=execution_time,
                worker_id=worker_id
            )
            
            self._handle_task_failure(task, task_result)
            return task_result
    
    def _handle_task_completion(self, result: TaskResult):
        """Handle successful task completion."""
        with self.lock:
            self.completed_tasks[result.task_id] = result
            
            # Remove from active tasks
            if result.task_id in self.active_tasks:
                del self.active_tasks[result.task_id]
            
            # Update statistics
            self.total_tasks_completed += 1
            self.total_execution_time += result.execution_time
            
            # Update worker statistics
            if result.worker_id:
                if result.worker_id not in self.worker_stats:
                    self.worker_stats[result.worker_id] = {
                        'tasks_completed': 0,
                        'total_time': 0.0,
                        'failures': 0
                    }
                
                self.worker_stats[result.worker_id]['tasks_completed'] += 1
                self.worker_stats[result.worker_id]['total_time'] += result.execution_time
        
        self.logger.debug(f"Task {result.task_id} completed successfully in {result.execution_time:.2f}s")
        
        # Process any dependent tasks
        self._process_pending_tasks()
    
    def _handle_task_failure(self, task: ComputationTask, result: TaskResult):
        """Handle task failure with retry logic."""
        with self.lock:
            # Check if we should retry
            if task.retries > 0:
                task.retries -= 1
                self.logger.warning(
                    f"Task {task.task_id} failed, retrying ({task.retries} attempts remaining)"
                )
                
                # Resubmit task
                priority_value = -task.priority.value
                self.task_queues[task.priority].put((priority_value, task.created_at, task))
            else:
                # No more retries, mark as failed
                self.completed_tasks[result.task_id] = result
                self.failed_tasks += 1
                
                self.logger.error(f"Task {task.task_id} failed permanently: {result.error}")
            
            # Remove from active tasks
            if result.task_id in self.active_tasks:
                del self.active_tasks[result.task_id]
            
            # Update worker statistics
            if result.worker_id and result.worker_id in self.worker_stats:
                self.worker_stats[result.worker_id]['failures'] += 1
    
    def get_result(self, task_id: str, timeout: Optional[float] = None) -> Optional[TaskResult]:
        """Get result for a specific task."""
        start_time = time.time()
        
        while True:
            with self.lock:
                if task_id in self.completed_tasks:
                    return self.completed_tasks[task_id]
            
            if timeout and (time.time() - start_time) > timeout:
                return None
            
            time.sleep(0.1)
    
    def wait_for_completion(self, task_ids: List[str], timeout: Optional[float] = None) -> Dict[str, TaskResult]:
        """Wait for multiple tasks to complete."""
        results = {}
        start_time = time.time()
        
        while len(results) < len(task_ids):
            with self.lock:
                for task_id in task_ids:
                    if task_id not in results and task_id in self.completed_tasks:
                        results[task_id] = self.completed_tasks[task_id]
            
            if timeout and (time.time() - start_time) > timeout:
                break
            
            time.sleep(0.1)
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get worker pool statistics."""
        with self.lock:
            avg_execution_time = (
                self.total_execution_time / self.total_tasks_completed 
                if self.total_tasks_completed > 0 else 0.0
            )
            
            # Calculate queue sizes
            queue_sizes = {
                priority.name: self.task_queues[priority].qsize()
                for priority in TaskPriority
            }
            
            return {
                'backend': self.backend.value,
                'max_workers': self.max_workers,
                'active_tasks': len(self.active_tasks),
                'completed_tasks': self.total_tasks_completed,
                'failed_tasks': self.failed_tasks,
                'success_rate': (
                    (self.total_tasks_completed - self.failed_tasks) / self.total_tasks_completed
                    if self.total_tasks_completed > 0 else 1.0
                ),
                'average_execution_time': avg_execution_time,
                'total_execution_time': self.total_execution_time,
                'queue_sizes': queue_sizes,
                'worker_stats': dict(self.worker_stats)
            }
    
    def _monitoring_loop(self):
        """Monitor worker health and performance."""
        while not self.shutdown_event.is_set():
            try:
                self._check_worker_health()
                self._optimize_worker_allocation()
                time.sleep(10)  # Check every 10 seconds
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
    
    def _check_worker_health(self):
        """Check health of active workers."""
        current_time = time.time()
        stalled_tasks = []
        
        with self.lock:
            for task_id, worker_info in self.workers.items():
                # Check for stalled tasks
                if current_time - worker_info['start_time'] > 600:  # 10 minutes
                    stalled_tasks.append(task_id)
        
        # Handle stalled tasks
        for task_id in stalled_tasks:
            self.logger.warning(f"Task {task_id} appears stalled, cancelling")
            self._cancel_task(task_id)
    
    def _cancel_task(self, task_id: str):
        """Cancel a running task."""
        with self.lock:
            if task_id in self.workers:
                worker_info = self.workers[task_id]
                future = worker_info['future']
                
                # Attempt to cancel
                if future.cancel():
                    self.logger.info(f"Cancelled task {task_id}")
                else:
                    self.logger.warning(f"Could not cancel task {task_id}")
                
                # Clean up
                del self.workers[task_id]
                
                if task_id in self.active_tasks:
                    del self.active_tasks[task_id]
    
    def _optimize_worker_allocation(self):
        """Optimize worker allocation based on performance."""
        if not self.worker_stats:
            return
        
        # Calculate worker efficiency
        worker_efficiency = {}
        for worker_id, stats in self.worker_stats.items():
            if stats['tasks_completed'] > 0:
                efficiency = stats['tasks_completed'] / (stats['total_time'] + 1)
                failure_rate = stats['failures'] / (stats['tasks_completed'] + stats['failures'])
                worker_efficiency[worker_id] = efficiency * (1 - failure_rate)
        
        if len(worker_efficiency) > 2:
            # Identify underperforming workers
            avg_efficiency = np.mean(list(worker_efficiency.values()))
            underperforming = [
                worker_id for worker_id, eff in worker_efficiency.items()
                if eff < avg_efficiency * 0.5
            ]
            
            if underperforming:
                self.logger.info(f"Identified underperforming workers: {underperforming}")
    
    def shutdown(self, wait: bool = True):
        """Shutdown the worker pool."""
        self.shutdown_event.set()
        
        if wait:
            # Wait for active tasks to complete
            timeout = 30  # 30 seconds timeout
            start_time = time.time()
            
            while self.active_tasks and (time.time() - start_time) < timeout:
                time.sleep(1)
        
        # Shutdown executor
        self.executor.shutdown(wait=wait)
        
        self.logger.info("Worker pool shutdown complete")


class DistributedTrainer:
    """
    Distributed training system for QECC-aware QML.
    
    Coordinates training across multiple workers with
    gradient aggregation and model synchronization.
    """
    
    def __init__(
        self,
        worker_pool: Optional[WorkerPool] = None,
        aggregation_strategy: str = "federated_averaging",
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize distributed trainer.
        
        Args:
            worker_pool: Worker pool for task execution
            aggregation_strategy: Gradient aggregation strategy
            logger: Optional logger instance
        """
        self.worker_pool = worker_pool or WorkerPool()
        self.aggregation_strategy = aggregation_strategy
        self.logger = logger or logging.getLogger(__name__)
        
        # Training state
        self.global_parameters: Optional[np.ndarray] = None
        self.worker_parameters: Dict[str, np.ndarray] = {}
        self.gradient_history: List[Dict[str, np.ndarray]] = []
        
        # Synchronization
        self.sync_lock = threading.RLock()
        
        self.logger.info(f"Initialized DistributedTrainer with {aggregation_strategy} aggregation")
    
    def distribute_training(
        self,
        training_function: Callable,
        data_partitions: List[Tuple[np.ndarray, np.ndarray]],
        initial_parameters: np.ndarray,
        epochs: int = 10,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Distribute training across workers.
        
        Args:
            training_function: Function to train on each worker
            data_partitions: List of (X, y) data partitions for each worker
            initial_parameters: Initial model parameters
            epochs: Number of training epochs
            **kwargs: Additional arguments for training function
            
        Returns:
            Training results
        """
        self.global_parameters = initial_parameters.copy()
        num_workers = len(data_partitions)
        
        training_history = {
            'global_loss': [],
            'worker_losses': {f'worker_{i}': [] for i in range(num_workers)},
            'parameter_updates': [],
            'synchronization_times': []
        }
        
        self.logger.info(f"Starting distributed training with {num_workers} workers for {epochs} epochs")
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Distribute current parameters to workers
            worker_tasks = []
            
            for worker_id, (X_partition, y_partition) in enumerate(data_partitions):
                task = ComputationTask(
                    task_id=f"train_epoch_{epoch}_worker_{worker_id}",
                    function=training_function,
                    args=(X_partition, y_partition, self.global_parameters.copy()),
                    kwargs=kwargs,
                    priority=TaskPriority.HIGH
                )
                
                task_id = self.worker_pool.submit_task(task)
                worker_tasks.append((worker_id, task_id))
            
            # Wait for all workers to complete
            task_ids = [task_id for _, task_id in worker_tasks]
            results = self.worker_pool.wait_for_completion(task_ids, timeout=300)  # 5 minutes timeout
            
            # Collect results and aggregate
            worker_parameters = {}
            worker_losses = {}
            
            for worker_id, task_id in worker_tasks:
                if task_id in results and results[task_id].success:
                    result_data = results[task_id].result
                    
                    if isinstance(result_data, dict):
                        worker_parameters[worker_id] = result_data.get('parameters')
                        worker_losses[worker_id] = result_data.get('loss', float('inf'))
                    else:
                        # Assume result is just parameters
                        worker_parameters[worker_id] = result_data
                        worker_losses[worker_id] = float('inf')
                else:
                    self.logger.warning(f"Worker {worker_id} failed for epoch {epoch}")
                    worker_parameters[worker_id] = self.global_parameters.copy()
                    worker_losses[worker_id] = float('inf')
            
            # Aggregate parameters
            self.global_parameters = self._aggregate_parameters(worker_parameters)
            
            # Record training history
            training_history['worker_losses'][f'worker_{worker_id}'].append(worker_losses.get(worker_id, float('inf')))
            training_history['global_loss'].append(np.mean(list(worker_losses.values())))
            training_history['parameter_updates'].append(np.linalg.norm(self.global_parameters))
            
            epoch_time = time.time() - epoch_start
            training_history['synchronization_times'].append(epoch_time)
            
            self.logger.info(
                f"Epoch {epoch + 1}/{epochs} completed in {epoch_time:.2f}s, "
                f"global loss: {training_history['global_loss'][-1]:.6f}"
            )
        
        return {
            'final_parameters': self.global_parameters,
            'training_history': training_history,
            'worker_pool_stats': self.worker_pool.get_statistics()
        }
    
    def _aggregate_parameters(self, worker_parameters: Dict[int, np.ndarray]) -> np.ndarray:
        """Aggregate parameters from workers."""
        if not worker_parameters:
            return self.global_parameters
        
        if self.aggregation_strategy == "federated_averaging":
            # Simple federated averaging
            parameter_arrays = list(worker_parameters.values())
            return np.mean(parameter_arrays, axis=0)
        
        elif self.aggregation_strategy == "weighted_averaging":
            # Weighted averaging (equal weights for now)
            parameter_arrays = list(worker_parameters.values())
            weights = np.ones(len(parameter_arrays)) / len(parameter_arrays)
            
            return np.average(parameter_arrays, axis=0, weights=weights)
        
        elif self.aggregation_strategy == "gradient_compression":
            # Gradient compression (simplified)
            parameter_arrays = list(worker_parameters.values())
            aggregated = np.mean(parameter_arrays, axis=0)
            
            # Simple compression: zero out small gradients
            gradient = aggregated - self.global_parameters
            threshold = np.percentile(np.abs(gradient), 90)  # Keep top 10%
            gradient[np.abs(gradient) < threshold] = 0
            
            return self.global_parameters + gradient
        
        else:
            # Default to simple averaging
            parameter_arrays = list(worker_parameters.values())
            return np.mean(parameter_arrays, axis=0)


class ParallelCircuitExecutor:
    """
    Parallel quantum circuit execution system.
    
    Executes quantum circuits in parallel across multiple backends
    with intelligent load balancing and resource management.
    """
    
    def __init__(
        self,
        backends: Optional[List[str]] = None,
        worker_pool: Optional[WorkerPool] = None,
        load_balancing: bool = True,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize parallel circuit executor.
        
        Args:
            backends: List of quantum backends to use
            worker_pool: Worker pool for parallel execution
            load_balancing: Whether to enable load balancing
            logger: Optional logger instance
        """
        self.backends = backends or ["aer_simulator"]
        self.worker_pool = worker_pool or WorkerPool(backend=ExecutionBackend.THREAD)
        self.load_balancing = load_balancing
        self.logger = logger or logging.getLogger(__name__)
        
        # Backend statistics
        self.backend_stats: Dict[str, Dict[str, Any]] = {
            backend: {
                'jobs_executed': 0,
                'total_time': 0.0,
                'failures': 0,
                'average_time': 0.0,
                'load_factor': 1.0
            }
            for backend in self.backends
        }
        
        # Circuit batching
        self.batch_size = 10
        self.max_batch_wait_time = 5.0  # seconds
        
        self.logger.info(f"Initialized ParallelCircuitExecutor with backends: {self.backends}")
    
    def execute_circuits_parallel(
        self,
        circuits: List[Any],
        parameters_list: Optional[List[np.ndarray]] = None,
        shots: int = 1024,
        max_concurrent: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute multiple circuits in parallel.
        
        Args:
            circuits: List of quantum circuits
            parameters_list: List of parameter arrays for each circuit
            shots: Number of measurement shots
            max_concurrent: Maximum concurrent executions
            
        Returns:
            List of execution results
        """
        if not circuits:
            return []
        
        if parameters_list is None:
            parameters_list = [None] * len(circuits)
        
        # Create execution tasks
        tasks = []
        for i, (circuit, parameters) in enumerate(zip(circuits, parameters_list)):
            # Select optimal backend
            backend = self._select_backend(circuit)
            
            task = ComputationTask(
                task_id=f"circuit_exec_{i}",
                function=self._execute_single_circuit,
                args=(circuit, parameters, backend, shots),
                priority=TaskPriority.NORMAL,
                metadata={'circuit_index': i, 'backend': backend}
            )
            
            tasks.append(task)
        
        # Submit tasks
        task_ids = []
        for task in tasks:
            task_id = self.worker_pool.submit_task(task)
            task_ids.append(task_id)
        
        # Wait for completion
        results = self.worker_pool.wait_for_completion(task_ids, timeout=600)  # 10 minutes
        
        # Process results in order
        ordered_results = []
        for i, task_id in enumerate(task_ids):
            if task_id in results and results[task_id].success:
                result_data = results[task_id].result
                ordered_results.append(result_data)
                
                # Update backend statistics
                backend = tasks[i].metadata['backend']
                self._update_backend_stats(backend, results[task_id].execution_time, success=True)
            else:
                # Handle failure
                error_result = {
                    'success': False,
                    'error': results[task_id].error if task_id in results else "Task not completed",
                    'counts': {},
                    'execution_time': 0.0
                }
                ordered_results.append(error_result)
                
                # Update failure statistics
                if task_id in results:
                    backend = tasks[i].metadata['backend']
                    self._update_backend_stats(backend, results[task_id].execution_time, success=False)
        
        return ordered_results
    
    def _execute_single_circuit(
        self,
        circuit: Any,
        parameters: Optional[np.ndarray],
        backend_name: str,
        shots: int
    ) -> Dict[str, Any]:
        """Execute a single quantum circuit."""
        start_time = time.time()
        
        try:
            # Import backend-specific execution
            if "aer" in backend_name or "simulator" in backend_name:
                result = self._execute_qiskit_circuit(circuit, parameters, backend_name, shots)
            else:
                result = self._execute_generic_circuit(circuit, parameters, backend_name, shots)
            
            execution_time = time.time() - start_time
            
            return {
                'success': True,
                'result': result,
                'execution_time': execution_time,
                'backend': backend_name,
                'shots': shots
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            return {
                'success': False,
                'error': str(e),
                'execution_time': execution_time,
                'backend': backend_name
            }
    
    def _execute_qiskit_circuit(
        self,
        circuit: Any,
        parameters: Optional[np.ndarray],
        backend_name: str,
        shots: int
    ) -> Dict[str, Any]:
        """Execute Qiskit circuit."""
        try:
            from qiskit_aer import AerSimulator
            from qiskit import transpile
            
            # Create backend
            if backend_name == "aer_simulator":
                backend = AerSimulator()
            else:
                # Try to import other simulators
                backend = AerSimulator(method='automatic')
            
            # Bind parameters if provided
            if parameters is not None and hasattr(circuit, 'bind_parameters'):
                bound_circuit = circuit.bind_parameters(parameters)
            else:
                bound_circuit = circuit
            
            # Transpile circuit
            transpiled_circuit = transpile(bound_circuit, backend)
            
            # Execute
            job = backend.run(transpiled_circuit, shots=shots)
            result = job.result()
            
            # Extract counts
            counts = result.get_counts(0) if result.get_counts() else {}
            
            return {
                'counts': counts,
                'metadata': result.to_dict() if hasattr(result, 'to_dict') else {}
            }
            
        except ImportError:
            raise RuntimeError(f"Qiskit not available for backend {backend_name}")
    
    def _execute_generic_circuit(
        self,
        circuit: Any,
        parameters: Optional[np.ndarray],
        backend_name: str,
        shots: int
    ) -> Dict[str, Any]:
        """Execute circuit with generic backend."""
        # Placeholder for other quantum frameworks
        return {
            'counts': {'0' * 4: shots},  # Dummy result
            'metadata': {'backend': backend_name}
        }
    
    def _select_backend(self, circuit: Any) -> str:
        """Select optimal backend for circuit execution."""
        if not self.load_balancing:
            return self.backends[0]
        
        # Simple load balancing based on backend statistics
        best_backend = min(
            self.backends,
            key=lambda b: self.backend_stats[b]['load_factor']
        )
        
        return best_backend
    
    def _update_backend_stats(self, backend: str, execution_time: float, success: bool):
        """Update backend performance statistics."""
        if backend not in self.backend_stats:
            return
        
        stats = self.backend_stats[backend]
        
        if success:
            stats['jobs_executed'] += 1
            stats['total_time'] += execution_time
            stats['average_time'] = stats['total_time'] / stats['jobs_executed']
        else:
            stats['failures'] += 1
        
        # Update load factor (higher means more loaded)
        total_jobs = stats['jobs_executed'] + stats['failures']
        failure_rate = stats['failures'] / total_jobs if total_jobs > 0 else 0
        
        # Load factor considers both average time and failure rate
        stats['load_factor'] = stats['average_time'] * (1 + failure_rate)
    
    def get_backend_statistics(self) -> Dict[str, Any]:
        """Get backend performance statistics."""
        return {
            'backends': dict(self.backend_stats),
            'worker_pool': self.worker_pool.get_statistics(),
            'load_balancing_enabled': self.load_balancing
        }
    
    def execute_circuits_batch(
        self,
        circuits: List[Any],
        batch_size: Optional[int] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Execute circuits in batches for better resource utilization."""
        if batch_size is None:
            batch_size = self.batch_size
        
        results = []
        
        for i in range(0, len(circuits), batch_size):
            batch_circuits = circuits[i:i + batch_size]
            batch_results = self.execute_circuits_parallel(batch_circuits, **kwargs)
            results.extend(batch_results)
        
        return results
    
    def optimize_execution_strategy(self, circuit_characteristics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Optimize execution strategy based on circuit characteristics."""
        if not circuit_characteristics:
            return {'strategy': 'default'}
        
        # Analyze circuit characteristics
        num_qubits = [c.get('num_qubits', 0) for c in circuit_characteristics]
        circuit_depths = [c.get('depth', 0) for c in circuit_characteristics]
        
        avg_qubits = np.mean(num_qubits) if num_qubits else 0
        avg_depth = np.mean(circuit_depths) if circuit_depths else 0
        
        # Recommend strategy based on characteristics
        if avg_qubits <= 10 and avg_depth <= 20:
            strategy = {
                'backend': 'aer_simulator',
                'batch_size': min(20, len(circuit_characteristics)),
                'max_concurrent': min(8, len(circuit_characteristics))
            }
        elif avg_qubits <= 20:
            strategy = {
                'backend': 'aer_simulator',
                'batch_size': min(10, len(circuit_characteristics)),
                'max_concurrent': min(4, len(circuit_characteristics))
            }
        else:
            strategy = {
                'backend': 'statevector_simulator',
                'batch_size': min(5, len(circuit_characteristics)),
                'max_concurrent': min(2, len(circuit_characteristics))
            }
        
        return strategy