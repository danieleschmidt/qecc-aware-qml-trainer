"""
Distributed Quantum Computing Orchestrator.

This module provides distributed execution, auto-scaling, and load balancing
for quantum machine learning workloads across multiple nodes and backends.
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
import threading
import asyncio
import json
import logging
import hashlib
import uuid
from typing import Dict, List, Optional, Callable, Any, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
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
from enum import Enum
from abc import ABC, abstractmethod

try:
    from qiskit import QuantumCircuit
    QISKIT_AVAILABLE = True
except ImportError:
    from ..core.fallback_imports import QuantumCircuit
    QISKIT_AVAILABLE = False

from ..utils.logging_config import get_logger

logger = get_logger(__name__)

class NodeStatus(Enum):
    """Status of distributed nodes."""
    IDLE = "idle"
    BUSY = "busy"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"

class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class QuantumNode:
    """Distributed quantum computing node."""
    node_id: str
    node_type: str  # 'simulator', 'quantum_hardware', 'hybrid'
    capabilities: Dict[str, Any]
    max_qubits: int
    max_concurrent_jobs: int
    current_load: float = 0.0
    status: NodeStatus = NodeStatus.IDLE
    last_heartbeat: float = 0.0
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.performance_metrics:
            self.performance_metrics = {
                'avg_execution_time': 0.0,
                'success_rate': 1.0,
                'error_rate': 0.0,
                'throughput': 0.0
            }

@dataclass
class QuantumTask:
    """Distributed quantum task."""
    task_id: str
    circuit: QuantumCircuit
    parameters: Dict[str, Any]
    priority: TaskPriority
    created_time: float
    required_qubits: int
    estimated_runtime: float
    callback: Optional[Callable] = None
    dependencies: List[str] = field(default_factory=list)
    retry_count: int = 0
    max_retries: int = 3
    
    def __post_init__(self):
        if not self.task_id:
            self.task_id = str(uuid.uuid4())
        if not self.created_time:
            self.created_time = time.time()

@dataclass
class ScalingPolicy:
    """Auto-scaling policy configuration."""
    min_nodes: int = 1
    max_nodes: int = 10
    target_cpu_utilization: float = 70.0
    scale_up_threshold: float = 80.0
    scale_down_threshold: float = 30.0
    scale_up_cooldown: float = 300.0  # 5 minutes
    scale_down_cooldown: float = 600.0  # 10 minutes
    enable_predictive_scaling: bool = True

class LoadBalancer:
    """Intelligent load balancer for quantum tasks."""
    
    def __init__(self):
        self.node_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.task_history: deque = deque(maxlen=1000)
        
    def select_optimal_node(self, task: QuantumTask, available_nodes: List[QuantumNode]) -> Optional[QuantumNode]:
        """Select optimal node for task execution."""
        if not available_nodes:
            return None
        
        # Filter nodes by capabilities
        suitable_nodes = [
            node for node in available_nodes
            if (node.max_qubits >= task.required_qubits and 
                node.status == NodeStatus.IDLE and
                self._can_handle_task(node, task))
        ]
        
        if not suitable_nodes:
            return None
        
        # Score nodes based on multiple factors
        node_scores = []
        for node in suitable_nodes:
            score = self._calculate_node_score(node, task)
            node_scores.append((node, score))
        
        # Select node with highest score
        best_node = max(node_scores, key=lambda x: x[1])[0]
        return best_node
    
    def _can_handle_task(self, node: QuantumNode, task: QuantumTask) -> bool:
        """Check if node can handle the task."""
        # Check concurrent job limit
        if node.current_load >= node.max_concurrent_jobs:
            return False
        
        # Check task-specific requirements
        if task.parameters.get('backend_type') and task.parameters['backend_type'] != node.node_type:
            return False
        
        return True
    
    def _calculate_node_score(self, node: QuantumNode, task: QuantumTask) -> float:
        """Calculate node suitability score for task."""
        score = 0.0
        
        # Performance score (30%)
        performance_score = (
            node.performance_metrics['success_rate'] * 0.5 +
            (1.0 - node.performance_metrics['error_rate']) * 0.3 +
            min(1.0, node.performance_metrics['throughput'] / 10.0) * 0.2
        )
        score += performance_score * 0.3
        
        # Load score (25%) - prefer less loaded nodes
        load_score = 1.0 - (node.current_load / node.max_concurrent_jobs)
        score += load_score * 0.25
        
        # Capability match score (20%)
        capability_score = 1.0
        if task.required_qubits <= node.max_qubits // 2:
            capability_score = 1.0  # Good fit
        elif task.required_qubits <= node.max_qubits:
            capability_score = 0.7  # Acceptable fit
        else:
            capability_score = 0.0  # Can't handle
        score += capability_score * 0.2
        
        # Priority bonus (15%)
        priority_score = task.priority.value / 4.0
        score += priority_score * 0.15
        
        # Historical performance (10%)
        if node.node_id in self.node_metrics:
            recent_metrics = list(self.node_metrics[node.node_id])[-10:]
            if recent_metrics:
                avg_success = np.mean([m.get('success', 0) for m in recent_metrics])
                score += avg_success * 0.1
        
        return score
    
    def update_node_metrics(self, node_id: str, task_result: Dict[str, Any]) -> None:
        """Update node performance metrics."""
        metrics = {
            'success': 1.0 if task_result.get('success', False) else 0.0,
            'execution_time': task_result.get('execution_time', 0.0),
            'timestamp': time.time()
        }
        self.node_metrics[node_id].append(metrics)

class AutoScaler:
    """Automatic scaling for distributed quantum systems."""
    
    def __init__(self, policy: ScalingPolicy):
        self.policy = policy
        self.last_scale_up = 0.0
        self.last_scale_down = 0.0
        self.metrics_history: deque = deque(maxlen=100)
        
    def should_scale_up(self, current_metrics: Dict[str, float]) -> bool:
        """Determine if system should scale up."""
        current_time = time.time()
        
        # Check cooldown period
        if current_time - self.last_scale_up < self.policy.scale_up_cooldown:
            return False
        
        # Check utilization threshold
        cpu_utilization = current_metrics.get('avg_cpu_utilization', 0.0)
        queue_length = current_metrics.get('queue_length', 0)
        
        if (cpu_utilization > self.policy.scale_up_threshold or 
            queue_length > 10):  # Queue backlog threshold
            return True
        
        # Predictive scaling
        if self.policy.enable_predictive_scaling:
            predicted_load = self._predict_future_load()
            if predicted_load > self.policy.scale_up_threshold:
                return True
        
        return False
    
    def should_scale_down(self, current_metrics: Dict[str, float]) -> bool:
        """Determine if system should scale down."""
        current_time = time.time()
        
        # Check cooldown period
        if current_time - self.last_scale_down < self.policy.scale_down_cooldown:
            return False
        
        # Check utilization threshold
        cpu_utilization = current_metrics.get('avg_cpu_utilization', 0.0)
        
        if cpu_utilization < self.policy.scale_down_threshold:
            # Check if we have more than minimum nodes
            active_nodes = current_metrics.get('active_nodes', 1)
            if active_nodes > self.policy.min_nodes:
                return True
        
        return False
    
    def _predict_future_load(self) -> float:
        """Predict future load based on historical trends."""
        if len(self.metrics_history) < 10:
            return 0.0
        
        recent_loads = [m.get('avg_cpu_utilization', 0.0) for m in list(self.metrics_history)[-10:]]
        
        # Simple linear prediction
        if len(recent_loads) >= 2:
            trend = recent_loads[-1] - recent_loads[-2]
            predicted_load = recent_loads[-1] + trend * 3  # Predict 3 time periods ahead
            return max(0.0, min(100.0, predicted_load))
        
        return recent_loads[-1] if recent_loads else 0.0
    
    def record_scaling_action(self, action: str) -> None:
        """Record scaling action with timestamp."""
        current_time = time.time()
        if action == 'scale_up':
            self.last_scale_up = current_time
        elif action == 'scale_down':
            self.last_scale_down = current_time

class DistributedQuantumOrchestrator:
    """
    Distributed quantum computing orchestrator.
    
    Provides distributed execution, auto-scaling, load balancing, and
    fault tolerance for quantum machine learning workloads.
    """
    
    def __init__(self, scaling_policy: Optional[ScalingPolicy] = None):
        """Initialize distributed orchestrator."""
        self.logger = get_logger(__name__)
        
        # Core components
        self.nodes: Dict[str, QuantumNode] = {}
        self.task_queue: List[QuantumTask] = []
        self.active_tasks: Dict[str, QuantumTask] = {}
        self.completed_tasks: Dict[str, Dict[str, Any]] = {}
        
        # Scaling and load balancing
        self.scaling_policy = scaling_policy or ScalingPolicy()
        self.load_balancer = LoadBalancer()
        self.auto_scaler = AutoScaler(self.scaling_policy)
        
        # Threading
        self.task_scheduler_thread: Optional[threading.Thread] = None
        self.health_monitor_thread: Optional[threading.Thread] = None
        self.is_running = False
        
        # Metrics
        self.orchestrator_metrics = {
            'total_tasks_executed': 0,
            'total_nodes_spawned': 0,
            'average_task_latency': 0.0,
            'system_uptime': time.time()
        }
        
        # Execution pool
        self.executor = ThreadPoolExecutor(max_workers=20)
        
        self.logger.info("Distributed quantum orchestrator initialized")
    
    def start(self) -> None:
        """Start orchestrator services."""
        if self.is_running:
            self.logger.warning("Orchestrator already running")
            return
        
        self.is_running = True
        
        # Start background threads
        self.task_scheduler_thread = threading.Thread(target=self._task_scheduler_loop, daemon=True)
        self.health_monitor_thread = threading.Thread(target=self._health_monitor_loop, daemon=True)
        
        self.task_scheduler_thread.start()
        self.health_monitor_thread.start()
        
        # Initialize with minimum nodes
        self._ensure_minimum_nodes()
        
        self.logger.info("Distributed orchestrator started")
    
    def stop(self) -> None:
        """Stop orchestrator services."""
        self.is_running = False
        
        # Wait for threads to finish
        if self.task_scheduler_thread:
            self.task_scheduler_thread.join(timeout=5.0)
        if self.health_monitor_thread:
            self.health_monitor_thread.join(timeout=5.0)
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        self.logger.info("Distributed orchestrator stopped")
    
    def register_node(self, node: QuantumNode) -> None:
        """Register a new quantum computing node."""
        self.nodes[node.node_id] = node
        node.last_heartbeat = time.time()
        
        self.logger.info(f"Registered node: {node.node_id} ({node.node_type}, {node.max_qubits} qubits)")
    
    def submit_task(self, task: QuantumTask) -> str:
        """Submit task for distributed execution."""
        # Add to queue with priority sorting
        self.task_queue.append(task)
        self.task_queue.sort(key=lambda t: (t.priority.value, t.created_time), reverse=True)
        
        self.logger.info(f"Task submitted: {task.task_id} (priority: {task.priority.name})")
        return task.task_id
    
    def submit_circuit(
        self,
        circuit: QuantumCircuit,
        parameters: Optional[Dict[str, Any]] = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        callback: Optional[Callable] = None
    ) -> str:
        """Submit quantum circuit for execution."""
        task = QuantumTask(
            task_id=str(uuid.uuid4()),
            circuit=circuit,
            parameters=parameters or {},
            priority=priority,
            created_time=time.time(),
            required_qubits=getattr(circuit, 'num_qubits', 4),
            estimated_runtime=self._estimate_runtime(circuit),
            callback=callback
        )
        
        return self.submit_task(task)
    
    def _estimate_runtime(self, circuit: QuantumCircuit) -> float:
        """Estimate circuit execution runtime."""
        num_qubits = getattr(circuit, 'num_qubits', 4)
        circuit_depth = getattr(circuit, 'depth', lambda: 10)()
        
        # Simple estimation formula
        base_time = 0.1  # Base execution time
        qubit_factor = np.log2(num_qubits + 1)
        depth_factor = np.log10(circuit_depth + 1)
        
        estimated_time = base_time * qubit_factor * depth_factor
        return max(0.1, min(estimated_time, 30.0))  # Clamp between 0.1 and 30 seconds
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of submitted task."""
        if task_id in self.completed_tasks:
            return {'status': 'completed', 'result': self.completed_tasks[task_id]}
        elif task_id in self.active_tasks:
            return {'status': 'running', 'task': self.active_tasks[task_id]}
        else:
            # Check if in queue
            for task in self.task_queue:
                if task.task_id == task_id:
                    return {'status': 'queued', 'position': self.task_queue.index(task)}
        
        return None
    
    def _task_scheduler_loop(self) -> None:
        """Main task scheduling loop."""
        while self.is_running:
            try:
                self._schedule_tasks()
                self._check_scaling_needs()
                time.sleep(1.0)  # Check every second
                
            except Exception as e:
                self.logger.error(f"Task scheduler error: {e}")
                time.sleep(5.0)
    
    def _schedule_tasks(self) -> None:
        """Schedule tasks to available nodes."""
        if not self.task_queue:
            return
        
        # Get available nodes
        available_nodes = [
            node for node in self.nodes.values()
            if node.status == NodeStatus.IDLE and node.current_load < node.max_concurrent_jobs
        ]
        
        if not available_nodes:
            return
        
        # Schedule tasks
        tasks_to_remove = []
        for task in self.task_queue[:]:  # Copy to avoid modification during iteration
            # Check dependencies
            if self._dependencies_satisfied(task):
                # Select optimal node
                selected_node = self.load_balancer.select_optimal_node(task, available_nodes)
                
                if selected_node:
                    # Assign task to node
                    self._execute_task_on_node(task, selected_node)
                    tasks_to_remove.append(task)
                    
                    # Update node status
                    selected_node.current_load += 1
                    if selected_node.current_load >= selected_node.max_concurrent_jobs:
                        available_nodes.remove(selected_node)
        
        # Remove scheduled tasks from queue
        for task in tasks_to_remove:
            self.task_queue.remove(task)
    
    def _dependencies_satisfied(self, task: QuantumTask) -> bool:
        """Check if task dependencies are satisfied."""
        for dep_id in task.dependencies:
            if dep_id not in self.completed_tasks:
                return False
        return True
    
    def _execute_task_on_node(self, task: QuantumTask, node: QuantumNode) -> None:
        """Execute task on selected node."""
        self.active_tasks[task.task_id] = task
        
        # Submit to thread pool
        future = self.executor.submit(self._run_task, task, node)
        future.add_done_callback(lambda f: self._handle_task_completion(f, task, node))
        
        self.logger.info(f"Task {task.task_id} assigned to node {node.node_id}")
    
    def _run_task(self, task: QuantumTask, node: QuantumNode) -> Dict[str, Any]:
        """Execute task on node (simulated)."""
        start_time = time.time()
        
        try:
            # Simulate quantum circuit execution
            time.sleep(min(task.estimated_runtime, 0.1))  # Limit actual sleep time
            
            # Generate mock results
            num_qubits = task.required_qubits
            counts = {}
            for i in range(min(2**num_qubits, 16)):
                bitstring = format(i, f'0{num_qubits}b')
                counts[bitstring] = np.random.poisson(1024 // (2**num_qubits))
            
            execution_time = time.time() - start_time
            
            result = {
                'success': True,
                'counts': counts,
                'execution_time': execution_time,
                'node_id': node.node_id,
                'task_id': task.task_id,
                'timestamp': time.time()
            }
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Task execution error: {e}")
            
            return {
                'success': False,
                'error': str(e),
                'execution_time': execution_time,
                'node_id': node.node_id,
                'task_id': task.task_id,
                'timestamp': time.time()
            }
    
    def _handle_task_completion(self, future: Future, task: QuantumTask, node: QuantumNode) -> None:
        """Handle task completion."""
        try:
            result = future.result()
            
            # Update node metrics
            self.load_balancer.update_node_metrics(node.node_id, result)
            
            # Update node load
            node.current_load = max(0, node.current_load - 1)
            
            # Store result
            self.completed_tasks[task.task_id] = result
            
            # Remove from active tasks
            if task.task_id in self.active_tasks:
                del self.active_tasks[task.task_id]
            
            # Update metrics
            self.orchestrator_metrics['total_tasks_executed'] += 1
            
            # Call callback if provided
            if task.callback:
                try:
                    task.callback(result)
                except Exception as e:
                    self.logger.error(f"Task callback error: {e}")
            
            self.logger.info(f"Task completed: {task.task_id} on node {node.node_id}")
            
            # Handle failed tasks with retry
            if not result['success'] and task.retry_count < task.max_retries:
                task.retry_count += 1
                self.task_queue.append(task)
                self.logger.info(f"Retrying task {task.task_id} (attempt {task.retry_count + 1})")
            
        except Exception as e:
            self.logger.error(f"Task completion handling error: {e}")
            node.current_load = max(0, node.current_load - 1)
    
    def _health_monitor_loop(self) -> None:
        """Monitor node health and system metrics."""
        while self.is_running:
            try:
                self._update_node_health()
                self._update_system_metrics()
                time.sleep(10.0)  # Check every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Health monitor error: {e}")
                time.sleep(10.0)
    
    def _update_node_health(self) -> None:
        """Update node health status."""
        current_time = time.time()
        
        for node in self.nodes.values():
            # Check heartbeat
            if current_time - node.last_heartbeat > 60.0:  # 1 minute timeout
                if node.status != NodeStatus.OFFLINE:
                    node.status = NodeStatus.OFFLINE
                    self.logger.warning(f"Node {node.node_id} went offline")
            else:
                if node.status == NodeStatus.OFFLINE:
                    node.status = NodeStatus.IDLE
                    self.logger.info(f"Node {node.node_id} came back online")
            
            # Update heartbeat for active nodes (simulate)
            if node.status != NodeStatus.OFFLINE:
                node.last_heartbeat = current_time
    
    def _update_system_metrics(self) -> None:
        """Update system-wide metrics."""
        active_nodes = len([n for n in self.nodes.values() if n.status != NodeStatus.OFFLINE])
        queue_length = len(self.task_queue)
        
        # Calculate average CPU utilization
        if active_nodes > 0:
            total_load = sum(n.current_load for n in self.nodes.values() if n.status != NodeStatus.OFFLINE)
            total_capacity = sum(n.max_concurrent_jobs for n in self.nodes.values() if n.status != NodeStatus.OFFLINE)
            avg_cpu_utilization = (total_load / total_capacity) * 100 if total_capacity > 0 else 0
        else:
            avg_cpu_utilization = 0
        
        metrics = {
            'active_nodes': active_nodes,
            'queue_length': queue_length,
            'avg_cpu_utilization': avg_cpu_utilization,
            'total_nodes': len(self.nodes),
            'timestamp': time.time()
        }
        
        # Record for auto-scaler
        self.auto_scaler.metrics_history.append(metrics)
    
    def _check_scaling_needs(self) -> None:
        """Check if scaling is needed."""
        if not self.auto_scaler.metrics_history:
            return
        
        current_metrics = self.auto_scaler.metrics_history[-1]
        
        # Check scale up
        if self.auto_scaler.should_scale_up(current_metrics):
            if len(self.nodes) < self.scaling_policy.max_nodes:
                self._scale_up()
        
        # Check scale down
        elif self.auto_scaler.should_scale_down(current_metrics):
            if len(self.nodes) > self.scaling_policy.min_nodes:
                self._scale_down()
    
    def _scale_up(self) -> None:
        """Scale up by adding a new node."""
        new_node_id = f"auto_node_{len(self.nodes) + 1}"
        new_node = QuantumNode(
            node_id=new_node_id,
            node_type="simulator",
            capabilities={"simulation": True},
            max_qubits=16,
            max_concurrent_jobs=4
        )
        
        self.register_node(new_node)
        self.auto_scaler.record_scaling_action('scale_up')
        self.orchestrator_metrics['total_nodes_spawned'] += 1
        
        self.logger.info(f"Scaled up: Added node {new_node_id}")
    
    def _scale_down(self) -> None:
        """Scale down by removing an idle node."""
        # Find idle nodes
        idle_nodes = [n for n in self.nodes.values() if n.status == NodeStatus.IDLE and n.current_load == 0]
        
        if idle_nodes:
            # Remove least recently used node
            node_to_remove = min(idle_nodes, key=lambda n: n.last_heartbeat)
            node_to_remove.status = NodeStatus.OFFLINE
            
            self.auto_scaler.record_scaling_action('scale_down')
            self.logger.info(f"Scaled down: Removed node {node_to_remove.node_id}")
    
    def _ensure_minimum_nodes(self) -> None:
        """Ensure minimum number of nodes are available."""
        active_nodes = len([n for n in self.nodes.values() if n.status != NodeStatus.OFFLINE])
        
        while active_nodes < self.scaling_policy.min_nodes:
            self._scale_up()
            active_nodes += 1
    
    def get_orchestrator_status(self) -> Dict[str, Any]:
        """Get comprehensive orchestrator status."""
        active_nodes = [n for n in self.nodes.values() if n.status != NodeStatus.OFFLINE]
        
        return {
            'is_running': self.is_running,
            'total_nodes': len(self.nodes),
            'active_nodes': len(active_nodes),
            'queued_tasks': len(self.task_queue),
            'active_tasks': len(self.active_tasks),
            'completed_tasks': len(self.completed_tasks),
            'system_uptime': time.time() - self.orchestrator_metrics['system_uptime'],
            'total_tasks_executed': self.orchestrator_metrics['total_tasks_executed'],
            'nodes': [
                {
                    'node_id': node.node_id,
                    'type': node.node_type,
                    'status': node.status.value,
                    'load': node.current_load,
                    'max_jobs': node.max_concurrent_jobs,
                    'qubits': node.max_qubits
                }
                for node in self.nodes.values()
            ]
        }

# Global orchestrator instance
_global_orchestrator = None

def get_distributed_orchestrator(policy: Optional[ScalingPolicy] = None) -> DistributedQuantumOrchestrator:
    """Get global distributed orchestrator instance."""
    global _global_orchestrator
    if _global_orchestrator is None:
        _global_orchestrator = DistributedQuantumOrchestrator(policy)
    return _global_orchestrator