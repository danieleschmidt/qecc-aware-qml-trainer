"""
Adaptive scaling and performance optimization for QECC-QML operations.
"""

import time
import threading
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from dataclasses import dataclass
import numpy as np
from queue import Queue, Empty
import psutil
import os

from ..utils.logging_config import get_logger
from ..monitoring.health_monitor import HealthMonitor

logger = get_logger(__name__)


@dataclass
class ResourceMetrics:
    """Resource utilization metrics."""
    cpu_usage: float
    memory_usage: float
    gpu_memory: float = 0.0
    network_io: float = 0.0
    disk_io: float = 0.0
    queue_length: int = 0
    response_time: float = 0.0


@dataclass
class ScalingDecision:
    """Auto-scaling decision."""
    action: str  # scale_up, scale_down, maintain
    target_workers: int
    reason: str
    confidence: float
    timestamp: float


class AdaptiveScaler:
    """
    Adaptive resource scaling based on workload and performance metrics.
    """
    
    def __init__(self,
                 min_workers: int = 1,
                 max_workers: int = None,
                 target_cpu_usage: float = 70.0,
                 target_memory_usage: float = 80.0,
                 scale_up_threshold: float = 85.0,
                 scale_down_threshold: float = 50.0,
                 monitoring_interval: float = 5.0,
                 scaling_cooldown: float = 30.0):
        
        self.min_workers = min_workers
        self.max_workers = max_workers or min(32, mp.cpu_count() * 2)
        self.target_cpu_usage = target_cpu_usage
        self.target_memory_usage = target_memory_usage
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.monitoring_interval = monitoring_interval
        self.scaling_cooldown = scaling_cooldown
        
        # Current state
        self.current_workers = min_workers
        self.last_scaling_time = 0.0
        self.scaling_history = []
        
        # Monitoring
        self.health_monitor = HealthMonitor()
        self.resource_history = []
        
        # Threading
        self.scaling_enabled = False
        self.scaling_thread = None
        self._lock = threading.Lock()
        
        # Performance prediction
        self.performance_predictor = PerformancePredictor()
        
    def start_adaptive_scaling(self):
        """Start adaptive scaling monitoring."""
        if self.scaling_enabled:
            return
            
        self.scaling_enabled = True
        self.health_monitor.start_monitoring()
        
        self.scaling_thread = threading.Thread(
            target=self._scaling_loop, 
            daemon=True
        )
        self.scaling_thread.start()
        
        logger.info(f"Adaptive scaling started with {self.current_workers} workers")
        
    def stop_adaptive_scaling(self):
        """Stop adaptive scaling."""
        self.scaling_enabled = False
        self.health_monitor.stop_monitoring()
        
        if self.scaling_thread:
            self.scaling_thread.join(timeout=5.0)
            
        logger.info("Adaptive scaling stopped")
        
    def _scaling_loop(self):
        """Main scaling decision loop."""
        while self.scaling_enabled:
            try:
                # Collect current metrics
                metrics = self._collect_resource_metrics()
                
                # Make scaling decision
                decision = self._make_scaling_decision(metrics)
                
                # Apply scaling decision
                if decision.action != "maintain":
                    self._apply_scaling_decision(decision)
                    
                # Store metrics for analysis
                with self._lock:
                    self.resource_history.append((time.time(), metrics))
                    if len(self.resource_history) > 1000:  # Limit history size
                        self.resource_history = self.resource_history[-500:]
                        
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Scaling loop error: {str(e)}")
                time.sleep(self.monitoring_interval)
                
    def _collect_resource_metrics(self) -> ResourceMetrics:
        """Collect current resource utilization metrics."""
        
        # System metrics
        cpu_usage = psutil.cpu_percent(interval=1)
        memory_usage = psutil.virtual_memory().percent
        
        # Network I/O
        net_io = psutil.net_io_counters()
        network_io = net_io.bytes_sent + net_io.bytes_recv
        
        # Disk I/O
        disk_io_counters = psutil.disk_io_counters()
        disk_io = disk_io_counters.read_bytes + disk_io_counters.write_bytes if disk_io_counters else 0
        
        # GPU metrics (if available)
        gpu_memory = self._get_gpu_memory_usage()
        
        return ResourceMetrics(
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            gpu_memory=gpu_memory,
            network_io=network_io,
            disk_io=disk_io,
            queue_length=0,  # Would be updated by task queue
            response_time=0.0  # Would be updated by performance monitoring
        )
        
    def _get_gpu_memory_usage(self) -> float:
        """Get GPU memory usage if available."""
        try:
            # This would integrate with actual GPU monitoring libraries
            # For now, return simulated value
            return 0.0
        except:
            return 0.0
            
    def _make_scaling_decision(self, metrics: ResourceMetrics) -> ScalingDecision:
        """Make intelligent scaling decision based on metrics."""
        
        current_time = time.time()
        
        # Check cooldown period
        if current_time - self.last_scaling_time < self.scaling_cooldown:
            return ScalingDecision(
                action="maintain",
                target_workers=self.current_workers,
                reason="Scaling cooldown active",
                confidence=1.0,
                timestamp=current_time
            )
            
        # Analyze trends
        trend_analysis = self._analyze_resource_trends()
        
        # Predict future resource needs
        prediction = self.performance_predictor.predict_resource_needs(
            metrics, self.current_workers
        )
        
        # Decision logic
        scale_up_signals = 0
        scale_down_signals = 0
        reasons = []
        
        # CPU-based decisions
        if metrics.cpu_usage > self.scale_up_threshold:
            scale_up_signals += 2
            reasons.append(f"High CPU usage: {metrics.cpu_usage:.1f}%")
        elif metrics.cpu_usage < self.scale_down_threshold:
            scale_down_signals += 1
            reasons.append(f"Low CPU usage: {metrics.cpu_usage:.1f}%")
            
        # Memory-based decisions
        if metrics.memory_usage > self.scale_up_threshold:
            scale_up_signals += 2
            reasons.append(f"High memory usage: {metrics.memory_usage:.1f}%")
        elif metrics.memory_usage < self.scale_down_threshold:
            scale_down_signals += 1
            reasons.append(f"Low memory usage: {metrics.memory_usage:.1f}%")
            
        # Queue length considerations
        if metrics.queue_length > self.current_workers * 2:
            scale_up_signals += 1
            reasons.append(f"High queue length: {metrics.queue_length}")
            
        # Response time considerations
        if metrics.response_time > 5.0:  # 5 second threshold
            scale_up_signals += 1
            reasons.append(f"High response time: {metrics.response_time:.2f}s")
            
        # Trend-based adjustments
        if trend_analysis.get("cpu_trend", 0) > 0.1:  # Growing trend
            scale_up_signals += 1
            reasons.append("Upward CPU trend detected")
        elif trend_analysis.get("cpu_trend", 0) < -0.1:  # Declining trend
            scale_down_signals += 1
            reasons.append("Downward CPU trend detected")
            
        # Prediction-based adjustments
        if prediction.get("recommended_workers", self.current_workers) > self.current_workers:
            scale_up_signals += 1
            reasons.append("Performance predictor suggests scale up")
        elif prediction.get("recommended_workers", self.current_workers) < self.current_workers:
            scale_down_signals += 1
            reasons.append("Performance predictor suggests scale down")
            
        # Make final decision
        if scale_up_signals > scale_down_signals and self.current_workers < self.max_workers:
            new_workers = min(self.current_workers + 1, self.max_workers)
            confidence = min(scale_up_signals / 5.0, 1.0)
            return ScalingDecision(
                action="scale_up",
                target_workers=new_workers,
                reason="; ".join(reasons),
                confidence=confidence,
                timestamp=current_time
            )
        elif scale_down_signals > scale_up_signals and self.current_workers > self.min_workers:
            new_workers = max(self.current_workers - 1, self.min_workers)
            confidence = min(scale_down_signals / 3.0, 1.0)
            return ScalingDecision(
                action="scale_down",
                target_workers=new_workers,
                reason="; ".join(reasons),
                confidence=confidence,
                timestamp=current_time
            )
        else:
            return ScalingDecision(
                action="maintain",
                target_workers=self.current_workers,
                reason="Balanced resource utilization",
                confidence=0.8,
                timestamp=current_time
            )
            
    def _analyze_resource_trends(self) -> Dict[str, float]:
        """Analyze resource utilization trends."""
        with self._lock:
            if len(self.resource_history) < 5:
                return {}
                
            recent_metrics = self.resource_history[-10:]  # Last 10 samples
            
        # Calculate trends
        timestamps = [m[0] for m in recent_metrics]
        cpu_values = [m[1].cpu_usage for m in recent_metrics]
        memory_values = [m[1].memory_usage for m in recent_metrics]
        
        cpu_trend = self._calculate_trend(timestamps, cpu_values)
        memory_trend = self._calculate_trend(timestamps, memory_values)
        
        return {
            "cpu_trend": cpu_trend,
            "memory_trend": memory_trend,
            "sample_count": len(recent_metrics)
        }
        
    def _calculate_trend(self, x: List[float], y: List[float]) -> float:
        """Calculate linear trend (slope)."""
        if len(x) < 2:
            return 0.0
            
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(x[i] ** 2 for i in range(n))
        
        denominator = n * sum_x2 - sum_x ** 2
        if denominator == 0:
            return 0.0
            
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        return slope
        
    def _apply_scaling_decision(self, decision: ScalingDecision):
        """Apply scaling decision."""
        with self._lock:
            old_workers = self.current_workers
            self.current_workers = decision.target_workers
            self.last_scaling_time = decision.timestamp
            self.scaling_history.append(decision)
            
            # Limit history size
            if len(self.scaling_history) > 100:
                self.scaling_history = self.scaling_history[-50:]
                
        logger.info(f"Scaling {decision.action}: {old_workers} -> {decision.target_workers} "
                   f"workers (confidence: {decision.confidence:.2f})")
        logger.info(f"Reason: {decision.reason}")
        
        # Here would be the actual implementation to scale worker pools
        # self._scale_worker_pool(decision.target_workers)
        
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get current scaling status and metrics."""
        with self._lock:
            recent_history = self.scaling_history[-10:] if self.scaling_history else []
            
        current_metrics = self._collect_resource_metrics()
        
        return {
            "current_workers": self.current_workers,
            "min_workers": self.min_workers,
            "max_workers": self.max_workers,
            "scaling_enabled": self.scaling_enabled,
            "last_scaling": self.last_scaling_time,
            "current_metrics": current_metrics.__dict__,
            "recent_decisions": [
                {
                    "action": d.action,
                    "workers": d.target_workers,
                    "reason": d.reason,
                    "confidence": d.confidence,
                    "timestamp": d.timestamp
                } for d in recent_history
            ]
        }


class PerformancePredictor:
    """
    Predicts future performance and resource needs.
    """
    
    def __init__(self):
        self.performance_history = []
        
    def predict_resource_needs(self, 
                             current_metrics: ResourceMetrics,
                             current_workers: int) -> Dict[str, Any]:
        """Predict future resource needs based on current state."""
        
        # Simple heuristic-based prediction
        # In practice, this could use ML models
        
        cpu_factor = current_metrics.cpu_usage / 100.0
        memory_factor = current_metrics.memory_usage / 100.0
        
        # Calculate efficiency score
        efficiency = self._calculate_efficiency(current_metrics, current_workers)
        
        # Predict optimal worker count
        if efficiency < 0.7:  # Low efficiency
            if cpu_factor > 0.8 or memory_factor > 0.8:
                recommended_workers = min(current_workers + 2, 32)
            else:
                recommended_workers = current_workers + 1
        elif efficiency > 0.9:  # High efficiency
            if cpu_factor < 0.3 and memory_factor < 0.3:
                recommended_workers = max(current_workers - 1, 1)
            else:
                recommended_workers = current_workers
        else:
            recommended_workers = current_workers
            
        return {
            "recommended_workers": recommended_workers,
            "efficiency_score": efficiency,
            "cpu_utilization_optimal": 0.6 <= cpu_factor <= 0.8,
            "memory_utilization_optimal": 0.6 <= memory_factor <= 0.8,
            "predicted_improvement": abs(efficiency - 0.8) * 0.1
        }
        
    def _calculate_efficiency(self, metrics: ResourceMetrics, workers: int) -> float:
        """Calculate current resource efficiency."""
        
        # Normalized efficiency based on resource utilization
        cpu_efficiency = 1.0 - abs(metrics.cpu_usage / 100.0 - 0.7)  # Target 70% CPU
        memory_efficiency = 1.0 - abs(metrics.memory_usage / 100.0 - 0.7)  # Target 70% memory
        
        # Worker utilization efficiency
        worker_efficiency = min(1.0, workers * 20.0 / max(1.0, metrics.cpu_usage))
        
        # Combined efficiency score
        efficiency = (cpu_efficiency + memory_efficiency + worker_efficiency) / 3.0
        return max(0.0, min(1.0, efficiency))


class LoadBalancer:
    """
    Intelligent load balancing for quantum computing tasks.
    """
    
    def __init__(self, workers: List[str] = None):
        self.workers = workers or []
        self.worker_loads = {worker: 0 for worker in self.workers}
        self.worker_performance = {worker: 1.0 for worker in self.workers}  # Performance scores
        self.task_history = []
        self._lock = threading.Lock()
        
    def add_worker(self, worker_id: str):
        """Add a new worker to the pool."""
        with self._lock:
            if worker_id not in self.workers:
                self.workers.append(worker_id)
                self.worker_loads[worker_id] = 0
                self.worker_performance[worker_id] = 1.0
                logger.info(f"Added worker: {worker_id}")
                
    def remove_worker(self, worker_id: str):
        """Remove a worker from the pool."""
        with self._lock:
            if worker_id in self.workers:
                self.workers.remove(worker_id)
                del self.worker_loads[worker_id]
                del self.worker_performance[worker_id]
                logger.info(f"Removed worker: {worker_id}")
                
    def select_worker(self, task_complexity: float = 1.0) -> Optional[str]:
        """Select optimal worker for task based on load and performance."""
        with self._lock:
            if not self.workers:
                return None
                
            # Calculate worker scores
            worker_scores = {}
            for worker in self.workers:
                load_factor = 1.0 / (1.0 + self.worker_loads[worker])
                performance_factor = self.worker_performance[worker]
                complexity_factor = 1.0 / max(0.1, task_complexity)  # Prefer high-perf for complex tasks
                
                score = load_factor * performance_factor * complexity_factor
                worker_scores[worker] = score
                
            # Select worker with highest score
            best_worker = max(worker_scores.keys(), key=lambda w: worker_scores[w])
            
            # Update load
            self.worker_loads[best_worker] += task_complexity
            
            return best_worker
            
    def task_completed(self, worker_id: str, task_complexity: float, execution_time: float):
        """Update worker statistics after task completion."""
        with self._lock:
            # Update load
            if worker_id in self.worker_loads:
                self.worker_loads[worker_id] = max(0, 
                    self.worker_loads[worker_id] - task_complexity
                )
                
            # Update performance score
            if worker_id in self.worker_performance:
                # Simple performance update based on execution time
                expected_time = task_complexity * 1.0  # 1 second per unit complexity
                performance_ratio = expected_time / max(0.1, execution_time)
                
                # Exponential moving average
                alpha = 0.1
                self.worker_performance[worker_id] = (
                    alpha * performance_ratio + 
                    (1 - alpha) * self.worker_performance[worker_id]
                )
                
            # Record task history
            self.task_history.append({
                "worker": worker_id,
                "complexity": task_complexity,
                "execution_time": execution_time,
                "timestamp": time.time()
            })
            
            # Limit history size
            if len(self.task_history) > 1000:
                self.task_history = self.task_history[-500:]
                
    def get_load_status(self) -> Dict[str, Any]:
        """Get current load balancing status."""
        with self._lock:
            total_load = sum(self.worker_loads.values())
            
            return {
                "total_workers": len(self.workers),
                "total_load": total_load,
                "average_load": total_load / max(1, len(self.workers)),
                "worker_loads": self.worker_loads.copy(),
                "worker_performance": self.worker_performance.copy(),
                "tasks_completed": len(self.task_history),
                "average_execution_time": self._calculate_average_execution_time()
            }
            
    def _calculate_average_execution_time(self) -> float:
        """Calculate average task execution time."""
        if not self.task_history:
            return 0.0
            
        recent_tasks = self.task_history[-100:]  # Last 100 tasks
        total_time = sum(task["execution_time"] for task in recent_tasks)
        return total_time / len(recent_tasks)


class PerformanceOptimizer:
    """
    Comprehensive performance optimization for QECC-QML operations.
    """
    
    def __init__(self):
        self.adaptive_scaler = AdaptiveScaler()
        self.load_balancer = LoadBalancer()
        self.performance_metrics = {}
        self.optimization_enabled = False
        
    def start_optimization(self):
        """Start comprehensive performance optimization."""
        self.optimization_enabled = True
        self.adaptive_scaler.start_adaptive_scaling()
        logger.info("Performance optimization started")
        
    def stop_optimization(self):
        """Stop performance optimization."""
        self.optimization_enabled = False
        self.adaptive_scaler.stop_adaptive_scaling()
        logger.info("Performance optimization stopped")
        
    def optimize_task_execution(self, task_func: Callable, 
                              task_complexity: float = 1.0,
                              **kwargs) -> Any:
        """Execute task with optimal resource allocation."""
        
        if not self.optimization_enabled:
            return task_func(**kwargs)
            
        # Select optimal worker
        worker_id = self.load_balancer.select_worker(task_complexity)
        
        start_time = time.time()
        
        try:
            # Execute task (would be distributed in real implementation)
            result = task_func(**kwargs)
            
            execution_time = time.time() - start_time
            
            # Update performance metrics
            if worker_id:
                self.load_balancer.task_completed(worker_id, task_complexity, execution_time)
                
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            if worker_id:
                self.load_balancer.task_completed(worker_id, task_complexity, execution_time)
            raise e
            
    def get_optimization_status(self) -> Dict[str, Any]:
        """Get comprehensive optimization status."""
        return {
            "optimization_enabled": self.optimization_enabled,
            "adaptive_scaling": self.adaptive_scaler.get_scaling_status(),
            "load_balancing": self.load_balancer.get_load_status(),
            "timestamp": time.time()
        }