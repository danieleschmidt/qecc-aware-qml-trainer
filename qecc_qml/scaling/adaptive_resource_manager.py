"""
Adaptive Resource Manager for Quantum-Classical Hybrid Systems
"""

import psutil
import time
import threading
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import json
import warnings

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    warnings.warn("GPUtil not available. GPU monitoring disabled.")

try:
    from kubernetes import client, config
    KUBERNETES_AVAILABLE = True
except ImportError:
    KUBERNETES_AVAILABLE = False
    warnings.warn("Kubernetes client not available. K8s scaling disabled.")


class ResourceType(Enum):
    """Types of computational resources."""
    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"
    QUANTUM_QPU = "quantum_qpu"
    NETWORK = "network"
    STORAGE = "storage"


class ScalingAction(Enum):
    """Types of scaling actions."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    SCALE_OUT = "scale_out"
    SCALE_IN = "scale_in"
    OPTIMIZE = "optimize"
    MIGRATE = "migrate"


@dataclass
class ResourceMetrics:
    """Resource utilization metrics."""
    resource_type: ResourceType
    current_usage: float  # 0.0 to 1.0
    average_usage: float
    peak_usage: float
    available_capacity: float
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScalingDecision:
    """Represents a scaling decision."""
    action: ScalingAction
    resource_type: ResourceType
    target_scale: float
    reasoning: str
    priority: int
    estimated_impact: float
    cost_estimate: float
    timestamp: float


class AdaptiveResourceManager:
    """
    Intelligent resource management system that dynamically adapts to workload
    demands and optimizes resource allocation across quantum-classical hybrid systems.
    
    Features:
    - Real-time resource monitoring (CPU, GPU, memory, quantum resources)
    - Predictive scaling based on workload patterns
    - Cost-aware resource optimization
    - Kubernetes integration for container orchestration
    - Quantum resource scheduling and allocation
    - Multi-objective optimization (performance vs cost vs energy)
    """
    
    def __init__(
        self,
        monitoring_interval: float = 30.0,
        prediction_horizon: int = 10,
        scaling_threshold_up: float = 0.8,
        scaling_threshold_down: float = 0.3,
        enable_predictive_scaling: bool = True,
        enable_cost_optimization: bool = True
    ):
        self.monitoring_interval = monitoring_interval
        self.prediction_horizon = prediction_horizon
        self.scaling_threshold_up = scaling_threshold_up
        self.scaling_threshold_down = scaling_threshold_down
        self.enable_predictive_scaling = enable_predictive_scaling
        self.enable_cost_optimization = enable_cost_optimization
        
        # Resource tracking
        self.resource_metrics: Dict[ResourceType, deque] = {
            resource_type: deque(maxlen=1000) for resource_type in ResourceType
        }
        self.current_resources: Dict[ResourceType, ResourceMetrics] = {}
        
        # Scaling decisions and history
        self.scaling_history: List[ScalingDecision] = []
        self.pending_scaling_actions: List[ScalingDecision] = []
        
        # Workload prediction
        self.workload_patterns: Dict[str, List[float]] = defaultdict(list)
        self.prediction_models: Dict[ResourceType, Any] = {}
        
        # Cost tracking
        self.resource_costs: Dict[ResourceType, float] = {
            ResourceType.CPU: 0.05,  # $ per CPU-hour
            ResourceType.MEMORY: 0.01,  # $ per GB-hour
            ResourceType.GPU: 0.50,  # $ per GPU-hour
            ResourceType.QUANTUM_QPU: 5.00,  # $ per QPU-hour
            ResourceType.NETWORK: 0.001,  # $ per GB transferred
            ResourceType.STORAGE: 0.001  # $ per GB-hour
        }
        
        # Monitoring control
        self._monitoring_active = False
        self._monitoring_thread = None
        self._scaling_thread = None
        
        # Kubernetes integration
        self.k8s_enabled = KUBERNETES_AVAILABLE
        if self.k8s_enabled:
            try:
                config.load_incluster_config()  # Try in-cluster config first
            except:
                try:
                    config.load_kube_config()  # Fall back to local config
                except:
                    self.k8s_enabled = False
                    warnings.warn("Kubernetes config not found. K8s features disabled.")
        
        # Resource pool management
        self.resource_pools: Dict[str, Dict[str, Any]] = {}
        self.allocation_strategy = "balanced"  # balanced, performance, cost
        
        # Logger
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging configuration."""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def start_monitoring(self):
        """Start resource monitoring and adaptive scaling."""
        if self._monitoring_active:
            self.logger.warning("Resource monitoring already active")
            return
        
        self._monitoring_active = True
        
        # Start monitoring thread
        self._monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self._monitoring_thread.daemon = True
        self._monitoring_thread.start()
        
        # Start scaling thread
        self._scaling_thread = threading.Thread(target=self._scaling_loop)
        self._scaling_thread.daemon = True
        self._scaling_thread.start()
        
        self.logger.info("Adaptive resource monitoring started")
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self._monitoring_active = False
        
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self._monitoring_thread.join(timeout=5.0)
        
        if self._scaling_thread and self._scaling_thread.is_alive():
            self._scaling_thread.join(timeout=5.0)
        
        self.logger.info("Resource monitoring stopped")
    
    def _monitoring_loop(self):
        """Main resource monitoring loop."""
        while self._monitoring_active:
            try:
                # Collect resource metrics
                self._collect_resource_metrics()
                
                # Update workload patterns
                self._update_workload_patterns()
                
                # Generate predictions if enabled
                if self.enable_predictive_scaling:
                    self._update_predictions()
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitoring_interval)
    
    def _scaling_loop(self):
        """Main scaling decision loop."""
        while self._monitoring_active:
            try:
                # Analyze current resource utilization
                scaling_decisions = self._analyze_scaling_needs()
                
                # Execute scaling decisions
                for decision in scaling_decisions:
                    self._execute_scaling_decision(decision)
                
                # Clean up completed scaling actions
                self._cleanup_scaling_actions()
                
                time.sleep(self.monitoring_interval * 2)  # Less frequent than monitoring
                
            except Exception as e:
                self.logger.error(f"Error in scaling loop: {e}")
                time.sleep(self.monitoring_interval)
    
    def _collect_resource_metrics(self):
        """Collect comprehensive resource metrics."""
        timestamp = time.time()
        
        # CPU metrics
        cpu_usage = psutil.cpu_percent(interval=1)
        cpu_metrics = ResourceMetrics(
            resource_type=ResourceType.CPU,
            current_usage=cpu_usage / 100.0,
            average_usage=self._calculate_average_usage(ResourceType.CPU),
            peak_usage=self._calculate_peak_usage(ResourceType.CPU),
            available_capacity=1.0 - (cpu_usage / 100.0),
            timestamp=timestamp,
            metadata={
                'cpu_count': psutil.cpu_count(),
                'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else {},
                'load_avg': psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
            }
        )
        self.current_resources[ResourceType.CPU] = cpu_metrics
        self.resource_metrics[ResourceType.CPU].append(cpu_metrics)
        
        # Memory metrics
        memory = psutil.virtual_memory()
        memory_metrics = ResourceMetrics(
            resource_type=ResourceType.MEMORY,
            current_usage=memory.percent / 100.0,
            average_usage=self._calculate_average_usage(ResourceType.MEMORY),
            peak_usage=self._calculate_peak_usage(ResourceType.MEMORY),
            available_capacity=memory.available / memory.total,
            timestamp=timestamp,
            metadata={
                'total_memory': memory.total,
                'available_memory': memory.available,
                'swap_usage': psutil.swap_memory().percent
            }
        )
        self.current_resources[ResourceType.MEMORY] = memory_metrics
        self.resource_metrics[ResourceType.MEMORY].append(memory_metrics)
        
        # GPU metrics (if available)
        if GPU_AVAILABLE:
            self._collect_gpu_metrics(timestamp)
        
        # Network metrics
        network = psutil.net_io_counters()
        network_metrics = ResourceMetrics(
            resource_type=ResourceType.NETWORK,
            current_usage=0.5,  # Simplified - would need more sophisticated calculation
            average_usage=self._calculate_average_usage(ResourceType.NETWORK),
            peak_usage=self._calculate_peak_usage(ResourceType.NETWORK),
            available_capacity=0.5,  # Simplified
            timestamp=timestamp,
            metadata={
                'bytes_sent': network.bytes_sent,
                'bytes_recv': network.bytes_recv,
                'packets_sent': network.packets_sent,
                'packets_recv': network.packets_recv
            }
        )
        self.current_resources[ResourceType.NETWORK] = network_metrics
        self.resource_metrics[ResourceType.NETWORK].append(network_metrics)
        
        # Storage metrics
        disk = psutil.disk_usage('/')
        storage_metrics = ResourceMetrics(
            resource_type=ResourceType.STORAGE,
            current_usage=disk.used / disk.total,
            average_usage=self._calculate_average_usage(ResourceType.STORAGE),
            peak_usage=self._calculate_peak_usage(ResourceType.STORAGE),
            available_capacity=disk.free / disk.total,
            timestamp=timestamp,
            metadata={
                'total_storage': disk.total,
                'used_storage': disk.used,
                'free_storage': disk.free
            }
        )
        self.current_resources[ResourceType.STORAGE] = storage_metrics
        self.resource_metrics[ResourceType.STORAGE].append(storage_metrics)
        
        # Quantum resource metrics (simulated)
        self._collect_quantum_metrics(timestamp)
    
    def _collect_gpu_metrics(self, timestamp: float):
        """Collect GPU resource metrics."""
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                # Use first GPU for simplicity (could aggregate across all GPUs)
                gpu = gpus[0]
                gpu_metrics = ResourceMetrics(
                    resource_type=ResourceType.GPU,
                    current_usage=gpu.load,
                    average_usage=self._calculate_average_usage(ResourceType.GPU),
                    peak_usage=self._calculate_peak_usage(ResourceType.GPU),
                    available_capacity=1.0 - gpu.load,
                    timestamp=timestamp,
                    metadata={
                        'gpu_name': gpu.name,
                        'memory_usage': gpu.memoryUtil,
                        'memory_total': gpu.memoryTotal,
                        'temperature': gpu.temperature
                    }
                )
                self.current_resources[ResourceType.GPU] = gpu_metrics
                self.resource_metrics[ResourceType.GPU].append(gpu_metrics)
        
        except Exception as e:
            self.logger.warning(f"Could not collect GPU metrics: {e}")
    
    def _collect_quantum_metrics(self, timestamp: float):
        """Collect quantum resource metrics (simulated)."""
        # In a real implementation, this would query actual quantum backends
        # For now, we simulate quantum resource usage
        
        # Simulate quantum resource utilization based on current workload
        base_usage = 0.1 + 0.3 * np.sin(timestamp / 3600)  # Hourly pattern
        noise = np.random.normal(0, 0.05)  # Add some noise
        current_usage = np.clip(base_usage + noise, 0.0, 1.0)
        
        quantum_metrics = ResourceMetrics(
            resource_type=ResourceType.QUANTUM_QPU,
            current_usage=current_usage,
            average_usage=self._calculate_average_usage(ResourceType.QUANTUM_QPU),
            peak_usage=self._calculate_peak_usage(ResourceType.QUANTUM_QPU),
            available_capacity=1.0 - current_usage,
            timestamp=timestamp,
            metadata={
                'qpu_count': 1,
                'queue_length': max(0, int(np.random.normal(5, 2))),
                'coherence_time': np.random.uniform(50, 100),  # microseconds
                'gate_fidelity': np.random.uniform(0.995, 0.999)
            }
        )
        
        self.current_resources[ResourceType.QUANTUM_QPU] = quantum_metrics
        self.resource_metrics[ResourceType.QUANTUM_QPU].append(quantum_metrics)
    
    def _calculate_average_usage(self, resource_type: ResourceType) -> float:
        """Calculate average usage for a resource type."""
        history = self.resource_metrics[resource_type]
        if not history:
            return 0.0
        
        recent_metrics = list(history)[-min(len(history), 20)]  # Last 20 measurements
        return np.mean([m.current_usage for m in recent_metrics])
    
    def _calculate_peak_usage(self, resource_type: ResourceType) -> float:
        """Calculate peak usage for a resource type."""
        history = self.resource_metrics[resource_type]
        if not history:
            return 0.0
        
        recent_metrics = list(history)[-min(len(history), 100)]  # Last 100 measurements
        return np.max([m.current_usage for m in recent_metrics])
    
    def _update_workload_patterns(self):
        """Update workload pattern analysis."""
        current_time = time.time()
        hour_of_day = int((current_time % 86400) / 3600)  # 0-23
        day_of_week = int((current_time / 86400) % 7)  # 0-6
        
        for resource_type, metrics in self.current_resources.items():
            # Store patterns by hour and day
            hour_key = f"{resource_type.value}_hour_{hour_of_day}"
            day_key = f"{resource_type.value}_day_{day_of_week}"
            
            self.workload_patterns[hour_key].append(metrics.current_usage)
            self.workload_patterns[day_key].append(metrics.current_usage)
            
            # Keep only recent patterns (last 30 days)
            max_samples = 30
            self.workload_patterns[hour_key] = self.workload_patterns[hour_key][-max_samples:]
            self.workload_patterns[day_key] = self.workload_patterns[day_key][-max_samples:]
    
    def _update_predictions(self):
        """Update resource utilization predictions."""
        for resource_type in ResourceType:
            try:
                prediction = self._predict_resource_usage(resource_type)
                if prediction is not None:
                    self.prediction_models[resource_type] = prediction
            except Exception as e:
                self.logger.warning(f"Could not generate prediction for {resource_type}: {e}")
    
    def _predict_resource_usage(self, resource_type: ResourceType) -> Optional[List[float]]:
        """Predict future resource usage."""
        history = self.resource_metrics[resource_type]
        if len(history) < 10:  # Need minimum history for prediction
            return None
        
        # Simple time series prediction using moving average and trend
        recent_values = [m.current_usage for m in list(history)[-20:]]
        
        # Calculate trend
        if len(recent_values) >= 2:
            x = np.arange(len(recent_values))
            trend = np.polyfit(x, recent_values, 1)[0]
        else:
            trend = 0.0
        
        # Generate predictions
        last_value = recent_values[-1]
        predictions = []
        
        for i in range(1, self.prediction_horizon + 1):
            # Simple linear trend with seasonal component
            predicted_value = last_value + trend * i
            
            # Add seasonal pattern if available
            current_time = time.time()
            future_hour = int(((current_time + i * self.monitoring_interval) % 86400) / 3600)
            hour_key = f"{resource_type.value}_hour_{future_hour}"
            
            if hour_key in self.workload_patterns and self.workload_patterns[hour_key]:
                seasonal_factor = np.mean(self.workload_patterns[hour_key])
                predicted_value = 0.7 * predicted_value + 0.3 * seasonal_factor
            
            # Clamp to valid range
            predicted_value = np.clip(predicted_value, 0.0, 1.0)
            predictions.append(predicted_value)
        
        return predictions
    
    def _analyze_scaling_needs(self) -> List[ScalingDecision]:
        """Analyze current metrics and determine scaling needs."""
        decisions = []
        
        for resource_type, metrics in self.current_resources.items():
            # Current utilization analysis
            if metrics.current_usage > self.scaling_threshold_up:
                decision = ScalingDecision(
                    action=ScalingAction.SCALE_UP,
                    resource_type=resource_type,
                    target_scale=1.5,  # 50% increase
                    reasoning=f"High utilization: {metrics.current_usage:.2%}",
                    priority=self._calculate_priority(metrics),
                    estimated_impact=0.8,
                    cost_estimate=self._estimate_scaling_cost(resource_type, 1.5),
                    timestamp=time.time()
                )
                decisions.append(decision)
            
            elif metrics.current_usage < self.scaling_threshold_down:
                decision = ScalingDecision(
                    action=ScalingAction.SCALE_DOWN,
                    resource_type=resource_type,
                    target_scale=0.7,  # 30% decrease
                    reasoning=f"Low utilization: {metrics.current_usage:.2%}",
                    priority=1,  # Lower priority for scale down
                    estimated_impact=0.3,
                    cost_estimate=self._estimate_scaling_cost(resource_type, 0.7),
                    timestamp=time.time()
                )
                decisions.append(decision)
            
            # Predictive scaling
            if self.enable_predictive_scaling and resource_type in self.prediction_models:
                predictions = self.prediction_models[resource_type]
                if predictions and max(predictions) > self.scaling_threshold_up:
                    decision = ScalingDecision(
                        action=ScalingAction.SCALE_UP,
                        resource_type=resource_type,
                        target_scale=1.3,  # Moderate increase
                        reasoning="Predictive scaling: high usage predicted",
                        priority=2,  # Lower priority than reactive scaling
                        estimated_impact=0.6,
                        cost_estimate=self._estimate_scaling_cost(resource_type, 1.3),
                        timestamp=time.time()
                    )
                    decisions.append(decision)
        
        # Sort decisions by priority and cost-effectiveness
        if self.enable_cost_optimization:
            decisions.sort(key=lambda d: (d.priority, -d.estimated_impact / max(d.cost_estimate, 0.01)), reverse=True)
        else:
            decisions.sort(key=lambda d: d.priority, reverse=True)
        
        return decisions[:3]  # Limit to top 3 decisions to avoid thrashing
    
    def _calculate_priority(self, metrics: ResourceMetrics) -> int:
        """Calculate priority for scaling decision."""
        if metrics.current_usage > 0.9:
            return 5  # Critical
        elif metrics.current_usage > 0.8:
            return 4  # High
        elif metrics.current_usage > 0.7:
            return 3  # Medium
        else:
            return 2  # Low
    
    def _estimate_scaling_cost(self, resource_type: ResourceType, scale_factor: float) -> float:
        """Estimate the cost of scaling a resource."""
        base_cost = self.resource_costs.get(resource_type, 0.1)
        
        if scale_factor > 1.0:
            # Scale up cost
            additional_resources = scale_factor - 1.0
            return base_cost * additional_resources * 24  # Cost per day
        else:
            # Scale down savings
            reduced_resources = 1.0 - scale_factor
            return -base_cost * reduced_resources * 24  # Negative cost (savings)
    
    def _execute_scaling_decision(self, decision: ScalingDecision):
        """Execute a scaling decision."""
        try:
            self.logger.info(f"Executing scaling decision: {decision.action.value} "
                           f"{decision.resource_type.value} to {decision.target_scale}x")
            
            if decision.resource_type == ResourceType.CPU:
                self._scale_cpu_resources(decision)
            elif decision.resource_type == ResourceType.MEMORY:
                self._scale_memory_resources(decision)
            elif decision.resource_type == ResourceType.GPU:
                self._scale_gpu_resources(decision)
            elif decision.resource_type == ResourceType.QUANTUM_QPU:
                self._scale_quantum_resources(decision)
            
            # Add to scaling history
            self.scaling_history.append(decision)
            
        except Exception as e:
            self.logger.error(f"Failed to execute scaling decision: {e}")
    
    def _scale_cpu_resources(self, decision: ScalingDecision):
        """Scale CPU resources."""
        if self.k8s_enabled:
            try:
                # Kubernetes HPA scaling
                self._scale_kubernetes_deployment("quantum-trainer", "cpu", decision.target_scale)
            except Exception as e:
                self.logger.warning(f"Kubernetes CPU scaling failed: {e}")
        
        # Log the scaling action (in practice, this would trigger actual scaling)
        self.logger.info(f"CPU scaling: {decision.target_scale}x (simulated)")
    
    def _scale_memory_resources(self, decision: ScalingDecision):
        """Scale memory resources."""
        if self.k8s_enabled:
            try:
                self._scale_kubernetes_deployment("quantum-trainer", "memory", decision.target_scale)
            except Exception as e:
                self.logger.warning(f"Kubernetes memory scaling failed: {e}")
        
        self.logger.info(f"Memory scaling: {decision.target_scale}x (simulated)")
    
    def _scale_gpu_resources(self, decision: ScalingDecision):
        """Scale GPU resources."""
        if self.k8s_enabled:
            try:
                self._scale_kubernetes_deployment("quantum-trainer", "gpu", decision.target_scale)
            except Exception as e:
                self.logger.warning(f"Kubernetes GPU scaling failed: {e}")
        
        self.logger.info(f"GPU scaling: {decision.target_scale}x (simulated)")
    
    def _scale_quantum_resources(self, decision: ScalingDecision):
        """Scale quantum resources (QPU allocation)."""
        # In practice, this would involve requesting more QPU time from providers
        # or switching to different quantum backends
        self.logger.info(f"Quantum QPU scaling: {decision.target_scale}x (simulated)")
    
    def _scale_kubernetes_deployment(self, deployment_name: str, resource_type: str, scale_factor: float):
        """Scale Kubernetes deployment resources."""
        if not self.k8s_enabled:
            return
        
        try:
            v1 = client.AppsV1Api()
            
            # Get current deployment
            deployment = v1.read_namespaced_deployment(
                name=deployment_name,
                namespace="default"
            )
            
            # Calculate new resource limits
            current_replicas = deployment.spec.replicas or 1
            new_replicas = max(1, int(current_replicas * scale_factor))
            
            # Update deployment
            deployment.spec.replicas = new_replicas
            
            v1.patch_namespaced_deployment(
                name=deployment_name,
                namespace="default",
                body=deployment
            )
            
            self.logger.info(f"Kubernetes deployment {deployment_name} scaled to {new_replicas} replicas")
            
        except Exception as e:
            self.logger.error(f"Kubernetes scaling failed: {e}")
    
    def _cleanup_scaling_actions(self):
        """Clean up completed scaling actions."""
        # Remove old scaling history (keep last 100 decisions)
        self.scaling_history = self.scaling_history[-100:]
        
        # Clear completed pending actions
        self.pending_scaling_actions = [
            action for action in self.pending_scaling_actions
            if time.time() - action.timestamp < 3600  # Keep for 1 hour
        ]
    
    def add_resource_pool(self, pool_name: str, resources: Dict[ResourceType, float]):
        """Add a resource pool for allocation management."""
        self.resource_pools[pool_name] = {
            'resources': resources,
            'allocated': {rt: 0.0 for rt in ResourceType},
            'created_at': time.time()
        }
        
        self.logger.info(f"Added resource pool: {pool_name}")
    
    def allocate_resources(
        self,
        requester_id: str,
        resource_requirements: Dict[ResourceType, float]
    ) -> Tuple[bool, Dict[ResourceType, float]]:
        """
        Allocate resources from available pools.
        
        Args:
            requester_id: ID of the requesting entity
            resource_requirements: Required resources by type
            
        Returns:
            Tuple of (success, allocated_resources)
        """
        allocated = {}
        
        for resource_type, required_amount in resource_requirements.items():
            available = self._get_available_capacity(resource_type)
            
            if available >= required_amount:
                allocated[resource_type] = required_amount
            else:
                # Partial allocation or failure
                if self.allocation_strategy == "balanced":
                    allocated[resource_type] = available
                else:
                    # Strict allocation - all or nothing
                    return False, {}
        
        # Record allocation
        self._record_allocation(requester_id, allocated)
        
        return True, allocated
    
    def _get_available_capacity(self, resource_type: ResourceType) -> float:
        """Get available capacity for a resource type."""
        if resource_type in self.current_resources:
            return self.current_resources[resource_type].available_capacity
        return 0.0
    
    def _record_allocation(self, requester_id: str, allocated: Dict[ResourceType, float]):
        """Record resource allocation."""
        # In practice, this would update a more sophisticated allocation tracking system
        self.logger.info(f"Allocated resources to {requester_id}: {allocated}")
    
    def get_resource_report(self) -> Dict[str, Any]:
        """Generate comprehensive resource report."""
        current_time = time.time()
        
        # Calculate overall system efficiency
        efficiency_scores = {}
        for resource_type, metrics in self.current_resources.items():
            # Efficiency = utilization balanced with available capacity
            utilization = metrics.current_usage
            capacity = metrics.available_capacity
            
            # Optimal range is 60-80% utilization
            if 0.6 <= utilization <= 0.8:
                efficiency = 1.0
            elif utilization < 0.6:
                efficiency = utilization / 0.6  # Underutilization penalty
            else:
                efficiency = 0.8 / utilization  # Over-utilization penalty
            
            efficiency_scores[resource_type.value] = efficiency
        
        overall_efficiency = np.mean(list(efficiency_scores.values()))
        
        return {
            'timestamp': current_time,
            'overall_efficiency': overall_efficiency,
            'resource_utilization': {
                rt.value: metrics.current_usage 
                for rt, metrics in self.current_resources.items()
            },
            'efficiency_by_resource': efficiency_scores,
            'scaling_decisions_last_hour': len([
                d for d in self.scaling_history 
                if current_time - d.timestamp < 3600
            ]),
            'cost_savings_last_day': sum([
                d.cost_estimate for d in self.scaling_history
                if current_time - d.timestamp < 86400 and d.cost_estimate < 0
            ]),
            'resource_pools': len(self.resource_pools),
            'predictions_available': len(self.prediction_models),
            'monitoring_active': self._monitoring_active
        }
    
    def optimize_allocation_strategy(self):
        """Optimize resource allocation strategy based on historical performance."""
        report = self.get_resource_report()
        
        if report['overall_efficiency'] < 0.6:
            self.logger.warning("Low system efficiency detected - adjusting allocation strategy")
            
            # Adjust thresholds
            if report['resource_utilization']['cpu'] > 0.9:
                self.scaling_threshold_up = min(self.scaling_threshold_up - 0.05, 0.7)
            
            # Consider switching allocation strategy
            if self.allocation_strategy == "balanced" and report['overall_efficiency'] < 0.5:
                self.allocation_strategy = "performance"
                self.logger.info("Switched to performance-focused allocation strategy")
    
    def cleanup(self):
        """Clean up resource manager."""
        self.stop_monitoring()
        self.resource_metrics.clear()
        self.current_resources.clear()
        self.scaling_history.clear()
        self.workload_patterns.clear()
        self.resource_pools.clear()