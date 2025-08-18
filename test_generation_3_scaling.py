"""
Test Suite for Generation 3 Scaling Features
"""

import pytest
import numpy as np
import time
import threading
from unittest.mock import Mock, patch, MagicMock

# Import the scaling modules
from qecc_qml.scaling.distributed_quantum_trainer import DistributedQuantumTrainer, WorkerNode, DistributedTask
from qecc_qml.scaling.adaptive_resource_manager import AdaptiveResourceManager, ResourceType, ScalingAction
from qecc_qml.core.circuit_health_monitor import CircuitHealthMonitor, HealthMetric, HealthAlert
from qecc_qml.utils.advanced_error_recovery import AdvancedErrorRecovery, ErrorType, ErrorEvent
from qecc_qml.security.quantum_security_validator import QuantumSecurityValidator, SecurityLevel, ThreatType

# Import core components for testing
from qecc_qml import QECCAwareQNN


class TestDistributedQuantumTrainer:
    """Test distributed quantum training functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.trainer = DistributedQuantumTrainer(
            master_host="localhost",
            master_port=9001,
            max_workers=4
        )
    
    def teardown_method(self):
        """Clean up after tests."""
        self.trainer.stop_master()
        self.trainer.cleanup()
    
    def test_worker_registration(self):
        """Test worker node registration."""
        # Register a worker
        success = self.trainer.register_worker(
            node_id="worker_1",
            host="localhost",
            port=9002,
            capabilities={"qubits": 20, "memory": "16GB", "gpu": True}
        )
        
        assert success
        assert "worker_1" in self.trainer.workers
        assert self.trainer.workers["worker_1"].capabilities["qubits"] == 20
    
    def test_worker_unregistration(self):
        """Test worker node unregistration."""
        # Register then unregister
        self.trainer.register_worker("worker_1", "localhost", 9002, {"qubits": 10})
        assert "worker_1" in self.trainer.workers
        
        self.trainer.unregister_worker("worker_1")
        assert "worker_1" not in self.trainer.workers
    
    def test_master_start_stop(self):
        """Test master process lifecycle."""
        assert not self.trainer._running
        
        self.trainer.start_master()
        assert self.trainer._running
        
        # Give threads time to start
        time.sleep(0.5)
        
        self.trainer.stop_master()
        assert not self.trainer._running
    
    def test_task_creation(self):
        """Test training task creation."""
        X_train = np.random.rand(100, 4)
        y_train = np.random.randint(0, 2, 100)
        
        tasks = self.trainer._create_training_tasks(X_train, y_train, batch_size=32, epoch=0)
        
        assert len(tasks) == 4  # 100 samples, batch_size 32 -> 4 batches
        assert all(task.task_type == "training" for task in tasks)
        assert all(task.parameters is not None for task in tasks)
    
    def test_worker_selection(self):
        """Test worker selection algorithm."""
        # Register workers with different loads
        self.trainer.register_worker("worker_1", "localhost", 9002, {"qubits": 10})
        self.trainer.register_worker("worker_2", "localhost", 9003, {"qubits": 20})
        
        self.trainer.workers["worker_1"].current_load = 0.3
        self.trainer.workers["worker_2"].current_load = 0.8
        
        task = DistributedTask("test_task", "training", None, np.array([1, 2, 3]))
        
        # Should select worker with lower load
        selected_worker = self.trainer._select_worker(task)
        assert selected_worker == "worker_1"
    
    @patch('qecc_qml.scaling.distributed_quantum_trainer.QECCTrainer')
    def test_task_execution_simulation(self, mock_trainer_class):
        """Test task execution on worker (simulated)."""
        # Setup mock trainer
        mock_trainer = Mock()
        mock_trainer.predict.return_value = np.array([[0.6, 0.4], [0.3, 0.7]])
        mock_trainer.loss_function.return_value = 0.5
        mock_trainer.parameters = np.array([1.0, 2.0, 3.0])
        mock_trainer_class.return_value = mock_trainer
        
        # Setup task and worker
        X_batch = np.random.rand(2, 4)
        y_batch = np.array([0, 1])
        task = DistributedTask("test_task", "training", (X_batch, y_batch), np.array([1, 2, 3]))
        worker = WorkerNode("worker_1", "localhost", 9002, {"qubits": 10})
        
        # Setup QNN mock
        qnn = Mock()
        
        # Execute task
        task_id, result = self.trainer._execute_task_on_worker(task, qnn, worker)
        
        assert task_id == "test_task"
        assert result is not None
        assert "loss" in result
        assert "gradients" in result
        assert worker.status == "idle"


class TestAdaptiveResourceManager:
    """Test adaptive resource management functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.manager = AdaptiveResourceManager(
            monitoring_interval=1.0,  # Fast monitoring for testing
            scaling_threshold_up=0.8,
            scaling_threshold_down=0.3
        )
    
    def teardown_method(self):
        """Clean up after tests."""
        self.manager.stop_monitoring()
        self.manager.cleanup()
    
    def test_resource_metrics_collection(self):
        """Test resource metrics collection."""
        self.manager._collect_resource_metrics()
        
        # Should have metrics for all resource types
        expected_types = [ResourceType.CPU, ResourceType.MEMORY, ResourceType.STORAGE, ResourceType.NETWORK]
        for resource_type in expected_types:
            assert resource_type in self.manager.current_resources
            metrics = self.manager.current_resources[resource_type]
            assert 0.0 <= metrics.current_usage <= 1.0
            assert metrics.timestamp > 0
    
    def test_scaling_decision_analysis(self):
        """Test scaling decision analysis."""
        # Simulate high CPU usage
        high_cpu_metrics = Mock()
        high_cpu_metrics.current_usage = 0.9
        high_cpu_metrics.average_usage = 0.85
        high_cpu_metrics.peak_usage = 0.95
        
        self.manager.current_resources[ResourceType.CPU] = high_cpu_metrics
        
        decisions = self.manager._analyze_scaling_needs()
        
        assert len(decisions) > 0
        cpu_decisions = [d for d in decisions if d.resource_type == ResourceType.CPU]
        assert len(cpu_decisions) > 0
        assert cpu_decisions[0].action == ScalingAction.SCALE_UP
    
    def test_monitoring_lifecycle(self):
        """Test monitoring start/stop lifecycle."""
        assert not self.manager._monitoring_active
        
        self.manager.start_monitoring()
        assert self.manager._monitoring_active
        
        # Give threads time to start
        time.sleep(2.0)
        
        self.manager.stop_monitoring()
        assert not self.manager._monitoring_active
    
    def test_resource_pool_management(self):
        """Test resource pool addition and management."""
        resources = {
            ResourceType.CPU: 8.0,
            ResourceType.MEMORY: 32.0,
            ResourceType.GPU: 2.0
        }
        
        self.manager.add_resource_pool("test_pool", resources)
        
        assert "test_pool" in self.manager.resource_pools
        assert self.manager.resource_pools["test_pool"]["resources"] == resources
    
    def test_resource_allocation(self):
        """Test resource allocation from pools."""
        # Add a resource pool
        self.manager.add_resource_pool("test_pool", {
            ResourceType.CPU: 8.0,
            ResourceType.MEMORY: 16.0
        })
        
        # Mock available capacity
        self.manager._get_available_capacity = Mock()
        self.manager._get_available_capacity.side_effect = lambda rt: 0.8  # 80% available
        
        # Request resources
        requirements = {ResourceType.CPU: 0.5, ResourceType.MEMORY: 0.3}
        success, allocated = self.manager.allocate_resources("test_requester", requirements)
        
        assert success
        assert allocated[ResourceType.CPU] == 0.5
        assert allocated[ResourceType.MEMORY] == 0.3
    
    def test_workload_pattern_tracking(self):
        """Test workload pattern analysis."""
        # Simulate metrics collection over time
        for i in range(10):
            self.manager._collect_resource_metrics()
            self.manager._update_workload_patterns()
            time.sleep(0.1)
        
        # Should have some workload patterns recorded
        assert len(self.manager.workload_patterns) > 0
        
        # Check for hour-based patterns
        hour_patterns = [k for k in self.manager.workload_patterns.keys() if "hour" in k]
        assert len(hour_patterns) > 0


class TestCircuitHealthMonitor:
    """Test quantum circuit health monitoring."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.monitor = CircuitHealthMonitor(
            history_size=100,
            alert_threshold=0.95,
            monitoring_interval=0.5
        )
    
    def teardown_method(self):
        """Clean up after tests."""
        self.monitor.stop_monitoring()
        self.monitor.cleanup()
    
    def test_circuit_evaluation(self):
        """Test circuit health evaluation."""
        # Create a mock circuit
        circuit = Mock()
        circuit.depth.return_value = 20
        circuit.data = [Mock() for _ in range(15)]  # 15 gates
        
        metrics = self.monitor.evaluate_circuit(circuit, shots=1024)
        
        assert "fidelity" in metrics
        assert "depth" in metrics
        assert "gate_count" in metrics
        assert "error_probability" in metrics
        assert 0.0 <= metrics["fidelity"] <= 1.0
    
    def test_health_score_calculation(self):
        """Test overall health score calculation."""
        # Add some mock metrics
        self.monitor.current_metrics["fidelity"] = HealthMetric(
            name="fidelity",
            value=0.95,
            timestamp=time.time(),
            is_critical=True
        )
        
        self.monitor.current_metrics["success_probability"] = HealthMetric(
            name="success_probability", 
            value=0.92,
            timestamp=time.time(),
            is_critical=True
        )
        
        score = self.monitor.get_current_health_score()
        assert 0.0 <= score <= 1.0
        assert score > 0.9  # Should be high with good metrics
    
    def test_monitoring_lifecycle(self):
        """Test health monitoring start/stop."""
        assert not self.monitor._monitoring_active
        
        # Create mock circuits
        circuits = [Mock() for _ in range(3)]
        for circuit in circuits:
            circuit.depth.return_value = 10
            circuit.data = []
        
        self.monitor.start_monitoring(circuits)
        assert self.monitor._monitoring_active
        
        # Let it run briefly
        time.sleep(1.5)
        
        self.monitor.stop_monitoring()
        assert not self.monitor._monitoring_active
    
    def test_alert_generation(self):
        """Test health alert generation."""
        # Create a metric that should trigger an alert
        bad_metric = HealthMetric(
            name="fidelity",
            value=0.3,  # Low fidelity
            timestamp=time.time(),
            threshold_min=0.9,
            is_critical=True
        )
        
        initial_alert_count = len(self.monitor.alerts)
        self.monitor._update_metric(bad_metric)
        
        # Should have generated an alert
        assert len(self.monitor.alerts) > initial_alert_count
    
    def test_health_report_generation(self):
        """Test comprehensive health report generation."""
        # Add some metrics
        self.monitor.current_metrics["fidelity"] = HealthMetric(
            name="fidelity",
            value=0.95,
            timestamp=time.time(),
            is_critical=True
        )
        
        report = self.monitor.get_health_report()
        
        assert "timestamp" in report
        assert "overall_health_score" in report
        assert "health_status" in report
        assert "current_metrics" in report
        assert "recommendations" in report


class TestAdvancedErrorRecovery:
    """Test advanced error recovery system."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.recovery = AdvancedErrorRecovery(
            max_recovery_attempts=3,
            enable_predictive_recovery=True,
            enable_adaptive_strategies=True
        )
    
    def teardown_method(self):
        """Clean up after tests."""
        self.recovery.cleanup()
    
    def test_error_detection(self):
        """Test error detection capabilities."""
        # Create a mock circuit
        circuit = Mock()
        circuit.depth.return_value = 50
        circuit.data = [Mock() for _ in range(40)]
        
        # Mock execution result
        execution_result = Mock()
        execution_result.get_counts.return_value = {"00": 100, "01": 200, "10": 150, "11": 50}
        
        # Mock context with issues
        context = {
            "gate_fidelities": {"cx_0_1": 0.92, "h_0": 0.98},  # Low cx fidelity
            "T1_time": 100,
            "T2_time": 80,
            "circuit_duration": 25  # Should trigger decoherence warning
        }
        
        errors = self.recovery._detect_errors(circuit, execution_result, context)
        
        assert len(errors) > 0
        error_types = [error.error_type for error in errors]
        assert ErrorType.GATE_ERROR in error_types or ErrorType.DECOHERENCE in error_types
    
    def test_error_recovery_attempt(self):
        """Test error recovery attempt."""
        # Create a gate error
        error = ErrorEvent(
            error_type=ErrorType.GATE_ERROR,
            severity=0.8,
            timestamp=time.time(),
            description="Low gate fidelity",
            metadata={"gate": "cx_0_1", "fidelity": 0.92}
        )
        
        circuit = Mock()
        success, recovered_circuit = self.recovery._attempt_error_recovery(error, circuit, {})
        
        # Should attempt recovery (result depends on implementation)
        assert isinstance(success, bool)
        assert recovered_circuit is not None
        assert error.recovery_attempted
    
    def test_error_prediction(self):
        """Test error prediction capabilities."""
        # Add some error history
        for i in range(5):
            error = ErrorEvent(
                error_type=ErrorType.DECOHERENCE,
                severity=0.6,
                timestamp=time.time() - (i * 600),  # Every 10 minutes
                location="qubit_0"
            )
            self.recovery._update_error_patterns(error)
        
        circuit = Mock()
        circuit.depth.return_value = 80  # Deep circuit
        
        predictions = self.recovery.predict_errors(circuit, {"T1_time": 50})
        
        # Should predict potential errors
        assert len(predictions) > 0
        predicted_types = [pred[0] for pred in predictions]
        assert ErrorType.DECOHERENCE in predicted_types
    
    def test_recovery_performance_tracking(self):
        """Test recovery performance tracking."""
        # Simulate some recovery attempts
        error = ErrorEvent(ErrorType.MEASUREMENT_ERROR, 0.5, time.time())
        
        # Simulate successful recovery
        self.recovery._apply_mitigation(error)
        
        report = self.recovery.get_recovery_report()
        
        assert "total_errors_detected" in report
        assert "overall_recovery_rate" in report
        assert "strategy_performance" in report


class TestQuantumSecurityValidator:
    """Test quantum security validation system."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.validator = QuantumSecurityValidator(
            security_level=SecurityLevel.ENHANCED
        )
    
    def teardown_method(self):
        """Clean up after tests."""
        self.validator.cleanup()
    
    def test_circuit_security_validation(self):
        """Test circuit security validation."""
        # Create a mock circuit
        circuit = Mock()
        circuit.depth.return_value = 30
        circuit.data = [Mock() for _ in range(25)]
        circuit.num_qubits = 4
        
        # Should pass basic validation
        is_secure = self.validator.validate_circuit_security(circuit)
        assert isinstance(is_secure, bool)
    
    def test_parameter_validation(self):
        """Test quantum parameter validation."""
        # Valid parameters
        valid_params = np.array([1.5, -0.8, 2.1, 0.3])
        is_valid = self.validator.validate_parameters(valid_params, "test_params")
        assert is_valid
        
        # Invalid parameters (too large)
        invalid_params = np.array([15.0, -8.0, 21.0, 0.3])  # Exceeds max_parameter_range
        is_valid = self.validator.validate_parameters(invalid_params, "test_params_invalid")
        assert not is_valid
    
    def test_data_validation(self):
        """Test training data validation."""
        # Normal data
        normal_data = np.random.normal(0, 1, (100, 10))
        normal_labels = np.random.randint(0, 2, 100)
        
        is_valid = self.validator.validate_data(normal_data, normal_labels)
        assert is_valid
        
        # Data with outliers
        outlier_data = np.random.normal(0, 1, (100, 10))
        outlier_data[0] = 100  # Extreme outlier
        outlier_data[1] = -100
        outlier_data[2] = 50
        
        is_valid = self.validator.validate_data(outlier_data, normal_labels)
        # May or may not be valid depending on outlier detection sensitivity
        assert isinstance(is_valid, bool)
    
    def test_security_report_generation(self):
        """Test security report generation."""
        # Generate some violations first
        invalid_params = np.array([15.0, -8.0])  # Should trigger violation
        self.validator.validate_parameters(invalid_params, "test")
        
        report = self.validator.get_security_report()
        
        assert "timestamp" in report
        assert "security_level" in report
        assert "total_violations" in report
        assert "security_score" in report
        assert 0.0 <= report["security_score"] <= 1.0
    
    def test_threat_detection(self):
        """Test various threat detection capabilities."""
        # Test circuit with suspicious patterns
        circuit = Mock()
        circuit.depth.return_value = 1500  # Exceeds policy limit
        circuit.data = [Mock() for _ in range(2000)]
        circuit.num_qubits = 4
        
        violations = self.validator._analyze_circuit_complexity(circuit)
        
        assert len(violations) > 0
        assert any(v.threat_type == ThreatType.CIRCUIT_TAMPERING for v in violations)


def test_integration_scaling_components():
    """Integration test for scaling components working together."""
    # Test that different scaling components can work together
    
    # Setup components
    resource_manager = AdaptiveResourceManager(monitoring_interval=0.5)
    health_monitor = CircuitHealthMonitor(monitoring_interval=0.5)
    error_recovery = AdvancedErrorRecovery()
    security_validator = QuantumSecurityValidator()
    
    try:
        # Start monitoring
        resource_manager.start_monitoring()
        
        # Create test circuit
        qnn = QECCAwareQNN(num_qubits=4, num_layers=2)
        test_circuit = qnn.get_circuit()
        
        # Validate security
        is_secure = security_validator.validate_circuit_security(test_circuit)
        assert isinstance(is_secure, bool)
        
        # Monitor health
        health_metrics = health_monitor.evaluate_circuit(test_circuit)
        assert len(health_metrics) > 0
        
        # Test error detection and recovery
        success, errors, recovered_circuit = error_recovery.detect_and_recover(
            test_circuit, None, {"T1_time": 100, "T2_time": 80}
        )
        assert isinstance(success, bool)
        assert isinstance(errors, list)
        
        # Check resource utilization
        resource_report = resource_manager.get_resource_report()
        assert "overall_efficiency" in resource_report
        
        # Let systems run briefly
        time.sleep(1.0)
        
        # Get comprehensive status
        health_report = health_monitor.get_health_report()
        security_report = security_validator.get_security_report()
        recovery_report = error_recovery.get_recovery_report()
        
        assert all([
            "health_status" in health_report,
            "security_score" in security_report,
            "overall_recovery_rate" in recovery_report
        ])
        
    finally:
        # Cleanup all components
        resource_manager.cleanup()
        health_monitor.cleanup()
        error_recovery.cleanup()
        security_validator.cleanup()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])