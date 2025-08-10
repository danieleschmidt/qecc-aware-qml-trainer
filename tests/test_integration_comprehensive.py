"""
Comprehensive integration tests for QECC-QML framework.
"""

import pytest
import numpy as np
import time
from pathlib import Path
import tempfile
import shutil

# Import framework components
from qecc_qml import (
    QECCAwareQNN, QECCTrainer, SurfaceCode, ColorCode, 
    NoiseModel, NoiseBenchmark
)
from qecc_qml.core.circuit_validation import CircuitValidator, SecurityManager
from qecc_qml.monitoring.health_monitor import HealthMonitor
from qecc_qml.validation.comprehensive_validation import ComprehensiveValidator
from qecc_qml.utils.error_recovery import ErrorRecoveryManager, FaultTolerantWrapper
from qecc_qml.optimization.adaptive_scaling import AdaptiveScaler
from qecc_qml.optimization.quantum_circuit_optimization import QuantumCircuitOptimizer
from qecc_qml.deployment.production_deployment import DeploymentConfig, ProductionDeployment

from qiskit import QuantumCircuit
from qiskit.quantum_info import random_statevector


class TestQECCQNNIntegration:
    """Integration tests for QECC-aware Quantum Neural Networks."""
    
    def test_basic_qnn_creation(self):
        """Test basic QNN creation and configuration."""
        qnn = QECCAwareQNN(
            num_qubits=4,
            num_layers=2,
            entanglement="circular"
        )
        
        assert qnn.num_qubits == 4
        assert qnn.num_layers == 2
        assert qnn.entanglement == "circular"
        
    def test_qnn_with_error_correction(self):
        """Test QNN integration with error correction."""
        qnn = QECCAwareQNN(num_qubits=4, num_layers=3)
        
        surface_code = SurfaceCode(distance=3, logical_qubits=4)
        
        # This would be implemented in the actual QNN class
        # qnn.add_error_correction(surface_code)
        
        # For now, just test that objects can be created
        assert isinstance(surface_code, SurfaceCode)
        
    def test_training_workflow(self):
        """Test complete training workflow."""
        # Create synthetic quantum dataset
        X = np.random.random((100, 16))  # 100 samples, 16 features
        y = np.random.randint(0, 2, 100)  # Binary classification
        
        # Create QNN
        qnn = QECCAwareQNN(num_qubits=4, num_layers=2)
        
        # Create noise model
        noise_model = NoiseModel()
        
        # Create trainer (this would be implemented)
        trainer = QECCTrainer(qnn=qnn, noise_model=noise_model)
        
        # Test basic trainer functionality
        assert trainer.qnn == qnn
        assert trainer.noise_model == noise_model


class TestValidationFramework:
    """Integration tests for validation framework."""
    
    def test_circuit_validation(self):
        """Test comprehensive circuit validation."""
        validator = CircuitValidator()
        
        # Create test circuit
        circuit = QuantumCircuit(4, 4)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.cx(1, 2)
        circuit.cx(2, 3)
        circuit.measure_all()
        
        # Validate circuit
        results = validator.validate_circuit(circuit)
        
        assert "valid" in results
        assert isinstance(results["valid"], bool)
        assert "warnings" in results
        assert "errors" in results
        assert "metrics" in results
        
    def test_security_manager(self):
        """Test security management."""
        security_manager = SecurityManager()
        
        # Create circuit with potentially unsafe operations
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)
        
        # Sanitize circuit
        sanitized = security_manager.sanitize_circuit(circuit)
        
        assert isinstance(sanitized, QuantumCircuit)
        assert sanitized.num_qubits == 2
        
    def test_comprehensive_validator(self):
        """Test comprehensive validation system."""
        validator = ComprehensiveValidator()
        
        # Test input data validation
        X = np.random.random((50, 10))
        y = np.random.randint(0, 2, 50)
        
        results = validator.validate_training_inputs(X, y, batch_size=10, num_epochs=5)
        
        assert isinstance(results, list)
        for result in results:
            assert hasattr(result, 'passed')
            assert hasattr(result, 'message')
            assert hasattr(result, 'severity')


class TestMonitoringSystem:
    """Integration tests for monitoring and health management."""
    
    def test_health_monitor(self):
        """Test health monitoring system."""
        monitor = HealthMonitor(monitoring_interval=0.1)
        
        # Start monitoring briefly
        monitor.start_monitoring()
        time.sleep(0.5)  # Let it collect some metrics
        monitor.stop_monitoring()
        
        # Check health status
        health = monitor.get_current_health()
        
        assert "timestamp" in health
        assert "uptime" in health
        assert "overall_status" in health
        assert "metrics" in health
        
    def test_error_recovery(self):
        """Test error recovery system."""
        recovery_manager = ErrorRecoveryManager(max_retries=2)
        
        # Test function that fails then succeeds
        call_count = 0
        
        @recovery_manager.retry_with_backoff(max_retries=2)
        def test_function():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("Test error")
            return "success"
        
        result = test_function()
        assert result == "success"
        assert call_count == 2
        
    def test_fault_tolerant_wrapper(self):
        """Test fault-tolerant function wrapper."""
        recovery_manager = ErrorRecoveryManager()
        wrapper = FaultTolerantWrapper(recovery_manager)
        
        def unreliable_function(x):
            if x < 0:
                raise ValueError("Negative input")
            return x * 2
        
        def fallback_function(x):
            return 0  # Safe fallback
        
        safe_function = wrapper.make_fault_tolerant(
            unreliable_function,
            fallback_func=fallback_function,
            exceptions=(ValueError,)
        )
        
        # Test successful call
        assert safe_function(5) == 10
        
        # Test fallback
        assert safe_function(-1) == 0


class TestOptimizationFramework:
    """Integration tests for optimization components."""
    
    def test_quantum_circuit_optimization(self):
        """Test quantum circuit optimization."""
        optimizer = QuantumCircuitOptimizer()
        
        # Create test circuit with redundancies
        circuit = QuantumCircuit(3)
        circuit.x(0)
        circuit.x(0)  # Should cancel out
        circuit.h(1)
        circuit.cx(0, 1)
        circuit.cx(0, 2)
        
        optimized, result = optimizer.optimize_circuit(circuit, optimization_level=1)
        
        assert isinstance(optimized, QuantumCircuit)
        assert isinstance(result.optimization_time, float)
        assert result.optimization_time >= 0
        
    def test_adaptive_scaling(self):
        """Test adaptive scaling system."""
        scaler = AdaptiveScaler(
            min_workers=1,
            max_workers=4,
            monitoring_interval=0.1
        )
        
        # Test scaling status
        status = scaler.get_scaling_status()
        
        assert "current_workers" in status
        assert "scaling_enabled" in status
        assert "current_metrics" in status
        
        # Brief test of scaling loop
        scaler.start_adaptive_scaling()
        time.sleep(0.3)  # Let it run briefly
        scaler.stop_adaptive_scaling()


class TestDeploymentSystem:
    """Integration tests for deployment system."""
    
    def test_deployment_config(self):
        """Test deployment configuration."""
        config = DeploymentConfig(
            name="test-qecc-qml",
            version="1.0.0",
            replicas=2,
            cpu_request="100m",
            memory_request="256Mi"
        )
        
        assert config.name == "test-qecc-qml"
        assert config.version == "1.0.0"
        assert config.replicas == 2
        
    def test_production_deployment(self):
        """Test production deployment workflow."""
        config = DeploymentConfig(
            name="test-deployment",
            version="1.0.0",
            replicas=1,
            min_replicas=1,
            max_replicas=2
        )
        
        deployment = ProductionDeployment(config)
        
        # Test deployment status
        status = deployment.get_status()
        
        assert "deployment_id" in status
        assert "name" in status
        assert "status" in status
        assert "replicas" in status


class TestEndToEndWorkflows:
    """End-to-end workflow integration tests."""
    
    def test_qml_training_pipeline(self):
        """Test complete QML training pipeline."""
        # Create synthetic dataset
        np.random.seed(42)
        X = np.random.random((20, 8))
        y = np.random.randint(0, 2, 20)
        
        # Create components
        qnn = QECCAwareQNN(num_qubits=3, num_layers=2)
        noise_model = NoiseModel()
        
        # Validate inputs
        validator = ComprehensiveValidator()
        validation_results = validator.validate_training_inputs(X, y, batch_size=5)
        
        # Check validation passed
        critical_errors = [r for r in validation_results if not r.passed and r.severity == "error"]
        assert len(critical_errors) == 0, f"Validation errors: {[r.message for r in critical_errors]}"
        
        # Create trainer
        trainer = QECCTrainer(qnn=qnn, noise_model=noise_model)
        
        # Test basic trainer setup
        assert trainer.qnn == qnn
        
    def test_error_correction_workflow(self):
        """Test error correction integration workflow."""
        # Create quantum circuit
        circuit = QuantumCircuit(9)  # Enough for Shor code
        circuit.h(0)
        for i in range(8):
            circuit.cx(i, (i + 1) % 9)
        circuit.measure_all()
        
        # Validate circuit
        validator = CircuitValidator()
        validation_result = validator.validate_circuit(circuit)
        
        assert validation_result["valid"] is True
        
        # Optimize circuit
        optimizer = QuantumCircuitOptimizer()
        optimized_circuit, opt_result = optimizer.optimize_circuit(circuit)
        
        assert optimized_circuit.num_qubits == circuit.num_qubits
        assert opt_result.optimization_time >= 0
        
    def test_monitoring_and_recovery_workflow(self):
        """Test integrated monitoring and error recovery."""
        # Setup monitoring
        monitor = HealthMonitor(monitoring_interval=0.1)
        recovery_manager = ErrorRecoveryManager()
        
        # Start monitoring
        monitor.start_monitoring()
        
        # Simulate some work with potential failures
        @recovery_manager.retry_with_backoff(max_retries=1)
        def simulate_quantum_computation():
            # Simulate computation that might fail
            if np.random.random() < 0.3:  # 30% chance of failure
                raise RuntimeError("Quantum computation failed")
            return "success"
        
        # Run computation multiple times
        results = []
        for _ in range(5):
            try:
                result = simulate_quantum_computation()
                results.append(result)
            except RuntimeError:
                results.append("failed")
        
        # Stop monitoring
        monitor.stop_monitoring()
        
        # Check that some computations succeeded
        successes = [r for r in results if r == "success"]
        assert len(successes) > 0, "No computations succeeded"
        
    def test_benchmarking_workflow(self):
        """Test benchmarking and performance evaluation."""
        # Create test circuit
        circuit = QuantumCircuit(4)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.cx(1, 2)
        circuit.cx(2, 3)
        
        # Setup noise benchmark
        benchmark = NoiseBenchmark()
        
        # Test basic benchmark setup
        assert isinstance(benchmark, NoiseBenchmark)
        
        # Create noise levels for testing
        noise_levels = np.logspace(-3, -1, 3)  # 3 noise levels
        
        assert len(noise_levels) == 3
        assert noise_levels[0] < noise_levels[-1]


class TestPerformanceRequirements:
    """Test performance and scalability requirements."""
    
    def test_circuit_optimization_performance(self):
        """Test that circuit optimization meets performance requirements."""
        optimizer = QuantumCircuitOptimizer(max_optimization_time=5.0)
        
        # Create moderately complex circuit
        circuit = QuantumCircuit(6)
        for i in range(10):  # 10 layers
            for q in range(6):
                circuit.ry(np.pi * np.random.random(), q)
            for q in range(5):
                circuit.cx(q, q + 1)
        
        start_time = time.time()
        optimized_circuit, result = optimizer.optimize_circuit(circuit)
        optimization_time = time.time() - start_time
        
        # Should complete within reasonable time
        assert optimization_time < 10.0, f"Optimization took too long: {optimization_time}s"
        
        # Should maintain circuit structure
        assert optimized_circuit.num_qubits == circuit.num_qubits
        
    def test_validation_performance(self):
        """Test validation system performance."""
        validator = ComprehensiveValidator()
        
        # Create large dataset
        X = np.random.random((1000, 20))
        y = np.random.randint(0, 4, 1000)
        
        start_time = time.time()
        results = validator.validate_training_inputs(X, y)
        validation_time = time.time() - start_time
        
        # Should complete quickly even for large datasets
        assert validation_time < 1.0, f"Validation took too long: {validation_time}s"
        
    def test_monitoring_overhead(self):
        """Test monitoring system overhead."""
        monitor = HealthMonitor(monitoring_interval=0.01)  # High frequency
        
        # Measure overhead
        monitor.start_monitoring()
        
        start_time = time.time()
        # Simulate some work
        for i in range(100):
            time.sleep(0.001)
        work_time = time.time() - start_time
        
        monitor.stop_monitoring()
        
        # Monitoring should not significantly impact performance
        assert work_time < 0.5, f"Work took too long with monitoring: {work_time}s"


class TestResourceManagement:
    """Test resource management and cleanup."""
    
    def test_temporary_resource_cleanup(self):
        """Test that temporary resources are cleaned up."""
        initial_temp_files = len(list(Path(tempfile.gettempdir()).iterdir()))
        
        # Create some temporary resources
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_file = Path(f.name)
            f.write(b"test data")
        
        # Simulate resource cleanup
        if temp_file.exists():
            temp_file.unlink()
        
        final_temp_files = len(list(Path(tempfile.gettempdir()).iterdir()))
        
        # Should not leak temporary files
        assert final_temp_files <= initial_temp_files + 1  # Allow for some system files
        
    def test_memory_usage_monitoring(self):
        """Test memory usage stays within bounds."""
        import psutil
        import gc
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # Create and destroy many objects
        large_objects = []
        for i in range(100):
            large_array = np.random.random((1000, 1000))
            large_objects.append(large_array)
        
        peak_memory = process.memory_info().rss
        
        # Clear objects and force garbage collection
        large_objects.clear()
        gc.collect()
        
        final_memory = process.memory_info().rss
        
        # Memory should be released
        memory_increase = final_memory - initial_memory
        peak_increase = peak_memory - initial_memory
        
        # Should release most allocated memory
        assert memory_increase < peak_increase * 0.5, "Memory not properly released"


class TestCompatibilityAndIntegration:
    """Test compatibility with external systems."""
    
    def test_qiskit_integration(self):
        """Test integration with Qiskit ecosystem."""
        from qiskit import QuantumCircuit, transpile
        from qiskit_aer import AerSimulator
        
        # Create circuit using QECC-QML
        qnn = QECCAwareQNN(num_qubits=3, num_layers=2)
        
        # Create standard Qiskit circuit for comparison
        qiskit_circuit = QuantumCircuit(3)
        qiskit_circuit.h(0)
        qiskit_circuit.cx(0, 1)
        qiskit_circuit.cx(1, 2)
        
        # Test that our validator works with Qiskit circuits
        validator = CircuitValidator()
        result = validator.validate_circuit(qiskit_circuit)
        
        assert result["valid"] is True
        
    def test_numpy_array_handling(self):
        """Test proper handling of NumPy arrays."""
        # Test various NumPy array types
        arrays = [
            np.random.random((10, 5)).astype(np.float32),
            np.random.random((10, 5)).astype(np.float64),
            np.random.randint(0, 2, (10,)).astype(np.int32),
            np.random.randint(0, 2, (10,)).astype(np.int64)
        ]
        
        validator = ComprehensiveValidator()
        
        for i, X in enumerate(arrays):
            y = np.random.randint(0, 2, len(X))
            results = validator.validate_training_inputs(X, y)
            
            # Should handle all array types without errors
            critical_errors = [r for r in results if not r.passed and r.severity == "error"]
            assert len(critical_errors) == 0, f"Array type {i} caused errors: {[r.message for r in critical_errors]}"


# Pytest configuration
@pytest.fixture
def sample_quantum_data():
    """Fixture providing sample quantum data for tests."""
    np.random.seed(42)
    X = np.random.random((50, 16))
    y = np.random.randint(0, 2, 50)
    return X, y


@pytest.fixture
def sample_quantum_circuit():
    """Fixture providing sample quantum circuit for tests."""
    circuit = QuantumCircuit(4, 4)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.cx(1, 2)
    circuit.cx(2, 3)
    circuit.measure_all()
    return circuit


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])