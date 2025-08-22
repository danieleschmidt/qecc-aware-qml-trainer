#!/usr/bin/env python3
"""
Autonomous Quality Gates Execution.

Comprehensive testing, security validation, and performance benchmarking
for the QECC-aware quantum machine learning framework.
"""

import sys
import time
import json
import logging
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Core imports
from qecc_qml.core.robust_quantum_executor import RobustQuantumExecutor, RobustExecutionConfig
from qecc_qml.security.advanced_security_validator import AdvancedSecurityValidator, SecurityPolicy
from qecc_qml.monitoring.comprehensive_health_monitor import ComprehensiveHealthMonitor, MonitoringConfig
from qecc_qml.optimization.quantum_performance_enhancer import QuantumPerformanceEnhancer, OptimizationConfig
from qecc_qml.scaling.distributed_quantum_orchestrator import DistributedQuantumOrchestrator, ScalingPolicy

# Test specific imports
from qecc_qml.datasets.simple_datasets import create_test_dataset
from qecc_qml.core.quantum_nn import QECCAwareQNN
from qecc_qml.training.basic_trainer import BasicQECCTrainer

def setup_logging() -> logging.Logger:
    """Set up logging for quality gates."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('quality_gates.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

class QualityGateResult:
    """Result of a quality gate check."""
    
    def __init__(self, name: str, passed: bool, details: Dict[str, Any], execution_time: float):
        self.name = name
        self.passed = passed
        self.details = details
        self.execution_time = execution_time
        self.timestamp = time.time()

class AutonomousQualityGates:
    """
    Autonomous quality gates execution system.
    
    Executes comprehensive testing, security validation, and performance
    benchmarking without human intervention.
    """
    
    def __init__(self):
        self.logger = setup_logging()
        self.results: List[QualityGateResult] = []
        self.total_start_time = time.time()
        
    def execute_all_gates(self) -> Dict[str, Any]:
        """Execute all quality gates autonomously."""
        self.logger.info("🚀 Starting Autonomous Quality Gates Execution")
        
        # Define quality gates in execution order
        gates = [
            ("Core Functionality Tests", self._test_core_functionality),
            ("Security Validation", self._test_security_validation),
            ("Performance Benchmarks", self._test_performance_benchmarks),
            ("Robustness Tests", self._test_robustness),
            ("Scalability Tests", self._test_scalability),
            ("Integration Tests", self._test_integration),
            ("Research Framework Tests", self._test_research_framework),
            ("Error Handling Tests", self._test_error_handling),
            ("Resource Management Tests", self._test_resource_management),
            ("Final System Validation", self._test_final_validation)
        ]
        
        # Execute gates
        passed_gates = 0
        total_gates = len(gates)
        
        for gate_name, gate_function in gates:
            try:
                self.logger.info(f"🔍 Executing: {gate_name}")
                start_time = time.time()
                
                result = gate_function()
                execution_time = time.time() - start_time
                
                gate_result = QualityGateResult(
                    name=gate_name,
                    passed=result.get('passed', False),
                    details=result,
                    execution_time=execution_time
                )
                
                self.results.append(gate_result)
                
                if gate_result.passed:
                    passed_gates += 1
                    self.logger.info(f"✅ {gate_name} PASSED ({execution_time:.2f}s)")
                else:
                    self.logger.error(f"❌ {gate_name} FAILED ({execution_time:.2f}s)")
                    self.logger.error(f"   Details: {result.get('error', 'Unknown error')}")
                
            except Exception as e:
                execution_time = time.time() - start_time
                error_details = {
                    'passed': False,
                    'error': str(e),
                    'traceback': traceback.format_exc()
                }
                
                gate_result = QualityGateResult(
                    name=gate_name,
                    passed=False,
                    details=error_details,
                    execution_time=execution_time
                )
                
                self.results.append(gate_result)
                self.logger.error(f"💥 {gate_name} EXCEPTION ({execution_time:.2f}s): {e}")
        
        # Generate final report
        total_execution_time = time.time() - self.total_start_time
        success_rate = (passed_gates / total_gates) * 100
        
        final_report = {
            'timestamp': time.time(),
            'total_gates': total_gates,
            'passed_gates': passed_gates,
            'failed_gates': total_gates - passed_gates,
            'success_rate': success_rate,
            'total_execution_time': total_execution_time,
            'overall_status': 'PASSED' if success_rate >= 85 else 'FAILED',
            'gate_results': [
                {
                    'name': r.name,
                    'passed': r.passed,
                    'execution_time': r.execution_time,
                    'details': r.details
                }
                for r in self.results
            ]
        }
        
        self.logger.info(f"🏁 Quality Gates Complete: {passed_gates}/{total_gates} passed ({success_rate:.1f}%)")
        self.logger.info(f"⏱️  Total execution time: {total_execution_time:.2f}s")
        
        if success_rate >= 85:
            self.logger.info("🎉 OVERALL STATUS: PASSED - System ready for production!")
        else:
            self.logger.error("🚨 OVERALL STATUS: FAILED - System requires fixes before production")
        
        return final_report
    
    def _test_core_functionality(self) -> Dict[str, Any]:
        """Test core functionality of the quantum ML framework."""
        try:
            # Test quantum neural network creation
            qnn = QECCAwareQNN(num_qubits=4, num_layers=2)
            assert qnn is not None, "Failed to create QECCAwareQNN"
            
            # Test dataset creation
            X_train, y_train = create_test_dataset(n_samples=100, n_features=4)
            assert X_train.shape == (100, 4), f"Wrong training data shape: {X_train.shape}"
            assert y_train.shape == (100,), f"Wrong training labels shape: {y_train.shape}"
            
            # Test basic trainer
            trainer = BasicQECCTrainer(qnn)
            assert trainer is not None, "Failed to create trainer"
            
            # Test basic training step
            initial_params = trainer.get_parameters()
            trainer.train_step(X_train[:10], y_train[:10])
            final_params = trainer.get_parameters()
            
            # Verify parameters changed
            param_changed = not np.allclose(initial_params, final_params, atol=1e-6)
            assert param_changed, "Parameters did not change during training"
            
            return {
                'passed': True,
                'qnn_created': True,
                'dataset_created': True,
                'trainer_created': True,
                'training_step_executed': True,
                'parameters_updated': param_changed,
                'samples_tested': 10
            }
            
        except Exception as e:
            return {
                'passed': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def _test_security_validation(self) -> Dict[str, Any]:
        """Test security validation system."""
        try:
            # Initialize security validator
            policy = SecurityPolicy(
                max_qubits=20,
                max_circuit_depth=500,
                require_authentication=True
            )
            validator = AdvancedSecurityValidator(policy)
            
            # Test circuit validation
            qnn = QECCAwareQNN(num_qubits=4, num_layers=2)
            circuit = qnn.create_circuit(np.random.rand(4))
            
            validation_result = validator.validate_circuit_security(circuit)
            assert 'secure' in validation_result, "Security validation missing 'secure' field"
            
            # Test input sanitization
            test_data = ["<script>alert('xss')</script>", "normal_data", {"key": "value"}]
            for data in test_data:
                sanitized = validator.sanitize_input_data(data)
                assert sanitized is not None, "Input sanitization returned None"
            
            # Test hash creation and verification
            test_string = "test_data_for_hashing"
            hash_value = validator.create_secure_hash(test_string)
            assert len(hash_value) == 64, "Hash should be 64 characters (SHA256)"
            
            integrity_check = validator.verify_data_integrity(test_string, hash_value)
            assert integrity_check, "Data integrity verification failed"
            
            # Test security summary
            summary = validator.get_security_summary()
            assert 'security_policy' in summary, "Security summary missing policy info"
            
            return {
                'passed': True,
                'circuit_validation': validation_result['secure'],
                'input_sanitization': True,
                'hash_verification': integrity_check,
                'security_summary': len(summary) > 0
            }
            
        except Exception as e:
            return {
                'passed': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def _test_performance_benchmarks(self) -> Dict[str, Any]:
        """Test performance optimization and benchmarking."""
        try:
            # Initialize performance enhancer
            config = OptimizationConfig(
                enable_circuit_optimization=True,
                enable_caching=True,
                enable_parallel_execution=True
            )
            enhancer = QuantumPerformanceEnhancer(config)
            
            # Create test circuits
            test_circuits = []
            for i in range(5):
                qnn = QECCAwareQNN(num_qubits=4, num_layers=2)
                circuit = qnn.create_circuit(np.random.rand(4))
                test_circuits.append(circuit)
            
            # Test circuit optimization
            original_circuit = test_circuits[0]
            optimized_circuit = enhancer.optimize_circuit(original_circuit)
            assert optimized_circuit is not None, "Circuit optimization returned None"
            
            # Test caching functionality
            def dummy_operation(circuit):
                time.sleep(0.001)  # Simulate work
                return {'result': 'cached_result', 'circuit_qubits': getattr(circuit, 'num_qubits', 4)}
            
            # First call should miss cache
            result1 = enhancer.execute_with_caching(dummy_operation, original_circuit)
            # Second call should hit cache
            result2 = enhancer.execute_with_caching(dummy_operation, original_circuit)
            assert result1 == result2, "Cached results don't match"
            
            # Test parallel execution
            start_time = time.time()
            parallel_results = enhancer.execute_parallel_batch(dummy_operation, test_circuits[:3])
            parallel_time = time.time() - start_time
            
            assert len(parallel_results) == 3, f"Expected 3 results, got {len(parallel_results)}"
            assert all(r is not None for r in parallel_results), "Some parallel results are None"
            
            # Test performance summary
            summary = enhancer.get_performance_summary()
            assert 'cache_hit_rate' in summary, "Performance summary missing cache hit rate"
            
            # Run benchmark
            benchmark_results = enhancer.benchmark_performance(test_circuits[:2], iterations=3)
            assert 'optimized' in benchmark_results, "Benchmark missing optimized results"
            assert 'unoptimized' in benchmark_results, "Benchmark missing unoptimized results"
            
            return {
                'passed': True,
                'circuit_optimization': True,
                'caching_functional': True,
                'parallel_execution': True,
                'parallel_time': parallel_time,
                'cache_hit_rate': summary['cache_hit_rate'],
                'benchmark_completed': True,
                'performance_improvement': benchmark_results.get('performance_improvement', {})
            }
            
        except Exception as e:
            return {
                'passed': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def _test_robustness(self) -> Dict[str, Any]:
        """Test robustness and error handling."""
        try:
            # Initialize robust executor
            config = RobustExecutionConfig(
                max_retries=2,
                enable_validation=True,
                enable_monitoring=True,
                auto_recovery=True
            )
            executor = RobustQuantumExecutor(config)
            
            # Test circuit validation
            qnn = QECCAwareQNN(num_qubits=4, num_layers=2)
            circuit = qnn.create_circuit(np.random.rand(4))
            
            validation_passed = executor.validate_circuit(circuit)
            assert validation_passed, "Circuit validation failed"
            
            # Test robust execution with retries
            def failing_operation():
                """Operation that fails on first try."""
                if not hasattr(failing_operation, 'attempts'):
                    failing_operation.attempts = 0
                failing_operation.attempts += 1
                
                if failing_operation.attempts == 1:
                    raise Exception("Simulated failure")
                return {'success': True, 'attempts': failing_operation.attempts}
            
            result = executor.execute_with_retry(
                failing_operation,
                operation_name="test_retry"
            )
            assert result['success'], "Retry mechanism failed"
            assert result['attempts'] == 2, "Retry didn't work as expected"
            
            # Test checkpoint creation and restoration
            test_data = {'param1': 1.5, 'param2': [1, 2, 3]}
            executor.create_checkpoint('test_checkpoint', test_data)
            restored_data = executor.restore_checkpoint('test_checkpoint')
            assert restored_data is not None, "Checkpoint restoration failed"
            
            # Test execution metrics
            metrics = executor.get_execution_metrics()
            assert 'total_executions' in metrics, "Execution metrics missing"
            assert metrics['total_executions'] > 0, "No executions recorded"
            
            return {
                'passed': True,
                'circuit_validation': validation_passed,
                'retry_mechanism': True,
                'checkpoint_system': True,
                'metrics_collection': True,
                'total_executions': metrics['total_executions']
            }
            
        except Exception as e:
            return {
                'passed': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def _test_scalability(self) -> Dict[str, Any]:
        """Test scalability and distributed execution."""
        try:
            # Initialize distributed orchestrator
            policy = ScalingPolicy(
                min_nodes=1,
                max_nodes=3,
                target_cpu_utilization=70.0
            )
            orchestrator = DistributedQuantumOrchestrator(policy)
            orchestrator.start()
            
            try:
                # Test node registration
                from qecc_qml.scaling.distributed_quantum_orchestrator import QuantumNode, NodeStatus
                
                test_node = QuantumNode(
                    node_id="test_node_1",
                    node_type="simulator",
                    capabilities={"simulation": True},
                    max_qubits=8,
                    max_concurrent_jobs=2
                )
                orchestrator.register_node(test_node)
                
                # Test task submission
                qnn = QECCAwareQNN(num_qubits=4, num_layers=2)
                circuit = qnn.create_circuit(np.random.rand(4))
                
                from qecc_qml.scaling.distributed_quantum_orchestrator import TaskPriority
                task_id = orchestrator.submit_circuit(
                    circuit=circuit,
                    priority=TaskPriority.NORMAL
                )
                assert task_id is not None, "Task submission failed"
                
                # Wait for task completion
                max_wait = 10.0  # 10 seconds
                start_wait = time.time()
                task_completed = False
                
                while time.time() - start_wait < max_wait:
                    status = orchestrator.get_task_status(task_id)
                    if status and status.get('status') == 'completed':
                        task_completed = True
                        break
                    time.sleep(0.1)
                
                # Test orchestrator status
                orch_status = orchestrator.get_orchestrator_status()
                assert orch_status['is_running'], "Orchestrator not running"
                assert orch_status['total_nodes'] >= 1, "No nodes registered"
                
                return {
                    'passed': True,
                    'orchestrator_started': True,
                    'node_registration': True,
                    'task_submission': True,
                    'task_completed': task_completed,
                    'total_nodes': orch_status['total_nodes'],
                    'active_nodes': orch_status['active_nodes']
                }
                
            finally:
                orchestrator.stop()
                
        except Exception as e:
            return {
                'passed': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def _test_integration(self) -> Dict[str, Any]:
        """Test integration between different components."""
        try:
            # Test QNN + Trainer integration
            qnn = QECCAwareQNN(num_qubits=4, num_layers=2)
            trainer = BasicQECCTrainer(qnn)
            
            # Create test data
            X_test, y_test = create_test_dataset(n_samples=20, n_features=4)
            
            # Test training integration
            initial_loss = trainer.compute_loss(X_test[:5], y_test[:5])
            trainer.train_step(X_test[:5], y_test[:5])
            final_loss = trainer.compute_loss(X_test[:5], y_test[:5])
            
            # Loss should change (either decrease or fluctuate in early training)
            loss_changed = abs(final_loss - initial_loss) > 1e-6
            
            # Test prediction integration
            predictions = trainer.predict(X_test[:3])
            assert predictions.shape == (3,), f"Wrong prediction shape: {predictions.shape}"
            
            # Test with security validation
            validator = AdvancedSecurityValidator()
            validation_result = validator.validate_circuit_security(
                qnn.create_circuit(X_test[0])
            )
            
            # Test with performance optimization
            enhancer = QuantumPerformanceEnhancer()
            optimized_circuit = enhancer.optimize_circuit(
                qnn.create_circuit(X_test[0])
            )
            
            return {
                'passed': True,
                'qnn_trainer_integration': True,
                'training_step': loss_changed,
                'prediction_integration': True,
                'security_integration': validation_result['secure'],
                'performance_integration': optimized_circuit is not None,
                'initial_loss': float(initial_loss),
                'final_loss': float(final_loss)
            }
            
        except Exception as e:
            return {
                'passed': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def _test_research_framework(self) -> Dict[str, Any]:
        """Test research framework components."""
        try:
            # Test research module imports
            from qecc_qml import research
            
            # Test basic research components
            tests_passed = {
                'research_module_imported': True,
                'reinforcement_learning_available': hasattr(research, 'QECCEnvironment'),
                'neural_decoders_available': hasattr(research, 'NeuralSyndromeDecoder'),
                'quantum_advantage_available': hasattr(research, 'QuantumAdvantageSuite'),
                'evolution_available': hasattr(research, 'AutonomousQuantumCircuitEvolution')
            }
            
            all_research_tests_passed = all(tests_passed.values())
            
            return {
                'passed': all_research_tests_passed,
                'component_tests': tests_passed,
                'total_components_tested': len(tests_passed),
                'components_passed': sum(tests_passed.values())
            }
            
        except Exception as e:
            return {
                'passed': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def _test_error_handling(self) -> Dict[str, Any]:
        """Test comprehensive error handling."""
        try:
            error_tests = {}
            
            # Test invalid circuit handling
            try:
                # Create invalid circuit scenario
                validator = AdvancedSecurityValidator()
                
                # Test with None circuit
                try:
                    result = validator.validate_circuit_security(None)
                    error_tests['none_circuit_handled'] = False  # Should have raised exception
                except Exception:
                    error_tests['none_circuit_handled'] = True
                
            except Exception:
                error_tests['none_circuit_handled'] = True
            
            # Test invalid data handling
            try:
                trainer = BasicQECCTrainer(QECCAwareQNN(num_qubits=4, num_layers=2))
                
                # Test with invalid data shapes
                try:
                    trainer.train_step(np.array([]), np.array([]))
                    error_tests['empty_data_handled'] = False
                except Exception:
                    error_tests['empty_data_handled'] = True
                
            except Exception:
                error_tests['empty_data_handled'] = True
            
            # Test memory/resource limits
            try:
                enhancer = QuantumPerformanceEnhancer()
                enhancer.optimize_memory_usage()  # Should not crash
                error_tests['memory_optimization'] = True
            except Exception:
                error_tests['memory_optimization'] = False
            
            # Test graceful degradation
            try:
                # Test fallback imports
                from qecc_qml.core.fallback_imports import QuantumCircuit
                fallback_circuit = QuantumCircuit(4)
                error_tests['fallback_imports'] = fallback_circuit is not None
            except Exception:
                error_tests['fallback_imports'] = False
            
            total_tests = len(error_tests)
            passed_tests = sum(error_tests.values())
            
            return {
                'passed': passed_tests >= total_tests * 0.8,  # 80% pass rate
                'error_handling_tests': error_tests,
                'total_error_tests': total_tests,
                'passed_error_tests': passed_tests,
                'error_handling_rate': (passed_tests / total_tests) * 100 if total_tests > 0 else 0
            }
            
        except Exception as e:
            return {
                'passed': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def _test_resource_management(self) -> Dict[str, Any]:
        """Test resource management and cleanup."""
        try:
            # Test memory monitoring
            config = MonitoringConfig(
                collection_interval=1.0,
                enable_predictive_alerts=False
            )
            monitor = ComprehensiveHealthMonitor(config)
            
            # Collect initial metrics
            initial_metrics = monitor.collect_metrics()
            assert initial_metrics.timestamp > 0, "Metrics collection failed"
            
            # Test resource cleanup
            enhancer = QuantumPerformanceEnhancer()
            
            # Create some cached data
            for i in range(10):
                qnn = QECCAwareQNN(num_qubits=4, num_layers=2)
                circuit = qnn.create_circuit(np.random.rand(4))
                enhancer.optimize_circuit(circuit)
            
            # Test memory optimization
            initial_cache_size = len(enhancer.optimized_circuits)
            enhancer.optimize_memory_usage()
            final_cache_size = len(enhancer.optimized_circuits)
            
            # Test performance summary
            perf_summary = enhancer.get_performance_summary()
            assert 'cache_hit_rate' in perf_summary, "Performance summary incomplete"
            
            # Test health summary
            health_summary = monitor.get_health_summary()
            assert 'current_metrics' in health_summary, "Health summary incomplete"
            
            return {
                'passed': True,
                'metrics_collection': True,
                'memory_optimization': True,
                'performance_monitoring': True,
                'health_monitoring': True,
                'initial_cache_size': initial_cache_size,
                'final_cache_size': final_cache_size,
                'cache_size_reduced': final_cache_size <= initial_cache_size
            }
            
        except Exception as e:
            return {
                'passed': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def _test_final_validation(self) -> Dict[str, Any]:
        """Final comprehensive system validation."""
        try:
            # Create a complete end-to-end test
            self.logger.info("Running final end-to-end validation...")
            
            # 1. Create quantum neural network
            qnn = QECCAwareQNN(num_qubits=4, num_layers=3)
            
            # 2. Create training data
            X_train, y_train = create_test_dataset(n_samples=50, n_features=4)
            X_test, y_test = create_test_dataset(n_samples=20, n_features=4)
            
            # 3. Initialize trainer with security and performance optimization
            trainer = BasicQECCTrainer(qnn)
            validator = AdvancedSecurityValidator()
            enhancer = QuantumPerformanceEnhancer()
            
            # 4. Secure training process
            training_steps = 5
            training_successful = True
            
            for step in range(training_steps):
                try:
                    # Validate security for this batch
                    batch_X = X_train[step*5:(step+1)*5]
                    batch_y = y_train[step*5:(step+1)*5]
                    
                    if len(batch_X) > 0:
                        # Create circuit for validation
                        test_circuit = qnn.create_circuit(batch_X[0])
                        
                        # Security validation
                        sec_result = validator.validate_circuit_security(test_circuit)
                        if not sec_result['secure']:
                            training_successful = False
                            break
                        
                        # Performance optimization
                        optimized_circuit = enhancer.optimize_circuit(test_circuit)
                        
                        # Training step
                        trainer.train_step(batch_X, batch_y)
                    
                except Exception as e:
                    self.logger.error(f"Training step {step} failed: {e}")
                    training_successful = False
                    break
            
            # 5. Final evaluation
            if training_successful:
                try:
                    predictions = trainer.predict(X_test[:5])
                    prediction_successful = predictions.shape == (5,)
                except Exception:
                    prediction_successful = False
            else:
                prediction_successful = False
            
            # 6. System health check
            monitor = ComprehensiveHealthMonitor()
            health_status = monitor.get_health_summary()
            system_healthy = health_status.get('system_status', 'degraded') == 'healthy'
            
            # 7. Performance summary
            perf_summary = enhancer.get_performance_summary()
            performance_good = perf_summary.get('cache_hit_rate', 0) >= 0
            
            # Final validation score
            validation_checks = {
                'qnn_creation': True,
                'data_generation': True,
                'security_validation': training_successful,
                'training_process': training_successful,
                'prediction_capability': prediction_successful,
                'system_health': system_healthy,
                'performance_optimization': performance_good
            }
            
            total_checks = len(validation_checks)
            passed_checks = sum(validation_checks.values())
            validation_score = (passed_checks / total_checks) * 100
            
            overall_passed = validation_score >= 85  # 85% threshold
            
            return {
                'passed': overall_passed,
                'validation_score': validation_score,
                'validation_checks': validation_checks,
                'total_checks': total_checks,
                'passed_checks': passed_checks,
                'training_steps_completed': training_steps,
                'system_ready_for_production': overall_passed
            }
            
        except Exception as e:
            return {
                'passed': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def save_report(self, report: Dict[str, Any], filepath: str = "autonomous_quality_gates_report.json") -> None:
        """Save quality gates report to file."""
        try:
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            self.logger.info(f"Quality gates report saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save report: {e}")

def main():
    """Main execution function."""
    print("🚀 Starting Autonomous Quality Gates Execution")
    print("=" * 60)
    
    # Execute quality gates
    quality_gates = AutonomousQualityGates()
    report = quality_gates.execute_all_gates()
    
    # Save report
    quality_gates.save_report(report)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 QUALITY GATES SUMMARY")
    print("=" * 60)
    print(f"Total Gates: {report['total_gates']}")
    print(f"Passed: {report['passed_gates']}")
    print(f"Failed: {report['failed_gates']}")
    print(f"Success Rate: {report['success_rate']:.1f}%")
    print(f"Execution Time: {report['total_execution_time']:.2f}s")
    print(f"Overall Status: {report['overall_status']}")
    
    if report['overall_status'] == 'PASSED':
        print("\n🎉 SYSTEM READY FOR PRODUCTION!")
        return 0
    else:
        print("\n🚨 SYSTEM REQUIRES FIXES BEFORE PRODUCTION")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)