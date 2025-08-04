#!/usr/bin/env python3
"""
Comprehensive test suite with edge cases and robustness testing.
"""

import unittest
import numpy as np
import tempfile
import os
import sys
from pathlib import Path
import warnings
import pickle
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qecc_qml.core.quantum_nn import QECCAwareQNN
from qecc_qml.core.noise_models import NoiseModel
from qecc_qml.core.error_correction import SimpleRepetitionCode
from qecc_qml.codes.surface_code import SurfaceCode
from qecc_qml.codes.color_code import ColorCode
from qecc_qml.training.qecc_trainer import QECCTrainer
from qecc_qml.training.optimizers import NoiseAwareAdam
from qecc_qml.training.loss_functions import QuantumCrossEntropy
from qecc_qml.evaluation.benchmarks import NoiseBenchmark
from qecc_qml.evaluation.fidelity_tracker import FidelityTracker
from qecc_qml.utils.validation import ValidationError, validate_qnn_config, validate_training_data
from qecc_qml.utils.security import sanitize_input, SecurityError
from qecc_qml.utils.diagnostics import SystemDiagnostics, HealthChecker


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""
    
    def test_minimal_qnn(self):
        """Test QNN with minimal configuration."""
        qnn = QECCAwareQNN(num_qubits=1, num_layers=1)
        
        self.assertEqual(qnn.num_qubits, 1)
        self.assertEqual(qnn.num_layers, 1)
        self.assertGreater(qnn.get_num_parameters(), 0)
        
        # Test forward pass with minimal data
        X = np.array([[0.1]])
        params = np.random.uniform(-np.pi, np.pi, qnn.get_num_parameters())
        result = qnn.forward(X, params, shots=10)
        
        self.assertEqual(result.shape[0], 1)
        self.assertAlmostEqual(np.sum(result[0]), 1.0, places=1)
    
    def test_large_qnn_configuration(self):
        """Test QNN with large but valid configuration."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress expected warnings
            
            qnn = QECCAwareQNN(num_qubits=10, num_layers=5)
            
            self.assertEqual(qnn.num_qubits, 10)
            self.assertEqual(qnn.num_layers, 5)
            
            # Should have many parameters
            self.assertGreater(qnn.get_num_parameters(), 100)
    
    def test_extreme_noise_model(self):
        """Test noise model with extreme but valid parameters."""
        # Very noisy model
        noisy_model = NoiseModel(
            gate_error_rate=0.5,  # 50% error rate
            readout_error_rate=0.5,
            T1=1e-6,  # 1 microsecond
            T2=0.5e-6  # 0.5 microseconds
        )
        
        self.assertEqual(noisy_model.gate_error_rate, 0.5)
        self.assertLess(noisy_model.T2, noisy_model.T1)
        
        # Very clean model
        clean_model = NoiseModel.ideal()
        
        self.assertEqual(clean_model.gate_error_rate, 0.0)
        self.assertEqual(clean_model.T1, float('inf'))
    
    def test_single_sample_training(self):
        """Test training with single sample."""
        qnn = QECCAwareQNN(num_qubits=2, num_layers=1)
        trainer = QECCTrainer(qnn=qnn, shots=10, learning_rate=0.1)
        
        # Single sample
        X = np.array([[0.1, 0.2]])
        y = np.array([0])
        
        # Should not crash
        history = trainer.fit(X, y, epochs=1, verbose=False)
        
        self.assertEqual(len(history['loss']), 1)
        self.assertEqual(len(history['accuracy']), 1)
    
    def test_empty_data_handling(self):
        """Test handling of empty datasets."""
        with self.assertRaises(ValidationError):
            validate_training_data(np.array([]), np.array([]))
    
    def test_mismatched_data_sizes(self):
        """Test handling of mismatched feature/label sizes."""
        X = np.random.rand(10, 4)
        y = np.random.randint(0, 2, 5)  # Wrong size
        
        with self.assertRaises(ValidationError):
            validate_training_data(X, y)
    
    def test_nan_and_inf_handling(self):
        """Test handling of NaN and infinite values."""
        # NaN in features
        X_nan = np.array([[1.0, np.nan, 2.0]])
        y = np.array([0])
        
        with self.assertRaises(ValidationError):
            validate_training_data(X_nan, y)
        
        # Infinite values in features
        X_inf = np.array([[1.0, np.inf, 2.0]])
        
        with self.assertRaises(ValidationError):
            validate_training_data(X_inf, y)
    
    def test_extreme_parameter_values(self):
        """Test handling of extreme parameter values."""
        qnn = QECCAwareQNN(num_qubits=2, num_layers=1)
        
        # Very large parameters
        large_params = np.full(qnn.get_num_parameters(), 100 * np.pi)
        X = np.array([[0.1, 0.2]])
        
        # Should not crash but may give warning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = qnn.forward(X, large_params, shots=10)
        
        self.assertEqual(result.shape[0], 1)


class TestErrorCorrection(unittest.TestCase):
    """Test error correction edge cases."""
    
    def test_invalid_surface_code_distance(self):
        """Test invalid surface code distances."""
        # Even distance
        with self.assertRaises(ValueError):
            SurfaceCode(distance=4)
        
        # Distance too small
        with self.assertRaises(ValueError):
            SurfaceCode(distance=1)
    
    def test_error_correction_with_extreme_noise(self):
        """Test error correction under extreme noise."""
        qnn = QECCAwareQNN(num_qubits=2, num_layers=1)
        rep_code = SimpleRepetitionCode()
        qnn.add_error_correction(rep_code)
        
        # Extreme noise that exceeds threshold
        extreme_noise = NoiseModel(gate_error_rate=0.9)
        
        trainer = QECCTrainer(qnn=qnn, noise_model=extreme_noise, shots=10)
        
        # Should still work but with poor performance
        X = np.array([[0.1, 0.2]])
        y = np.array([0])
        
        history = trainer.fit(X, y, epochs=1, verbose=False)
        self.assertIsInstance(history['logical_error_rate'][0], float)
    
    def test_syndrome_decoding_edge_cases(self):
        """Test syndrome decoding with edge cases."""
        rep_code = SimpleRepetitionCode()
        
        # Empty syndrome
        errors = rep_code.decode_syndrome("")
        self.assertEqual(len(errors), 0)
        
        # Invalid syndrome length
        errors = rep_code.decode_syndrome("0")  # Too short
        self.assertEqual(len(errors), 0)
        
        # All zeros (no error)
        errors = rep_code.decode_syndrome("00")
        self.assertEqual(len(errors), 0)


class TestSecurity(unittest.TestCase):
    """Test security and input sanitization."""
    
    def test_malicious_input_detection(self):
        """Test detection of potentially malicious inputs."""
        dangerous_strings = [
            "__import__('os').system('rm -rf /')",
            "exec('print(hello)')",
            "eval('1+1')",
            "import os; os.system('ls')",
            "open('/etc/passwd', 'r')",
        ]
        
        for dangerous in dangerous_strings:
            with self.assertRaises(SecurityError):
                sanitize_input(dangerous, 'string')
    
    def test_path_traversal_prevention(self):
        """Test prevention of path traversal attacks."""
        from qecc_qml.utils.security import validate_file_path
        
        dangerous_paths = [
            "../../../etc/passwd",
            "/etc/passwd",
            "~/../../etc/passwd",
            "file.txt; rm -rf /",
            "file.txt|cat /etc/passwd",
        ]
        
        for dangerous_path in dangerous_paths:
            with self.assertRaises(SecurityError):
                validate_file_path(dangerous_path, allow_absolute=False)
    
    def test_large_input_handling(self):
        """Test handling of excessively large inputs."""
        # Very large string
        large_string = "a" * 10000
        
        with self.assertRaises(SecurityError):
            sanitize_input(large_string, 'string', max_length=1000)
        
        # Very large array
        large_array = np.ones((1000, 1000))  # 1M elements
        
        with self.assertRaises(SecurityError):
            sanitize_input(large_array, 'array', max_size=100000)
    
    def test_numeric_bounds_checking(self):
        """Test numeric bounds checking."""
        # Value too large
        with self.assertRaises(SecurityError):
            sanitize_input(1000.0, 'numeric', max_val=100.0)
        
        # Value too small
        with self.assertRaises(SecurityError):
            sanitize_input(-1000.0, 'numeric', min_val=-100.0)


class TestRobustness(unittest.TestCase):
    """Test system robustness under stress."""
    
    def test_memory_stress(self):
        """Test behavior under memory stress."""
        # Create large QNN but don't execute expensive operations
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            qnn = QECCAwareQNN(num_qubits=8, num_layers=3)
            self.assertGreater(qnn.get_num_parameters(), 50)
    
    def test_concurrent_operations(self):
        """Test concurrent operations (basic)."""
        import threading
        import time
        
        results = []
        errors = []
        
        def worker():
            try:
                qnn = QECCAwareQNN(num_qubits=2, num_layers=1)
                X = np.array([[0.1, 0.2]])
                params = np.random.uniform(-np.pi, np.pi, qnn.get_num_parameters())
                result = qnn.forward(X, params, shots=10)
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=10)
        
        # Check results
        self.assertEqual(len(errors), 0, f"Errors in concurrent execution: {errors}")
        self.assertGreater(len(results), 0)
    
    def test_repeated_operations(self):
        """Test repeated operations for stability."""
        qnn = QECCAwareQNN(num_qubits=2, num_layers=1)
        X = np.array([[0.1, 0.2]])
        params = np.random.uniform(-np.pi, np.pi, qnn.get_num_parameters())
        
        # Repeat many times
        for i in range(20):
            result = qnn.forward(X, params, shots=10)
            self.assertEqual(result.shape[0], 1)
            # Results should be probabilistic but bounded
            self.assertTrue(np.all(result >= 0))
            self.assertTrue(np.all(result <= 1))
    
    def test_configuration_validation_stress(self):
        """Test configuration validation with many invalid inputs."""
        invalid_configs = [
            {'num_qubits': -1, 'num_layers': 1},
            {'num_qubits': 0, 'num_layers': 1},
            {'num_qubits': 1, 'num_layers': -1},
            {'num_qubits': 1, 'num_layers': 0},
            {'num_qubits': 1.5, 'num_layers': 1},  # Non-integer
            {'num_qubits': 1, 'num_layers': 1, 'entanglement': 'invalid'},
            {'num_qubits': 1, 'num_layers': 1, 'feature_map': 'invalid'},
        ]
        
        for config in invalid_configs:
            with self.assertRaises((ValidationError, ValueError, TypeError)):
                validate_qnn_config(**config)


class TestDiagnostics(unittest.TestCase):
    """Test diagnostics and health checking."""
    
    def test_system_diagnostics(self):
        """Test system diagnostics."""
        diagnostics = SystemDiagnostics()
        results = diagnostics.run_all_checks()
        
        self.assertGreater(len(results), 0)
        
        # Should have summary as first result
        summary = results[0]
        self.assertEqual(summary.name, 'summary')
        self.assertIn(summary.status, ['pass', 'warning', 'fail'])
    
    def test_health_checker(self):
        """Test health checker."""
        checker = HealthChecker()
        health = checker.quick_health_check()
        
        self.assertIn('status', health)
        self.assertIn('memory_usage_percent', health)
        self.assertIn('cpu_usage_percent', health)
        
        # Should be able to determine if healthy
        is_healthy = checker.is_healthy()
        self.assertIsInstance(is_healthy, bool)
    
    def test_resource_monitoring(self):
        """Test resource usage monitoring."""
        checker = HealthChecker()
        usage = checker.get_resource_usage()
        
        required_metrics = ['memory_percent', 'cpu_percent', 'disk_percent']
        for metric in required_metrics:
            self.assertIn(metric, usage)
            self.assertIsInstance(usage[metric], (int, float))
            self.assertGreaterEqual(usage[metric], 0)


class TestPersistence(unittest.TestCase):
    """Test data persistence and serialization."""
    
    def test_model_saving_and_loading(self):
        """Test model saving and loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and train a small model
            qnn = QECCAwareQNN(num_qubits=2, num_layers=1)
            trainer = QECCTrainer(qnn=qnn, shots=10)
            
            X = np.array([[0.1, 0.2], [0.3, 0.4]])
            y = np.array([0, 1])
            
            trainer.fit(X, y, epochs=1, verbose=False)
            
            # Save model
            save_path = os.path.join(tmpdir, 'test_model.pkl')
            trainer.save_model(save_path)
            
            # Check file exists
            self.assertTrue(os.path.exists(save_path))
            self.assertGreater(os.path.getsize(save_path), 0)
    
    def test_benchmark_results_export(self):
        """Test benchmark results export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            qnn = QECCAwareQNN(num_qubits=2, num_layers=1)
            benchmark = NoiseBenchmark(
                model=qnn,
                noise_levels=np.array([0.001, 0.01]),
                shots=10
            )
            
            X = np.array([[0.1, 0.2]])
            y = np.array([0])
            params = np.random.uniform(-np.pi, np.pi, qnn.get_num_parameters())
            
            # Run benchmark
            results = benchmark.run(X, y, model_parameters=params, verbose=False)
            
            # Export to different formats
            for format_type in ['csv', 'json']:
                export_path = os.path.join(tmpdir, f'results.{format_type}')
                benchmark.export_results(export_path, format=format_type)
                
                self.assertTrue(os.path.exists(export_path))
                self.assertGreater(os.path.getsize(export_path), 0)
    
    def test_fidelity_tracker_export(self):
        """Test fidelity tracker data export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = FidelityTracker()
            
            # Add some tracking data
            for i in range(5):
                tracker.update(
                    circuit_fidelity=0.9 - i * 0.05,
                    physical_error_rate=0.001 + i * 0.001
                )
            
            # Export data
            export_path = os.path.join(tmpdir, 'fidelity_data.json')
            tracker.export_tracking_data(export_path)
            
            self.assertTrue(os.path.exists(export_path))
            
            # Verify content
            with open(export_path, 'r') as f:
                data = json.load(f)
            
            self.assertIn('fidelity_history', data)
            self.assertIn('error_rate_history', data)
            self.assertEqual(len(data['fidelity_history']), 5)


class TestPerformanceRegression(unittest.TestCase):
    """Test for performance regressions."""
    
    def test_qnn_forward_pass_timing(self):
        """Test QNN forward pass timing."""
        import time
        
        qnn = QECCAwareQNN(num_qubits=4, num_layers=2)
        X = np.random.uniform(0, np.pi, (5, 4))
        params = np.random.uniform(-np.pi, np.pi, qnn.get_num_parameters())
        
        # Warm up
        qnn.forward(X[:1], params, shots=10)
        
        # Time the operation
        start_time = time.time()
        result = qnn.forward(X, params, shots=50)
        elapsed = time.time() - start_time
        
        # Should complete within reasonable time (adjust threshold as needed)
        self.assertLess(elapsed, 10.0, "Forward pass took too long")
        self.assertEqual(result.shape[0], 5)
    
    def test_training_performance(self):
        """Test training performance."""
        import time
        
        qnn = QECCAwareQNN(num_qubits=2, num_layers=1)
        trainer = QECCTrainer(qnn=qnn, shots=20, learning_rate=0.1)
        
        X = np.random.uniform(0, np.pi, (10, 2))
        y = np.random.randint(0, 2, 10)
        
        # Time training
        start_time = time.time()
        history = trainer.fit(X, y, epochs=2, verbose=False)
        elapsed = time.time() - start_time
        
        # Should complete within reasonable time
        self.assertLess(elapsed, 30.0, "Training took too long")
        self.assertEqual(len(history['loss']), 2)


if __name__ == '__main__':
    # Set up test environment
    import logging
    logging.basicConfig(level=logging.WARNING)  # Reduce noise during testing
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestEdgeCases,
        TestErrorCorrection,
        TestSecurity,
        TestRobustness,
        TestDiagnostics,
        TestPersistence,
        TestPerformanceRegression,
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestClass(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(suite)
    
    # Print detailed summary
    print(f"\n{'='*60}")
    print(f"Comprehensive Test Suite Results")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    
    if result.failures:
        print(f"\nFailures:")
        for test, traceback in result.failures:
            print(f"  ❌ {test}")
    
    if result.errors:
        print(f"\nErrors:")
        for test, traceback in result.errors:
            print(f"  💥 {test}")
    
    success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / 
                   result.testsRun * 100) if result.testsRun > 0 else 0
    
    print(f"\nSuccess rate: {success_rate:.1f}%")
    
    if result.wasSuccessful():
        print(f"\n🎉 All tests passed! System is robust and ready.")
        sys.exit(0)
    else:
        print(f"\n⚠️  Some tests failed. Please review and fix issues.")
        sys.exit(1)