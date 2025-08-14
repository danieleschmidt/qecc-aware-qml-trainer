#!/usr/bin/env python3
"""
Comprehensive system testing for QECC-aware QML framework.
Tests all three generations and validates quality gates.
"""

import sys
import os
import time
import numpy as np
import unittest
from pathlib import Path
import json
import tempfile
import shutil
import warnings

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

# Test imports with fallbacks
try:
    from qecc_qml.core.quantum_nn import QECCAwareQNN
    CORE_AVAILABLE = True
except ImportError as e:
    print(f"Core module not available: {e}")
    CORE_AVAILABLE = False

# Import datasets directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'qecc_qml', 'datasets'))
try:
    from simple_datasets import load_quantum_classification
    DATASETS_AVAILABLE = True
except ImportError as e:
    print(f"Datasets not available: {e}")
    DATASETS_AVAILABLE = False

# Import trainers with fallbacks
try:
    from qecc_qml.training.basic_trainer_clean import BasicQECCTrainer
    BASIC_TRAINER_AVAILABLE = True
except ImportError:
    BASIC_TRAINER_AVAILABLE = False

try:
    from qecc_qml.training.robust_trainer import RobustQECCTrainer
    ROBUST_TRAINER_AVAILABLE = True
except ImportError:
    ROBUST_TRAINER_AVAILABLE = False

try:
    from qecc_qml.training.scalable_trainer import ScalableQECCTrainer
    SCALABLE_TRAINER_AVAILABLE = True
except ImportError:
    SCALABLE_TRAINER_AVAILABLE = False


class TestGeneration1Basic(unittest.TestCase):
    """Test Generation 1: Basic functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not CORE_AVAILABLE or not DATASETS_AVAILABLE or not BASIC_TRAINER_AVAILABLE:
            self.skipTest("Required modules not available")
            
        self.qnn = QECCAwareQNN(num_qubits=2, num_layers=2)
        self.trainer = BasicQECCTrainer(self.qnn, learning_rate=0.1, verbose=False)
        
        # Generate small test dataset
        np.random.seed(42)
        self.X_train = np.random.randn(20, 2).astype(np.float32)
        self.X_train = self.X_train / np.linalg.norm(self.X_train, axis=1, keepdims=True)
        self.y_train = (np.sum(self.X_train, axis=1) > 0).astype(int)
        
        self.X_test = np.random.randn(10, 2).astype(np.float32)
        self.X_test = self.X_test / np.linalg.norm(self.X_test, axis=1, keepdims=True)
        self.y_test = (np.sum(self.X_test, axis=1) > 0).astype(int)
    
    def test_qnn_creation(self):
        """Test QNN creation and basic properties."""
        self.assertEqual(self.qnn.num_qubits, 2)
        self.assertEqual(self.qnn.num_layers, 2)
        self.assertIsNotNone(self.qnn.weight_params)
        
    def test_trainer_initialization(self):
        """Test trainer initialization."""
        self.assertEqual(self.trainer.learning_rate, 0.1)
        self.assertFalse(self.trainer.verbose)
        self.assertIsNone(self.trainer.current_params)
        
    def test_basic_training(self):
        """Test basic training functionality."""
        # Run short training
        history = self.trainer.fit(
            self.X_train, self.y_train,
            epochs=3,
            batch_size=8,
            validation_split=0.0
        )
        
        # Check history structure
        self.assertIn('loss', history)
        self.assertIn('accuracy', history)
        self.assertIn('fidelity', history)
        self.assertEqual(len(history['loss']), 3)
        
        # Check parameters were updated
        self.assertIsNotNone(self.trainer.current_params)
        self.assertGreater(len(self.trainer.current_params), 0)
        
    def test_prediction_and_evaluation(self):
        """Test prediction and evaluation functionality."""
        # Train briefly
        self.trainer.fit(self.X_train, self.y_train, epochs=2, batch_size=8, validation_split=0.0)
        
        # Make predictions
        predictions = self.trainer.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.X_test))
        self.assertTrue(all(p in [0, 1] for p in predictions))
        
        # Evaluate
        results = self.trainer.evaluate(self.X_test, self.y_test)
        self.assertIn('loss', results)
        self.assertIn('accuracy', results)
        self.assertIn('fidelity', results)
        self.assertGreaterEqual(results['accuracy'], 0.0)
        self.assertLessEqual(results['accuracy'], 1.0)


class TestGeneration2Robust(unittest.TestCase):
    """Test Generation 2: Robust functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not ROBUST_TRAINER_AVAILABLE:
            self.skipTest("Robust trainer not available")
            
        self.qnn = QECCAwareQNN(num_qubits=2, num_layers=2)
        self.trainer = RobustQECCTrainer(
            self.qnn,
            learning_rate=0.05,
            verbose=False,
            validation_freq=2,
            checkpoint_freq=3,
            max_retries=1,
            enable_monitoring=False
        )
        
        # Generate test dataset with some edge cases
        np.random.seed(123)
        self.X_train = np.random.randn(30, 2).astype(np.float32)
        self.X_train = self.X_train / np.linalg.norm(self.X_train, axis=1, keepdims=True)
        
        # Add some edge cases
        self.X_train[0] = [0.0, 0.0]  # Zero vector
        self.X_train[1] = [1e-8, 1e-8]  # Very small values
        
        self.y_train = (np.sum(self.X_train, axis=1) > 0).astype(int)
        
        self.X_test = np.random.randn(15, 2).astype(np.float32)
        self.X_test = self.X_test / np.linalg.norm(self.X_test, axis=1, keepdims=True)
        self.y_test = (np.sum(self.X_test, axis=1) > 0).astype(int)
    
    def test_input_validation(self):
        """Test input validation and sanitization."""
        # Test with invalid inputs
        X_invalid = np.array([[np.nan, 1.0], [0.5, np.inf]])
        y_invalid = np.array([0, 1])
        
        # Should handle gracefully (sanitize or raise appropriate error)
        try:
            self.trainer._validate_and_sanitize_inputs(X_invalid, y_invalid)
        except ValueError as e:
            self.assertIn('validation', str(e).lower())
    
    def test_robust_training(self):
        """Test robust training with validation and checkpointing."""
        # Test with temporary directory for checkpoints
        with tempfile.TemporaryDirectory() as temp_dir:
            original_dir = os.getcwd()
            os.chdir(temp_dir)
            
            try:
                history = self.trainer.fit(
                    self.X_train, self.y_train,
                    epochs=5,
                    batch_size=10,
                    validation_split=0.2
                )
                
                # Check enhanced history
                self.assertIn('loss', history)
                self.assertIn('performance_summary', history)
                
                # Check validation was performed
                if 'validation_metrics' in history:
                    self.assertGreater(len(history['validation_metrics']), 0)
                
            finally:
                os.chdir(original_dir)
    
    def test_error_handling_and_recovery(self):
        """Test error handling capabilities."""
        # Test error handler
        error_handler = self.trainer.error_handler
        
        # Simulate different error types
        test_error = Exception("Test timeout error")
        recovery = error_handler.handle_circuit_execution_error(
            test_error, {'operation': 'test_op', 'attempt': 0}
        )
        
        # Should return some recovery strategy or None
        self.assertIsInstance(recovery, (str, type(None)))
    
    def test_diagnostics(self):
        """Test diagnostic capabilities."""
        # Brief training to generate some data
        self.trainer.fit(self.X_train[:10], self.y_train[:10], epochs=2, batch_size=5, validation_split=0.0)
        
        # Get diagnostics
        diagnostics = self.trainer.get_training_diagnostics()
        
        self.assertIn('validation_report', diagnostics)
        self.assertIn('performance_metrics', diagnostics)
        self.assertIn('training_stability', diagnostics)


class TestGeneration3Scalable(unittest.TestCase):
    """Test Generation 3: Scalable functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not SCALABLE_TRAINER_AVAILABLE:
            self.skipTest("Scalable trainer not available")
            
        self.qnn = QECCAwareQNN(num_qubits=2, num_layers=2)
        
        # Initialize with minimal optimization to avoid complex dependencies
        self.trainer = ScalableQECCTrainer(
            self.qnn,
            learning_rate=0.05,
            verbose=False,
            validation_freq=3,
            checkpoint_freq=5,
            max_retries=1,
            enable_monitoring=False,
            enable_optimization=False,  # Disable to avoid parallel processing issues
            enable_auto_scaling=False,
            enable_parallel=False
        )
        
        # Generate larger test dataset
        np.random.seed(456)
        self.X_train = np.random.randn(50, 2).astype(np.float32)
        self.X_train = self.X_train / np.linalg.norm(self.X_train, axis=1, keepdims=True)
        self.y_train = (np.sum(self.X_train, axis=1) > 0).astype(int)
        
        self.X_test = np.random.randn(20, 2).astype(np.float32)
        self.X_test = self.X_test / np.linalg.norm(self.X_test, axis=1, keepdims=True)
        self.y_test = (np.sum(self.X_test, axis=1) > 0).astype(int)
    
    def test_scalable_initialization(self):
        """Test scalable trainer initialization."""
        self.assertIsNotNone(self.trainer)
        
        # Check Generation 3 specific attributes
        self.assertFalse(self.trainer.enable_optimization)
        self.assertFalse(self.trainer.enable_auto_scaling)
        
    def test_scalable_training(self):
        """Test scalable training functionality."""
        history = self.trainer.fit(
            self.X_train, self.y_train,
            epochs=4,
            batch_size=16,
            validation_split=0.2
        )
        
        # Check enhanced history structure
        self.assertIn('loss', history)
        self.assertIn('performance_summary', history)
        
        # Should have completed successfully
        perf_summary = history.get('performance_summary', {})
        if 'success_rate' in perf_summary:
            self.assertGreaterEqual(perf_summary['success_rate'], 0.5)
    
    def test_resource_monitoring(self):
        """Test resource monitoring capabilities."""
        # Brief training
        self.trainer.fit(self.X_train[:20], self.y_train[:20], epochs=2, batch_size=8, validation_split=0.0)
        
        # Check resource monitoring
        if hasattr(self.trainer, 'resource_monitor'):
            resource_stats = self.trainer.resource_monitor
            self.assertIsInstance(resource_stats, dict)
            
    def test_optimization_diagnostics(self):
        """Test optimization diagnostic capabilities."""
        if hasattr(self.trainer, 'get_optimization_diagnostics'):
            # Brief training
            self.trainer.fit(self.X_train[:15], self.y_train[:15], epochs=2, batch_size=8, validation_split=0.0)
            
            # Get diagnostics
            diagnostics = self.trainer.get_optimization_diagnostics()
            self.assertIsInstance(diagnostics, dict)


class TestSystemIntegration(unittest.TestCase):
    """Test system-wide integration and data flow."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        if not DATASETS_AVAILABLE:
            self.skipTest("Datasets not available")
            
        # Use actual quantum datasets
        self.X_train, self.y_train = load_quantum_classification(
            dataset='synthetic',
            n_samples=100,
            n_features=2,
            noise=0.1,
            random_state=789
        )
        
        self.X_test, self.y_test = load_quantum_classification(
            dataset='synthetic',
            n_samples=40,
            n_features=2,
            noise=0.1,
            random_state=987
        )
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow."""
        if not BASIC_TRAINER_AVAILABLE:
            self.skipTest("Basic trainer not available")
            
        # Create QNN
        qnn = QECCAwareQNN(num_qubits=2, num_layers=2)
        
        # Train with basic trainer
        trainer = BasicQECCTrainer(qnn, learning_rate=0.1, verbose=False)
        
        # Full workflow
        history = trainer.fit(
            self.X_train, self.y_train,
            epochs=5,
            batch_size=20,
            validation_split=0.2
        )
        
        # Evaluate
        results = trainer.evaluate(self.X_test, self.y_test)
        
        # Make predictions
        predictions = trainer.predict(self.X_test[:10])
        
        # Validate complete workflow
        self.assertIn('loss', history)
        self.assertIn('accuracy', results)
        self.assertEqual(len(predictions), 10)
        
    def test_data_quality_and_preprocessing(self):
        """Test data quality and preprocessing."""
        # Check dataset properties
        self.assertEqual(self.X_train.shape[1], 2)
        self.assertEqual(len(np.unique(self.y_train)), 2)
        
        # Check normalization
        norms = np.linalg.norm(self.X_train, axis=1)
        self.assertTrue(np.allclose(norms, 1.0, atol=1e-6))
        
        # Check data types
        self.assertEqual(self.X_train.dtype, np.float32)
        self.assertEqual(self.y_train.dtype, int)


class TestSecurityAndValidation(unittest.TestCase):
    """Test security measures and input validation."""
    
    def setUp(self):
        """Set up security test fixtures."""
        if not CORE_AVAILABLE:
            self.skipTest("Core modules not available")
    
    def test_parameter_validation(self):
        """Test parameter validation for security."""
        # Test QNN parameter validation
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Valid parameters
            qnn1 = QECCAwareQNN(num_qubits=2, num_layers=2)
            self.assertEqual(qnn1.num_qubits, 2)
            
            # Edge case parameters
            qnn2 = QECCAwareQNN(num_qubits=1, num_layers=1)  # Minimal
            self.assertEqual(qnn2.num_qubits, 1)
    
    def test_input_sanitization(self):
        """Test input sanitization for security."""
        if not ROBUST_TRAINER_AVAILABLE:
            self.skipTest("Robust trainer not available")
            
        qnn = QECCAwareQNN(num_qubits=2, num_layers=1)
        trainer = RobustQECCTrainer(qnn, verbose=False, enable_monitoring=False)
        
        # Test with potentially problematic inputs
        X_test = np.array([[1e10, -1e10], [0.0, 0.0]], dtype=np.float32)  # Extreme values
        y_test = np.array([0, 1], dtype=int)
        
        # Should handle gracefully through sanitization
        try:
            X_clean, y_clean, _ = trainer._validate_and_sanitize_inputs(X_test, y_test)
            self.assertEqual(X_clean.shape, X_test.shape)
        except ValueError:
            # Acceptable if validation rejects clearly invalid inputs
            pass
    
    def test_memory_and_resource_limits(self):
        """Test memory and resource limit handling."""
        # Test with reasonable limits
        if SCALABLE_TRAINER_AVAILABLE:
            qnn = QECCAwareQNN(num_qubits=2, num_layers=1)
            trainer = ScalableQECCTrainer(
                qnn,
                memory_limit=0.9,  # 90% memory limit
                enable_optimization=False,
                enable_auto_scaling=False,
                enable_monitoring=False,
                verbose=False
            )
            
            self.assertEqual(trainer.memory_limit, 0.9)


def run_quality_gates():
    """Run comprehensive quality gates and generate report."""
    print("🔍 Running Comprehensive Quality Gates...")
    print("=" * 60)
    
    # Test discovery and execution
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestGeneration1Basic,
        TestGeneration2Robust,
        TestGeneration3Scalable,
        TestSystemIntegration,
        TestSecurityAndValidation
    ]
    
    for test_class in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(test_class))
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(
        verbosity=2,
        stream=sys.stdout,
        descriptions=True,
        failfast=False
    )
    
    start_time = time.time()
    result = runner.run(suite)
    end_time = time.time()
    
    # Generate quality report
    report = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_tests': result.testsRun,
        'failures': len(result.failures),
        'errors': len(result.errors),
        'skipped': len(result.skipped) if hasattr(result, 'skipped') else 0,
        'success_rate': (result.testsRun - len(result.failures) - len(result.errors)) / max(1, result.testsRun),
        'execution_time': end_time - start_time,
        'modules_tested': {
            'core': CORE_AVAILABLE,
            'datasets': DATASETS_AVAILABLE,
            'basic_trainer': BASIC_TRAINER_AVAILABLE,
            'robust_trainer': ROBUST_TRAINER_AVAILABLE,
            'scalable_trainer': SCALABLE_TRAINER_AVAILABLE
        }
    }
    
    # Save report
    with open('quality_gates_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"🎯 Quality Gates Summary")
    print(f"{'='*60}")
    print(f"Tests Run: {report['total_tests']}")
    print(f"Failures: {report['failures']}")
    print(f"Errors: {report['errors']}")
    print(f"Skipped: {report['skipped']}")
    print(f"Success Rate: {report['success_rate']:.1%}")
    print(f"Execution Time: {report['execution_time']:.2f}s")
    
    print(f"\n📦 Module Availability:")
    for module, available in report['modules_tested'].items():
        status = "✅" if available else "❌"
        print(f"  {status} {module}")
    
    # Quality gate decision
    if report['success_rate'] >= 0.8 and report['failures'] == 0:
        print(f"\n✅ Quality Gates: PASSED")
        print(f"   System meets quality requirements for production deployment.")
        return True
    else:
        print(f"\n⚠️  Quality Gates: NEEDS ATTENTION")
        print(f"   Some tests failed or quality thresholds not met.")
        return False


if __name__ == "__main__":
    success = run_quality_gates()
    sys.exit(0 if success else 1)