#!/usr/bin/env python3
"""
Basic functionality tests for QECC-aware QML library.
"""

import unittest
import numpy as np
import sys
import os

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


class TestQuantumNeuralNetwork(unittest.TestCase):
    """Test quantum neural network functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.qnn = QECCAwareQNN(
            num_qubits=4,
            num_layers=2,
            entanglement="circular"
        )
    
    def test_qnn_initialization(self):
        """Test QNN initialization."""
        self.assertEqual(self.qnn.num_qubits, 4)
        self.assertEqual(self.qnn.num_layers, 2)
        self.assertEqual(self.qnn.entanglement, "circular")
        self.assertIsNotNone(self.qnn._full_circuit)
    
    def test_parameter_count(self):
        """Test parameter counting."""
        # Default rotation gates: ['rx', 'ry', 'rz']
        # Expected: num_layers * len(rotation_gates) * num_qubits
        expected_params = 2 * 3 * 4  # 24 parameters
        self.assertEqual(self.qnn.get_num_parameters(), expected_params)
    
    def test_circuit_depth(self):
        """Test circuit depth calculation."""
        depth = self.qnn.get_circuit_depth()
        self.assertGreater(depth, 0)
        self.assertIsInstance(depth, int)
    
    def test_forward_pass(self):
        """Test forward pass functionality."""
        X = np.random.uniform(0, np.pi, (2, 4))
        params = np.random.uniform(-np.pi, np.pi, self.qnn.get_num_parameters())
        
        # Test forward pass
        results = self.qnn.forward(X, params, shots=100)
        
        self.assertEqual(results.shape[0], 2)  # Batch size
        self.assertGreater(results.shape[1], 0)  # Output dimensions
        
        # Check probabilities sum to 1
        for i in range(results.shape[0]):
            self.assertAlmostEqual(np.sum(results[i]), 1.0, places=2)


class TestNoiseModels(unittest.TestCase):
    """Test noise model functionality."""
    
    def test_noise_model_creation(self):
        """Test noise model creation."""
        noise_model = NoiseModel(
            gate_error_rate=0.001,
            readout_error_rate=0.01
        )
        
        self.assertEqual(noise_model.gate_error_rate, 0.001)
        self.assertEqual(noise_model.readout_error_rate, 0.01)
        self.assertIsNotNone(noise_model.get_qiskit_noise_model())
    
    def test_predefined_backends(self):
        """Test predefined backend noise models."""
        noise_model = NoiseModel.from_backend("ibm_lagos")
        
        self.assertGreater(noise_model.gate_error_rate, 0)
        self.assertGreater(noise_model.readout_error_rate, 0)
        self.assertGreater(noise_model.T1, 0)
        self.assertGreater(noise_model.T2, 0)
    
    def test_ideal_noise_model(self):
        """Test ideal (noiseless) model."""
        ideal_model = NoiseModel.ideal()
        
        self.assertEqual(ideal_model.gate_error_rate, 0.0)
        self.assertEqual(ideal_model.readout_error_rate, 0.0)
        self.assertEqual(ideal_model.T1, float('inf'))
        self.assertEqual(ideal_model.T2, float('inf'))
    
    def test_noise_scaling(self):
        """Test noise model scaling."""
        base_model = NoiseModel(gate_error_rate=0.001)
        scaled_model = base_model.scale_noise(2.0)
        
        self.assertEqual(scaled_model.gate_error_rate, 0.002)
        self.assertEqual(scaled_model.T1, base_model.T1 / 2.0)


class TestErrorCorrection(unittest.TestCase):
    """Test error correction functionality."""
    
    def test_repetition_code(self):
        """Test simple repetition code."""
        rep_code = SimpleRepetitionCode()
        
        self.assertEqual(rep_code.get_physical_qubits(1), 3)
        self.assertEqual(rep_code.get_code_distance(), 3)
        self.assertEqual(rep_code.get_error_threshold(), 0.5)
        
        # Test syndrome decoding
        syndrome = "01"  # Error on qubit 2
        errors = rep_code.decode_syndrome(syndrome)
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0], ("X", 2))
    
    def test_surface_code(self):
        """Test surface code implementation."""
        surface_code = SurfaceCode(distance=3)
        
        self.assertEqual(surface_code.distance, 3)
        self.assertEqual(surface_code.get_code_distance(), 3)
        self.assertAlmostEqual(surface_code.get_error_threshold(), 0.01, places=3)
        
        # Test physical qubit count
        physical_qubits = surface_code.get_physical_qubits(1)
        self.assertGreater(physical_qubits, 9)  # Should need more than 9 physical qubits
    
    def test_color_code(self):
        """Test color code implementation."""
        color_code = ColorCode(distance=3)
        
        self.assertEqual(color_code.distance, 3)
        self.assertEqual(color_code.get_code_distance(), 3)
        self.assertGreater(color_code.get_error_threshold(), 0.005)
        
        # Test face coloring
        face_colors = color_code.get_face_colors()
        self.assertGreater(len(face_colors), 0)
        
        # Check all three colors are used
        used_colors = set(face_colors.values())
        self.assertTrue(any(color in used_colors for color in ['R', 'G', 'B']))


class TestTraining(unittest.TestCase):
    """Test training functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.qnn = QECCAwareQNN(num_qubits=2, num_layers=1)
        self.noise_model = NoiseModel(gate_error_rate=0.001)
        
        # Simple training data
        self.X_train = np.random.uniform(0, np.pi, (10, 2))
        self.y_train = np.random.randint(0, 2, 10)
    
    def test_trainer_initialization(self):
        """Test trainer initialization."""
        trainer = QECCTrainer(
            qnn=self.qnn,
            noise_model=self.noise_model,
            shots=100
        )
        
        self.assertEqual(trainer.shots, 100)
        self.assertIsNotNone(trainer.optimizer)
        self.assertIsNotNone(trainer.loss_function)
    
    def test_noise_aware_optimizer(self):
        """Test noise-aware Adam optimizer."""
        optimizer = NoiseAwareAdam(
            learning_rate=0.01,
            noise_model=self.noise_model
        )
        
        # Test parameter update
        params = np.random.uniform(-1, 1, 10)
        gradients = np.random.uniform(-0.1, 0.1, 10)
        
        new_params = optimizer.step(params, gradients)
        
        self.assertEqual(len(new_params), len(params))
        self.assertFalse(np.array_equal(params, new_params))
    
    def test_quantum_cross_entropy(self):
        """Test quantum cross-entropy loss."""
        loss_fn = QuantumCrossEntropy()
        
        # Mock predictions and targets
        predictions = np.array([[0.7, 0.3], [0.4, 0.6]])
        targets = np.array([0, 1])
        
        loss = loss_fn(predictions, targets)
        
        self.assertGreater(loss, 0)
        self.assertIsInstance(loss, float)
    
    def test_training_step(self):
        """Test single training step."""
        trainer = QECCTrainer(
            qnn=self.qnn,
            noise_model=self.noise_model,
            shots=64  # Small number for fast testing
        )
        
        # Get initial parameters
        initial_params = trainer.get_parameters()
        
        # Run one training epoch
        history = trainer.fit(
            self.X_train, self.y_train,
            epochs=1,
            batch_size=5,
            verbose=False
        )
        
        # Check that parameters changed
        final_params = trainer.get_parameters()
        self.assertFalse(np.allclose(initial_params, final_params))
        
        # Check history structure
        self.assertIn('loss', history)
        self.assertIn('accuracy', history)
        self.assertEqual(len(history['loss']), 1)


class TestEvaluation(unittest.TestCase):
    """Test evaluation and benchmarking."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.qnn = QECCAwareQNN(num_qubits=2, num_layers=1)
        self.X_test = np.random.uniform(0, np.pi, (5, 2))
        self.y_test = np.random.randint(0, 2, 5)
    
    def test_fidelity_tracker(self):
        """Test fidelity tracking."""
        tracker = FidelityTracker(window_size=10)
        
        # Add some fidelity measurements
        for i in range(5):
            fidelity = 0.9 - i * 0.05
            error_rate = 0.001 + i * 0.001
            tracker.update(fidelity, error_rate)
        
        stats = tracker.get_current_stats()
        
        self.assertIn('current_fidelity', stats)
        self.assertIn('mean_fidelity', stats)
        self.assertEqual(stats['current_fidelity'], 0.7)
    
    def test_noise_benchmark(self):
        """Test noise benchmarking."""
        benchmark = NoiseBenchmark(
            model=self.qnn,
            noise_levels=np.array([0.001, 0.01]),
            shots=64  # Small for fast testing
        )
        
        # Mock trained parameters
        params = np.random.uniform(-np.pi, np.pi, self.qnn.get_num_parameters())
        
        results = benchmark.run(
            self.X_test, self.y_test,
            model_parameters=params,
            verbose=False
        )
        
        # Check results structure
        self.assertIn('accuracy', results)
        self.assertIn('fidelity', results)
        self.assertEqual(len(results['accuracy']), 2)  # Two noise levels
        
        # Check threshold analysis
        analysis = benchmark.get_threshold_analysis()
        self.assertIsInstance(analysis, dict)


class TestIntegration(unittest.TestCase):
    """Integration tests."""
    
    def test_qecc_integration(self):
        """Test full QECC integration."""
        # Create QNN with error correction
        qnn = QECCAwareQNN(num_qubits=2, num_layers=1)
        surface_code = SurfaceCode(distance=3, logical_qubits=2)
        qnn.add_error_correction(surface_code)
        
        # Verify error correction is added
        self.assertIsNotNone(qnn.error_correction)
        self.assertGreater(qnn.num_physical_qubits, qnn.num_qubits)
        
        # Test circuit construction
        circuit = qnn.get_circuit()
        self.assertIsNotNone(circuit)
        self.assertGreater(circuit.num_qubits, qnn.num_qubits)
    
    def test_end_to_end_training(self):
        """Test end-to-end training with error correction."""
        # Small-scale test
        qnn = QECCAwareQNN(num_qubits=2, num_layers=1)
        rep_code = SimpleRepetitionCode()
        qnn.add_error_correction(rep_code)
        
        noise_model = NoiseModel(gate_error_rate=0.01)
        trainer = QECCTrainer(
            qnn=qnn,
            noise_model=noise_model,
            shots=64,
            learning_rate=0.1
        )
        
        # Generate simple data
        X = np.random.uniform(0, np.pi, (6, 2))
        y = np.random.randint(0, 2, 6)
        
        # Train for a few epochs
        history = trainer.fit(X, y, epochs=2, verbose=False)
        
        # Check training completed
        self.assertEqual(len(history['loss']), 2)
        self.assertGreater(len(history['accuracy']), 0)
        
        # Test evaluation
        results = trainer.evaluate(X, y)
        self.assertIn('accuracy', results)
        self.assertIn('logical_error_rate', results)


if __name__ == '__main__':
    # Configure test runner
    unittest_loader = unittest.TestLoader()
    test_suite = unittest_loader.discover('.', pattern='test_*.py')
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print(f"\nFailures:")
        for test, traceback in result.failures:
            print(f"  - {test}")
    
    if result.errors:
        print(f"\nErrors:")
        for test, traceback in result.errors:
            print(f"  - {test}")
    
    if result.wasSuccessful():
        print(f"\n✅ All tests passed!")
    else:
        print(f"\n❌ Some tests failed!")
    
    print(f"{'='*50}")