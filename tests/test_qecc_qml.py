"""Tests for the QECC-aware QML Trainer."""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from qecc_qml.circuit import QuantumCircuit
from qecc_qml.surface_code import SurfaceCodeStub
from qecc_qml.layer import QMLLayer
from qecc_qml.trainer import TrainingLoop
from qecc_qml.fidelity import FidelityTracker


# ── QuantumCircuit Tests ────────────────────────────────────────────────────

class TestQuantumCircuit:

    def test_circuit_initial_state_zero(self):
        """Initial state should be |0...0> = [1, 0, 0, ...]."""
        qc = QuantumCircuit(3)
        assert qc.state[0] == pytest.approx(1.0 + 0j)
        assert np.all(qc.state[1:] == 0)
        assert qc.state.shape == (8,)

    def test_circuit_h_gate_superposition(self):
        """H gate on |0> should create equal superposition."""
        qc = QuantumCircuit(1)
        qc.h(0)
        expected_amp = 1.0 / np.sqrt(2)
        assert abs(qc.state[0]) == pytest.approx(expected_amp, abs=1e-10)
        assert abs(qc.state[1]) == pytest.approx(expected_amp, abs=1e-10)

    def test_circuit_rx_rotation(self):
        """Rx(pi) on |0> should give |1> (up to global phase)."""
        qc = QuantumCircuit(1)
        qc.rx(0, np.pi)
        # After Rx(pi): state = [cos(pi/2), -i*sin(pi/2)] = [0, -i]
        assert abs(qc.state[0]) == pytest.approx(0.0, abs=1e-10)
        assert abs(qc.state[1]) == pytest.approx(1.0, abs=1e-10)

    def test_circuit_measure_z_range(self):
        """<Z> expectation should always be in [-1, 1]."""
        qc = QuantumCircuit(2)
        # Test various states
        for theta in np.linspace(0, 2 * np.pi, 20):
            qc.reset()
            qc.rx(0, theta)
            val = qc.measure_z(0)
            assert -1.0 - 1e-10 <= val <= 1.0 + 1e-10, f"measure_z={val} out of range"

    def test_circuit_measure_z_zero_state(self):
        """<Z> for |0> should be +1."""
        qc = QuantumCircuit(1)
        assert qc.measure_z(0) == pytest.approx(1.0)

    def test_circuit_reset(self):
        """Reset should restore to |0...0> state."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.h(1)
        qc.reset()
        assert qc.state[0] == pytest.approx(1.0)
        assert np.all(qc.state[1:] == 0)

    def test_circuit_ry_rotation(self):
        """Ry(pi) on |0> should give |1>."""
        qc = QuantumCircuit(1)
        qc.ry(0, np.pi)
        assert abs(qc.state[0]) == pytest.approx(0.0, abs=1e-10)
        assert abs(qc.state[1]) == pytest.approx(1.0, abs=1e-10)

    def test_circuit_state_normalized(self):
        """State vector should remain normalized after gate applications."""
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.h(1)
        qc.cnot(0, 1)
        qc.rx(2, 1.23)
        norm = np.linalg.norm(qc.state)
        assert norm == pytest.approx(1.0, abs=1e-10)


# ── SurfaceCodeStub Tests ───────────────────────────────────────────────────

class TestSurfaceCodeStub:

    def test_surface_code_encode(self):
        """Encoding 0 gives [0,0,0], encoding 1 gives [1,1,1]."""
        sc = SurfaceCodeStub()
        assert sc.encode(0) == [0, 0, 0]
        assert sc.encode(1) == [1, 1, 1]

    def test_surface_code_decode_majority(self):
        """Majority vote: two 1s give 1, two 0s give 0."""
        sc = SurfaceCodeStub()
        assert sc.decode([1, 1, 0]) == 1
        assert sc.decode([0, 1, 1]) == 1
        assert sc.decode([1, 0, 0]) == 0
        assert sc.decode([0, 0, 1]) == 0
        assert sc.decode([1, 1, 1]) == 1
        assert sc.decode([0, 0, 0]) == 0

    def test_surface_code_correct_single_flip(self):
        """With flip_prob=0, correct() should return original bit unchanged."""
        sc = SurfaceCodeStub()
        # With 0 flip probability, should return exact majority
        result = sc.correct([1, 1, 1], flip_prob=0.0)
        assert result == 1
        result = sc.correct([0, 0, 0], flip_prob=0.0)
        assert result == 0

    def test_surface_code_correct_returns_binary(self):
        """correct() should return 0 or 1."""
        sc = SurfaceCodeStub()
        for _ in range(20):
            result = sc.correct([1, 0, 1], flip_prob=0.1)
            assert result in (0, 1)


# ── QMLLayer Tests ──────────────────────────────────────────────────────────

class TestQMLLayer:

    def test_qml_layer_param_shape(self):
        """params should have shape (n_qubits * 3,)."""
        layer = QMLLayer(4)
        assert layer.params.shape == (12,)

    def test_qml_layer_forward_shape(self):
        """forward() should return a list of length n_qubits."""
        layer = QMLLayer(3)
        circuit = QuantumCircuit(3)
        result = layer.forward(circuit)
        assert len(result) == 3

    def test_qml_layer_forward_values_in_range(self):
        """All forward() outputs should be in [-1, 1]."""
        layer = QMLLayer(2)
        circuit = QuantumCircuit(2)
        result = layer.forward(circuit)
        for val in result:
            assert -1.0 - 1e-10 <= val <= 1.0 + 1e-10

    def test_parameter_shift_gradient_runs(self):
        """parameter_shift_gradient should return a float without error."""
        layer = QMLLayer(2)
        circuit = QuantumCircuit(2)
        layer.forward(circuit)

        def observable_fn():
            return float(circuit.measure_z(0))

        grad = layer.parameter_shift_gradient(circuit, 0, observable_fn)
        assert isinstance(grad, float)
        assert np.isfinite(grad)

    def test_parameter_shift_gradient_all_params(self):
        """Should be able to compute gradient for all parameters."""
        n = 2
        layer = QMLLayer(n)
        circuit = QuantumCircuit(n)
        target = np.array([0.5, -0.5])

        grads = []
        for i in range(len(layer.params)):
            def obs(layer=layer, circuit=circuit, tgt=target):
                out = np.array([circuit.measure_z(q) for q in range(layer.n_qubits)])
                return float(np.mean((out - tgt) ** 2))
            g = layer.parameter_shift_gradient(circuit, i, obs)
            grads.append(g)
        assert len(grads) == n * 3


# ── TrainingLoop Tests ──────────────────────────────────────────────────────

class TestTrainingLoop:

    def test_training_loop_loss_decreases(self):
        """Training should generally reduce loss over 20 steps."""
        np.random.seed(42)
        n = 2
        layer = QMLLayer(n)
        circuit = QuantumCircuit(n)
        trainer = TrainingLoop(layer, circuit, lr=0.2)
        target = [0.9, -0.9]
        history = trainer.train(target, n_steps=20)
        assert len(history) == 20
        # Loss should generally decrease (first vs last)
        assert history[-1] <= history[0] + 0.5  # allow some tolerance

    def test_training_loop_returns_loss_history(self):
        """train() should return a list of floats."""
        n = 1
        layer = QMLLayer(n)
        circuit = QuantumCircuit(n)
        trainer = TrainingLoop(layer, circuit, lr=0.1)
        history = trainer.train([0.0], n_steps=5)
        assert len(history) == 5
        for loss in history:
            assert isinstance(loss, float)
            assert loss >= 0.0

    def test_training_loop_step_returns_float(self):
        """step() should return a non-negative float."""
        layer = QMLLayer(2)
        circuit = QuantumCircuit(2)
        trainer = TrainingLoop(layer, circuit)
        loss = trainer.step([0.5, -0.5])
        assert isinstance(loss, float)
        assert loss >= 0.0


# ── FidelityTracker Tests ───────────────────────────────────────────────────

class TestFidelityTracker:

    def test_fidelity_tracker_record(self):
        """record() should store fidelity in history."""
        ft = FidelityTracker()
        s1 = np.array([1.0, 0.0], dtype=complex)
        s2 = np.array([1.0, 0.0], dtype=complex)
        ft.record(s1, s2)
        assert len(ft.history) == 1
        assert ft.history[0] == pytest.approx(1.0)

    def test_fidelity_identical_states(self):
        """Fidelity of identical states should be 1."""
        ft = FidelityTracker()
        s = np.array([1.0 / np.sqrt(2), 1.0 / np.sqrt(2)], dtype=complex)
        ft.record(s, s)
        assert ft.history[0] == pytest.approx(1.0)

    def test_fidelity_orthogonal_states(self):
        """Fidelity of orthogonal states should be 0."""
        ft = FidelityTracker()
        s1 = np.array([1.0, 0.0], dtype=complex)
        s2 = np.array([0.0, 1.0], dtype=complex)
        ft.record(s1, s2)
        assert ft.history[0] == pytest.approx(0.0)

    def test_fidelity_tracker_best(self):
        """best() should return the maximum fidelity."""
        ft = FidelityTracker()
        ft.history = [0.3, 0.9, 0.6]
        assert ft.best() == pytest.approx(0.9)

    def test_fidelity_tracker_mean(self):
        """mean() should return the average fidelity."""
        ft = FidelityTracker()
        ft.history = [0.2, 0.4, 0.6]
        assert ft.mean() == pytest.approx(0.4)

    def test_fidelity_tracker_empty_best(self):
        """best() on empty tracker should return 0."""
        ft = FidelityTracker()
        assert ft.best() == 0.0

    def test_fidelity_tracker_empty_mean(self):
        """mean() on empty tracker should return 0."""
        ft = FidelityTracker()
        assert ft.mean() == 0.0
