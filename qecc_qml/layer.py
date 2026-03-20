"""Parameterized quantum machine learning layer."""

import numpy as np
from typing import List, Callable
from .circuit import QuantumCircuit


class QMLLayer:
    """Parameterized quantum layer with RY-RZ-RY ansatz and CNOT entanglement.

    Parameters are stored as a flat array of shape (n_qubits * 3,), where
    each qubit i has params [theta_ry1, theta_rz, theta_ry2].
    """

    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        # Initialize params with small random values
        self.params = np.random.uniform(-np.pi, np.pi, size=(n_qubits * 3,))

    def _apply_layer(self, circuit: QuantumCircuit):
        """Apply the variational ansatz to the circuit (modifies in place)."""
        # Apply RY-RZ-RY rotations to each qubit
        for i in range(self.n_qubits):
            circuit.ry(i, self.params[i * 3])
            circuit.rz(i, self.params[i * 3 + 1])
            circuit.ry(i, self.params[i * 3 + 2])
        # Apply CNOTs between adjacent qubits
        for i in range(self.n_qubits - 1):
            circuit.cnot(i, i + 1)

    def forward(self, circuit: QuantumCircuit) -> List[float]:
        """Apply layer to circuit and return Z-expectation values.

        Resets the circuit, applies the variational ansatz (RY-RZ-RY per qubit
        plus adjacent CNOTs), and measures <Z> for each qubit.

        Args:
            circuit: QuantumCircuit to apply the layer to (will be reset first).

        Returns:
            List of <Z> expectation values, one per qubit.
        """
        circuit.reset()
        self._apply_layer(circuit)
        return [circuit.measure_z(i) for i in range(self.n_qubits)]

    def parameter_shift_gradient(
        self,
        circuit: QuantumCircuit,
        i: int,
        observable_fn: Callable[[], float],
    ) -> float:
        """Compute gradient of observable w.r.t. params[i] via parameter shift rule.

        Gradient = [f(params[i] + pi/2) - f(params[i] - pi/2)] / 2

        Args:
            circuit: QuantumCircuit to use for evaluation.
            i: Index into self.params to differentiate.
            observable_fn: Callable that takes no arguments and returns a scalar.
                           Called after forward() to compute the observable.

        Returns:
            The estimated gradient as a float.
        """
        original = self.params[i]

        # f(params[i] + pi/2)
        self.params[i] = original + np.pi / 2
        self.forward(circuit)
        f_plus = observable_fn()

        # f(params[i] - pi/2)
        self.params[i] = original - np.pi / 2
        self.forward(circuit)
        f_minus = observable_fn()

        # Restore
        self.params[i] = original

        return (f_plus - f_minus) / 2.0
