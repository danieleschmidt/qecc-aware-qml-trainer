"""Training loop for quantum machine learning with parameter-shift gradients."""

import numpy as np
from typing import List
from .circuit import QuantumCircuit
from .layer import QMLLayer


class TrainingLoop:
    """Trains a QMLLayer to match target expectation values using parameter-shift gradients.

    Uses mean squared error as the loss and gradient descent for parameter updates.
    """

    def __init__(self, layer: QMLLayer, circuit: QuantumCircuit, lr: float = 0.1):
        self.layer = layer
        self.circuit = circuit
        self.lr = lr

    def step(self, target_expectations: List[float]) -> float:
        """Perform a single gradient descent step.

        Computes MSE loss between forward pass output and targets, then updates
        all parameters via parameter-shift gradients.

        Args:
            target_expectations: List of target <Z> values, one per qubit.

        Returns:
            The MSE loss before the parameter update.
        """
        target = np.array(target_expectations, dtype=float)

        # Current forward pass for loss computation
        current_output = np.array(self.layer.forward(self.circuit), dtype=float)
        loss = float(np.mean((current_output - target) ** 2))

        # Compute gradients for each parameter
        gradients = np.zeros_like(self.layer.params)

        for i in range(len(self.layer.params)):
            def observable_fn(layer=self.layer, circuit=self.circuit, tgt=target):
                output = np.array([circuit.measure_z(q) for q in range(layer.n_qubits)])
                return float(np.mean((output - tgt) ** 2))

            gradients[i] = self.layer.parameter_shift_gradient(
                self.circuit, i, observable_fn
            )

        # Gradient descent update
        self.layer.params -= self.lr * gradients

        return loss

    def train(self, target_expectations: List[float], n_steps: int = 50) -> List[float]:
        """Run n_steps of gradient descent.

        Args:
            target_expectations: List of target <Z> values, one per qubit.
            n_steps: Number of optimization steps.

        Returns:
            List of loss values (one per step).
        """
        loss_history = []
        for _ in range(n_steps):
            loss = self.step(target_expectations)
            loss_history.append(loss)
        return loss_history
