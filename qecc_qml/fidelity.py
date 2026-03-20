"""Fidelity tracker for quantum state comparison."""

import numpy as np
from typing import List


class FidelityTracker:
    """Tracks fidelity between quantum states over time.

    Fidelity between two pure states |ψ⟩ and |φ⟩ is defined as |⟨ψ|φ⟩|².
    """

    def __init__(self):
        self.history: List[float] = []

    def record(self, state1: np.ndarray, state2: np.ndarray):
        """Compute and record the fidelity between two state vectors.

        Fidelity = |<state1|state2>|^2

        Args:
            state1: Complex state vector (will be normalized if not already).
            state2: Complex state vector (will be normalized if not already).
        """
        s1 = np.asarray(state1, dtype=complex)
        s2 = np.asarray(state2, dtype=complex)

        # Normalize
        norm1 = np.linalg.norm(s1)
        norm2 = np.linalg.norm(s2)
        if norm1 > 0:
            s1 = s1 / norm1
        if norm2 > 0:
            s2 = s2 / norm2

        fidelity = float(abs(np.dot(np.conj(s1), s2)) ** 2)
        self.history.append(fidelity)

    def plot_ascii(self):
        """Print a simple ASCII bar chart of fidelity history."""
        if not self.history:
            print("No fidelity data recorded.")
            return

        width = 40
        print(f"Fidelity History ({len(self.history)} samples)")
        print(f"{'0.0':>5} {'':{'<'}{width}} {'1.0'}")
        print(f"{'':>5} {'─' * width}")

        for i, f in enumerate(self.history):
            bar_len = int(round(f * width))
            bar = "█" * bar_len
            print(f"{i:>4}: {bar:<{width}} {f:.4f}")

    def best(self) -> float:
        """Return the maximum recorded fidelity."""
        if not self.history:
            return 0.0
        return float(max(self.history))

    def mean(self) -> float:
        """Return the mean recorded fidelity."""
        if not self.history:
            return 0.0
        return float(np.mean(self.history))
