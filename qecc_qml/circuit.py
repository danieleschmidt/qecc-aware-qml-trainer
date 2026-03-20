"""State-vector quantum circuit simulator."""

import numpy as np


class QuantumCircuit:
    """State-vector simulator for n-qubit quantum circuits.

    Qubit ordering: qubit 0 is the most significant bit (leftmost in tensor product).
    State vector has length 2^n_qubits.
    """

    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.state = np.zeros(2 ** n_qubits, dtype=complex)
        self.state[0] = 1.0  # |0...0>

    def reset(self):
        """Reset circuit to |0...0> state."""
        self.state = np.zeros(2 ** self.n_qubits, dtype=complex)
        self.state[0] = 1.0

    # ── Single-qubit gate helpers ──────────────────────────────────────────

    def _apply_single_qubit_gate(self, qubit: int, gate: np.ndarray):
        """Apply a 2x2 unitary gate to the given qubit index."""
        n = self.n_qubits
        # Reshape state to (2, 2, ..., 2) — one axis per qubit
        state = self.state.reshape([2] * n)
        # Move the target qubit axis to the front
        state = np.moveaxis(state, qubit, 0)
        # Reshape to (2, 2^(n-1)) and apply gate
        shape = state.shape
        state = state.reshape(2, -1)
        state = gate @ state
        state = state.reshape(shape)
        # Move axis back
        state = np.moveaxis(state, 0, qubit)
        self.state = state.reshape(2 ** n)

    # ── Single-qubit gates ─────────────────────────────────────────────────

    def rx(self, qubit: int, theta: float):
        """Rx rotation gate."""
        c = np.cos(theta / 2)
        s = np.sin(theta / 2)
        gate = np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)
        self._apply_single_qubit_gate(qubit, gate)

    def ry(self, qubit: int, theta: float):
        """Ry rotation gate."""
        c = np.cos(theta / 2)
        s = np.sin(theta / 2)
        gate = np.array([[c, -s], [s, c]], dtype=complex)
        self._apply_single_qubit_gate(qubit, gate)

    def rz(self, qubit: int, theta: float):
        """Rz rotation gate."""
        e_neg = np.exp(-1j * theta / 2)
        e_pos = np.exp(1j * theta / 2)
        gate = np.array([[e_neg, 0], [0, e_pos]], dtype=complex)
        self._apply_single_qubit_gate(qubit, gate)

    def h(self, qubit: int):
        """Hadamard gate."""
        gate = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        self._apply_single_qubit_gate(qubit, gate)

    # ── Two-qubit gates ────────────────────────────────────────────────────

    def cnot(self, control: int, target: int):
        """CNOT (controlled-X) gate."""
        n = self.n_qubits
        state = self.state.reshape([2] * n)
        # Iterate over all basis states where control qubit == 1
        # Use index slicing: control=1 → flip target
        new_state = state.copy()
        # Build index tuples for control=1 slice
        # We do this by looping over all 2^(n-2) combinations of the other qubits
        dim = 2 ** n
        for idx in range(dim):
            bits = [(idx >> (n - 1 - q)) & 1 for q in range(n)]
            if bits[control] == 1:
                # Flip target bit
                bits_flipped = bits.copy()
                bits_flipped[target] ^= 1
                idx_flipped = sum(b << (n - 1 - q) for q, b in enumerate(bits_flipped))
                new_state.reshape(dim)[idx_flipped] = state.reshape(dim)[idx]
                new_state.reshape(dim)[idx] = state.reshape(dim)[idx_flipped]
        # The above double-swaps — use a cleaner approach
        self._apply_cnot_clean(control, target)

    def _apply_cnot_clean(self, control: int, target: int):
        """Apply CNOT gate cleanly."""
        n = self.n_qubits
        dim = 2 ** n
        new_state = self.state.copy()
        for idx in range(dim):
            # Check if control qubit is |1>
            control_bit = (idx >> (n - 1 - control)) & 1
            if control_bit == 1:
                # Flip the target bit
                target_mask = 1 << (n - 1 - target)
                flipped_idx = idx ^ target_mask
                new_state[idx] = self.state[flipped_idx]
        self.state = new_state

    # ── Measurement ────────────────────────────────────────────────────────

    def measure_z(self, qubit: int) -> float:
        """Compute <Z> expectation value for the given qubit.

        Returns sum of |amp|^2 for states where qubit=0
                minus sum of |amp|^2 for states where qubit=1.
        Range: [-1, 1]
        """
        n = self.n_qubits
        dim = 2 ** n
        exp_z = 0.0
        for idx in range(dim):
            bit = (idx >> (n - 1 - qubit)) & 1
            sign = 1 if bit == 0 else -1
            exp_z += sign * (abs(self.state[idx]) ** 2)
        return float(exp_z)
