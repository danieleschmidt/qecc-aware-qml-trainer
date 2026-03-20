# qecc-aware-qml-trainer

Quantum Machine Learning trainer with integrated surface code error correction. Features state-vector simulation, parameterized quantum layers with parameter-shift gradients, and 3-qubit majority vote error correction stubs.

## Install

```bash
pip install numpy
```

## Usage

```python
from qecc_qml.circuit import QuantumCircuit
from qecc_qml.layer import QMLLayer
from qecc_qml.trainer import TrainingLoop
```

## Modules

- **`qecc_qml.circuit`** — `QuantumCircuit`: state-vector simulator with Rx, Ry, Rz, H, CNOT gates and Z-expectation measurement
- **`qecc_qml.surface_code`** — `SurfaceCodeStub`: 3-qubit repetition code with majority vote error correction
- **`qecc_qml.layer`** — `QMLLayer`: parameterized RY-RZ-RY ansatz with parameter-shift gradient computation
- **`qecc_qml.trainer`** — `TrainingLoop`: MSE loss optimization via parameter-shift gradient descent
- **`qecc_qml.fidelity`** — `FidelityTracker`: track and visualize quantum state fidelity over time

## Example

```python
import numpy as np
from qecc_qml.circuit import QuantumCircuit
from qecc_qml.layer import QMLLayer
from qecc_qml.trainer import TrainingLoop
from qecc_qml.fidelity import FidelityTracker

# Setup
n_qubits = 2
circuit = QuantumCircuit(n_qubits)
layer = QMLLayer(n_qubits)
trainer = TrainingLoop(layer, circuit, lr=0.1)

# Train to match target expectations
target = [0.8, -0.8]
losses = trainer.train(target, n_steps=50)
print(f"Initial loss: {losses[0]:.4f}, Final loss: {losses[-1]:.4f}")

# Track fidelity
ft = FidelityTracker()
s1 = np.array([1, 0], dtype=complex)
s2 = np.array([1, 1], dtype=complex) / np.sqrt(2)
ft.record(s1, s2)
ft.plot_ascii()
print(f"Best fidelity: {ft.best():.4f}")
```

## Running Tests

```bash
python -m pytest tests/ -v
```
