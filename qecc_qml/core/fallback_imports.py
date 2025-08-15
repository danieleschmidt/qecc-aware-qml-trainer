"""
Fallback imports for optional dependencies to ensure core functionality works.
"""

# Fallback for torch
try:
    import torch
except ImportError:
    class FallbackTorch:
        @staticmethod
        def tensor(data):
            import numpy as np
            return np.array(data)
        
        @staticmethod
        def from_numpy(data):
            return data
            
        class optim:
            class Adam:
                def __init__(self, params, lr=0.01):
                    self.params = list(params)
                    self.lr = lr
                    
                def zero_grad(self):
                    pass
                    
                def step(self):
                    pass
    
    torch = FallbackTorch()

# Fallback for sklearn
try:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, mean_squared_error
except ImportError:
    def train_test_split(X, y, test_size=0.2, random_state=None):
        import numpy as np
        if random_state:
            np.random.seed(random_state)
        
        n_samples = len(X)
        n_test = int(n_samples * test_size)
        indices = np.random.permutation(n_samples)
        
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]
        
        return X[train_indices], X[test_indices], y[train_indices], y[test_indices]
    
    def accuracy_score(y_true, y_pred):
        import numpy as np
        return np.mean(y_true == y_pred)
    
    def mean_squared_error(y_true, y_pred):
        import numpy as np
        return np.mean((y_true - y_pred) ** 2)

# Fallback for pandas
try:
    import pandas as pd
except ImportError:
    class FallbackPandas:
        @staticmethod
        def DataFrame(data=None, columns=None):
            import numpy as np
            if data is None:
                return {}
            if isinstance(data, dict):
                return data
            return {"data": np.array(data)}
    
    pd = FallbackPandas()

# Fallback for qiskit
try:
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
    from qiskit.circuit import Parameter, ParameterVector
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_aer import AerSimulator
except ImportError:
    import warnings
    import numpy as np
    
    warnings.warn(
        "Qiskit not found. Using fallback implementations. "
        "Install qiskit for full quantum functionality: pip install qiskit qiskit-aer",
        UserWarning
    )
    
    class QuantumCircuit:
        """Mock quantum circuit implementation."""
        
        def __init__(self, num_qubits=1, num_clbits=None, name="circuit"):
            self.num_qubits = num_qubits
            self.num_clbits = num_clbits or num_qubits
            self.name = name
            self.gates = []
            self.parameters = []
            self.data = []
            self.qregs = []
            self.cregs = []
        
        def add_register(self, register):
            if hasattr(register, 'size') and hasattr(register, 'name'):
                if 'q' in register.name.lower():
                    self.qregs.append(register)
                else:
                    self.cregs.append(register)
        
        def measure_all(self):
            self.gates.append(('measure_all',))
        
        def measure(self, qubit, clbit):
            self.gates.append(('measure', qubit, clbit))
        
        def rx(self, theta, qubit):
            self.gates.append(('rx', theta, qubit))
        
        def ry(self, theta, qubit):
            self.gates.append(('ry', theta, qubit))
        
        def rz(self, theta, qubit):
            self.gates.append(('rz', theta, qubit))
        
        def cx(self, control, target):
            self.gates.append(('cx', control, target))
        
        def h(self, qubit):
            self.gates.append(('h', qubit))
        
        def x(self, qubit):
            self.gates.append(('x', qubit))
        
        def y(self, qubit):
            self.gates.append(('y', qubit))
        
        def z(self, qubit):
            self.gates.append(('z', qubit))
        
        def barrier(self, qubits=None):
            self.gates.append(('barrier', qubits))
        
        def append(self, gate, qargs, cargs=None):
            self.gates.append(('append', gate, qargs, cargs))
        
        def bind_parameters(self, values):
            return self
        
        def decompose(self):
            return self
        
        def depth(self):
            return len(self.gates)
        
        def copy(self):
            new_circuit = QuantumCircuit(self.num_qubits, self.num_clbits, self.name)
            new_circuit.gates = self.gates.copy()
            new_circuit.parameters = self.parameters.copy()
            return new_circuit
        
        def compose(self, other, inplace=False):
            """Compose this circuit with another circuit."""
            if inplace:
                self.gates.extend(other.gates)
                self.parameters.extend(other.parameters)
                return self
            else:
                new_circuit = self.copy()
                new_circuit.gates.extend(other.gates)
                new_circuit.parameters.extend(other.parameters)
                return new_circuit
    
    class QuantumRegister:
        """Mock quantum register."""
        
        def __init__(self, size, name="q"):
            self.size = size
            self.name = name
            self._bits = list(range(size))
        
        def __getitem__(self, index):
            return self._bits[index]
        
        def __len__(self):
            return self.size
    
    class ClassicalRegister:
        """Mock classical register."""
        
        def __init__(self, size, name="c"):
            self.size = size
            self.name = name
            self._bits = list(range(size))
        
        def __getitem__(self, index):
            return self._bits[index]
        
        def __len__(self):
            return self.size
    
    class Parameter:
        """Mock parameter."""
        
        def __init__(self, name):
            self.name = name
            self._value = 0.0
        
        def __str__(self):
            return self.name
        
        def __repr__(self):
            return f"Parameter('{self.name}')"
    
    class ParameterVector:
        """Mock parameter vector."""
        
        def __init__(self, name, length):
            self.name = name
            self.length = length
            self._parameters = [Parameter(f"{name}[{i}]") for i in range(length)]
        
        def __getitem__(self, index):
            return self._parameters[index]
        
        def __len__(self):
            return self.length
    
    class SparsePauliOp:
        """Mock Sparse Pauli operator."""
        
        def __init__(self, paulis, coeffs=None):
            self.paulis = paulis
            self.coeffs = coeffs or [1.0] * len(paulis)
            self.num_qubits = len(paulis[0]) if paulis else 1
        
        def evolve(self, operator):
            return self
        
        def expectation_value(self, state):
            return np.random.random() - 0.5
    
    class Pauli:
        """Mock Pauli operator."""
        
        def __init__(self, label):
            self.label = label
        
        def __str__(self):
            return self.label
        
        def evolve(self, operator):
            return self
        
        def tensor(self, other):
            return Pauli(self.label + other.label)
        
        @staticmethod
        def from_label(label):
            return Pauli(label)
    
    class AerSimulator:
        """Mock Aer simulator."""
        
        def __init__(self, method='statevector'):
            self.name = "mock_simulator"
            self.method = method
        
        def run(self, circuits, shots=1024, **kwargs):
            return MockJob(circuits, shots)
    
    class MockJob:
        """Mock quantum job."""
        
        def __init__(self, circuits=None, shots=1024):
            self.circuits = circuits
            self.shots = shots
            self._result = None
        
        def result(self):
            if self._result is None:
                self._result = MockResult(self.circuits, self.shots)
            return self._result
        
        def status(self):
            return 'DONE'
    
    class MockResult:
        """Mock quantum result."""
        
        def __init__(self, circuits=None, shots=1024):
            self.circuits = circuits
            self.shots = shots
        
        def get_counts(self, circuit=None):
            if hasattr(self, 'circuits') and self.circuits:
                num_qubits = getattr(self.circuits[0], 'num_qubits', 2)
            else:
                num_qubits = 2
            
            states = [format(i, f'0{num_qubits}b') for i in range(min(4, 2**num_qubits))]
            counts = {}
            remaining_shots = self.shots
            
            for i, state in enumerate(states[:-1]):
                count = np.random.randint(0, remaining_shots // (len(states) - i))
                counts[state] = count
                remaining_shots -= count
            
            if states:
                counts[states[-1]] = remaining_shots
            
            return counts
        
        def get_statevector(self, circuit=None):
            if hasattr(self, 'circuits') and self.circuits:
                num_qubits = getattr(self.circuits[0], 'num_qubits', 2)
            else:
                num_qubits = 2
            
            size = 2 ** num_qubits
            real_parts = np.random.randn(size)
            imag_parts = np.random.randn(size)
            statevector = real_parts + 1j * imag_parts
            return statevector / np.linalg.norm(statevector)