"""
QECC-aware training for quantum neural networks.
"""

# Import with fallback support
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from qecc_qml.core.fallback_imports import create_fallback_implementations
    create_fallback_implementations()
except ImportError:
    pass
from typing import Tuple, Optional, Dict, Any, Callable, List
try:
    import numpy as np
except ImportError:
    import sys
    if 'numpy' in sys.modules:
        np = sys.modules['numpy']
    else:
        class MockNumPy:
            @staticmethod
            def array(x): return list(x) if isinstance(x, (list, tuple)) else x
            @staticmethod
            def zeros(shape): return [0] * (shape if isinstance(shape, int) else shape[0])
            @staticmethod  
            def ones(shape): return [1] * (shape if isinstance(shape, int) else shape[0])
            ndarray = list
        np = MockNumPy()
try:
    from tqdm import tqdm
except ImportError:
    # Fallback for environments without tqdm
    class tqdm:
        def __init__(self, iterable=None, **kwargs):
            self.iterable = iterable or []
            self.kwargs = kwargs
        def __iter__(self):
            return iter(self.iterable)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def set_description(self, desc):
            print(f"Progress: {desc}")
        def update(self, n=1):
            pass
        def close(self):
            pass
import time
try:
    from qiskit_aer import AerSimulator
except ImportError:
    from qecc_qml.core.fallback_imports import AerSimulator

from ..core.quantum_nn import QECCAwareQNN
from ..core.noise_models import NoiseModel
from .optimizers import NoiseAwareAdam
from .loss_functions import QuantumCrossEntropy


class QECCTrainer:
    """
    Trainer for QECC-aware quantum neural networks.
    
    Handles noise-aware training with error correction, fidelity tracking,
    and adaptive optimization strategies.
    """
    
    def __init__(
        self,
        qnn: QECCAwareQNN,
        noise_model: Optional[NoiseModel] = None,
        optimizer: str = "noise_aware_adam",
        loss: str = "cross_entropy",
        shots: int = 1024,
        backend=None,
        learning_rate: float = 0.01,
        track_fidelity: bool = True,
        use_error_mitigation: bool = True,
    ):
        """
        Initialize QECC trainer.
        
        Args:
            qnn: The quantum neural network to train
            noise_model: Noise model for simulation
            optimizer: Optimizer type ('noise_aware_adam', 'adam', 'sgd')
            loss: Loss function ('cross_entropy', 'mse')
            shots: Number of measurement shots
            backend: Quantum backend (defaults to AerSimulator)
            learning_rate: Learning rate for optimization
            track_fidelity: Whether to track fidelity during training
            use_error_mitigation: Whether to use error mitigation techniques
        """
        self.qnn = qnn
        self.noise_model = noise_model
        self.shots = shots
        self.learning_rate = learning_rate
        self.track_fidelity = track_fidelity
        self.use_error_mitigation = use_error_mitigation
        
        # Set up backend
        if backend is None:
            self.backend = AerSimulator()
            if noise_model:
                self.backend.set_options(noise_model=noise_model.get_qiskit_noise_model())
        else:
            self.backend = backend
        
        # Set up optimizer
        if optimizer == "noise_aware_adam":
            self.optimizer = NoiseAwareAdam(
                learning_rate=learning_rate,
                noise_model=noise_model
            )
        else:
            # Fallback to simple gradient descent
            self.optimizer = self._simple_sgd
        
        # Set up loss function
        if loss == "cross_entropy":
            self.loss_function = QuantumCrossEntropy()
        else:
            self.loss_function = self._mse_loss
        
        # Training history
        self.history = {
            'loss': [],
            'accuracy': [],
            'fidelity': [],
            'logical_error_rate': [],
            'training_time': [],
        }
        
        # Initialize parameters
        self.parameters = np.random.uniform(
            -np.pi, np.pi, self.qnn.get_num_parameters()
        )
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 50,
        batch_size: int = 32,
        validation_split: float = 0.0,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Train the QECC-aware QNN.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Fraction of training data for validation
            verbose: Whether to print training progress
            
        Returns:
            Training history dictionary
        """
        # Handle validation split
        if validation_split > 0 and X_val is None:
            split_idx = int(len(X_train) * (1 - validation_split))
            X_val = X_train[split_idx:]
            y_val = y_train[split_idx:]
            X_train = X_train[:split_idx]
            y_train = y_train[:split_idx]
        
        if verbose:
            print(f"Training QECC-aware QNN for {epochs} epochs")
            print(f"Training samples: {len(X_train)}")
            if X_val is not None:
                print(f"Validation samples: {len(X_val)}")
            print(f"Error correction: {self.qnn.error_correction}")
        
        # Training loop
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            # Training phase
            train_loss, train_acc = self._train_epoch(
                X_train, y_train, batch_size, verbose and epoch % 10 == 0
            )
            
            # Validation phase
            val_loss, val_acc = None, None
            if X_val is not None:
                val_loss, val_acc = self._validate_epoch(X_val, y_val)
            
            # Fidelity tracking
            fidelity = self._estimate_fidelity() if self.track_fidelity else 1.0
            
            # Logical error rate estimation
            logical_error_rate = self._estimate_logical_error_rate()
            
            epoch_time = time.time() - epoch_start_time
            
            # Update history
            self.history['loss'].append(train_loss)
            self.history['accuracy'].append(train_acc)
            self.history['fidelity'].append(fidelity)
            self.history['logical_error_rate'].append(logical_error_rate)
            self.history['training_time'].append(epoch_time)
            
            # Print progress
            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch + 1}/{epochs}")
                print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
                if val_loss is not None:
                    print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                print(f"  Fidelity: {fidelity:.4f}, Logic Err: {logical_error_rate:.2e}")
                print(f"  Time: {epoch_time:.2f}s")
        
        return self.history
    
    def _train_epoch(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray, 
        batch_size: int,
        verbose: bool = False
    ) -> Tuple[float, float]:
        """Train for one epoch."""
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        # Create batches
        num_batches = (len(X_train) + batch_size - 1) // batch_size
        
        if verbose:
            pbar = tqdm(range(num_batches), desc="Training")
        else:
            pbar = range(num_batches)
        
        for batch_idx in pbar:
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(X_train))
            
            X_batch = X_train[start_idx:end_idx]
            y_batch = y_train[start_idx:end_idx]
            
            # Forward pass
            predictions = self.qnn.forward(
                X_batch, self.parameters, shots=self.shots, backend=self.backend
            )
            
            # Compute loss and gradients
            loss = self.loss_function(predictions, y_batch)
            gradients = self._compute_gradients(X_batch, y_batch, predictions)
            
            # Update parameters
            self.parameters = self.optimizer.step(self.parameters, gradients)
            
            # Track metrics
            total_loss += loss * len(X_batch)
            correct_predictions += self._count_correct_predictions(predictions, y_batch)
            total_samples += len(X_batch)
        
        avg_loss = total_loss / total_samples
        accuracy = correct_predictions / total_samples
        
        return avg_loss, accuracy
    
    def _validate_epoch(
        self, 
        X_val: np.ndarray, 
        y_val: np.ndarray
    ) -> Tuple[float, float]:
        """Validate for one epoch."""
        predictions = self.qnn.forward(
            X_val, self.parameters, shots=self.shots, backend=self.backend
        )
        
        loss = self.loss_function(predictions, y_val)
        accuracy = self._count_correct_predictions(predictions, y_val) / len(y_val)
        
        return loss, accuracy
    
    def _compute_gradients(
        self, 
        X_batch: np.ndarray, 
        y_batch: np.ndarray, 
        predictions: np.ndarray,
        epsilon: float = 0.01
    ) -> np.ndarray:
        """
        Compute gradients using parameter shift rule.
        
        For quantum circuits, we use the parameter shift rule:
        ∂⟨ψ(θ)|H|ψ(θ)⟩/∂θ = (⟨ψ(θ+π/2)|H|ψ(θ+π/2)⟩ - ⟨ψ(θ-π/2)|H|ψ(θ-π/2)⟩) / 2
        """
        gradients = np.zeros_like(self.parameters)
        
        for i in range(len(self.parameters)):
            # Forward shift
            params_plus = self.parameters.copy()
            params_plus[i] += np.pi / 2
            predictions_plus = self.qnn.forward(
                X_batch, params_plus, shots=self.shots, backend=self.backend
            )
            loss_plus = self.loss_function(predictions_plus, y_batch)
            
            # Backward shift
            params_minus = self.parameters.copy()
            params_minus[i] -= np.pi / 2
            predictions_minus = self.qnn.forward(
                X_batch, params_minus, shots=self.shots, backend=self.backend
            )
            loss_minus = self.loss_function(predictions_minus, y_batch)
            
            # Parameter shift rule
            gradients[i] = (loss_plus - loss_minus) / 2
        
        return gradients
    
    def _count_correct_predictions(
        self, 
        predictions: np.ndarray, 
        y_true: np.ndarray
    ) -> int:
        """Count correct predictions."""
        # Convert probabilities to class predictions
        if predictions.ndim > 1:
            y_pred = np.argmax(predictions, axis=1)
        else:
            y_pred = (predictions > 0.5).astype(int)
        
        if y_true.ndim > 1:
            y_true = np.argmax(y_true, axis=1)
        
        return np.sum(y_pred == y_true)
    
    def _estimate_fidelity(self) -> float:
        """Estimate current circuit fidelity."""
        if self.noise_model is None:
            return 1.0
        
        circuit_depth = self.qnn.get_circuit_depth()
        return self.noise_model.get_fidelity_estimate(circuit_depth)
    
    def _estimate_logical_error_rate(self) -> float:
        """Estimate logical error rate with current error correction."""
        if self.qnn.error_correction is None or self.noise_model is None:
            return 0.0
        
        physical_error_rate = self.noise_model.gate_error_rate
        threshold = self.qnn.error_correction.get_error_threshold()
        
        if physical_error_rate < threshold:
            # Below threshold - exponential suppression
            distance = self.qnn.error_correction.get_code_distance()
            return (physical_error_rate / threshold) ** ((distance + 1) // 2)
        else:
            # Above threshold - no improvement
            return physical_error_rate
    
    def _simple_sgd(self, parameters: np.ndarray, gradients: np.ndarray) -> np.ndarray:
        """Simple SGD optimizer fallback."""
        return parameters - self.learning_rate * gradients
    
    def _mse_loss(self, predictions: np.ndarray, y_true: np.ndarray) -> float:
        """Mean squared error loss."""
        if predictions.ndim > 1:
            predictions = predictions[:, 0]  # Take first output
        if y_true.ndim > 1:
            y_true = y_true[:, 0]
        return np.mean((predictions - y_true) ** 2)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with the trained model."""
        return self.qnn.forward(
            X, self.parameters, shots=self.shots, backend=self.backend
        )
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate the trained model."""
        predictions = self.predict(X_test)
        loss = self.loss_function(predictions, y_test)
        accuracy = self._count_correct_predictions(predictions, y_test) / len(y_test)
        fidelity = self._estimate_fidelity() if self.track_fidelity else 1.0
        logical_error_rate = self._estimate_logical_error_rate()
        
        return {
            'loss': loss,
            'accuracy': accuracy,
            'fidelity': fidelity,
            'logical_error_rate': logical_error_rate,
        }
    
    def get_parameters(self) -> np.ndarray:
        """Get current model parameters."""
        return self.parameters.copy()
    
    def set_parameters(self, parameters: np.ndarray):
        """Set model parameters."""
        if len(parameters) != len(self.parameters):
            raise ValueError(f"Expected {len(self.parameters)} parameters, got {len(parameters)}")
        self.parameters = parameters.copy()
    
    def save_model(self, filepath: str):
        """Save model parameters and configuration."""
        import pickle
        
        model_data = {
            'parameters': self.parameters,
            'qnn_config': {
                'num_qubits': self.qnn.num_qubits,
                'num_layers': self.qnn.num_layers,
                'entanglement': self.qnn.entanglement,
                'feature_map': self.qnn.feature_map,
                'rotation_gates': self.qnn.rotation_gates,
            },
            'error_correction': self.qnn.error_correction,
            'history': self.history,
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def __str__(self) -> str:
        ec_info = f" with {self.qnn.error_correction}" if self.qnn.error_correction else ""
        return f"QECCTrainer({self.qnn.num_qubits} qubits{ec_info})"
    
    def __repr__(self) -> str:
        return self.__str__()