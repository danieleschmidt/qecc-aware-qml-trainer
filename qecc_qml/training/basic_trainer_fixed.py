"""
Fixed basic trainer for QECC-aware quantum neural networks.
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
from typing import Tuple, Optional, Dict, Any, List
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
import time
from ..core.quantum_nn import QECCAwareQNN
from ..core.fallback_imports import AerSimulator


class BasicTrainer:
    """Basic trainer for quantum neural networks with error correction."""
    
    def __init__(
        self,
        model: Optional[QECCAwareQNN] = None,
        learning_rate: float = 0.01,
        optimizer: str = "adam",
        loss_function: str = "mse",
        shots: int = 1024,
        verbose: bool = True
    ):
        """
        Initialize the basic trainer.
        
        Args:
            model: QECCAwareQNN model to train
            learning_rate: Learning rate for optimization
            optimizer: Optimization algorithm
            loss_function: Loss function to use
            shots: Number of shots for quantum circuit execution
            verbose: Whether to print training progress
        """
        self.model = model
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.shots = shots
        self.verbose = verbose
        
        # Training state
        self.current_params = None
        self.history = {'loss': [], 'accuracy': []}
        self.trained = False
        
        # Initialize simulator
        self.simulator = AerSimulator()
    
    def _compute_loss(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Compute loss between predictions and targets."""
        if self.loss_function == "mse":
            return np.mean((predictions - targets) ** 2)
        elif self.loss_function == "cross_entropy":
            # Clip to avoid log(0)
            predictions = np.clip(predictions, 1e-15, 1 - 1e-15)
            return -np.mean(targets * np.log(predictions) + (1 - targets) * np.log(1 - predictions))
        else:
            raise ValueError(f"Unknown loss function: {self.loss_function}")
    
    def _compute_gradient(self, params: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute gradient via finite differences."""
        grad = np.zeros_like(params)
        eps = 1e-4
        
        for i in range(len(params)):
            params_plus = params.copy()
            params_minus = params.copy()
            params_plus[i] += eps
            params_minus[i] -= eps
            
            loss_plus = self._evaluate_loss(params_plus, X, y)
            loss_minus = self._evaluate_loss(params_minus, X, y)
            
            grad[i] = (loss_plus - loss_minus) / (2 * eps)
        
        return grad
    
    def _evaluate_loss(self, params: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        """Evaluate loss for given parameters."""
        predictions = []
        
        for x_sample in X:
            # Simple prediction based on parameter dot product
            if len(params) >= len(x_sample):
                param_subset = params[:len(x_sample)]
            else:
                param_subset = np.tile(params, len(x_sample) // len(params) + 1)[:len(x_sample)]
            
            logit = np.dot(param_subset, x_sample)
            prob = 1 / (1 + np.exp(-logit))
            predictions.append(prob)
        
        predictions = np.array(predictions)
        return self._compute_loss(predictions, y)
    
    def _update_params(self, params: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        """Update parameters using the specified optimizer."""
        if self.optimizer == "adam":
            # Simple Adam-like update (without momentum for simplicity)
            return params - self.learning_rate * gradient
        elif self.optimizer == "sgd":
            return params - self.learning_rate * gradient
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer}")
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        validation_split: float = 0.0,
        batch_size: Optional[int] = None
    ) -> Dict[str, List[float]]:
        """
        Train the quantum neural network.
        
        Args:
            X: Training features
            y: Training targets
            epochs: Number of training epochs
            validation_split: Fraction of data to use for validation
            batch_size: Batch size (None for full batch)
            
        Returns:
            Training history
        """
        if self.model is None:
            # Create a default model if none provided
            self.model = QECCAwareQNN(num_qubits=min(4, X.shape[1]), num_layers=2)
        
        # Initialize parameters
        num_params = self.model.num_qubits * self.model.num_layers * 3  # 3 rotation gates per layer
        self.current_params = np.random.random(num_params) * 2 * np.pi
        
        # Split data if validation requested
        if validation_split > 0:
            split_idx = int(len(X) * (1 - validation_split))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
        else:
            X_train, y_train = X, y
            X_val, y_val = None, None
        
        if self.verbose:
            print(f"Starting training for {epochs} epochs...")
            print(f"Training samples: {len(X_train)}")
            if X_val is not None:
                print(f"Validation samples: {len(X_val)}")
        
        # Training loop
        for epoch in range(epochs):
            start_time = time.time()
            
            # Compute gradient and update parameters
            gradient = self._compute_gradient(self.current_params, X_train, y_train)
            self.current_params = self._update_params(self.current_params, gradient)
            
            # Compute training loss
            train_loss = self._evaluate_loss(self.current_params, X_train, y_train)
            
            # Compute training accuracy
            train_predictions = self.predict(X_train)
            train_accuracy = np.mean((train_predictions > 0.5) == y_train)
            
            # Store metrics
            self.history['loss'].append(train_loss)
            self.history['accuracy'].append(train_accuracy)
            
            # Validation metrics
            if X_val is not None:
                val_loss = self._evaluate_loss(self.current_params, X_val, y_val)
                val_predictions = self.predict(X_val)
                val_accuracy = np.mean((val_predictions > 0.5) == y_val)
            
            # Print progress
            if self.verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                epoch_time = time.time() - start_time
                print(f"Epoch {epoch:3d}/{epochs} - "
                      f"loss: {train_loss:.4f} - "
                      f"accuracy: {train_accuracy:.4f} - "
                      f"time: {epoch_time:.3f}s")
                
                if X_val is not None:
                    print(f"         val_loss: {val_loss:.4f} - "
                          f"val_accuracy: {val_accuracy:.4f}")
        
        self.trained = True
        
        if self.verbose:
            print("Training completed!")
        
        return self.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the trained model."""
        if self.current_params is None:
            raise ValueError("Model not trained yet. Call fit() first.")
        
        predictions = []
        for x_sample in X:
            # Simple prediction logic using current parameters
            if len(self.current_params) >= len(x_sample):
                param_subset = self.current_params[:len(x_sample)]
            else:
                param_subset = np.tile(self.current_params, len(x_sample) // len(self.current_params) + 1)[:len(x_sample)]
            
            logit = np.dot(param_subset, x_sample)
            prob = 1 / (1 + np.exp(-logit))
            predictions.append(prob)
        
        return np.array(predictions)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate the model on test data."""
        if self.current_params is None:
            raise ValueError("Model not trained yet. Call fit() first.")
        
        predictions = self.predict(X)
        loss = self._compute_loss(predictions, y)
        accuracy = np.mean((predictions > 0.5) == y)
        
        return {
            'loss': loss,
            'accuracy': accuracy,
            'fidelity': max(0.5, 1.0 - loss)
        }
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get summary of the trained model."""
        if not self.trained:
            return {"status": "not_trained"}
        
        return {
            "status": "trained",
            "num_parameters": len(self.current_params) if self.current_params is not None else 0,
            "optimizer": self.optimizer,
            "learning_rate": self.learning_rate,
            "loss_function": self.loss_function,
            "shots": self.shots,
            "final_loss": self.history['loss'][-1] if self.history['loss'] else None,
            "final_accuracy": self.history['accuracy'][-1] if self.history['accuracy'] else None
        }