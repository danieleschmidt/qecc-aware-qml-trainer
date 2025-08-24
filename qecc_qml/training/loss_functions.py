"""
Loss functions for quantum machine learning with error correction.
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
from typing import Optional, Callable
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


class QuantumCrossEntropy:
    """
    Cross-entropy loss function adapted for quantum probability distributions.
    
    Handles measurement outcomes and probability distributions from quantum circuits,
    with optional regularization for QECC overhead.
    """
    
    def __init__(
        self,
        qecc_regularization: float = 0.0,
        smoothing: float = 1e-12,
    ):
        """
        Initialize quantum cross-entropy loss.
        
        Args:
            qecc_regularization: Regularization weight for QECC overhead
            smoothing: Small value to prevent log(0)
        """
        self.qecc_regularization = qecc_regularization
        self.smoothing = smoothing
    
    def __call__(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """
        Compute quantum cross-entropy loss.
        
        Args:
            predictions: Predicted probability distributions from quantum measurements
            targets: Target labels or distributions
            
        Returns:
            Cross-entropy loss value
        """
        # Handle different input formats
        predictions = np.atleast_2d(predictions)
        targets = np.atleast_1d(targets)
        
        # Convert targets to one-hot if necessary
        if targets.ndim == 1 and len(np.unique(targets)) > 1:
            num_classes = predictions.shape[1] if predictions.ndim > 1 else 2
            targets_onehot = np.zeros((len(targets), num_classes))
            targets_onehot[np.arange(len(targets)), targets.astype(int)] = 1
            targets = targets_onehot
        elif targets.ndim == 1:
            targets = targets.reshape(-1, 1)
        
        # Ensure predictions are probabilities
        if predictions.shape[1] > 1:
            # Multi-class: apply softmax if needed
            predictions = self._softmax(predictions)
        else:
            # Binary: apply sigmoid if needed
            predictions = self._sigmoid(predictions)
        
        # Add smoothing to prevent log(0)
        predictions = np.clip(predictions, self.smoothing, 1 - self.smoothing)
        
        # Compute cross-entropy
        if targets.shape[1] == 1:
            # Binary classification
            loss = -(targets * np.log(predictions) + 
                    (1 - targets) * np.log(1 - predictions))
        else:
            # Multi-class classification
            loss = -np.sum(targets * np.log(predictions), axis=1)
        
        mean_loss = np.mean(loss)
        
        # Add QECC regularization if specified
        if self.qecc_regularization > 0:
            # Penalize for using error correction (encourages efficiency)
            regularization = self.qecc_regularization * self._compute_qecc_penalty(predictions)
            mean_loss += regularization
        
        return mean_loss
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Apply softmax function."""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Apply sigmoid function."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def _compute_qecc_penalty(self, predictions: np.ndarray) -> float:
        """
        Compute penalty term for QECC overhead.
        
        This encourages the model to be confident in its predictions
        to justify the overhead of error correction.
        """
        # Measure prediction confidence (entropy)
        entropy = -np.sum(predictions * np.log(predictions + self.smoothing), axis=1)
        max_entropy = np.log(predictions.shape[1])
        
        # Penalty increases with uncertainty
        uncertainty = entropy / max_entropy
        return np.mean(uncertainty)


class QuantumMeanSquaredError:
    """
    Mean squared error loss for quantum regression tasks.
    """
    
    def __init__(self, qecc_regularization: float = 0.0):
        """
        Initialize quantum MSE loss.
        
        Args:
            qecc_regularization: Regularization weight for QECC overhead
        """
        self.qecc_regularization = qecc_regularization
    
    def __call__(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """
        Compute mean squared error.
        
        Args:
            predictions: Predicted values from quantum measurements
            targets: Target values
            
        Returns:
            MSE loss value
        """
        predictions = np.atleast_1d(predictions)
        targets = np.atleast_1d(targets)
        
        # Take first column if multi-dimensional
        if predictions.ndim > 1:
            predictions = predictions[:, 0]
        if targets.ndim > 1:
            targets = targets[:, 0]
        
        # Compute MSE
        mse = np.mean((predictions - targets) ** 2)
        
        # Add QECC regularization
        if self.qecc_regularization > 0:
            variance_penalty = self.qecc_regularization * np.var(predictions)
            mse += variance_penalty
        
        return mse


class FidelityAwareLoss:
    """
    Loss function that incorporates quantum fidelity considerations.
    
    Adjusts the loss based on estimated circuit fidelity to account for
    noise-induced errors in the learning process.
    """
    
    def __init__(
        self,
        base_loss: Callable[[np.ndarray, np.ndarray], float],
        fidelity_weight: float = 0.1,
        min_fidelity: float = 0.1,
    ):
        """
        Initialize fidelity-aware loss.
        
        Args:
            base_loss: Base loss function (e.g., cross-entropy, MSE)
            fidelity_weight: Weight for fidelity penalty term
            min_fidelity: Minimum fidelity threshold
        """
        self.base_loss = base_loss
        self.fidelity_weight = fidelity_weight
        self.min_fidelity = min_fidelity
    
    def __call__(
        self, 
        predictions: np.ndarray, 
        targets: np.ndarray,
        fidelity: Optional[float] = None
    ) -> float:
        """
        Compute fidelity-aware loss.
        
        Args:
            predictions: Predicted values
            targets: Target values
            fidelity: Estimated circuit fidelity (0 to 1)
            
        Returns:
            Adjusted loss value
        """
        base_loss_value = self.base_loss(predictions, targets)
        
        if fidelity is None:
            return base_loss_value
        
        # Fidelity penalty: higher loss for lower fidelity
        fidelity = max(fidelity, self.min_fidelity)
        fidelity_penalty = self.fidelity_weight * (1 - fidelity) ** 2
        
        # Scale base loss by fidelity factor
        fidelity_factor = 1 / fidelity
        adjusted_loss = base_loss_value * fidelity_factor + fidelity_penalty
        
        return adjusted_loss


class ExpectationValueLoss:
    """
    Loss function based on quantum expectation values.
    
    Directly uses expectation values of quantum observables as targets,
    which is natural for many quantum machine learning applications.
    """
    
    def __init__(
        self,
        observable: Optional[np.ndarray] = None,
        loss_type: str = "mse",
    ):
        """
        Initialize expectation value loss.
        
        Args:
            observable: Quantum observable operator (default: Pauli-Z on first qubit)
            loss_type: Type of loss ("mse", "mae", "huber")
        """
        self.observable = observable
        self.loss_type = loss_type
        
        if observable is None:
            # Default to Pauli-Z measurement on first qubit
            self.observable = np.array([[1, 0], [0, -1]])
    
    def __call__(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """
        Compute expectation value loss.
        
        Args:
            predictions: Probability distributions from quantum measurements
            targets: Target expectation values
            
        Returns:
            Loss value
        """
        # Convert probability distributions to expectation values
        if predictions.ndim > 1 and predictions.shape[1] > 1:
            # For multi-outcome measurements, compute expectation value
            expectation_values = self._compute_expectation_values(predictions)
        else:
            # For binary measurements, simple expectation
            predictions = np.atleast_1d(predictions)
            expectation_values = 2 * predictions - 1  # Map [0,1] to [-1,1]
        
        targets = np.atleast_1d(targets)
        
        # Compute loss based on type
        if self.loss_type == "mse":
            loss = np.mean((expectation_values - targets) ** 2)
        elif self.loss_type == "mae":
            loss = np.mean(np.abs(expectation_values - targets))
        elif self.loss_type == "huber":
            delta = 1.0
            residual = np.abs(expectation_values - targets)
            loss = np.where(
                residual <= delta,
                0.5 * residual ** 2,
                delta * residual - 0.5 * delta ** 2
            )
            loss = np.mean(loss)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        return loss
    
    def _compute_expectation_values(self, probabilities: np.ndarray) -> np.ndarray:
        """
        Compute expectation values from measurement probabilities.
        
        Args:
            probabilities: Measurement outcome probabilities
            
        Returns:
            Expectation values for each sample
        """
        # For computational basis measurements of Pauli-Z
        # ⟨Z⟩ = P(0) - P(1) for single qubit
        if probabilities.shape[1] == 2:
            return probabilities[:, 0] - probabilities[:, 1]
        else:
            # For multi-qubit case, more complex calculation needed
            # This is a simplified version
            even_parity = np.sum(probabilities[:, ::2], axis=1)
            odd_parity = np.sum(probabilities[:, 1::2], axis=1)
            return even_parity - odd_parity


class VariationalQuantumLoss:
    """
    Loss function specifically designed for variational quantum algorithms.
    
    Incorporates both the cost function optimization and circuit ansatz efficiency.
    """
    
    def __init__(
        self, 
        cost_hamiltonian: Optional[np.ndarray] = None,
        circuit_depth_penalty: float = 0.01,
        parameter_regularization: float = 0.001,
    ):
        """
        Initialize variational quantum loss.
        
        Args:
            cost_hamiltonian: Hamiltonian defining the cost function
            circuit_depth_penalty: Penalty for circuit depth
            parameter_regularization: L2 regularization for parameters
        """
        self.cost_hamiltonian = cost_hamiltonian
        self.circuit_depth_penalty = circuit_depth_penalty
        self.parameter_regularization = parameter_regularization
    
    def __call__(
        self, 
        predictions: np.ndarray, 
        targets: np.ndarray,
        parameters: Optional[np.ndarray] = None,
        circuit_depth: Optional[int] = None
    ) -> float:
        """
        Compute variational quantum loss.
        
        Args:
            predictions: Quantum state measurements or expectation values
            targets: Target values or ground truth
            parameters: Current circuit parameters
            circuit_depth: Current circuit depth
            
        Returns:
            Total loss value
        """
        # Base loss (expectation value of cost Hamiltonian)
        if self.cost_hamiltonian is not None:
            # Use Hamiltonian expectation as loss
            base_loss = np.mean(predictions)  # Simplified
        else:
            # Fallback to MSE
            base_loss = np.mean((predictions - targets) ** 2)
        
        total_loss = base_loss
        
        # Circuit depth penalty
        if circuit_depth is not None and self.circuit_depth_penalty > 0:
            depth_penalty = self.circuit_depth_penalty * circuit_depth
            total_loss += depth_penalty
        
        # Parameter regularization
        if parameters is not None and self.parameter_regularization > 0:
            l2_penalty = self.parameter_regularization * np.sum(parameters ** 2)
            total_loss += l2_penalty
        
        return total_loss