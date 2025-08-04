"""
Noise-aware optimizers for QECC quantum machine learning.
"""

from typing import Optional, Dict, Any
import numpy as np

from ..core.noise_models import NoiseModel


class NoiseAwareAdam:
    """
    Adam optimizer adapted for noisy quantum circuits.
    
    Incorporates noise model information to adapt learning rates
    and gradient estimates for better convergence in noisy environments.
    """
    
    def __init__(
        self,
        learning_rate: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        noise_model: Optional[NoiseModel] = None,
        noise_adaptation: bool = True,
        gradient_clipping: bool = True,
        clip_value: float = 1.0,
    ):
        """
        Initialize noise-aware Adam optimizer.
        
        Args:
            learning_rate: Base learning rate
            beta1: Exponential decay rate for first moment estimates
            beta2: Exponential decay rate for second moment estimates
            epsilon: Small constant for numerical stability
            noise_model: Noise model for adaptation
            noise_adaptation: Whether to adapt to noise characteristics
            gradient_clipping: Whether to clip gradients
            clip_value: Maximum gradient norm
        """
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.noise_model = noise_model
        self.noise_adaptation = noise_adaptation
        self.gradient_clipping = gradient_clipping
        self.clip_value = clip_value
        
        # Adam state
        self.m = None  # First moment
        self.v = None  # Second moment
        self.t = 0     # Time step
        
        # Noise-aware adaptations
        self.noise_scale = 1.0
        self.effective_lr = learning_rate
        
        if noise_model and noise_adaptation:
            self._compute_noise_adaptations()
    
    def _compute_noise_adaptations(self):
        """Compute noise-aware adaptations to optimizer parameters."""
        if self.noise_model is None:
            return
        
        # Adapt learning rate based on noise level
        gate_error_rate = self.noise_model.gate_error_rate
        
        # Lower learning rate for higher noise to improve stability
        self.noise_scale = 1.0 / (1.0 + 10 * gate_error_rate)
        self.effective_lr = self.learning_rate * self.noise_scale
        
        # Adapt momentum parameters for noisy gradients
        # Higher noise → more smoothing
        noise_factor = min(gate_error_rate * 100, 0.1)
        self.beta1 = min(0.95, self.beta1 + noise_factor)
        self.beta2 = min(0.999, self.beta2 + noise_factor * 0.1)
    
    def step(self, parameters: np.ndarray, gradients: np.ndarray) -> np.ndarray:
        """
        Perform one optimization step.
        
        Args:
            parameters: Current parameter values
            gradients: Computed gradients
            
        Returns:
            Updated parameters
        """
        if self.m is None:
            self.m = np.zeros_like(parameters)
            self.v = np.zeros_like(parameters)
        
        self.t += 1
        
        # Apply gradient clipping if enabled
        if self.gradient_clipping:
            grad_norm = np.linalg.norm(gradients)
            if grad_norm > self.clip_value:
                gradients = gradients * (self.clip_value / grad_norm)
        
        # Apply noise-aware gradient filtering
        if self.noise_model and self.noise_adaptation:
            gradients = self._filter_noisy_gradients(gradients)
        
        # Update biased first moment estimate
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradients
        
        # Update biased second raw moment estimate
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradients ** 2)
        
        # Compute bias-corrected first moment estimate
        m_hat = self.m / (1 - self.beta1 ** self.t)
        
        # Compute bias-corrected second raw moment estimate
        v_hat = self.v / (1 - self.beta2 ** self.t)
        
        # Update parameters
        update = self.effective_lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        new_parameters = parameters - update
        
        return new_parameters
    
    def _filter_noisy_gradients(self, gradients: np.ndarray) -> np.ndarray:
        """
        Apply noise-aware filtering to gradients.
        
        Reduces impact of gradient noise from quantum measurement statistics.
        """
        if self.noise_model is None:
            return gradients
        
        # Estimate gradient noise level
        shot_noise_scale = 1.0 / np.sqrt(self.noise_model.gate_error_rate * 1000 + 1)
        
        # Apply exponential moving average for high-noise regime
        if hasattr(self, '_prev_gradients'):
            alpha = 0.7 if shot_noise_scale < 0.5 else 0.3
            filtered_gradients = alpha * gradients + (1 - alpha) * self._prev_gradients
        else:
            filtered_gradients = gradients
        
        self._prev_gradients = gradients.copy()
        return filtered_gradients
    
    def reset(self):
        """Reset optimizer state."""
        self.m = None
        self.v = None
        self.t = 0
        if hasattr(self, '_prev_gradients'):
            delattr(self, '_prev_gradients')
    
    def get_state(self) -> Dict[str, Any]:
        """Get optimizer state for saving/loading."""
        return {
            'learning_rate': self.learning_rate,
            'effective_lr': self.effective_lr,
            'beta1': self.beta1,
            'beta2': self.beta2,
            'epsilon': self.epsilon,
            'noise_scale': self.noise_scale,
            'm': self.m.copy() if self.m is not None else None,
            'v': self.v.copy() if self.v is not None else None,
            't': self.t,
        }
    
    def set_state(self, state: Dict[str, Any]):
        """Set optimizer state from saved data."""
        self.learning_rate = state['learning_rate']
        self.effective_lr = state['effective_lr']
        self.beta1 = state['beta1']
        self.beta2 = state['beta2']
        self.epsilon = state['epsilon']
        self.noise_scale = state['noise_scale']
        self.m = state['m'].copy() if state['m'] is not None else None
        self.v = state['v'].copy() if state['v'] is not None else None
        self.t = state['t']
    
    def __str__(self) -> str:
        return (f"NoiseAwareAdam(lr={self.effective_lr:.4f}, "
                f"noise_scale={self.noise_scale:.3f})")
    
    def __repr__(self) -> str:
        return self.__str__()


class QuantumNaturalGradient:
    """
    Quantum Natural Gradient optimizer.
    
    Uses the quantum Fisher information metric to perform natural gradient
    descent, which can be more efficient for quantum parameter landscapes.
    """
    
    def __init__(
        self,
        learning_rate: float = 0.1,
        regularization: float = 1e-6,
        noise_model: Optional[NoiseModel] = None,
    ):
        """
        Initialize quantum natural gradient optimizer.
        
        Args:
            learning_rate: Learning rate
            regularization: Regularization parameter for Fisher information matrix
            noise_model: Noise model for adaptation
        """
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.noise_model = noise_model
        
        # Cache for Fisher information matrix
        self._cached_fisher = None
        self._cache_params = None
    
    def step(
        self, 
        parameters: np.ndarray, 
        gradients: np.ndarray,
        fisher_info_matrix: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Perform quantum natural gradient step.
        
        Args:
            parameters: Current parameters
            gradients: Computed gradients
            fisher_info_matrix: Pre-computed Fisher information matrix
            
        Returns:
            Updated parameters
        """
        if fisher_info_matrix is None:
            fisher_info_matrix = self._compute_fisher_information(parameters)
        
        # Add regularization
        fisher_regularized = fisher_info_matrix + self.regularization * np.eye(len(parameters))
        
        # Compute natural gradient
        try:
            natural_gradient = np.linalg.solve(fisher_regularized, gradients)
        except np.linalg.LinAlgError:
            # Fallback to regular gradient if Fisher matrix is singular
            natural_gradient = gradients
        
        # Update parameters
        new_parameters = parameters - self.learning_rate * natural_gradient
        
        return new_parameters
    
    def _compute_fisher_information(self, parameters: np.ndarray) -> np.ndarray:
        """
        Compute quantum Fisher information matrix.
        
        This is a simplified approximation - real QFI computation requires
        more sophisticated quantum circuit analysis.
        """
        n_params = len(parameters)
        fisher_matrix = np.zeros((n_params, n_params))
        
        # Diagonal approximation of Fisher information
        # Real implementation would compute off-diagonal terms
        for i in range(n_params):
            # Approximate Fisher information as inverse variance of gradient
            fisher_matrix[i, i] = 1.0 / (1.0 + np.abs(parameters[i]) ** 2)
        
        return fisher_matrix
    
    def reset(self):
        """Reset optimizer state."""
        self._cached_fisher = None
        self._cache_params = None


class AdaptiveLearningRate:
    """
    Adaptive learning rate scheduler for quantum machine learning.
    
    Adjusts learning rate based on training progress and noise characteristics.
    """
    
    def __init__(
        self,
        initial_lr: float = 0.01,
        decay_factor: float = 0.95,
        patience: int = 10,
        min_lr: float = 1e-6,
        noise_model: Optional[NoiseModel] = None,
    ):
        """
        Initialize adaptive learning rate scheduler.
        
        Args:
            initial_lr: Initial learning rate
            decay_factor: Factor to multiply learning rate on plateau
            patience: Number of steps to wait before reducing learning rate
            min_lr: Minimum learning rate
            noise_model: Noise model for adaptation
        """
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
        self.decay_factor = decay_factor
        self.patience = patience
        self.min_lr = min_lr
        self.noise_model = noise_model
        
        # State for plateau detection
        self.best_loss = float('inf')
        self.wait_count = 0
        self.step_count = 0
    
    def step(self, loss: float) -> float:
        """
        Update learning rate based on loss.
        
        Args:
            loss: Current training loss
            
        Returns:
            Updated learning rate
        """
        self.step_count += 1
        
        if loss < self.best_loss:
            self.best_loss = loss
            self.wait_count = 0
        else:
            self.wait_count += 1
        
        # Reduce learning rate on plateau
        if self.wait_count >= self.patience:
            old_lr = self.current_lr
            self.current_lr = max(self.min_lr, self.current_lr * self.decay_factor)
            self.wait_count = 0
            
            if self.current_lr < old_lr:
                print(f"Reducing learning rate to {self.current_lr:.6f}")
        
        # Noise-aware adaptation
        if self.noise_model:
            noise_factor = 1.0 / (1.0 + self.noise_model.gate_error_rate * 10)
            self.current_lr *= noise_factor
        
        return self.current_lr
    
    def reset(self):
        """Reset scheduler state."""
        self.current_lr = self.initial_lr
        self.best_loss = float('inf')
        self.wait_count = 0
        self.step_count = 0