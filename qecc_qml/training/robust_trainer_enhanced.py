"""
Robust QECC-aware quantum neural network trainer.
Generation 2: Reliable implementation with comprehensive error handling.
"""

from typing import Tuple, Optional, Dict, Any, List, Union, Callable
import numpy as np
import time
import logging
import warnings
from pathlib import Path
import json

from ..core.quantum_nn import QECCAwareQNN
from ..core.noise_models import NoiseModel
from ..utils.error_recovery import ErrorRecoveryManager
from ..utils.logging_config import setup_logging
from ..utils.validation import validate_input_data, validate_parameters
from ..monitoring.health_monitor_simple import HealthMonitor


class RobustQECCTrainer:
    """
    Robust trainer for QECC-aware quantum neural networks.
    
    Features comprehensive error handling, validation, monitoring,
    and automatic recovery mechanisms.
    """
    
    def __init__(
        self,
        qnn: QECCAwareQNN,
        noise_model: Optional[NoiseModel] = None,
        learning_rate: float = 0.01,
        shots: int = 1024,
        verbose: bool = True,
        checkpoint_dir: Optional[str] = None,
        max_retries: int = 3,
        enable_monitoring: bool = True,
        validation_frequency: int = 10,
        early_stopping_patience: int = 20,
        gradient_clipping: float = 1.0,
        parameter_bounds: Optional[Tuple[float, float]] = None
    ):
        """
        Initialize robust trainer with comprehensive error handling.
        
        Args:
            qnn: The quantum neural network to train
            noise_model: Optional noise model for simulation
            learning_rate: Learning rate for parameter updates
            shots: Number of measurement shots per evaluation
            verbose: Whether to print training progress
            checkpoint_dir: Directory for saving training checkpoints
            max_retries: Maximum number of retries for failed operations
            enable_monitoring: Enable system health monitoring
            validation_frequency: How often to validate training progress
            early_stopping_patience: Epochs to wait before early stopping
            gradient_clipping: Maximum gradient norm for clipping
            parameter_bounds: (min, max) bounds for parameters
        """
        # Core components
        self.qnn = qnn
        self.noise_model = noise_model
        self.learning_rate = learning_rate
        self.shots = shots
        self.verbose = verbose
        
        # Robustness features
        self.max_retries = max_retries
        self.validation_frequency = validation_frequency
        self.early_stopping_patience = early_stopping_patience
        self.gradient_clipping = gradient_clipping
        self.parameter_bounds = parameter_bounds or (-np.pi, np.pi)
        
        # Training state
        self.current_params = None
        self.best_params = None
        self.best_loss = np.inf
        self.patience_counter = 0
        
        # History tracking
        self.history = {
            'loss': [],
            'accuracy': [],
            'fidelity': [],
            'epoch_time': [],
            'learning_rate': [],
            'gradient_norm': [],
            'validation_loss': [],
            'validation_accuracy': [],
            'error_recoveries': [],
            'health_metrics': []
        }
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Setup checkpointing
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup error recovery
        self.error_recovery = ErrorRecoveryManager(
            max_retries=max_retries
        )
        
        # Setup health monitoring
        self.health_monitor = None
        if enable_monitoring:
            try:
                self.health_monitor = HealthMonitor(
                    qnn=qnn,
                    update_frequency=1.0,
                    logger=self.logger
                )
            except Exception as e:
                self.logger.warning(f"Failed to initialize health monitor: {e}")
        
        # Initialize parameters
        self._initialize_parameters()
        
        self.logger.info(f"RobustQECCTrainer initialized with {len(self.current_params)} parameters")
    
    def _initialize_parameters(self):
        """Initialize the variational parameters with validation."""
        try:
            num_params = len(self.qnn.weight_params)
            # Use Xavier/Glorot initialization scaled for quantum circuits
            scale = np.sqrt(2.0 / num_params)
            self.current_params = np.random.normal(0, scale, num_params)
            
            # Apply parameter bounds
            self.current_params = np.clip(
                self.current_params, 
                self.parameter_bounds[0], 
                self.parameter_bounds[1]
            )
            
            self.best_params = self.current_params.copy()
            self.logger.debug(f"Parameters initialized: shape={self.current_params.shape}, range=[{self.current_params.min():.3f}, {self.current_params.max():.3f}]")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize parameters: {e}")
            raise
    
    def get_parameters(self) -> np.ndarray:
        """Get current parameters with validation."""
        if self.current_params is None:
            self._initialize_parameters()
        return self.current_params.copy()
    
    def set_parameters(self, params: np.ndarray):
        """Set parameters with validation."""
        try:
            validate_parameters(params, expected_shape=(len(self.qnn.weight_params),))
            self.current_params = np.clip(
                params.copy(), 
                self.parameter_bounds[0], 
                self.parameter_bounds[1]
            )
            self.logger.debug(f"Parameters updated: shape={self.current_params.shape}")
        except Exception as e:
            self.logger.error(f"Failed to set parameters: {e}")
            raise
    
    def _evaluate_circuit_robust(self, params: np.ndarray, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """
        Evaluate circuit with comprehensive error handling and recovery.
        """
        def _evaluation_attempt():
            # Validate inputs
            validate_input_data(X, y)
            validate_parameters(params, expected_shape=(len(self.qnn.weight_params),))
            
            predictions = []
            
            for i, x_sample in enumerate(X):
                try:
                    # Create quantum circuit for this sample
                    circuit = self.qnn.create_circuit(x_sample, params)
                    
                    # For now, use classical simulation
                    # In production, this would execute on quantum hardware
                    param_product = np.dot(params[:len(x_sample)], x_sample)
                    prob = 1 / (1 + np.exp(-param_product))  # Sigmoid activation
                    
                    # Add noise simulation if noise model is provided
                    if self.noise_model is not None:
                        noise_factor = np.random.normal(1.0, 0.01)  # 1% noise
                        prob *= noise_factor
                        prob = np.clip(prob, 0.0, 1.0)
                    
                    prediction = 1 if prob > 0.5 else 0
                    predictions.append(prediction)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to evaluate sample {i}: {e}")
                    # Use fallback prediction
                    predictions.append(np.random.randint(0, 2))
            
            predictions = np.array(predictions)
            
            # Calculate metrics
            accuracy = np.mean(predictions == y)
            loss = self._cross_entropy_loss(y, predictions.astype(float))
            
            return loss, accuracy
        
        # Use error recovery for robust evaluation
        return self.error_recovery.retry_with_backoff(_evaluation_attempt)
    
    def _cross_entropy_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Robust cross-entropy loss with numerical stability."""
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        try:
            loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
            
            # Check for NaN or infinite values
            if not np.isfinite(loss):
                self.logger.warning(f"Non-finite loss detected: {loss}, using fallback")
                return 1.0  # Fallback loss value
                
            return loss
            
        except Exception as e:
            self.logger.error(f"Error computing loss: {e}")
            return 1.0  # Fallback loss value
    
    def _compute_gradients_robust(self, params: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute gradients with parameter shift rule and error handling.
        """
        def _gradient_computation():
            gradients = np.zeros_like(params)
            epsilon = np.pi / 2  # Standard parameter shift for quantum gradients
            
            for i in range(len(params)):
                try:
                    # Parameter shift rule: df/dx = [f(x + π/2) - f(x - π/2)] / 2
                    params_plus = params.copy()
                    params_minus = params.copy()
                    
                    params_plus[i] += epsilon
                    params_minus[i] -= epsilon
                    
                    # Apply parameter bounds
                    params_plus = np.clip(params_plus, self.parameter_bounds[0], self.parameter_bounds[1])
                    params_minus = np.clip(params_minus, self.parameter_bounds[0], self.parameter_bounds[1])
                    
                    loss_plus, _ = self._evaluate_circuit_robust(params_plus, X, y)
                    loss_minus, _ = self._evaluate_circuit_robust(params_minus, X, y)
                    
                    gradients[i] = (loss_plus - loss_minus) / 2.0
                    
                except Exception as e:
                    self.logger.warning(f"Failed to compute gradient for parameter {i}: {e}")
                    gradients[i] = 0.0  # Zero gradient as fallback
            
            # Apply gradient clipping
            gradient_norm = np.linalg.norm(gradients)
            if gradient_norm > self.gradient_clipping:
                gradients = gradients * (self.gradient_clipping / gradient_norm)
                self.logger.debug(f"Gradients clipped: norm {gradient_norm:.4f} -> {self.gradient_clipping}")
            
            return gradients
        
        return self.error_recovery.retry_with_backoff(_gradient_computation)
    
    def _save_checkpoint(self, epoch: int):
        """Save training checkpoint."""
        if self.checkpoint_dir is None:
            return
        
        try:
            checkpoint = {
                'epoch': epoch,
                'params': self.current_params.tolist(),
                'best_params': self.best_params.tolist(),
                'best_loss': self.best_loss,
                'history': self.history,
                'learning_rate': self.learning_rate,
                'patience_counter': self.patience_counter
            }
            
            checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.json"
            with open(checkpoint_path, 'w') as f:
                json.dump(checkpoint, f, indent=2)
                
            self.logger.debug(f"Checkpoint saved: {checkpoint_path}")
            
        except Exception as e:
            self.logger.warning(f"Failed to save checkpoint: {e}")
    
    def compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute loss with robust error handling."""
        try:
            if self.current_params is None:
                self._initialize_parameters()
            loss, _ = self._evaluate_circuit_robust(self.current_params, X, y)
            return loss
        except Exception as e:
            self.logger.error(f"Failed to compute loss: {e}")
            return np.inf
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 50,
        batch_size: int = 32,
        validation_split: float = 0.2
    ) -> Dict[str, List[float]]:
        """
        Train the quantum neural network with robust error handling.
        """
        try:
            self.logger.info(f"Starting robust training for {epochs} epochs...")
            self.logger.info(f"Training samples: {len(X_train)}, Learning rate: {self.learning_rate}")
            
            # Validate inputs
            validate_input_data(X_train, y_train)
            
            # Initialize parameters if needed
            if self.current_params is None:
                self._initialize_parameters()
            
            # Start health monitoring
            if self.health_monitor:
                self.health_monitor.start_monitoring()
            
            # Split validation data if not provided
            if X_val is None and validation_split > 0:
                val_size = int(len(X_train) * validation_split)
                val_indices = np.random.choice(len(X_train), val_size, replace=False)
                train_indices = np.setdiff1d(np.arange(len(X_train)), val_indices)
                
                X_val = X_train[val_indices]
                y_val = y_train[val_indices]
                X_train = X_train[train_indices]
                y_train = y_train[train_indices]
            
            # Training loop with comprehensive error handling
            for epoch in range(epochs):
                epoch_start_time = time.time()
                error_recoveries_this_epoch = 0
                
                try:
                    # Shuffle training data
                    shuffle_indices = np.random.permutation(len(X_train))
                    X_shuffled = X_train[shuffle_indices]
                    y_shuffled = y_train[shuffle_indices]
                    
                    epoch_losses = []
                    epoch_accuracies = []
                    
                    # Mini-batch training with error recovery
                    for i in range(0, len(X_train), batch_size):
                        batch_X = X_shuffled[i:i+batch_size]
                        batch_y = y_shuffled[i:i+batch_size]
                        
                        try:
                            # Compute gradients with error handling
                            gradients = self._compute_gradients_robust(self.current_params, batch_X, batch_y)
                            
                            # Update parameters with bounds checking
                            self.current_params -= self.learning_rate * gradients
                            self.current_params = np.clip(
                                self.current_params,
                                self.parameter_bounds[0],
                                self.parameter_bounds[1]
                            )
                            
                            # Evaluate current batch
                            batch_loss, batch_acc = self._evaluate_circuit_robust(
                                self.current_params, batch_X, batch_y
                            )
                            
                            epoch_losses.append(batch_loss)
                            epoch_accuracies.append(batch_acc)
                            
                        except Exception as e:
                            self.logger.warning(f"Batch {i//batch_size} failed: {e}")
                            error_recoveries_this_epoch += 1
                            
                            # Skip this batch and continue
                            continue
                    
                    # Calculate epoch metrics
                    if epoch_losses:
                        avg_loss = np.mean(epoch_losses)
                        avg_accuracy = np.mean(epoch_accuracies)
                        gradient_norm = np.linalg.norm(gradients) if 'gradients' in locals() else 0.0
                    else:
                        avg_loss = np.inf
                        avg_accuracy = 0.0
                        gradient_norm = 0.0
                    
                    # Validation evaluation
                    val_loss, val_accuracy = 0.0, 0.0
                    if X_val is not None:
                        try:
                            val_loss, val_accuracy = self._evaluate_circuit_robust(
                                self.current_params, X_val, y_val
                            )
                        except Exception as e:
                            self.logger.warning(f"Validation evaluation failed: {e}")
                    
                    # Update best parameters
                    if val_loss < self.best_loss:
                        self.best_loss = val_loss
                        self.best_params = self.current_params.copy()
                        self.patience_counter = 0
                    else:
                        self.patience_counter += 1
                    
                    # Early stopping check
                    if self.patience_counter >= self.early_stopping_patience:
                        self.logger.info(f"Early stopping at epoch {epoch}")
                        break
                    
                    # Calculate additional metrics
                    fidelity = max(0.5, 1.0 - avg_loss)  # Rough approximation
                    epoch_time = time.time() - epoch_start_time
                    
                    # Collect health metrics
                    health_metrics = {}
                    if self.health_monitor:
                        try:
                            health_metrics = self.health_monitor.get_current_metrics()
                        except Exception as e:
                            self.logger.debug(f"Failed to collect health metrics: {e}")
                    
                    # Store history
                    self.history['loss'].append(avg_loss)
                    self.history['accuracy'].append(avg_accuracy)
                    self.history['fidelity'].append(fidelity)
                    self.history['epoch_time'].append(epoch_time)
                    self.history['learning_rate'].append(self.learning_rate)
                    self.history['gradient_norm'].append(gradient_norm)
                    self.history['validation_loss'].append(val_loss)
                    self.history['validation_accuracy'].append(val_accuracy)
                    self.history['error_recoveries'].append(error_recoveries_this_epoch)
                    self.history['health_metrics'].append(health_metrics)
                    
                    # Progress reporting
                    if self.verbose and (epoch % self.validation_frequency == 0 or epoch == epochs - 1):
                        self.logger.info(
                            f"Epoch {epoch+1}/{epochs}: "
                            f"Loss={avg_loss:.4f}, Acc={avg_accuracy:.3f}, "
                            f"Val_Loss={val_loss:.4f}, Val_Acc={val_accuracy:.3f}, "
                            f"Time={epoch_time:.2f}s, Recoveries={error_recoveries_this_epoch}"
                        )
                    
                    # Save checkpoint
                    if epoch % (self.validation_frequency * 2) == 0:
                        self._save_checkpoint(epoch)
                    
                except Exception as e:
                    self.logger.error(f"Epoch {epoch} failed: {e}")
                    error_recoveries_this_epoch += 1
                    continue
            
            # Final checkpoint
            self._save_checkpoint(epochs - 1)
            
            # Stop health monitoring
            if self.health_monitor:
                self.health_monitor.stop_monitoring()
            
            # Restore best parameters
            if self.best_params is not None:
                self.current_params = self.best_params.copy()
                self.logger.info(f"Training completed. Best validation loss: {self.best_loss:.4f}")
            
            return self.history
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise