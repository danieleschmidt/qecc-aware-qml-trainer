"""
Robust QECC-aware trainer with comprehensive error handling and validation.
Generation 2: Enhanced reliability, monitoring, and recovery.
"""

from typing import Tuple, Optional, Dict, Any, List, Union, Callable
import numpy as np
import time
import logging
from datetime import datetime
from pathlib import Path
import json

from ..core.quantum_nn import QECCAwareQNN
from ..core.noise_models import NoiseModel
from ..validation.circuit_validation import CircuitValidator, RobustErrorHandler
from .basic_trainer_clean import BasicQECCTrainer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('qecc_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class RobustQECCTrainer(BasicQECCTrainer):
    """
    Robust trainer with comprehensive validation, error handling, and monitoring.
    
    Generation 2 enhancements:
    - Input validation and sanitization
    - Comprehensive error handling and recovery
    - Training monitoring and checkpointing
    - Performance metrics tracking
    - Security measures for parameter validation
    """
    
    def __init__(
        self,
        qnn: QECCAwareQNN,
        noise_model: Optional[NoiseModel] = None,
        learning_rate: float = 0.01,
        shots: int = 1024,
        verbose: bool = True,
        validation_freq: int = 5,
        checkpoint_freq: int = 10,
        max_retries: int = 3,
        enable_monitoring: bool = True,
        log_level: str = 'INFO'
    ):
        """
        Initialize robust trainer with enhanced capabilities.
        
        Args:
            qnn: The quantum neural network to train
            noise_model: Optional noise model for simulation
            learning_rate: Learning rate for parameter updates
            shots: Number of measurement shots per evaluation
            verbose: Whether to print training progress
            validation_freq: How often to run validation (epochs)
            checkpoint_freq: How often to save checkpoints (epochs) 
            max_retries: Maximum retry attempts for failed operations
            enable_monitoring: Whether to enable performance monitoring
            log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        """
        super().__init__(qnn, noise_model, learning_rate, shots, verbose)
        
        # Enhanced components
        self.validator = CircuitValidator(strict_mode=False)
        self.error_handler = RobustErrorHandler(max_retries=max_retries)
        
        # Configuration
        self.validation_freq = validation_freq
        self.checkpoint_freq = checkpoint_freq
        self.enable_monitoring = enable_monitoring
        
        # Set logging level
        logger.setLevel(getattr(logging, log_level.upper()))
        
        # Enhanced tracking
        self.detailed_history = {
            'loss': [],
            'accuracy': [],
            'fidelity': [],
            'epoch_time': [],
            'validation_metrics': [],
            'error_events': [],
            'performance_metrics': {},
            'checkpoints': []
        }
        
        # Performance monitoring
        self.performance_monitor = {
            'peak_memory_usage': 0,
            'total_computation_time': 0,
            'successful_epochs': 0,
            'failed_epochs': 0,
            'recovery_events': 0
        }
        
        # Security and validation state
        self._validated_params = None
        self._last_checkpoint = None
        
        logger.info(f"Initialized RobustQECCTrainer with enhanced capabilities")
    
    def _validate_and_sanitize_inputs(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Comprehensive input validation and sanitization.
        
        Returns:
            Tuple of (validated_X, validated_y, sanitized_kwargs)
        """
        logger.debug("Validating and sanitizing inputs...")
        
        # Validate data compatibility
        data_validation = self.validator.validate_data_compatibility(
            X_train, y_train, self.qnn.num_qubits
        )
        
        if not data_validation['valid']:
            raise ValueError(f"Data validation failed: {data_validation['errors']}")
            
        # Log warnings
        for warning in data_validation['warnings']:
            logger.warning(warning)
            
        # Sanitize data
        X_sanitized = self._sanitize_features(X_train)
        y_sanitized = self._sanitize_labels(y_train)
        
        # Validate training parameters
        training_validation = self.validator.validate_training_parameters(
            self.learning_rate,
            kwargs.get('epochs', 50),
            kwargs.get('batch_size', 32),
            self.shots
        )
        
        if not training_validation['valid']:
            raise ValueError(f"Training parameter validation failed: {training_validation['errors']}")
            
        # Apply parameter corrections
        sanitized_kwargs = kwargs.copy()
        sanitized_kwargs.update(training_validation.get('corrected_params', {}))
        
        logger.info(f"Input validation completed successfully")
        return X_sanitized, y_sanitized, sanitized_kwargs
    
    def _sanitize_features(self, X: np.ndarray) -> np.ndarray:
        """Sanitize feature matrix for quantum processing."""
        X_clean = X.copy()
        
        # Handle NaN and infinite values
        if np.any(np.isnan(X_clean)):
            logger.warning("Found NaN values in features, replacing with zeros")
            X_clean = np.nan_to_num(X_clean, nan=0.0)
            
        if np.any(np.isinf(X_clean)):
            logger.warning("Found infinite values in features, clipping to [-1, 1]")
            X_clean = np.clip(X_clean, -1.0, 1.0)
            
        # Normalize to quantum-friendly range
        data_range = X_clean.max() - X_clean.min()
        if data_range > 2.0:  # Outside [-1, 1]
            X_clean = 2 * (X_clean - X_clean.min()) / data_range - 1
            logger.info("Normalized features to [-1, 1] range")
            
        return X_clean.astype(np.float32)
    
    def _sanitize_labels(self, y: np.ndarray) -> np.ndarray:
        """Sanitize label vector."""
        y_clean = y.copy()
        
        # Ensure binary classification
        unique_labels = np.unique(y_clean)
        if len(unique_labels) > 2:
            logger.warning(f"Found {len(unique_labels)} classes, converting to binary")
            y_clean = (y_clean == unique_labels[0]).astype(int)
            
        return y_clean.astype(int)
    
    def _execute_with_retry(
        self,
        operation: Callable,
        operation_name: str,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute operation with retry logic and error handling.
        
        Args:
            operation: Function to execute
            operation_name: Name for logging
            *args: Position arguments for operation
            **kwargs: Keyword arguments for operation
            
        Returns:
            Operation result
        """
        last_error = None
        
        for attempt in range(self.error_handler.max_retries + 1):
            try:
                if attempt > 0:
                    # Exponential backoff
                    sleep_time = self.error_handler.backoff_factor ** attempt
                    time.sleep(sleep_time)
                    logger.info(f"Retry attempt {attempt} for {operation_name}")
                
                result = operation(*args, **kwargs)
                
                if attempt > 0:
                    self.performance_monitor['recovery_events'] += 1
                    logger.info(f"Successfully recovered {operation_name} after {attempt} attempts")
                
                return result
                
            except Exception as e:
                last_error = e
                recovery_strategy = self.error_handler.handle_circuit_execution_error(
                    e, {'operation': operation_name, 'attempt': attempt}
                )
                
                logger.warning(f"Error in {operation_name} (attempt {attempt}): {str(e)}")
                
                if recovery_strategy and attempt < self.error_handler.max_retries:
                    logger.info(f"Applying recovery strategy: {recovery_strategy}")
                    # Apply recovery strategy
                    if recovery_strategy == 'reduce_batch_size' and 'batch_size' in kwargs:
                        kwargs['batch_size'] = max(1, kwargs['batch_size'] // 2)
                        logger.info(f"Reduced batch size to {kwargs['batch_size']}")
                else:
                    break
        
        # All retries failed
        logger.error(f"All retry attempts failed for {operation_name}")
        self.detailed_history['error_events'].append({
            'timestamp': datetime.now().isoformat(),
            'operation': operation_name,
            'error': str(last_error),
            'attempts': self.error_handler.max_retries + 1
        })
        raise last_error
    
    def _create_checkpoint(self, epoch: int) -> Dict[str, Any]:
        """Create training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
            'parameters': self.current_params.copy() if self.current_params is not None else None,
            'history': self.detailed_history.copy(),
            'performance_monitor': self.performance_monitor.copy(),
            'learning_rate': self.learning_rate,
            'qnn_config': {
                'num_qubits': self.qnn.num_qubits,
                'num_layers': self.qnn.num_layers,
                'entanglement': self.qnn.entanglement
            }
        }
        
        # Save checkpoint to file
        checkpoint_path = Path(f"checkpoint_epoch_{epoch}.json")
        try:
            with open(checkpoint_path, 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                checkpoint_serializable = checkpoint.copy()
                if checkpoint_serializable['parameters'] is not None:
                    checkpoint_serializable['parameters'] = checkpoint_serializable['parameters'].tolist()
                json.dump(checkpoint_serializable, f, indent=2)
                
            self._last_checkpoint = checkpoint_path
            self.detailed_history['checkpoints'].append(str(checkpoint_path))
            logger.info(f"Created checkpoint at epoch {epoch}: {checkpoint_path}")
            
        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")
            
        return checkpoint
    
    def _monitor_performance(self, epoch_start_time: float, epoch: int):
        """Monitor and track performance metrics."""
        if not self.enable_monitoring:
            return
            
        epoch_time = time.time() - epoch_start_time
        self.performance_monitor['total_computation_time'] += epoch_time
        
        # Memory usage monitoring (simplified)
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            self.performance_monitor['peak_memory_usage'] = max(
                self.performance_monitor['peak_memory_usage'], memory_mb
            )
        except ImportError:
            # psutil not available, skip memory monitoring
            pass
        
        # Calculate performance metrics
        if epoch > 0:
            avg_epoch_time = self.performance_monitor['total_computation_time'] / (epoch + 1)
            self.detailed_history['performance_metrics'][f'epoch_{epoch}'] = {
                'epoch_time': epoch_time,
                'avg_epoch_time': avg_epoch_time,
                'memory_usage_mb': self.performance_monitor.get('peak_memory_usage', 0)
            }
    
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
        
        Enhanced with:
        - Comprehensive input validation
        - Error handling and recovery
        - Performance monitoring
        - Checkpointing
        - Security measures
        """
        logger.info(f"Starting robust training for {epochs} epochs...")
        
        try:
            # Validate and sanitize inputs
            X_clean, y_clean, kwargs_clean = self._validate_and_sanitize_inputs(
                X_train, y_train, 
                epochs=epochs, batch_size=batch_size, validation_split=validation_split
            )
            
            # Extract cleaned parameters
            epochs = kwargs_clean.get('epochs', epochs)
            batch_size = kwargs_clean.get('batch_size', batch_size)
            validation_split = kwargs_clean.get('validation_split', validation_split)
            
            # Initialize parameters with validation
            if self.current_params is None:
                num_params = len(self.qnn.weight_params)
                self.current_params = np.random.normal(0, 0.1, num_params)
                logger.info(f"Initialized {num_params} parameters")
            
            # Split validation data if not provided
            if X_val is None and validation_split > 0:
                val_size = int(len(X_clean) * validation_split)
                val_indices = np.random.choice(len(X_clean), val_size, replace=False)
                train_indices = np.setdiff1d(np.arange(len(X_clean)), val_indices)
                
                X_val = X_clean[val_indices]
                y_val = y_clean[val_indices]
                X_train = X_clean[train_indices]
                y_train = y_clean[train_indices]
            else:
                X_train, y_train = X_clean, y_clean
            
            logger.info(f"Training on {len(X_train)} samples, validating on {len(X_val) if X_val is not None else 0}")
            
            # Training loop with enhanced error handling
            for epoch in range(epochs):
                epoch_start_time = time.time()
                
                try:
                    # Execute training epoch with retry logic
                    epoch_result = self._execute_with_retry(
                        self._train_epoch,
                        f"epoch_{epoch}",
                        X_train, y_train, X_val, y_val, batch_size, epoch
                    )
                    
                    # Update history
                    self.detailed_history['loss'].append(epoch_result['loss'])
                    self.detailed_history['accuracy'].append(epoch_result['accuracy'])
                    self.detailed_history['fidelity'].append(epoch_result['fidelity'])
                    
                    # Monitor performance
                    self._monitor_performance(epoch_start_time, epoch)
                    
                    # Validation
                    if epoch % self.validation_freq == 0 and X_val is not None:
                        val_result = self._execute_with_retry(
                            self._validate_epoch,
                            f"validation_{epoch}",
                            X_val, y_val
                        )
                        self.detailed_history['validation_metrics'].append({
                            'epoch': epoch,
                            'val_loss': val_result['loss'],
                            'val_accuracy': val_result['accuracy']
                        })
                    
                    # Checkpointing
                    if epoch % self.checkpoint_freq == 0:
                        self._create_checkpoint(epoch)
                    
                    # Progress reporting
                    if self.verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                        self._print_progress(epoch, epochs, epoch_result, 
                                           time.time() - epoch_start_time)
                    
                    self.performance_monitor['successful_epochs'] += 1
                    
                except Exception as e:
                    logger.error(f"Epoch {epoch} failed: {e}")
                    self.performance_monitor['failed_epochs'] += 1
                    
                    # Decide whether to continue or abort
                    if self.performance_monitor['failed_epochs'] > epochs * 0.1:  # >10% failure rate
                        logger.error("Too many failed epochs, aborting training")
                        raise
                    
                    continue
            
            # Final validation and reporting
            final_results = self._generate_final_report(X_val, y_val if X_val is not None else None)
            logger.info("Robust training completed successfully")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            raise
    
    def _train_epoch(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray],
        batch_size: int,
        epoch: int
    ) -> Dict[str, float]:
        """Execute a single training epoch."""
        
        # Shuffle training data
        shuffle_indices = np.random.permutation(len(X_train))
        X_shuffled = X_train[shuffle_indices]
        y_shuffled = y_train[shuffle_indices]
        
        epoch_losses = []
        epoch_accuracies = []
        
        # Mini-batch training
        for i in range(0, len(X_train), batch_size):
            batch_X = X_shuffled[i:i+batch_size]
            batch_y = y_shuffled[i:i+batch_size]
            
            # Compute gradients with error handling
            gradients = self._compute_gradients(self.current_params, batch_X, batch_y)
            
            # Parameter update with validation
            param_update = self.learning_rate * gradients
            if np.any(np.isnan(param_update)) or np.any(np.isinf(param_update)):
                logger.warning("Invalid parameter update detected, skipping batch")
                continue
                
            self.current_params -= param_update
            
            # Evaluate current batch
            batch_loss, batch_acc = self._evaluate_circuit(self.current_params, batch_X, batch_y)
            epoch_losses.append(batch_loss)
            epoch_accuracies.append(batch_acc)
        
        # Calculate epoch metrics
        avg_loss = np.mean(epoch_losses) if epoch_losses else float('inf')
        avg_accuracy = np.mean(epoch_accuracies) if epoch_accuracies else 0.0
        fidelity = max(0.5, 1.0 - avg_loss)
        
        return {
            'loss': avg_loss,
            'accuracy': avg_accuracy,
            'fidelity': fidelity
        }
    
    def _validate_epoch(self, X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, float]:
        """Execute validation for current epoch."""
        val_loss, val_accuracy = self._evaluate_circuit(self.current_params, X_val, y_val)
        return {'loss': val_loss, 'accuracy': val_accuracy}
    
    def _print_progress(self, epoch: int, total_epochs: int, epoch_result: Dict, epoch_time: float):
        """Print enhanced progress information."""
        print(f"Epoch {epoch+1}/{total_epochs}: "
              f"Loss: {epoch_result['loss']:.4f}, "
              f"Acc: {epoch_result['accuracy']:.4f}, "
              f"Fidelity: {epoch_result['fidelity']:.4f}, "
              f"Time: {epoch_time:.2f}s, "
              f"Success Rate: {self.performance_monitor['successful_epochs']/(epoch+1):.2%}")
    
    def _generate_final_report(self, X_val: Optional[np.ndarray], y_val: Optional[np.ndarray]) -> Dict[str, Any]:
        """Generate comprehensive training report."""
        
        # Base history
        final_report = self.detailed_history.copy()
        
        # Performance summary
        final_report['performance_summary'] = {
            'total_epochs': len(self.detailed_history['loss']),
            'successful_epochs': self.performance_monitor['successful_epochs'],
            'failed_epochs': self.performance_monitor['failed_epochs'],
            'success_rate': (self.performance_monitor['successful_epochs'] / 
                           max(1, self.performance_monitor['successful_epochs'] + self.performance_monitor['failed_epochs'])),
            'total_training_time': self.performance_monitor['total_computation_time'],
            'avg_epoch_time': (self.performance_monitor['total_computation_time'] / 
                              max(1, len(self.detailed_history['loss']))),
            'recovery_events': self.performance_monitor['recovery_events']
        }
        
        # Validation summary
        if X_val is not None and y_val is not None:
            final_validation = self.evaluate(X_val, y_val)
            final_report['final_validation'] = final_validation
        
        # Error analysis
        final_report['error_analysis'] = {
            'total_errors': len(self.detailed_history['error_events']),
            'error_statistics': self.error_handler.get_error_statistics(),
            'validation_report': self.validator.get_validation_report()
        }
        
        return final_report
    
    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """Load training state from checkpoint."""
        try:
            with open(checkpoint_path, 'r') as f:
                checkpoint = json.load(f)
            
            if checkpoint.get('parameters'):
                self.current_params = np.array(checkpoint['parameters'])
            
            self.detailed_history = checkpoint.get('history', self.detailed_history)
            self.performance_monitor = checkpoint.get('performance_monitor', self.performance_monitor)
            
            logger.info(f"Successfully loaded checkpoint from {checkpoint_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return False
    
    def get_training_diagnostics(self) -> Dict[str, Any]:
        """Get comprehensive training diagnostics."""
        return {
            'validation_report': self.validator.get_validation_report(),
            'error_statistics': self.error_handler.get_error_statistics(),
            'performance_metrics': self.performance_monitor,
            'training_stability': {
                'parameter_variance': np.var(self.current_params) if self.current_params is not None else 0,
                'loss_trend': np.polyfit(range(len(self.detailed_history['loss'])), 
                                       self.detailed_history['loss'], 1)[0] if self.detailed_history['loss'] else 0,
                'convergence_rate': len([l for l in self.detailed_history['loss'][-10:] 
                                       if abs(l - self.detailed_history['loss'][-1]) < 0.01]) / 10 if len(self.detailed_history['loss']) >= 10 else 0
            }
        }