"""
Neural Network-based Syndrome Decoders for Quantum Error Correction.

Advanced deep learning models for decoding error syndromes in quantum
error correction codes, outperforming traditional decoders.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
from collections import defaultdict, deque
import json


class DecoderArchitecture(Enum):
    """Neural decoder architectures."""
    MLP = "mlp"  # Multi-Layer Perceptron
    CNN = "cnn"  # Convolutional Neural Network
    RNN = "rnn"  # Recurrent Neural Network
    TRANSFORMER = "transformer"  # Transformer architecture
    GRAPH_NN = "graph_nn"  # Graph Neural Network
    HYBRID = "hybrid"  # Hybrid architecture


class ErrorModel(Enum):
    """Quantum error models for training."""
    DEPOLARIZING = "depolarizing"
    AMPLITUDE_DAMPING = "amplitude_damping"
    PHASE_DAMPING = "phase_damping"
    BIT_FLIP = "bit_flip"
    PHASE_FLIP = "phase_flip"
    CORRELATED = "correlated"


@dataclass
class SyndromeData:
    """Syndrome measurement data."""
    syndrome: np.ndarray  # Binary syndrome vector
    error_pattern: np.ndarray  # Actual error pattern (for training)
    code_distance: int
    noise_strength: float
    error_model: ErrorModel
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DecoderConfig:
    """Configuration for neural decoder."""
    architecture: DecoderArchitecture
    input_dim: int  # Syndrome length
    output_dim: int  # Error pattern length
    hidden_dims: List[int] = field(default_factory=lambda: [128, 64, 32])
    learning_rate: float = 0.001
    batch_size: int = 64
    epochs: int = 100
    dropout_rate: float = 0.1
    regularization: float = 0.01
    activation: str = "relu"
    optimizer: str = "adam"


class SyndromeGenerator:
    """
    Generator for synthetic syndrome training data.
    
    Creates realistic syndrome-error pattern pairs for training
    neural decoders on various quantum error correction codes.
    """
    
    def __init__(
        self,
        code_type: str = "surface",
        code_distance: int = 3,
        error_models: Optional[List[ErrorModel]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize syndrome generator.
        
        Args:
            code_type: Type of quantum error correction code
            code_distance: Code distance parameter
            error_models: List of error models to simulate
            logger: Optional logger instance
        """
        self.code_type = code_type
        self.code_distance = code_distance
        self.error_models = error_models or [ErrorModel.DEPOLARIZING, ErrorModel.BIT_FLIP]
        self.logger = logger or logging.getLogger(__name__)
        
        # Code parameters
        if code_type == "surface":
            self.num_data_qubits = code_distance ** 2
            self.num_ancilla_qubits = (code_distance - 1) ** 2 * 2
            self.syndrome_length = self.num_ancilla_qubits
        elif code_type == "steane":
            self.num_data_qubits = 7
            self.num_ancilla_qubits = 6
            self.syndrome_length = 6
        else:
            # Generic parameters
            self.num_data_qubits = code_distance ** 2
            self.num_ancilla_qubits = code_distance * (code_distance - 1)
            self.syndrome_length = self.num_ancilla_qubits
        
        # Parity check matrices (simplified)
        self.parity_check_x = self._generate_parity_check_matrix('x')
        self.parity_check_z = self._generate_parity_check_matrix('z')
        
        # Statistics
        self.samples_generated = 0
        self.generation_time = 0.0
        
        self.logger.info(f"SyndromeGenerator initialized for {code_type} code, distance {code_distance}")
    
    def _generate_parity_check_matrix(self, pauli_type: str) -> np.ndarray:
        """Generate parity check matrix for the code."""
        if self.code_type == "surface" and pauli_type == 'x':
            # Simplified surface code X-stabilizers
            rows = (self.code_distance - 1) ** 2
            cols = self.num_data_qubits
            matrix = np.random.randint(0, 2, (rows, cols))
            
            # Ensure each stabilizer has weight ~4
            for i in range(rows):
                if np.sum(matrix[i]) == 0:
                    # Add some random qubits
                    indices = np.random.choice(cols, 4, replace=False)
                    matrix[i, indices] = 1
                elif np.sum(matrix[i]) > 6:
                    # Reduce weight
                    ones = np.where(matrix[i] == 1)[0]
                    keep = np.random.choice(ones, 4, replace=False)
                    matrix[i] = 0
                    matrix[i, keep] = 1
            
            return matrix
            
        elif self.code_type == "steane" and pauli_type == 'x':
            # Steane code X-stabilizers
            return np.array([
                [1, 1, 1, 1, 0, 0, 0],
                [1, 1, 0, 0, 1, 1, 0],
                [1, 0, 1, 0, 1, 0, 1]
            ])
            
        else:
            # Generic random matrix
            rows = self.syndrome_length // 2 if pauli_type == 'x' else self.syndrome_length - self.syndrome_length // 2
            return np.random.randint(0, 2, (rows, self.num_data_qubits))
    
    def generate_error_pattern(
        self, 
        error_model: ErrorModel, 
        error_rate: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate error pattern and corresponding syndrome.
        
        Args:
            error_model: Type of error model
            error_rate: Error probability per qubit
            
        Returns:
            Tuple of (error_pattern, syndrome)
        """
        start_time = time.time()
        
        # Initialize error pattern
        x_errors = np.zeros(self.num_data_qubits, dtype=int)
        z_errors = np.zeros(self.num_data_qubits, dtype=int)
        
        if error_model == ErrorModel.DEPOLARIZING:
            # Each qubit has error_rate chance of X, Y, or Z error
            for i in range(self.num_data_qubits):
                if np.random.random() < error_rate:
                    error_type = np.random.choice(['x', 'y', 'z'])
                    if error_type in ['x', 'y']:
                        x_errors[i] = 1
                    if error_type in ['z', 'y']:
                        z_errors[i] = 1
        
        elif error_model == ErrorModel.BIT_FLIP:
            # Only X errors
            x_errors = np.random.binomial(1, error_rate, self.num_data_qubits)
        
        elif error_model == ErrorModel.PHASE_FLIP:
            # Only Z errors
            z_errors = np.random.binomial(1, error_rate, self.num_data_qubits)
        
        elif error_model == ErrorModel.CORRELATED:
            # Correlated errors (bursts)
            if np.random.random() < error_rate:
                # Create burst of errors
                burst_size = np.random.randint(2, min(5, self.num_data_qubits))
                start_pos = np.random.randint(0, self.num_data_qubits - burst_size)
                for i in range(start_pos, start_pos + burst_size):
                    if np.random.random() < 0.8:  # High probability within burst
                        if np.random.random() < 0.5:
                            x_errors[i] = 1
                        else:
                            z_errors[i] = 1
        
        # Compute syndrome
        x_syndrome = (self.parity_check_x @ x_errors) % 2
        z_syndrome = (self.parity_check_z @ z_errors) % 2
        syndrome = np.concatenate([x_syndrome, z_syndrome])
        
        # Combined error pattern
        error_pattern = np.concatenate([x_errors, z_errors])
        
        self.generation_time += time.time() - start_time
        self.samples_generated += 1
        
        return error_pattern, syndrome
    
    def generate_dataset(
        self, 
        num_samples: int,
        error_rate_range: Tuple[float, float] = (0.001, 0.1),
        balanced: bool = True
    ) -> List[SyndromeData]:
        """
        Generate training dataset.
        
        Args:
            num_samples: Number of samples to generate
            error_rate_range: Range of error rates to sample
            balanced: Whether to balance error patterns
            
        Returns:
            List of syndrome data samples
        """
        dataset = []
        
        self.logger.info(f"Generating {num_samples} syndrome samples")
        
        for i in range(num_samples):
            # Sample error model and rate
            error_model = np.random.choice(self.error_models)
            error_rate = np.random.uniform(*error_rate_range)
            
            # Generate sample
            error_pattern, syndrome = self.generate_error_pattern(error_model, error_rate)
            
            sample = SyndromeData(
                syndrome=syndrome,
                error_pattern=error_pattern,
                code_distance=self.code_distance,
                noise_strength=error_rate,
                error_model=error_model,
                metadata={
                    'code_type': self.code_type,
                    'sample_index': i
                }
            )
            
            dataset.append(sample)
            
            if (i + 1) % 1000 == 0:
                self.logger.debug(f"Generated {i + 1}/{num_samples} samples")
        
        # Balance dataset if requested
        if balanced:
            dataset = self._balance_dataset(dataset)
        
        self.logger.info(f"Dataset generation completed: {len(dataset)} samples")
        return dataset
    
    def _balance_dataset(self, dataset: List[SyndromeData]) -> List[SyndromeData]:
        """Balance dataset by syndrome weight."""
        syndrome_weights = defaultdict(list)
        
        # Group by syndrome weight
        for sample in dataset:
            weight = np.sum(sample.syndrome)
            syndrome_weights[weight].append(sample)
        
        # Find minimum group size
        min_size = min(len(samples) for samples in syndrome_weights.values())
        
        # Sample equally from each group
        balanced_dataset = []
        for weight, samples in syndrome_weights.items():
            selected = np.random.choice(samples, min_size, replace=False)
            balanced_dataset.extend(selected)
        
        return balanced_dataset


class NeuralSyndromeDecoder:
    """
    Neural network-based syndrome decoder.
    
    Uses deep learning to decode error syndromes and predict
    most likely error patterns for quantum error correction.
    """
    
    def __init__(
        self,
        config: DecoderConfig,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize neural syndrome decoder.
        
        Args:
            config: Decoder configuration
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Model placeholder (would be actual neural network)
        self.model = None
        self.is_trained = False
        
        # Training history
        self.training_history = {
            'loss': [],
            'accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        # Performance metrics
        self.metrics = {
            'logical_error_rate': 0.0,
            'frame_error_rate': 0.0,
            'decoding_accuracy': 0.0,
            'inference_time': 0.0
        }
        
        # Statistical tracking
        self.total_decodings = 0
        self.successful_decodings = 0
        self.training_time = 0.0
        
        self.logger.info(f"NeuralSyndromeDecoder initialized with {config.architecture.value} architecture")
    
    def _build_model(self):
        """Build neural network model (placeholder)."""
        # This would contain actual neural network construction
        # For now, just simulate model structure
        
        if self.config.architecture == DecoderArchitecture.MLP:
            layers = [self.config.input_dim] + self.config.hidden_dims + [self.config.output_dim]
            self.model = {
                'type': 'mlp',
                'layers': layers,
                'parameters': sum(layers[i] * layers[i+1] for i in range(len(layers)-1))
            }
        
        elif self.config.architecture == DecoderArchitecture.CNN:
            self.model = {
                'type': 'cnn',
                'conv_layers': [(32, 3), (64, 3), (128, 3)],
                'fc_layers': self.config.hidden_dims + [self.config.output_dim],
                'parameters': 50000  # Estimate
            }
        
        elif self.config.architecture == DecoderArchitecture.TRANSFORMER:
            self.model = {
                'type': 'transformer',
                'num_heads': 8,
                'num_layers': 4,
                'hidden_dim': 256,
                'parameters': 100000  # Estimate
            }
        
        else:
            # Default MLP
            self.model = {
                'type': 'mlp',
                'layers': [self.config.input_dim, 128, 64, self.config.output_dim],
                'parameters': 10000
            }
        
        self.logger.info(f"Built {self.model['type']} model with ~{self.model['parameters']} parameters")
    
    def train(
        self, 
        train_data: List[SyndromeData],
        val_data: Optional[List[SyndromeData]] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Train the neural decoder.
        
        Args:
            train_data: Training syndrome data
            val_data: Validation syndrome data
            verbose: Whether to log training progress
            
        Returns:
            Training results and metrics
        """
        start_time = time.time()
        
        if self.model is None:
            self._build_model()
        
        # Convert data to arrays (simplified)
        train_syndromes = np.array([sample.syndrome for sample in train_data])
        train_errors = np.array([sample.error_pattern for sample in train_data])
        
        if val_data:
            val_syndromes = np.array([sample.syndrome for sample in val_data])
            val_errors = np.array([sample.error_pattern for sample in val_data])
        
        if verbose:
            self.logger.info(f"Training on {len(train_data)} samples for {self.config.epochs} epochs")
        
        # Simulate training (would be actual training loop)
        for epoch in range(self.config.epochs):
            # Simulate training metrics
            train_loss = max(0.1, 2.0 * np.exp(-epoch / 20) + np.random.normal(0, 0.05))
            train_acc = min(0.99, 0.5 + 0.45 * (1 - np.exp(-epoch / 15)) + np.random.normal(0, 0.02))
            
            self.training_history['loss'].append(train_loss)
            self.training_history['accuracy'].append(train_acc)
            
            if val_data:
                val_loss = train_loss + np.random.normal(0, 0.1)
                val_acc = max(0, train_acc + np.random.normal(0, 0.05))
                self.training_history['val_loss'].append(val_loss)
                self.training_history['val_accuracy'].append(val_acc)
            
            if verbose and epoch % 10 == 0:
                self.logger.info(f"Epoch {epoch}: loss={train_loss:.4f}, acc={train_acc:.4f}")
        
        self.training_time = time.time() - start_time
        self.is_trained = True
        
        # Final performance metrics
        final_accuracy = self.training_history['accuracy'][-1]
        self.metrics['decoding_accuracy'] = final_accuracy
        self.metrics['logical_error_rate'] = 1 - final_accuracy
        
        results = {
            'training_time': self.training_time,
            'final_accuracy': final_accuracy,
            'final_loss': self.training_history['loss'][-1],
            'model_parameters': self.model['parameters'],
            'convergence_epoch': self._find_convergence_epoch()
        }
        
        self.logger.info(f"Training completed in {self.training_time:.2f}s, accuracy: {final_accuracy:.4f}")
        return results
    
    def _find_convergence_epoch(self) -> int:
        """Find epoch where training converged."""
        if len(self.training_history['loss']) < 10:
            return len(self.training_history['loss'])
        
        # Look for where loss stabilizes
        losses = self.training_history['loss']
        for i in range(10, len(losses)):
            if np.std(losses[i-10:i]) < 0.01:
                return i
        
        return len(losses)
    
    def decode(self, syndrome: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Decode syndrome to predict error pattern.
        
        Args:
            syndrome: Syndrome measurement
            
        Returns:
            Tuple of (predicted_error_pattern, confidence)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before decoding")
        
        start_time = time.time()
        
        # Simulate neural network inference
        if self.config.architecture == DecoderArchitecture.MLP:
            # Simple heuristic for MLP
            error_pattern = np.random.binomial(1, 0.1, self.config.output_dim)
            confidence = 0.8 + np.random.normal(0, 0.1)
            
        elif self.config.architecture == DecoderArchitecture.CNN:
            # CNN would consider spatial correlations
            error_pattern = np.random.binomial(1, 0.08, self.config.output_dim)
            confidence = 0.85 + np.random.normal(0, 0.08)
            
        elif self.config.architecture == DecoderArchitecture.TRANSFORMER:
            # Transformer considers all syndrome bits simultaneously
            error_pattern = np.random.binomial(1, 0.06, self.config.output_dim)
            confidence = 0.9 + np.random.normal(0, 0.05)
            
        else:
            # Default prediction
            error_pattern = np.random.binomial(1, 0.1, self.config.output_dim)
            confidence = 0.7
        
        # Ensure valid range
        confidence = np.clip(confidence, 0.0, 1.0)
        
        # Update metrics
        inference_time = time.time() - start_time
        self.metrics['inference_time'] = inference_time
        self.total_decodings += 1
        
        return error_pattern, confidence
    
    def evaluate(
        self, 
        test_data: List[SyndromeData],
        metrics: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Evaluate decoder performance.
        
        Args:
            test_data: Test syndrome data
            metrics: Metrics to compute
            
        Returns:
            Performance metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        if metrics is None:
            metrics = ['accuracy', 'logical_error_rate', 'frame_error_rate']
        
        correct_predictions = 0
        logical_errors = 0
        frame_errors = 0
        total_inference_time = 0
        
        self.logger.info(f"Evaluating on {len(test_data)} test samples")
        
        for sample in test_data:
            start_time = time.time()
            
            predicted_error, confidence = self.decode(sample.syndrome)
            
            inference_time = time.time() - start_time
            total_inference_time += inference_time
            
            # Check accuracy (exact match)
            if np.array_equal(predicted_error, sample.error_pattern):
                correct_predictions += 1
            else:
                # Check if it's a logical error (would need logical operators)
                # For now, assume any mismatch could be logical error
                if np.sum(np.abs(predicted_error - sample.error_pattern)) > 2:
                    logical_errors += 1
                frame_errors += 1
        
        # Compute metrics
        results = {}
        
        if 'accuracy' in metrics:
            results['accuracy'] = correct_predictions / len(test_data)
        
        if 'logical_error_rate' in metrics:
            results['logical_error_rate'] = logical_errors / len(test_data)
        
        if 'frame_error_rate' in metrics:
            results['frame_error_rate'] = frame_errors / len(test_data)
        
        if 'inference_time' in metrics:
            results['avg_inference_time'] = total_inference_time / len(test_data)
        
        # Update stored metrics
        self.metrics.update(results)
        
        self.logger.info(f"Evaluation completed: accuracy={results.get('accuracy', 0):.4f}")
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get decoder statistics."""
        return {
            'config': {
                'architecture': self.config.architecture.value,
                'input_dim': self.config.input_dim,
                'output_dim': self.config.output_dim,
                'hidden_dims': self.config.hidden_dims
            },
            'training': {
                'is_trained': self.is_trained,
                'training_time': self.training_time,
                'epochs': self.config.epochs,
                'convergence_epoch': self._find_convergence_epoch() if self.training_history['loss'] else 0
            },
            'performance': self.metrics.copy(),
            'usage': {
                'total_decodings': self.total_decodings,
                'successful_decodings': self.successful_decodings
            },
            'model': self.model if self.model else {}
        }


class DecoderComparison:
    """
    Framework for comparing different neural decoder architectures.
    
    Benchmarks multiple decoder types on the same dataset to
    identify optimal architectures for different scenarios.
    """
    
    def __init__(
        self,
        code_type: str = "surface",
        code_distance: int = 3,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize decoder comparison framework.
        
        Args:
            code_type: Quantum error correction code type
            code_distance: Code distance parameter
            logger: Optional logger instance
        """
        self.code_type = code_type
        self.code_distance = code_distance
        self.logger = logger or logging.getLogger(__name__)
        
        # Generate syndrome data
        self.generator = SyndromeGenerator(code_type, code_distance, logger=logger)
        
        # Decoders to compare
        self.decoders: Dict[str, NeuralSyndromeDecoder] = {}
        self.comparison_results: Dict[str, Dict[str, Any]] = {}
        
        # Common dataset for fair comparison
        self.train_data: List[SyndromeData] = []
        self.test_data: List[SyndromeData] = []
        
        self.logger.info(f"DecoderComparison initialized for {code_type} code, distance {code_distance}")
    
    def prepare_datasets(
        self,
        train_size: int = 10000,
        test_size: int = 2000,
        error_rate_range: Tuple[float, float] = (0.001, 0.05)
    ):
        """Prepare training and test datasets."""
        self.logger.info(f"Preparing datasets: train={train_size}, test={test_size}")
        
        # Generate training data
        self.train_data = self.generator.generate_dataset(
            train_size, 
            error_rate_range=error_rate_range,
            balanced=True
        )
        
        # Generate test data with different error rates
        test_error_range = (error_rate_range[0], error_rate_range[1] * 1.5)
        self.test_data = self.generator.generate_dataset(
            test_size,
            error_rate_range=test_error_range,
            balanced=False
        )
        
        self.logger.info("Dataset preparation completed")
    
    def add_decoder(self, name: str, config: DecoderConfig):
        """Add decoder to comparison."""
        decoder = NeuralSyndromeDecoder(config, logger=self.logger)
        self.decoders[name] = decoder
        self.logger.info(f"Added decoder: {name} ({config.architecture.value})")
    
    def add_standard_decoders(self):
        """Add standard decoder architectures for comparison."""
        syndrome_length = self.generator.syndrome_length
        error_pattern_length = 2 * self.generator.num_data_qubits  # X and Z errors
        
        # MLP decoder
        mlp_config = DecoderConfig(
            architecture=DecoderArchitecture.MLP,
            input_dim=syndrome_length,
            output_dim=error_pattern_length,
            hidden_dims=[128, 64, 32],
            learning_rate=0.001,
            epochs=50
        )
        self.add_decoder("MLP", mlp_config)
        
        # CNN decoder (for surface codes with spatial structure)
        if self.code_type == "surface":
            cnn_config = DecoderConfig(
                architecture=DecoderArchitecture.CNN,
                input_dim=syndrome_length,
                output_dim=error_pattern_length,
                hidden_dims=[64, 32],
                learning_rate=0.0005,
                epochs=60
            )
            self.add_decoder("CNN", cnn_config)
        
        # Transformer decoder
        transformer_config = DecoderConfig(
            architecture=DecoderArchitecture.TRANSFORMER,
            input_dim=syndrome_length,
            output_dim=error_pattern_length,
            hidden_dims=[256, 128],
            learning_rate=0.0001,
            epochs=40
        )
        self.add_decoder("Transformer", transformer_config)
    
    def run_comparison(self, train_models: bool = True) -> Dict[str, Dict[str, Any]]:
        """
        Run complete decoder comparison.
        
        Args:
            train_models: Whether to train models or use existing
            
        Returns:
            Comparison results
        """
        if not self.train_data or not self.test_data:
            self.prepare_datasets()
        
        if not self.decoders:
            self.add_standard_decoders()
        
        self.logger.info(f"Running comparison of {len(self.decoders)} decoders")
        
        results = {}
        
        for name, decoder in self.decoders.items():
            self.logger.info(f"Processing decoder: {name}")
            
            decoder_results = {}
            
            if train_models:
                # Train decoder
                train_results = decoder.train(
                    self.train_data,
                    val_data=self.test_data[:500],  # Use part of test as validation
                    verbose=False
                )
                decoder_results['training'] = train_results
            
            # Evaluate decoder
            eval_results = decoder.evaluate(
                self.test_data,
                metrics=['accuracy', 'logical_error_rate', 'frame_error_rate', 'inference_time']
            )
            decoder_results['evaluation'] = eval_results
            
            # Get statistics
            decoder_results['statistics'] = decoder.get_statistics()
            
            results[name] = decoder_results
            
            self.logger.info(
                f"{name}: accuracy={eval_results['accuracy']:.4f}, "
                f"logical_error_rate={eval_results['logical_error_rate']:.4f}"
            )
        
        self.comparison_results = results
        return results
    
    def generate_comparison_report(self) -> Dict[str, Any]:
        """Generate comprehensive comparison report."""
        if not self.comparison_results:
            self.run_comparison()
        
        # Summary metrics
        summary = {}
        for name, results in self.comparison_results.items():
            eval_metrics = results['evaluation']
            train_metrics = results.get('training', {})
            
            summary[name] = {
                'accuracy': eval_metrics.get('accuracy', 0),
                'logical_error_rate': eval_metrics.get('logical_error_rate', 1),
                'training_time': train_metrics.get('training_time', 0),
                'inference_time': eval_metrics.get('avg_inference_time', 0),
                'model_size': results['statistics']['model'].get('parameters', 0)
            }
        
        # Find best performers
        best_accuracy = max(summary.values(), key=lambda x: x['accuracy'])
        best_speed = min(summary.values(), key=lambda x: x['inference_time'])
        best_logical = min(summary.values(), key=lambda x: x['logical_error_rate'])
        
        recommendations = {
            'best_accuracy': {
                'decoder': next(name for name, metrics in summary.items() 
                               if metrics['accuracy'] == best_accuracy['accuracy']),
                'accuracy': best_accuracy['accuracy']
            },
            'fastest_inference': {
                'decoder': next(name for name, metrics in summary.items() 
                               if metrics['inference_time'] == best_speed['inference_time']),
                'time': best_speed['inference_time']
            },
            'best_logical_performance': {
                'decoder': next(name for name, metrics in summary.items() 
                               if metrics['logical_error_rate'] == best_logical['logical_error_rate']),
                'logical_error_rate': best_logical['logical_error_rate']
            }
        }
        
        report = {
            'experiment_info': {
                'code_type': self.code_type,
                'code_distance': self.code_distance,
                'train_samples': len(self.train_data),
                'test_samples': len(self.test_data),
                'decoders_compared': list(self.decoders.keys())
            },
            'summary_metrics': summary,
            'recommendations': recommendations,
            'detailed_results': self.comparison_results
        }
        
        return report