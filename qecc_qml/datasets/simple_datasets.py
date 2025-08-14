"""
Simple quantum datasets for training and testing.
Generation 1: Basic synthetic datasets with minimal dependencies.
"""

import numpy as np
from typing import Tuple, Optional


class SimpleQuantumDatasets:
    """
    Generator for simple quantum machine learning datasets.
    """
    
    @staticmethod
    def generate_classification_data(
        n_samples: int = 1000,
        n_features: int = 4,
        n_classes: int = 2,
        noise: float = 0.1,
        random_state: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic classification data suitable for quantum ML.
        
        Args:
            n_samples: Number of samples to generate
            n_features: Number of features (should match quantum circuit qubits)
            n_classes: Number of classes (currently supports 2)
            noise: Amount of noise to add to the data
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (X, y) where X is features and y is labels
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        # Generate linearly separable data with quantum-inspired structure
        X = np.random.randn(n_samples, n_features)
        
        # Create quantum-inspired patterns
        # Features represent quantum state amplitudes (normalized)
        for i in range(n_samples):
            norm = np.linalg.norm(X[i])
            if norm > 0:
                X[i] = X[i] / norm  # Normalize like quantum state amplitudes
        
        # Create labels based on quantum-inspired decision boundary
        # Use a combination of features that mimics quantum interference patterns
        decision_boundary = np.zeros(n_samples)
        
        for i in range(n_samples):
            # Quantum-inspired decision rule: based on phases and amplitudes
            phase_factor = np.sum(X[i] * np.array([1, -1, 1, -1][:n_features]))
            amplitude_factor = np.sum(X[i] ** 2)
            
            decision_boundary[i] = phase_factor + 0.5 * amplitude_factor
        
        y = (decision_boundary > np.median(decision_boundary)).astype(int)
        
        # Add noise
        if noise > 0:
            n_flip = int(noise * n_samples)
            flip_indices = np.random.choice(n_samples, n_flip, replace=False)
            y[flip_indices] = 1 - y[flip_indices]
        
        return X.astype(np.float32), y.astype(int)
    
    @staticmethod
    def generate_entangled_data(
        n_samples: int = 500,
        n_qubits: int = 4,
        entanglement_strength: float = 0.8,
        random_state: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate data with quantum entanglement-like correlations.
        
        Args:
            n_samples: Number of samples
            n_qubits: Number of qubits (features)
            entanglement_strength: Strength of correlations between features
            random_state: Random seed
            
        Returns:
            Tuple of (X, y) with entangled features
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        X = np.zeros((n_samples, n_qubits))
        y = np.zeros(n_samples, dtype=int)
        
        for i in range(n_samples):
            # Generate base random state
            base_state = np.random.randn(n_qubits)
            
            # Create entanglement-like correlations
            if n_qubits >= 2:
                # Pair-wise correlations (like Bell states)
                for j in range(0, n_qubits - 1, 2):
                    if np.random.random() < entanglement_strength:
                        # Strong correlation between qubit pairs
                        correlation = np.random.choice([-1, 1])
                        base_state[j+1] = correlation * base_state[j] + 0.1 * np.random.randn()
            
            # Normalize
            norm = np.linalg.norm(base_state)
            if norm > 0:
                X[i] = base_state / norm
            
            # Label based on entanglement pattern
            # High entanglement -> class 1, low entanglement -> class 0
            entanglement_measure = np.sum(np.abs(np.correlate(X[i][::2],  X[i][1::2], mode='valid')))
            y[i] = 1 if entanglement_measure > np.median([
                np.sum(np.abs(np.correlate(X[j][::2],  X[j][1::2], mode='valid')))
                for j in range(min(i+1, 100))  # Use running median
            ]) else 0
        
        return X.astype(np.float32), y.astype(int)
    
    @staticmethod
    def quantum_mnist_subset(
        n_samples: int = 200,
        classes: Tuple[int, int] = (0, 1),
        feature_reduction: str = 'pca',
        n_features: int = 4,
        random_state: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a quantum-inspired MNIST subset.
        
        Since we don't have sklearn, we'll create synthetic data that
        mimics the characteristics of reduced MNIST data.
        
        Args:
            n_samples: Number of samples per class
            classes: Which classes to include (e.g., (0, 1) for digits 0 and 1)
            feature_reduction: Type of reduction ('pca', 'amplitude')
            n_features: Number of features after reduction
            random_state: Random seed
            
        Returns:
            Reduced dataset suitable for quantum ML
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        total_samples = n_samples * len(classes)
        X = np.zeros((total_samples, n_features))
        y = np.zeros(total_samples, dtype=int)
        
        for class_idx, class_label in enumerate(classes):
            start_idx = class_idx * n_samples
            end_idx = start_idx + n_samples
            
            # Generate synthetic data that mimics digit characteristics
            if class_label == 0:
                # "Digit 0" - circular pattern
                angles = np.linspace(0, 2*np.pi, n_samples)
                for i, angle in enumerate(angles):
                    X[start_idx + i] = [
                        np.cos(angle) + 0.1 * np.random.randn(),
                        np.sin(angle) + 0.1 * np.random.randn(),
                        0.5 + 0.1 * np.random.randn(),
                        0.2 * np.random.randn()
                    ][:n_features]
                    
            elif class_label == 1:
                # "Digit 1" - linear pattern
                positions = np.linspace(-1, 1, n_samples)
                for i, pos in enumerate(positions):
                    X[start_idx + i] = [
                        pos + 0.1 * np.random.randn(),
                        0.8 * pos + 0.1 * np.random.randn(),
                        0.2 + 0.1 * np.random.randn(),
                        0.1 * np.random.randn()
                    ][:n_features]
            
            y[start_idx:end_idx] = class_label
        
        # Normalize features to quantum state amplitudes
        for i in range(total_samples):
            norm = np.linalg.norm(X[i])
            if norm > 0:
                X[i] = X[i] / norm
        
        # Shuffle data
        shuffle_indices = np.random.permutation(total_samples)
        X = X[shuffle_indices]
        y = y[shuffle_indices]
        
        return X.astype(np.float32), y.astype(int)
    
    @staticmethod
    def get_iris_quantum(
        n_features: int = 4,
        classes: Tuple[int, int] = (0, 1),
        random_state: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate quantum-inspired Iris dataset.
        
        Args:
            n_features: Number of features (max 4)
            classes: Which classes to include
            random_state: Random seed
            
        Returns:
            Quantum-ready Iris-like dataset
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        # Iris-inspired data generation
        n_samples_per_class = 50
        total_samples = n_samples_per_class * len(classes)
        
        X = np.zeros((total_samples, n_features))
        y = np.zeros(total_samples, dtype=int)
        
        # Class characteristics (inspired by real Iris dataset)
        class_params = {
            0: {'sepal_length': (5.0, 0.4), 'sepal_width': (3.4, 0.4), 
                'petal_length': (1.5, 0.2), 'petal_width': (0.2, 0.1)},
            1: {'sepal_length': (6.0, 0.5), 'sepal_width': (2.8, 0.3),
                'petal_length': (4.3, 0.5), 'petal_width': (1.3, 0.2)},
            2: {'sepal_length': (6.6, 0.6), 'sepal_width': (3.0, 0.3),
                'petal_length': (5.6, 0.6), 'petal_width': (2.0, 0.3)}
        }
        
        feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        
        for class_idx, class_label in enumerate(classes):
            start_idx = class_idx * n_samples_per_class
            end_idx = start_idx + n_samples_per_class
            
            params = class_params[class_label]
            
            for i in range(n_samples_per_class):
                for j in range(n_features):
                    feature_name = feature_names[j]
                    mean, std = params[feature_name]
                    X[start_idx + i, j] = np.random.normal(mean, std)
            
            y[start_idx:end_idx] = class_label
        
        # Normalize for quantum processing
        for i in range(total_samples):
            # Min-max normalize to [0,1] then to quantum amplitudes
            X[i] = (X[i] - X[i].min()) / (X[i].max() - X[i].min() + 1e-8)
            norm = np.linalg.norm(X[i])
            if norm > 0:
                X[i] = X[i] / norm
        
        # Shuffle
        shuffle_indices = np.random.permutation(total_samples)
        X = X[shuffle_indices]
        y = y[shuffle_indices]
        
        return X.astype(np.float32), y.astype(int)


# Convenience functions
def load_quantum_classification(dataset: str = 'synthetic', **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a quantum classification dataset.
    
    Args:
        dataset: Dataset name ('synthetic', 'entangled', 'mnist', 'iris')
        **kwargs: Additional parameters for dataset generation
        
    Returns:
        Tuple of (X, y)
    """
    if dataset == 'synthetic':
        return SimpleQuantumDatasets.generate_classification_data(**kwargs)
    elif dataset == 'entangled':
        return SimpleQuantumDatasets.generate_entangled_data(**kwargs)
    elif dataset == 'mnist':
        return SimpleQuantumDatasets.quantum_mnist_subset(**kwargs)
    elif dataset == 'iris':
        return SimpleQuantumDatasets.get_iris_quantum(**kwargs)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")