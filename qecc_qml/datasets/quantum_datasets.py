"""
Quantum datasets for machine learning experiments.
"""

from typing import Tuple
import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler


class QuantumDatasets:
    """
    Collection of datasets suitable for quantum machine learning experiments.
    """
    
    def __init__(self):
        self.scaler_cache = {}
    
    def load_quantum_mnist(
        self,
        subset_size: int = 1000,
        num_classes: int = 2,
        target_dim: int = 4,
        normalize: bool = True,
        random_state: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Load quantum-friendly MNIST-like dataset."""
        X, y = make_classification(
            n_samples=subset_size,
            n_features=target_dim,
            n_informative=target_dim,
            n_redundant=0,
            n_classes=num_classes,
            class_sep=1.5,
            random_state=random_state
        )
        
        if normalize:
            scaler = MinMaxScaler(feature_range=(0, np.pi))
            X = scaler.fit_transform(X)
            self.scaler_cache['quantum_mnist'] = scaler
        
        return X, y