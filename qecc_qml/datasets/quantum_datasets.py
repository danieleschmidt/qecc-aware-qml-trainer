"""
Quantum datasets for machine learning experiments.
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
from typing import Tuple
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