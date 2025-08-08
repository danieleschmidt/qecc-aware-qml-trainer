"""
Quantum datasets for machine learning experiments.
"""

from .quantum_datasets import QuantumDatasets
from .quantum_datasets_enhanced import EnhancedQuantumDatasets

# Convenience instances
quantum_datasets = QuantumDatasets()
enhanced_quantum_datasets = EnhancedQuantumDatasets()

__all__ = [
    "QuantumDatasets",
    "EnhancedQuantumDatasets",
    "quantum_datasets",
    "enhanced_quantum_datasets",
]