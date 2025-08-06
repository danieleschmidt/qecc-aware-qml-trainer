"""
Quantum datasets for machine learning experiments.
"""

from .quantum_datasets import QuantumDatasets

# Convenience instance
quantum_datasets = QuantumDatasets()

__all__ = [
    "QuantumDatasets",
    "quantum_datasets",
]