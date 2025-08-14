"""
Quantum datasets for machine learning experiments.
"""

# Import essential modules first
from .simple_datasets import SimpleQuantumDatasets, load_quantum_classification

# Try to import optional modules
try:
    from .quantum_datasets import QuantumDatasets
    quantum_datasets = QuantumDatasets()
    QUANTUM_DATASETS_AVAILABLE = True
except ImportError:
    QUANTUM_DATASETS_AVAILABLE = False

try:
    from .quantum_datasets_enhanced import EnhancedQuantumDatasets
    enhanced_quantum_datasets = EnhancedQuantumDatasets()
    ENHANCED_DATASETS_AVAILABLE = True
except ImportError:
    ENHANCED_DATASETS_AVAILABLE = False

# Build __all__ based on what's available
__all__ = ["SimpleQuantumDatasets", "load_quantum_classification"]

if QUANTUM_DATASETS_AVAILABLE:
    __all__.extend(["QuantumDatasets", "quantum_datasets"])

if ENHANCED_DATASETS_AVAILABLE:
    __all__.extend(["EnhancedQuantumDatasets", "enhanced_quantum_datasets"])