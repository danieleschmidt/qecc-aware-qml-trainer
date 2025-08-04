"""
QECC-Aware Quantum Machine Learning Trainer

A quantum-classical framework that seamlessly integrates error correction codes (QECC) 
into Quantum Machine Learning (QML) circuits with real-time fidelity tracking and 
noise-aware training.
"""

__version__ = "0.1.0"
__author__ = "Terragon Labs"
__email__ = "contact@terragonlabs.com"

from .core.quantum_nn import QECCAwareQNN
from .core.error_correction import ErrorCorrectionScheme
from .core.noise_models import NoiseModel
from .training.qecc_trainer import QECCTrainer
from .codes.surface_code import SurfaceCode
from .codes.color_code import ColorCode
from .evaluation.benchmarks import NoiseBenchmark

__all__ = [
    "QECCAwareQNN",
    "ErrorCorrectionScheme", 
    "NoiseModel",
    "QECCTrainer",
    "SurfaceCode",
    "ColorCode",
    "NoiseBenchmark",
]