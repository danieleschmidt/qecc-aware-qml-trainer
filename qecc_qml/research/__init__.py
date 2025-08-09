"""
QECC-QML Research Module.

Advanced research implementations for quantum error correction
aware quantum machine learning systems.
"""

from .reinforcement_learning_qecc import (
    QECCEnvironment,
    QECCRLAgent, 
    create_rl_qecc_trainer,
    RLAction,
    RLState,
    QECCAction,
    EnvironmentState
)

from .neural_syndrome_decoders import (
    NeuralSyndromeDecoder,
    SyndromeGenerator,
    DecoderComparison,
    DecoderArchitecture,
    DecoderConfig,
    SyndromeData,
    ErrorModel
)

from .quantum_advantage_benchmarks import (
    QuantumAdvantageSuite,
    LearningEfficiencyBenchmark,
    ErrorCorrectionThresholdBenchmark,
    ScalingAdvantageBenchmark,
    BenchmarkType,
    QuantumAdvantageMetric,
    BenchmarkConfig,
    BenchmarkResult
)

__all__ = [
    # RL for QECC
    'QECCEnvironment',
    'QECCRLAgent',
    'create_rl_qecc_trainer',
    'RLAction',
    'RLState', 
    'QECCAction',
    'EnvironmentState',
    
    # Neural Decoders
    'NeuralSyndromeDecoder',
    'SyndromeGenerator',
    'DecoderComparison',
    'DecoderArchitecture',
    'DecoderConfig',
    'SyndromeData',
    'ErrorModel',
    
    # Quantum Advantage
    'QuantumAdvantageSuite',
    'LearningEfficiencyBenchmark',
    'ErrorCorrectionThresholdBenchmark',
    'ScalingAdvantageBenchmark',
    'BenchmarkType',
    'QuantumAdvantageMetric',
    'BenchmarkConfig',
    'BenchmarkResult'
]