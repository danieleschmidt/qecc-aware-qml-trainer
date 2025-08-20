"""
QECC-QML Research Module.

Advanced research implementations for quantum error correction
aware quantum machine learning systems, including cutting-edge
co-evolution algorithms and quantum-classical optimization.
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

# Advanced Co-evolution Algorithms
from .autonomous_quantum_circuit_evolution import (
    AutonomousQuantumCircuitEvolution,
    CircuitGenome,
    EvolutionMetrics,
    QuantumMachineLearningFitnessEvaluator,
    run_autonomous_evolution_research
)

from .quantum_advantage_prover import (
    QuantumAdvantageProver,
    QuantumComplexityAnalyzer,
    ExperimentalValidation,
    AdvantageProof,
    ComplexityBounds,
    PerformanceMetrics,
    run_quantum_advantage_research
)

from .federated_quantum_learning import (
    FederatedQuantumLearningOrchestrator,
    QuantumNode,
    FederatedModel,
    FederatedUpdate,
    DifferentialPrivacyProtocol,
    QuantumSecureMultipartyComputation,
    ConsensusProtocol,
    run_federated_quantum_learning_research
)

from .quantum_classical_coevolution import (
    QuantumClassicalCoevolution,
    CoevolutionStrategy,
    QuantumComponent,
    ClassicalComponent,
    HybridInterface,
    CoevolutionGenome,
    QECCQMLCoevolutionEvaluator,
    run_quantum_classical_coevolution_research
)

from .coevolutionary_optimizer import (
    CoevolutionaryOptimizer,
    OptimizationStrategy,
    CoevolutionaryIndividual,
    OptimizationTarget,
    QECCQMLHybridObjective,
    run_coevolutionary_optimization_research
)

from .adaptive_neural_architecture_search import (
    AdaptiveNeuralArchitectureSearch,
    SearchStrategy,
    ArchitectureGene,
    SearchSpace,
    QuantumAwareArchitectureEvaluator,
    LayerType,
    ActivationFunction,
    run_adaptive_neural_architecture_search_research
)

from .hybrid_evolution_engine import (
    HybridEvolutionEngine,
    HybridStrategy,
    HybridIndividual,
    EvolutionPhase,
    QECCQMLHybridObjective,
    run_hybrid_evolution_engine_research
)

from .coevolution_benchmarks import (
    CoevolutionBenchmarkFramework,
    CoevolutionBenchmarkSuite,
    BenchmarkType as CoevolutionBenchmarkType,
    MetricType,
    BenchmarkConfig as CoevolutionBenchmarkConfig,
    BenchmarkResult as CoevolutionBenchmarkResult,
    ComparativeAnalysis,
    run_comprehensive_coevolution_benchmarks
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
    'BenchmarkResult',
    
    # Autonomous Circuit Evolution
    'AutonomousQuantumCircuitEvolution',
    'CircuitGenome',
    'EvolutionMetrics',
    'QuantumMachineLearningFitnessEvaluator',
    'run_autonomous_evolution_research',
    
    # Quantum Advantage Prover
    'QuantumAdvantageProver',
    'QuantumComplexityAnalyzer',
    'ExperimentalValidation',
    'AdvantageProof',
    'ComplexityBounds',
    'PerformanceMetrics',
    'run_quantum_advantage_research',
    
    # Federated Quantum Learning
    'FederatedQuantumLearningOrchestrator',
    'QuantumNode',
    'FederatedModel',
    'FederatedUpdate',
    'DifferentialPrivacyProtocol',
    'QuantumSecureMultipartyComputation',
    'ConsensusProtocol',
    'run_federated_quantum_learning_research',
    
    # Quantum-Classical Co-evolution
    'QuantumClassicalCoevolution',
    'CoevolutionStrategy',
    'QuantumComponent',
    'ClassicalComponent',
    'HybridInterface',
    'CoevolutionGenome',
    'QECCQMLCoevolutionEvaluator',
    'run_quantum_classical_coevolution_research',
    
    # Co-evolutionary Optimizer
    'CoevolutionaryOptimizer',
    'OptimizationStrategy',
    'CoevolutionaryIndividual',
    'OptimizationTarget',
    'QECCQMLHybridObjective',
    'run_coevolutionary_optimization_research',
    
    # Adaptive Neural Architecture Search
    'AdaptiveNeuralArchitectureSearch',
    'SearchStrategy',
    'ArchitectureGene',
    'SearchSpace',
    'QuantumAwareArchitectureEvaluator',
    'LayerType',
    'ActivationFunction',
    'run_adaptive_neural_architecture_search_research',
    
    # Hybrid Evolution Engine
    'HybridEvolutionEngine',
    'HybridStrategy',
    'HybridIndividual',
    'EvolutionPhase',
    'QECCQMLHybridObjective',
    'run_hybrid_evolution_engine_research',
    
    # Co-evolution Benchmarks
    'CoevolutionBenchmarkFramework',
    'CoevolutionBenchmarkSuite',
    'CoevolutionBenchmarkType',
    'MetricType',
    'CoevolutionBenchmarkConfig',
    'CoevolutionBenchmarkResult',
    'ComparativeAnalysis',
    'run_comprehensive_coevolution_benchmarks'
]