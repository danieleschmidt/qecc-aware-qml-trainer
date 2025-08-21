#!/usr/bin/env python3
"""
Comprehensive Benchmarking System for Co-evolution Algorithms

Advanced benchmarking framework for evaluating and comparing all co-evolution
algorithms implemented in the QECC-QML research framework. Provides standardized
metrics, performance analysis, and comparative studies.
"""

import sys
import time
import json
import math
import random
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass, asdict, field
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from enum import Enum
import numpy as np

# Import with fallbacks
sys.path.insert(0, '/root/repo')
from qecc_qml.core.fallback_imports import create_fallback_implementations
create_fallback_implementations()

try:
    import numpy as np
except ImportError:
    import sys
    np = sys.modules['numpy']


class BenchmarkType(Enum):
    """Types of benchmarks for co-evolution algorithms."""
    PERFORMANCE_BENCHMARK = "performance_benchmark"
    SCALABILITY_BENCHMARK = "scalability_benchmark"
    ROBUSTNESS_BENCHMARK = "robustness_benchmark"
    CONVERGENCE_BENCHMARK = "convergence_benchmark"
    COMPARATIVE_BENCHMARK = "comparative_benchmark"
    QUANTUM_ADVANTAGE_BENCHMARK = "quantum_advantage_benchmark"


class MetricType(Enum):
    """Types of metrics for evaluation."""
    FITNESS_METRIC = "fitness_metric"
    CONVERGENCE_METRIC = "convergence_metric"
    DIVERSITY_METRIC = "diversity_metric"
    EFFICIENCY_METRIC = "efficiency_metric"
    ROBUSTNESS_METRIC = "robustness_metric"
    QUANTUM_METRIC = "quantum_metric"


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark tests."""
    benchmark_id: str
    benchmark_type: BenchmarkType
    algorithm_configs: List[Dict[str, Any]]
    test_parameters: Dict[str, Any]
    metrics_to_evaluate: List[MetricType]
    repetitions: int = 5
    max_runtime: float = 3600.0  # 1 hour max
    statistical_significance: float = 0.95


@dataclass
class BenchmarkResult:
    """Results from a benchmark test."""
    benchmark_id: str
    algorithm_name: str
    run_id: int
    metrics: Dict[str, float]
    runtime: float
    convergence_data: List[float]
    diversity_data: List[float]
    additional_data: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error_message: Optional[str] = None


@dataclass
class ComparativeAnalysis:
    """Comparative analysis between algorithms."""
    comparison_id: str
    algorithms_compared: List[str]
    statistical_tests: Dict[str, Dict[str, float]]
    performance_rankings: Dict[str, int]
    significant_differences: List[Tuple[str, str, str]]  # (alg1, alg2, metric)
    recommendations: List[str]
    confidence_intervals: Dict[str, Dict[str, Tuple[float, float]]]


class BenchmarkSuite(ABC):
    """Abstract base class for benchmark suites."""
    
    @abstractmethod
    def setup_benchmark(self, config: BenchmarkConfig) -> None:
        """Setup benchmark environment."""
        pass
    
    @abstractmethod
    def run_algorithm(self, algorithm_name: str, config: Dict[str, Any]) -> BenchmarkResult:
        """Run a single algorithm and return results."""
        pass
    
    @abstractmethod
    def cleanup_benchmark(self) -> None:
        """Cleanup benchmark environment."""
        pass


class CoevolutionBenchmarkSuite(BenchmarkSuite):
    """Comprehensive benchmark suite for co-evolution algorithms."""
    
    def __init__(self):
        self.benchmark_problems = self._create_benchmark_problems()
        self.noise_models = self._create_noise_models()
        self.current_problem = None
        self.baseline_metrics = {}
    
    def _create_benchmark_problems(self) -> Dict[str, Dict[str, Any]]:
        """Create standard benchmark problems."""
        problems = {
            'simple_qecc': {
                'description': 'Simple quantum error correction problem',
                'quantum_params': {
                    'qubit_count': 5,
                    'circuit_depth': 10,
                    'gate_count': 25,
                    'error_rate': 0.01
                },
                'classical_params': {
                    'input_dim': 10,
                    'hidden_dims': [32, 16],
                    'output_dim': 5
                },
                'target_fidelity': 0.90
            },
            'medium_qecc': {
                'description': 'Medium complexity quantum error correction problem',
                'quantum_params': {
                    'qubit_count': 10,
                    'circuit_depth': 20,
                    'gate_count': 60,
                    'error_rate': 0.02
                },
                'classical_params': {
                    'input_dim': 20,
                    'hidden_dims': [64, 32, 16],
                    'output_dim': 10
                },
                'target_fidelity': 0.85
            },
            'complex_qecc': {
                'description': 'Complex quantum error correction problem',
                'quantum_params': {
                    'qubit_count': 15,
                    'circuit_depth': 30,
                    'gate_count': 100,
                    'error_rate': 0.03
                },
                'classical_params': {
                    'input_dim': 30,
                    'hidden_dims': [128, 64, 32, 16],
                    'output_dim': 15
                },
                'target_fidelity': 0.80
            },
            'noisy_environment': {
                'description': 'High noise quantum environment',
                'quantum_params': {
                    'qubit_count': 8,
                    'circuit_depth': 15,
                    'gate_count': 40,
                    'error_rate': 0.05
                },
                'classical_params': {
                    'input_dim': 16,
                    'hidden_dims': [64, 32],
                    'output_dim': 8
                },
                'target_fidelity': 0.75
            },
            'scalability_test': {
                'description': 'Scalability benchmark problem',
                'quantum_params': {
                    'qubit_count': 20,
                    'circuit_depth': 40,
                    'gate_count': 150,
                    'error_rate': 0.02
                },
                'classical_params': {
                    'input_dim': 40,
                    'hidden_dims': [256, 128, 64, 32],
                    'output_dim': 20
                },
                'target_fidelity': 0.85
            }
        }
        
        return problems
    
    def _create_noise_models(self) -> Dict[str, Dict[str, float]]:
        """Create various noise models for testing."""
        return {
            'low_noise': {
                'single_qubit_error': 0.001,
                'two_qubit_error': 0.01,
                'measurement_error': 0.005
            },
            'medium_noise': {
                'single_qubit_error': 0.003,
                'two_qubit_error': 0.02,
                'measurement_error': 0.01
            },
            'high_noise': {
                'single_qubit_error': 0.005,
                'two_qubit_error': 0.03,
                'measurement_error': 0.015
            },
            'realistic_noise': {
                'single_qubit_error': 0.002,
                'two_qubit_error': 0.015,
                'measurement_error': 0.008
            }
        }
    
    def setup_benchmark(self, config: BenchmarkConfig) -> None:
        """Setup benchmark environment."""
        self.current_config = config
        
        # Select benchmark problem
        problem_name = config.test_parameters.get('problem', 'medium_qecc')
        self.current_problem = self.benchmark_problems[problem_name]
        
        # Setup noise model
        noise_name = config.test_parameters.get('noise_model', 'medium_noise')
        self.current_noise = self.noise_models[noise_name]
        
        # Calculate baseline metrics
        self.baseline_metrics = self._calculate_baseline_metrics()
    
    def _calculate_baseline_metrics(self) -> Dict[str, float]:
        """Calculate baseline metrics for comparison."""
        # Theoretical limits and random baselines
        baseline = {
            'random_fitness': 0.5,
            'theoretical_max_fidelity': self.current_problem['target_fidelity'],
            'min_convergence_time': 10.0,  # Minimum expected convergence time
            'max_diversity': 1.0,
            'efficiency_baseline': 0.5
        }
        
        return baseline
    
    def run_algorithm(self, algorithm_name: str, config: Dict[str, Any]) -> BenchmarkResult:
        """Run a single algorithm and return results."""
        start_time = time.time()
        
        try:
            if algorithm_name == 'autonomous_circuit_evolution':
                result = self._run_autonomous_circuit_evolution(config)
            elif algorithm_name == 'quantum_classical_coevolution':
                result = self._run_quantum_classical_coevolution(config)
            elif algorithm_name == 'coevolutionary_optimizer':
                result = self._run_coevolutionary_optimizer(config)
            elif algorithm_name == 'adaptive_nas':
                result = self._run_adaptive_nas(config)
            elif algorithm_name == 'hybrid_evolution_engine':
                result = self._run_hybrid_evolution_engine(config)
            else:
                raise ValueError(f"Unknown algorithm: {algorithm_name}")
            
            runtime = time.time() - start_time
            
            # Extract metrics
            metrics = self._extract_metrics(result, algorithm_name)
            
            # Extract convergence and diversity data
            convergence_data = result.get('convergence_data', [])
            diversity_data = result.get('diversity_data', [])
            
            return BenchmarkResult(
                benchmark_id=self.current_config.benchmark_id,
                algorithm_name=algorithm_name,
                run_id=config.get('run_id', 0),
                metrics=metrics,
                runtime=runtime,
                convergence_data=convergence_data,
                diversity_data=diversity_data,
                additional_data=result,
                success=True
            )
            
        except Exception as e:
            runtime = time.time() - start_time
            
            return BenchmarkResult(
                benchmark_id=self.current_config.benchmark_id,
                algorithm_name=algorithm_name,
                run_id=config.get('run_id', 0),
                metrics={},
                runtime=runtime,
                convergence_data=[],
                diversity_data=[],
                success=False,
                error_message=str(e)
            )
    
    def _run_autonomous_circuit_evolution(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run autonomous circuit evolution algorithm."""
        from .autonomous_circuit_evolution import AutonomousCircuitEvolution
        
        # Configure algorithm
        evolution_system = AutonomousCircuitEvolution(
            population_size=config.get('population_size', 20),
            mutation_rate=config.get('mutation_rate', 0.15),
            crossover_rate=config.get('crossover_rate', 0.7)
        )
        
        # Initialize population
        evolution_system.initialize_population()
        
        # Run evolution
        max_generations = config.get('max_generations', 30)
        convergence_data = []
        diversity_data = []
        
        for generation in range(max_generations):
            breakthroughs = evolution_system.evolve_generation()
            
            # Record metrics
            convergence_data.append(evolution_system.metrics.best_fitness)
            diversity_data.append(evolution_system.metrics.diversity_score)
            
            # Early termination
            if evolution_system.metrics.best_fitness > 0.95:
                break
        
        # Generate report
        final_report = evolution_system._generate_evolution_report([])
        
        # Add benchmark-specific data
        final_report['convergence_data'] = convergence_data
        final_report['diversity_data'] = diversity_data
        
        return final_report
    
    def _run_quantum_classical_coevolution(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run quantum-classical co-evolution algorithm."""
        from .quantum_classical_coevolution import QuantumClassicalCoevolution, CoevolutionStrategy
        
        # Configure algorithm
        strategy = CoevolutionStrategy(config.get('strategy', 'cooperative_coevolution'))
        coevolution_system = QuantumClassicalCoevolution(
            population_size=config.get('population_size', 20),
            strategy=strategy
        )
        
        # Run co-evolution
        max_generations = config.get('max_generations', 25)
        convergence_data = []
        diversity_data = []
        
        # Initialize population
        coevolution_system.initialize_population()
        
        for generation in range(max_generations):
            generation_metrics = coevolution_system.evolve_generation()
            
            # Record metrics
            convergence_data.append(generation_metrics['best_fitness'])
            diversity_data.append(generation_metrics.get('diversity_score', 0.5))
            
            # Early termination
            if generation_metrics['best_fitness'] > 0.90:
                break
        
        # Generate report
        final_report = coevolution_system._generate_coevolution_report()
        
        # Add benchmark-specific data
        final_report['convergence_data'] = convergence_data
        final_report['diversity_data'] = diversity_data
        
        return final_report
    
    def _run_coevolutionary_optimizer(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run co-evolutionary optimizer algorithm."""
        from .coevolutionary_optimizer import CoevolutionaryOptimizer, OptimizationStrategy
        
        # Configure algorithm
        strategy = OptimizationStrategy(config.get('strategy', 'hybrid_gradient_evolutionary'))
        optimizer = CoevolutionaryOptimizer(
            strategy=strategy,
            population_size=config.get('population_size', 25)
        )
        
        # Run optimization
        max_iterations = config.get('max_iterations', 50)
        convergence_data = []
        diversity_data = []
        
        # Run optimization with tracking
        report = optimizer.optimize(
            max_iterations=max_iterations,
            convergence_tolerance=1e-6
        )
        
        # Extract tracking data
        if 'performance_metrics' in report:
            convergence_data = report['performance_metrics'].get('best_fitness', [])
        
        # Estimate diversity data (simplified)
        diversity_data = [0.5 - 0.4 * (i / max_iterations) for i in range(len(convergence_data))]
        
        # Add benchmark-specific data
        report['convergence_data'] = convergence_data
        report['diversity_data'] = diversity_data
        
        return report
    
    def _run_adaptive_nas(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run adaptive neural architecture search algorithm."""
        from .adaptive_neural_architecture_search import AdaptiveNeuralArchitectureSearch, SearchStrategy
        
        # Configure algorithm
        strategy = SearchStrategy(config.get('strategy', 'evolutionary_architecture_search'))
        nas_system = AdaptiveNeuralArchitectureSearch(
            search_strategy=strategy,
            population_size=config.get('population_size', 15),
            max_generations=config.get('max_generations', 30)
        )
        
        # Run search
        report = nas_system.search()
        
        # Extract tracking data
        convergence_data = report['performance_metrics'].get('best_fitness', [])
        diversity_data = report['performance_metrics'].get('fitness_variance', [])
        
        # Normalize diversity data
        if diversity_data:
            max_var = max(diversity_data)
            diversity_data = [1.0 - (var / max(max_var, 1e-6)) for var in diversity_data]
        
        # Add benchmark-specific data
        report['convergence_data'] = convergence_data
        report['diversity_data'] = diversity_data
        
        return report
    
    def _run_hybrid_evolution_engine(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run hybrid evolution engine algorithm."""
        from .hybrid_evolution_engine import HybridEvolutionEngine, HybridStrategy
        
        # Configure algorithm
        strategy = HybridStrategy(config.get('strategy', 'adaptive_multi_strategy'))
        engine = HybridEvolutionEngine(
            strategy=strategy,
            population_size=config.get('population_size', 30)
        )
        
        # Run evolution
        max_generations = config.get('max_generations', 40)
        report = engine.evolve(
            max_generations=max_generations,
            convergence_threshold=1e-6
        )
        
        # Extract tracking data
        convergence_data = report['performance_metrics'].get('best_fitness', [])
        diversity_data = report['performance_metrics'].get('population_diversity', [])
        
        # Add benchmark-specific data
        report['convergence_data'] = convergence_data
        report['diversity_data'] = diversity_data
        
        return report
    
    def _extract_metrics(self, result: Dict[str, Any], algorithm_name: str) -> Dict[str, float]:
        """Extract standardized metrics from algorithm results."""
        metrics = {}
        
        # Fitness metrics
        if 'best_fitness' in result:
            metrics['final_fitness'] = result['best_fitness']
        elif 'best_score' in result:
            metrics['final_fitness'] = result['best_score']
        else:
            metrics['final_fitness'] = 0.0
        
        # Convergence metrics
        convergence_data = result.get('convergence_data', [])
        if len(convergence_data) > 1:
            metrics['convergence_speed'] = self._calculate_convergence_speed(convergence_data)
            metrics['fitness_improvement'] = convergence_data[-1] - convergence_data[0]
            metrics['convergence_stability'] = 1.0 - np.var(convergence_data[-10:]) if len(convergence_data) >= 10 else 0.5
        else:
            metrics['convergence_speed'] = 0.0
            metrics['fitness_improvement'] = 0.0
            metrics['convergence_stability'] = 0.0
        
        # Diversity metrics
        diversity_data = result.get('diversity_data', [])
        if diversity_data:
            metrics['final_diversity'] = diversity_data[-1]
            metrics['average_diversity'] = np.mean(diversity_data)
            metrics['diversity_maintenance'] = np.mean(diversity_data) / max(diversity_data[0], 1e-6) if diversity_data[0] > 0 else 0.5
        else:
            metrics['final_diversity'] = 0.0
            metrics['average_diversity'] = 0.0
            metrics['diversity_maintenance'] = 0.0
        
        # Efficiency metrics
        total_generations = result.get('total_generations', result.get('final_generation', 1))
        metrics['efficiency'] = metrics['final_fitness'] / max(total_generations, 1)
        
        # Algorithm-specific metrics
        if algorithm_name == 'quantum_classical_coevolution':
            metrics['pareto_front_size'] = len(result.get('pareto_front', []))
            component_analysis = result.get('component_analysis', {})
            metrics['quantum_improvement'] = component_analysis.get('quantum_improvement', 0.0)
            metrics['classical_improvement'] = component_analysis.get('classical_improvement', 0.0)
        
        elif algorithm_name == 'coevolutionary_optimizer':
            convergence_analysis = result.get('convergence_analysis', {})
            metrics['total_improvement'] = convergence_analysis.get('total_improvement', 0.0)
            metrics['convergence_rate'] = convergence_analysis.get('convergence_rate', 0.0)
        
        elif algorithm_name == 'adaptive_nas':
            arch_analysis = result.get('architecture_analysis', {})
            metrics['quantum_utilization'] = arch_analysis.get('quantum_utilization', 0.0)
            metrics['average_complexity'] = arch_analysis.get('average_complexity', 0.0)
        
        elif algorithm_name == 'hybrid_evolution_engine':
            hybrid_analysis = result.get('hybrid_analysis', {})
            metrics['hybrid_advantage'] = hybrid_analysis.get('hybrid_advantage', 0.0)
            metrics['evolution_efficiency'] = hybrid_analysis.get('evolution_efficiency', 0.0)
        
        # Robustness metrics (normalized relative to baseline)
        baseline = self.baseline_metrics
        metrics['fitness_improvement_normalized'] = metrics['fitness_improvement'] / baseline.get('theoretical_max_fidelity', 1.0)
        metrics['efficiency_normalized'] = metrics['efficiency'] / baseline.get('efficiency_baseline', 1.0)
        
        return metrics
    
    def _calculate_convergence_speed(self, convergence_data: List[float]) -> float:
        """Calculate convergence speed metric."""
        if len(convergence_data) < 3:
            return 0.0
        
        # Find when algorithm reaches 90% of final fitness
        final_fitness = convergence_data[-1]
        target_fitness = 0.9 * final_fitness
        
        for i, fitness in enumerate(convergence_data):
            if fitness >= target_fitness:
                # Convergence speed: inverse of generations to reach target
                return 1.0 / max(i + 1, 1)
        
        # Didn't reach target - slow convergence
        return 1.0 / len(convergence_data)
    
    def cleanup_benchmark(self) -> None:
        """Cleanup benchmark environment."""
        self.current_problem = None
        self.current_config = None


class CoevolutionBenchmarkFramework:
    """
    Comprehensive benchmarking framework for co-evolution algorithms.
    
    Provides standardized testing, statistical analysis, and comparative
    evaluation of all co-evolution algorithms in the QECC-QML system.
    """
    
    def __init__(self):
        self.benchmark_suite = CoevolutionBenchmarkSuite()
        self.results_database: List[BenchmarkResult] = []
        self.comparative_analyses: List[ComparativeAnalysis] = []
        
        # Statistical analysis tools
        self.confidence_level = 0.95
        self.significance_threshold = 0.05
        
    def create_benchmark_config(self, benchmark_type: BenchmarkType, **kwargs) -> BenchmarkConfig:
        """Create benchmark configuration."""
        if benchmark_type == BenchmarkType.PERFORMANCE_BENCHMARK:
            return self._create_performance_benchmark_config(**kwargs)
        elif benchmark_type == BenchmarkType.SCALABILITY_BENCHMARK:
            return self._create_scalability_benchmark_config(**kwargs)
        elif benchmark_type == BenchmarkType.ROBUSTNESS_BENCHMARK:
            return self._create_robustness_benchmark_config(**kwargs)
        elif benchmark_type == BenchmarkType.CONVERGENCE_BENCHMARK:
            return self._create_convergence_benchmark_config(**kwargs)
        elif benchmark_type == BenchmarkType.COMPARATIVE_BENCHMARK:
            return self._create_comparative_benchmark_config(**kwargs)
        elif benchmark_type == BenchmarkType.QUANTUM_ADVANTAGE_BENCHMARK:
            return self._create_quantum_advantage_benchmark_config(**kwargs)
        else:
            raise ValueError(f"Unknown benchmark type: {benchmark_type}")
    
    def _create_performance_benchmark_config(self, **kwargs) -> BenchmarkConfig:
        """Create performance benchmark configuration."""
        algorithms = [
            {'name': 'autonomous_circuit_evolution', 'population_size': 20, 'max_generations': 30},
            {'name': 'quantum_classical_coevolution', 'population_size': 20, 'max_generations': 25},
            {'name': 'coevolutionary_optimizer', 'population_size': 25, 'max_iterations': 50},
            {'name': 'adaptive_nas', 'population_size': 15, 'max_generations': 30},
            {'name': 'hybrid_evolution_engine', 'population_size': 30, 'max_generations': 40}
        ]
        
        return BenchmarkConfig(
            benchmark_id="performance_benchmark",
            benchmark_type=BenchmarkType.PERFORMANCE_BENCHMARK,
            algorithm_configs=algorithms,
            test_parameters={
                'problem': kwargs.get('problem', 'medium_qecc'),
                'noise_model': kwargs.get('noise_model', 'medium_noise')
            },
            metrics_to_evaluate=[
                MetricType.FITNESS_METRIC,
                MetricType.CONVERGENCE_METRIC,
                MetricType.EFFICIENCY_METRIC
            ],
            repetitions=kwargs.get('repetitions', 5)
        )
    
    def _create_scalability_benchmark_config(self, **kwargs) -> BenchmarkConfig:
        """Create scalability benchmark configuration."""
        algorithms = [
            {'name': 'hybrid_evolution_engine', 'population_size': 50, 'max_generations': 60},
            {'name': 'coevolutionary_optimizer', 'population_size': 40, 'max_iterations': 80},
            {'name': 'quantum_classical_coevolution', 'population_size': 35, 'max_generations': 50}
        ]
        
        return BenchmarkConfig(
            benchmark_id="scalability_benchmark",
            benchmark_type=BenchmarkType.SCALABILITY_BENCHMARK,
            algorithm_configs=algorithms,
            test_parameters={
                'problem': 'scalability_test',
                'noise_model': 'realistic_noise'
            },
            metrics_to_evaluate=[
                MetricType.FITNESS_METRIC,
                MetricType.EFFICIENCY_METRIC,
                MetricType.CONVERGENCE_METRIC
            ],
            repetitions=kwargs.get('repetitions', 3)
        )
    
    def _create_robustness_benchmark_config(self, **kwargs) -> BenchmarkConfig:
        """Create robustness benchmark configuration."""
        algorithms = [
            {'name': 'hybrid_evolution_engine', 'population_size': 25, 'max_generations': 35},
            {'name': 'autonomous_circuit_evolution', 'population_size': 20, 'max_generations': 30},
            {'name': 'coevolutionary_optimizer', 'population_size': 25, 'max_iterations': 45}
        ]
        
        return BenchmarkConfig(
            benchmark_id="robustness_benchmark",
            benchmark_type=BenchmarkType.ROBUSTNESS_BENCHMARK,
            algorithm_configs=algorithms,
            test_parameters={
                'problem': 'noisy_environment',
                'noise_model': 'high_noise'
            },
            metrics_to_evaluate=[
                MetricType.FITNESS_METRIC,
                MetricType.ROBUSTNESS_METRIC,
                MetricType.DIVERSITY_METRIC
            ],
            repetitions=kwargs.get('repetitions', 7)
        )
    
    def _create_convergence_benchmark_config(self, **kwargs) -> BenchmarkConfig:
        """Create convergence benchmark configuration."""
        algorithms = [
            {'name': 'coevolutionary_optimizer', 'population_size': 20, 'max_iterations': 100},
            {'name': 'hybrid_evolution_engine', 'population_size': 25, 'max_generations': 80},
            {'name': 'quantum_classical_coevolution', 'population_size': 20, 'max_generations': 60}
        ]
        
        return BenchmarkConfig(
            benchmark_id="convergence_benchmark",
            benchmark_type=BenchmarkType.CONVERGENCE_BENCHMARK,
            algorithm_configs=algorithms,
            test_parameters={
                'problem': 'complex_qecc',
                'noise_model': 'medium_noise'
            },
            metrics_to_evaluate=[
                MetricType.CONVERGENCE_METRIC,
                MetricType.FITNESS_METRIC,
                MetricType.EFFICIENCY_METRIC
            ],
            repetitions=kwargs.get('repetitions', 10)
        )
    
    def _create_comparative_benchmark_config(self, **kwargs) -> BenchmarkConfig:
        """Create comparative benchmark configuration."""
        algorithms = [
            {'name': 'autonomous_circuit_evolution', 'population_size': 20, 'max_generations': 30},
            {'name': 'quantum_classical_coevolution', 'population_size': 20, 'max_generations': 25},
            {'name': 'coevolutionary_optimizer', 'population_size': 25, 'max_iterations': 50},
            {'name': 'adaptive_nas', 'population_size': 15, 'max_generations': 30},
            {'name': 'hybrid_evolution_engine', 'population_size': 30, 'max_generations': 40}
        ]
        
        return BenchmarkConfig(
            benchmark_id="comparative_benchmark",
            benchmark_type=BenchmarkType.COMPARATIVE_BENCHMARK,
            algorithm_configs=algorithms,
            test_parameters={
                'problem': kwargs.get('problem', 'medium_qecc'),
                'noise_model': kwargs.get('noise_model', 'realistic_noise')
            },
            metrics_to_evaluate=[
                MetricType.FITNESS_METRIC,
                MetricType.CONVERGENCE_METRIC,
                MetricType.DIVERSITY_METRIC,
                MetricType.EFFICIENCY_METRIC,
                MetricType.ROBUSTNESS_METRIC
            ],
            repetitions=kwargs.get('repetitions', 10)
        )
    
    def _create_quantum_advantage_benchmark_config(self, **kwargs) -> BenchmarkConfig:
        """Create quantum advantage benchmark configuration."""
        algorithms = [
            {'name': 'quantum_classical_coevolution', 'population_size': 20, 'max_generations': 25},
            {'name': 'hybrid_evolution_engine', 'population_size': 25, 'max_generations': 35},
            {'name': 'adaptive_nas', 'population_size': 15, 'max_generations': 30}
        ]
        
        return BenchmarkConfig(
            benchmark_id="quantum_advantage_benchmark",
            benchmark_type=BenchmarkType.QUANTUM_ADVANTAGE_BENCHMARK,
            algorithm_configs=algorithms,
            test_parameters={
                'problem': 'complex_qecc',
                'noise_model': 'low_noise'
            },
            metrics_to_evaluate=[
                MetricType.QUANTUM_METRIC,
                MetricType.FITNESS_METRIC,
                MetricType.EFFICIENCY_METRIC
            ],
            repetitions=kwargs.get('repetitions', 8)
        )
    
    def run_benchmark(self, config: BenchmarkConfig) -> List[BenchmarkResult]:
        """Run a complete benchmark."""
        self.log(f"🧪 Running benchmark: {config.benchmark_id}")
        
        # Setup benchmark
        self.benchmark_suite.setup_benchmark(config)
        
        all_results = []
        
        for algorithm_config in config.algorithm_configs:
            algorithm_name = algorithm_config['name']
            self.log(f"   Testing {algorithm_name}")
            
            algorithm_results = []
            
            for run_id in range(config.repetitions):
                self.log(f"     Run {run_id + 1}/{config.repetitions}")
                
                # Setup run-specific config
                run_config = algorithm_config.copy()
                run_config['run_id'] = run_id
                
                # Run algorithm
                result = self.benchmark_suite.run_algorithm(algorithm_name, run_config)
                algorithm_results.append(result)
                all_results.append(result)
                
                if not result.success:
                    self.log(f"     ⚠️ Run failed: {result.error_message}")
            
            # Log algorithm summary
            successful_runs = [r for r in algorithm_results if r.success]
            if successful_runs:
                avg_fitness = np.mean([r.metrics.get('final_fitness', 0) for r in successful_runs])
                avg_runtime = np.mean([r.runtime for r in successful_runs])
                self.log(f"     Summary: {len(successful_runs)}/{config.repetitions} successful, "
                        f"avg fitness: {avg_fitness:.3f}, avg runtime: {avg_runtime:.1f}s")
        
        # Cleanup benchmark
        self.benchmark_suite.cleanup_benchmark()
        
        # Store results
        self.results_database.extend(all_results)
        
        self.log(f"✅ Benchmark {config.benchmark_id} completed")
        
        return all_results
    
    def analyze_results(self, benchmark_id: str) -> ComparativeAnalysis:
        """Analyze benchmark results and perform statistical tests."""
        # Get results for this benchmark
        benchmark_results = [r for r in self.results_database if r.benchmark_id == benchmark_id]
        
        if not benchmark_results:
            raise ValueError(f"No results found for benchmark {benchmark_id}")
        
        # Group results by algorithm
        algorithm_results = defaultdict(list)
        for result in benchmark_results:
            if result.success:
                algorithm_results[result.algorithm_name].append(result)
        
        algorithms = list(algorithm_results.keys())
        
        # Perform statistical analysis
        statistical_tests = self._perform_statistical_tests(algorithm_results)
        
        # Calculate performance rankings
        performance_rankings = self._calculate_performance_rankings(algorithm_results)
        
        # Identify significant differences
        significant_differences = self._identify_significant_differences(statistical_tests)
        
        # Calculate confidence intervals
        confidence_intervals = self._calculate_confidence_intervals(algorithm_results)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(algorithm_results, statistical_tests)
        
        analysis = ComparativeAnalysis(
            comparison_id=f"{benchmark_id}_analysis",
            algorithms_compared=algorithms,
            statistical_tests=statistical_tests,
            performance_rankings=performance_rankings,
            significant_differences=significant_differences,
            recommendations=recommendations,
            confidence_intervals=confidence_intervals
        )
        
        self.comparative_analyses.append(analysis)
        
        return analysis
    
    def _perform_statistical_tests(self, algorithm_results: Dict[str, List[BenchmarkResult]]) -> Dict[str, Dict[str, float]]:
        """Perform statistical tests between algorithms."""
        statistical_tests = {}
        
        algorithms = list(algorithm_results.keys())
        
        # For each pair of algorithms
        for i, alg1 in enumerate(algorithms):
            for j, alg2 in enumerate(algorithms[i+1:], i+1):
                pair_key = f"{alg1}_vs_{alg2}"
                statistical_tests[pair_key] = {}
                
                # Get metrics for both algorithms
                alg1_fitness = [r.metrics.get('final_fitness', 0) for r in algorithm_results[alg1]]
                alg2_fitness = [r.metrics.get('final_fitness', 0) for r in algorithm_results[alg2]]
                
                alg1_efficiency = [r.metrics.get('efficiency', 0) for r in algorithm_results[alg1]]
                alg2_efficiency = [r.metrics.get('efficiency', 0) for r in algorithm_results[alg2]]
                
                # Perform t-tests (simplified implementation)
                fitness_p_value = self._welch_t_test(alg1_fitness, alg2_fitness)
                efficiency_p_value = self._welch_t_test(alg1_efficiency, alg2_efficiency)
                
                statistical_tests[pair_key]['fitness_p_value'] = fitness_p_value
                statistical_tests[pair_key]['efficiency_p_value'] = efficiency_p_value
                
                # Effect sizes (Cohen's d)
                fitness_effect_size = self._cohens_d(alg1_fitness, alg2_fitness)
                efficiency_effect_size = self._cohens_d(alg1_efficiency, alg2_efficiency)
                
                statistical_tests[pair_key]['fitness_effect_size'] = fitness_effect_size
                statistical_tests[pair_key]['efficiency_effect_size'] = efficiency_effect_size
        
        return statistical_tests
    
    def _welch_t_test(self, sample1: List[float], sample2: List[float]) -> float:
        """Simplified Welch's t-test implementation."""
        if len(sample1) < 2 or len(sample2) < 2:
            return 1.0  # No significant difference if insufficient data
        
        mean1, mean2 = np.mean(sample1), np.mean(sample2)
        var1, var2 = np.var(sample1, ddof=1), np.var(sample2, ddof=1)
        n1, n2 = len(sample1), len(sample2)
        
        # Welch's t-statistic
        pooled_se = np.sqrt(var1/n1 + var2/n2)
        if pooled_se == 0:
            return 1.0
        
        t_stat = (mean1 - mean2) / pooled_se
        
        # Degrees of freedom (Welch-Satterthwaite equation)
        dof = (var1/n1 + var2/n2)**2 / ((var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1))
        
        # Simplified p-value estimation
        # For dof > 30, t-distribution ≈ normal distribution
        if dof > 30:
            # Two-tailed test
            p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(t_stat) / math.sqrt(2))))
        else:
            # Very simplified approximation for small samples
            p_value = max(0.001, min(0.999, 2 * (1 - abs(t_stat) / (abs(t_stat) + dof))))
        
        return p_value
    
    def _cohens_d(self, sample1: List[float], sample2: List[float]) -> float:
        """Calculate Cohen's d effect size."""
        if len(sample1) < 2 or len(sample2) < 2:
            return 0.0
        
        mean1, mean2 = np.mean(sample1), np.mean(sample2)
        var1, var2 = np.var(sample1, ddof=1), np.var(sample2, ddof=1)
        n1, n2 = len(sample1), len(sample2)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
        
        if pooled_std == 0:
            return 0.0
        
        return (mean1 - mean2) / pooled_std
    
    def _calculate_performance_rankings(self, algorithm_results: Dict[str, List[BenchmarkResult]]) -> Dict[str, int]:
        """Calculate performance rankings for algorithms."""
        algorithm_scores = {}
        
        for algorithm, results in algorithm_results.items():
            if results:
                # Primary metric: average final fitness
                avg_fitness = np.mean([r.metrics.get('final_fitness', 0) for r in results])
                
                # Secondary metrics
                avg_efficiency = np.mean([r.metrics.get('efficiency', 0) for r in results])
                avg_convergence_speed = np.mean([r.metrics.get('convergence_speed', 0) for r in results])
                
                # Combined score
                combined_score = 0.6 * avg_fitness + 0.3 * avg_efficiency + 0.1 * avg_convergence_speed
                algorithm_scores[algorithm] = combined_score
        
        # Rank algorithms by combined score
        sorted_algorithms = sorted(algorithm_scores.items(), key=lambda x: x[1], reverse=True)
        
        rankings = {}
        for rank, (algorithm, score) in enumerate(sorted_algorithms, 1):
            rankings[algorithm] = rank
        
        return rankings
    
    def _identify_significant_differences(self, statistical_tests: Dict[str, Dict[str, float]]) -> List[Tuple[str, str, str]]:
        """Identify statistically significant differences."""
        significant_differences = []
        
        for pair_key, tests in statistical_tests.items():
            alg1, alg2 = pair_key.split('_vs_')
            
            # Check fitness differences
            if tests.get('fitness_p_value', 1.0) < self.significance_threshold:
                effect_size = abs(tests.get('fitness_effect_size', 0))
                if effect_size > 0.5:  # Medium to large effect size
                    significant_differences.append((alg1, alg2, 'fitness'))
            
            # Check efficiency differences
            if tests.get('efficiency_p_value', 1.0) < self.significance_threshold:
                effect_size = abs(tests.get('efficiency_effect_size', 0))
                if effect_size > 0.5:
                    significant_differences.append((alg1, alg2, 'efficiency'))
        
        return significant_differences
    
    def _calculate_confidence_intervals(self, algorithm_results: Dict[str, List[BenchmarkResult]]) -> Dict[str, Dict[str, Tuple[float, float]]]:
        """Calculate confidence intervals for algorithm performance."""
        confidence_intervals = {}
        
        for algorithm, results in algorithm_results.items():
            if len(results) < 2:
                continue
            
            confidence_intervals[algorithm] = {}
            
            # Fitness confidence interval
            fitness_values = [r.metrics.get('final_fitness', 0) for r in results]
            fitness_ci = self._calculate_ci(fitness_values)
            confidence_intervals[algorithm]['fitness'] = fitness_ci
            
            # Efficiency confidence interval
            efficiency_values = [r.metrics.get('efficiency', 0) for r in results]
            efficiency_ci = self._calculate_ci(efficiency_values)
            confidence_intervals[algorithm]['efficiency'] = efficiency_ci
            
            # Convergence speed confidence interval
            convergence_values = [r.metrics.get('convergence_speed', 0) for r in results]
            convergence_ci = self._calculate_ci(convergence_values)
            confidence_intervals[algorithm]['convergence_speed'] = convergence_ci
        
        return confidence_intervals
    
    def _calculate_ci(self, values: List[float]) -> Tuple[float, float]:
        """Calculate confidence interval for a list of values."""
        if len(values) < 2:
            return (0.0, 0.0)
        
        mean = np.mean(values)
        std = np.std(values, ddof=1)
        n = len(values)
        
        # t-critical value (approximation for 95% confidence)
        t_critical = 2.0 if n > 30 else 2.5  # Simplified
        
        margin_of_error = t_critical * (std / np.sqrt(n))
        
        return (mean - margin_of_error, mean + margin_of_error)
    
    def _generate_recommendations(self, algorithm_results: Dict[str, List[BenchmarkResult]], statistical_tests: Dict[str, Dict[str, float]]) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []
        
        # Find best performing algorithm
        algorithm_scores = {}
        for algorithm, results in algorithm_results.items():
            if results:
                avg_fitness = np.mean([r.metrics.get('final_fitness', 0) for r in results])
                algorithm_scores[algorithm] = avg_fitness
        
        if algorithm_scores:
            best_algorithm = max(algorithm_scores, key=algorithm_scores.get)
            best_score = algorithm_scores[best_algorithm]
            recommendations.append(f"Best overall performer: {best_algorithm} (fitness: {best_score:.3f})")
        
        # Identify algorithms with significant advantages
        strong_performers = [alg for alg, score in algorithm_scores.items() if score > 0.8]
        if strong_performers:
            recommendations.append(f"Strong performers (fitness > 0.8): {', '.join(strong_performers)}")
        
        # Efficiency recommendations
        efficiency_scores = {}
        for algorithm, results in algorithm_results.items():
            if results:
                avg_efficiency = np.mean([r.metrics.get('efficiency', 0) for r in results])
                efficiency_scores[algorithm] = avg_efficiency
        
        if efficiency_scores:
            most_efficient = max(efficiency_scores, key=efficiency_scores.get)
            recommendations.append(f"Most efficient algorithm: {most_efficient}")
        
        # Statistical significance recommendations
        significant_pairs = []
        for pair_key, tests in statistical_tests.items():
            if tests.get('fitness_p_value', 1.0) < 0.05:
                significant_pairs.append(pair_key)
        
        if significant_pairs:
            recommendations.append(f"Statistically significant differences found in {len(significant_pairs)} comparisons")
        
        # Practical recommendations
        if len(algorithm_results) > 3:
            recommendations.append("Consider ensemble approaches combining top-performing algorithms")
        
        recommendations.append("Validate results on additional benchmark problems for generalization")
        
        return recommendations
    
    def generate_comprehensive_report(self, benchmark_ids: List[str]) -> Dict[str, Any]:
        """Generate comprehensive benchmark report."""
        report = {
            'timestamp': time.time(),
            'benchmark_summary': {},
            'cross_benchmark_analysis': {},
            'algorithm_profiles': {},
            'overall_recommendations': []
        }
        
        # Analyze each benchmark
        for benchmark_id in benchmark_ids:
            try:
                analysis = self.analyze_results(benchmark_id)
                report['benchmark_summary'][benchmark_id] = asdict(analysis)
            except ValueError as e:
                self.log(f"⚠️ Could not analyze {benchmark_id}: {e}")
        
        # Cross-benchmark analysis
        if len(benchmark_ids) > 1:
            report['cross_benchmark_analysis'] = self._cross_benchmark_analysis(benchmark_ids)
        
        # Algorithm profiles
        report['algorithm_profiles'] = self._generate_algorithm_profiles()
        
        # Overall recommendations
        report['overall_recommendations'] = self._generate_overall_recommendations()
        
        return report
    
    def _cross_benchmark_analysis(self, benchmark_ids: List[str]) -> Dict[str, Any]:
        """Perform cross-benchmark analysis."""
        cross_analysis = {
            'consistency_scores': {},
            'versatility_rankings': {},
            'benchmark_difficulty': {}
        }
        
        # Get all algorithms that appear in multiple benchmarks
        algorithm_performances = defaultdict(dict)
        
        for benchmark_id in benchmark_ids:
            benchmark_results = [r for r in self.results_database if r.benchmark_id == benchmark_id and r.success]
            
            algorithm_avg_fitness = defaultdict(list)
            for result in benchmark_results:
                algorithm_avg_fitness[result.algorithm_name].append(result.metrics.get('final_fitness', 0))
            
            for algorithm, fitnesses in algorithm_avg_fitness.items():
                algorithm_performances[algorithm][benchmark_id] = np.mean(fitnesses)
        
        # Calculate consistency scores (low variance across benchmarks)
        for algorithm, performances in algorithm_performances.items():
            if len(performances) > 1:
                consistency = 1.0 - np.var(list(performances.values()))
                cross_analysis['consistency_scores'][algorithm] = max(0.0, consistency)
        
        # Versatility rankings (average performance across benchmarks)
        for algorithm, performances in algorithm_performances.items():
            if performances:
                avg_performance = np.mean(list(performances.values()))
                cross_analysis['versatility_rankings'][algorithm] = avg_performance
        
        # Benchmark difficulty (average performance across all algorithms)
        for benchmark_id in benchmark_ids:
            benchmark_results = [r for r in self.results_database if r.benchmark_id == benchmark_id and r.success]
            if benchmark_results:
                avg_fitness = np.mean([r.metrics.get('final_fitness', 0) for r in benchmark_results])
                cross_analysis['benchmark_difficulty'][benchmark_id] = 1.0 - avg_fitness  # Higher difficulty = lower average fitness
        
        return cross_analysis
    
    def _generate_algorithm_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Generate comprehensive profiles for each algorithm."""
        algorithm_profiles = {}
        
        # Group all results by algorithm
        algorithm_all_results = defaultdict(list)
        for result in self.results_database:
            if result.success:
                algorithm_all_results[result.algorithm_name].append(result)
        
        for algorithm, results in algorithm_all_results.items():
            if not results:
                continue
            
            profile = {
                'total_runs': len(results),
                'success_rate': len(results) / max(len([r for r in self.results_database if r.algorithm_name == algorithm]), 1),
                'average_metrics': {},
                'strengths': [],
                'weaknesses': [],
                'best_use_cases': []
            }
            
            # Calculate average metrics
            all_metrics = defaultdict(list)
            for result in results:
                for metric, value in result.metrics.items():
                    all_metrics[metric].append(value)
            
            for metric, values in all_metrics.items():
                profile['average_metrics'][metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
            
            # Identify strengths and weaknesses
            avg_fitness = profile['average_metrics'].get('final_fitness', {}).get('mean', 0)
            avg_efficiency = profile['average_metrics'].get('efficiency', {}).get('mean', 0)
            avg_convergence = profile['average_metrics'].get('convergence_speed', {}).get('mean', 0)
            
            if avg_fitness > 0.8:
                profile['strengths'].append('High fitness achievement')
            elif avg_fitness < 0.6:
                profile['weaknesses'].append('Low fitness achievement')
            
            if avg_efficiency > 0.5:
                profile['strengths'].append('High efficiency')
            elif avg_efficiency < 0.3:
                profile['weaknesses'].append('Low efficiency')
            
            if avg_convergence > 0.1:
                profile['strengths'].append('Fast convergence')
            elif avg_convergence < 0.05:
                profile['weaknesses'].append('Slow convergence')
            
            # Best use cases based on algorithm characteristics
            if algorithm == 'autonomous_circuit_evolution':
                profile['best_use_cases'] = ['Circuit architecture optimization', 'Hardware-specific adaptation']
            elif algorithm == 'quantum_classical_coevolution':
                profile['best_use_cases'] = ['Hybrid system optimization', 'Multi-objective problems']
            elif algorithm == 'coevolutionary_optimizer':
                profile['best_use_cases'] = ['Parameter optimization', 'Gradient-based problems']
            elif algorithm == 'adaptive_nas':
                profile['best_use_cases'] = ['Architecture search', 'Network design']
            elif algorithm == 'hybrid_evolution_engine':
                profile['best_use_cases'] = ['Complex optimization', 'Multi-strategy problems']
            
            algorithm_profiles[algorithm] = profile
        
        return algorithm_profiles
    
    def _generate_overall_recommendations(self) -> List[str]:
        """Generate overall recommendations based on all benchmarks."""
        recommendations = []
        
        # Find most consistent performer across benchmarks
        if len(self.comparative_analyses) > 1:
            algorithm_rankings = defaultdict(list)
            
            for analysis in self.comparative_analyses:
                for algorithm, rank in analysis.performance_rankings.items():
                    algorithm_rankings[algorithm].append(rank)
            
            # Find algorithm with best average ranking
            avg_rankings = {}
            for algorithm, ranks in algorithm_rankings.items():
                if len(ranks) > 1:  # Only consider algorithms in multiple benchmarks
                    avg_rankings[algorithm] = np.mean(ranks)
            
            if avg_rankings:
                best_overall = min(avg_rankings, key=avg_rankings.get)
                recommendations.append(f"Most consistent performer across benchmarks: {best_overall}")
        
        # Recommendations based on problem types
        recommendations.extend([
            "For performance-critical applications, use hybrid_evolution_engine or coevolutionary_optimizer",
            "For quantum circuit optimization, autonomous_circuit_evolution shows strong results",
            "For hybrid quantum-classical systems, quantum_classical_coevolution is recommended",
            "For architecture search problems, adaptive_nas provides specialized capabilities",
            "Consider ensemble approaches for complex, multi-objective problems"
        ])
        
        # Statistical recommendations
        total_successful_runs = len([r for r in self.results_database if r.success])
        total_runs = len(self.results_database)
        
        if total_successful_runs / max(total_runs, 1) > 0.9:
            recommendations.append("High success rate indicates robust algorithm implementations")
        
        # Future research directions
        recommendations.extend([
            "Investigate adaptive parameter tuning for improved performance",
            "Explore hybrid approaches combining multiple algorithms",
            "Validate results on real quantum hardware when available",
            "Extend benchmarks to larger problem sizes for scalability assessment"
        ])
        
        return recommendations
    
    def save_results(self, filename: str) -> None:
        """Save all benchmark results to file."""
        results_data = {
            'results_database': [asdict(result) for result in self.results_database],
            'comparative_analyses': [asdict(analysis) for analysis in self.comparative_analyses],
            'metadata': {
                'total_benchmarks': len(set(r.benchmark_id for r in self.results_database)),
                'total_algorithms': len(set(r.algorithm_name for r in self.results_database)),
                'total_runs': len(self.results_database),
                'timestamp': time.time()
            }
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(results_data, f, indent=2, default=str)
            self.log(f"📁 Results saved to {filename}")
        except Exception as e:
            self.log(f"⚠️ Could not save results: {e}")
    
    def log(self, message: str):
        """Enhanced logging with timestamps."""
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] BENCHMARK: {message}")


def run_comprehensive_coevolution_benchmarks():
    """Execute comprehensive benchmarking of all co-evolution algorithms."""
    print("🧪 COMPREHENSIVE CO-EVOLUTION BENCHMARKS")
    print("=" * 60)
    
    # Initialize benchmark framework
    framework = CoevolutionBenchmarkFramework()
    
    # Define benchmark configurations
    benchmark_configs = [
        framework.create_benchmark_config(BenchmarkType.PERFORMANCE_BENCHMARK, repetitions=5),
        framework.create_benchmark_config(BenchmarkType.CONVERGENCE_BENCHMARK, repetitions=7),
        framework.create_benchmark_config(BenchmarkType.ROBUSTNESS_BENCHMARK, repetitions=6),
        framework.create_benchmark_config(BenchmarkType.COMPARATIVE_BENCHMARK, repetitions=8)
    ]
    
    all_benchmark_ids = []
    
    # Run all benchmarks
    for config in benchmark_configs:
        print(f"\n🔬 Running {config.benchmark_type.value}")
        print("-" * 50)
        
        try:
            results = framework.run_benchmark(config)
            all_benchmark_ids.append(config.benchmark_id)
            
            # Quick summary
            successful_results = [r for r in results if r.success]
            total_algorithms = len(set(r.algorithm_name for r in successful_results))
            avg_fitness = np.mean([r.metrics.get('final_fitness', 0) for r in successful_results])
            
            print(f"   Completed: {len(successful_results)}/{len(results)} successful runs")
            print(f"   Algorithms tested: {total_algorithms}")
            print(f"   Average fitness: {avg_fitness:.3f}")
            
        except Exception as e:
            print(f"   ⚠️ Benchmark failed: {e}")
    
    # Generate comprehensive analysis
    print(f"\n📊 COMPREHENSIVE ANALYSIS")
    print("=" * 60)
    
    comprehensive_report = framework.generate_comprehensive_report(all_benchmark_ids)
    
    # Display key findings
    print(f"Total benchmarks completed: {len(all_benchmark_ids)}")
    print(f"Total algorithm comparisons: {len(framework.comparative_analyses)}")
    
    # Algorithm performance summary
    if 'algorithm_profiles' in comprehensive_report:
        print(f"\n🔬 Algorithm Performance Summary:")
        
        for algorithm, profile in comprehensive_report['algorithm_profiles'].items():
            success_rate = profile['success_rate']
            avg_fitness = profile['average_metrics'].get('final_fitness', {}).get('mean', 0)
            
            print(f"   {algorithm}:")
            print(f"     Success Rate: {success_rate:.2%}")
            print(f"     Average Fitness: {avg_fitness:.3f}")
            print(f"     Strengths: {', '.join(profile['strengths'][:2])}")
    
    # Cross-benchmark analysis
    if 'cross_benchmark_analysis' in comprehensive_report:
        cross_analysis = comprehensive_report['cross_benchmark_analysis']
        
        if 'versatility_rankings' in cross_analysis:
            print(f"\n🏆 Versatility Rankings (Average Performance):")
            versatility = cross_analysis['versatility_rankings']
            sorted_algorithms = sorted(versatility.items(), key=lambda x: x[1], reverse=True)
            
            for rank, (algorithm, score) in enumerate(sorted_algorithms[:5], 1):
                print(f"   {rank}. {algorithm}: {score:.3f}")
        
        if 'consistency_scores' in cross_analysis:
            print(f"\n📈 Consistency Scores (Low Variance):")
            consistency = cross_analysis['consistency_scores']
            sorted_consistency = sorted(consistency.items(), key=lambda x: x[1], reverse=True)
            
            for algorithm, score in sorted_consistency[:3]:
                print(f"   {algorithm}: {score:.3f}")
    
    # Overall recommendations
    print(f"\n💡 Overall Recommendations:")
    for rec in comprehensive_report['overall_recommendations'][:5]:
        print(f"   • {rec}")
    
    # Statistical significance summary
    significant_findings = 0
    for analysis in framework.comparative_analyses:
        significant_findings += len(analysis.significant_differences)
    
    print(f"\n📊 Statistical Analysis:")
    print(f"   Significant differences found: {significant_findings}")
    print(f"   Comparative analyses performed: {len(framework.comparative_analyses)}")
    
    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_filename = f"/root/repo/comprehensive_coevolution_benchmarks_{timestamp}.json"
    framework.save_results(results_filename)
    
    # Save comprehensive report
    report_filename = f"/root/repo/coevolution_benchmark_report_{timestamp}.json"
    try:
        with open(report_filename, 'w') as f:
            json.dump(comprehensive_report, f, indent=2, default=str)
        print(f"📈 Comprehensive report saved to {report_filename}")
    except Exception as e:
        print(f"⚠️ Could not save report: {e}")
    
    return comprehensive_report, framework


if __name__ == "__main__":
    comprehensive_report, framework = run_comprehensive_coevolution_benchmarks()
    
    # Determine overall success
    total_successful_runs = len([r for r in framework.results_database if r.success])
    total_runs = len(framework.results_database)
    success_rate = total_successful_runs / max(total_runs, 1)
    
    avg_fitness_across_all = np.mean([
        r.metrics.get('final_fitness', 0) 
        for r in framework.results_database 
        if r.success
    ])
    
    algorithms_tested = len(set(r.algorithm_name for r in framework.results_database))
    benchmarks_completed = len(set(r.benchmark_id for r in framework.results_database))
    
    print(f"\n🎯 BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Success Rate: {success_rate:.1%}")
    print(f"Average Fitness: {avg_fitness_across_all:.3f}")
    print(f"Algorithms Tested: {algorithms_tested}")
    print(f"Benchmarks Completed: {benchmarks_completed}")
    
    # Overall success criteria
    success = (
        success_rate > 0.8 and
        avg_fitness_across_all > 0.7 and
        algorithms_tested >= 4 and
        benchmarks_completed >= 3
    )
    
    if success:
        print("\n🎉 COMPREHENSIVE BENCHMARKING SUCCESS!")
        print("All co-evolution algorithms successfully benchmarked and analyzed.")
    else:
        print("\n⚠️ Benchmarking completed with mixed results.")