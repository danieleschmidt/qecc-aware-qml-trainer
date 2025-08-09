"""
Advanced benchmarking suite for QECC-aware QML systems.

Provides comprehensive benchmarks for comparative studies, scalability analysis,
and performance validation across different quantum hardware platforms.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time
import logging
import json
import multiprocessing as mp
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from ..core.quantum_nn import QECCAwareQNN
from ..core.noise_models import NoiseModel
from ..training.qecc_trainer import QECCTrainer
from ..codes.surface_code import SurfaceCode
from ..codes.color_code import ColorCode
from ..codes.steane_code import SteaneCode
from ..adaptive.adaptive_qecc import AdaptiveQECC, QECCSelectionStrategy


class BenchmarkType(Enum):
    """Types of benchmarks."""
    NOISE_RESILIENCE = "noise_resilience"
    SCALABILITY = "scalability"
    HARDWARE_COMPARISON = "hardware_comparison"
    CODE_COMPARISON = "code_comparison"
    ADAPTIVE_PERFORMANCE = "adaptive_performance"
    TRAINING_EFFICIENCY = "training_efficiency"
    RESOURCE_UTILIZATION = "resource_utilization"


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""
    name: str
    benchmark_type: BenchmarkType
    datasets: List[str] = field(default_factory=list)
    noise_levels: List[float] = field(default_factory=list)
    qubit_counts: List[int] = field(default_factory=list)
    code_distances: List[int] = field(default_factory=list)
    repetitions: int = 5
    timeout_seconds: int = 3600
    parallel: bool = True
    max_workers: Optional[int] = None
    save_results: bool = True
    output_dir: str = "benchmark_results"
    metrics: List[str] = field(default_factory=lambda: ["accuracy", "fidelity", "error_rate", "training_time"])


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    config_id: str
    benchmark_type: BenchmarkType
    parameters: Dict[str, Any]
    metrics: Dict[str, float]
    execution_time: float
    success: bool
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class AdvancedBenchmarkSuite:
    """
    Comprehensive benchmarking suite for QECC-aware QML systems.
    
    Provides automated benchmarking across multiple dimensions including
    noise resilience, scalability, hardware platforms, and code comparisons.
    """
    
    def __init__(
        self,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize benchmark suite.
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        
        # Benchmark configurations
        self.benchmark_configs: Dict[str, BenchmarkConfig] = {}
        
        # Results storage
        self.results: List[BenchmarkResult] = []
        self.summary_stats: Dict[str, Any] = {}
        
        # Execution control
        self.is_running = False
        self.current_benchmark = None
        
        # Default configurations
        self._initialize_default_benchmarks()
    
    def _initialize_default_benchmarks(self):
        """Initialize default benchmark configurations."""
        # Noise resilience benchmark
        self.add_benchmark_config(BenchmarkConfig(
            name="noise_resilience_sweep",
            benchmark_type=BenchmarkType.NOISE_RESILIENCE,
            noise_levels=np.logspace(-4, -1, 10).tolist(),
            datasets=["synthetic_4qubit", "quantum_mnist"],
            repetitions=3,
            metrics=["accuracy", "fidelity", "logical_error_rate"]
        ))
        
        # Scalability benchmark
        self.add_benchmark_config(BenchmarkConfig(
            name="qubit_scalability",
            benchmark_type=BenchmarkType.SCALABILITY,
            qubit_counts=[4, 6, 8, 10, 12, 16],
            code_distances=[3, 5, 7],
            repetitions=2,
            metrics=["accuracy", "training_time", "memory_usage", "circuit_depth"]
        ))
        
        # Code comparison benchmark
        self.add_benchmark_config(BenchmarkConfig(
            name="qecc_code_comparison",
            benchmark_type=BenchmarkType.CODE_COMPARISON,
            noise_levels=[0.001, 0.005, 0.01],
            code_distances=[3, 5],
            repetitions=5,
            metrics=["accuracy", "fidelity", "logical_error_rate", "overhead"]
        ))
        
        # Adaptive performance benchmark
        self.add_benchmark_config(BenchmarkConfig(
            name="adaptive_qecc_performance",
            benchmark_type=BenchmarkType.ADAPTIVE_PERFORMANCE,
            noise_levels=np.logspace(-3, -1, 8).tolist(),
            repetitions=3,
            metrics=["accuracy", "adaptation_count", "final_code_distance", "training_time"]
        ))
    
    def add_benchmark_config(self, config: BenchmarkConfig):
        """Add a benchmark configuration."""
        self.benchmark_configs[config.name] = config
        self.logger.info(f"Added benchmark config: {config.name}")
    
    def run_benchmark(self, config_name: str) -> List[BenchmarkResult]:
        """
        Run a specific benchmark.
        
        Args:
            config_name: Name of benchmark configuration
            
        Returns:
            List of benchmark results
        """
        if config_name not in self.benchmark_configs:
            raise ValueError(f"Unknown benchmark config: {config_name}")
        
        config = self.benchmark_configs[config_name]
        self.current_benchmark = config_name
        self.is_running = True
        
        self.logger.info(f"Starting benchmark: {config_name}")
        
        try:
            if config.benchmark_type == BenchmarkType.NOISE_RESILIENCE:
                results = self._run_noise_resilience_benchmark(config)
            elif config.benchmark_type == BenchmarkType.SCALABILITY:
                results = self._run_scalability_benchmark(config)
            elif config.benchmark_type == BenchmarkType.CODE_COMPARISON:
                results = self._run_code_comparison_benchmark(config)
            elif config.benchmark_type == BenchmarkType.ADAPTIVE_PERFORMANCE:
                results = self._run_adaptive_performance_benchmark(config)
            elif config.benchmark_type == BenchmarkType.HARDWARE_COMPARISON:
                results = self._run_hardware_comparison_benchmark(config)
            elif config.benchmark_type == BenchmarkType.TRAINING_EFFICIENCY:
                results = self._run_training_efficiency_benchmark(config)
            else:
                raise NotImplementedError(f"Benchmark type {config.benchmark_type} not implemented")
            
            self.results.extend(results)
            
            # Save results if requested
            if config.save_results:
                self._save_benchmark_results(config_name, results)
            
            self.logger.info(f"Completed benchmark: {config_name} ({len(results)} results)")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Benchmark {config_name} failed: {e}")
            raise
        finally:
            self.is_running = False
            self.current_benchmark = None
    
    def _run_noise_resilience_benchmark(self, config: BenchmarkConfig) -> List[BenchmarkResult]:
        """Run noise resilience benchmark."""
        results = []
        
        # Generate test cases
        test_cases = []
        for noise_level in config.noise_levels:
            for dataset in config.datasets:
                for rep in range(config.repetitions):
                    test_cases.append({
                        'noise_level': noise_level,
                        'dataset': dataset,
                        'repetition': rep,
                        'config_id': f"noise_{noise_level:.4f}_{dataset}_{rep}"
                    })
        
        self.logger.info(f"Running {len(test_cases)} noise resilience test cases")
        
        # Execute test cases
        if config.parallel:
            results = self._run_parallel_benchmark(test_cases, self._execute_noise_resilience_case, config)
        else:
            results = self._run_sequential_benchmark(test_cases, self._execute_noise_resilience_case, config)
        
        return results
    
    def _execute_noise_resilience_case(self, case: Dict[str, Any], config: BenchmarkConfig) -> BenchmarkResult:
        """Execute single noise resilience test case."""
        start_time = time.time()
        
        try:
            # Create noise model
            noise_model = NoiseModel(
                gate_error_rate=case['noise_level'],
                readout_error_rate=case['noise_level'] * 10,
                T1=50e-6 / (1 + case['noise_level'] * 100),
                T2=70e-6 / (1 + case['noise_level'] * 100)
            )
            
            # Create QNN with and without error correction
            qnn_baseline = QECCAwareQNN(num_qubits=4, num_layers=2)
            qnn_protected = QECCAwareQNN(num_qubits=4, num_layers=2)
            
            # Add surface code protection
            surface_code = SurfaceCode(distance=3)
            qnn_protected.add_error_correction(surface_code)
            
            # Generate synthetic dataset
            X_train, y_train, X_test, y_test = self._generate_dataset(case['dataset'])
            
            # Train baseline model
            trainer_baseline = QECCTrainer(
                qnn=qnn_baseline,
                noise_model=noise_model,
                learning_rate=0.05,
                shots=256
            )
            
            trainer_baseline.fit(
                X_train, y_train,
                epochs=10,
                batch_size=8,
                verbose=False
            )
            
            baseline_results = trainer_baseline.evaluate(X_test, y_test)
            
            # Train protected model
            trainer_protected = QECCTrainer(
                qnn=qnn_protected,
                noise_model=noise_model,
                learning_rate=0.05,
                shots=256
            )
            
            trainer_protected.fit(
                X_train, y_train,
                epochs=10,
                batch_size=8,
                verbose=False
            )
            
            protected_results = trainer_protected.evaluate(X_test, y_test)
            
            # Calculate metrics
            metrics = {
                'baseline_accuracy': baseline_results.get('accuracy', 0.0),
                'protected_accuracy': protected_results.get('accuracy', 0.0),
                'accuracy_improvement': (protected_results.get('accuracy', 0.0) - baseline_results.get('accuracy', 0.0)),
                'baseline_fidelity': baseline_results.get('fidelity', 0.0),
                'protected_fidelity': protected_results.get('fidelity', 0.0),
                'logical_error_rate': protected_results.get('logical_error_rate', 0.0),
                'noise_level': case['noise_level']
            }
            
            execution_time = time.time() - start_time
            
            return BenchmarkResult(
                config_id=case['config_id'],
                benchmark_type=config.benchmark_type,
                parameters=case,
                metrics=metrics,
                execution_time=execution_time,
                success=True
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            return BenchmarkResult(
                config_id=case['config_id'],
                benchmark_type=config.benchmark_type,
                parameters=case,
                metrics={},
                execution_time=execution_time,
                success=False,
                error_message=str(e)
            )
    
    def _run_scalability_benchmark(self, config: BenchmarkConfig) -> List[BenchmarkResult]:
        """Run scalability benchmark."""
        results = []
        
        # Generate test cases
        test_cases = []
        for qubit_count in config.qubit_counts:
            for distance in config.code_distances:
                for rep in range(config.repetitions):
                    test_cases.append({
                        'qubit_count': qubit_count,
                        'code_distance': distance,
                        'repetition': rep,
                        'config_id': f"scale_{qubit_count}q_d{distance}_{rep}"
                    })
        
        self.logger.info(f"Running {len(test_cases)} scalability test cases")
        
        # Execute test cases
        if config.parallel:
            results = self._run_parallel_benchmark(test_cases, self._execute_scalability_case, config)
        else:
            results = self._run_sequential_benchmark(test_cases, self._execute_scalability_case, config)
        
        return results
    
    def _execute_scalability_case(self, case: Dict[str, Any], config: BenchmarkConfig) -> BenchmarkResult:
        """Execute single scalability test case."""
        start_time = time.time()
        
        try:
            # Create QNN
            qnn = QECCAwareQNN(
                num_qubits=case['qubit_count'],
                num_layers=max(2, case['qubit_count'] // 2)
            )
            
            # Add error correction
            surface_code = SurfaceCode(distance=case['code_distance'])
            qnn.add_error_correction(surface_code)
            
            # Generate dataset
            X_train, y_train, X_test, y_test = self._generate_dataset("synthetic", num_features=case['qubit_count'])
            
            # Measure memory usage (simplified)
            circuit_depth = qnn.get_circuit_depth()
            physical_qubits = qnn.num_physical_qubits
            parameters = qnn.get_num_parameters()
            
            # Quick training for timing
            noise_model = NoiseModel(gate_error_rate=0.001)
            trainer = QECCTrainer(qnn=qnn, noise_model=noise_model, shots=128)
            
            training_start = time.time()
            trainer.fit(X_train, y_train, epochs=5, batch_size=4, verbose=False)
            training_time = time.time() - training_start
            
            results = trainer.evaluate(X_test, y_test)
            
            metrics = {
                'accuracy': results.get('accuracy', 0.0),
                'training_time': training_time,
                'circuit_depth': circuit_depth,
                'physical_qubits': physical_qubits,
                'parameters': parameters,
                'memory_usage': physical_qubits * circuit_depth,  # Simplified metric
                'overhead': physical_qubits / case['qubit_count']
            }
            
            execution_time = time.time() - start_time
            
            return BenchmarkResult(
                config_id=case['config_id'],
                benchmark_type=config.benchmark_type,
                parameters=case,
                metrics=metrics,
                execution_time=execution_time,
                success=True
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            return BenchmarkResult(
                config_id=case['config_id'],
                benchmark_type=config.benchmark_type,
                parameters=case,
                metrics={},
                execution_time=execution_time,
                success=False,
                error_message=str(e)
            )
    
    def _run_code_comparison_benchmark(self, config: BenchmarkConfig) -> List[BenchmarkResult]:
        """Run QECC code comparison benchmark."""
        results = []
        
        # Define codes to compare
        codes = {
            'surface_code': lambda d: SurfaceCode(distance=d),
            'color_code': lambda d: ColorCode(distance=d),
            'steane_code': lambda d: SteaneCode() if d == 3 else None  # Steane code is fixed distance 3
        }
        
        # Generate test cases
        test_cases = []
        for noise_level in config.noise_levels:
            for distance in config.code_distances:
                for code_name, code_factory in codes.items():
                    code = code_factory(distance)
                    if code is None:  # Skip invalid combinations
                        continue
                        
                    for rep in range(config.repetitions):
                        test_cases.append({
                            'noise_level': noise_level,
                            'code_distance': distance,
                            'code_name': code_name,
                            'code': code,
                            'repetition': rep,
                            'config_id': f"code_{code_name}_d{distance}_n{noise_level:.3f}_{rep}"
                        })
        
        self.logger.info(f"Running {len(test_cases)} code comparison test cases")
        
        # Execute test cases
        if config.parallel:
            results = self._run_parallel_benchmark(test_cases, self._execute_code_comparison_case, config)
        else:
            results = self._run_sequential_benchmark(test_cases, self._execute_code_comparison_case, config)
        
        return results
    
    def _execute_code_comparison_case(self, case: Dict[str, Any], config: BenchmarkConfig) -> BenchmarkResult:
        """Execute single code comparison test case."""
        start_time = time.time()
        
        try:
            # Create QNN with specific code
            qnn = QECCAwareQNN(num_qubits=4, num_layers=2)
            qnn.add_error_correction(case['code'])
            
            # Create noise model
            noise_model = NoiseModel(
                gate_error_rate=case['noise_level'],
                readout_error_rate=case['noise_level'] * 5,
                T1=50e-6,
                T2=70e-6
            )
            
            # Generate dataset
            X_train, y_train, X_test, y_test = self._generate_dataset("synthetic_4qubit")
            
            # Train model
            trainer = QECCTrainer(
                qnn=qnn,
                noise_model=noise_model,
                learning_rate=0.05,
                shots=256
            )
            
            trainer.fit(
                X_train, y_train,
                epochs=15,
                batch_size=8,
                verbose=False
            )
            
            results = trainer.evaluate(X_test, y_test)
            
            # Calculate code-specific metrics
            physical_qubits = qnn.num_physical_qubits
            logical_qubits = 4  # We use 4 logical qubits
            overhead = physical_qubits / logical_qubits
            
            metrics = {
                'accuracy': results.get('accuracy', 0.0),
                'fidelity': results.get('fidelity', 0.0),
                'logical_error_rate': results.get('logical_error_rate', 0.0),
                'physical_qubits': physical_qubits,
                'overhead': overhead,
                'code_distance': case['code_distance'],
                'noise_level': case['noise_level']
            }
            
            execution_time = time.time() - start_time
            
            return BenchmarkResult(
                config_id=case['config_id'],
                benchmark_type=config.benchmark_type,
                parameters=case,
                metrics=metrics,
                execution_time=execution_time,
                success=True,
                metadata={'code_name': case['code_name']}
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            return BenchmarkResult(
                config_id=case['config_id'],
                benchmark_type=config.benchmark_type,
                parameters=case,
                metrics={},
                execution_time=execution_time,
                success=False,
                error_message=str(e),
                metadata={'code_name': case['code_name']}
            )
    
    def _run_adaptive_performance_benchmark(self, config: BenchmarkConfig) -> List[BenchmarkResult]:
        """Run adaptive QECC performance benchmark."""
        results = []
        
        # Generate test cases with different adaptation strategies
        strategies = [
            QECCSelectionStrategy.THRESHOLD_BASED,
            QECCSelectionStrategy.PERFORMANCE_OPTIMIZED,
            QECCSelectionStrategy.HYBRID_ADAPTIVE
        ]
        
        test_cases = []
        for noise_level in config.noise_levels:
            for strategy in strategies:
                for rep in range(config.repetitions):
                    test_cases.append({
                        'noise_level': noise_level,
                        'strategy': strategy,
                        'repetition': rep,
                        'config_id': f"adaptive_{strategy.value}_n{noise_level:.3f}_{rep}"
                    })
        
        self.logger.info(f"Running {len(test_cases)} adaptive performance test cases")
        
        # Execute test cases
        if config.parallel:
            results = self._run_parallel_benchmark(test_cases, self._execute_adaptive_case, config)
        else:
            results = self._run_sequential_benchmark(test_cases, self._execute_adaptive_case, config)
        
        return results
    
    def _execute_adaptive_case(self, case: Dict[str, Any], config: BenchmarkConfig) -> BenchmarkResult:
        """Execute single adaptive performance test case."""
        start_time = time.time()
        
        try:
            # Create adaptive QECC system
            adaptive_qecc = AdaptiveQECC(
                strategy=case['strategy'],
                adaptation_frequency=5,  # Adapt every 5 epochs
                performance_window=20
            )
            
            # Create QNN (will be dynamically managed by adaptive system)
            qnn = QECCAwareQNN(num_qubits=4, num_layers=2)
            
            # Create varying noise model (simulates changing conditions)
            base_noise_level = case['noise_level']
            noise_model = NoiseModel(
                gate_error_rate=base_noise_level,
                readout_error_rate=base_noise_level * 5,
                T1=50e-6,
                T2=70e-6
            )
            
            # Generate dataset
            X_train, y_train, X_test, y_test = self._generate_dataset("synthetic_4qubit")
            
            # Custom trainer with adaptation
            trainer = QECCTrainer(
                qnn=qnn,
                noise_model=noise_model,
                learning_rate=0.05,
                shots=256
            )
            
            # Training with adaptation
            adaptation_count = 0
            history = {'accuracy': [], 'fidelity': [], 'logical_error_rate': []}
            
            for epoch in range(20):
                # Simulate noise variations
                current_noise = base_noise_level * (1 + 0.3 * np.sin(epoch * 0.5))
                noise_model.gate_error_rate = current_noise
                
                # Single epoch training
                trainer.fit(
                    X_train, y_train,
                    epochs=1,
                    batch_size=8,
                    verbose=False,
                    initial_epoch=epoch
                )
                
                # Evaluate and track performance
                results = trainer.evaluate(X_test, y_test)
                history['accuracy'].append(results.get('accuracy', 0.0))
                history['fidelity'].append(results.get('fidelity', 0.0))
                history['logical_error_rate'].append(results.get('logical_error_rate', 0.0))
                
                # Update adaptive system
                adaptive_qecc.update_performance(results)
                adaptive_qecc.update_noise_characteristics({'gate_error_rate': current_noise})
                
                # Check for adaptation
                if adaptive_qecc.should_adapt(results):
                    resource_constraints = {'available_qubits': 30}
                    performance_requirements = {'target_fidelity': 0.9}
                    
                    adapted = adaptive_qecc.adapt(
                        noise_model,
                        resource_constraints,
                        performance_requirements
                    )
                    
                    if adapted:
                        adaptation_count += 1
                        # Apply new scheme to QNN
                        new_scheme = adaptive_qecc.get_current_scheme()
                        if new_scheme:
                            qnn.add_error_correction(new_scheme)
            
            # Final evaluation
            final_results = trainer.evaluate(X_test, y_test)
            
            # Get adaptation statistics
            adaptation_stats = adaptive_qecc.get_adaptation_statistics()
            
            metrics = {
                'final_accuracy': final_results.get('accuracy', 0.0),
                'final_fidelity': final_results.get('fidelity', 0.0),
                'adaptation_count': adaptation_count,
                'final_code_distance': adaptation_stats.get('current_distance', 3),
                'average_accuracy': np.mean(history['accuracy']),
                'accuracy_stability': 1.0 - np.std(history['accuracy']),
                'performance_trend': adaptation_stats.get('performance_trend', 0.0)
            }
            
            execution_time = time.time() - start_time
            
            return BenchmarkResult(
                config_id=case['config_id'],
                benchmark_type=config.benchmark_type,
                parameters=case,
                metrics=metrics,
                execution_time=execution_time,
                success=True,
                metadata={'strategy': case['strategy'].value}
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            return BenchmarkResult(
                config_id=case['config_id'],
                benchmark_type=config.benchmark_type,
                parameters=case,
                metrics={},
                execution_time=execution_time,
                success=False,
                error_message=str(e),
                metadata={'strategy': case['strategy'].value}
            )
    
    def _run_hardware_comparison_benchmark(self, config: BenchmarkConfig) -> List[BenchmarkResult]:
        """Run hardware comparison benchmark (placeholder)."""
        # This would compare performance across different quantum backends
        # For now, return empty results
        self.logger.info("Hardware comparison benchmark not fully implemented")
        return []
    
    def _run_training_efficiency_benchmark(self, config: BenchmarkConfig) -> List[BenchmarkResult]:
        """Run training efficiency benchmark (placeholder)."""
        # This would benchmark training speed, convergence, and resource usage
        self.logger.info("Training efficiency benchmark not fully implemented")
        return []
    
    def _generate_dataset(self, dataset_name: str, num_features: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate synthetic dataset for benchmarking."""
        if num_features is None:
            num_features = 4
        
        from sklearn.datasets import make_classification
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        
        # Generate synthetic classification data
        X, y = make_classification(
            n_samples=200,
            n_features=num_features,
            n_informative=num_features,
            n_redundant=0,
            n_clusters_per_class=1,
            class_sep=1.0,
            random_state=42
        )
        
        # Scale features to [0, π] for quantum encoding
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        X = (X + 3) * np.pi / 6
        
        # Split into train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        return X_train, y_train, X_test, y_test
    
    def _run_parallel_benchmark(self, test_cases: List[Dict], executor_func: Callable, config: BenchmarkConfig) -> List[BenchmarkResult]:
        """Run benchmark test cases in parallel."""
        max_workers = config.max_workers or min(8, mp.cpu_count())
        
        with mp.Pool(max_workers) as pool:
            # Create tasks
            tasks = [(case, config) for case in test_cases]
            
            # Execute in parallel
            results = pool.starmap(executor_func, tasks)
        
        return results
    
    def _run_sequential_benchmark(self, test_cases: List[Dict], executor_func: Callable, config: BenchmarkConfig) -> List[BenchmarkResult]:
        """Run benchmark test cases sequentially."""
        results = []
        
        for i, case in enumerate(test_cases):
            self.logger.info(f"Executing test case {i+1}/{len(test_cases)}: {case['config_id']}")
            
            result = executor_func(case, config)
            results.append(result)
            
            # Check timeout
            if hasattr(config, 'timeout_seconds') and result.execution_time > config.timeout_seconds:
                self.logger.warning(f"Test case {case['config_id']} exceeded timeout")
        
        return results
    
    def _save_benchmark_results(self, benchmark_name: str, results: List[BenchmarkResult]):
        """Save benchmark results to files."""
        output_dir = Path("benchmark_results")
        output_dir.mkdir(exist_ok=True)
        
        # Convert results to DataFrame
        data = []
        for result in results:
            row = {
                'config_id': result.config_id,
                'benchmark_type': result.benchmark_type.value,
                'success': result.success,
                'execution_time': result.execution_time,
                'error_message': result.error_message
            }
            
            # Add parameters
            for key, value in result.parameters.items():
                row[f'param_{key}'] = value
            
            # Add metrics
            for key, value in result.metrics.items():
                row[f'metric_{key}'] = value
            
            # Add metadata
            for key, value in result.metadata.items():
                row[f'meta_{key}'] = value
            
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Save as CSV
        csv_path = output_dir / f"{benchmark_name}_results.csv"
        df.to_csv(csv_path, index=False)
        
        # Save as JSON (for metadata preservation)
        json_data = [
            {
                'config_id': r.config_id,
                'benchmark_type': r.benchmark_type.value,
                'parameters': r.parameters,
                'metrics': r.metrics,
                'execution_time': r.execution_time,
                'success': r.success,
                'error_message': r.error_message,
                'metadata': r.metadata
            }
            for r in results
        ]
        
        json_path = output_dir / f"{benchmark_name}_results.json"
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2, default=str)
        
        self.logger.info(f"Saved benchmark results: {csv_path}, {json_path}")
    
    def analyze_results(self, benchmark_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze benchmark results and generate summary statistics.
        
        Args:
            benchmark_name: Specific benchmark to analyze (None for all)
            
        Returns:
            Analysis results
        """
        # Filter results if specific benchmark requested
        if benchmark_name:
            results = [r for r in self.results if benchmark_name in r.config_id]
        else:
            results = self.results
        
        if not results:
            return {'message': 'No results to analyze'}
        
        # Convert to DataFrame for analysis
        data = []
        for result in results:
            if not result.success:
                continue
                
            row = {
                'benchmark_type': result.benchmark_type.value,
                'execution_time': result.execution_time,
                **result.parameters,
                **result.metrics,
                **result.metadata
            }
            data.append(row)
        
        if not data:
            return {'message': 'No successful results to analyze'}
        
        df = pd.DataFrame(data)
        
        # Generate summary statistics
        analysis = {
            'total_results': len(results),
            'successful_results': len(data),
            'success_rate': len(data) / len(results),
            'benchmark_types': df['benchmark_type'].value_counts().to_dict(),
            'execution_time_stats': {
                'mean': df['execution_time'].mean(),
                'median': df['execution_time'].median(),
                'std': df['execution_time'].std(),
                'min': df['execution_time'].min(),
                'max': df['execution_time'].max()
            }
        }
        
        # Metric-specific analysis
        metric_columns = [col for col in df.columns if col.startswith('metric_') or col in ['accuracy', 'fidelity', 'logical_error_rate']]
        
        for metric in metric_columns:
            if metric in df.columns and df[metric].dtype in [np.float64, np.int64]:
                analysis[f'{metric}_stats'] = {
                    'mean': df[metric].mean(),
                    'median': df[metric].median(),
                    'std': df[metric].std(),
                    'min': df[metric].min(),
                    'max': df[metric].max()
                }
        
        # Correlation analysis (if multiple metrics)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            correlation_matrix = df[numeric_cols].corr()
            analysis['correlations'] = correlation_matrix.to_dict()
        
        self.summary_stats = analysis
        return analysis
    
    def generate_report(self, output_path: Optional[str] = None) -> str:
        """
        Generate comprehensive benchmark report.
        
        Args:
            output_path: Path to save report
            
        Returns:
            Path to generated report
        """
        if output_path is None:
            output_path = "benchmark_report.html"
        
        # Analyze results if not done already
        if not self.summary_stats:
            self.analyze_results()
        
        # Generate visualizations
        self._create_benchmark_plots()
        
        # Create HTML report
        html_content = self._generate_html_report()
        
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        self.logger.info(f"Generated benchmark report: {output_path}")
        return output_path
    
    def _create_benchmark_plots(self):
        """Create visualizations for benchmark results."""
        output_dir = Path("benchmark_results")
        output_dir.mkdir(exist_ok=True)
        
        # Convert successful results to DataFrame
        data = []
        for result in self.results:
            if result.success:
                row = {
                    'benchmark_type': result.benchmark_type.value,
                    **result.parameters,
                    **result.metrics
                }
                data.append(row)
        
        if not data:
            return
        
        df = pd.DataFrame(data)
        
        # Create plots based on available data
        plt.style.use('seaborn-v0_8')
        
        # Execution time distribution
        if 'execution_time' in [r.execution_time for r in self.results]:
            plt.figure(figsize=(10, 6))
            execution_times = [r.execution_time for r in self.results if r.success]
            plt.hist(execution_times, bins=20, alpha=0.7, edgecolor='black')
            plt.title('Execution Time Distribution')
            plt.xlabel('Execution Time (seconds)')
            plt.ylabel('Frequency')
            plt.savefig(output_dir / 'execution_time_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # Noise resilience plot (if available)
        noise_results = df[df['benchmark_type'] == 'noise_resilience']
        if not noise_results.empty and 'noise_level' in noise_results.columns:
            plt.figure(figsize=(12, 8))
            
            # Plot accuracy vs noise level
            if 'baseline_accuracy' in noise_results.columns and 'protected_accuracy' in noise_results.columns:
                plt.subplot(2, 2, 1)
                plt.semilogx(noise_results['noise_level'], noise_results['baseline_accuracy'], 'b-o', label='Baseline')
                plt.semilogx(noise_results['noise_level'], noise_results['protected_accuracy'], 'r-s', label='Protected')
                plt.xlabel('Noise Level')
                plt.ylabel('Accuracy')
                plt.legend()
                plt.title('Accuracy vs Noise Level')
                plt.grid(True, alpha=0.3)
            
            # Plot fidelity vs noise level
            if 'baseline_fidelity' in noise_results.columns and 'protected_fidelity' in noise_results.columns:
                plt.subplot(2, 2, 2)
                plt.semilogx(noise_results['noise_level'], noise_results['baseline_fidelity'], 'b-o', label='Baseline')
                plt.semilogx(noise_results['noise_level'], noise_results['protected_fidelity'], 'r-s', label='Protected')
                plt.xlabel('Noise Level')
                plt.ylabel('Fidelity')
                plt.legend()
                plt.title('Fidelity vs Noise Level')
                plt.grid(True, alpha=0.3)
            
            # Plot improvement vs noise level
            if 'accuracy_improvement' in noise_results.columns:
                plt.subplot(2, 2, 3)
                plt.semilogx(noise_results['noise_level'], noise_results['accuracy_improvement'], 'g-^')
                plt.xlabel('Noise Level')
                plt.ylabel('Accuracy Improvement')
                plt.title('QECC Improvement vs Noise')
                plt.grid(True, alpha=0.3)
            
            # Plot logical error rate
            if 'logical_error_rate' in noise_results.columns:
                plt.subplot(2, 2, 4)
                plt.loglog(noise_results['noise_level'], noise_results['logical_error_rate'], 'purple', marker='d')
                plt.xlabel('Physical Error Rate')
                plt.ylabel('Logical Error Rate')
                plt.title('Error Suppression')
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'noise_resilience_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # Scalability plot (if available)
        scale_results = df[df['benchmark_type'] == 'scalability']
        if not scale_results.empty and 'qubit_count' in scale_results.columns:
            plt.figure(figsize=(12, 6))
            
            plt.subplot(1, 2, 1)
            if 'training_time' in scale_results.columns:
                for distance in scale_results['code_distance'].unique():
                    subset = scale_results[scale_results['code_distance'] == distance]
                    plt.plot(subset['qubit_count'], subset['training_time'], 'o-', label=f'Distance {distance}')
                
                plt.xlabel('Number of Logical Qubits')
                plt.ylabel('Training Time (seconds)')
                plt.title('Training Time Scaling')
                plt.legend()
                plt.grid(True, alpha=0.3)
            
            plt.subplot(1, 2, 2)
            if 'overhead' in scale_results.columns:
                for distance in scale_results['code_distance'].unique():
                    subset = scale_results[scale_results['code_distance'] == distance]
                    plt.plot(subset['qubit_count'], subset['overhead'], 's-', label=f'Distance {distance}')
                
                plt.xlabel('Number of Logical Qubits')
                plt.ylabel('Physical/Logical Qubit Ratio')
                plt.title('Resource Overhead')
                plt.legend()
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'scalability_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _generate_html_report(self) -> str:
        """Generate HTML report content."""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>QECC-Aware QML Benchmark Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                h1, h2, h3 { color: #2c3e50; }
                table { border-collapse: collapse; width: 100%; margin: 20px 0; }
                th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
                th { background-color: #f2f2f2; }
                .metric { background-color: #e8f5e8; }
                .summary { background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin: 15px 0; }
                img { max-width: 100%; height: auto; margin: 20px 0; }
            </style>
        </head>
        <body>
            <h1>🚀 QECC-Aware QML Benchmark Report</h1>
            <div class="summary">
                <h2>Executive Summary</h2>
        """
        
        # Add summary statistics
        if self.summary_stats:
            html += f"""
                <p><strong>Total Results:</strong> {self.summary_stats.get('total_results', 0)}</p>
                <p><strong>Success Rate:</strong> {self.summary_stats.get('success_rate', 0):.1%}</p>
                <p><strong>Average Execution Time:</strong> {self.summary_stats.get('execution_time_stats', {}).get('mean', 0):.2f} seconds</p>
            """
        
        html += """
            </div>
            
            <h2>Benchmark Results</h2>
        """
        
        # Add benchmark type breakdown
        if self.summary_stats and 'benchmark_types' in self.summary_stats:
            html += "<h3>Benchmark Types</h3><ul>"
            for bench_type, count in self.summary_stats['benchmark_types'].items():
                html += f"<li>{bench_type}: {count} results</li>"
            html += "</ul>"
        
        # Add visualizations
        html += """
            <h2>Performance Analysis</h2>
            <img src="benchmark_results/execution_time_distribution.png" alt="Execution Time Distribution">
            <img src="benchmark_results/noise_resilience_analysis.png" alt="Noise Resilience Analysis">
            <img src="benchmark_results/scalability_analysis.png" alt="Scalability Analysis">
            
            <h2>Detailed Results</h2>
            <p>See CSV and JSON files in the benchmark_results directory for detailed data.</p>
            
            <footer>
                <p><em>Report generated by QECC-Aware QML Benchmark Suite</em></p>
            </footer>
        </body>
        </html>
        """
        
        return html
    
    def run_all_benchmarks(self) -> Dict[str, List[BenchmarkResult]]:
        """
        Run all configured benchmarks.
        
        Returns:
            Dictionary of benchmark name to results
        """
        all_results = {}
        
        for config_name in self.benchmark_configs.keys():
            self.logger.info(f"Running benchmark suite: {config_name}")
            
            try:
                results = self.run_benchmark(config_name)
                all_results[config_name] = results
            except Exception as e:
                self.logger.error(f"Failed to run benchmark {config_name}: {e}")
                all_results[config_name] = []
        
        # Generate comprehensive report
        self.analyze_results()
        report_path = self.generate_report()
        
        self.logger.info(f"Completed all benchmarks. Report: {report_path}")
        
        return all_results
    
    def get_benchmark_summary(self) -> Dict[str, Any]:
        """Get summary of all benchmark configurations and results."""
        return {
            'configurations': {name: {
                'type': config.benchmark_type.value,
                'repetitions': config.repetitions,
                'metrics': config.metrics
            } for name, config in self.benchmark_configs.items()},
            'results_count': len(self.results),
            'successful_results': len([r for r in self.results if r.success]),
            'summary_stats': self.summary_stats
        }