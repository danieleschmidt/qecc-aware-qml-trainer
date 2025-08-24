"""
Performance benchmarking suite for QECC-aware QML.

Comprehensive evaluation of quantum machine learning models with and without
error correction across various metrics and datasets.

Author: Terragon Labs SDLC System
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
from typing import Dict, List, Tuple, Optional, Any, Callable
import time
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
import pandas as pd
from dataclasses import dataclass, field
from collections import defaultdict
import warnings

try:
    try:
    import matplotlib.pyplot as plt
except ImportError:
    class MockPlt:
        def figure(self, *args, **kwargs): return None
        def plot(self, *args, **kwargs): return None
        def show(self): pass
        def savefig(self, *args, **kwargs): pass
    plt = MockPlt()
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    warnings.warn("Matplotlib/Seaborn not available. Plotting disabled.")

from ..datasets import EnhancedQuantumDatasets
from ..training.qecc_trainer import QECCTrainer
from ..core.quantum_nn import QECCAwareQNN
from ..codes import SurfaceCode, SteaneCode, ColorCode
from ..backends import EnhancedQuantumBackendManager


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    experiment_name: str
    model_name: str
    dataset_name: str
    qecc_enabled: bool
    qecc_code: Optional[str] = None
    
    # Performance metrics
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    
    # Quantum-specific metrics
    fidelity: float = 0.0
    logical_error_rate: float = 0.0
    physical_error_rate: float = 0.0
    circuit_depth: int = 0
    
    # Training metrics
    training_time: float = 0.0
    convergence_epochs: int = 0
    final_loss: float = 0.0
    
    # Resource metrics
    num_qubits: int = 0
    num_parameters: int = 0
    num_shots: int = 1024
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


class PerformanceBenchmark:
    """
    Comprehensive performance benchmarking for QECC-aware QML.
    
    Features:
    - Multi-dataset evaluation
    - QECC vs non-QECC comparison
    - Multiple quantum error correction codes
    - Noise-level scaling studies
    - Resource utilization analysis
    - Statistical significance testing
    """
    
    def __init__(self, random_state: int = 42):
        """Initialize performance benchmark."""
        self.random_state = random_state
        self.datasets = EnhancedQuantumDatasets(random_state=random_state)
        self.backend_manager = EnhancedQuantumBackendManager(enable_cloud=False)
        self.results: List[BenchmarkResult] = []
        
        # Default benchmark configuration
        self.default_config = {
            'epochs': 50,
            'batch_size': 32,
            'learning_rate': 0.01,
            'shots': 1024,
            'num_trials': 5,  # For statistical significance
            'timeout': 3600,  # 1 hour timeout per experiment
        }
    
    def run_comprehensive_benchmark(
        self,
        models: Optional[List[Dict[str, Any]]] = None,
        datasets: Optional[List[str]] = None,
        qecc_codes: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Run comprehensive benchmark across models, datasets, and QECC codes.
        
        Args:
            models: List of model configurations
            datasets: List of dataset names to benchmark
            qecc_codes: List of QECC codes to test
            config: Benchmark configuration
            
        Returns:
            DataFrame with benchmark results
        """
        # Set defaults
        if models is None:
            models = self._get_default_models()
        if datasets is None:
            datasets = ['quantum_iris', 'quantum_wine', 'quantum_moons', 'quantum_circles']
        if qecc_codes is None:
            qecc_codes = ['none', 'steane', 'surface_d3']
        if config is None:
            config = self.default_config.copy()
        
        self.results = []
        total_experiments = len(models) * len(datasets) * len(qecc_codes)
        
        print(f"Running {total_experiments} benchmark experiments...")
        
        for model_config in models:
            for dataset_name in datasets:
                for qecc_code in qecc_codes:
                    try:
                        result = self._run_single_benchmark(
                            model_config=model_config,
                            dataset_name=dataset_name,
                            qecc_code=qecc_code,
                            config=config
                        )
                        self.results.append(result)
                        
                        print(f"✓ {model_config['name']} + {dataset_name} + {qecc_code}: "
                              f"Acc={result.accuracy:.3f}, Fidelity={result.fidelity:.3f}")
                        
                    except Exception as e:
                        print(f"✗ Failed: {model_config['name']} + {dataset_name} + {qecc_code}: {e}")
                        continue
        
        return self.get_results_dataframe()
    
    def _run_single_benchmark(
        self,
        model_config: Dict[str, Any],
        dataset_name: str,
        qecc_code: str,
        config: Dict[str, Any]
    ) -> BenchmarkResult:
        """Run a single benchmark experiment."""
        # Load dataset
        X, y = self._load_dataset(dataset_name)
        
        # Split train/test
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Create model
        qnn = self._create_model(model_config, X_train.shape[1])
        
        # Add QECC if specified
        if qecc_code != 'none':
            error_correction = self._create_qecc(qecc_code)
            qnn.add_error_correction(error_correction)
        
        # Create trainer
        backend = self.backend_manager.get_backend(
            provider='simulator',
            noise_model='realistic' if qecc_code != 'none' else 'noiseless'
        )
        
        trainer = QECCTrainer(
            qnn=qnn,
            backend=backend.backend,
            shots=config['shots'],
            learning_rate=config['learning_rate']
        )
        
        # Train model
        start_time = time.time()
        history = trainer.fit(
            X_train, y_train,
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            validation_split=0.2,
            track_fidelity=True
        )
        training_time = time.time() - start_time
        
        # Evaluate model
        predictions = trainer.predict(X_test)
        metrics = self._compute_metrics(y_test, predictions)
        
        # Create result
        result = BenchmarkResult(
            experiment_name=f"{model_config['name']}_{dataset_name}_{qecc_code}",
            model_name=model_config['name'],
            dataset_name=dataset_name,
            qecc_enabled=(qecc_code != 'none'),
            qecc_code=qecc_code if qecc_code != 'none' else None,
            accuracy=metrics['accuracy'],
            precision=metrics['precision'],
            recall=metrics['recall'],
            f1_score=metrics['f1_score'],
            fidelity=history.get('fidelity', [0])[-1] if history.get('fidelity') else 0,
            logical_error_rate=history.get('logical_error_rate', [0])[-1] if history.get('logical_error_rate') else 0,
            circuit_depth=qnn.get_circuit_depth() if hasattr(qnn, 'get_circuit_depth') else 0,
            training_time=training_time,
            convergence_epochs=len(history.get('loss', [])),
            final_loss=history.get('loss', [float('inf')])[-1] if history.get('loss') else float('inf'),
            num_qubits=qnn.num_qubits,
            num_parameters=qnn.get_num_parameters() if hasattr(qnn, 'get_num_parameters') else 0,
            num_shots=config['shots'],
            metadata={
                'dataset_size': len(X),
                'features': X.shape[1],
                'classes': len(np.unique(y)),
                'backend': backend.name
            }
        )
        
        return result
    
    def _get_default_models(self) -> List[Dict[str, Any]]:
        """Get default model configurations."""
        return [
            {
                'name': 'VQC_2layer',
                'type': 'variational',
                'num_layers': 2,
                'entanglement': 'circular',
                'feature_map': 'angle_encoding'
            },
            {
                'name': 'VQC_4layer',
                'type': 'variational',
                'num_layers': 4,
                'entanglement': 'circular',
                'feature_map': 'angle_encoding'
            },
            {
                'name': 'QAOA_style',
                'type': 'qaoa',
                'num_layers': 3,
                'entanglement': 'full',
                'feature_map': 'amplitude_encoding'
            }
        ]
    
    def _load_dataset(self, dataset_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load dataset by name."""
        if dataset_name == 'quantum_iris':
            return self.datasets.load_quantum_iris()
        elif dataset_name == 'quantum_wine':
            return self.datasets.load_quantum_wine()
        elif dataset_name == 'quantum_moons':
            return self.datasets.load_quantum_moons()
        elif dataset_name == 'quantum_circles':
            return self.datasets.load_quantum_circles()
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    
    def _create_model(self, model_config: Dict[str, Any], num_features: int) -> QECCAwareQNN:
        """Create quantum neural network from configuration."""
        # Determine number of qubits (must be at least num_features)
        num_qubits = max(num_features, 2)
        
        qnn = QECCAwareQNN(
            num_qubits=num_qubits,
            num_layers=model_config.get('num_layers', 2),
            entanglement=model_config.get('entanglement', 'circular'),
            feature_map=model_config.get('feature_map', 'angle_encoding')
        )
        
        return qnn
    
    def _create_qecc(self, qecc_code: str):
        """Create error correction code from string."""
        if qecc_code == 'steane':
            return SteaneCode()
        elif qecc_code == 'surface_d3':
            return SurfaceCode(distance=3)
        elif qecc_code == 'surface_d5':
            return SurfaceCode(distance=5)
        elif qecc_code == 'color_d3':
            return ColorCode(distance=3)
        else:
            raise ValueError(f"Unknown QECC code: {qecc_code}")
    
    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Compute classification metrics."""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        # Handle binary classification
        average = 'binary' if len(np.unique(y_true)) == 2 else 'weighted'
        
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
            'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average=average, zero_division=0)
        }
    
    def get_results_dataframe(self) -> pd.DataFrame:
        """Convert results to pandas DataFrame."""
        data = []
        for result in self.results:
            row = {
                'experiment_name': result.experiment_name,
                'model': result.model_name,
                'dataset': result.dataset_name,
                'qecc_enabled': result.qecc_enabled,
                'qecc_code': result.qecc_code,
                'accuracy': result.accuracy,
                'precision': result.precision,
                'recall': result.recall,
                'f1_score': result.f1_score,
                'fidelity': result.fidelity,
                'logical_error_rate': result.logical_error_rate,
                'circuit_depth': result.circuit_depth,
                'training_time': result.training_time,
                'convergence_epochs': result.convergence_epochs,
                'final_loss': result.final_loss,
                'num_qubits': result.num_qubits,
                'num_parameters': result.num_parameters,
                'num_shots': result.num_shots
            }
            data.append(row)
        
        return pd.DataFrame(data)
    
    def analyze_qecc_benefit(self) -> Dict[str, Any]:
        """Analyze benefit of QECC across experiments."""
        df = self.get_results_dataframe()
        
        analysis = {}
        
        # Group by model and dataset
        for (model, dataset), group in df.groupby(['model', 'dataset']):
            qecc_results = group[group['qecc_enabled'] == True]
            no_qecc_results = group[group['qecc_enabled'] == False]
            
            if len(qecc_results) > 0 and len(no_qecc_results) > 0:
                qecc_acc = qecc_results['accuracy'].mean()
                no_qecc_acc = no_qecc_results['accuracy'].mean()
                improvement = (qecc_acc - no_qecc_acc) / no_qecc_acc * 100
                
                analysis[f"{model}_{dataset}"] = {
                    'qecc_accuracy': qecc_acc,
                    'no_qecc_accuracy': no_qecc_acc,
                    'improvement_percent': improvement,
                    'fidelity_gain': qecc_results['fidelity'].mean() - no_qecc_results['fidelity'].mean(),
                    'overhead': qecc_results['training_time'].mean() / no_qecc_results['training_time'].mean()
                }
        
        return analysis
    
    def plot_benchmark_results(self, save_path: Optional[str] = None) -> None:
        """Plot comprehensive benchmark results."""
        if not PLOTTING_AVAILABLE:
            warnings.warn("Plotting not available. Install matplotlib and seaborn.")
            return
        
        df = self.get_results_dataframe()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Accuracy comparison
        sns.boxplot(data=df, x='dataset', y='accuracy', hue='qecc_enabled', ax=axes[0, 0])
        axes[0, 0].set_title('Accuracy: QECC vs No QECC')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Fidelity vs Accuracy
        sns.scatterplot(data=df, x='fidelity', y='accuracy', hue='qecc_enabled', 
                       style='model', s=100, ax=axes[0, 1])
        axes[0, 1].set_title('Fidelity vs Accuracy')
        
        # Training time overhead
        sns.boxplot(data=df, x='model', y='training_time', hue='qecc_enabled', ax=axes[1, 0])
        axes[1, 0].set_title('Training Time Overhead')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Error rates
        qecc_df = df[df['qecc_enabled'] == True]
        if len(qecc_df) > 0:
            sns.scatterplot(data=qecc_df, x='logical_error_rate', y='accuracy', 
                           hue='qecc_code', s=100, ax=axes[1, 1])
            axes[1, 1].set_title('Logical Error Rate vs Accuracy')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_benchmark_report(self) -> str:
        """Generate comprehensive benchmark report."""
        df = self.get_results_dataframe()
        analysis = self.analyze_qecc_benefit()
        
        report = []
        report.append("# QECC-Aware QML Benchmark Report")
        report.append(f"## Overview")
        report.append(f"- Total experiments: {len(self.results)}")
        report.append(f"- Models tested: {df['model'].nunique()}")
        report.append(f"- Datasets tested: {df['dataset'].nunique()}")
        report.append(f"- QECC codes tested: {df['qecc_code'].nunique()}")
        
        report.append("\\n## Summary Statistics")
        report.append(df.groupby('qecc_enabled')[['accuracy', 'fidelity', 'training_time']].describe().to_string())
        
        report.append("\\n## QECC Benefit Analysis")
        for exp_name, results in analysis.items():
            report.append(f"### {exp_name}")
            report.append(f"- Accuracy improvement: {results['improvement_percent']:.1f}%")
            report.append(f"- Fidelity gain: {results['fidelity_gain']:.3f}")
            report.append(f"- Training overhead: {results['overhead']:.1f}x")
        
        report.append("\\n## Best Performing Configurations")
        best_configs = df.nlargest(5, 'accuracy')[['model', 'dataset', 'qecc_code', 'accuracy', 'fidelity']]
        report.append(best_configs.to_string(index=False))
        
        return "\\n".join(report)
    
    def run_noise_scaling_study(
        self,
        model_config: Dict[str, Any],
        dataset_name: str,
        noise_levels: List[float] = None
    ) -> pd.DataFrame:
        """Run scaling study across noise levels."""
        if noise_levels is None:
            noise_levels = np.logspace(-4, -1, 8)  # 0.0001 to 0.1
        
        scaling_results = []
        
        for noise_level in noise_levels:
            # Run with and without QECC
            for qecc_code in ['none', 'steane']:
                try:
                    # Modify backend to use specific noise level
                    config = self.default_config.copy()
                    config['noise_level'] = noise_level
                    
                    result = self._run_single_benchmark(
                        model_config=model_config,
                        dataset_name=dataset_name,
                        qecc_code=qecc_code,
                        config=config
                    )
                    result.metadata['noise_level'] = noise_level
                    scaling_results.append(result)
                    
                except Exception as e:
                    print(f"Failed at noise level {noise_level}: {e}")
                    continue
        
        # Convert to DataFrame
        data = []
        for result in scaling_results:
            data.append({
                'noise_level': result.metadata['noise_level'],
                'qecc_enabled': result.qecc_enabled,
                'accuracy': result.accuracy,
                'fidelity': result.fidelity,
                'logical_error_rate': result.logical_error_rate
            })
        
        return pd.DataFrame(data)
    
    def __str__(self) -> str:
        """String representation."""
        return f"PerformanceBenchmark({len(self.results)} results)"
    
    def __repr__(self) -> str:
        """Detailed representation."""
        return (f"PerformanceBenchmark(results={len(self.results)}, "
                f"datasets={self.datasets}, backend_manager={self.backend_manager})")