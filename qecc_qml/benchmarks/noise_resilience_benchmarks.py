"""
Noise resilience benchmarking for QECC-aware QML.

Specialized benchmarks for evaluating how well quantum machine learning models
perform under various noise conditions, with and without error correction.

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
from typing import Dict, List, Tuple, Optional, Any, Union
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
from dataclasses import dataclass
import warnings
import time

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

from ..core.noise_models import NoiseModel
from ..datasets import EnhancedQuantumDatasets
from ..training.qecc_trainer import QECCTrainer
from ..core.quantum_nn import QECCAwareQNN
from ..codes import SurfaceCode, SteaneCode
from ..backends import EnhancedQuantumBackendManager


@dataclass
class NoiseResilienceResult:
    """Container for noise resilience benchmark results."""
    noise_type: str
    noise_level: float
    qecc_enabled: bool
    qecc_code: Optional[str]
    model_name: str
    dataset_name: str
    
    # Performance under noise
    accuracy: float
    fidelity: float
    logical_error_rate: float
    physical_error_rate: float
    
    # Noise-specific metrics
    threshold_exceeded: bool
    degradation_rate: float
    recovery_factor: float  # How much QECC helps
    
    # Metadata
    circuit_depth: int
    num_qubits: int
    shots: int


class NoiseResilienceBenchmark:
    """
    Comprehensive noise resilience benchmarking for QECC-aware QML.
    
    Features:
    - Multiple noise models (depolarizing, amplitude damping, etc.)
    - Noise level scaling studies
    - Error threshold identification
    - Recovery factor analysis
    - Hardware-specific noise characterization
    - Statistical analysis of noise robustness
    """
    
    def __init__(self, random_state: int = 42):
        """Initialize noise resilience benchmark."""
        self.random_state = random_state
        self.datasets = EnhancedQuantumDatasets(random_state=random_state)
        self.backend_manager = EnhancedQuantumBackendManager(enable_cloud=False)
        self.results: List[NoiseResilienceResult] = []
        
        # Noise model configurations
        self.noise_configs = {
            'depolarizing': {
                'type': 'depolarizing',
                'levels': np.logspace(-4, -1, 12),  # 0.0001 to 0.1
                'description': 'Symmetric depolarizing noise'
            },
            'amplitude_damping': {
                'type': 'amplitude_damping', 
                'levels': np.logspace(-3, -1, 10),  # 0.001 to 0.1
                'description': 'T1 relaxation / energy loss'
            },
            'phase_damping': {
                'type': 'phase_damping',
                'levels': np.logspace(-3, -1, 10),
                'description': 'T2 dephasing / phase loss'
            },
            'readout_error': {
                'type': 'readout_error',
                'levels': np.linspace(0.001, 0.1, 10),
                'description': 'Measurement errors'
            }
        }
    
    def run_comprehensive_noise_study(
        self,
        models: Optional[List[Dict[str, Any]]] = None,
        datasets: Optional[List[str]] = None,
        noise_types: Optional[List[str]] = None,
        qecc_codes: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Run comprehensive noise resilience study.
        
        Args:
            models: Model configurations to test
            datasets: Datasets to use for testing
            noise_types: Types of noise to evaluate
            qecc_codes: Error correction codes to compare
            
        Returns:
            DataFrame with detailed results
        """
        # Set defaults
        if models is None:
            models = self._get_default_models()
        if datasets is None:
            datasets = ['quantum_iris', 'quantum_moons']
        if noise_types is None:
            noise_types = ['depolarizing', 'amplitude_damping']
        if qecc_codes is None:
            qecc_codes = ['none', 'steane', 'surface_d3']
        
        self.results = []
        total_experiments = (len(models) * len(datasets) * len(noise_types) * 
                           len(qecc_codes) * len(self.noise_configs['depolarizing']['levels']))
        
        print(f"Running {total_experiments} noise resilience experiments...")
        
        experiment_count = 0
        for model_config in models:
            for dataset_name in datasets:
                for noise_type in noise_types:
                    for qecc_code in qecc_codes:
                        noise_levels = self.noise_configs[noise_type]['levels']
                        
                        for noise_level in noise_levels:
                            try:
                                result = self._run_noise_experiment(
                                    model_config=model_config,
                                    dataset_name=dataset_name,
                                    noise_type=noise_type,
                                    noise_level=noise_level,
                                    qecc_code=qecc_code
                                )
                                self.results.append(result)
                                
                                experiment_count += 1
                                if experiment_count % 50 == 0:
                                    print(f"Progress: {experiment_count}/{total_experiments} completed")
                                
                            except Exception as e:
                                print(f"Failed experiment: {e}")
                                continue
        
        print(f"Completed {len(self.results)} experiments")
        return self.get_results_dataframe()
    
    def _run_noise_experiment(
        self,
        model_config: Dict[str, Any],
        dataset_name: str,
        noise_type: str,
        noise_level: float,
        qecc_code: str
    ) -> NoiseResilienceResult:
        """Run single noise resilience experiment."""
        # Load dataset
        X, y = self._load_dataset(dataset_name)
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Create model
        qnn = self._create_model(model_config, X_train.shape[1])
        
        # Add QECC if specified
        if qecc_code != 'none':
            error_correction = self._create_qecc(qecc_code)
            qnn.add_error_correction(error_correction)
        
        # Create noise model
        noise_model = self._create_noise_model(noise_type, noise_level)
        
        # Create trainer with noise
        backend = self.backend_manager.get_backend(provider='simulator', noise_model='noiseless')
        trainer = QECCTrainer(
            qnn=qnn,
            backend=backend.backend,
            noise_model=noise_model,
            shots=1024
        )
        
        # Train model
        history = trainer.fit(
            X_train, y_train,
            epochs=30,  # Reduced for noise studies
            batch_size=16,
            validation_split=0.2,
            track_fidelity=True
        )
        
        # Evaluate
        predictions = trainer.predict(X_test)
        accuracy = np.mean(predictions == y_test)
        
        # Compute noise-specific metrics
        fidelity = history.get('fidelity', [0])[-1] if history.get('fidelity') else 0
        logical_error_rate = (history.get('logical_error_rate', [noise_level])[-1] 
                            if history.get('logical_error_rate') else noise_level)
        
        # Check if error threshold exceeded
        threshold = self._get_error_threshold(qecc_code)
        threshold_exceeded = noise_level > threshold
        
        # Compute degradation rate (compared to noiseless)
        degradation_rate = self._compute_degradation_rate(accuracy, noise_level)
        
        # Create result
        result = NoiseResilienceResult(
            noise_type=noise_type,
            noise_level=noise_level,
            qecc_enabled=(qecc_code != 'none'),
            qecc_code=qecc_code if qecc_code != 'none' else None,
            model_name=model_config['name'],
            dataset_name=dataset_name,
            accuracy=accuracy,
            fidelity=fidelity,
            logical_error_rate=logical_error_rate,
            physical_error_rate=noise_level,
            threshold_exceeded=threshold_exceeded,
            degradation_rate=degradation_rate,
            recovery_factor=0.0,  # Will be computed in analysis
            circuit_depth=qnn.get_circuit_depth() if hasattr(qnn, 'get_circuit_depth') else 0,
            num_qubits=qnn.num_qubits,
            shots=1024
        )
        
        return result
    
    def _create_noise_model(self, noise_type: str, noise_level: float) -> NoiseModel:
        """Create noise model with specified type and level."""
        if noise_type == 'depolarizing':
            return NoiseModel(
                gate_error_rate=noise_level,
                readout_error_rate=noise_level * 0.1,
                T1=100e-6,  # 100 microseconds
                T2=80e-6    # 80 microseconds
            )
        elif noise_type == 'amplitude_damping':
            # T1 process - energy relaxation
            t1 = 1.0 / noise_level  # Convert error rate to coherence time
            return NoiseModel(
                gate_error_rate=noise_level * 0.1,
                readout_error_rate=0.01,
                T1=t1 * 1e-6,  # Convert to seconds
                T2=t1 * 1.5e-6  # T2 usually longer than T1
            )
        elif noise_type == 'phase_damping':
            # T2 process - phase decoherence
            t2 = 1.0 / noise_level
            return NoiseModel(
                gate_error_rate=noise_level * 0.1,
                readout_error_rate=0.01,
                T1=100e-6,  # Fixed T1
                T2=t2 * 1e-6  # Variable T2
            )
        elif noise_type == 'readout_error':
            return NoiseModel(
                gate_error_rate=0.001,  # Low gate error
                readout_error_rate=noise_level,
                T1=100e-6,
                T2=80e-6
            )
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")
    
    def _get_error_threshold(self, qecc_code: str) -> float:
        """Get theoretical error threshold for QECC code."""
        thresholds = {
            'none': 0.0,  # No threshold without error correction
            'steane': 0.0073,  # Steane code threshold ~0.73%
            'surface_d3': 0.01,   # Surface code threshold ~1%
            'surface_d5': 0.01,
            'color_d3': 0.008     # Color code threshold ~0.8%
        }
        return thresholds.get(qecc_code, 0.0)
    
    def _compute_degradation_rate(self, accuracy: float, noise_level: float) -> float:
        """Compute performance degradation rate with noise."""
        # Assume perfect performance at zero noise
        ideal_accuracy = 1.0
        if noise_level == 0:
            return 0.0
        return (ideal_accuracy - accuracy) / noise_level
    
    def analyze_noise_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Analyze error thresholds from experimental data."""
        df = self.get_results_dataframe()
        thresholds = {}
        
        for qecc_code in df['qecc_code'].unique():
            if pd.isna(qecc_code):
                continue
                
            code_data = df[df['qecc_code'] == qecc_code]
            thresholds[qecc_code] = {}
            
            for noise_type in code_data['noise_type'].unique():
                noise_data = code_data[code_data['noise_type'] == noise_type]
                
                # Find threshold where accuracy drops below 0.5 (random guessing)
                threshold_data = noise_data[noise_data['accuracy'] < 0.5]
                if len(threshold_data) > 0:
                    threshold = threshold_data['noise_level'].min()
                else:
                    threshold = noise_data['noise_level'].max()  # No threshold found
                
                thresholds[qecc_code][noise_type] = threshold
        
        return thresholds
    
    def compute_recovery_factors(self) -> pd.DataFrame:
        """Compute how much QECC helps at each noise level."""
        df = self.get_results_dataframe()
        recovery_data = []
        
        # Group by noise type, level, model, and dataset
        group_cols = ['noise_type', 'noise_level', 'model_name', 'dataset_name']
        
        for name, group in df.groupby(group_cols):
            qecc_results = group[group['qecc_enabled'] == True]
            no_qecc_results = group[group['qecc_enabled'] == False]
            
            if len(qecc_results) > 0 and len(no_qecc_results) > 0:
                qecc_acc = qecc_results['accuracy'].mean()
                no_qecc_acc = no_qecc_results['accuracy'].mean()
                
                # Recovery factor: how much performance is recovered
                if no_qecc_acc > 0:
                    recovery_factor = (qecc_acc - no_qecc_acc) / no_qecc_acc
                else:
                    recovery_factor = 0
                
                recovery_data.append({
                    'noise_type': name[0],
                    'noise_level': name[1], 
                    'model_name': name[2],
                    'dataset_name': name[3],
                    'qecc_accuracy': qecc_acc,
                    'no_qecc_accuracy': no_qecc_acc,
                    'recovery_factor': recovery_factor,
                    'absolute_improvement': qecc_acc - no_qecc_acc
                })
        
        return pd.DataFrame(recovery_data)
    
    def plot_noise_resilience(self, save_path: Optional[str] = None) -> None:
        """Plot comprehensive noise resilience results."""
        if not PLOTTING_AVAILABLE:
            warnings.warn("Plotting not available.")
            return
        
        df = self.get_results_dataframe()
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Accuracy vs noise level
        for noise_type in df['noise_type'].unique():
            noise_data = df[df['noise_type'] == noise_type]
            
            for qecc_enabled in [True, False]:
                subset = noise_data[noise_data['qecc_enabled'] == qecc_enabled]
                if len(subset) > 0:
                    label = f"{noise_type} ({'QECC' if qecc_enabled else 'No QECC'})"
                    axes[0, 0].semilogx(subset['noise_level'], subset['accuracy'], 
                                       'o-', label=label, alpha=0.7)
        
        axes[0, 0].set_xlabel('Noise Level')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_title('Accuracy vs Noise Level')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Fidelity vs noise level
        qecc_data = df[df['qecc_enabled'] == True]
        if len(qecc_data) > 0:
            for noise_type in qecc_data['noise_type'].unique():
                subset = qecc_data[qecc_data['noise_type'] == noise_type]
                axes[0, 1].semilogx(subset['noise_level'], subset['fidelity'], 
                                   'o-', label=f"{noise_type}", alpha=0.7)
            
            axes[0, 1].set_xlabel('Noise Level')
            axes[0, 1].set_ylabel('Fidelity')
            axes[0, 1].set_title('Quantum Fidelity vs Noise Level')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        # Error threshold visualization
        thresholds = self.analyze_noise_thresholds()
        if thresholds:
            codes = list(thresholds.keys())
            noise_types = list(self.noise_configs.keys())
            
            threshold_matrix = np.zeros((len(codes), len(noise_types)))
            for i, code in enumerate(codes):
                for j, noise_type in enumerate(noise_types):
                    threshold_matrix[i, j] = thresholds[code].get(noise_type, 0)
            
            im = axes[1, 0].imshow(threshold_matrix, cmap='viridis', aspect='auto')
            axes[1, 0].set_xticks(range(len(noise_types)))
            axes[1, 0].set_xticklabels(noise_types, rotation=45)
            axes[1, 0].set_yticks(range(len(codes)))
            axes[1, 0].set_yticklabels(codes)
            axes[1, 0].set_title('Error Thresholds by Code and Noise Type')
            plt.colorbar(im, ax=axes[1, 0], label='Threshold')
        
        # Recovery factor analysis
        recovery_df = self.compute_recovery_factors()
        if len(recovery_df) > 0:
            sns.boxplot(data=recovery_df, x='noise_type', y='recovery_factor', ax=axes[1, 1])
            axes[1, 1].set_title('QECC Recovery Factor by Noise Type')
            axes[1, 1].set_ylabel('Recovery Factor')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_noise_resilience_report(self) -> str:
        """Generate comprehensive noise resilience report."""
        df = self.get_results_dataframe()
        thresholds = self.analyze_noise_thresholds()
        recovery_df = self.compute_recovery_factors()
        
        report = []
        report.append("# Noise Resilience Benchmark Report")
        
        report.append("## Overview")
        report.append(f"- Total noise experiments: {len(df)}")
        report.append(f"- Noise types tested: {df['noise_type'].nunique()}")
        report.append(f"- QECC codes evaluated: {df['qecc_code'].nunique()}")
        report.append(f"- Noise levels per type: {len(self.noise_configs['depolarizing']['levels'])}")
        
        report.append("\\n## Error Thresholds")
        for code, threshold_data in thresholds.items():
            report.append(f"### {code}")
            for noise_type, threshold in threshold_data.items():
                report.append(f"- {noise_type}: {threshold:.2e}")
        
        report.append("\\n## Recovery Factor Analysis")
        if len(recovery_df) > 0:
            recovery_stats = recovery_df.groupby('noise_type')['recovery_factor'].agg(['mean', 'std', 'max'])
            report.append(recovery_stats.to_string())
        
        report.append("\\n## Best Noise-Resilient Configurations")
        # Find configs that perform best at high noise levels
        high_noise = df[df['noise_level'] > 0.01]  # Above 1% error rate
        if len(high_noise) > 0:
            best_high_noise = high_noise.nlargest(5, 'accuracy')[
                ['model_name', 'qecc_code', 'noise_type', 'noise_level', 'accuracy']
            ]
            report.append(best_high_noise.to_string(index=False))
        
        report.append("\\n## Noise Type Comparison")
        noise_comparison = df.groupby(['noise_type', 'qecc_enabled'])['accuracy'].agg(['mean', 'std'])
        report.append(noise_comparison.to_string())
        
        return "\\n".join(report)
    
    def _get_default_models(self) -> List[Dict[str, Any]]:
        """Get default model configurations for noise studies."""
        return [
            {
                'name': 'VQC_shallow',
                'num_layers': 2,
                'entanglement': 'circular'
            },
            {
                'name': 'VQC_deep', 
                'num_layers': 6,
                'entanglement': 'full'
            }
        ]
    
    def _load_dataset(self, dataset_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load dataset by name."""
        if dataset_name == 'quantum_iris':
            return self.datasets.load_quantum_iris()
        elif dataset_name == 'quantum_moons':
            return self.datasets.load_quantum_moons(n_samples=100)  # Smaller for noise studies
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    
    def _create_model(self, model_config: Dict[str, Any], num_features: int) -> QECCAwareQNN:
        """Create quantum neural network."""
        num_qubits = max(num_features, 2)
        return QECCAwareQNN(
            num_qubits=num_qubits,
            num_layers=model_config['num_layers'],
            entanglement=model_config['entanglement']
        )
    
    def _create_qecc(self, qecc_code: str):
        """Create error correction code."""
        if qecc_code == 'steane':
            return SteaneCode()
        elif qecc_code == 'surface_d3':
            return SurfaceCode(distance=3)
        else:
            raise ValueError(f"Unknown QECC code: {qecc_code}")
    
    def get_results_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame."""
        data = []
        for result in self.results:
            data.append({
                'noise_type': result.noise_type,
                'noise_level': result.noise_level,
                'qecc_enabled': result.qecc_enabled,
                'qecc_code': result.qecc_code,
                'model_name': result.model_name,
                'dataset_name': result.dataset_name,
                'accuracy': result.accuracy,
                'fidelity': result.fidelity,
                'logical_error_rate': result.logical_error_rate,
                'physical_error_rate': result.physical_error_rate,
                'threshold_exceeded': result.threshold_exceeded,
                'degradation_rate': result.degradation_rate,
                'recovery_factor': result.recovery_factor,
                'circuit_depth': result.circuit_depth,
                'num_qubits': result.num_qubits,
                'shots': result.shots
            })
        
        return pd.DataFrame(data)
    
    def __str__(self) -> str:
        """String representation."""
        return f"NoiseResilienceBenchmark({len(self.results)} results)"
    
    def __repr__(self) -> str:
        """Detailed representation."""
        return (f"NoiseResilienceBenchmark(results={len(self.results)}, "
                f"noise_types={list(self.noise_configs.keys())})")