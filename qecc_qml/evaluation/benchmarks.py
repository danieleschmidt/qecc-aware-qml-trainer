"""
Benchmarking tools for QECC-aware quantum machine learning.
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
from typing import List, Dict, Optional, Tuple, Any
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
try:
    import matplotlib.pyplot as plt
except ImportError:
    class MockPlt:
        def figure(self, *args, **kwargs): return None
        def plot(self, *args, **kwargs): return None
        def show(self): pass
        def savefig(self, *args, **kwargs): pass
    plt = MockPlt()
import time
from tqdm import tqdm

from ..core.quantum_nn import QECCAwareQNN
from ..core.noise_models import NoiseModel
from ..training.qecc_trainer import QECCTrainer


class NoiseBenchmark:
    """
    Comprehensive benchmarking tool for evaluating QML performance across noise levels.
    
    Tests model performance, fidelity, and error correction effectiveness
    across various noise configurations.
    """
    
    def __init__(
        self,
        model: QECCAwareQNN,
        noise_levels: np.ndarray = None,
        metrics: List[str] = None,
        shots: int = 1024,
    ):
        """
        Initialize noise benchmark.
        
        Args:
            model: The quantum neural network to benchmark
            noise_levels: Array of noise levels to test
            metrics: List of metrics to compute
            shots: Number of measurement shots per evaluation
        """
        self.model = model
        self.noise_levels = noise_levels or np.logspace(-4, -1, 10)
        self.metrics = metrics or ["accuracy", "fidelity", "effective_error_rate"]
        self.shots = shots
        
        # Results storage
        self.results = {}
        self.benchmark_complete = False
    
    def run(
        self, 
        X_test: np.ndarray, 
        y_test: np.ndarray,
        model_parameters: Optional[np.ndarray] = None,
        verbose: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Run the noise benchmark across all noise levels.
        
        Args:
            X_test: Test features
            y_test: Test labels
            model_parameters: Pre-trained model parameters
            verbose: Whether to show progress
            
        Returns:
            Dictionary of benchmark results
        """
        if verbose:
            print(f"Running noise benchmark across {len(self.noise_levels)} noise levels")
            print(f"Testing {len(X_test)} samples with {self.shots} shots each")
        
        # Initialize results storage
        for metric in self.metrics:
            self.results[metric] = []
        
        # Add timing and overhead metrics
        self.results['runtime'] = []
        self.results['circuit_depth'] = []
        self.results['physical_qubits'] = []
        
        # Progress bar
        pbar = tqdm(self.noise_levels, desc="Noise levels") if verbose else self.noise_levels
        
        for noise_level in pbar:
            if verbose:
                pbar.set_postfix({'noise': f'{noise_level:.1e}'})
            
            # Create noise model for this level
            noise_model = NoiseModel(
                gate_error_rate=noise_level,
                readout_error_rate=noise_level * 10,
                T1=50e-6 / (1 + noise_level * 1000),
                T2=70e-6 / (1 + noise_level * 1000),
            )
            
            # Run evaluation at this noise level
            level_results = self._evaluate_at_noise_level(
                noise_model, X_test, y_test, model_parameters
            )
            
            # Store results
            for metric in self.metrics:
                if metric in level_results:
                    self.results[metric].append(level_results[metric])
                else:
                    self.results[metric].append(0.0)
            
            self.results['runtime'].append(level_results.get('runtime', 0.0))
            self.results['circuit_depth'].append(level_results.get('circuit_depth', 0))
            self.results['physical_qubits'].append(level_results.get('physical_qubits', 0))
        
        # Convert lists to numpy arrays
        for key in self.results:
            self.results[key] = np.array(self.results[key])
        
        self.benchmark_complete = True
        
        if verbose:
            print("Benchmark complete!")
            self._print_summary()
        
        return self.results
    
    def _evaluate_at_noise_level(
        self,
        noise_model: NoiseModel,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_parameters: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Evaluate model at a specific noise level."""
        start_time = time.time()
        
        # Create trainer with noise model
        trainer = QECCTrainer(
            qnn=self.model,
            noise_model=noise_model,
            shots=self.shots,
            track_fidelity=True,
        )
        
        if model_parameters is not None:
            trainer.set_parameters(model_parameters)
        
        # Evaluate model
        eval_results = trainer.evaluate(X_test, y_test)
        
        runtime = time.time() - start_time
        
        # Collect additional metrics
        results = {
            'runtime': runtime,
            'circuit_depth': self.model.get_circuit_depth(),
            'physical_qubits': self.model.num_physical_qubits,
        }
        
        # Add evaluation metrics
        results.update(eval_results)
        
        # Compute effective error rate
        if 'effective_error_rate' in self.metrics:
            results['effective_error_rate'] = noise_model.get_effective_error_rate(
                self.model.get_circuit_depth()
            )
        
        return results
    
    def _print_summary(self):
        """Print benchmark summary."""
        if not self.benchmark_complete:
            print("Benchmark not yet complete")
            return
        
        print("\n=== Noise Benchmark Summary ===")
        
        for metric in self.metrics:
            if metric in self.results:
                values = self.results[metric]
                print(f"{metric}:")
                print(f"  Min: {np.min(values):.4f} at noise {self.noise_levels[np.argmin(values)]:.1e}")
                print(f"  Max: {np.max(values):.4f} at noise {self.noise_levels[np.argmax(values)]:.1e}")
                print(f"  Mean: {np.mean(values):.4f} ± {np.std(values):.4f}")
        
        print(f"\nAverage runtime per evaluation: {np.mean(self.results['runtime']):.2f}s")
        print(f"Total benchmark time: {np.sum(self.results['runtime']):.2f}s")
    
    def plot_noise_resilience(
        self,
        save_path: Optional[str] = None,
        compare_with_uncorrected: bool = False,
        figsize: Tuple[int, int] = (12, 8)
    ):
        """
        Plot noise resilience curves.
        
        Args:
            save_path: Path to save the plot
            compare_with_uncorrected: Whether to compare with uncorrected version
            figsize: Figure size
        """
        if not self.benchmark_complete:
            raise ValueError("Run benchmark first before plotting")
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('QECC-Aware QML Noise Resilience Benchmark', fontsize=16)
        
        # Plot 1: Accuracy vs Noise
        ax1 = axes[0, 0]
        if 'accuracy' in self.results:
            ax1.semilogx(self.noise_levels, self.results['accuracy'], 'b-o', 
                        label='With QECC', linewidth=2, markersize=6)
        
        if compare_with_uncorrected:
            # TODO: Implement uncorrected comparison
            pass
        
        ax1.set_xlabel('Noise Level (Gate Error Rate)')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Classification Accuracy vs Noise')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Fidelity vs Noise
        ax2 = axes[0, 1]
        if 'fidelity' in self.results:
            ax2.semilogx(self.noise_levels, self.results['fidelity'], 'g-o',
                        label='Circuit Fidelity', linewidth=2, markersize=6)
        
        ax2.set_xlabel('Noise Level (Gate Error Rate)')
        ax2.set_ylabel('Fidelity')
        ax2.set_title('Circuit Fidelity vs Noise')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Plot 3: Error Rate vs Noise
        ax3 = axes[1, 0]
        if 'logical_error_rate' in self.results:
            ax3.loglog(self.noise_levels, self.results['logical_error_rate'], 'r-o',
                      label='Logical Error Rate', linewidth=2, markersize=6)
        if 'effective_error_rate' in self.results:
            ax3.loglog(self.noise_levels, self.results['effective_error_rate'], 'k--',
                      label='Physical Error Rate', linewidth=2)
        
        ax3.set_xlabel('Noise Level (Gate Error Rate)')
        ax3.set_ylabel('Error Rate')
        ax3.set_title('Error Rates vs Noise')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Plot 4: Runtime vs Noise
        ax4 = axes[1, 1]
        ax4.semilogx(self.noise_levels, self.results['runtime'], 'm-o',
                    label='Runtime', linewidth=2, markersize=6)
        
        ax4.set_xlabel('Noise Level (Gate Error Rate)')
        ax4.set_ylabel('Runtime (s)')
        ax4.set_title('Evaluation Runtime vs Noise')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def get_threshold_analysis(self) -> Dict[str, Any]:
        """
        Analyze error correction thresholds.
        
        Returns:
            Dictionary with threshold analysis results
        """
        if not self.benchmark_complete:
            raise ValueError("Run benchmark first")
        
        analysis = {}
        
        # Find threshold where error correction becomes beneficial
        if 'logical_error_rate' in self.results and 'effective_error_rate' in self.results:
            logical_errors = self.results['logical_error_rate']
            physical_errors = self.results['effective_error_rate']
            
            # Find crossover point
            improvement_ratio = logical_errors / physical_errors
            threshold_idx = np.argmin(np.abs(improvement_ratio - 1.0))
            
            analysis['threshold_noise_level'] = self.noise_levels[threshold_idx]
            analysis['threshold_improvement'] = improvement_ratio[threshold_idx]
        
        # Find optimal operating point (best accuracy)
        if 'accuracy' in self.results:
            best_acc_idx = np.argmax(self.results['accuracy'])
            analysis['optimal_noise_level'] = self.noise_levels[best_acc_idx]
            analysis['optimal_accuracy'] = self.results['accuracy'][best_acc_idx]
        
        # Calculate overhead
        if 'runtime' in self.results:
            min_runtime = np.min(self.results['runtime'])
            max_runtime = np.max(self.results['runtime'])
            analysis['runtime_overhead'] = max_runtime / min_runtime
        
        return analysis
    
    def export_results(self, filepath: str, format: str = 'csv'):
        """
        Export benchmark results to file.
        
        Args:
            filepath: Output file path
            format: Export format ('csv', 'json', 'pkl')
        """
        if not self.benchmark_complete:
            raise ValueError("Run benchmark first")
        
        if format == 'csv':
            import pandas as pd
            
            # Create DataFrame
            data = {'noise_level': self.noise_levels}
            data.update(self.results)
            
            df = pd.DataFrame(data)
            df.to_csv(filepath, index=False)
            
        elif format == 'json':
            import json
            
            # Convert numpy arrays to lists for JSON serialization
            export_data = {
                'noise_levels': self.noise_levels.tolist(),
                'results': {k: v.tolist() for k, v in self.results.items()}
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
                
        elif format == 'pkl':
            import pickle
            
            export_data = {
                'noise_levels': self.noise_levels,
                'results': self.results,
                'model_info': {
                    'num_qubits': self.model.num_qubits,
                    'num_layers': self.model.num_layers,
                    'error_correction': str(self.model.error_correction),
                }
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(export_data, f)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"Results exported to {filepath}")


class ComparisonBenchmark:
    """
    Benchmark for comparing different QECC schemes and QML approaches.
    """
    
    def __init__(self, models: Dict[str, QECCAwareQNN]):
        """
        Initialize comparison benchmark.
        
        Args:
            models: Dictionary of {name: model} to compare
        """
        self.models = models
        self.results = {}
    
    def run_comparison(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        noise_model: NoiseModel,
        metrics: List[str] = None,
        verbose: bool = True
    ) -> Dict[str, Dict[str, float]]:
        """
        Run comparison across all models.
        
        Args:
            X_test: Test data
            y_test: Test labels
            noise_model: Noise model to use
            metrics: Metrics to compute
            verbose: Whether to show progress
            
        Returns:
            Nested dictionary of results
        """
        if metrics is None:
            metrics = ["accuracy", "fidelity", "runtime", "logical_error_rate"]
        
        if verbose:
            print(f"Comparing {len(self.models)} models on {len(X_test)} test samples")
        
        for model_name, model in tqdm(self.models.items()) if verbose else self.models.items():
            # Create trainer
            trainer = QECCTrainer(qnn=model, noise_model=noise_model)
            
            # Evaluate
            start_time = time.time()
            eval_results = trainer.evaluate(X_test, y_test)
            runtime = time.time() - start_time
            
            # Store results
            self.results[model_name] = eval_results.copy()
            self.results[model_name]['runtime'] = runtime
            self.results[model_name]['circuit_depth'] = model.get_circuit_depth()
            self.results[model_name]['physical_qubits'] = model.num_physical_qubits
        
        if verbose:
            self._print_comparison_summary()
        
        return self.results
    
    def _print_comparison_summary(self):
        """Print comparison summary."""
        print("\n=== Model Comparison Summary ===")
        
        # Find best model for each metric
        metrics = ['accuracy', 'fidelity', 'runtime', 'logical_error_rate']
        
        for metric in metrics:
            if all(metric in results for results in self.results.values()):
                values = {name: results[metric] for name, results in self.results.items()}
                
                if metric == 'runtime' or metric == 'logical_error_rate':
                    best_model = min(values, key=values.get)
                    best_value = values[best_model]
                else:
                    best_model = max(values, key=values.get)
                    best_value = values[best_model]
                
                print(f"\nBest {metric}: {best_model} ({best_value:.4f})")
                
                # Show all values
                for name, value in sorted(values.items(), key=lambda x: x[1], reverse=(metric not in ['runtime', 'logical_error_rate'])):
                    print(f"  {name}: {value:.4f}")
    
    def plot_comparison(self, save_path: Optional[str] = None):
        """Plot comparison results."""
        if not self.results:
            raise ValueError("Run comparison first")
        
        metrics = ['accuracy', 'fidelity', 'logical_error_rate', 'runtime']
        model_names = list(self.results.keys())
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('QECC Scheme Comparison', fontsize=16)
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            
            values = [self.results[name].get(metric, 0) for name in model_names]
            
            bars = ax.bar(model_names, values)
            ax.set_title(f'{metric.replace("_", " ").title()}')
            ax.set_ylabel(metric.replace("_", " ").title())
            
            # Rotate x-axis labels if needed
            if len(max(model_names, key=len)) > 10:
                ax.tick_params(axis='x', rotation=45)
            
            # Color bars
            colors = plt.cm.Set3(np.linspace(0, 1, len(model_names)))
            for bar, color in zip(bars, colors):
                bar.set_color(color)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()