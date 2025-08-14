#!/usr/bin/env python3
"""
Scalable training example with advanced optimization and auto-scaling.
Generation 3: High-performance, optimized, and scalable implementation.
"""

import numpy as np
import sys
import os
import time
from typing import Tuple, Dict, Any

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from qecc_qml.core.quantum_nn import QECCAwareQNN

# Import the scalable trainer
try:
    from qecc_qml.training.scalable_trainer import ScalableQECCTrainer
    SCALABLE_TRAINER_AVAILABLE = True
except ImportError as e:
    print(f"Scalable trainer not available: {e}")
    SCALABLE_TRAINER_AVAILABLE = False

# Fallback to robust trainer
if not SCALABLE_TRAINER_AVAILABLE:
    try:
        from qecc_qml.training.robust_trainer import RobustQECCTrainer as ScalableQECCTrainer
        print("Using RobustQECCTrainer as fallback")
    except ImportError:
        from qecc_qml.training.basic_trainer_clean import BasicQECCTrainer as ScalableQECCTrainer
        print("Using BasicQECCTrainer as fallback")

# Direct import to avoid dependency issues
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'qecc_qml', 'datasets'))
from simple_datasets import load_quantum_classification

try:
    from qecc_qml.codes.surface_code import SurfaceCode
    SURFACE_CODE_AVAILABLE = True
except ImportError:
    SURFACE_CODE_AVAILABLE = False


def generate_large_dataset(base_size: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a large dataset for scalability testing."""
    datasets = []
    
    # Multiple dataset types for comprehensive testing
    for i in range(3):
        if i % 2 == 0:  # Synthetic dataset
            X, y = load_quantum_classification(
                dataset='synthetic',
                n_samples=base_size,
                n_features=4,
                noise=0.1 + 0.05 * i,
                random_state=42 + i * 100
            )
        else:  # Entangled dataset
            X, y = load_quantum_classification(
                dataset='entangled',
                n_samples=base_size,
                n_qubits=4,
                entanglement_strength=0.8 - 0.1 * i,
                random_state=42 + i * 100
            )
        datasets.append((X, y))
    
    # Combine all datasets
    X_combined = np.vstack([X for X, y in datasets])
    y_combined = np.hstack([y for X, y in datasets])
    
    return X_combined, y_combined


def benchmark_training_performance(trainer, X_train, y_train, X_test, y_test) -> Dict[str, Any]:
    """Benchmark training performance across different configurations."""
    
    benchmark_results = {}
    
    # Test different batch sizes
    batch_sizes = [16, 32, 64, 128]
    
    for batch_size in batch_sizes:
        print(f"\n🔍 Benchmarking batch size: {batch_size}")
        
        try:
            start_time = time.time()
            
            # Reset trainer state
            trainer.current_params = None
            if hasattr(trainer, 'optimizer') and trainer.optimizer:
                trainer.optimizer.reset_stats()
            
            # Train with specific batch size
            history = trainer.fit(
                X_train[:500],  # Use subset for benchmarking
                y_train[:500],
                epochs=10,
                batch_size=batch_size,
                validation_split=0.2
            )
            
            training_time = time.time() - start_time
            
            # Evaluate
            test_results = trainer.evaluate(X_test, y_test)
            
            benchmark_results[f'batch_{batch_size}'] = {
                'training_time': training_time,
                'final_accuracy': test_results['accuracy'],
                'final_loss': test_results['loss'],
                'avg_epoch_time': training_time / 10,
                'throughput': len(X_train[:500]) / training_time
            }
            
            print(f"   ✓ Batch {batch_size}: {training_time:.2f}s, Acc: {test_results['accuracy']:.4f}")
            
        except Exception as e:
            print(f"   ❌ Batch {batch_size} failed: {e}")
            benchmark_results[f'batch_{batch_size}'] = {'error': str(e)}
    
    return benchmark_results


def main():
    """Demonstrate scalable QECC-aware QML training with Generation 3 optimizations."""
    
    print("🚀 QECC-Aware QML Scalable Training Demo - Generation 3")
    print("=" * 70)
    
    # Step 1: Create advanced quantum neural network
    print("📊 Step 1: Creating Advanced Quantum Neural Network...")
    qnn = QECCAwareQNN(
        num_qubits=4,
        num_layers=4,  # More layers for scalability testing
        entanglement="circular",
        feature_map="amplitude_encoding"
    )
    print(f"   ✓ Created advanced QNN with {qnn.num_qubits} logical qubits and {qnn.num_layers} layers")
    
    # Step 2: Add error correction with validation
    print("\n🛡️  Step 2: Adding Error Correction...")
    if SURFACE_CODE_AVAILABLE:
        try:
            surface_code = SurfaceCode(distance=3, logical_qubits=4)
            qnn.add_error_correction(
                scheme=surface_code,
                syndrome_extraction_frequency=2,
                decoder="minimum_weight_matching"
            )
            print(f"   ✓ Added Surface Code protection (distance=3)")
            print(f"   ✓ Physical qubits required: {qnn.num_physical_qubits}")
        except Exception as e:
            print(f"   ⚠️  Error correction setup failed: {e}")
            print("   → Continuing without error correction for demo")
    else:
        print("   ⚠️  Surface code not available, continuing without error correction")
    
    # Step 3: Generate large-scale datasets for scalability testing
    print("\n📚 Step 3: Generating Large-Scale Datasets...")
    
    # Generate training dataset
    X_train, y_train = generate_large_dataset(base_size=800)
    X_test, y_test = load_quantum_classification(
        dataset='synthetic',
        n_samples=300,
        n_features=4,
        noise=0.1,
        random_state=999
    )
    
    print(f"   ✓ Training samples: {len(X_train)} (large-scale)")
    print(f"   ✓ Test samples: {len(X_test)}")
    print(f"   ✓ Features: {X_train.shape[1]}, Classes: {len(np.unique(y_train))}")
    
    # Step 4: Initialize scalable trainer with full optimization
    print("\n🏋️  Step 4: Initializing Scalable Trainer...")
    
    # Check if we have the full scalable trainer or fallback
    trainer_config = {
        'qnn': qnn,
        'learning_rate': 0.02,
        'shots': 1024,
        'verbose': True,
        'validation_freq': 3,
        'checkpoint_freq': 5,
        'max_retries': 2,
        'enable_monitoring': True,
        'log_level': 'INFO'
    }
    
    # Add Generation 3 parameters if available
    if SCALABLE_TRAINER_AVAILABLE:
        trainer_config.update({
            'enable_optimization': True,
            'enable_auto_scaling': True,
            'enable_parallel': True,
            'initial_batch_size': 64,
            'max_workers': 4,
            'cache_size': 2000,
            'memory_limit': 0.8,
            'performance_target': 1.5
        })
    
    trainer = ScalableQECCTrainer(**trainer_config)
    
    print(f"   ✓ Scalable trainer initialized")
    if SCALABLE_TRAINER_AVAILABLE:
        print(f"   ✓ Generation 3 optimizations: enabled")
        print(f"   ✓ Auto-scaling: enabled")
        print(f"   ✓ Parallel processing: enabled")
        print(f"   ✓ Intelligent caching: enabled")
    else:
        print(f"   ⚠️  Using fallback trainer (Generation 1/2)")
    
    # Step 5: Scalable training with optimization
    print("\n🚂 Step 5: Scalable Training with Advanced Optimization...")
    print("-" * 60)
    
    training_start_time = time.time()
    
    try:
        history = trainer.fit(
            X_train, y_train,
            epochs=25,
            batch_size=64,
            validation_split=0.2
        )
        
        training_total_time = time.time() - training_start_time
        
        print("-" * 60)
        print(f"   ✅ Scalable training completed in {training_total_time:.2f}s")
        
    except Exception as e:
        print(f"   ❌ Training failed: {e}")
        return None
    
    # Step 6: Performance evaluation and optimization analysis
    print("\n📊 Step 6: Performance Evaluation and Optimization Analysis...")
    
    # Evaluate on test set
    test_results = trainer.evaluate(X_test, y_test)
    print(f"   🎯 Test Results:")
    print(f"      → Accuracy: {test_results['accuracy']:.4f}")
    print(f"      → Loss: {test_results['loss']:.4f}")
    print(f"      → Fidelity: {test_results['fidelity']:.4f}")
    
    # Get optimization diagnostics if available
    if hasattr(trainer, 'get_optimization_diagnostics'):
        diagnostics = trainer.get_optimization_diagnostics()
        
        print(f"\n   🔧 Optimization Performance:")
        
        # Resource usage
        if 'resource_stats' in diagnostics:
            resource_stats = diagnostics['resource_stats']
            print(f"      → Peak Memory: {resource_stats.get('peak_memory_mb', 0):.1f} MB")
            print(f"      → Avg CPU: {resource_stats.get('avg_cpu_percent', 0):.1f}%")
            print(f"      → Parallel Tasks: {resource_stats.get('parallel_tasks_completed', 0)}")
            print(f"      → Scaling Adjustments: {resource_stats.get('scaling_adjustments', 0)}")
        
        # Optimization stats
        if 'optimization_stats' in diagnostics:
            opt_stats = diagnostics['optimization_stats']
            print(f"      → Cache Hit Rate: {opt_stats.get('cache_efficiency', 0)*100:.1f}%")
            print(f"      → Total Evaluations: {opt_stats.get('total_evaluations', 0)}")
            print(f"      → Cache Saves: {opt_stats.get('cache_saves', 0)}")
            print(f"      → Parallel Tasks: {opt_stats.get('parallel_tasks', 0)}")
        
        # Scaling effectiveness
        if 'scaling_stats' in diagnostics:
            scaling_stats = diagnostics['scaling_stats']
            print(f"      → Resource Changes: {scaling_stats.get('scaling_events', 0)} events")
            if 'performance_trend' in scaling_stats:
                trend = scaling_stats['performance_trend']
                trend_desc = "improving" if trend > 0 else "stable" if abs(trend) < 0.01 else "declining"
                print(f"      → Performance Trend: {trend_desc} ({trend:.4f})")
    
    # Step 7: Scalability benchmarking
    print(f"\n📈 Step 7: Scalability Benchmarking...")
    
    print("   🔬 Running performance benchmarks across different configurations...")
    
    benchmark_results = benchmark_training_performance(trainer, X_train, y_train, X_test, y_test)
    
    # Analyze benchmark results
    successful_benchmarks = {k: v for k, v in benchmark_results.items() if 'error' not in v}
    
    if successful_benchmarks:
        print(f"\n   📊 Benchmark Results:")
        
        best_throughput = max(successful_benchmarks.values(), key=lambda x: x.get('throughput', 0))
        best_accuracy = max(successful_benchmarks.values(), key=lambda x: x.get('final_accuracy', 0))
        fastest_training = min(successful_benchmarks.values(), key=lambda x: x.get('training_time', float('inf')))
        
        print(f"      → Best Throughput: {best_throughput['throughput']:.1f} samples/s")
        print(f"      → Best Accuracy: {best_accuracy['final_accuracy']:.4f}")
        print(f"      → Fastest Training: {fastest_training['training_time']:.2f}s")
        
        # Calculate scalability metrics
        batch_sizes = [int(k.split('_')[1]) for k in successful_benchmarks.keys()]
        throughputs = [v['throughput'] for v in successful_benchmarks.values()]
        
        if len(throughputs) > 1:
            scalability_factor = max(throughputs) / min(throughputs)
            print(f"      → Scalability Factor: {scalability_factor:.2f}x")
    
    # Step 8: Summary and recommendations
    print(f"\n🎯 Step 8: Performance Summary and Recommendations...")
    
    # Calculate overall metrics
    total_samples_processed = len(X_train)
    avg_throughput = total_samples_processed / training_total_time
    samples_per_epoch = total_samples_processed / 25  # 25 epochs
    
    print(f"   📈 Overall Performance:")
    print(f"      → Total Samples Processed: {total_samples_processed:,}")
    print(f"      → Total Training Time: {training_total_time:.2f}s")
    print(f"      → Average Throughput: {avg_throughput:.1f} samples/s")
    print(f"      → Samples per Epoch: {samples_per_epoch:.0f}")
    print(f"      → Final Test Accuracy: {test_results['accuracy']:.4f}")
    
    # Performance classification
    if avg_throughput > 100:
        performance_class = "🚀 High Performance"
    elif avg_throughput > 50:
        performance_class = "⚡ Good Performance"
    elif avg_throughput > 20:
        performance_class = "✅ Acceptable Performance"
    else:
        performance_class = "⚠️ Optimization Needed"
    
    print(f"      → Performance Class: {performance_class}")
    
    # Recommendations
    print(f"\n   💡 Optimization Recommendations:")
    
    if SCALABLE_TRAINER_AVAILABLE and hasattr(trainer, 'optimization_history'):
        cache_hit_rates = getattr(trainer, 'optimization_history', {}).get('cache_hit_rates', [])
        if cache_hit_rates and np.mean(cache_hit_rates) < 0.5:
            print("      → Consider increasing cache size for better performance")
        
        memory_usage = getattr(trainer, 'optimization_history', {}).get('memory_usage', [])
        if memory_usage and np.mean(memory_usage) > 0.8:
            print("      → High memory usage detected - consider reducing batch size")
        elif memory_usage and np.mean(memory_usage) < 0.3:
            print("      → Low memory usage - could increase batch size for better throughput")
    
    if test_results['accuracy'] > 0.8:
        print("      → Excellent model performance - system is well-tuned!")
    elif test_results['accuracy'] < 0.6:
        print("      → Consider adjusting learning rate or model architecture")
    
    print(f"\n🎉 Scalable training demo completed!")
    print(f"   Generation 3 implementation demonstrates advanced optimization and scaling.")
    
    return {
        'trainer': trainer,
        'history': history,
        'test_results': test_results,
        'benchmark_results': benchmark_results,
        'performance_metrics': {
            'total_time': training_total_time,
            'throughput': avg_throughput,
            'test_accuracy': test_results['accuracy']
        }
    }


if __name__ == "__main__":
    try:
        results = main()
        if results:
            print(f"\n✅ Scalable quantum ML system operational!")
            print(f"   🎯 Achieved {results['performance_metrics']['test_accuracy']:.1%} accuracy")
            print(f"   ⚡ Processing {results['performance_metrics']['throughput']:.1f} samples/second")
        else:
            print(f"\n⚠️  Demo completed with limitations")
        
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)