#!/usr/bin/env python3
"""
Robust training example with comprehensive error handling and monitoring.
Generation 2: Enhanced reliability and validation.
"""

import numpy as np
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from qecc_qml.core.quantum_nn import QECCAwareQNN
from qecc_qml.training.robust_trainer import RobustQECCTrainer

# Direct import to avoid dependency issues
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'qecc_qml', 'datasets'))
from simple_datasets import load_quantum_classification

try:
    from qecc_qml.codes.surface_code import SurfaceCode
    SURFACE_CODE_AVAILABLE = True
except ImportError:
    SURFACE_CODE_AVAILABLE = False


def main():
    """Demonstrate robust QECC-aware QML training with Generation 2 enhancements."""
    
    print("🚀 QECC-Aware QML Robust Training Demo - Generation 2")
    print("=" * 60)
    
    # Step 1: Create quantum neural network with validation
    print("📊 Step 1: Creating and Validating Quantum Neural Network...")
    qnn = QECCAwareQNN(
        num_qubits=4,
        num_layers=3,
        entanglement="circular",
        feature_map="amplitude_encoding"
    )
    print(f"   ✓ Created QNN with {qnn.num_qubits} logical qubits and {qnn.num_layers} layers")
    
    # Step 2: Add error correction with validation
    print("\n🛡️  Step 2: Adding Error Correction with Validation...")
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
    
    # Step 3: Generate comprehensive training datasets
    print("\n📚 Step 3: Generating and Validating Training Data...")
    
    # Generate multiple dataset types for robustness testing
    datasets = []
    
    # Synthetic dataset
    X_syn, y_syn = load_quantum_classification(
        dataset='synthetic',
        n_samples=500,
        n_features=4,
        noise=0.15,
        random_state=42
    )
    datasets.append(('Synthetic', X_syn, y_syn))
    
    # Entangled dataset
    X_ent, y_ent = load_quantum_classification(
        dataset='entangled',
        n_samples=300,
        n_qubits=4,
        entanglement_strength=0.7,
        random_state=123
    )
    datasets.append(('Entangled', X_ent, y_ent))
    
    # Test dataset
    X_test, y_test = load_quantum_classification(
        dataset='synthetic',
        n_samples=200,
        n_features=4,
        noise=0.1,
        random_state=456
    )
    
    print(f"   ✓ Generated {len(datasets)} training datasets")
    print(f"   ✓ Test samples: {len(X_test)}, Features: {X_test.shape[1]}")
    
    # Step 4: Initialize robust trainer with full configuration
    print("\n🏋️  Step 4: Initializing Robust Trainer...")
    trainer = RobustQECCTrainer(
        qnn=qnn,
        learning_rate=0.03,
        shots=1024,
        verbose=True,
        validation_freq=3,      # Validate every 3 epochs
        checkpoint_freq=5,      # Checkpoint every 5 epochs  
        max_retries=2,          # Retry failed operations twice
        enable_monitoring=True,  # Enable performance monitoring
        log_level='INFO'        # Detailed logging
    )
    print(f"   ✓ Robust trainer initialized with enhanced capabilities")
    print(f"   ✓ Validation frequency: every {trainer.validation_freq} epochs")
    print(f"   ✓ Checkpoint frequency: every {trainer.checkpoint_freq} epochs")
    
    # Step 5: Train on multiple datasets with comprehensive validation
    print("\n🚂 Step 5: Robust Training with Validation...")
    print("-" * 50)
    
    training_results = {}
    
    for dataset_name, X_train, y_train in datasets:
        print(f"\n🔄 Training on {dataset_name} dataset ({len(X_train)} samples)...")
        
        try:
            # Train with enhanced error handling
            history = trainer.fit(
                X_train, y_train,
                epochs=15,
                batch_size=16,
                validation_split=0.25
            )
            
            training_results[dataset_name] = {
                'history': history,
                'final_loss': history['loss'][-1] if history['loss'] else float('inf'),
                'final_accuracy': history['accuracy'][-1] if history['accuracy'] else 0.0,
                'training_time': sum(history.get('epoch_time', []))
            }
            
            print(f"   ✓ Training on {dataset_name} completed successfully")
            
        except Exception as e:
            print(f"   ❌ Training on {dataset_name} failed: {e}")
            training_results[dataset_name] = {'error': str(e)}
    
    print("-" * 50)
    
    # Step 6: Comprehensive evaluation and diagnostics
    print("\n📊 Step 6: Comprehensive Evaluation and Diagnostics...")
    
    # Evaluate on test set
    test_results = trainer.evaluate(X_test, y_test)
    print(f"   🎯 Test Results:")
    print(f"      → Accuracy: {test_results['accuracy']:.4f}")
    print(f"      → Loss: {test_results['loss']:.4f}")
    print(f"      → Fidelity: {test_results['fidelity']:.4f}")
    
    # Get training diagnostics
    diagnostics = trainer.get_training_diagnostics()
    print(f"\n   🔍 Training Diagnostics:")
    print(f"      → Total Validations: {diagnostics['validation_report']['total_validations']}")
    print(f"      → Failed Validations: {diagnostics['validation_report']['failed_validations']}")
    print(f"      → Error Events: {diagnostics['validation_report']['total_errors']}")
    print(f"      → Warning Events: {diagnostics['validation_report']['total_warnings']}")
    print(f"      → Successful Epochs: {diagnostics['performance_metrics']['successful_epochs']}")
    print(f"      → Failed Epochs: {diagnostics['performance_metrics']['failed_epochs']}")
    print(f"      → Recovery Events: {diagnostics['performance_metrics']['recovery_events']}")
    
    # Training stability analysis
    stability = diagnostics['training_stability']
    print(f"\n   📈 Training Stability:")
    print(f"      → Parameter Variance: {stability['parameter_variance']:.6f}")
    print(f"      → Loss Trend: {stability['loss_trend']:.6f}")
    print(f"      → Convergence Rate: {stability['convergence_rate']:.2%}")
    
    # Step 7: Performance summary across datasets
    print(f"\n📈 Step 7: Multi-Dataset Performance Summary...")
    
    successful_trainings = 0
    total_accuracy = 0
    total_training_time = 0
    
    for dataset_name, results in training_results.items():
        if 'error' not in results:
            successful_trainings += 1
            total_accuracy += results['final_accuracy']
            total_training_time += results['training_time']
            
            print(f"   📊 {dataset_name} Dataset:")
            print(f"      → Final Accuracy: {results['final_accuracy']:.4f}")
            print(f"      → Final Loss: {results['final_loss']:.4f}")
            print(f"      → Training Time: {results['training_time']:.2f}s")
        else:
            print(f"   ❌ {dataset_name} Dataset: {results['error']}")
    
    if successful_trainings > 0:
        avg_accuracy = total_accuracy / successful_trainings
        avg_training_time = total_training_time / successful_trainings
        
        print(f"\n   🎯 Overall Performance:")
        print(f"      → Success Rate: {successful_trainings}/{len(datasets)} ({100*successful_trainings/len(datasets):.1f}%)")
        print(f"      → Average Accuracy: {avg_accuracy:.4f}")
        print(f"      → Average Training Time: {avg_training_time:.2f}s")
        print(f"      → Test Set Accuracy: {test_results['accuracy']:.4f}")
    
    # Step 8: Error analysis and recommendations
    print(f"\n🔧 Step 8: Error Analysis and Recommendations...")
    
    error_stats = diagnostics.get('error_statistics', {})
    if error_stats:
        print("   🚨 Error Statistics:")
        for error_type, count in error_stats.items():
            print(f"      → {error_type}: {count} occurrences")
    else:
        print("   ✅ No errors detected during training")
    
    # Recommendations based on performance
    print("\n   💡 Recommendations:")
    if stability['convergence_rate'] < 0.5:
        print("   → Consider reducing learning rate for better convergence")
    if stability['loss_trend'] > 0:
        print("   → Loss is increasing - possible overfitting or learning rate too high")
    if diagnostics['performance_metrics']['recovery_events'] > 0:
        print("   → Multiple recovery events detected - consider parameter tuning")
    if successful_trainings == len(datasets):
        print("   → All datasets trained successfully - system is robust!")
    
    print(f"\n🎉 Robust training demo completed!")
    print(f"   Generation 2 implementation demonstrates enhanced reliability.")
    
    return {
        'trainer': trainer,
        'training_results': training_results,
        'test_results': test_results,
        'diagnostics': diagnostics
    }


if __name__ == "__main__":
    try:
        results = main()
        print(f"\n✅ Robust training system operational!")
        
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)