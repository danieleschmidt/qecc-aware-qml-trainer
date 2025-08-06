#!/usr/bin/env python3
"""
Basic example of QECC-aware quantum machine learning.

This example demonstrates:
1. Creating a quantum neural network with error correction
2. Training on synthetic data with noise
3. Evaluating performance improvements from QECC
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import our QECC-aware QML library
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qecc_qml import QECCAwareQNN, NoiseModel, QECCTrainer, SurfaceCode, NoiseBenchmark


def generate_quantum_dataset(n_samples: int = 200, n_features: int = 4, noise: float = 0.1):
    """Generate a synthetic quantum-inspired dataset."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features,
        n_redundant=0,
        n_clusters_per_class=1,
        class_sep=1.5,
        random_state=42
    )
    
    # Scale features to [0, π] range for quantum encoding
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = (X + 3) * np.pi / 6  # Map to [0, π] approximately
    
    return X, y


def main():
    """Main example execution."""
    print("🚀 QECC-Aware Quantum Machine Learning Example")
    print("=" * 50)
    
    # Step 1: Generate dataset
    print("\n1. Generating quantum dataset...")
    X, y = generate_quantum_dataset(n_samples=100, n_features=4)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    print(f"   Features: {X.shape[1]}")
    
    # Step 2: Create quantum neural network without error correction
    print("\n2. Creating baseline QNN (no error correction)...")
    qnn_baseline = QECCAwareQNN(
        num_qubits=4,
        num_layers=2,
        entanglement="circular",
        feature_map="amplitude_encoding"
    )
    print(f"   {qnn_baseline}")
    print(f"   Circuit depth: {qnn_baseline.get_circuit_depth()}")
    print(f"   Parameters: {qnn_baseline.get_num_parameters()}")
    
    # Step 3: Create quantum neural network with error correction
    print("\n3. Creating QECC-aware QNN...")
    qnn_protected = QECCAwareQNN(
        num_qubits=4,
        num_layers=2,
        entanglement="circular",
        feature_map="amplitude_encoding"
    )
    
    # Add Surface Code error correction
    surface_code = SurfaceCode(distance=3, logical_qubits=4)
    qnn_protected.add_error_correction(
        scheme=surface_code,
        syndrome_extraction_frequency=1,
        decoder="minimum_weight_matching"
    )
    
    print(f"   {qnn_protected}")
    print(f"   Physical qubits: {qnn_protected.num_physical_qubits}")
    print(f"   Error correction: {surface_code}")
    
    # Step 4: Define noise model
    print("\n4. Setting up noise model...")
    noise_model = NoiseModel(
        gate_error_rate=0.001,
        readout_error_rate=0.01,
        T1=50e-6,
        T2=70e-6
    )
    print(f"   {noise_model}")
    
    # Step 5: Train baseline model
    print("\n5. Training baseline QNN...")
    trainer_baseline = QECCTrainer(
        qnn=qnn_baseline,
        noise_model=noise_model,
        optimizer="noise_aware_adam",
        learning_rate=0.05,
        shots=512
    )
    
    history_baseline = trainer_baseline.fit(
        X_train, y_train,
        X_val=X_test, y_val=y_test,
        epochs=20,
        batch_size=16,
        verbose=False
    )
    
    baseline_results = trainer_baseline.evaluate(X_test, y_test)
    print(f"   Baseline accuracy: {baseline_results['accuracy']:.3f}")
    print(f"   Baseline fidelity: {baseline_results['fidelity']:.3f}")
    
    # Step 6: Train QECC-protected model
    print("\n6. Training QECC-protected QNN...")
    trainer_protected = QECCTrainer(
        qnn=qnn_protected,
        noise_model=noise_model,
        optimizer="noise_aware_adam",
        learning_rate=0.05,
        shots=512
    )
    
    history_protected = trainer_protected.fit(
        X_train, y_train,
        X_val=X_test, y_val=y_test,
        epochs=20,
        batch_size=16,
        verbose=False
    )
    
    protected_results = trainer_protected.evaluate(X_test, y_test)
    print(f"   Protected accuracy: {protected_results['accuracy']:.3f}")
    print(f"   Protected fidelity: {protected_results['fidelity']:.3f}")
    print(f"   Logical error rate: {protected_results['logical_error_rate']:.2e}")
    
    # Step 7: Compare results
    print("\n7. Performance Comparison:")
    print("   " + "=" * 40)
    print(f"   {'Metric':<20} {'Baseline':<12} {'Protected':<12} {'Improvement':<12}")
    print("   " + "-" * 40)
    
    metrics = ['accuracy', 'fidelity']
    for metric in metrics:
        baseline_val = baseline_results[metric]
        protected_val = protected_results[metric]
        improvement = ((protected_val - baseline_val) / baseline_val) * 100
        
        print(f"   {metric.capitalize():<20} {baseline_val:<12.3f} {protected_val:<12.3f} {improvement:<12.1f}%")
    
    # Step 8: Noise resilience benchmark
    print("\n8. Running noise resilience benchmark...")
    benchmark = NoiseBenchmark(
        model=qnn_protected,
        noise_levels=np.logspace(-4, -2, 8),
        metrics=["accuracy", "fidelity", "logical_error_rate"],
        shots=256
    )
    
    # Run benchmark with pre-trained parameters
    benchmark_results = benchmark.run(
        X_test, y_test,
        model_parameters=trainer_protected.get_parameters(),
        verbose=False
    )
    
    print("   Benchmark complete!")
    
    # Step 9: Plot results
    print("\n9. Generating plots...")
    
    # Plot 1: Training history comparison
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('QECC-Aware QML Training Results', fontsize=16)
    
    # Training loss
    ax1.plot(history_baseline['loss'], 'b-', label='Baseline', linewidth=2)
    ax1.plot(history_protected['loss'], 'r-', label='Protected', linewidth=2)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Training accuracy
    ax2.plot(history_baseline['accuracy'], 'b-', label='Baseline', linewidth=2)
    ax2.plot(history_protected['accuracy'], 'r-', label='Protected', linewidth=2)
    ax2.set_title('Training Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Fidelity tracking
    ax3.plot(history_baseline['fidelity'], 'b-', label='Baseline', linewidth=2)
    ax3.plot(history_protected['fidelity'], 'r-', label='Protected', linewidth=2)
    ax3.set_title('Circuit Fidelity')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Fidelity')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Logical error rate (protected model only)
    ax4.semilogy(history_protected['logical_error_rate'], 'r-', linewidth=2)
    ax4.set_title('Logical Error Rate (Protected)')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Error Rate')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('qecc_training_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot 2: Noise resilience
    benchmark.plot_noise_resilience(
        save_path='noise_resilience.png',
        compare_with_uncorrected=False
    )
    
    # Step 10: Summary
    print("\n10. Summary:")
    print("    " + "=" * 45)
    
    accuracy_improvement = ((protected_results['accuracy'] - baseline_results['accuracy']) / 
                           baseline_results['accuracy']) * 100
    
    print(f"    ✅ Successfully trained QECC-aware QNN")
    print(f"    ✅ Accuracy improvement: {accuracy_improvement:.1f}%")
    print(f"    ✅ Logical error rate: {protected_results['logical_error_rate']:.2e}")
    print(f"    ✅ Physical qubits used: {qnn_protected.num_physical_qubits}")
    print(f"    ✅ Error correction overhead: {qnn_protected.num_physical_qubits / qnn_baseline.num_qubits:.1f}x")
    
    print(f"\n    📊 Plots saved:")
    print(f"       - qecc_training_results.png")
    print(f"       - noise_resilience.png")
    
    print(f"\n🎉 Example completed successfully!")


if __name__ == "__main__":
    main()