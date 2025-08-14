#!/usr/bin/env python3
"""
Basic training example with minimal dependencies.
Generation 1: Simple, working demonstration.
"""

import numpy as np
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from qecc_qml.core.quantum_nn import QECCAwareQNN
from qecc_qml.training.basic_trainer_clean import BasicQECCTrainer
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'qecc_qml', 'datasets'))
from simple_datasets import load_quantum_classification
from qecc_qml.codes.surface_code import SurfaceCode


def main():
    """Demonstrate basic QECC-aware QML training."""
    
    print("🚀 QECC-Aware QML Training Demo - Generation 1")
    print("=" * 50)
    
    # Step 1: Create quantum neural network
    print("📊 Step 1: Creating Quantum Neural Network...")
    qnn = QECCAwareQNN(
        num_qubits=4,
        num_layers=2,
        entanglement="circular",
        feature_map="amplitude_encoding"
    )
    print(f"   ✓ Created QNN with {qnn.num_qubits} logical qubits and {qnn.num_layers} layers")
    
    # Step 2: Add error correction
    print("\n🛡️  Step 2: Adding Error Correction...")
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
        print("   → Continuing without error correction for basic demo")
    
    # Step 3: Generate training data
    print("\n📚 Step 3: Generating Training Data...")
    X_train, y_train = load_quantum_classification(
        dataset='synthetic',
        n_samples=400,
        n_features=4,
        noise=0.1,
        random_state=42
    )
    
    X_test, y_test = load_quantum_classification(
        dataset='synthetic',
        n_samples=100,
        n_features=4,
        noise=0.05,
        random_state=123
    )
    
    print(f"   ✓ Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"   ✓ Features: {X_train.shape[1]}, Classes: {len(np.unique(y_train))}")
    
    # Step 4: Initialize trainer
    print("\n🏋️  Step 4: Initializing Trainer...")
    trainer = BasicQECCTrainer(
        qnn=qnn,
        learning_rate=0.05,
        shots=1024,
        verbose=True
    )
    print(f"   ✓ Trainer initialized with learning rate: {trainer.learning_rate}")
    
    # Step 5: Train the model
    print("\n🚂 Step 5: Training Model...")
    print("-" * 30)
    
    history = trainer.fit(
        X_train, y_train,
        epochs=20,
        batch_size=32,
        validation_split=0.2
    )
    
    print("-" * 30)
    print("   ✓ Training completed!")
    
    # Step 6: Evaluate performance
    print("\n📊 Step 6: Evaluating Performance...")
    
    # Training set evaluation
    train_results = trainer.evaluate(X_train, y_train)
    print(f"   📈 Training Results:")
    print(f"      → Accuracy: {train_results['accuracy']:.4f}")
    print(f"      → Loss: {train_results['loss']:.4f}")
    print(f"      → Fidelity: {train_results['fidelity']:.4f}")
    
    # Test set evaluation
    test_results = trainer.evaluate(X_test, y_test)
    print(f"   🎯 Test Results:")
    print(f"      → Accuracy: {test_results['accuracy']:.4f}")
    print(f"      → Loss: {test_results['loss']:.4f}")
    print(f"      → Fidelity: {test_results['fidelity']:.4f}")
    
    # Step 7: Display training history
    print(f"\n📈 Step 7: Training Summary...")
    final_loss = history['loss'][-1] if history['loss'] else 0
    final_acc = history['accuracy'][-1] if history['accuracy'] else 0
    final_fidelity = history['fidelity'][-1] if history['fidelity'] else 0
    avg_epoch_time = np.mean(history['epoch_time']) if history['epoch_time'] else 0
    
    print(f"   🎯 Final Training Metrics:")
    print(f"      → Loss: {final_loss:.4f}")
    print(f"      → Accuracy: {final_acc:.4f}") 
    print(f"      → Fidelity: {final_fidelity:.4f}")
    print(f"      → Avg Epoch Time: {avg_epoch_time:.2f}s")
    
    # Step 8: Make predictions on new data
    print(f"\n🔮 Step 8: Making Predictions...")
    sample_predictions = trainer.predict(X_test[:5])
    print(f"   Sample predictions: {sample_predictions}")
    print(f"   Actual labels:      {y_test[:5]}")
    
    print(f"\n🎉 Demo completed successfully!")
    print(f"   Generation 1 implementation is working with basic functionality.")
    
    return {
        'qnn': qnn,
        'trainer': trainer,
        'history': history,
        'test_results': test_results
    }


if __name__ == "__main__":
    try:
        results = main()
        print(f"\n✅ All systems operational!")
        
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)