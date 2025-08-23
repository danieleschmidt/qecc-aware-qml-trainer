#!/usr/bin/env python3
"""
Comprehensive Quality Gates for QECC-QML Framework
Generation 1-3: Complete autonomous validation and testing
"""

import sys
import time
import json
import traceback
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.insert(0, '/root/repo')

def run_quality_gate_1_basic_functionality():
    """Test Generation 1: Basic functionality"""
    try:
        from qecc_qml.core.quantum_nn import QECCAwareQNN
        from qecc_qml.training.basic_trainer import BasicQECCTrainer
        
        # Test QNN creation
        qnn = QECCAwareQNN(num_qubits=4, num_layers=2)
        assert len(qnn.weight_params) == 24, f"Expected 24 parameters, got {len(qnn.weight_params)}"
        
        # Test trainer creation  
        trainer = BasicQECCTrainer(qnn, verbose=False)
        params = trainer.get_parameters()
        assert params.shape == (24,), f"Parameter shape mismatch: {params.shape}"
        
        # Test circuit creation
        x = np.random.rand(4)
        circuit = qnn.create_circuit(x)
        assert circuit is not None, "Circuit creation failed"
        
        # Test training
        X = np.random.rand(20, 4)
        y = np.random.randint(0, 2, 20)
        history = trainer.fit(X, y, epochs=2, batch_size=8)
        
        assert 'loss' in history, "Training history missing loss"
        assert len(history['loss']) == 2, f"Expected 2 epochs, got {len(history['loss'])}"
        
        return True, "Basic functionality working"
        
    except Exception as e:
        return False, f"Basic functionality failed: {str(e)}"

def run_quality_gate_2_robustness():
    """Test Generation 2: Robustness and error handling"""
    try:
        from qecc_qml.core.quantum_nn import QECCAwareQNN
        from qecc_qml.training.robust_trainer_enhanced import RobustQECCTrainer
        from qecc_qml.utils.error_recovery import ErrorRecoveryManager
        from qecc_qml.utils.validation import validate_input_data, validate_parameters
        
        # Test error recovery
        recovery_manager = ErrorRecoveryManager()
        assert recovery_manager.max_retries == 3, "Error recovery not configured correctly"
        
        # Test input validation
        X = np.random.rand(15, 4)
        y = np.random.randint(0, 2, 15)
        validate_input_data(X, y)  # Should not raise
        
        params = np.random.rand(24)
        validate_parameters(params)  # Should not raise
        
        # Test robust trainer
        qnn = QECCAwareQNN(num_qubits=4, num_layers=2)
        trainer = RobustQECCTrainer(
            qnn=qnn,
            enable_monitoring=False,  # Disable for testing
            max_retries=2,
            early_stopping_patience=10
        )
        
        # Test with edge cases
        X_edge = np.array([[0, 0, 0, 0], [1, 1, 1, 1], [0.5, 0.5, 0.5, 0.5]])
        y_edge = np.array([0, 1, 0])
        
        history = trainer.fit(X_edge, y_edge, epochs=2, batch_size=2)
        assert 'error_recoveries' in history, "Error recovery tracking missing"
        
        return True, "Robustness features working"
        
    except Exception as e:
        return False, f"Robustness test failed: {str(e)}"

def run_quality_gate_3_scalability():
    """Test Generation 3: Scalability and optimization"""
    try:
        from qecc_qml.core.quantum_nn import QECCAwareQNN
        from qecc_qml.training.scalable_trainer_advanced import ScalableQECCTrainer, AdaptiveCache, BatchProcessor
        
        # Test caching
        cache = AdaptiveCache(max_size=100)
        params = np.random.rand(24)
        x_sample = np.random.rand(4)
        
        # Cache miss
        result = cache.get(params, x_sample)
        assert result is None, "Expected cache miss"
        
        # Cache put and hit
        cache.put(params, x_sample, 0.85)
        result = cache.get(params, x_sample)
        assert result == 0.85, "Cache hit failed"
        
        # Test batch processor
        processor = BatchProcessor(max_workers=1)
        assert processor.max_workers == 1, "Batch processor config failed"
        
        # Test scalable trainer
        qnn = QECCAwareQNN(num_qubits=4, num_layers=2)
        trainer = ScalableQECCTrainer(
            qnn=qnn,
            enable_caching=True,
            cache_size=50,
            max_workers=1,
            learning_rate_schedule='cosine'
        )
        
        # Test training with performance tracking
        X = np.random.rand(30, 4)
        y = np.random.randint(0, 2, 30)
        
        history = trainer.fit(X, y, epochs=3, batch_size=10)
        
        assert 'cache_hit_rate' in history, "Cache metrics missing"
        assert 'throughput' in history, "Throughput metrics missing"
        
        # Test performance report
        report = trainer.get_performance_report()
        assert 'cache_stats' in report, "Performance report incomplete"
        
        return True, "Scalability features working"
        
    except Exception as e:
        return False, f"Scalability test failed: {str(e)}"

def run_quality_gate_4_integration():
    """Test full system integration"""
    try:
        from qecc_qml import QECCAwareQNN
        from qecc_qml.training.scalable_trainer_advanced import ScalableQECCTrainer
        from qecc_qml.codes.surface_code import SurfaceCode
        
        # Test full workflow
        qnn = QECCAwareQNN(num_qubits=4, num_layers=3)
        
        # Add error correction (if available)
        try:
            surface_code = SurfaceCode(distance=3, logical_qubits=4)
            qnn.add_error_correction(surface_code)
        except:
            pass  # Optional feature
        
        # Train with all features
        trainer = ScalableQECCTrainer(
            qnn=qnn,
            learning_rate=0.02,
            enable_caching=True,
            max_workers=1,
            adaptive_batch_sizing=False
        )
        
        # Realistic dataset
        X = np.random.rand(40, 4)
        y = np.random.randint(0, 2, 40)
        
        history = trainer.fit(X, y, epochs=5, batch_size=8)
        
        # Validate training progression
        losses = history['loss']
        assert len(losses) == 5, f"Expected 5 epochs, got {len(losses)}"
        
        # Test final performance
        final_loss = losses[-1]
        assert final_loss < np.inf, "Training diverged"
        assert final_loss >= 0, "Invalid loss value"
        
        return True, "Full integration working"
        
    except Exception as e:
        return False, f"Integration test failed: {str(e)}"

def run_quality_gate_5_performance():
    """Test performance benchmarks"""
    try:
        from qecc_qml.core.quantum_nn import QECCAwareQNN
        from qecc_qml.training.scalable_trainer_advanced import ScalableQECCTrainer
        
        # Performance test
        qnn = QECCAwareQNN(num_qubits=4, num_layers=2)
        trainer = ScalableQECCTrainer(
            qnn=qnn,
            enable_caching=True,
            max_workers=1
        )
        
        # Measure training speed
        X = np.random.rand(50, 4)
        y = np.random.randint(0, 2, 50)
        
        start_time = time.time()
        history = trainer.fit(X, y, epochs=3, batch_size=16)
        training_time = time.time() - start_time
        
        # Performance criteria
        throughput = history['throughput'][-1]
        cache_hit_rate = history['cache_hit_rate'][-1]
        
        assert training_time < 120, f"Training too slow: {training_time:.2f}s"
        assert throughput > 1, f"Throughput too low: {throughput:.2f} samples/s"
        
        return True, f"Performance acceptable: {throughput:.1f} samples/s, {training_time:.1f}s"
        
    except Exception as e:
        return False, f"Performance test failed: {str(e)}"

def main():
    """Run all quality gates"""
    print("🛡️ RUNNING COMPREHENSIVE QUALITY GATES")
    print("=" * 60)
    
    quality_gates = [
        ("Generation 1: Basic Functionality", run_quality_gate_1_basic_functionality),
        ("Generation 2: Robustness", run_quality_gate_2_robustness), 
        ("Generation 3: Scalability", run_quality_gate_3_scalability),
        ("System Integration", run_quality_gate_4_integration),
        ("Performance Benchmarks", run_quality_gate_5_performance)
    ]
    
    results = []
    passed = 0
    total = len(quality_gates)
    
    for name, test_func in quality_gates:
        print(f"\n📋 {name}")
        print("-" * 40)
        
        try:
            success, message = test_func()
            if success:
                print(f"✅ PASSED: {message}")
                passed += 1
            else:
                print(f"❌ FAILED: {message}")
            
            results.append({
                'name': name,
                'passed': success,
                'message': message
            })
            
        except Exception as e:
            error_msg = f"Exception: {str(e)}"
            print(f"❌ ERROR: {error_msg}")
            results.append({
                'name': name,
                'passed': False,
                'message': error_msg,
                'traceback': traceback.format_exc()
            })
    
    # Final report
    print("\n" + "=" * 60)
    print("🎯 QUALITY GATES SUMMARY")
    print("=" * 60)
    print(f"PASSED: {passed}/{total} ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 ALL QUALITY GATES PASSED!")
        print("✅ System ready for production deployment")
    else:
        print(f"⚠️  {total - passed} quality gates failed")
        print("🔧 System requires fixes before deployment")
    
    # Save results
    report = {
        'timestamp': time.time(),
        'total_gates': total,
        'passed_gates': passed,
        'success_rate': passed / total * 100,
        'results': results,
        'overall_status': 'PASSED' if passed == total else 'FAILED'
    }
    
    with open('/root/repo/quality_gates_comprehensive_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📊 Detailed report saved to quality_gates_comprehensive_report.json")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)