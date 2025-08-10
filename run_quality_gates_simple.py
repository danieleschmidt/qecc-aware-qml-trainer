#!/usr/bin/env python3
"""
Simplified quality gates execution for QECC-QML framework.
"""

import sys
import os
import time
import json

# Add current directory to path
sys.path.insert(0, '/root/repo')

def run_test(test_name, test_function):
    """Run a single test and report results."""
    print(f"\n🔄 Running {test_name}...")
    
    try:
        start_time = time.time()
        result = test_function()
        execution_time = time.time() - start_time
        
        if result:
            print(f"✅ {test_name} PASSED ({execution_time:.2f}s)")
            return True, execution_time
        else:
            print(f"❌ {test_name} FAILED ({execution_time:.2f}s)")
            return False, execution_time
            
    except Exception as e:
        execution_time = time.time() - start_time
        print(f"🚫 {test_name} ERROR: {str(e)} ({execution_time:.2f}s)")
        return False, execution_time

def test_imports():
    """Test critical imports."""
    try:
        import numpy as np
        print("✓ NumPy imported")
        
        import qiskit
        print("✓ Qiskit imported")
        
        import qiskit_aer
        print("✓ Qiskit Aer imported")
        
        from qiskit import QuantumCircuit
        print("✓ QuantumCircuit imported")
        
        import qecc_qml
        print("✓ QECC-QML imported")
        
        from qecc_qml.core.quantum_nn import QECCAwareQNN
        print("✓ QECCAwareQNN imported")
        
        from qecc_qml.core.circuit_validation import CircuitValidator
        print("✓ CircuitValidator imported")
        
        return True
    except Exception as e:
        print(f"Import error: {e}")
        return False

def test_basic_functionality():
    """Test basic framework functionality."""
    try:
        import numpy as np
        from qiskit import QuantumCircuit
        
        # Test circuit creation
        circuit = QuantumCircuit(4, 4)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.measure_all()
        print("✓ Quantum circuit created")
        
        # Test QNN creation
        from qecc_qml.core.quantum_nn import QECCAwareQNN
        qnn = QECCAwareQNN(num_qubits=3, num_layers=2)
        assert qnn.num_qubits == 3
        print("✓ QECCAwareQNN created")
        
        # Test validation
        from qecc_qml.core.circuit_validation import CircuitValidator
        validator = CircuitValidator()
        result = validator.validate_circuit(circuit)
        assert 'valid' in result
        print("✓ Circuit validation working")
        
        return True
    except Exception as e:
        print(f"Basic functionality error: {e}")
        return False

def test_monitoring_system():
    """Test monitoring capabilities."""
    try:
        from qecc_qml.monitoring.health_monitor import HealthMonitor
        
        monitor = HealthMonitor(monitoring_interval=0.1)
        print("✓ HealthMonitor created")
        
        monitor.start_monitoring()
        time.sleep(0.3)
        monitor.stop_monitoring()
        print("✓ Health monitoring lifecycle")
        
        health = monitor.get_current_health()
        assert 'timestamp' in health
        print("✓ Health status retrieval")
        
        return True
    except Exception as e:
        print(f"Monitoring system error: {e}")
        return False

def test_validation_framework():
    """Test validation framework."""
    try:
        import numpy as np
        from qecc_qml.validation.comprehensive_validation import ComprehensiveValidator
        
        validator = ComprehensiveValidator()
        print("✓ ComprehensiveValidator created")
        
        # Test data validation
        X = np.random.random((20, 8))
        y = np.random.randint(0, 2, 20)
        
        results = validator.validate_training_inputs(X, y)
        assert isinstance(results, list)
        print("✓ Training data validation")
        
        return True
    except Exception as e:
        print(f"Validation framework error: {e}")
        return False

def test_optimization_components():
    """Test optimization components."""
    try:
        import numpy as np
        from qiskit import QuantumCircuit
        from qecc_qml.optimization.quantum_circuit_optimization import QuantumCircuitOptimizer
        
        optimizer = QuantumCircuitOptimizer()
        print("✓ QuantumCircuitOptimizer created")
        
        # Create test circuit
        circuit = QuantumCircuit(3)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.cx(1, 2)
        
        optimized, result = optimizer.optimize_circuit(circuit)
        assert isinstance(optimized, QuantumCircuit)
        print("✓ Circuit optimization working")
        
        return True
    except Exception as e:
        print(f"Optimization error: {e}")
        return False

def test_error_recovery():
    """Test error recovery system."""
    try:
        from qecc_qml.utils.error_recovery import ErrorRecoveryManager
        
        recovery_manager = ErrorRecoveryManager(max_retries=2)
        print("✓ ErrorRecoveryManager created")
        
        call_count = 0
        
        @recovery_manager.retry_with_backoff(max_retries=2)
        def test_function():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("Test error")
            return "success"
        
        result = test_function()
        assert result == "success"
        assert call_count == 2
        print("✓ Error recovery working")
        
        return True
    except Exception as e:
        print(f"Error recovery error: {e}")
        return False

def test_performance():
    """Test performance requirements."""
    try:
        import numpy as np
        from qiskit import QuantumCircuit
        from qecc_qml.optimization.quantum_circuit_optimization import QuantumCircuitOptimizer
        
        # Test optimization performance
        circuit = QuantumCircuit(5)
        for i in range(3):
            for q in range(5):
                circuit.ry(np.pi * 0.5, q)
            for q in range(4):
                circuit.cx(q, q + 1)
        
        optimizer = QuantumCircuitOptimizer()
        start_time = time.time()
        optimized, result = optimizer.optimize_circuit(circuit)
        optimization_time = time.time() - start_time
        
        assert optimization_time < 5.0  # Should be fast
        print(f"✓ Circuit optimization time: {optimization_time:.3f}s")
        
        # Test validation performance
        from qecc_qml.validation.comprehensive_validation import ComprehensiveValidator
        validator = ComprehensiveValidator()
        
        X = np.random.random((100, 10))
        y = np.random.randint(0, 2, 100)
        
        start_time = time.time()
        results = validator.validate_training_inputs(X, y)
        validation_time = time.time() - start_time
        
        assert validation_time < 1.0  # Should be fast
        print(f"✓ Validation time: {validation_time:.3f}s")
        
        return True
    except Exception as e:
        print(f"Performance test error: {e}")
        return False

def main():
    """Run all quality gates."""
    print("🚀 QECC-QML QUALITY GATES EXECUTION")
    print("=" * 50)
    
    # Define test suite
    tests = [
        ("Core Imports", test_imports),
        ("Basic Functionality", test_basic_functionality),
        ("Monitoring System", test_monitoring_system),
        ("Validation Framework", test_validation_framework),
        ("Optimization Components", test_optimization_components),
        ("Error Recovery", test_error_recovery),
        ("Performance Tests", test_performance)
    ]
    
    results = {}
    total_time = 0
    
    # Run all tests
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name.upper()} {'='*20}")
        success, exec_time = run_test(test_name, test_func)
        
        results[test_name] = {
            "success": success,
            "execution_time": exec_time,
            "status": "PASS" if success else "FAIL"
        }
        total_time += exec_time
    
    # Generate report
    print("\n" + "="*50)
    print("📊 QUALITY GATES SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for r in results.values() if r["success"])
    total = len(results)
    success_rate = (passed / total) * 100
    
    print(f"Total Tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Success Rate: {success_rate:.1f}%")
    print(f"Total Time: {total_time:.2f}s")
    
    # Detailed results
    print(f"\n📋 DETAILED RESULTS:")
    for test_name, result in results.items():
        status_emoji = "✅" if result["success"] else "❌"
        print(f"{status_emoji} {test_name}: {result['status']} ({result['execution_time']:.2f}s)")
    
    # Save report
    report = {
        "timestamp": time.time(),
        "framework": "QECC-QML",
        "version": "0.1.0",
        "summary": {
            "total_tests": total,
            "passed_tests": passed,
            "failed_tests": total - passed,
            "success_rate": success_rate,
            "total_execution_time": total_time
        },
        "detailed_results": results,
        "overall_status": "PASS" if passed == total else "FAIL"
    }
    
    try:
        with open("/root/repo/quality_gates_report.json", "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n📈 Report saved to quality_gates_report.json")
    except Exception as e:
        print(f"⚠️  Could not save report: {e}")
    
    # Final status
    overall_status = "PASS" if passed == total else "FAIL"
    status_emoji = "🎉" if overall_status == "PASS" else "💥"
    
    print(f"\n{status_emoji} OVERALL STATUS: {overall_status}")
    
    return overall_status == "PASS"

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)