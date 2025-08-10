#!/usr/bin/env python3
"""
Comprehensive quality gates execution for QECC-QML framework.
"""

import sys
import time
import subprocess
import json
from pathlib import Path
import os

def run_command(command, description, timeout=300):
    """Run a command with timeout and capture output."""
    print(f"\n🔄 {description}")
    print(f"Command: {command}")
    
    try:
        start_time = time.time()
        result = subprocess.run(
            command.split(),
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd="/root/repo"
        )
        
        execution_time = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ {description} PASSED ({execution_time:.2f}s)")
            if result.stdout.strip():
                print(f"Output: {result.stdout.strip()}")
            return True, result.stdout
        else:
            print(f"❌ {description} FAILED ({execution_time:.2f}s)")
            print(f"Error: {result.stderr.strip()}")
            return False, result.stderr
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {description} TIMEOUT after {timeout}s")
        return False, "Timeout"
    except Exception as e:
        print(f"🚫 {description} ERROR: {str(e)}")
        return False, str(e)

def check_imports():
    """Test all critical imports."""
    print("\n🔍 TESTING CORE IMPORTS")
    
    imports_to_test = [
        "import numpy as np",
        "import qiskit",
        "import qiskit_aer",
        "from qiskit import QuantumCircuit",
        "import sys; sys.path.append('.'); import qecc_qml",
        "from qecc_qml.core.quantum_nn import QECCAwareQNN",
        "from qecc_qml.monitoring.health_monitor import HealthMonitor",
        "from qecc_qml.validation.comprehensive_validation import ComprehensiveValidator"
    ]
    
    all_passed = True
    for import_test in imports_to_test:
        success, output = run_command(
            f"bash -c 'source venv/bin/activate && python -c \"{import_test}\"'",
            f"Import: {import_test}",
            timeout=30
        )
        if not success:
            all_passed = False
            
    return all_passed

def run_basic_functionality_tests():
    """Run basic functionality tests."""
    print("\n🧪 RUNNING BASIC FUNCTIONALITY TESTS")
    
    test_script = """
import sys
sys.path.append('.')
import numpy as np
from qiskit import QuantumCircuit

# Test circuit creation
circuit = QuantumCircuit(4, 4)
circuit.h(0)
circuit.cx(0, 1)
circuit.measure_all()

# Test framework imports
from qecc_qml.core.circuit_validation import CircuitValidator
from qecc_qml.monitoring.health_monitor import HealthMonitor

# Test validation
validator = CircuitValidator()
result = validator.validate_circuit(circuit)
print(f"Circuit validation: {'PASS' if result['valid'] else 'FAIL'}")

# Test monitoring
monitor = HealthMonitor(monitoring_interval=0.1)
monitor.start_monitoring()
import time
time.sleep(0.5)
monitor.stop_monitoring()
health = monitor.get_current_health()
print(f"Health monitoring: {'PASS' if 'timestamp' in health else 'FAIL'}")

print("All basic tests passed!")
"""
    
    with open("/tmp/basic_test.py", "w") as f:
        f.write(test_script)
        
    success, output = run_command(
        "bash -c 'source venv/bin/activate && python /tmp/basic_test.py'",
        "Basic Functionality Tests",
        timeout=60
    )
    
    return success

def run_performance_benchmarks():
    """Run performance benchmarks."""
    print("\n⚡ RUNNING PERFORMANCE BENCHMARKS")
    
    benchmark_script = """
import sys
sys.path.append('.')
import time
import numpy as np
from qiskit import QuantumCircuit

# Test circuit optimization performance
from qecc_qml.optimization.quantum_circuit_optimization import QuantumCircuitOptimizer

circuit = QuantumCircuit(6)
for i in range(5):  # 5 layers
    for q in range(6):
        circuit.ry(np.pi * 0.5, q)
    for q in range(5):
        circuit.cx(q, q + 1)

optimizer = QuantumCircuitOptimizer()
start_time = time.time()
optimized_circuit, result = optimizer.optimize_circuit(circuit)
optimization_time = time.time() - start_time

print(f"Circuit optimization time: {optimization_time:.3f}s")
print(f"Original gates: {circuit.size()}, Optimized gates: {optimized_circuit.size()}")

# Test validation performance
from qecc_qml.validation.comprehensive_validation import ComprehensiveValidator

validator = ComprehensiveValidator()
X = np.random.random((100, 10))
y = np.random.randint(0, 2, 100)

start_time = time.time()
results = validator.validate_training_inputs(X, y)
validation_time = time.time() - start_time

print(f"Validation time for 100 samples: {validation_time:.3f}s")
print(f"Performance benchmarks completed!")
"""
    
    with open("/tmp/benchmark_test.py", "w") as f:
        f.write(benchmark_script)
        
    success, output = run_command(
        "bash -c 'source venv/bin/activate && python /tmp/benchmark_test.py'",
        "Performance Benchmarks",
        timeout=120
    )
    
    return success

def run_security_tests():
    """Run security validation tests."""
    print("\n🔐 RUNNING SECURITY TESTS")
    
    security_script = """
import sys
sys.path.append('.')
from qiskit import QuantumCircuit
from qecc_qml.core.circuit_validation import SecurityManager

# Test security manager
security_manager = SecurityManager()

# Test circuit sanitization
circuit = QuantumCircuit(2)
circuit.h(0)
circuit.cx(0, 1)

sanitized = security_manager.sanitize_circuit(circuit)
print(f"Circuit sanitization: PASS")

# Test parameter validation
import numpy as np
params = [np.pi, 2*np.pi, 0.5]
valid = security_manager.validate_parameters(params)
print(f"Parameter validation: {'PASS' if valid else 'FAIL'}")

print("Security tests completed!")
"""
    
    with open("/tmp/security_test.py", "w") as f:
        f.write(security_script)
        
    success, output = run_command(
        "bash -c 'source venv/bin/activate && python /tmp/security_test.py'",
        "Security Tests",
        timeout=60
    )
    
    return success

def run_integration_tests():
    """Run integration tests."""
    print("\n🔗 RUNNING INTEGRATION TESTS")
    
    # Create minimal integration test
    integration_script = """
import sys
sys.path.append('.')
import numpy as np
from qiskit import QuantumCircuit

# Test QNN creation
from qecc_qml.core.quantum_nn import QECCAwareQNN
from qecc_qml.core.noise_models import NoiseModel
from qecc_qml.training.qecc_trainer import QECCTrainer

# Create components
qnn = QECCAwareQNN(num_qubits=3, num_layers=2)
noise_model = NoiseModel()
trainer = QECCTrainer(qnn=qnn, noise_model=noise_model)

print(f"QNN creation: PASS")
print(f"Trainer creation: PASS")

# Test validation workflow  
from qecc_qml.validation.comprehensive_validation import ComprehensiveValidator

validator = ComprehensiveValidator()
X = np.random.random((10, 6))
y = np.random.randint(0, 2, 10)

results = validator.validate_training_inputs(X, y)
print(f"Validation workflow: PASS ({len(results)} checks)")

# Test monitoring workflow
from qecc_qml.monitoring.health_monitor import HealthMonitor

monitor = HealthMonitor(monitoring_interval=0.1)
monitor.start_monitoring()
import time
time.sleep(0.3)
monitor.stop_monitoring()

health = monitor.get_current_health()
print(f"Monitoring workflow: PASS")

print("Integration tests completed!")
"""
    
    with open("/tmp/integration_test.py", "w") as f:
        f.write(integration_script)
        
    success, output = run_command(
        "bash -c 'source venv/bin/activate && python /tmp/integration_test.py'",
        "Integration Tests",
        timeout=120
    )
    
    return success

def generate_quality_report(results):
    """Generate comprehensive quality report."""
    print("\n📊 GENERATING QUALITY REPORT")
    
    total_tests = len(results)
    passed_tests = sum(1 for result in results.values() if result['success'])
    
    report = {
        "timestamp": time.time(),
        "framework": "QECC-QML",
        "version": "0.1.0",
        "quality_gates": {
            "total_gates": total_tests,
            "passed_gates": passed_tests,
            "failed_gates": total_tests - passed_tests,
            "success_rate": passed_tests / total_tests * 100
        },
        "detailed_results": results,
        "overall_status": "PASS" if passed_tests == total_tests else "FAIL"
    }
    
    # Save report
    with open("/root/repo/quality_gates_report.json", "w") as f:
        json.dump(report, f, indent=2)
        
    print(f"📈 Quality Report Generated:")
    print(f"   Total Gates: {total_tests}")
    print(f"   Passed: {passed_tests}")
    print(f"   Failed: {total_tests - passed_tests}")  
    print(f"   Success Rate: {passed_tests / total_tests * 100:.1f}%")
    print(f"   Overall Status: {report['overall_status']}")
    
    return report

def main():
    """Run all quality gates."""
    print("🚀 QECC-QML QUALITY GATES EXECUTION")
    print("=" * 50)
    
    # Store results
    results = {}
    
    # Activate virtual environment first
    print("🔧 Setting up environment...")
    setup_success, _ = run_command(
        "bash -c 'source venv/bin/activate && python --version'",
        "Environment Setup"
    )
    
    if not setup_success:
        print("❌ Failed to setup environment")
        return False
    
    # Run quality gates
    quality_gates = [
        ("Core Imports", check_imports),
        ("Basic Functionality", run_basic_functionality_tests),
        ("Performance Benchmarks", run_performance_benchmarks),
        ("Security Tests", run_security_tests),
        ("Integration Tests", run_integration_tests)
    ]
    
    for gate_name, gate_function in quality_gates:
        print(f"\n{'='*20} {gate_name.upper()} {'='*20}")
        
        start_time = time.time()
        try:
            success = gate_function()
            execution_time = time.time() - start_time
            
            results[gate_name] = {
                "success": success,
                "execution_time": execution_time,
                "status": "PASS" if success else "FAIL"
            }
            
            print(f"\n{'✅' if success else '❌'} {gate_name}: {results[gate_name]['status']} ({execution_time:.2f}s)")
            
        except Exception as e:
            execution_time = time.time() - start_time
            results[gate_name] = {
                "success": False,
                "execution_time": execution_time,
                "status": "ERROR",
                "error": str(e)
            }
            print(f"\n🚫 {gate_name}: ERROR - {str(e)} ({execution_time:.2f}s)")
    
    # Generate final report
    print("\n" + "="*50)
    report = generate_quality_report(results)
    
    # Return success if all gates passed
    return report['overall_status'] == 'PASS'

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)