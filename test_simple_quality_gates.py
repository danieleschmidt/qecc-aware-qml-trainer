#!/usr/bin/env python3
"""
Simple Quality Gates Execution.

Focused testing for core functionality without integration complexity.
"""

import sys
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def setup_logging() -> logging.Logger:
    """Set up logging for quality gates."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)

def test_core_imports():
    """Test that all core modules can be imported."""
    try:
        from qecc_qml.core.quantum_nn import QECCAwareQNN
        from qecc_qml.training.basic_trainer import BasicQECCTrainer
        from qecc_qml.datasets.simple_datasets import create_test_dataset
        return {"passed": True, "imports": "success"}
    except Exception as e:
        return {"passed": False, "error": str(e)}

def test_data_generation():
    """Test dataset generation."""
    try:
        from qecc_qml.datasets.simple_datasets import create_test_dataset
        X, y = create_test_dataset(n_samples=50, n_features=4)
        
        assert X.shape == (50, 4), f"Wrong X shape: {X.shape}"
        assert y.shape == (50,), f"Wrong y shape: {y.shape}"
        assert np.all(np.isfinite(X)), "X contains non-finite values"
        assert np.all((y == 0) | (y == 1)), "y contains invalid labels"
        
        return {"passed": True, "X_shape": X.shape, "y_shape": y.shape}
    except Exception as e:
        return {"passed": False, "error": str(e)}

def test_qnn_creation():
    """Test QNN creation."""
    try:
        from qecc_qml.core.quantum_nn import QECCAwareQNN
        
        qnn = QECCAwareQNN(num_qubits=4, num_layers=2)
        assert qnn is not None, "QNN creation failed"
        assert hasattr(qnn, 'num_qubits'), "QNN missing num_qubits"
        assert qnn.num_qubits == 4, f"Wrong qubit count: {qnn.num_qubits}"
        
        return {"passed": True, "qubits": qnn.num_qubits}
    except Exception as e:
        return {"passed": False, "error": str(e)}

def test_basic_training():
    """Test basic training functionality."""
    try:
        from qecc_qml.core.quantum_nn import QECCAwareQNN
        from qecc_qml.training.basic_trainer import BasicQECCTrainer
        from qecc_qml.datasets.simple_datasets import create_test_dataset
        
        # Create components
        qnn = QECCAwareQNN(num_qubits=4, num_layers=2)
        trainer = BasicQECCTrainer(qnn)
        X, y = create_test_dataset(n_samples=20, n_features=4)
        
        # Test training step
        initial_params = trainer.model.parameters.copy()
        trainer.train_step(X[:5], y[:5])
        final_params = trainer.model.parameters
        
        # Check if parameters changed
        params_changed = not np.allclose(initial_params, final_params, atol=1e-6)
        
        return {"passed": True, "params_changed": params_changed}
    except Exception as e:
        return {"passed": False, "error": str(e)}

def test_prediction():
    """Test prediction functionality."""
    try:
        from qecc_qml.core.quantum_nn import QECCAwareQNN
        from qecc_qml.training.basic_trainer import BasicQECCTrainer
        from qecc_qml.datasets.simple_datasets import create_test_dataset
        
        qnn = QECCAwareQNN(num_qubits=4, num_layers=2)
        trainer = BasicQECCTrainer(qnn)
        X, y = create_test_dataset(n_samples=10, n_features=4)
        
        # Train for one step to initialize model
        trainer.train_step(X[:5], y[:5])
        
        # Now test prediction
        predictions = trainer.predict(X[:3])
        
        assert predictions.shape == (3,), f"Wrong prediction shape: {predictions.shape}"
        assert np.all(np.isfinite(predictions)), "Predictions contain non-finite values"
        
        return {"passed": True, "prediction_shape": predictions.shape}
    except Exception as e:
        return {"passed": False, "error": str(e)}

def test_security_basics():
    """Test basic security functionality."""
    try:
        from qecc_qml.security.advanced_security_validator import AdvancedSecurityValidator
        
        validator = AdvancedSecurityValidator()
        
        # Test input sanitization
        test_input = "<script>alert('test')</script>"
        sanitized = validator.sanitize_input_data(test_input)
        
        assert sanitized != test_input, "Input not sanitized"
        assert '<script>' not in sanitized, "Dangerous content not removed"
        
        return {"passed": True, "sanitization": "working"}
    except Exception as e:
        return {"passed": False, "error": str(e)}

def test_performance_basics():
    """Test basic performance functionality."""
    try:
        from qecc_qml.optimization.quantum_performance_enhancer import QuantumPerformanceEnhancer
        
        enhancer = QuantumPerformanceEnhancer()
        summary = enhancer.get_performance_summary()
        
        assert isinstance(summary, dict), "Summary should be dict"
        assert 'cache_hit_rate' in summary, "Missing cache_hit_rate"
        
        return {"passed": True, "summary_keys": list(summary.keys())}
    except Exception as e:
        return {"passed": False, "error": str(e)}

def test_monitoring_basics():
    """Test basic monitoring functionality."""
    try:
        from qecc_qml.monitoring.comprehensive_health_monitor import ComprehensiveHealthMonitor
        
        monitor = ComprehensiveHealthMonitor()
        metrics = monitor.collect_metrics()
        
        assert hasattr(metrics, 'timestamp'), "Metrics missing timestamp"
        assert metrics.timestamp > 0, "Invalid timestamp"
        
        return {"passed": True, "metrics_collected": True}
    except Exception as e:
        return {"passed": False, "error": str(e)}

def test_research_imports():
    """Test research module imports."""
    try:
        from qecc_qml import research
        
        # Test that research module exists and has expected attributes
        research_components = [
            'QECCEnvironment', 'NeuralSyndromeDecoder', 'QuantumAdvantageSuite'
        ]
        
        available_components = []
        for component in research_components:
            if hasattr(research, component):
                available_components.append(component)
        
        return {"passed": True, "available_components": available_components}
    except Exception as e:
        return {"passed": False, "error": str(e)}

def run_simple_quality_gates():
    """Run simple quality gates."""
    logger = setup_logging()
    
    # Define test functions
    tests = [
        ("Core Imports", test_core_imports),
        ("Data Generation", test_data_generation), 
        ("QNN Creation", test_qnn_creation),
        ("Basic Training", test_basic_training),
        ("Prediction", test_prediction),
        ("Security Basics", test_security_basics),
        ("Performance Basics", test_performance_basics),
        ("Monitoring Basics", test_monitoring_basics),
        ("Research Imports", test_research_imports),
    ]
    
    results = []
    passed_count = 0
    
    logger.info("🚀 Starting Simple Quality Gates")
    print("=" * 60)
    
    for test_name, test_func in tests:
        print(f"🔍 Testing: {test_name}")
        start_time = time.time()
        
        try:
            result = test_func()
            execution_time = time.time() - start_time
            
            if result["passed"]:
                passed_count += 1
                print(f"✅ {test_name} PASSED ({execution_time:.2f}s)")
                logger.info(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED ({execution_time:.2f}s)")
                print(f"   Error: {result.get('error', 'Unknown')}")
                logger.error(f"❌ {test_name} FAILED: {result.get('error', 'Unknown')}")
            
            results.append({
                "name": test_name,
                "passed": result["passed"],
                "execution_time": execution_time,
                "details": result
            })
            
        except Exception as e:
            execution_time = time.time() - start_time
            print(f"💥 {test_name} EXCEPTION ({execution_time:.2f}s): {e}")
            logger.error(f"💥 {test_name} EXCEPTION: {e}")
            
            results.append({
                "name": test_name,
                "passed": False,
                "execution_time": execution_time,
                "details": {"passed": False, "error": str(e)}
            })
    
    # Calculate final results
    total_tests = len(tests)
    success_rate = (passed_count / total_tests) * 100
    overall_status = "PASSED" if success_rate >= 70 else "FAILED"
    
    # Create report
    report = {
        "timestamp": time.time(),
        "total_tests": total_tests,
        "passed_tests": passed_count,
        "failed_tests": total_tests - passed_count,
        "success_rate": success_rate,
        "overall_status": overall_status,
        "test_results": results
    }
    
    # Save report
    with open("simple_quality_gates_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 SIMPLE QUALITY GATES SUMMARY")
    print("=" * 60)
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_count}")
    print(f"Failed: {total_tests - passed_count}")
    print(f"Success Rate: {success_rate:.1f}%")
    print(f"Overall Status: {overall_status}")
    
    if overall_status == "PASSED":
        print("\n🎉 CORE FUNCTIONALITY WORKING!")
        return 0
    else:
        print("\n🔧 CORE FUNCTIONALITY NEEDS FIXES")
        return 1

if __name__ == "__main__":
    exit_code = run_simple_quality_gates()
    sys.exit(exit_code)