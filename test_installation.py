#!/usr/bin/env python3
"""
Test script to verify that the quantum machine learning framework dependencies
are installed correctly and basic functionality works.
"""

import sys

def test_basic_imports():
    """Test that all essential packages can be imported."""
    print("Testing basic scientific computing packages...")
    
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__} imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import NumPy: {e}")
        return False
    
    try:
        import scipy
        print(f"✓ SciPy {scipy.__version__} imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import SciPy: {e}")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("✓ Matplotlib imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import Matplotlib: {e}")
        return False
    
    try:
        import networkx as nx
        print(f"✓ NetworkX {nx.__version__} imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import NetworkX: {e}")
        return False
    
    try:
        import sympy as sp
        print(f"✓ SymPy {sp.__version__} imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import SymPy: {e}")
        return False
    
    return True

def test_quantum_packages():
    """Test quantum computing packages."""
    print("\nTesting quantum computing packages...")
    
    try:
        import qiskit
        print(f"✓ Qiskit {qiskit.__version__} imported successfully")
        
        from qiskit import QuantumCircuit
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        print("✓ Basic quantum circuit creation works")
        
        return True
    except ImportError as e:
        print(f"! Qiskit not available: {e}")
        print("! Framework will use fallback implementations")
        return True  # This is acceptable

def test_ml_packages():
    """Test machine learning packages."""
    print("\nTesting machine learning packages...")
    
    try:
        import pandas as pd
        print("✓ Pandas imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import Pandas: {e}")
        return False
    
    try:
        import sklearn
        print(f"✓ Scikit-learn {sklearn.__version__} imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import Scikit-learn: {e}")
        return False
    
    try:
        import joblib
        print("✓ Joblib imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import Joblib: {e}")
        return False
    
    try:
        import tqdm
        print("✓ TQDM imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import TQDM: {e}")
        return False
    
    return True

def test_framework_components():
    """Test the quantum machine learning framework components."""
    print("\nTesting quantum ML framework components...")
    
    try:
        from qecc_qml.core.quantum_nn import QECCAwareQNN
        qnn = QECCAwareQNN(num_qubits=3, num_layers=2)
        print("✓ QECCAwareQNN created successfully")
    except Exception as e:
        print(f"✗ Failed to create QECCAwareQNN: {e}")
        return False
    
    try:
        from qecc_qml.training.basic_trainer import BasicQECCTrainer
        trainer = BasicQECCTrainer(qnn=qnn, learning_rate=0.01)
        print("✓ BasicQECCTrainer created successfully")
    except Exception as e:
        print(f"✗ Failed to create BasicQECCTrainer: {e}")
        return False
    
    try:
        from qecc_qml.datasets.simple_datasets import load_quantum_classification
        X, y = load_quantum_classification(n_samples=5, n_features=3)
        print(f"✓ Dataset generated: X shape={len(X)}, y shape={len(y)}")
    except Exception as e:
        print(f"✗ Failed to generate dataset: {e}")
        return False
    
    return True

def main():
    """Run all tests."""
    print("=" * 60)
    print("QUANTUM MACHINE LEARNING FRAMEWORK INSTALLATION TEST")
    print("=" * 60)
    
    all_passed = True
    
    # Test basic imports
    if not test_basic_imports():
        all_passed = False
    
    # Test quantum packages
    if not test_quantum_packages():
        all_passed = False
    
    # Test ML packages
    if not test_ml_packages():
        all_passed = False
    
    # Test framework components
    if not test_framework_components():
        all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL TESTS PASSED - Installation successful!")
        print("The quantum machine learning framework is ready to use.")
    else:
        print("✗ Some tests failed - Please check the installation.")
        sys.exit(1)
    print("=" * 60)

if __name__ == "__main__":
    main()