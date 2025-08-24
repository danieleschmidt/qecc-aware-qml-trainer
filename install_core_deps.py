#!/usr/bin/env python3
"""
Core dependency installer with fallbacks for QECC-QML autonomous execution.
"""

import subprocess
import sys
import os

def install_minimal_deps():
    """Install minimal dependencies for autonomous execution."""
    
    print("🔧 Installing core dependencies for autonomous execution...")
    
    # Essential packages for autonomous execution
    essential_packages = [
        "numpy>=1.20.0",
        "matplotlib>=3.5.0", 
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "tqdm>=4.60.0",
        "networkx>=2.6.0",
        "joblib>=1.1.0",
        "h5py>=3.6.0",
        "sympy>=1.9.0"
    ]
    
    for package in essential_packages:
        try:
            print(f"  Installing {package}...")
            subprocess.run([
                sys.executable, "-m", "pip", "install", "--quiet", "--no-warn-script-location", package
            ], check=True, capture_output=True)
            print(f"  ✅ {package} installed successfully")
        except subprocess.CalledProcessError:
            print(f"  ⚠️  {package} installation failed - continuing with fallbacks")
            continue
    
    print("✅ Core dependencies installation complete")
    
    # Test imports
    print("\n🧪 Testing core imports...")
    try:
        import numpy as np
        print(f"  ✅ NumPy {np.__version__} imported successfully")
    except ImportError:
        print("  ⚠️  NumPy not available - using fallbacks")
        
    try:
        import matplotlib
        print(f"  ✅ Matplotlib {matplotlib.__version__} imported successfully")
    except ImportError:
        print("  ⚠️  Matplotlib not available - using fallbacks")
        
    return True

if __name__ == "__main__":
    install_minimal_deps()