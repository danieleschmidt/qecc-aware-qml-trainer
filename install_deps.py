#!/usr/bin/env python3
"""Simple dependency installer for QECC-QML framework."""

import subprocess
import sys

# Core dependencies in order of installation
CORE_DEPS = [
    "numpy>=1.24.0",
    "scipy>=1.10.0", 
    "qiskit>=1.0.0",
    "torch>=2.0.0",
    "matplotlib>=3.7.0",
    "scikit-learn>=1.3.0",
    "tqdm>=4.65.0",
    "networkx>=3.1.0"
]

def install_package(package):
    """Install a single package."""
    try:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ {package} installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package}: {e}")
        return False

def main():
    """Install core dependencies."""
    print("🚀 Installing QECC-QML core dependencies...")
    
    success_count = 0
    for package in CORE_DEPS:
        if install_package(package):
            success_count += 1
    
    print(f"\n📊 Installation complete: {success_count}/{len(CORE_DEPS)} packages installed")
    
    # Test imports
    try:
        import numpy
        import qiskit
        print("✅ Core packages importable")
        return True
    except ImportError as e:
        print(f"❌ Import test failed: {e}")
        return False

if __name__ == "__main__":
    main()