#!/usr/bin/env python3
"""
Quick script to fix qiskit imports across the codebase.
"""

import os
import re

def fix_qiskit_imports(file_path):
    """Fix qiskit imports in a Python file."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Skip files that already have fallback imports
    if 'fallback_imports' in content:
        return False
    
    # Store original content
    original_content = content
    
    # Replace qiskit imports
    replacements = [
        (r'from qiskit import (.+)', r'try:\n    from qiskit import \1\nexcept ImportError:\n    from qecc_qml.core.fallback_imports import \1'),
        (r'from qiskit\.circuit import (.+)', r'try:\n    from qiskit.circuit import \1\nexcept ImportError:\n    from qecc_qml.core.fallback_imports import \1'),
        (r'from qiskit\.quantum_info import (.+)', r'try:\n    from qiskit.quantum_info import \1\nexcept ImportError:\n    from qecc_qml.core.fallback_imports import \1'),
        (r'from qiskit_aer import (.+)', r'try:\n    from qiskit_aer import \1\nexcept ImportError:\n    from qecc_qml.core.fallback_imports import \1'),
    ]
    
    for pattern, replacement in replacements:
        content = re.sub(pattern, replacement, content)
    
    # Only write if changes were made
    if content != original_content:
        with open(file_path, 'w') as f:
            f.write(content)
        return True
    return False

def main():
    """Fix imports in all Python files."""
    python_files = []
    
    # Find all Python files
    for root, dirs, files in os.walk('.'):
        # Skip certain directories
        if any(skip in root for skip in ['.git', '__pycache__', '.pytest_cache']):
            continue
            
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    print(f"Found {len(python_files)} Python files")
    
    fixed_count = 0
    for file_path in python_files:
        try:
            if fix_qiskit_imports(file_path):
                print(f"Fixed: {file_path}")
                fixed_count += 1
        except Exception as e:
            print(f"Error fixing {file_path}: {e}")
    
    print(f"Fixed imports in {fixed_count} files")

if __name__ == "__main__":
    main()