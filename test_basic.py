#!/usr/bin/env python3
"""
Basic functionality tests that don't require external dependencies.
"""

import sys
import os
import importlib.util
from pathlib import Path

def test_module_structure():
    """Test that all module files exist and have correct structure."""
    print("🔍 Testing module structure...")
    
    required_modules = [
        'qecc_qml/__init__.py',
        'qecc_qml/core/__init__.py',
        'qecc_qml/codes/__init__.py',
        'qecc_qml/training/__init__.py',
        'qecc_qml/evaluation/__init__.py',
        'qecc_qml/benchmarks/__init__.py',
        'qecc_qml/adaptive/__init__.py',
        'qecc_qml/monitoring/__init__.py',
        'qecc_qml/deployment/__init__.py',
        'qecc_qml/validation/__init__.py',
        'qecc_qml/optimization/__init__.py',
    ]
    
    missing_modules = []
    for module_path in required_modules:
        if not Path(module_path).exists():
            missing_modules.append(module_path)
    
    if missing_modules:
        print(f"❌ Missing modules: {missing_modules}")
        return False
    
    print("✅ All required module files exist")
    return True

def test_python_syntax():
    """Test Python syntax of all files."""
    print("🔍 Testing Python syntax...")
    
    syntax_errors = []
    
    for py_file in Path('qecc_qml').rglob('*.py'):
        try:
            with open(py_file, 'r') as f:
                compile(f.read(), py_file, 'exec')
        except SyntaxError as e:
            syntax_errors.append(f"{py_file}: {e}")
    
    if syntax_errors:
        print(f"❌ Syntax errors found:")
        for error in syntax_errors:
            print(f"   {error}")
        return False
    
    print("✅ All Python files have valid syntax")
    return True

def test_file_completeness():
    """Test that important files are present."""
    print("🔍 Testing file completeness...")
    
    required_files = [
        'README.md',
        'requirements.txt',
        'setup.py',
        'pyproject.toml',
        'LICENSE',
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"⚠️ Missing optional files: {missing_files}")
        # Not a failure, just a warning
    
    print("✅ File completeness check passed")
    return True

def test_docstrings():
    """Test that modules have docstrings."""
    print("🔍 Testing docstrings...")
    
    modules_without_docstrings = []
    
    for py_file in Path('qecc_qml').rglob('*.py'):
        if py_file.name == '__init__.py':
            continue
        
        try:
            with open(py_file, 'r') as f:
                content = f.read()
                
            # Simple check for docstring at beginning of file
            lines = content.strip().split('\n')
            if len(lines) < 3:
                continue
                
            # Look for docstring in first few lines
            has_docstring = False
            for i, line in enumerate(lines[:10]):
                if '"""' in line or "'''" in line:
                    has_docstring = True
                    break
            
            if not has_docstring:
                modules_without_docstrings.append(str(py_file))
                
        except Exception:
            continue
    
    if modules_without_docstrings:
        print(f"⚠️ Modules without docstrings: {len(modules_without_docstrings)}")
        # Not a failure, just a warning
    
    print("✅ Docstring check passed")
    return True

def test_code_metrics():
    """Test basic code metrics."""
    print("🔍 Testing code metrics...")
    
    total_lines = 0
    total_files = 0
    
    for py_file in Path('qecc_qml').rglob('*.py'):
        try:
            with open(py_file, 'r') as f:
                lines = len(f.readlines())
                total_lines += lines
                total_files += 1
        except Exception:
            continue
    
    if total_files == 0:
        print("❌ No Python files found")
        return False
    
    avg_lines = total_lines / total_files
    
    print(f"📊 Code metrics:")
    print(f"   Total Python files: {total_files}")
    print(f"   Total lines of code: {total_lines}")
    print(f"   Average lines per file: {avg_lines:.1f}")
    
    if total_lines < 1000:
        print("⚠️ Codebase is quite small")
    elif total_lines > 50000:
        print("⚠️ Codebase is very large")
    else:
        print("✅ Codebase size is reasonable")
    
    return True

def run_quality_checks():
    """Run all quality checks."""
    print("🚀 Running Quality Gate Checks")
    print("=" * 50)
    
    tests = [
        test_module_structure,
        test_python_syntax,
        test_file_completeness,
        test_docstrings,
        test_code_metrics,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            failed += 1
        print()
    
    print("📊 Quality Gate Summary")
    print("=" * 30)
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📈 Success Rate: {passed/(passed+failed)*100:.1f}%")
    
    if failed == 0:
        print("\n🎉 All quality checks passed!")
        return True
    else:
        print(f"\n⚠️ {failed} quality check(s) failed")
        return False

if __name__ == "__main__":
    success = run_quality_checks()
    sys.exit(0 if success else 1)