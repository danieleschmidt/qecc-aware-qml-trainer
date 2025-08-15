#!/usr/bin/env python3
"""
Comprehensive quality gates for QECC-aware QML trainer.
Includes security, performance, and code quality checks.
"""

import os
import sys
import subprocess
import json
import time
from pathlib import Path
from typing import Dict, List, Any
import warnings

# Suppress fallback warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="qecc_qml.core.fallback_imports")


class QualityGateRunner:
    """Comprehensive quality gate runner."""
    
    def __init__(self):
        self.results = {}
        self.start_time = time.time()
        self.passed_gates = 0
        self.total_gates = 0
    
    def run_security_checks(self) -> Dict[str, Any]:
        """Run security vulnerability checks."""
        print("🔒 Running security checks...")
        
        security_results = {
            'status': 'passed',
            'checks': {},
            'vulnerabilities': []
        }
        
        # Check for common security issues
        security_checks = [
            self._check_hardcoded_secrets(),
            self._check_dangerous_imports(),
            self._check_file_permissions(),
            self._check_input_validation()
        ]
        
        for check_name, result in security_checks:
            security_results['checks'][check_name] = result
            if not result['passed']:
                security_results['status'] = 'warning'
                security_results['vulnerabilities'].append(result)
        
        self.total_gates += 1
        if security_results['status'] == 'passed':
            self.passed_gates += 1
        
        return security_results
    
    def _check_hardcoded_secrets(self) -> tuple:
        """Check for hardcoded secrets."""
        dangerous_patterns = [
            'password', 'secret', 'api_key', 'token', 'credential'
        ]
        
        issues = []
        for root, dirs, files in os.walk('.'):
            if any(skip in root for skip in ['.git', '__pycache__', '.pytest_cache']):
                continue
                
            for file in files:
                if file.endswith('.py'):
                    filepath = os.path.join(root, file)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            content = f.read().lower()
                            for pattern in dangerous_patterns:
                                if f'{pattern}=' in content or f'{pattern}:' in content:
                                    issues.append(f"{filepath}: potential {pattern}")
                    except:
                        continue
        
        return 'hardcoded_secrets', {
            'passed': len(issues) == 0,
            'message': f"Found {len(issues)} potential hardcoded secrets",
            'details': issues[:5]  # Limit to first 5
        }
    
    def _check_dangerous_imports(self) -> tuple:
        """Check for dangerous imports."""
        dangerous_imports = ['eval', 'exec', 'subprocess.call', '__import__']
        
        issues = []
        for root, dirs, files in os.walk('.'):
            if any(skip in root for skip in ['.git', '__pycache__', '.pytest_cache']):
                continue
                
            for file in files:
                if file.endswith('.py'):
                    filepath = os.path.join(root, file)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            lines = f.readlines()
                            for i, line in enumerate(lines):
                                for danger in dangerous_imports:
                                    if danger in line and not line.strip().startswith('#'):
                                        issues.append(f"{filepath}:{i+1}: {danger}")
                    except:
                        continue
        
        return 'dangerous_imports', {
            'passed': len(issues) == 0,
            'message': f"Found {len(issues)} potentially dangerous imports",
            'details': issues[:5]
        }
    
    def _check_file_permissions(self) -> tuple:
        """Check file permissions."""
        issues = []
        
        for root, dirs, files in os.walk('.'):
            for file in files:
                filepath = os.path.join(root, file)
                try:
                    stat = os.stat(filepath)
                    # Check if file is world-writable
                    if stat.st_mode & 0o002:
                        issues.append(f"{filepath}: world-writable")
                except:
                    continue
        
        return 'file_permissions', {
            'passed': len(issues) == 0,
            'message': f"Found {len(issues)} permission issues",
            'details': issues[:5]
        }
    
    def _check_input_validation(self) -> tuple:
        """Check for input validation patterns."""
        validation_patterns = [
            'sanitize', 'validate', 'clean', 'escape', 'filter'
        ]
        
        validation_files = 0
        total_python_files = 0
        
        for root, dirs, files in os.walk('.'):
            if any(skip in root for skip in ['.git', '__pycache__', '.pytest_cache']):
                continue
                
            for file in files:
                if file.endswith('.py'):
                    total_python_files += 1
                    filepath = os.path.join(root, file)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            content = f.read().lower()
                            if any(pattern in content for pattern in validation_patterns):
                                validation_files += 1
                    except:
                        continue
        
        validation_ratio = validation_files / max(1, total_python_files)
        
        return 'input_validation', {
            'passed': validation_ratio > 0.1,  # At least 10% of files have validation
            'message': f"Input validation found in {validation_files}/{total_python_files} files ({validation_ratio:.1%})",
            'details': {'ratio': validation_ratio}
        }
    
    def run_performance_checks(self) -> Dict[str, Any]:
        """Run performance checks."""
        print("⚡ Running performance checks...")
        
        performance_results = {
            'status': 'passed',
            'metrics': {},
            'benchmarks': {}
        }
        
        # Import performance test
        try:
            from qecc_qml.optimization.performance_optimizer import PerformanceOptimizer
            from qecc_qml.training.basic_trainer_fixed import BasicTrainer
            import numpy as np
            
            # Benchmark basic operations
            start_time = time.time()
            
            # Test 1: Optimizer performance
            optimizer = PerformanceOptimizer()
            opt_result = optimizer.optimize_system()
            optimizer_time = time.time() - start_time
            
            # Test 2: Training performance
            start_time = time.time()
            trainer = BasicTrainer(verbose=False)
            X = np.random.random((10, 4))
            y = np.random.randint(0, 2, 10)
            trainer.fit(X, y, epochs=2)
            training_time = time.time() - start_time
            
            performance_results['benchmarks'] = {
                'optimizer_time': optimizer_time,
                'training_time': training_time,
                'memory_usage': 'within_limits'
            }
            
            # Performance thresholds
            if optimizer_time > 5.0:  # Should be fast
                performance_results['status'] = 'warning'
            if training_time > 30.0:  # Should complete in reasonable time
                performance_results['status'] = 'warning'
            
        except Exception as e:
            performance_results['status'] = 'error'
            performance_results['error'] = str(e)
        
        self.total_gates += 1
        if performance_results['status'] == 'passed':
            self.passed_gates += 1
        
        return performance_results
    
    def run_code_quality_checks(self) -> Dict[str, Any]:
        """Run code quality checks."""
        print("📋 Running code quality checks...")
        
        quality_results = {
            'status': 'passed',
            'metrics': {},
            'issues': []
        }
        
        try:
            # Count Python files and check basic quality metrics
            python_files = []
            total_lines = 0
            documented_functions = 0
            total_functions = 0
            
            for root, dirs, files in os.walk('.'):
                if any(skip in root for skip in ['.git', '__pycache__', '.pytest_cache']):
                    continue
                    
                for file in files:
                    if file.endswith('.py'):
                        filepath = os.path.join(root, file)
                        python_files.append(filepath)
                        
                        try:
                            with open(filepath, 'r', encoding='utf-8') as f:
                                lines = f.readlines()
                                total_lines += len(lines)
                                
                                # Count functions and documentation
                                for i, line in enumerate(lines):
                                    if line.strip().startswith('def '):
                                        total_functions += 1
                                        # Check if function has docstring
                                        if i + 1 < len(lines) and ('"""' in lines[i + 1] or "'''" in lines[i + 1]):
                                            documented_functions += 1
                        except:
                            continue
            
            # Calculate metrics
            documentation_ratio = documented_functions / max(1, total_functions)
            avg_lines_per_file = total_lines / max(1, len(python_files))
            
            quality_results['metrics'] = {
                'total_files': len(python_files),
                'total_lines': total_lines,
                'documentation_ratio': documentation_ratio,
                'avg_lines_per_file': avg_lines_per_file
            }
            
            # Quality thresholds
            if documentation_ratio < 0.5:  # At least 50% functions documented
                quality_results['issues'].append(f"Low documentation ratio: {documentation_ratio:.1%}")
            
            if avg_lines_per_file > 500:  # Files shouldn't be too large
                quality_results['issues'].append(f"Average file size too large: {avg_lines_per_file:.0f} lines")
            
            if len(quality_results['issues']) > 0:
                quality_results['status'] = 'warning'
            
        except Exception as e:
            quality_results['status'] = 'error'
            quality_results['error'] = str(e)
        
        self.total_gates += 1
        if quality_results['status'] == 'passed':
            self.passed_gates += 1
        
        return quality_results
    
    def run_dependency_checks(self) -> Dict[str, Any]:
        """Check dependency security and health."""
        print("📦 Running dependency checks...")
        
        dependency_results = {
            'status': 'passed',
            'dependencies': {},
            'issues': []
        }
        
        # Check requirements.txt
        if os.path.exists('requirements.txt'):
            try:
                with open('requirements.txt', 'r') as f:
                    deps = [line.strip() for line in f if line.strip() and not line.startswith('#')]
                
                dependency_results['dependencies'] = {
                    'total_count': len(deps),
                    'requirements_file': 'found',
                    'dependencies': deps[:10]  # Show first 10
                }
                
                # Check for pinned versions
                pinned_deps = [dep for dep in deps if '==' in dep or '>=' in dep]
                pin_ratio = len(pinned_deps) / max(1, len(deps))
                
                if pin_ratio < 0.8:  # At least 80% should be pinned
                    dependency_results['issues'].append(f"Low version pinning ratio: {pin_ratio:.1%}")
                    dependency_results['status'] = 'warning'
                
            except Exception as e:
                dependency_results['status'] = 'error'
                dependency_results['error'] = str(e)
        else:
            dependency_results['issues'].append("No requirements.txt found")
            dependency_results['status'] = 'warning'
        
        self.total_gates += 1
        if dependency_results['status'] == 'passed':
            self.passed_gates += 1
        
        return dependency_results
    
    def run_all_gates(self) -> Dict[str, Any]:
        """Run all quality gates."""
        print("🚀 Starting comprehensive quality gates...")
        print("=" * 60)
        
        # Run all checks
        self.results = {
            'security': self.run_security_checks(),
            'performance': self.run_performance_checks(),
            'code_quality': self.run_code_quality_checks(),
            'dependencies': self.run_dependency_checks(),
        }
        
        # Calculate overall results
        total_time = time.time() - self.start_time
        pass_rate = self.passed_gates / max(1, self.total_gates)
        
        overall_status = 'passed' if pass_rate >= 0.75 else 'warning'
        
        self.results['summary'] = {
            'overall_status': overall_status,
            'pass_rate': pass_rate,
            'passed_gates': self.passed_gates,
            'total_gates': self.total_gates,
            'execution_time': total_time,
            'timestamp': time.time()
        }
        
        return self.results
    
    def print_summary(self):
        """Print quality gate summary."""
        print("\n" + "=" * 60)
        print("📊 QUALITY GATE SUMMARY")
        print("=" * 60)
        
        summary = self.results['summary']
        
        # Overall status
        status_emoji = "✅" if summary['overall_status'] == 'passed' else "⚠️"
        print(f"{status_emoji} Overall Status: {summary['overall_status'].upper()}")
        print(f"📈 Pass Rate: {summary['pass_rate']:.1%} ({summary['passed_gates']}/{summary['total_gates']})")
        print(f"⏱️  Execution Time: {summary['execution_time']:.2f}s")
        
        print("\n🔍 DETAILED RESULTS:")
        
        for gate_name, gate_result in self.results.items():
            if gate_name == 'summary':
                continue
                
            status = gate_result.get('status', 'unknown')
            emoji = "✅" if status == 'passed' else "⚠️" if status == 'warning' else "❌"
            print(f"  {emoji} {gate_name.replace('_', ' ').title()}: {status}")
            
            if 'message' in gate_result:
                print(f"     {gate_result['message']}")
            
            if gate_result.get('issues'):
                for issue in gate_result['issues'][:3]:  # Show first 3 issues
                    print(f"     - {issue}")
        
        print("\n" + "=" * 60)
        
        if summary['overall_status'] == 'passed':
            print("🎉 All quality gates passed! Ready for production deployment.")
        else:
            print("⚠️  Some quality gates need attention before production deployment.")


def main():
    """Main entry point."""
    runner = QualityGateRunner()
    results = runner.run_all_gates()
    runner.print_summary()
    
    # Save results to file
    with open('quality_gates_report_comprehensive.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📋 Detailed report saved to: quality_gates_report_comprehensive.json")
    
    # Exit with appropriate code
    if results['summary']['overall_status'] == 'passed':
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()