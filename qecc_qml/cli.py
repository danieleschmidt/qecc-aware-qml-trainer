#!/usr/bin/env python3
"""
Command-line interface for QECC-aware QML library.
"""

import argparse
import sys
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

from .utils.logging_config import setup_logging, get_logger
from .utils.diagnostics import SystemDiagnostics, HealthChecker
from .config.settings import load_config, save_config, Settings
from .config.validation import ConfigValidator


def setup_cli_logging(verbose: bool = False, quiet: bool = False):
    """Setup logging for CLI usage."""
    if quiet:
        level = "ERROR"
    elif verbose:
        level = "DEBUG"
    else:
        level = "INFO"
    
    setup_logging(
        level=level,
        structured_logging=False,
        enable_performance_logging=False
    )


def cmd_diagnose(args):
    """Run system diagnostics."""
    logger = get_logger(__name__)
    
    print("🔍 QECC-Aware QML System Diagnostics")
    print("=" * 50)
    
    diagnostics = SystemDiagnostics()
    results = diagnostics.run_all_checks()
    
    if args.format == 'json':
        # Output as JSON
        json_results = [result.to_dict() for result in results]
        print(json.dumps(json_results, indent=2))
    else:
        # Human-readable output
        diagnostics.print_report(show_details=args.detailed)
    
    # Return appropriate exit code
    failed_checks = [r for r in results if r.status == 'fail']
    return 1 if failed_checks else 0


def cmd_config(args):
    """Manage configuration."""
    logger = get_logger(__name__)
    
    if args.action == 'show':
        # Show current configuration
        try:
            settings = load_config(args.config_file)
            
            if args.format == 'json':
                print(settings.to_json())
            elif args.format == 'yaml':
                print(settings.to_yaml())
            else:
                # Pretty print
                print("Current Configuration:")
                print("=" * 30)
                print(settings.to_yaml())
                
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return 1
    
    elif args.action == 'validate':
        # Validate configuration file
        try:
            validator = ConfigValidator()
            is_valid, report = validator.validate_config_file(args.config_file)
            
            print(report)
            return 0 if is_valid else 1
            
        except Exception as e:
            logger.error(f"Failed to validate configuration: {e}")
            return 1
    
    elif args.action == 'create':
        # Create default configuration
        try:
            settings = Settings()
            
            if args.config_file:
                save_config(settings, args.config_file)
                print(f"Created configuration file: {args.config_file}")
            else:
                print("Default configuration:")
                print(settings.to_yaml())
                
        except Exception as e:
            logger.error(f"Failed to create configuration: {e}")
            return 1
    
    return 0


def cmd_benchmark(args):
    """Run performance benchmarks."""
    logger = get_logger(__name__)
    
    print("🚀 QECC-Aware QML Benchmark")
    print("=" * 40)
    
    try:
        # Import here to avoid circular imports
        from .core.quantum_nn import QECCAwareQNN
        from .core.noise_models import NoiseModel
        from .codes.surface_code import SurfaceCode
        from .training.qecc_trainer import QECCTrainer
        from .evaluation.benchmarks import NoiseBenchmark
        import numpy as np
        
        # Create test model
        print("Setting up test model...")
        qnn = QECCAwareQNN(
            num_qubits=args.qubits,
            num_layers=args.layers,
            entanglement="circular"
        )
        
        if args.error_correction:
            surface_code = SurfaceCode(distance=3)
            qnn.add_error_correction(surface_code)
            print(f"Added error correction: {surface_code}")
        
        # Create noise model
        noise_model = NoiseModel(gate_error_rate=args.noise_level)
        print(f"Using noise model: {noise_model}")
        
        # Generate test data
        print("Generating test data...")
        X_test = np.random.uniform(0, np.pi, (args.samples, args.qubits))
        y_test = np.random.randint(0, 2, args.samples)
        
        # Run benchmark
        print("Running benchmark...")
        benchmark = NoiseBenchmark(
            model=qnn,
            noise_levels=np.array([args.noise_level]),
            shots=args.shots
        )
        
        # Random parameters for testing
        params = np.random.uniform(-np.pi, np.pi, qnn.get_num_parameters())
        
        results = benchmark.run(
            X_test, y_test,
            model_parameters=params,
            verbose=False
        )
        
        # Display results
        print(f"\nBenchmark Results:")
        print(f"Accuracy: {results['accuracy'][0]:.3f}")
        print(f"Fidelity: {results['fidelity'][0]:.3f}")
        print(f"Runtime: {results['runtime'][0]:.2f}s")
        print(f"Circuit Depth: {results['circuit_depth'][0]}")
        print(f"Physical Qubits: {results['physical_qubits'][0]}")
        
        if args.output:
            # Save results
            benchmark.export_results(args.output, format='json')
            print(f"Results saved to: {args.output}")
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        return 1
    
    return 0


def cmd_train(args):
    """Train a quantum neural network."""
    logger = get_logger(__name__)
    
    print("🎯 QECC-Aware QML Training")
    print("=" * 35)
    
    try:
        # Load configuration
        settings = load_config(args.config_file)
        
        # Import training modules
        from .core.quantum_nn import QECCAwareQNN
        from .core.noise_models import NoiseModel
        from .codes.surface_code import SurfaceCode
        from .training.qecc_trainer import QECCTrainer
        import numpy as np
        
        # Create model from config
        print("Creating quantum neural network...")
        qnn = QECCAwareQNN(
            num_qubits=args.qubits or settings.quantum.max_qubits,
            num_layers=args.layers or 3,
            entanglement="circular"
        )
        
        # Add error correction if enabled
        if args.error_correction or settings.error_correction.default_scheme != "none":
            if settings.error_correction.default_scheme == "surface_code":
                surface_code = SurfaceCode(
                    distance=settings.error_correction.default_distance
                )
                qnn.add_error_correction(surface_code)
                print(f"Added error correction: {surface_code}")
        
        # Create noise model
        noise_model = NoiseModel(
            gate_error_rate=settings.noise.default_gate_error_rate,
            readout_error_rate=settings.noise.default_readout_error_rate,
            T1=settings.noise.default_T1,
            T2=settings.noise.default_T2
        )
        
        # Load or generate training data
        if args.data_file:
            print(f"Loading training data from: {args.data_file}")
            data = np.load(args.data_file)
            X_train = data['X_train']
            y_train = data['y_train']
            X_test = data.get('X_test', X_train[:10])
            y_test = data.get('y_test', y_train[:10])
        else:
            print("Generating synthetic training data...")
            n_samples = args.samples or 100
            n_features = qnn.num_qubits
            
            X_train = np.random.uniform(0, np.pi, (n_samples, n_features))
            y_train = np.random.randint(0, 2, n_samples)
            X_test = np.random.uniform(0, np.pi, (20, n_features))
            y_test = np.random.randint(0, 2, 20)
        
        print(f"Training data: {X_train.shape[0]} samples, {X_train.shape[1]} features")
        
        # Create trainer
        trainer = QECCTrainer(
            qnn=qnn,
            noise_model=noise_model,
            learning_rate=args.learning_rate or settings.training.default_learning_rate,
            shots=args.shots or settings.quantum.default_shots
        )
        
        # Train model
        print("Starting training...")
        history = trainer.fit(
            X_train, y_train,
            X_val=X_test, y_val=y_test,
            epochs=args.epochs or settings.training.default_epochs,
            batch_size=args.batch_size or settings.training.default_batch_size,
            verbose=True
        )
        
        # Evaluate final model
        print("\nEvaluating trained model...")
        results = trainer.evaluate(X_test, y_test)
        
        print(f"Final Results:")
        print(f"Accuracy: {results['accuracy']:.3f}")
        print(f"Loss: {results['loss']:.4f}")
        print(f"Fidelity: {results['fidelity']:.3f}")
        
        if 'logical_error_rate' in results:
            print(f"Logical Error Rate: {results['logical_error_rate']:.2e}")
        
        # Save model if requested
        if args.output:
            trainer.save_model(args.output)
            print(f"Model saved to: {args.output}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 1
    
    return 0


def cmd_health(args):
    """Check system health."""
    logger = get_logger(__name__)
    
    health_checker = HealthChecker()
    health_data = health_checker.quick_health_check()
    
    if args.format == 'json':
        print(json.dumps(health_data, indent=2))
    else:
        print("System Health Check")
        print("=" * 30)
        print(f"Status: {health_data['status'].upper()}")
        print(f"Memory Usage: {health_data['memory_usage_percent']:.1f}%")
        print(f"CPU Usage: {health_data['cpu_usage_percent']:.1f}%")
        print(f"Disk Usage: {health_data['disk_usage_percent']:.1f}%")
        print(f"Process Memory: {health_data['process_memory_mb']:.1f} MB")
        
        if 'issues' in health_data:
            print(f"\nIssues:")
            for issue in health_data['issues']:
                print(f"  • {issue}")
    
    return 0 if health_data['status'] == 'healthy' else 1


def create_parser():
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        prog='qecc-qml',
        description='QECC-Aware Quantum Machine Learning CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  qecc-qml diagnose                    # Run system diagnostics
  qecc-qml config show                 # Show current configuration
  qecc-qml config validate config.yaml # Validate configuration file
  qecc-qml benchmark --qubits 4        # Run performance benchmark
  qecc-qml train --epochs 50           # Train a quantum model
  qecc-qml health                      # Check system health
        """
    )
    
    # Global options
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--quiet', '-q', action='store_true', help='Quiet output')
    parser.add_argument('--config-file', '-c', help='Configuration file path')
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Diagnose command
    diagnose_parser = subparsers.add_parser('diagnose', help='Run system diagnostics')
    diagnose_parser.add_argument('--format', choices=['text', 'json'], default='text', help='Output format')
    diagnose_parser.add_argument('--detailed', action='store_true', help='Show detailed information')
    
    # Config command
    config_parser = subparsers.add_parser('config', help='Manage configuration')
    config_parser.add_argument('action', choices=['show', 'validate', 'create'], help='Configuration action')
    config_parser.add_argument('--format', choices=['text', 'json', 'yaml'], default='text', help='Output format')
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser('benchmark', help='Run performance benchmarks')
    benchmark_parser.add_argument('--qubits', type=int, default=4, help='Number of qubits')
    benchmark_parser.add_argument('--layers', type=int, default=2, help='Number of layers')
    benchmark_parser.add_argument('--samples', type=int, default=50, help='Number of test samples')
    benchmark_parser.add_argument('--shots', type=int, default=1024, help='Number of shots')
    benchmark_parser.add_argument('--noise-level', type=float, default=0.01, help='Noise level')
    benchmark_parser.add_argument('--error-correction', action='store_true', help='Enable error correction')
    benchmark_parser.add_argument('--output', '-o', help='Output file for results')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train quantum neural network')
    train_parser.add_argument('--qubits', type=int, help='Number of qubits')
    train_parser.add_argument('--layers', type=int, help='Number of layers')
    train_parser.add_argument('--epochs', type=int, help='Number of training epochs')
    train_parser.add_argument('--batch-size', type=int, help='Batch size')
    train_parser.add_argument('--learning-rate', type=float, help='Learning rate')
    train_parser.add_argument('--shots', type=int, help='Number of shots')
    train_parser.add_argument('--samples', type=int, help='Number of training samples')
    train_parser.add_argument('--data-file', help='Training data file (.npz)')
    train_parser.add_argument('--error-correction', action='store_true', help='Enable error correction')
    train_parser.add_argument('--output', '-o', help='Output file for trained model')
    
    # Health command
    health_parser = subparsers.add_parser('health', help='Check system health')
    health_parser.add_argument('--format', choices=['text', 'json'], default='text', help='Output format')
    
    return parser


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Setup logging
    setup_cli_logging(verbose=args.verbose, quiet=args.quiet)
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Route to appropriate command handler
    command_handlers = {
        'diagnose': cmd_diagnose,
        'config': cmd_config,
        'benchmark': cmd_benchmark,
        'train': cmd_train,
        'health': cmd_health,
    }
    
    handler = command_handlers.get(args.command)
    if handler:
        try:
            return handler(args)
        except KeyboardInterrupt:
            print("\nOperation cancelled by user")
            return 130
        except Exception as e:
            logger = get_logger(__name__)
            logger.error(f"Command failed: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            return 1
    else:
        print(f"Unknown command: {args.command}")
        return 1


if __name__ == '__main__':
    sys.exit(main())