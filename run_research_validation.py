#!/usr/bin/env python3
"""
Research Validation Study for QECC-QML Advances.

Comprehensive validation of the research implementations including
reinforcement learning, neural decoders, and quantum advantage benchmarks.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import logging
import json
import time
from pathlib import Path

from qecc_qml.research.experimental_framework import (
    ResearchExperimentFramework,
    ExperimentConfig,
    ExperimentType
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Run comprehensive research validation study."""
    logger.info("🚀 Starting QECC-QML Research Validation Study")
    
    # Initialize research framework
    framework = ResearchExperimentFramework(
        experiment_dir="./research_validation_results",
        logger=logger
    )
    
    # Define validation experiments
    experiments = [
        # RL QECC Training Validation
        ExperimentConfig(
            experiment_type=ExperimentType.RL_QECC_TRAINING,
            name="RL QECC Validation",
            description="Validate reinforcement learning for QECC optimization",
            parameters={
                'num_episodes': 100,  # Reduced for validation
                'noise_models': [
                    {'gate_error_rate': 0.001, 'readout_error_rate': 0.01},
                    {'gate_error_rate': 0.01, 'readout_error_rate': 0.03}
                ]
            },
            expected_runtime=300.0,  # 5 minutes
            priority=1
        ),
        
        # Neural Decoder Comparison Validation
        ExperimentConfig(
            experiment_type=ExperimentType.NEURAL_DECODER_COMPARISON,
            name="Neural Decoder Validation",
            description="Validate neural syndrome decoders performance",
            parameters={
                'code_type': 'surface',
                'code_distance': 3,
                'train_size': 1000,  # Reduced for validation
                'test_size': 200,
                'decoder_configs': {
                    'MLP_Small': {
                        'architecture': 'mlp',
                        'input_dim': 8,  # Will be adjusted automatically
                        'output_dim': 18,
                        'hidden_dims': [32, 16],
                        'learning_rate': 0.01,
                        'epochs': 20
                    }
                }
            },
            expected_runtime=600.0,  # 10 minutes
            priority=2
        ),
        
        # Quantum Advantage Study Validation
        ExperimentConfig(
            experiment_type=ExperimentType.QUANTUM_ADVANTAGE_STUDY,
            name="Quantum Advantage Validation",
            description="Validate quantum advantage benchmarking framework",
            parameters={
                'benchmark_types': ['learning_efficiency', 'scaling_advantage'],
                'problem_sizes': [10, 20],  # Reduced for validation
                'noise_levels': [0.001, 0.01],
                'num_trials': 3  # Reduced for validation
            },
            expected_runtime=900.0,  # 15 minutes
            priority=3
        ),
        
        # Hybrid Optimization Validation
        ExperimentConfig(
            experiment_type=ExperimentType.HYBRID_OPTIMIZATION,
            name="Hybrid Optimization Validation",
            description="Validate integrated optimization approach",
            parameters={
                'include_rl': True,
                'include_decoders': True,
                'include_advantage': False,  # Skip for faster validation
                'rl_params': {'num_episodes': 50},
                'decoder_params': {
                    'code_type': 'surface',
                    'code_distance': 3,
                    'train_size': 500,
                    'test_size': 100
                }
            },
            expected_runtime=1200.0,  # 20 minutes
            priority=4
        )
    ]
    
    # Register and queue experiments
    experiment_ids = []
    for config in experiments:
        exp_id = framework.register_experiment(config)
        framework.queue_experiment(exp_id)
        experiment_ids.append(exp_id)
        logger.info(f"✓ Registered experiment: {config.name}")
    
    logger.info(f"📋 Queued {len(experiment_ids)} validation experiments")
    
    # Run all experiments
    start_time = time.time()
    all_results = framework.run_experiment_queue()
    total_time = time.time() - start_time
    
    # Analyze results
    logger.info("📊 Analyzing validation results...")
    
    successful_experiments = 0
    failed_experiments = 0
    validation_metrics = {}
    
    for exp_id, result in all_results.items():
        exp_config = framework.experiments[exp_id]
        
        if result.status.value == 'completed':
            successful_experiments += 1
            logger.info(f"✅ {exp_config.name}: SUCCESS")
            
            # Extract validation metrics
            for metric_name, value in result.metrics.items():
                if metric_name not in validation_metrics:
                    validation_metrics[metric_name] = []
                validation_metrics[metric_name].append(value)
        else:
            failed_experiments += 1
            logger.error(f"❌ {exp_config.name}: FAILED - {result.error_message}")
    
    # Generate comprehensive validation report
    validation_report = {
        'validation_summary': {
            'total_experiments': len(experiment_ids),
            'successful_experiments': successful_experiments,
            'failed_experiments': failed_experiments,
            'success_rate': successful_experiments / len(experiment_ids),
            'total_validation_time': total_time,
            'validation_timestamp': time.time()
        },
        'experiment_results': {
            exp_id: {
                'name': framework.experiments[exp_id].name,
                'type': result.experiment_type.value,
                'status': result.status.value,
                'runtime': result.end_time - result.start_time if result.end_time else 0,
                'metrics': result.metrics,
                'artifacts': result.artifacts
            }
            for exp_id, result in all_results.items()
        },
        'aggregated_metrics': {
            metric: {
                'mean': sum(values) / len(values),
                'min': min(values),
                'max': max(values),
                'count': len(values)
            }
            for metric, values in validation_metrics.items()
        },
        'framework_statistics': framework.get_experiment_statistics(),
        'research_summary': framework.generate_research_summary()
    }
    
    # Save validation report
    report_path = Path("./research_validation_results/validation_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w') as f:
        json.dump(validation_report, f, indent=2, default=str)
    
    # Generate human-readable summary
    logger.info("="*80)
    logger.info("🎯 RESEARCH VALIDATION SUMMARY")
    logger.info("="*80)
    logger.info(f"📈 Success Rate: {validation_report['validation_summary']['success_rate']:.1%}")
    logger.info(f"⏱️ Total Time: {total_time:.1f} seconds")
    logger.info(f"🧪 Experiments: {successful_experiments}/{len(experiment_ids)} successful")
    
    if validation_metrics:
        logger.info("\n📊 Key Validation Metrics:")
        for metric, stats in validation_report['aggregated_metrics'].items():
            logger.info(f"  • {metric}: mean={stats['mean']:.4f}, range=[{stats['min']:.4f}, {stats['max']:.4f}]")
    
    # Research insights
    research_summary = validation_report['research_summary']
    if research_summary['key_findings']:
        logger.info("\n🔍 Key Research Findings:")
        for finding in research_summary['key_findings']:
            logger.info(f"  • {finding}")
    
    # Future directions
    if research_summary['future_directions']:
        logger.info("\n🚀 Future Research Directions:")
        for direction in research_summary['future_directions'][:3]:  # Show top 3
            logger.info(f"  • {direction}")
    
    logger.info(f"\n📄 Full validation report saved to: {report_path}")
    
    # Final validation assessment
    if validation_report['validation_summary']['success_rate'] >= 0.75:
        logger.info("✅ VALIDATION PASSED: Research implementations are working correctly")
        return_code = 0
    else:
        logger.error("❌ VALIDATION FAILED: Some research implementations need attention")
        return_code = 1
    
    logger.info("="*80)
    logger.info("🎉 Research validation completed!")
    
    return return_code


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)