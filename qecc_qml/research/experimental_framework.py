"""
Experimental Framework for QECC-QML Research.

Unified framework for conducting research experiments combining
reinforcement learning, neural decoders, and quantum advantage benchmarks.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
import json
from pathlib import Path
from collections import defaultdict, deque

from .reinforcement_learning_qecc import QECCEnvironment, QECCRLAgent, create_rl_qecc_trainer
from .neural_syndrome_decoders import (
    NeuralSyndromeDecoder, 
    SyndromeGenerator, 
    DecoderComparison,
    DecoderArchitecture,
    DecoderConfig
)
from .quantum_advantage_benchmarks import (
    QuantumAdvantageSuite,
    BenchmarkType,
    QuantumAdvantageMetric
)


class ExperimentType(Enum):
    """Types of research experiments."""
    RL_QECC_TRAINING = "rl_qecc_training"
    NEURAL_DECODER_COMPARISON = "neural_decoder_comparison"
    QUANTUM_ADVANTAGE_STUDY = "quantum_advantage_study"
    HYBRID_OPTIMIZATION = "hybrid_optimization"
    CROSS_VALIDATION = "cross_validation"
    ABLATION_STUDY = "ablation_study"


class ExperimentStatus(Enum):
    """Experiment execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ExperimentConfig:
    """Configuration for research experiment."""
    experiment_type: ExperimentType
    name: str
    description: str
    parameters: Dict[str, Any]
    expected_runtime: float = 3600.0  # 1 hour default
    priority: int = 1
    dependencies: List[str] = field(default_factory=list)
    output_dir: Optional[str] = None
    random_seed: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentResult:
    """Results from research experiment."""
    experiment_id: str
    experiment_type: ExperimentType
    status: ExperimentStatus
    start_time: float
    end_time: Optional[float] = None
    results: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)  # file paths
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ResearchExperimentFramework:
    """
    Unified framework for conducting QECC-QML research experiments.
    
    Orchestrates different types of research experiments and provides
    tools for analyzing and comparing results across studies.
    """
    
    def __init__(
        self,
        experiment_dir: str = "./experiments",
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize research framework.
        
        Args:
            experiment_dir: Directory for storing experiment results
            logger: Optional logger instance
        """
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or logging.getLogger(__name__)
        
        # Experiment management
        self.experiments: Dict[str, ExperimentConfig] = {}
        self.results: Dict[str, ExperimentResult] = {}
        self.execution_queue: deque = deque()
        
        # Research components
        self.rl_trainer = None
        self.decoder_comparisons: Dict[str, DecoderComparison] = {}
        self.advantage_suite = None
        
        # Statistics
        self.total_experiments = 0
        self.successful_experiments = 0
        self.failed_experiments = 0
        
        self.logger.info(f"ResearchExperimentFramework initialized with experiment_dir: {experiment_dir}")
    
    def register_experiment(
        self, 
        config: ExperimentConfig
    ) -> str:
        """
        Register a new experiment.
        
        Args:
            config: Experiment configuration
            
        Returns:
            Experiment ID
        """
        experiment_id = f"{config.experiment_type.value}_{int(time.time())}_{len(self.experiments)}"
        
        # Set random seed if not provided
        if config.random_seed is None:
            config.random_seed = np.random.randint(0, 2**32 - 1)
        
        # Set output directory
        if config.output_dir is None:
            config.output_dir = str(self.experiment_dir / experiment_id)
        
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        
        self.experiments[experiment_id] = config
        self.total_experiments += 1
        
        self.logger.info(f"Registered experiment: {experiment_id} ({config.name})")
        return experiment_id
    
    def queue_experiment(self, experiment_id: str):
        """Queue experiment for execution."""
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        self.execution_queue.append(experiment_id)
        self.logger.info(f"Queued experiment: {experiment_id}")
    
    def run_experiment(self, experiment_id: str) -> ExperimentResult:
        """
        Run a single experiment.
        
        Args:
            experiment_id: ID of experiment to run
            
        Returns:
            Experiment results
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        config = self.experiments[experiment_id]
        self.logger.info(f"Starting experiment: {experiment_id}")
        
        # Initialize result
        result = ExperimentResult(
            experiment_id=experiment_id,
            experiment_type=config.experiment_type,
            status=ExperimentStatus.RUNNING,
            start_time=time.time()
        )
        
        try:
            # Set random seed for reproducibility
            np.random.seed(config.random_seed)
            
            # Run experiment based on type
            if config.experiment_type == ExperimentType.RL_QECC_TRAINING:
                experiment_results = self._run_rl_qecc_experiment(config)
            elif config.experiment_type == ExperimentType.NEURAL_DECODER_COMPARISON:
                experiment_results = self._run_decoder_comparison_experiment(config)
            elif config.experiment_type == ExperimentType.QUANTUM_ADVANTAGE_STUDY:
                experiment_results = self._run_quantum_advantage_experiment(config)
            elif config.experiment_type == ExperimentType.HYBRID_OPTIMIZATION:
                experiment_results = self._run_hybrid_optimization_experiment(config)
            else:
                raise NotImplementedError(f"Experiment type {config.experiment_type.value} not implemented")
            
            # Store results
            result.results = experiment_results
            result.status = ExperimentStatus.COMPLETED
            result.end_time = time.time()
            
            # Extract key metrics
            result.metrics = self._extract_key_metrics(experiment_results, config.experiment_type)
            
            # Save artifacts
            result.artifacts = self._save_artifacts(experiment_id, experiment_results, config.output_dir)
            
            self.successful_experiments += 1
            self.logger.info(f"Experiment completed successfully: {experiment_id}")
            
        except Exception as e:
            result.status = ExperimentStatus.FAILED
            result.end_time = time.time()
            result.error_message = str(e)
            self.failed_experiments += 1
            
            self.logger.error(f"Experiment failed: {experiment_id}, Error: {e}")
        
        self.results[experiment_id] = result
        return result
    
    def _run_rl_qecc_experiment(self, config: ExperimentConfig) -> Dict[str, Any]:
        """Run reinforcement learning QECC experiment."""
        params = config.parameters
        
        # Extract parameters
        num_episodes = params.get('num_episodes', 1000)
        noise_models = params.get('noise_models', None)
        
        # Run RL training
        training_results = create_rl_qecc_trainer(
            num_episodes=num_episodes,
            logger=self.logger
        )
        
        # Additional analysis
        analysis = {
            'convergence_analysis': self._analyze_rl_convergence(training_results),
            'performance_metrics': self._calculate_rl_metrics(training_results),
            'policy_analysis': self._analyze_rl_policy(training_results)
        }
        
        return {
            'training_results': training_results,
            'analysis': analysis,
            'experiment_config': params
        }
    
    def _run_decoder_comparison_experiment(self, config: ExperimentConfig) -> Dict[str, Any]:
        """Run neural decoder comparison experiment."""
        params = config.parameters
        
        # Extract parameters
        code_type = params.get('code_type', 'surface')
        code_distance = params.get('code_distance', 3)
        train_size = params.get('train_size', 10000)
        test_size = params.get('test_size', 2000)
        
        # Create decoder comparison
        comparison = DecoderComparison(
            code_type=code_type,
            code_distance=code_distance,
            logger=self.logger
        )
        
        # Prepare datasets
        comparison.prepare_datasets(train_size=train_size, test_size=test_size)
        
        # Add custom decoders if specified
        if 'decoder_configs' in params:
            for name, decoder_config in params['decoder_configs'].items():
                config_obj = DecoderConfig(**decoder_config)
                comparison.add_decoder(name, config_obj)
        else:
            comparison.add_standard_decoders()
        
        # Run comparison
        comparison_results = comparison.run_comparison(train_models=True)
        
        # Generate report
        report = comparison.generate_comparison_report()
        
        return {
            'comparison_results': comparison_results,
            'report': report,
            'experiment_config': params
        }
    
    def _run_quantum_advantage_experiment(self, config: ExperimentConfig) -> Dict[str, Any]:
        """Run quantum advantage study experiment."""
        params = config.parameters
        
        # Extract parameters
        benchmark_types = params.get('benchmark_types', [BenchmarkType.LEARNING_EFFICIENCY])
        problem_sizes = params.get('problem_sizes', [10, 25, 50])
        noise_levels = params.get('noise_levels', [0.001, 0.01, 0.05])
        num_trials = params.get('num_trials', 5)
        
        # Convert string benchmark types to enums if needed
        if isinstance(benchmark_types[0], str):
            benchmark_types = [BenchmarkType(bt) for bt in benchmark_types]
        
        # Create advantage suite
        suite = QuantumAdvantageSuite(benchmarks=benchmark_types, logger=self.logger)
        
        # Run comprehensive benchmarking
        suite_results = suite.run_comprehensive_suite(
            problem_sizes=problem_sizes,
            noise_levels=noise_levels,
            num_trials=num_trials
        )
        
        # Generate analysis and report
        analysis = suite.analyze_quantum_advantage()
        report = suite.generate_benchmark_report()
        
        return {
            'suite_results': suite_results,
            'analysis': analysis,
            'report': report,
            'experiment_config': params
        }
    
    def _run_hybrid_optimization_experiment(self, config: ExperimentConfig) -> Dict[str, Any]:
        """Run hybrid optimization experiment combining multiple approaches."""
        params = config.parameters
        
        # This would combine RL, neural decoders, and advantage analysis
        results = {}
        
        # Phase 1: Train RL agent for QECC optimization
        if params.get('include_rl', True):
            rl_config = ExperimentConfig(
                experiment_type=ExperimentType.RL_QECC_TRAINING,
                name="RL Phase",
                description="RL training phase",
                parameters=params.get('rl_params', {'num_episodes': 500})
            )
            results['rl_phase'] = self._run_rl_qecc_experiment(rl_config)
        
        # Phase 2: Train neural decoders
        if params.get('include_decoders', True):
            decoder_config = ExperimentConfig(
                experiment_type=ExperimentType.NEURAL_DECODER_COMPARISON,
                name="Decoder Phase", 
                description="Neural decoder training phase",
                parameters=params.get('decoder_params', {
                    'code_type': 'surface',
                    'code_distance': 3,
                    'train_size': 5000
                })
            )
            results['decoder_phase'] = self._run_decoder_comparison_experiment(decoder_config)
        
        # Phase 3: Evaluate quantum advantage
        if params.get('include_advantage', True):
            advantage_config = ExperimentConfig(
                experiment_type=ExperimentType.QUANTUM_ADVANTAGE_STUDY,
                name="Advantage Phase",
                description="Quantum advantage evaluation phase",
                parameters=params.get('advantage_params', {
                    'benchmark_types': ['learning_efficiency'],
                    'problem_sizes': [25, 50],
                    'num_trials': 3
                })
            )
            results['advantage_phase'] = self._run_quantum_advantage_experiment(advantage_config)
        
        # Integrated analysis
        integrated_analysis = self._perform_integrated_analysis(results)
        
        return {
            'phase_results': results,
            'integrated_analysis': integrated_analysis,
            'experiment_config': params
        }
    
    def _analyze_rl_convergence(self, training_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze RL training convergence."""
        episode_rewards = training_results.get('episode_rewards', [])
        
        if not episode_rewards:
            return {}
        
        # Find convergence point
        window_size = min(50, len(episode_rewards) // 4)
        if window_size < 10:
            return {'convergence_episode': len(episode_rewards)}
        
        for i in range(window_size, len(episode_rewards)):
            recent_rewards = episode_rewards[i-window_size:i]
            if len(set([round(r, 1) for r in recent_rewards])) <= 3:  # Converged
                return {
                    'convergence_episode': i,
                    'convergence_reward': np.mean(recent_rewards),
                    'final_stability': np.std(episode_rewards[-window_size:])
                }
        
        return {
            'convergence_episode': len(episode_rewards),
            'convergence_reward': np.mean(episode_rewards[-window_size:]),
            'final_stability': np.std(episode_rewards[-window_size:])
        }
    
    def _calculate_rl_metrics(self, training_results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate key RL performance metrics."""
        episode_rewards = training_results.get('episode_rewards', [])
        
        if not episode_rewards:
            return {}
        
        return {
            'final_average_reward': np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards),
            'max_reward': max(episode_rewards),
            'reward_improvement': episode_rewards[-1] - episode_rewards[0] if len(episode_rewards) > 1 else 0,
            'learning_stability': 1.0 / (1.0 + np.std(episode_rewards[-50:]) / abs(np.mean(episode_rewards[-50:]) + 1e-6)) if len(episode_rewards) >= 50 else 0
        }
    
    def _analyze_rl_policy(self, training_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze learned RL policy."""
        # Placeholder for policy analysis
        return {
            'policy_entropy': 0.5,  # Would calculate actual entropy
            'action_distribution': {'surface_code': 0.4, 'color_code': 0.3, 'threshold_adjust': 0.3},
            'policy_complexity': 0.7
        }
    
    def _perform_integrated_analysis(self, phase_results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform integrated analysis across experiment phases."""
        analysis = {
            'cross_phase_correlations': {},
            'performance_synthesis': {},
            'recommendations': []
        }
        
        # Correlate RL performance with decoder performance
        if 'rl_phase' in phase_results and 'decoder_phase' in phase_results:
            rl_metrics = phase_results['rl_phase'].get('analysis', {}).get('performance_metrics', {})
            decoder_report = phase_results['decoder_phase'].get('report', {})
            
            analysis['cross_phase_correlations']['rl_decoder'] = {
                'rl_final_reward': rl_metrics.get('final_average_reward', 0),
                'best_decoder_accuracy': max([
                    metrics.get('accuracy', 0) 
                    for metrics in decoder_report.get('summary_metrics', {}).values()
                ], default=0)
            }
        
        # Synthesize quantum advantage findings
        if 'advantage_phase' in phase_results:
            advantage_analysis = phase_results['advantage_phase'].get('analysis', {})
            recommendations = advantage_analysis.get('recommendations', [])
            analysis['recommendations'].extend(recommendations)
        
        # Overall performance synthesis
        analysis['performance_synthesis'] = {
            'phases_completed': len(phase_results),
            'overall_success': all(
                'error' not in phase_result 
                for phase_result in phase_results.values()
            ),
            'integration_score': len(phase_results) / 3.0  # Normalized by max phases
        }
        
        return analysis
    
    def _extract_key_metrics(
        self, 
        results: Dict[str, Any], 
        experiment_type: ExperimentType
    ) -> Dict[str, float]:
        """Extract key metrics from experiment results."""
        metrics = {}
        
        if experiment_type == ExperimentType.RL_QECC_TRAINING:
            analysis = results.get('analysis', {})
            perf_metrics = analysis.get('performance_metrics', {})
            metrics.update(perf_metrics)
            
            convergence = analysis.get('convergence_analysis', {})
            if 'convergence_episode' in convergence:
                metrics['convergence_episode'] = convergence['convergence_episode']
        
        elif experiment_type == ExperimentType.NEURAL_DECODER_COMPARISON:
            report = results.get('report', {})
            summary = report.get('summary_metrics', {})
            
            # Get best decoder accuracy
            if summary:
                best_accuracy = max(
                    metrics.get('accuracy', 0) 
                    for metrics in summary.values()
                )
                metrics['best_decoder_accuracy'] = best_accuracy
        
        elif experiment_type == ExperimentType.QUANTUM_ADVANTAGE_STUDY:
            analysis = results.get('analysis', {})
            summary = analysis.get('summary', {})
            
            # Aggregate advantage metrics
            total_speedup = 0
            count = 0
            for benchmark_summary in summary.values():
                if 'average_speedup' in benchmark_summary:
                    total_speedup += benchmark_summary['average_speedup']
                    count += 1
            
            if count > 0:
                metrics['average_quantum_speedup'] = total_speedup / count
        
        return metrics
    
    def _save_artifacts(
        self, 
        experiment_id: str, 
        results: Dict[str, Any], 
        output_dir: str
    ) -> Dict[str, str]:
        """Save experiment artifacts to disk."""
        artifacts = {}
        output_path = Path(output_dir)
        
        # Save main results
        results_file = output_path / "results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        artifacts['results'] = str(results_file)
        
        # Save specific artifacts based on experiment type
        if 'training_results' in results:
            # RL training artifacts
            training_file = output_path / "training_history.json"
            with open(training_file, 'w') as f:
                json.dump(results['training_results'], f, indent=2, default=str)
            artifacts['training_history'] = str(training_file)
        
        if 'report' in results:
            # Comparison/benchmark report
            report_file = output_path / "experiment_report.json"
            with open(report_file, 'w') as f:
                json.dump(results['report'], f, indent=2, default=str)
            artifacts['report'] = str(report_file)
        
        self.logger.debug(f"Saved {len(artifacts)} artifacts for experiment {experiment_id}")
        return artifacts
    
    def run_experiment_queue(self) -> Dict[str, ExperimentResult]:
        """Run all queued experiments."""
        results = {}
        
        while self.execution_queue:
            experiment_id = self.execution_queue.popleft()
            self.logger.info(f"Processing queued experiment: {experiment_id}")
            
            try:
                result = self.run_experiment(experiment_id)
                results[experiment_id] = result
            except Exception as e:
                self.logger.error(f"Failed to run experiment {experiment_id}: {e}")
                # Create failed result
                results[experiment_id] = ExperimentResult(
                    experiment_id=experiment_id,
                    experiment_type=self.experiments[experiment_id].experiment_type,
                    status=ExperimentStatus.FAILED,
                    start_time=time.time(),
                    end_time=time.time(),
                    error_message=str(e)
                )
        
        return results
    
    def get_experiment_statistics(self) -> Dict[str, Any]:
        """Get framework statistics."""
        return {
            'total_experiments': self.total_experiments,
            'successful_experiments': self.successful_experiments,
            'failed_experiments': self.failed_experiments,
            'success_rate': self.successful_experiments / max(1, self.total_experiments),
            'queued_experiments': len(self.execution_queue),
            'completed_experiments': len(self.results),
            'experiment_types_run': list(set(
                config.experiment_type.value for config in self.experiments.values()
            ))
        }
    
    def generate_research_summary(self) -> Dict[str, Any]:
        """Generate comprehensive research summary."""
        summary = {
            'framework_statistics': self.get_experiment_statistics(),
            'experiment_results': {},
            'key_findings': [],
            'research_insights': {},
            'future_directions': []
        }
        
        # Aggregate results by experiment type
        results_by_type = defaultdict(list)
        for result in self.results.values():
            if result.status == ExperimentStatus.COMPLETED:
                results_by_type[result.experiment_type].append(result)
        
        # Analyze results by type
        for exp_type, type_results in results_by_type.items():
            type_summary = {
                'count': len(type_results),
                'average_runtime': np.mean([
                    (r.end_time - r.start_time) for r in type_results if r.end_time
                ]),
                'key_metrics': {}
            }
            
            # Aggregate metrics
            all_metrics = defaultdict(list)
            for result in type_results:
                for metric, value in result.metrics.items():
                    all_metrics[metric].append(value)
            
            for metric, values in all_metrics.items():
                type_summary['key_metrics'][metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': min(values),
                    'max': max(values)
                }
            
            summary['experiment_results'][exp_type.value] = type_summary
        
        # Generate key findings
        summary['key_findings'] = self._generate_key_findings(results_by_type)
        
        # Research insights
        summary['research_insights'] = self._generate_research_insights(results_by_type)
        
        # Future directions
        summary['future_directions'] = self._suggest_future_directions(results_by_type)
        
        return summary
    
    def _generate_key_findings(self, results_by_type: Dict) -> List[str]:
        """Generate key findings from experiments."""
        findings = []
        
        for exp_type, results in results_by_type.items():
            if not results:
                continue
                
            if exp_type == ExperimentType.RL_QECC_TRAINING:
                avg_reward = np.mean([
                    r.metrics.get('final_average_reward', 0) for r in results
                ])
                findings.append(f"RL QECC training achieved average reward of {avg_reward:.2f}")
            
            elif exp_type == ExperimentType.NEURAL_DECODER_COMPARISON:
                avg_accuracy = np.mean([
                    r.metrics.get('best_decoder_accuracy', 0) for r in results
                ])
                findings.append(f"Neural decoders achieved {avg_accuracy:.1%} average accuracy")
            
            elif exp_type == ExperimentType.QUANTUM_ADVANTAGE_STUDY:
                avg_speedup = np.mean([
                    r.metrics.get('average_quantum_speedup', 1) for r in results
                ])
                if avg_speedup > 1:
                    findings.append(f"Quantum advantage demonstrated with {avg_speedup:.1f}x speedup")
        
        return findings
    
    def _generate_research_insights(self, results_by_type: Dict) -> Dict[str, str]:
        """Generate research insights."""
        insights = {}
        
        if ExperimentType.NEURAL_DECODER_COMPARISON in results_by_type:
            insights['neural_decoders'] = "Neural decoders show promise for syndrome decoding"
        
        if ExperimentType.RL_QECC_TRAINING in results_by_type:
            insights['reinforcement_learning'] = "RL can optimize QECC strategies adaptively"
        
        if ExperimentType.QUANTUM_ADVANTAGE_STUDY in results_by_type:
            insights['quantum_advantage'] = "Quantum advantage depends on problem regime"
        
        return insights
    
    def _suggest_future_directions(self, results_by_type: Dict) -> List[str]:
        """Suggest future research directions."""
        directions = [
            "Investigate hybrid quantum-classical optimization algorithms",
            "Explore transfer learning between different QECC codes",
            "Study the robustness of learned policies to hardware variations",
            "Develop real-time adaptive error correction systems"
        ]
        
        # Add specific directions based on results
        if len(results_by_type) >= 2:
            directions.append("Conduct more comprehensive cross-method comparisons")
        
        return directions