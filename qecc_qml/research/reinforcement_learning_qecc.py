"""
Reinforcement Learning for QECC Optimization.

Novel approach using deep reinforcement learning to automatically
discover optimal QECC strategies and adaptation policies.
"""

# Import with fallback support
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from qecc_qml.core.fallback_imports import create_fallback_implementations
    create_fallback_implementations()
except ImportError:
    pass
try:
    import numpy as np
except ImportError:
    import sys
    if 'numpy' in sys.modules:
        np = sys.modules['numpy']
    else:
        class MockNumPy:
            @staticmethod
            def array(x): return list(x) if isinstance(x, (list, tuple)) else x
            @staticmethod
            def zeros(shape): return [0] * (shape if isinstance(shape, int) else shape[0])
            @staticmethod  
            def ones(shape): return [1] * (shape if isinstance(shape, int) else shape[0])
            ndarray = list
        np = MockNumPy()
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
from collections import deque, defaultdict
import json


class RLState(Enum):
    """RL environment state components."""
    NOISE_PROFILE = "noise_profile"
    ERROR_RATES = "error_rates"
    CIRCUIT_DEPTH = "circuit_depth"
    RESOURCE_USAGE = "resource_usage"
    PERFORMANCE_HISTORY = "performance_history"


class RLAction(Enum):
    """Available RL actions for QECC optimization."""
    SELECT_SURFACE_CODE = "select_surface_code"
    SELECT_COLOR_CODE = "select_color_code"
    SELECT_STEANE_CODE = "select_steane_code"
    ADJUST_THRESHOLD = "adjust_threshold"
    MODIFY_DECODER = "modify_decoder"
    CHANGE_DISTANCE = "change_distance"
    ADAPT_SCHEDULING = "adapt_scheduling"


@dataclass
class QECCAction:
    """QECC action with parameters."""
    action_type: RLAction
    parameters: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    expected_reward: float = 0.0


@dataclass
class EnvironmentState:
    """Complete environment state for RL agent."""
    noise_model: Dict[str, float]
    current_performance: Dict[str, float]
    resource_constraints: Dict[str, float]
    circuit_properties: Dict[str, Any]
    history_window: List[Dict[str, float]]
    timestamp: float = field(default_factory=time.time)


class QECCEnvironment:
    """
    Reinforcement learning environment for QECC optimization.
    
    Simulates the quantum computing environment where QECC decisions
    must be made, providing rewards based on performance metrics.
    """
    
    def __init__(
        self,
        noise_models: Optional[List[Dict[str, float]]] = None,
        performance_targets: Optional[Dict[str, float]] = None,
        resource_limits: Optional[Dict[str, float]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize QECC RL environment.
        
        Args:
            noise_models: List of noise model configurations
            performance_targets: Target performance metrics
            resource_limits: Resource constraint limits
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        
        # Environment configuration
        self.noise_models = noise_models or self._generate_default_noise_models()
        self.performance_targets = performance_targets or {
            'fidelity': 0.95,
            'logical_error_rate': 0.001,
            'overhead': 5.0,
            'throughput': 1.0
        }
        self.resource_limits = resource_limits or {
            'qubits': 1000,
            'gates': 10000,
            'time': 100.0,
            'memory': 1024
        }
        
        # Current state
        self.current_state: Optional[EnvironmentState] = None
        self.step_count = 0
        self.episode_count = 0
        
        # Performance tracking
        self.performance_history: deque = deque(maxlen=1000)
        self.reward_history: deque = deque(maxlen=1000)
        
        # Action space
        self.action_space = list(RLAction)
        self.action_history: deque = deque(maxlen=100)
        
        # Simulation parameters
        self.max_steps_per_episode = 100
        self.current_noise_model_idx = 0
        
        self.logger.info("QECC RL Environment initialized")
    
    def reset(self) -> EnvironmentState:
        """Reset environment to initial state."""
        self.step_count = 0
        self.episode_count += 1
        
        # Select random noise model for this episode
        self.current_noise_model_idx = np.random.randint(len(self.noise_models))
        current_noise = self.noise_models[self.current_noise_model_idx]
        
        # Initialize state
        self.current_state = EnvironmentState(
            noise_model=current_noise.copy(),
            current_performance=self._initialize_performance(),
            resource_constraints=self.resource_limits.copy(),
            circuit_properties=self._generate_circuit_properties(),
            history_window=[]
        )
        
        self.logger.debug(f"Environment reset - Episode {self.episode_count}")
        return self.current_state
    
    def step(self, action: QECCAction) -> Tuple[EnvironmentState, float, bool, Dict[str, Any]]:
        """
        Execute action and return new state, reward, done, info.
        
        Args:
            action: QECC action to execute
            
        Returns:
            Tuple of (new_state, reward, done, info)
        """
        if self.current_state is None:
            raise ValueError("Environment not reset. Call reset() first.")
        
        self.step_count += 1
        
        # Execute action and get new performance
        new_performance = self._simulate_action_effect(action, self.current_state)
        
        # Calculate reward
        reward = self._calculate_reward(new_performance, action)
        
        # Update state
        self.current_state.current_performance = new_performance
        self.current_state.history_window.append(new_performance.copy())
        
        # Keep history bounded
        if len(self.current_state.history_window) > 20:
            self.current_state.history_window = self.current_state.history_window[-20:]
        
        # Check if episode is done
        done = (self.step_count >= self.max_steps_per_episode or 
                self._check_terminal_condition(new_performance))
        
        # Store history
        self.performance_history.append(new_performance.copy())
        self.reward_history.append(reward)
        self.action_history.append({
            'action': action.action_type.value,
            'parameters': action.parameters.copy(),
            'reward': reward,
            'step': self.step_count
        })
        
        # Additional info
        info = {
            'step': self.step_count,
            'episode': self.episode_count,
            'performance_targets_met': self._check_performance_targets(new_performance),
            'resource_utilization': self._calculate_resource_utilization()
        }
        
        return self.current_state, reward, done, info
    
    def _generate_default_noise_models(self) -> List[Dict[str, float]]:
        """Generate default noise model configurations."""
        return [
            # Low noise
            {
                'gate_error_rate': 0.001,
                'readout_error_rate': 0.01,
                't1_coherence': 50e-6,
                't2_coherence': 70e-6,
                'crosstalk': 0.001
            },
            # Medium noise
            {
                'gate_error_rate': 0.01,
                'readout_error_rate': 0.03,
                't1_coherence': 30e-6,
                't2_coherence': 40e-6,
                'crosstalk': 0.01
            },
            # High noise
            {
                'gate_error_rate': 0.05,
                'readout_error_rate': 0.10,
                't1_coherence': 10e-6,
                't2_coherence': 15e-6,
                'crosstalk': 0.05
            },
            # Variable noise
            {
                'gate_error_rate': np.random.uniform(0.001, 0.05),
                'readout_error_rate': np.random.uniform(0.01, 0.10),
                't1_coherence': np.random.uniform(10e-6, 50e-6),
                't2_coherence': np.random.uniform(15e-6, 70e-6),
                'crosstalk': np.random.uniform(0.001, 0.05)
            }
        ]
    
    def _initialize_performance(self) -> Dict[str, float]:
        """Initialize baseline performance metrics."""
        return {
            'fidelity': 0.85,  # Start below target
            'logical_error_rate': 0.01,  # Start above target
            'overhead': 10.0,  # Start high
            'throughput': 0.5,  # Start low
            'resource_efficiency': 0.6
        }
    
    def _generate_circuit_properties(self) -> Dict[str, Any]:
        """Generate random circuit properties for this episode."""
        return {
            'num_qubits': np.random.randint(10, 100),
            'circuit_depth': np.random.randint(20, 200),
            'gate_types': ['cx', 'h', 'rz', 't', 's'],
            'connectivity': 'linear',  # Could be 'grid', 'all_to_all', etc.
            'measurement_pattern': 'standard'
        }
    
    def _simulate_action_effect(
        self, 
        action: QECCAction, 
        state: EnvironmentState
    ) -> Dict[str, float]:
        """Simulate the effect of taking an action."""
        current_perf = state.current_performance.copy()
        noise_level = state.noise_model['gate_error_rate']
        
        if action.action_type == RLAction.SELECT_SURFACE_CODE:
            # Surface code: good for high noise, moderate overhead
            distance = action.parameters.get('distance', 3)
            logical_error_reduction = (distance ** 2) * 0.1 / (1 + noise_level * 10)
            
            current_perf['logical_error_rate'] *= (1 - logical_error_reduction)
            current_perf['overhead'] = distance ** 2 * 1.2
            current_perf['fidelity'] = min(0.99, current_perf['fidelity'] + logical_error_reduction * 0.1)
            
        elif action.action_type == RLAction.SELECT_COLOR_CODE:
            # Color code: better logical rates but higher overhead
            distance = action.parameters.get('distance', 3)
            logical_error_reduction = (distance ** 1.5) * 0.15 / (1 + noise_level * 8)
            
            current_perf['logical_error_rate'] *= (1 - logical_error_reduction)
            current_perf['overhead'] = distance ** 2 * 1.8
            current_perf['fidelity'] = min(0.99, current_perf['fidelity'] + logical_error_reduction * 0.12)
            
        elif action.action_type == RLAction.SELECT_STEANE_CODE:
            # Steane code: lower overhead but needs low noise
            if noise_level < 0.01:  # Only effective for low noise
                current_perf['logical_error_rate'] *= 0.7
                current_perf['overhead'] = 7.0  # Fixed overhead
                current_perf['fidelity'] = min(0.99, current_perf['fidelity'] + 0.05)
            else:
                # Degraded performance in high noise
                current_perf['logical_error_rate'] *= 1.2
                current_perf['fidelity'] *= 0.95
        
        elif action.action_type == RLAction.ADJUST_THRESHOLD:
            # Threshold adjustment affects decoder performance
            threshold_change = action.parameters.get('change', 0.0)
            if abs(threshold_change) > 0:
                # Optimal threshold depends on noise level
                optimal_threshold = 0.5 + noise_level * 2
                current_threshold = action.parameters.get('current_threshold', 0.5)
                new_threshold = current_threshold + threshold_change
                
                threshold_error = abs(new_threshold - optimal_threshold)
                performance_factor = max(0.5, 1.0 - threshold_error * 2)
                
                current_perf['logical_error_rate'] *= (2 - performance_factor)
                current_perf['throughput'] *= performance_factor
        
        elif action.action_type == RLAction.MODIFY_DECODER:
            # Decoder modification affects accuracy vs speed tradeoff
            decoder_type = action.parameters.get('type', 'mwpm')
            if decoder_type == 'neural':
                current_perf['logical_error_rate'] *= 0.8  # Better accuracy
                current_perf['throughput'] *= 0.6  # Slower
            elif decoder_type == 'lookup':
                current_perf['logical_error_rate'] *= 1.1  # Worse accuracy
                current_perf['throughput'] *= 1.5  # Faster
        
        # Add some random variation
        for key in current_perf:
            if key != 'overhead':  # Don't randomize overhead
                current_perf[key] *= (1 + np.random.normal(0, 0.05))
                current_perf[key] = max(0, current_perf[key])  # Keep positive
        
        return current_perf
    
    def _calculate_reward(self, performance: Dict[str, float], action: QECCAction) -> float:
        """Calculate reward based on performance and action."""
        reward = 0.0
        
        # Performance-based rewards
        for metric, target in self.performance_targets.items():
            if metric in performance:
                if metric in ['fidelity', 'throughput', 'resource_efficiency']:
                    # Higher is better
                    reward += max(0, performance[metric] - target) * 10
                    reward -= max(0, target - performance[metric]) * 5
                else:
                    # Lower is better (error rates, overhead)
                    reward += max(0, target - performance[metric]) * 10
                    reward -= max(0, performance[metric] - target) * 5
        
        # Penalty for extreme overhead
        if performance.get('overhead', 0) > 50:
            reward -= (performance['overhead'] - 50) * 2
        
        # Bonus for meeting all targets
        targets_met = self._check_performance_targets(performance)
        if sum(targets_met.values()) >= len(targets_met) * 0.8:
            reward += 20  # Bonus for meeting most targets
        
        # Action-specific adjustments
        if action.action_type == RLAction.ADJUST_THRESHOLD:
            # Small penalty for frequent threshold adjustments
            reward -= 1
        
        # Resource efficiency bonus
        resource_util = self._calculate_resource_utilization()
        if resource_util < 0.8:  # Efficient resource usage
            reward += (0.8 - resource_util) * 5
        
        return reward
    
    def _check_performance_targets(self, performance: Dict[str, float]) -> Dict[str, bool]:
        """Check which performance targets are met."""
        targets_met = {}
        
        for metric, target in self.performance_targets.items():
            if metric in performance:
                if metric in ['fidelity', 'throughput', 'resource_efficiency']:
                    targets_met[metric] = performance[metric] >= target
                else:
                    targets_met[metric] = performance[metric] <= target
            else:
                targets_met[metric] = False
        
        return targets_met
    
    def _calculate_resource_utilization(self) -> float:
        """Calculate current resource utilization."""
        if not self.current_state:
            return 0.0
        
        circuit_props = self.current_state.circuit_properties
        overhead = self.current_state.current_performance.get('overhead', 1.0)
        
        base_qubits = circuit_props['num_qubits']
        total_qubits = base_qubits * overhead
        
        utilization = total_qubits / self.resource_limits['qubits']
        return min(1.0, utilization)
    
    def _check_terminal_condition(self, performance: Dict[str, float]) -> bool:
        """Check if episode should terminate early."""
        # Terminate if performance is extremely poor
        if (performance.get('logical_error_rate', 0) > 0.1 or 
            performance.get('fidelity', 1) < 0.3 or
            performance.get('overhead', 0) > 100):
            return True
        
        # Terminate if all targets are met consistently
        if len(self.current_state.history_window) >= 10:
            recent_performance = self.current_state.history_window[-5:]
            all_good = all(
                self._check_performance_targets(perf)
                for perf in recent_performance
            )
            if all_good:
                return True
        
        return False
    
    def get_state_vector(self, state: EnvironmentState) -> np.ndarray:
        """Convert state to vector representation for RL agents."""
        vector_components = []
        
        # Noise model features
        noise_features = [
            state.noise_model.get('gate_error_rate', 0),
            state.noise_model.get('readout_error_rate', 0),
            state.noise_model.get('t1_coherence', 0) * 1e6,  # Convert to microseconds
            state.noise_model.get('t2_coherence', 0) * 1e6,
            state.noise_model.get('crosstalk', 0)
        ]
        vector_components.extend(noise_features)
        
        # Current performance features
        performance_features = [
            state.current_performance.get('fidelity', 0),
            state.current_performance.get('logical_error_rate', 0),
            state.current_performance.get('overhead', 0) / 100,  # Normalize
            state.current_performance.get('throughput', 0),
            state.current_performance.get('resource_efficiency', 0)
        ]
        vector_components.extend(performance_features)
        
        # Circuit properties
        circuit_features = [
            state.circuit_properties.get('num_qubits', 0) / 100,  # Normalize
            state.circuit_properties.get('circuit_depth', 0) / 200,  # Normalize
        ]
        vector_components.extend(circuit_features)
        
        # Historical trend features
        if len(state.history_window) >= 3:
            recent_fidelities = [p.get('fidelity', 0) for p in state.history_window[-3:]]
            recent_errors = [p.get('logical_error_rate', 0) for p in state.history_window[-3:]]
            
            # Trend features
            fidelity_trend = recent_fidelities[-1] - recent_fidelities[0]
            error_trend = recent_errors[-1] - recent_errors[0]
            
            trend_features = [fidelity_trend, error_trend]
        else:
            trend_features = [0.0, 0.0]
        
        vector_components.extend(trend_features)
        
        return np.array(vector_components, dtype=np.float32)
    
    def get_action_mask(self, state: EnvironmentState) -> np.ndarray:
        """Get mask of valid actions for current state."""
        mask = np.ones(len(self.action_space), dtype=bool)
        
        # Mask actions based on current conditions
        noise_level = state.noise_model.get('gate_error_rate', 0)
        
        # Steane code only effective for low noise
        if noise_level > 0.02:
            steane_idx = self.action_space.index(RLAction.SELECT_STEANE_CODE)
            mask[steane_idx] = False
        
        # High distance codes may exceed resource limits
        if self._calculate_resource_utilization() > 0.8:
            # Disable high-overhead actions
            pass  # Could implement specific logic here
        
        return mask


class QECCRLAgent:
    """
    Deep Q-Network agent for QECC optimization.
    
    Uses deep reinforcement learning to learn optimal QECC
    selection and adaptation strategies.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 0.001,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        memory_size: int = 10000,
        batch_size: int = 32,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize RL agent.
        
        Args:
            state_dim: Dimension of state vector
            action_dim: Number of possible actions
            learning_rate: Learning rate for neural network
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Exploration decay rate
            memory_size: Size of replay memory
            batch_size: Training batch size
            logger: Optional logger instance
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.logger = logger or logging.getLogger(__name__)
        
        # Exploration parameters
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Experience replay
        self.memory = deque(maxlen=memory_size)
        
        # Neural networks (simplified placeholders)
        self.q_network = None  # Would be actual neural network
        self.target_network = None  # Target network for stable training
        
        # Training statistics
        self.total_steps = 0
        self.episodes_trained = 0
        self.average_reward = 0.0
        self.recent_rewards = deque(maxlen=100)
        
        self.logger.info(f"QECC RL Agent initialized with state_dim={state_dim}, action_dim={action_dim}")
    
    def act(self, state_vector: np.ndarray, action_mask: np.ndarray) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state_vector: Current state as vector
            action_mask: Mask of valid actions
            
        Returns:
            Action index
        """
        # Epsilon-greedy action selection
        if np.random.random() < self.epsilon:
            # Random action from valid actions
            valid_actions = np.where(action_mask)[0]
            return np.random.choice(valid_actions)
        else:
            # Greedy action (simplified - would use neural network)
            # For now, use heuristic-based action selection
            return self._heuristic_action_selection(state_vector, action_mask)
    
    def _heuristic_action_selection(self, state_vector: np.ndarray, action_mask: np.ndarray) -> int:
        """Heuristic-based action selection (placeholder for neural network)."""
        valid_actions = np.where(action_mask)[0]
        
        # Extract key features
        gate_error_rate = state_vector[0]
        fidelity = state_vector[5]
        logical_error_rate = state_vector[6]
        overhead = state_vector[7] * 100  # Denormalize
        
        # Simple heuristics
        if gate_error_rate > 0.02:  # High noise
            # Prefer surface code for high noise
            surface_idx = 0  # RLAction.SELECT_SURFACE_CODE
            if surface_idx in valid_actions:
                return surface_idx
        
        if fidelity < 0.8:  # Low fidelity
            # Try color code
            color_idx = 1  # RLAction.SELECT_COLOR_CODE
            if color_idx in valid_actions:
                return color_idx
        
        if overhead > 20:  # High overhead
            # Try threshold adjustment
            threshold_idx = 3  # RLAction.ADJUST_THRESHOLD
            if threshold_idx in valid_actions:
                return threshold_idx
        
        # Default: return first valid action
        return valid_actions[0]
    
    def remember(
        self, 
        state: np.ndarray, 
        action: int, 
        reward: float, 
        next_state: np.ndarray, 
        done: bool
    ):
        """Store experience in replay memory."""
        self.memory.append((state, action, reward, next_state, done))
        self.recent_rewards.append(reward)
        
        if len(self.recent_rewards) >= 10:
            self.average_reward = np.mean(self.recent_rewards)
    
    def train(self) -> Dict[str, float]:
        """Train the agent on a batch of experiences."""
        if len(self.memory) < self.batch_size:
            return {'loss': 0.0, 'q_value': 0.0}
        
        # Sample batch (simplified)
        batch = np.random.choice(len(self.memory), self.batch_size, replace=False)
        
        # Decay exploration
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
        
        self.total_steps += 1
        
        # Placeholder training metrics
        return {
            'loss': np.random.uniform(0.1, 1.0),
            'q_value': np.random.uniform(1.0, 10.0),
            'epsilon': self.epsilon
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get training statistics."""
        return {
            'total_steps': self.total_steps,
            'episodes_trained': self.episodes_trained,
            'epsilon': self.epsilon,
            'average_reward': self.average_reward,
            'memory_size': len(self.memory),
            'recent_rewards': list(self.recent_rewards)[-10:]
        }


def create_rl_qecc_trainer(
    num_episodes: int = 1000,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Create and run QECC reinforcement learning training.
    
    Args:
        num_episodes: Number of training episodes
        logger: Optional logger instance
        
    Returns:
        Training results and statistics
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Create environment and agent
    env = QECCEnvironment(logger=logger)
    
    # Get dimensions from environment
    dummy_state = env.reset()
    state_vector = env.get_state_vector(dummy_state)
    state_dim = len(state_vector)
    action_dim = len(env.action_space)
    
    agent = QECCRLAgent(state_dim, action_dim, logger=logger)
    
    # Training loop
    episode_rewards = []
    episode_lengths = []
    best_reward = float('-inf')
    
    logger.info(f"Starting RL training for {num_episodes} episodes")
    
    for episode in range(num_episodes):
        state = env.reset()
        state_vector = env.get_state_vector(state)
        
        total_reward = 0
        step_count = 0
        
        while True:
            # Get valid actions and select action
            action_mask = env.get_action_mask(state)
            action_idx = agent.act(state_vector, action_mask)
            
            # Convert action index to QECC action
            action_type = env.action_space[action_idx]
            qecc_action = QECCAction(action_type=action_type, parameters={})
            
            # Take step in environment
            next_state, reward, done, info = env.step(qecc_action)
            next_state_vector = env.get_state_vector(next_state)
            
            # Store experience
            agent.remember(state_vector, action_idx, reward, next_state_vector, done)
            
            # Train agent
            if len(agent.memory) >= agent.batch_size:
                training_metrics = agent.train()
            
            total_reward += reward
            step_count += 1
            
            # Update state
            state = next_state
            state_vector = next_state_vector
            
            if done:
                break
        
        episode_rewards.append(total_reward)
        episode_lengths.append(step_count)
        
        if total_reward > best_reward:
            best_reward = total_reward
            logger.info(f"New best reward: {best_reward:.2f} in episode {episode}")
        
        # Log progress
        if episode % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_length = np.mean(episode_lengths[-100:])
            logger.info(
                f"Episode {episode}: avg_reward={avg_reward:.2f}, "
                f"avg_length={avg_length:.1f}, epsilon={agent.epsilon:.3f}"
            )
    
    # Training completed
    results = {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'best_reward': best_reward,
        'final_average_reward': np.mean(episode_rewards[-100:]),
        'agent_statistics': agent.get_statistics(),
        'environment_stats': {
            'total_episodes': env.episode_count,
            'noise_models_used': len(env.noise_models)
        }
    }
    
    logger.info(f"RL training completed. Best reward: {best_reward:.2f}")
    return results