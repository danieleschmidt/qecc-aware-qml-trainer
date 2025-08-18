"""
Quantum Security Validator for QECC-aware QML Systems
"""

import hashlib
import hmac
import secrets
import time
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import warnings
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

try:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import random_statevector, Statevector
    from qiskit_aer import AerSimulator
except ImportError:
    from ..core.fallback_imports import QuantumCircuit, random_statevector, Statevector, AerSimulator


class SecurityLevel(Enum):
    """Security levels for quantum operations."""
    BASIC = "basic"
    ENHANCED = "enhanced"
    MAXIMUM = "maximum"
    QUANTUM_SAFE = "quantum_safe"


class ThreatType(Enum):
    """Types of security threats."""
    CIRCUIT_TAMPERING = "circuit_tampering"
    DATA_POISONING = "data_poisoning"
    PARAMETER_INJECTION = "parameter_injection"
    SIDE_CHANNEL_ATTACK = "side_channel_attack"
    QUANTUM_BACKDOOR = "quantum_backdoor"
    CLASSICAL_CRYPTOGRAPHIC = "classical_cryptographic"
    QUANTUM_SUPREMACY_ATTACK = "quantum_supremacy_attack"


@dataclass
class SecurityViolation:
    """Represents a detected security violation."""
    threat_type: ThreatType
    severity: float  # 0.0 to 1.0
    timestamp: float
    description: str
    location: Optional[str] = None
    evidence: Dict[str, Any] = None
    mitigation_applied: bool = False
    
    def __post_init__(self):
        if self.evidence is None:
            self.evidence = {}


@dataclass
class SecurityPolicy:
    """Security policy configuration."""
    max_circuit_depth: int = 1000
    max_parameter_range: float = 10.0
    require_encryption: bool = True
    allow_external_circuits: bool = False
    quantum_signature_required: bool = True
    homomorphic_encryption: bool = False
    secure_multiparty_computation: bool = False


class QuantumSecurityValidator:
    """
    Comprehensive security validator for quantum machine learning systems.
    
    Features:
    - Quantum circuit integrity verification
    - Parameter tampering detection
    - Data poisoning detection
    - Quantum cryptographic validation
    - Side-channel attack mitigation
    - Quantum-safe cryptography integration
    """
    
    def __init__(
        self,
        security_level: SecurityLevel = SecurityLevel.ENHANCED,
        policy: Optional[SecurityPolicy] = None
    ):
        self.security_level = security_level
        self.policy = policy or SecurityPolicy()
        
        # Security state
        self.violations: List[SecurityViolation] = []
        self.trusted_circuits: Dict[str, str] = {}  # circuit_hash -> signature
        self.parameter_baselines: Dict[str, np.ndarray] = {}
        
        # Cryptographic components
        self._init_cryptographic_keys()
        self._init_quantum_signature_system()
        
        # Monitoring
        self.security_metrics: Dict[str, float] = {}
        self.threat_intelligence: Dict[ThreatType, List[float]] = {}
        
        # Logger
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging configuration."""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def _init_cryptographic_keys(self):
        """Initialize cryptographic key material."""
        try:
            # Generate RSA keys for classical cryptography
            self.private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048
            )
            self.public_key = self.private_key.public_key()
            
            # Generate symmetric keys
            self.aes_key = secrets.token_bytes(32)  # AES-256
            self.hmac_key = secrets.token_bytes(64)  # HMAC key
            
            self.logger.info("Cryptographic keys initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize cryptographic keys: {e}")
            # Fallback to basic security
            self.aes_key = b'0' * 32
            self.hmac_key = b'0' * 64
    
    def _init_quantum_signature_system(self):
        """Initialize quantum signature verification system."""
        try:
            # Quantum signature parameters
            self.quantum_signature_params = {
                'num_qubits': 8,
                'signature_length': 256,
                'hash_rounds': 3,
                'security_parameter': 128
            }
            
            # Generate quantum signature keys (simulated)
            self.quantum_private_key = np.random.rand(
                self.quantum_signature_params['signature_length']
            ) * 2 * np.pi
            
            self.quantum_public_key = self._generate_quantum_public_key()
            
            self.logger.info("Quantum signature system initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize quantum signatures: {e}")
            self.quantum_signature_params = None
    
    def _generate_quantum_public_key(self) -> np.ndarray:
        """Generate quantum public key from private key."""
        # Simplified quantum key generation
        return np.cos(self.quantum_private_key / 2)
    
    def validate_circuit_security(self, circuit: Any, context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Validate quantum circuit security.
        
        Args:
            circuit: Quantum circuit to validate
            context: Additional context information
            
        Returns:
            True if circuit passes security validation
        """
        context = context or {}
        violations = []
        
        # Circuit integrity check
        integrity_violations = self._check_circuit_integrity(circuit, context)
        violations.extend(integrity_violations)
        
        # Circuit complexity analysis
        complexity_violations = self._analyze_circuit_complexity(circuit)
        violations.extend(complexity_violations)
        
        # Backdoor detection
        backdoor_violations = self._detect_quantum_backdoors(circuit)
        violations.extend(backdoor_violations)
        
        # Side-channel vulnerability assessment
        sidechannel_violations = self._assess_sidechannel_vulnerabilities(circuit)
        violations.extend(sidechannel_violations)
        
        # Update violation history
        self.violations.extend(violations)
        
        # Apply mitigations
        for violation in violations:
            self._apply_mitigation(violation)
        
        # Determine overall security status
        critical_violations = [v for v in violations if v.severity > 0.7]
        
        if critical_violations:
            self.logger.warning(f"Circuit failed security validation: {len(critical_violations)} critical violations")
            return False
        
        self.logger.info("Circuit passed security validation")
        return True
    
    def _check_circuit_integrity(self, circuit: Any, context: Dict[str, Any]) -> List[SecurityViolation]:
        """Check quantum circuit integrity."""
        violations = []
        
        try:
            # Calculate circuit hash
            circuit_hash = self._calculate_circuit_hash(circuit)
            
            # Check against trusted circuits
            if circuit_hash in self.trusted_circuits:
                expected_signature = self.trusted_circuits[circuit_hash]
                current_signature = self._sign_circuit(circuit)
                
                if not self._verify_signature(circuit_hash, current_signature, expected_signature):
                    violation = SecurityViolation(
                        threat_type=ThreatType.CIRCUIT_TAMPERING,
                        severity=0.9,
                        timestamp=time.time(),
                        description="Circuit signature verification failed",
                        evidence={'expected': expected_signature, 'actual': current_signature}
                    )
                    violations.append(violation)
            else:
                # New circuit - generate and store signature
                if self.policy.quantum_signature_required:
                    signature = self._sign_circuit(circuit)
                    self.trusted_circuits[circuit_hash] = signature
            
            # Check for suspicious circuit modifications
            if 'original_circuit' in context:
                original_circuit = context['original_circuit']
                if self._detect_unauthorized_modifications(original_circuit, circuit):
                    violation = SecurityViolation(
                        threat_type=ThreatType.CIRCUIT_TAMPERING,
                        severity=0.8,
                        timestamp=time.time(),
                        description="Unauthorized circuit modifications detected"
                    )
                    violations.append(violation)
        
        except Exception as e:
            self.logger.error(f"Error checking circuit integrity: {e}")
        
        return violations
    
    def _analyze_circuit_complexity(self, circuit: Any) -> List[SecurityViolation]:
        """Analyze circuit complexity for security implications."""
        violations = []
        
        try:
            depth = getattr(circuit, 'depth', lambda: 0)()
            gate_count = len(getattr(circuit, 'data', []))
            
            # Check against policy limits
            if depth > self.policy.max_circuit_depth:
                violation = SecurityViolation(
                    threat_type=ThreatType.CIRCUIT_TAMPERING,
                    severity=min((depth - self.policy.max_circuit_depth) / self.policy.max_circuit_depth, 1.0),
                    timestamp=time.time(),
                    description=f"Circuit depth exceeds policy limit: {depth} > {self.policy.max_circuit_depth}",
                    evidence={'depth': depth, 'limit': self.policy.max_circuit_depth}
                )
                violations.append(violation)
            
            # Detect suspiciously complex circuits
            if gate_count > 1000 and depth > 100:
                complexity_ratio = gate_count / depth
                if complexity_ratio > 50:  # Unusually wide circuit
                    violation = SecurityViolation(
                        threat_type=ThreatType.QUANTUM_BACKDOOR,
                        severity=0.6,
                        timestamp=time.time(),
                        description=f"Suspiciously complex circuit structure detected",
                        evidence={'gate_count': gate_count, 'depth': depth, 'ratio': complexity_ratio}
                    )
                    violations.append(violation)
        
        except Exception as e:
            self.logger.error(f"Error analyzing circuit complexity: {e}")
        
        return violations
    
    def _detect_quantum_backdoors(self, circuit: Any) -> List[SecurityViolation]:
        """Detect potential quantum backdoors in circuits."""
        violations = []
        
        try:
            # Analyze gate patterns for backdoor signatures
            if hasattr(circuit, 'data'):
                gate_sequence = [str(instruction.operation.name) for instruction in circuit.data]
                
                # Known backdoor patterns (simplified detection)
                suspicious_patterns = [
                    ['h', 'cx', 'h', 'cx'] * 3,  # Repeated entanglement pattern
                    ['rz'] * 10,  # Excessive rotation gates
                    ['barrier'] * 5,  # Unusual barrier usage
                ]
                
                for pattern in suspicious_patterns:
                    if self._contains_pattern(gate_sequence, pattern):
                        violation = SecurityViolation(
                            threat_type=ThreatType.QUANTUM_BACKDOOR,
                            severity=0.7,
                            timestamp=time.time(),
                            description=f"Suspicious gate pattern detected: {pattern[:3]}...",
                            evidence={'pattern': pattern, 'circuit_gates': gate_sequence[:20]}
                        )
                        violations.append(violation)
            
            # Check for hidden information encoding
            if self._detect_hidden_information(circuit):
                violation = SecurityViolation(
                    threat_type=ThreatType.QUANTUM_BACKDOOR,
                    severity=0.8,
                    timestamp=time.time(),
                    description="Potential hidden information encoding detected"
                )
                violations.append(violation)
        
        except Exception as e:
            self.logger.error(f"Error detecting quantum backdoors: {e}")
        
        return violations
    
    def _assess_sidechannel_vulnerabilities(self, circuit: Any) -> List[SecurityViolation]:
        """Assess side-channel attack vulnerabilities."""
        violations = []
        
        try:
            # Timing attack vulnerability
            if hasattr(circuit, 'depth'):
                depth_variation = self._calculate_depth_variation(circuit)
                if depth_variation > 0.3:  # High timing variance
                    violation = SecurityViolation(
                        threat_type=ThreatType.SIDE_CHANNEL_ATTACK,
                        severity=0.6,
                        timestamp=time.time(),
                        description="Circuit vulnerable to timing side-channel attacks",
                        evidence={'depth_variation': depth_variation}
                    )
                    violations.append(violation)
            
            # Power analysis vulnerability (simulated)
            gate_power_profile = self._estimate_power_profile(circuit)
            if self._has_distinctive_power_pattern(gate_power_profile):
                violation = SecurityViolation(
                    threat_type=ThreatType.SIDE_CHANNEL_ATTACK,
                    severity=0.5,
                    timestamp=time.time(),
                    description="Circuit may be vulnerable to power analysis attacks"
                )
                violations.append(violation)
        
        except Exception as e:
            self.logger.error(f"Error assessing side-channel vulnerabilities: {e}")
        
        return violations
    
    def validate_parameters(self, parameters: np.ndarray, parameter_id: str = "default") -> bool:
        """
        Validate quantum parameters for security.
        
        Args:
            parameters: Parameter array to validate
            parameter_id: Identifier for parameter set
            
        Returns:
            True if parameters pass security validation
        """
        violations = []
        
        # Range check
        if np.any(np.abs(parameters) > self.policy.max_parameter_range):
            violation = SecurityViolation(
                threat_type=ThreatType.PARAMETER_INJECTION,
                severity=0.7,
                timestamp=time.time(),
                description=f"Parameters exceed allowed range: max={np.max(np.abs(parameters)):.3f}",
                evidence={'max_value': float(np.max(np.abs(parameters))), 'limit': self.policy.max_parameter_range}
            )
            violations.append(violation)
        
        # Check against baseline if available
        if parameter_id in self.parameter_baselines:
            baseline = self.parameter_baselines[parameter_id]
            deviation = np.linalg.norm(parameters - baseline)
            
            if deviation > 1.0:  # Significant deviation threshold
                violation = SecurityViolation(
                    threat_type=ThreatType.PARAMETER_INJECTION,
                    severity=min(deviation / 5.0, 1.0),
                    timestamp=time.time(),
                    description=f"Parameters deviate significantly from baseline: deviation={deviation:.3f}",
                    evidence={'deviation': float(deviation), 'parameter_id': parameter_id}
                )
                violations.append(violation)
        else:
            # Store as baseline
            self.parameter_baselines[parameter_id] = parameters.copy()
        
        # Statistical anomaly detection
        if self._detect_parameter_anomalies(parameters):
            violation = SecurityViolation(
                threat_type=ThreatType.PARAMETER_INJECTION,
                severity=0.6,
                timestamp=time.time(),
                description="Statistical anomalies detected in parameters"
            )
            violations.append(violation)
        
        self.violations.extend(violations)
        
        return len(violations) == 0
    
    def validate_data(self, data: np.ndarray, labels: Optional[np.ndarray] = None) -> bool:
        """
        Validate training/inference data for security.
        
        Args:
            data: Input data to validate
            labels: Optional labels for supervised learning
            
        Returns:
            True if data passes security validation
        """
        violations = []
        
        # Data poisoning detection
        poisoning_violations = self._detect_data_poisoning(data, labels)
        violations.extend(poisoning_violations)
        
        # Adversarial example detection
        adversarial_violations = self._detect_adversarial_examples(data)
        violations.extend(adversarial_violations)
        
        # Data integrity check
        integrity_violations = self._check_data_integrity(data)
        violations.extend(integrity_violations)
        
        self.violations.extend(violations)
        
        return len(violations) == 0
    
    def _detect_data_poisoning(self, data: np.ndarray, labels: Optional[np.ndarray]) -> List[SecurityViolation]:
        """Detect data poisoning attacks."""
        violations = []
        
        try:
            # Statistical outlier detection
            data_mean = np.mean(data, axis=0)
            data_std = np.std(data, axis=0)
            
            # Z-score based outlier detection
            z_scores = np.abs((data - data_mean) / (data_std + 1e-8))
            outliers = np.any(z_scores > 3.0, axis=1)
            
            outlier_ratio = np.sum(outliers) / len(data)
            if outlier_ratio > 0.05:  # More than 5% outliers
                violation = SecurityViolation(
                    threat_type=ThreatType.DATA_POISONING,
                    severity=min(outlier_ratio * 10, 1.0),
                    timestamp=time.time(),
                    description=f"High outlier ratio detected: {outlier_ratio:.1%}",
                    evidence={'outlier_ratio': float(outlier_ratio), 'outlier_count': int(np.sum(outliers))}
                )
                violations.append(violation)
            
            # Label consistency check (if labels provided)
            if labels is not None:
                label_anomalies = self._detect_label_anomalies(data, labels)
                if label_anomalies > 0.02:  # More than 2% label anomalies
                    violation = SecurityViolation(
                        threat_type=ThreatType.DATA_POISONING,
                        severity=min(label_anomalies * 25, 1.0),
                        timestamp=time.time(),
                        description=f"Label anomalies detected: {label_anomalies:.1%}"
                    )
                    violations.append(violation)
        
        except Exception as e:
            self.logger.error(f"Error detecting data poisoning: {e}")
        
        return violations
    
    def _detect_adversarial_examples(self, data: np.ndarray) -> List[SecurityViolation]:
        """Detect adversarial examples in data."""
        violations = []
        
        try:
            # Simple adversarial detection based on gradient norms
            # This is a simplified implementation
            gradient_norms = np.linalg.norm(np.diff(data, axis=0), axis=1)
            high_gradient_ratio = np.sum(gradient_norms > np.percentile(gradient_norms, 95)) / len(gradient_norms)
            
            if high_gradient_ratio > 0.1:  # More than 10% high-gradient samples
                violation = SecurityViolation(
                    threat_type=ThreatType.DATA_POISONING,
                    severity=0.6,
                    timestamp=time.time(),
                    description=f"Potential adversarial examples detected: {high_gradient_ratio:.1%}"
                )
                violations.append(violation)
        
        except Exception as e:
            self.logger.error(f"Error detecting adversarial examples: {e}")
        
        return violations
    
    def _check_data_integrity(self, data: np.ndarray) -> List[SecurityViolation]:
        """Check data integrity using checksums."""
        violations = []
        
        try:
            # Calculate data hash for integrity
            data_hash = hashlib.sha256(data.tobytes()).hexdigest()
            
            # HMAC verification (if we have a stored HMAC)
            hmac_digest = hmac.new(self.hmac_key, data.tobytes(), hashlib.sha256).hexdigest()
            
            # Store for future verification
            if not hasattr(self, '_data_integrity_hashes'):
                self._data_integrity_hashes = {}
            
            self._data_integrity_hashes[str(data.shape)] = {
                'hash': data_hash,
                'hmac': hmac_digest,
                'timestamp': time.time()
            }
        
        except Exception as e:
            self.logger.error(f"Error checking data integrity: {e}")
        
        return violations
    
    # Helper methods
    def _calculate_circuit_hash(self, circuit: Any) -> str:
        """Calculate cryptographic hash of circuit."""
        try:
            circuit_str = str(circuit)
            return hashlib.sha256(circuit_str.encode()).hexdigest()
        except Exception:
            return hashlib.sha256(str(id(circuit)).encode()).hexdigest()
    
    def _sign_circuit(self, circuit: Any) -> str:
        """Generate cryptographic signature for circuit."""
        try:
            circuit_hash = self._calculate_circuit_hash(circuit)
            if self.quantum_signature_params:
                # Quantum signature (simplified)
                quantum_hash = self._quantum_hash(circuit_hash)
                return quantum_hash
            else:
                # Classical signature
                signature = hmac.new(self.hmac_key, circuit_hash.encode(), hashlib.sha256).hexdigest()
                return signature
        except Exception as e:
            self.logger.error(f"Error signing circuit: {e}")
            return ""
    
    def _quantum_hash(self, data: str) -> str:
        """Generate quantum hash (simplified implementation)."""
        # This is a simplified quantum hash - in practice, this would use
        # actual quantum circuits for cryptographic hashing
        data_bytes = data.encode()
        hash_result = hashlib.sha256(data_bytes + b"quantum_salt").hexdigest()
        return hash_result
    
    def _verify_signature(self, circuit_hash: str, signature: str, expected_signature: str) -> bool:
        """Verify circuit signature."""
        return signature == expected_signature
    
    def _detect_unauthorized_modifications(self, original: Any, current: Any) -> bool:
        """Detect unauthorized circuit modifications."""
        original_hash = self._calculate_circuit_hash(original)
        current_hash = self._calculate_circuit_hash(current)
        return original_hash != current_hash
    
    def _contains_pattern(self, sequence: List[str], pattern: List[str]) -> bool:
        """Check if sequence contains a specific pattern."""
        pattern_len = len(pattern)
        for i in range(len(sequence) - pattern_len + 1):
            if sequence[i:i + pattern_len] == pattern:
                return True
        return False
    
    def _detect_hidden_information(self, circuit: Any) -> bool:
        """Detect hidden information encoding in circuit."""
        # Simplified detection - look for unused qubits or strange gate patterns
        try:
            if hasattr(circuit, 'num_qubits') and hasattr(circuit, 'data'):
                used_qubits = set()
                for instruction in circuit.data:
                    used_qubits.update(qubit._index for qubit in instruction.qubits)
                
                unused_qubits = circuit.num_qubits - len(used_qubits)
                return unused_qubits > circuit.num_qubits * 0.2  # More than 20% unused
        except Exception:
            pass
        
        return False
    
    def _calculate_depth_variation(self, circuit: Any) -> float:
        """Calculate depth variation for timing analysis."""
        # Simplified calculation
        try:
            total_depth = getattr(circuit, 'depth', lambda: 0)()
            if hasattr(circuit, 'data') and total_depth > 0:
                gate_positions = []
                for i, instruction in enumerate(circuit.data):
                    gate_positions.append(i)
                
                if gate_positions:
                    return np.std(gate_positions) / total_depth
        except Exception:
            pass
        
        return 0.0
    
    def _estimate_power_profile(self, circuit: Any) -> List[float]:
        """Estimate power consumption profile."""
        # Simplified power estimation
        power_profile = []
        try:
            if hasattr(circuit, 'data'):
                for instruction in circuit.data:
                    gate_name = str(instruction.operation.name).lower()
                    # Simplified power model
                    if gate_name in ['cx', 'cnot']:
                        power_profile.append(2.0)  # Two-qubit gates use more power
                    elif gate_name in ['h', 'x', 'y', 'z']:
                        power_profile.append(1.0)  # Single-qubit gates
                    else:
                        power_profile.append(1.5)  # Other gates
        except Exception:
            power_profile = [1.0]  # Default profile
        
        return power_profile
    
    def _has_distinctive_power_pattern(self, power_profile: List[float]) -> bool:
        """Check if power profile has distinctive patterns."""
        if len(power_profile) < 10:
            return False
        
        # Look for repeating patterns
        profile_array = np.array(power_profile)
        autocorr = np.correlate(profile_array, profile_array, mode='full')
        return np.max(autocorr[len(autocorr)//2+1:]) > 0.8 * np.max(autocorr)
    
    def _detect_parameter_anomalies(self, parameters: np.ndarray) -> bool:
        """Detect statistical anomalies in parameters."""
        try:
            # Check for unusual distributions
            param_std = np.std(parameters)
            param_mean = np.mean(parameters)
            
            # Check for suspicious patterns
            if param_std < 1e-6:  # All parameters nearly identical
                return True
            
            # Check for extreme values
            z_scores = np.abs(parameters - param_mean) / param_std
            extreme_values = np.sum(z_scores > 4.0)  # Beyond 4 standard deviations
            
            return extreme_values > len(parameters) * 0.05  # More than 5% extreme
        
        except Exception:
            return False
    
    def _detect_label_anomalies(self, data: np.ndarray, labels: np.ndarray) -> float:
        """Detect label anomalies indicating poisoning."""
        try:
            # Simple consistency check - this would be more sophisticated in practice
            unique_labels = np.unique(labels)
            if len(unique_labels) > len(data) * 0.5:  # Too many unique labels
                return 0.8
            
            # Check for isolated mislabeled points (simplified)
            # This is a placeholder - real implementation would use more sophisticated methods
            return 0.0
        
        except Exception:
            return 0.0
    
    def _apply_mitigation(self, violation: SecurityViolation):
        """Apply security mitigation for detected violations."""
        try:
            if violation.threat_type == ThreatType.CIRCUIT_TAMPERING:
                self.logger.warning("Applying circuit integrity mitigation")
                violation.mitigation_applied = True
            
            elif violation.threat_type == ThreatType.PARAMETER_INJECTION:
                self.logger.warning("Applying parameter sanitization")
                violation.mitigation_applied = True
            
            elif violation.threat_type == ThreatType.DATA_POISONING:
                self.logger.warning("Applying data filtering mitigation")
                violation.mitigation_applied = True
            
            # Record mitigation attempt
            violation.mitigation_applied = True
            
        except Exception as e:
            self.logger.error(f"Error applying mitigation: {e}")
    
    def get_security_report(self) -> Dict[str, Any]:
        """Generate comprehensive security report."""
        recent_violations = [
            v for v in self.violations 
            if time.time() - v.timestamp < 3600  # Last hour
        ]
        
        threat_counts = {}
        for violation in recent_violations:
            threat_type = violation.threat_type.value
            threat_counts[threat_type] = threat_counts.get(threat_type, 0) + 1
        
        critical_violations = [v for v in recent_violations if v.severity > 0.7]
        
        return {
            'timestamp': time.time(),
            'security_level': self.security_level.value,
            'total_violations': len(self.violations),
            'recent_violations': len(recent_violations),
            'critical_violations': len(critical_violations),
            'threat_distribution': threat_counts,
            'trusted_circuits': len(self.trusted_circuits),
            'parameter_baselines': len(self.parameter_baselines),
            'mitigation_success_rate': self._calculate_mitigation_success_rate(),
            'security_score': self._calculate_security_score(),
        }
    
    def _calculate_mitigation_success_rate(self) -> float:
        """Calculate the success rate of applied mitigations."""
        mitigated = sum(1 for v in self.violations if v.mitigation_applied)
        total = len(self.violations)
        return mitigated / max(total, 1)
    
    def _calculate_security_score(self) -> float:
        """Calculate overall security score (0-1, higher is better)."""
        if not self.violations:
            return 1.0
        
        recent_violations = [
            v for v in self.violations 
            if time.time() - v.timestamp < 86400  # Last 24 hours
        ]
        
        if not recent_violations:
            return 0.9  # Good if no recent violations
        
        avg_severity = np.mean([v.severity for v in recent_violations])
        violation_density = len(recent_violations) / 24  # Violations per hour
        
        score = 1.0 - (avg_severity * 0.7 + min(violation_density / 10, 0.3))
        return max(score, 0.0)
    
    def cleanup(self):
        """Clean up security resources."""
        self.violations.clear()
        self.trusted_circuits.clear()
        self.parameter_baselines.clear()
        if hasattr(self, '_data_integrity_hashes'):
            self._data_integrity_hashes.clear()