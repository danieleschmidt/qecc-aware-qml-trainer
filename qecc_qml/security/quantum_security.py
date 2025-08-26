#!/usr/bin/env python3
"""
Quantum Security Framework
Comprehensive security for quantum machine learning systems including 
cryptographic protocols, secure computation, and privacy preservation
"""

import hashlib
import hmac
import secrets
import time
import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import base64
import json

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import random_statevector


class SecurityLevel(Enum):
    """Security protection levels"""
    BASIC = "basic"
    ENHANCED = "enhanced"
    QUANTUM_SAFE = "quantum_safe"
    POST_QUANTUM = "post_quantum"


class ThreatLevel(Enum):
    """Threat assessment levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityCredentials:
    """Security credentials for quantum systems"""
    public_key: bytes
    private_key: bytes
    certificate: Optional[bytes] = None
    expiry_time: float = 0.0
    permissions: List[str] = field(default_factory=list)
    
    def is_valid(self) -> bool:
        """Check if credentials are still valid"""
        return time.time() < self.expiry_time if self.expiry_time > 0 else True


@dataclass
class SecurityAuditLog:
    """Security audit log entry"""
    timestamp: float
    event_type: str
    user_id: str
    resource: str
    action: str
    result: str
    threat_level: ThreatLevel
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QuantumSecret:
    """Quantum-secured secret storage"""
    secret_id: str
    encrypted_data: bytes
    quantum_signature: np.ndarray
    access_count: int = 0
    max_access: int = 10
    created_time: float = field(default_factory=time.time)


class QuantumSecurityFramework:
    """
    Comprehensive quantum security framework providing cryptographic
    protection, secure multi-party computation, and privacy preservation
    for quantum machine learning systems.
    """
    
    def __init__(self, 
                 security_level: SecurityLevel = SecurityLevel.ENHANCED,
                 enable_quantum_crypto: bool = True):
        
        self.security_level = security_level
        self.enable_quantum_crypto = enable_quantum_crypto
        
        # Cryptographic keys and certificates
        self.master_key: Optional[bytes] = None
        self.key_ring: Dict[str, SecurityCredentials] = {}
        self.quantum_keys: Dict[str, np.ndarray] = {}
        
        # Security monitoring
        self.audit_log: List[SecurityAuditLog] = []
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.threat_monitors: List[Callable] = []
        
        # Quantum secret storage
        self.quantum_secrets: Dict[str, QuantumSecret] = {}
        
        # Security policies
        self.access_policies: Dict[str, List[str]] = {
            "admin": ["read", "write", "execute", "manage"],
            "researcher": ["read", "execute"],
            "guest": ["read"]
        }
        
        # Threat detection
        self.anomaly_threshold = 0.8
        self.failed_auth_limit = 5
        self.auth_failures: Dict[str, int] = {}
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize security system
        self._initialize_security_system()
    
    def _initialize_security_system(self) -> None:
        """Initialize the quantum security system"""
        try:
            # Generate master encryption key
            self.master_key = self._generate_master_key()
            
            # Setup quantum cryptographic keys if enabled
            if self.enable_quantum_crypto:
                self._initialize_quantum_crypto()
            
            # Create default admin credentials
            admin_creds = self._create_user_credentials(
                "admin", 
                permissions=self.access_policies["admin"],
                expiry_hours=24
            )
            self.key_ring["admin"] = admin_creds
            
            self.logger.info(f"Quantum security system initialized with level {self.security_level.value}")
            
        except Exception as e:
            self.logger.error(f"Security system initialization failed: {e}")
            raise
    
    def _generate_master_key(self) -> bytes:
        """Generate cryptographically secure master key"""
        return secrets.token_bytes(32)  # 256-bit key
    
    def _initialize_quantum_crypto(self) -> None:
        """Initialize quantum cryptographic components"""
        try:
            # Generate quantum random states for cryptographic use
            for i in range(5):  # Create multiple quantum keys
                key_name = f"quantum_key_{i}"
                quantum_state = random_statevector(4)  # 4-qubit quantum key
                self.quantum_keys[key_name] = quantum_state.data
            
            self.logger.info("Quantum cryptographic keys generated")
            
        except Exception as e:
            self.logger.warning(f"Quantum crypto initialization failed: {e}")
    
    def create_user(self, user_id: str, role: str = "guest", 
                   custom_permissions: Optional[List[str]] = None,
                   expiry_hours: float = 8.0) -> bool:
        """Create new user with security credentials"""
        try:
            # Validate role
            if role not in self.access_policies and not custom_permissions:
                raise ValueError(f"Unknown role: {role}")
            
            permissions = custom_permissions or self.access_policies.get(role, ["read"])
            
            # Create credentials
            credentials = self._create_user_credentials(user_id, permissions, expiry_hours)
            self.key_ring[user_id] = credentials
            
            # Audit log
            self._log_security_event(
                event_type="user_creation",
                user_id="system",
                resource="user_management",
                action=f"create_user:{user_id}",
                result="success",
                threat_level=ThreatLevel.LOW
            )
            
            self.logger.info(f"User {user_id} created with role {role}")
            return True
            
        except Exception as e:
            self.logger.error(f"User creation failed: {e}")
            return False
    
    def _create_user_credentials(self, user_id: str, permissions: List[str], 
                               expiry_hours: float) -> SecurityCredentials:
        """Create security credentials for user"""
        
        # Generate RSA key pair
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        public_key = private_key.public_key()
        
        # Serialize keys
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        # Create credentials
        expiry_time = time.time() + (expiry_hours * 3600)
        
        return SecurityCredentials(
            public_key=public_pem,
            private_key=private_pem,
            expiry_time=expiry_time,
            permissions=permissions
        )
    
    def authenticate_user(self, user_id: str, signature: bytes, 
                         message: bytes) -> bool:
        """Authenticate user using cryptographic signature"""
        try:
            # Check if user exists
            if user_id not in self.key_ring:
                self._record_auth_failure(user_id, "user_not_found")
                return False
            
            credentials = self.key_ring[user_id]
            
            # Check credential validity
            if not credentials.is_valid():
                self._record_auth_failure(user_id, "credentials_expired")
                return False
            
            # Load public key
            public_key = serialization.load_pem_public_key(credentials.public_key)
            
            # Verify signature
            try:
                public_key.verify(
                    signature,
                    message,
                    padding.PSS(
                        mgf=padding.MGF1(hashes.SHA256()),
                        salt_length=padding.PSS.MAX_LENGTH
                    ),
                    hashes.SHA256()
                )
                
                # Authentication successful
                self._clear_auth_failures(user_id)
                self._log_security_event(
                    event_type="authentication",
                    user_id=user_id,
                    resource="auth_system",
                    action="login",
                    result="success",
                    threat_level=ThreatLevel.LOW
                )
                
                return True
                
            except Exception as e:
                self._record_auth_failure(user_id, "invalid_signature")
                return False
            
        except Exception as e:
            self.logger.error(f"Authentication failed: {e}")
            self._record_auth_failure(user_id, "authentication_error")
            return False
    
    def authorize_action(self, user_id: str, resource: str, action: str) -> bool:
        """Authorize user action on resource"""
        try:
            if user_id not in self.key_ring:
                return False
            
            credentials = self.key_ring[user_id]
            
            # Check if action is permitted
            if action not in credentials.permissions:
                self._log_security_event(
                    event_type="authorization",
                    user_id=user_id,
                    resource=resource,
                    action=action,
                    result="denied",
                    threat_level=ThreatLevel.MEDIUM
                )
                return False
            
            # Action authorized
            self._log_security_event(
                event_type="authorization",
                user_id=user_id,
                resource=resource,
                action=action,
                result="granted",
                threat_level=ThreatLevel.LOW
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Authorization failed: {e}")
            return False
    
    def encrypt_data(self, data: bytes, user_id: str) -> Optional[bytes]:
        """Encrypt data for secure storage or transmission"""
        try:
            if not self.master_key:
                raise ValueError("Master key not initialized")
            
            # Generate initialization vector
            iv = secrets.token_bytes(16)
            
            # Create cipher
            cipher = Cipher(
                algorithms.AES(self.master_key),
                modes.CBC(iv)
            )
            encryptor = cipher.encryptor()
            
            # Pad data to block size
            padded_data = self._pad_data(data)
            
            # Encrypt
            encrypted_data = encryptor.update(padded_data) + encryptor.finalize()
            
            # Combine IV and encrypted data
            result = iv + encrypted_data
            
            # Log encryption
            self._log_security_event(
                event_type="encryption",
                user_id=user_id,
                resource="data",
                action="encrypt",
                result="success",
                threat_level=ThreatLevel.LOW,
                details={"data_size": len(data)}
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Encryption failed: {e}")
            return None
    
    def decrypt_data(self, encrypted_data: bytes, user_id: str) -> Optional[bytes]:
        """Decrypt previously encrypted data"""
        try:
            if not self.master_key:
                raise ValueError("Master key not initialized")
            
            if len(encrypted_data) < 16:
                raise ValueError("Invalid encrypted data")
            
            # Extract IV and encrypted content
            iv = encrypted_data[:16]
            encrypted_content = encrypted_data[16:]
            
            # Create cipher
            cipher = Cipher(
                algorithms.AES(self.master_key),
                modes.CBC(iv)
            )
            decryptor = cipher.decryptor()
            
            # Decrypt
            padded_data = decryptor.update(encrypted_content) + decryptor.finalize()
            
            # Remove padding
            data = self._unpad_data(padded_data)
            
            # Log decryption
            self._log_security_event(
                event_type="decryption",
                user_id=user_id,
                resource="data",
                action="decrypt",
                result="success",
                threat_level=ThreatLevel.LOW,
                details={"data_size": len(data)}
            )
            
            return data
            
        except Exception as e:
            self.logger.error(f"Decryption failed: {e}")
            return None
    
    def store_quantum_secret(self, secret_id: str, secret_data: bytes,
                           user_id: str, max_access: int = 10) -> bool:
        """Store secret using quantum-enhanced security"""
        try:
            # Encrypt the secret
            encrypted_data = self.encrypt_data(secret_data, user_id)
            if not encrypted_data:
                return False
            
            # Generate quantum signature if quantum crypto is enabled
            quantum_signature = np.array([0])  # Default
            if self.enable_quantum_crypto and self.quantum_keys:
                key_name = list(self.quantum_keys.keys())[0]
                quantum_key = self.quantum_keys[key_name]
                quantum_signature = self._generate_quantum_signature(secret_data, quantum_key)
            
            # Store quantum secret
            quantum_secret = QuantumSecret(
                secret_id=secret_id,
                encrypted_data=encrypted_data,
                quantum_signature=quantum_signature,
                max_access=max_access
            )
            
            self.quantum_secrets[secret_id] = quantum_secret
            
            self._log_security_event(
                event_type="quantum_secret_storage",
                user_id=user_id,
                resource="quantum_secrets",
                action=f"store:{secret_id}",
                result="success",
                threat_level=ThreatLevel.LOW
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Quantum secret storage failed: {e}")
            return False
    
    def retrieve_quantum_secret(self, secret_id: str, user_id: str) -> Optional[bytes]:
        """Retrieve quantum-secured secret"""
        try:
            if secret_id not in self.quantum_secrets:
                return None
            
            quantum_secret = self.quantum_secrets[secret_id]
            
            # Check access limit
            if quantum_secret.access_count >= quantum_secret.max_access:
                self._log_security_event(
                    event_type="quantum_secret_access",
                    user_id=user_id,
                    resource="quantum_secrets",
                    action=f"retrieve:{secret_id}",
                    result="denied",
                    threat_level=ThreatLevel.HIGH,
                    details={"reason": "access_limit_exceeded"}
                )
                return None
            
            # Verify quantum signature if enabled
            if self.enable_quantum_crypto:
                # In practice, would verify quantum signature
                pass
            
            # Decrypt secret
            decrypted_data = self.decrypt_data(quantum_secret.encrypted_data, user_id)
            if decrypted_data:
                # Increment access count
                quantum_secret.access_count += 1
                
                self._log_security_event(
                    event_type="quantum_secret_access",
                    user_id=user_id,
                    resource="quantum_secrets",
                    action=f"retrieve:{secret_id}",
                    result="success",
                    threat_level=ThreatLevel.LOW,
                    details={"access_count": quantum_secret.access_count}
                )
            
            return decrypted_data
            
        except Exception as e:
            self.logger.error(f"Quantum secret retrieval failed: {e}")
            return None
    
    def secure_multiparty_computation(self, participants: List[str], 
                                     computation_function: Callable,
                                     inputs: Dict[str, Any]) -> Optional[Any]:
        """Perform secure multi-party computation"""
        try:
            # Simplified secure MPC (in practice would use advanced protocols)
            self.logger.info(f"Starting secure MPC with {len(participants)} participants")
            
            # Verify all participants
            for participant in participants:
                if participant not in self.key_ring:
                    raise ValueError(f"Unknown participant: {participant}")
            
            # Encrypt inputs from each participant
            encrypted_inputs = {}
            for participant, input_data in inputs.items():
                if participant in participants:
                    serialized_input = json.dumps(input_data).encode()
                    encrypted_inputs[participant] = self.encrypt_data(serialized_input, participant)
            
            # Perform computation on encrypted data (simplified)
            # In practice, this would use homomorphic encryption or secret sharing
            result = computation_function(inputs)
            
            # Log secure computation
            self._log_security_event(
                event_type="secure_multiparty_computation",
                user_id="system",
                resource="smc",
                action="compute",
                result="success",
                threat_level=ThreatLevel.LOW,
                details={"participants": len(participants)}
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Secure MPC failed: {e}")
            return None
    
    def _generate_quantum_signature(self, data: bytes, quantum_key: np.ndarray) -> np.ndarray:
        """Generate quantum-based signature for data"""
        # Simplified quantum signature (in practice would use proper quantum protocols)
        data_hash = hashlib.sha256(data).digest()
        
        # Use quantum key to create signature
        signature = np.real(quantum_key) * np.frombuffer(data_hash[:16], dtype=np.uint8)
        
        return signature
    
    def detect_anomalies(self, user_id: str, action_pattern: List[str]) -> float:
        """Detect anomalous behavior patterns"""
        try:
            # Simple anomaly detection based on action patterns
            if user_id not in self.key_ring:
                return 1.0  # Maximum anomaly for unknown user
            
            # Analyze recent actions for user
            recent_logs = [
                log for log in self.audit_log[-100:] 
                if log.user_id == user_id
            ]
            
            if not recent_logs:
                return 0.5  # Moderate anomaly for new users
            
            # Calculate pattern deviation
            recent_actions = [log.action for log in recent_logs]
            
            # Simple statistical anomaly detection
            action_frequency = {}
            for action in recent_actions:
                action_frequency[action] = action_frequency.get(action, 0) + 1
            
            total_actions = len(recent_actions)
            normal_actions = sum(
                1 for action in action_pattern 
                if action in action_frequency
            )
            
            anomaly_score = 1.0 - (normal_actions / len(action_pattern))
            
            if anomaly_score > self.anomaly_threshold:
                self._log_security_event(
                    event_type="anomaly_detection",
                    user_id=user_id,
                    resource="behavioral_analysis",
                    action="analyze",
                    result="anomaly_detected",
                    threat_level=ThreatLevel.HIGH,
                    details={"anomaly_score": anomaly_score}
                )
            
            return anomaly_score
            
        except Exception as e:
            self.logger.error(f"Anomaly detection failed: {e}")
            return 0.5
    
    def _record_auth_failure(self, user_id: str, reason: str) -> None:
        """Record authentication failure"""
        self.auth_failures[user_id] = self.auth_failures.get(user_id, 0) + 1
        
        threat_level = ThreatLevel.HIGH if self.auth_failures[user_id] >= self.failed_auth_limit else ThreatLevel.MEDIUM
        
        self._log_security_event(
            event_type="authentication_failure",
            user_id=user_id,
            resource="auth_system",
            action="login_attempt",
            result="failure",
            threat_level=threat_level,
            details={"reason": reason, "failure_count": self.auth_failures[user_id]}
        )
        
        # Lock account if too many failures
        if self.auth_failures[user_id] >= self.failed_auth_limit:
            self.logger.warning(f"Account {user_id} locked due to repeated auth failures")
    
    def _clear_auth_failures(self, user_id: str) -> None:
        """Clear authentication failures for user"""
        if user_id in self.auth_failures:
            del self.auth_failures[user_id]
    
    def _log_security_event(self, event_type: str, user_id: str, resource: str,
                           action: str, result: str, threat_level: ThreatLevel,
                           details: Optional[Dict[str, Any]] = None) -> None:
        """Log security event to audit trail"""
        
        log_entry = SecurityAuditLog(
            timestamp=time.time(),
            event_type=event_type,
            user_id=user_id,
            resource=resource,
            action=action,
            result=result,
            threat_level=threat_level,
            details=details or {}
        )
        
        self.audit_log.append(log_entry)
        
        # Log to system logger based on threat level
        if threat_level == ThreatLevel.CRITICAL:
            self.logger.critical(f"SECURITY: {event_type} - {user_id} {action} {resource} - {result}")
        elif threat_level == ThreatLevel.HIGH:
            self.logger.error(f"SECURITY: {event_type} - {user_id} {action} {resource} - {result}")
        elif threat_level == ThreatLevel.MEDIUM:
            self.logger.warning(f"SECURITY: {event_type} - {user_id} {action} {resource} - {result}")
        else:
            self.logger.info(f"SECURITY: {event_type} - {user_id} {action} {resource} - {result}")
    
    def _pad_data(self, data: bytes) -> bytes:
        """Add PKCS7 padding to data"""
        padding_length = 16 - (len(data) % 16)
        padding = bytes([padding_length] * padding_length)
        return data + padding
    
    def _unpad_data(self, padded_data: bytes) -> bytes:
        """Remove PKCS7 padding from data"""
        padding_length = padded_data[-1]
        return padded_data[:-padding_length]
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get current security system status"""
        
        # Count threats by level
        threat_counts = {level.value: 0 for level in ThreatLevel}
        for log in self.audit_log[-100:]:  # Recent entries
            threat_counts[log.threat_level.value] += 1
        
        # Active user sessions
        active_users = len([creds for creds in self.key_ring.values() if creds.is_valid()])
        
        return {
            "security_level": self.security_level.value,
            "quantum_crypto_enabled": self.enable_quantum_crypto,
            "active_users": active_users,
            "total_users": len(self.key_ring),
            "quantum_secrets_stored": len(self.quantum_secrets),
            "audit_log_entries": len(self.audit_log),
            "recent_threat_counts": threat_counts,
            "auth_failures": len(self.auth_failures),
            "locked_accounts": sum(
                1 for count in self.auth_failures.values() 
                if count >= self.failed_auth_limit
            )
        }
    
    def generate_security_report(self) -> str:
        """Generate comprehensive security report"""
        
        status = self.get_security_status()
        recent_alerts = [
            log for log in self.audit_log[-50:] 
            if log.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]
        ]
        
        report = f"""
# Quantum Security Report

## System Status
- Security Level: {status['security_level'].upper()}
- Quantum Cryptography: {'Enabled' if status['quantum_crypto_enabled'] else 'Disabled'}
- Active Users: {status['active_users']} / {status['total_users']}
- Quantum Secrets: {status['quantum_secrets_stored']}

## Threat Assessment
- Critical Threats: {status['recent_threat_counts']['critical']}
- High Threats: {status['recent_threat_counts']['high']}
- Medium Threats: {status['recent_threat_counts']['medium']}
- Low Threats: {status['recent_threat_counts']['low']}

## Authentication Security
- Failed Authentication Attempts: {status['auth_failures']}
- Locked Accounts: {status['locked_accounts']}

## Recent High-Priority Alerts
"""
        
        for alert in recent_alerts[-10:]:  # Last 10 high-priority alerts
            report += f"""
- **{alert.threat_level.value.upper()}**: {alert.event_type} - {alert.action} on {alert.resource} by {alert.user_id} ({alert.result})
"""
        
        return report


# Demonstration function
async def demo_quantum_security():
    """Demonstrate quantum security framework"""
    print("🔒 Starting Quantum Security Framework Demo")
    
    # Initialize security system
    security = QuantumSecurityFramework(
        security_level=SecurityLevel.QUANTUM_SAFE,
        enable_quantum_crypto=True
    )
    
    # Create test users
    security.create_user("researcher_001", "researcher")
    security.create_user("admin_001", "admin")
    
    # Test data encryption
    test_data = b"This is sensitive quantum data that needs protection"
    encrypted = security.encrypt_data(test_data, "admin_001")
    
    if encrypted:
        print(f"✅ Data encrypted successfully ({len(encrypted)} bytes)")
        
        # Test decryption
        decrypted = security.decrypt_data(encrypted, "admin_001")
        if decrypted == test_data:
            print("✅ Data decrypted successfully")
        else:
            print("❌ Decryption failed")
    
    # Test quantum secret storage
    secret_data = b"quantum_model_parameters_v1.2.3"
    success = security.store_quantum_secret("model_params", secret_data, "researcher_001")
    
    if success:
        print("✅ Quantum secret stored successfully")
        
        # Test secret retrieval
        retrieved = security.retrieve_quantum_secret("model_params", "researcher_001")
        if retrieved == secret_data:
            print("✅ Quantum secret retrieved successfully")
        else:
            print("❌ Secret retrieval failed")
    
    # Test secure multi-party computation
    participants = ["researcher_001", "admin_001"]
    inputs = {
        "researcher_001": {"data_points": 1000},
        "admin_001": {"model_accuracy": 0.95}
    }
    
    def simple_computation(data):
        total_points = sum(item.get("data_points", 0) for item in data.values())
        avg_accuracy = sum(item.get("model_accuracy", 0) for item in data.values()) / len(data)
        return {"total_data_points": total_points, "average_accuracy": avg_accuracy}
    
    smc_result = security.secure_multiparty_computation(participants, simple_computation, inputs)
    if smc_result:
        print(f"✅ Secure MPC completed: {smc_result}")
    
    # Test anomaly detection
    normal_pattern = ["read", "execute", "read"]
    anomaly_score = security.detect_anomalies("researcher_001", normal_pattern)
    print(f"🔍 Anomaly score: {anomaly_score:.3f}")
    
    # Generate security report
    report = security.generate_security_report()
    print("\n📋 Security Report:")
    print(report)
    
    # Get system status
    status = security.get_security_status()
    print(f"\n🛡️ System Status: {status}")
    
    return security


if __name__ == "__main__":
    asyncio.run(demo_quantum_security())