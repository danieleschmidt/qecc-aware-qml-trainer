"""
Production deployment and orchestration system for QECC-QML.
"""

import os
import json
import time
import threading
import subprocess
import shutil
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import socket
import requests
from pathlib import Path

from ..utils.logging_config import get_logger
from ..monitoring.health_monitor import HealthMonitor
from ..utils.error_recovery import ErrorRecoveryManager
from ..optimization.adaptive_scaling import AdaptiveScaler

logger = get_logger(__name__)


class DeploymentStatus(Enum):
    """Deployment status states."""
    PENDING = "pending"
    DEPLOYING = "deploying"
    RUNNING = "running"
    SCALING = "scaling"
    UPDATING = "updating"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass
class DeploymentConfig:
    """Production deployment configuration."""
    name: str
    version: str
    image: str = "qecc-qml:latest"
    replicas: int = 3
    min_replicas: int = 1
    max_replicas: int = 10
    cpu_request: str = "500m"
    cpu_limit: str = "2"
    memory_request: str = "1Gi"
    memory_limit: str = "4Gi"
    gpu_required: bool = False
    gpu_count: int = 0
    environment: Dict[str, str] = None
    secrets: Dict[str, str] = None
    health_check_path: str = "/health"
    readiness_check_path: str = "/ready"
    port: int = 8000
    load_balancer: bool = True
    auto_scaling: bool = True
    monitoring: bool = True
    logging_level: str = "INFO"
    
    def __post_init__(self):
        if self.environment is None:
            self.environment = {}
        if self.secrets is None:
            self.secrets = {}


class ProductionDeployment:
    """
    Production deployment orchestrator for QECC-QML applications.
    """
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.status = DeploymentStatus.PENDING
        self.deployment_id = self._generate_deployment_id()
        
        # Component managers
        self.health_monitor = HealthMonitor()
        self.error_recovery = ErrorRecoveryManager()
        self.adaptive_scaler = AdaptiveScaler(
            min_workers=config.min_replicas,
            max_workers=config.max_replicas
        )
        
        # State tracking
        self.start_time = None
        self.deployment_metrics = {}
        self.active_replicas = {}
        self.failed_replicas = {}
        
        # Deployment callbacks
        self.status_callbacks = []
        
    def deploy(self, wait_for_ready: bool = True, timeout: int = 300) -> bool:
        """
        Deploy the application to production.
        
        Args:
            wait_for_ready: Wait for deployment to be ready
            timeout: Timeout in seconds
            
        Returns:
            True if deployment successful, False otherwise
        """
        logger.info(f"Starting deployment: {self.config.name} v{self.config.version}")
        
        self.status = DeploymentStatus.DEPLOYING
        self.start_time = time.time()
        self._notify_status_change()
        
        try:
            # Validate deployment configuration
            if not self._validate_deployment_config():
                raise DeploymentError("Invalid deployment configuration")
                
            # Setup deployment environment
            self._setup_deployment_environment()
            
            # Deploy application containers
            self._deploy_containers()
            
            # Setup load balancing
            if self.config.load_balancer:
                self._setup_load_balancer()
                
            # Start health monitoring
            if self.config.monitoring:
                self._start_monitoring()
                
            # Start auto-scaling
            if self.config.auto_scaling:
                self._start_auto_scaling()
                
            # Wait for readiness
            if wait_for_ready:
                if not self._wait_for_ready(timeout):
                    raise DeploymentError("Deployment failed to become ready")
                    
            self.status = DeploymentStatus.RUNNING
            self._notify_status_change()
            
            deployment_time = time.time() - self.start_time
            logger.info(f"Deployment successful in {deployment_time:.2f}s")
            
            return True
            
        except Exception as e:
            self.status = DeploymentStatus.FAILED
            self._notify_status_change()
            logger.error(f"Deployment failed: {str(e)}")
            
            # Cleanup failed deployment
            self._cleanup_failed_deployment()
            
            return False
            
    def scale(self, target_replicas: int) -> bool:
        """
        Scale the deployment to target number of replicas.
        """
        if self.status != DeploymentStatus.RUNNING:
            logger.warning("Cannot scale deployment that is not running")
            return False
            
        if target_replicas < self.config.min_replicas:
            target_replicas = self.config.min_replicas
        elif target_replicas > self.config.max_replicas:
            target_replicas = self.config.max_replicas
            
        logger.info(f"Scaling deployment from {self.config.replicas} to {target_replicas}")
        
        old_status = self.status
        self.status = DeploymentStatus.SCALING
        self._notify_status_change()
        
        try:
            if target_replicas > self.config.replicas:
                # Scale up
                self._scale_up(target_replicas - self.config.replicas)
            else:
                # Scale down
                self._scale_down(self.config.replicas - target_replicas)
                
            self.config.replicas = target_replicas
            self.status = old_status
            self._notify_status_change()
            
            logger.info(f"Scaling completed to {target_replicas} replicas")
            return True
            
        except Exception as e:
            self.status = old_status
            self._notify_status_change()
            logger.error(f"Scaling failed: {str(e)}")
            return False
            
    def update(self, new_config: DeploymentConfig, 
              strategy: str = "rolling") -> bool:
        """
        Update the deployment with new configuration.
        
        Args:
            new_config: New deployment configuration
            strategy: Update strategy ('rolling', 'blue_green', 'canary')
        """
        logger.info(f"Updating deployment with strategy: {strategy}")
        
        old_status = self.status
        self.status = DeploymentStatus.UPDATING
        self._notify_status_change()
        
        try:
            if strategy == "rolling":
                success = self._rolling_update(new_config)
            elif strategy == "blue_green":
                success = self._blue_green_update(new_config)
            elif strategy == "canary":
                success = self._canary_update(new_config)
            else:
                raise DeploymentError(f"Unknown update strategy: {strategy}")
                
            if success:
                self.config = new_config
                self.status = old_status
                logger.info("Deployment update successful")
            else:
                self.status = old_status
                logger.error("Deployment update failed")
                
            self._notify_status_change()
            return success
            
        except Exception as e:
            self.status = old_status
            self._notify_status_change()
            logger.error(f"Deployment update failed: {str(e)}")
            return False
            
    def stop(self, graceful: bool = True, timeout: int = 30) -> bool:
        """
        Stop the deployment.
        
        Args:
            graceful: Perform graceful shutdown
            timeout: Timeout for graceful shutdown
        """
        logger.info("Stopping deployment")
        
        self.status = DeploymentStatus.STOPPING
        self._notify_status_change()
        
        try:
            # Stop auto-scaling
            self.adaptive_scaler.stop_adaptive_scaling()
            
            # Stop monitoring
            self.health_monitor.stop_monitoring()
            
            # Stop application containers
            self._stop_containers(graceful, timeout)
            
            # Cleanup resources
            self._cleanup_deployment_resources()
            
            self.status = DeploymentStatus.STOPPED
            self._notify_status_change()
            
            logger.info("Deployment stopped successfully")
            return True
            
        except Exception as e:
            self.status = DeploymentStatus.FAILED
            self._notify_status_change()
            logger.error(f"Failed to stop deployment: {str(e)}")
            return False
            
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive deployment status."""
        return {
            "deployment_id": self.deployment_id,
            "name": self.config.name,
            "version": self.config.version,
            "status": self.status.value,
            "uptime": time.time() - self.start_time if self.start_time else 0,
            "replicas": {
                "desired": self.config.replicas,
                "running": len(self.active_replicas),
                "failed": len(self.failed_replicas)
            },
            "health": self._get_health_status(),
            "metrics": self.deployment_metrics,
            "auto_scaling_enabled": self.config.auto_scaling,
            "load_balancer_enabled": self.config.load_balancer
        }
        
    def _validate_deployment_config(self) -> bool:
        """Validate deployment configuration."""
        
        # Check required fields
        if not self.config.name or not self.config.version:
            logger.error("Deployment name and version are required")
            return False
            
        # Check resource limits
        if self.config.replicas < 1:
            logger.error("At least 1 replica is required")
            return False
            
        if self.config.min_replicas > self.config.max_replicas:
            logger.error("min_replicas cannot be greater than max_replicas")
            return False
            
        # Check port availability
        if not self._check_port_available(self.config.port):
            logger.warning(f"Port {self.config.port} may not be available")
            
        return True
        
    def _setup_deployment_environment(self):
        """Setup deployment environment."""
        
        # Create deployment directory
        deployment_dir = Path(f"/tmp/qecc-qml-deployment-{self.deployment_id}")
        deployment_dir.mkdir(exist_ok=True)
        
        # Generate configuration files
        self._generate_config_files(deployment_dir)
        
        # Setup secrets
        self._setup_secrets(deployment_dir)
        
        logger.info(f"Deployment environment setup at {deployment_dir}")
        
    def _deploy_containers(self):
        """Deploy application containers."""
        
        logger.info(f"Deploying {self.config.replicas} replicas")
        
        for i in range(self.config.replicas):
            replica_id = f"{self.config.name}-{i}"
            
            try:
                self._deploy_single_container(replica_id)
                self.active_replicas[replica_id] = {
                    "id": replica_id,
                    "status": "running",
                    "start_time": time.time()
                }
            except Exception as e:
                logger.error(f"Failed to deploy replica {replica_id}: {str(e)}")
                self.failed_replicas[replica_id] = {
                    "id": replica_id,
                    "error": str(e),
                    "timestamp": time.time()
                }
                
    def _deploy_single_container(self, replica_id: str):
        """Deploy a single container replica."""
        
        # This would integrate with container orchestration platforms
        # For now, simulate container deployment
        
        logger.info(f"Deploying container: {replica_id}")
        
        # Simulate deployment time
        time.sleep(1)
        
        # Check if deployment would succeed
        if replica_id.endswith("-fail"):  # Simulate failure for testing
            raise DeploymentError(f"Simulated failure for {replica_id}")
            
        logger.info(f"Container {replica_id} deployed successfully")
        
    def _setup_load_balancer(self):
        """Setup load balancer for the deployment."""
        
        logger.info("Setting up load balancer")
        
        # This would integrate with load balancing solutions
        # For now, log the configuration
        
        lb_config = {
            "service_name": self.config.name,
            "port": self.config.port,
            "replicas": list(self.active_replicas.keys()),
            "health_check": self.config.health_check_path
        }
        
        logger.info(f"Load balancer config: {lb_config}")
        
    def _start_monitoring(self):
        """Start deployment monitoring."""
        
        # Register custom alert callback
        def deployment_alert_callback(alert_data):
            logger.warning(f"Deployment alert: {alert_data}")
            
        self.health_monitor.register_alert_callback(deployment_alert_callback)
        self.health_monitor.start_monitoring()
        
        logger.info("Deployment monitoring started")
        
    def _start_auto_scaling(self):
        """Start auto-scaling for the deployment."""
        
        self.adaptive_scaler.start_adaptive_scaling()
        logger.info("Auto-scaling started")
        
    def _wait_for_ready(self, timeout: int) -> bool:
        """Wait for deployment to be ready."""
        
        logger.info(f"Waiting for deployment readiness (timeout: {timeout}s)")
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            ready_count = 0
            
            for replica_id in self.active_replicas:
                if self._check_replica_ready(replica_id):
                    ready_count += 1
                    
            if ready_count >= self.config.replicas:
                logger.info("Deployment is ready")
                return True
                
            time.sleep(2)  # Check every 2 seconds
            
        logger.error("Deployment readiness timeout")
        return False
        
    def _check_replica_ready(self, replica_id: str) -> bool:
        """Check if a replica is ready."""
        
        # This would perform actual health checks
        # For now, simulate readiness check
        
        try:
            # Simulate HTTP health check
            # response = requests.get(f"http://{replica_id}:{self.config.port}{self.config.readiness_check_path}")
            # return response.status_code == 200
            
            # For simulation, assume replica is ready after 5 seconds
            replica_info = self.active_replicas.get(replica_id)
            if replica_info and time.time() - replica_info["start_time"] > 5:
                return True
                
        except Exception as e:
            logger.debug(f"Readiness check failed for {replica_id}: {str(e)}")
            
        return False
        
    def _scale_up(self, additional_replicas: int):
        """Scale up by adding replicas."""
        
        for i in range(additional_replicas):
            replica_id = f"{self.config.name}-{len(self.active_replicas) + i}"
            
            try:
                self._deploy_single_container(replica_id)
                self.active_replicas[replica_id] = {
                    "id": replica_id,
                    "status": "running",
                    "start_time": time.time()
                }
                logger.info(f"Scaled up: added replica {replica_id}")
            except Exception as e:
                logger.error(f"Failed to scale up replica {replica_id}: {str(e)}")
                
    def _scale_down(self, replicas_to_remove: int):
        """Scale down by removing replicas."""
        
        replicas_list = list(self.active_replicas.keys())
        
        for i in range(min(replicas_to_remove, len(replicas_list))):
            replica_id = replicas_list[i]
            
            try:
                self._stop_single_container(replica_id)
                del self.active_replicas[replica_id]
                logger.info(f"Scaled down: removed replica {replica_id}")
            except Exception as e:
                logger.error(f"Failed to scale down replica {replica_id}: {str(e)}")
                
    def _rolling_update(self, new_config: DeploymentConfig) -> bool:
        """Perform rolling update."""
        
        logger.info("Performing rolling update")
        
        # Update replicas one by one
        for replica_id in list(self.active_replicas.keys()):
            try:
                # Stop old replica
                self._stop_single_container(replica_id)
                
                # Deploy new replica
                self._deploy_single_container(replica_id)
                
                # Wait for readiness
                time.sleep(5)
                
                if not self._check_replica_ready(replica_id):
                    logger.error(f"New replica {replica_id} not ready")
                    return False
                    
            except Exception as e:
                logger.error(f"Rolling update failed for {replica_id}: {str(e)}")
                return False
                
        return True
        
    def _blue_green_update(self, new_config: DeploymentConfig) -> bool:
        """Perform blue-green update."""
        
        logger.info("Performing blue-green update")
        
        # Deploy green environment
        green_replicas = {}
        
        for i in range(new_config.replicas):
            replica_id = f"{new_config.name}-green-{i}"
            
            try:
                self._deploy_single_container(replica_id)
                green_replicas[replica_id] = {
                    "id": replica_id,
                    "status": "running", 
                    "start_time": time.time()
                }
            except Exception as e:
                logger.error(f"Failed to deploy green replica {replica_id}: {str(e)}")
                # Cleanup green environment
                for green_id in green_replicas:
                    self._stop_single_container(green_id)
                return False
                
        # Wait for green environment readiness
        time.sleep(10)
        
        # Switch traffic to green
        logger.info("Switching traffic to green environment")
        
        # Stop blue environment
        for replica_id in list(self.active_replicas.keys()):
            self._stop_single_container(replica_id)
            
        # Update active replicas
        self.active_replicas = green_replicas
        
        return True
        
    def _canary_update(self, new_config: DeploymentConfig) -> bool:
        """Perform canary update."""
        
        logger.info("Performing canary update")
        
        # Deploy canary replica
        canary_id = f"{new_config.name}-canary"
        
        try:
            self._deploy_single_container(canary_id)
            
            # Monitor canary for health
            time.sleep(30)  # Monitor for 30 seconds
            
            if self._check_replica_ready(canary_id):
                # Canary is healthy, proceed with full update
                self._stop_single_container(canary_id)
                return self._rolling_update(new_config)
            else:
                # Canary failed, rollback
                self._stop_single_container(canary_id)
                logger.error("Canary update failed - rolling back")
                return False
                
        except Exception as e:
            logger.error(f"Canary update failed: {str(e)}")
            return False
            
    def _stop_containers(self, graceful: bool, timeout: int):
        """Stop all containers."""
        
        for replica_id in list(self.active_replicas.keys()):
            try:
                self._stop_single_container(replica_id, graceful, timeout)
            except Exception as e:
                logger.error(f"Failed to stop container {replica_id}: {str(e)}")
                
        self.active_replicas.clear()
        
    def _stop_single_container(self, replica_id: str, 
                              graceful: bool = True, timeout: int = 30):
        """Stop a single container."""
        
        logger.info(f"Stopping container: {replica_id}")
        
        if graceful:
            # Send graceful shutdown signal
            logger.debug(f"Sending graceful shutdown to {replica_id}")
            time.sleep(2)  # Simulate graceful shutdown time
            
        # Force stop if needed
        logger.info(f"Container {replica_id} stopped")
        
    def _cleanup_failed_deployment(self):
        """Cleanup resources from failed deployment."""
        
        logger.info("Cleaning up failed deployment")
        
        # Stop any running containers
        for replica_id in list(self.active_replicas.keys()):
            try:
                self._stop_single_container(replica_id, graceful=False)
            except:
                pass
                
        self.active_replicas.clear()
        self.failed_replicas.clear()
        
    def _cleanup_deployment_resources(self):
        """Cleanup all deployment resources."""
        
        logger.info("Cleaning up deployment resources")
        
        # Remove deployment directory
        deployment_dir = Path(f"/tmp/qecc-qml-deployment-{self.deployment_id}")
        if deployment_dir.exists():
            shutil.rmtree(deployment_dir)
            
    def _generate_config_files(self, deployment_dir: Path):
        """Generate configuration files for deployment."""
        
        config_file = deployment_dir / "config.json"
        
        with open(config_file, 'w') as f:
            json.dump(asdict(self.config), f, indent=2)
            
    def _setup_secrets(self, deployment_dir: Path):
        """Setup secrets for deployment."""
        
        if self.config.secrets:
            secrets_dir = deployment_dir / "secrets"
            secrets_dir.mkdir(exist_ok=True)
            
            for key, value in self.config.secrets.items():
                secret_file = secrets_dir / key
                with open(secret_file, 'w') as f:
                    f.write(value)
                    
    def _get_health_status(self) -> Dict[str, Any]:
        """Get health status of deployment."""
        
        return {
            "healthy_replicas": len([r for r in self.active_replicas.values() 
                                   if r["status"] == "running"]),
            "unhealthy_replicas": len(self.failed_replicas),
            "health_checks_enabled": self.config.monitoring,
            "last_health_check": time.time()
        }
        
    def _check_port_available(self, port: int) -> bool:
        """Check if port is available."""
        
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return True
        except OSError:
            return False
            
    def _generate_deployment_id(self) -> str:
        """Generate unique deployment ID."""
        return f"{self.config.name}-{int(time.time())}"
        
    def _notify_status_change(self):
        """Notify registered callbacks of status change."""
        for callback in self.status_callbacks:
            try:
                callback(self.status, self.get_status())
            except Exception as e:
                logger.error(f"Status callback failed: {str(e)}")
                
    def register_status_callback(self, callback: Callable):
        """Register callback for status changes."""
        self.status_callbacks.append(callback)


class DeploymentError(Exception):
    """Deployment-specific exception."""
    pass


class DeploymentManager:
    """
    Manages multiple deployments across different environments.
    """
    
    def __init__(self):
        self.deployments: Dict[str, ProductionDeployment] = {}
        
    def create_deployment(self, config: DeploymentConfig) -> ProductionDeployment:
        """Create a new deployment."""
        
        deployment = ProductionDeployment(config)
        self.deployments[deployment.deployment_id] = deployment
        
        logger.info(f"Created deployment: {deployment.deployment_id}")
        
        return deployment
        
    def get_deployment(self, deployment_id: str) -> Optional[ProductionDeployment]:
        """Get deployment by ID."""
        return self.deployments.get(deployment_id)
        
    def list_deployments(self) -> List[Dict[str, Any]]:
        """List all deployments."""
        return [deployment.get_status() for deployment in self.deployments.values()]
        
    def cleanup_stopped_deployments(self):
        """Remove stopped deployments from tracking."""
        
        stopped_deployments = [
            dep_id for dep_id, deployment in self.deployments.items()
            if deployment.status == DeploymentStatus.STOPPED
        ]
        
        for dep_id in stopped_deployments:
            del self.deployments[dep_id]
            
        logger.info(f"Cleaned up {len(stopped_deployments)} stopped deployments")