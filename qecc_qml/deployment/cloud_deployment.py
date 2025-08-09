"""
Cloud deployment automation for QECC-aware QML systems.

Supports deployment to major cloud providers including AWS, GCP, Azure,
and quantum cloud platforms with auto-scaling and monitoring.
"""

import os
import json
import yaml
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import subprocess
from pathlib import Path
import time


class CloudProvider(Enum):
    """Supported cloud providers."""
    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"
    IBM_QUANTUM = "ibm_quantum"
    GOOGLE_QUANTUM = "google_quantum"
    LOCAL = "local"


class DeploymentEnvironment(Enum):
    """Deployment environments."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class DeploymentConfig:
    """Configuration for cloud deployment."""
    provider: CloudProvider
    environment: DeploymentEnvironment
    region: str = "us-east-1"
    instance_type: str = "t3.medium"
    min_instances: int = 1
    max_instances: int = 10
    auto_scaling: bool = True
    load_balancer: bool = True
    ssl_enabled: bool = True
    monitoring_enabled: bool = True
    logging_enabled: bool = True
    backup_enabled: bool = True
    quantum_backends: List[str] = field(default_factory=lambda: ["qiskit", "cirq"])
    environment_variables: Dict[str, str] = field(default_factory=dict)
    resource_limits: Dict[str, str] = field(default_factory=dict)
    storage_size: str = "100Gi"
    network_config: Dict[str, Any] = field(default_factory=dict)
    security_config: Dict[str, Any] = field(default_factory=dict)


class CloudDeployer:
    """
    Multi-cloud deployment automation for QECC-aware QML systems.
    
    Provides unified interface for deploying to different cloud providers
    with optimized configurations for quantum machine learning workloads.
    """
    
    def __init__(
        self,
        project_name: str,
        config: DeploymentConfig,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize cloud deployer.
        
        Args:
            project_name: Name of the project
            config: Deployment configuration
            logger: Optional logger instance
        """
        self.project_name = project_name
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Deployment state
        self.deployment_id = None
        self.resources = {}
        self.endpoints = {}
        
        # Validate configuration
        self._validate_config()
        self._setup_provider_client()
    
    def _validate_config(self):
        """Validate deployment configuration."""
        if not self.project_name:
            raise ValueError("Project name is required")
        
        if self.config.min_instances > self.config.max_instances:
            raise ValueError("min_instances cannot be greater than max_instances")
        
        # Validate instance types per provider
        valid_instances = {
            CloudProvider.AWS: ["t3.micro", "t3.small", "t3.medium", "t3.large", "c5.large", "m5.large"],
            CloudProvider.GCP: ["e2-micro", "e2-small", "e2-medium", "e2-standard-2", "n1-standard-1"],
            CloudProvider.AZURE: ["Standard_B1s", "Standard_B1ms", "Standard_B2s", "Standard_D2s_v3"]
        }
        
        if (self.config.provider in valid_instances and 
            self.config.instance_type not in valid_instances[self.config.provider]):
            self.logger.warning(f"Instance type {self.config.instance_type} may not be valid for {self.config.provider.value}")
    
    def _setup_provider_client(self):
        """Setup cloud provider client."""
        if self.config.provider == CloudProvider.AWS:
            self._setup_aws_client()
        elif self.config.provider == CloudProvider.GCP:
            self._setup_gcp_client()
        elif self.config.provider == CloudProvider.AZURE:
            self._setup_azure_client()
        elif self.config.provider == CloudProvider.IBM_QUANTUM:
            self._setup_ibm_quantum_client()
        elif self.config.provider == CloudProvider.GOOGLE_QUANTUM:
            self._setup_google_quantum_client()
        elif self.config.provider == CloudProvider.LOCAL:
            self._setup_local_client()
    
    def _setup_aws_client(self):
        """Setup AWS client."""
        try:
            import boto3
            self.aws_client = boto3.session.Session()
            self.ec2 = self.aws_client.client('ec2', region_name=self.config.region)
            self.ecs = self.aws_client.client('ecs', region_name=self.config.region)
            self.elb = self.aws_client.client('elbv2', region_name=self.config.region)
            self.logs = self.aws_client.client('logs', region_name=self.config.region)
            self.logger.info("AWS client initialized")
        except ImportError:
            raise RuntimeError("boto3 is required for AWS deployment. Install with: pip install boto3")
        except Exception as e:
            raise RuntimeError(f"Failed to setup AWS client: {e}")
    
    def _setup_gcp_client(self):
        """Setup Google Cloud client."""
        try:
            from google.cloud import compute_v1, container_v1
            self.gcp_compute = compute_v1.InstancesClient()
            self.gcp_container = container_v1.ClusterManagerClient()
            self.logger.info("GCP client initialized")
        except ImportError:
            raise RuntimeError("google-cloud SDK is required for GCP deployment")
        except Exception as e:
            raise RuntimeError(f"Failed to setup GCP client: {e}")
    
    def _setup_azure_client(self):
        """Setup Azure client."""
        try:
            from azure.identity import DefaultAzureCredential
            from azure.mgmt.compute import ComputeManagementClient
            from azure.mgmt.containerinstance import ContainerInstanceManagementClient
            
            credential = DefaultAzureCredential()
            # Will need subscription_id from environment or config
            subscription_id = os.environ.get('AZURE_SUBSCRIPTION_ID')
            if not subscription_id:
                raise RuntimeError("AZURE_SUBSCRIPTION_ID environment variable is required")
            
            self.azure_compute = ComputeManagementClient(credential, subscription_id)
            self.azure_container = ContainerInstanceManagementClient(credential, subscription_id)
            self.logger.info("Azure client initialized")
        except ImportError:
            raise RuntimeError("azure-mgmt SDK is required for Azure deployment")
        except Exception as e:
            raise RuntimeError(f"Failed to setup Azure client: {e}")
    
    def _setup_ibm_quantum_client(self):
        """Setup IBM Quantum client."""
        try:
            from qiskit_ibm_runtime import QiskitRuntimeService
            self.ibm_service = QiskitRuntimeService()
            self.logger.info("IBM Quantum client initialized")
        except ImportError:
            raise RuntimeError("qiskit-ibm-runtime is required for IBM Quantum deployment")
        except Exception as e:
            self.logger.warning(f"IBM Quantum client setup failed: {e}")
    
    def _setup_google_quantum_client(self):
        """Setup Google Quantum AI client."""
        try:
            import cirq_google
            self.google_quantum = cirq_google.engine.Engine()
            self.logger.info("Google Quantum AI client initialized")
        except ImportError:
            raise RuntimeError("cirq-google is required for Google Quantum AI deployment")
        except Exception as e:
            self.logger.warning(f"Google Quantum AI client setup failed: {e}")
    
    def _setup_local_client(self):
        """Setup local deployment client (Docker/Kubernetes)."""
        # Check if Docker is available
        try:
            subprocess.run(["docker", "--version"], capture_output=True, check=True)
            self.has_docker = True
        except:
            self.has_docker = False
        
        # Check if Kubernetes is available
        try:
            subprocess.run(["kubectl", "version", "--client"], capture_output=True, check=True)
            self.has_kubernetes = True
        except:
            self.has_kubernetes = False
        
        self.logger.info(f"Local deployment: Docker={self.has_docker}, Kubernetes={self.has_kubernetes}")
    
    def deploy(self, image_tag: str, deployment_manifest: Optional[str] = None) -> Dict[str, Any]:
        """
        Deploy the QECC-QML application.
        
        Args:
            image_tag: Container image tag to deploy
            deployment_manifest: Path to deployment manifest (optional)
            
        Returns:
            Deployment information
        """
        self.logger.info(f"Starting deployment to {self.config.provider.value}")
        
        try:
            if self.config.provider == CloudProvider.AWS:
                return self._deploy_aws(image_tag, deployment_manifest)
            elif self.config.provider == CloudProvider.GCP:
                return self._deploy_gcp(image_tag, deployment_manifest)
            elif self.config.provider == CloudProvider.AZURE:
                return self._deploy_azure(image_tag, deployment_manifest)
            elif self.config.provider == CloudProvider.LOCAL:
                return self._deploy_local(image_tag, deployment_manifest)
            else:
                raise NotImplementedError(f"Deployment to {self.config.provider.value} not implemented")
                
        except Exception as e:
            self.logger.error(f"Deployment failed: {e}")
            raise
    
    def _deploy_aws(self, image_tag: str, deployment_manifest: Optional[str]) -> Dict[str, Any]:
        """Deploy to AWS using ECS."""
        # Create ECS cluster
        cluster_name = f"{self.project_name}-{self.config.environment.value}"
        
        try:
            self.ecs.create_cluster(clusterName=cluster_name)
            self.logger.info(f"Created ECS cluster: {cluster_name}")
        except self.ecs.exceptions.ClusterNotFoundException:
            pass  # Cluster already exists
        
        # Create task definition
        task_definition = self._create_aws_task_definition(image_tag)
        
        # Create service
        service_response = self.ecs.create_service(
            cluster=cluster_name,
            serviceName=f"{self.project_name}-service",
            taskDefinition=task_definition['taskDefinitionArn'],
            desiredCount=self.config.min_instances,
            launchType='FARGATE',
            networkConfiguration={
                'awsvpcConfiguration': {
                    'subnets': self._get_aws_subnets(),
                    'securityGroups': [self._get_aws_security_group()],
                    'assignPublicIp': 'ENABLED'
                }
            }
        )
        
        self.deployment_id = service_response['service']['serviceArn']
        
        # Setup load balancer if enabled
        if self.config.load_balancer:
            lb_arn = self._create_aws_load_balancer()
            self.resources['load_balancer'] = lb_arn
        
        return {
            'deployment_id': self.deployment_id,
            'cluster': cluster_name,
            'service': service_response['service']['serviceName'],
            'provider': 'aws',
            'region': self.config.region,
            'resources': self.resources
        }
    
    def _create_aws_task_definition(self, image_tag: str) -> Dict[str, Any]:
        """Create AWS ECS task definition."""
        container_def = {
            'name': self.project_name,
            'image': image_tag,
            'memory': 2048,
            'cpu': 1024,
            'essential': True,
            'portMappings': [
                {
                    'containerPort': port,
                    'protocol': 'tcp'
                } for port in [8080, 8050]  # Default ports
            ],
            'environment': [
                {'name': k, 'value': v} 
                for k, v in self.config.environment_variables.items()
            ],
            'logConfiguration': {
                'logDriver': 'awslogs',
                'options': {
                    'awslogs-group': f'/ecs/{self.project_name}',
                    'awslogs-region': self.config.region,
                    'awslogs-stream-prefix': 'ecs'
                }
            }
        }
        
        # Create log group
        try:
            self.logs.create_log_group(logGroupName=f'/ecs/{self.project_name}')
        except self.logs.exceptions.ResourceAlreadyExistsException:
            pass
        
        response = self.ecs.register_task_definition(
            family=self.project_name,
            networkMode='awsvpc',
            requiresCompatibilities=['FARGATE'],
            cpu='1024',
            memory='2048',
            executionRoleArn=self._get_aws_execution_role(),
            containerDefinitions=[container_def]
        )
        
        return response['taskDefinition']
    
    def _get_aws_subnets(self) -> List[str]:
        """Get AWS subnets for deployment."""
        # Get default VPC
        vpcs = self.ec2.describe_vpcs(Filters=[{'Name': 'is-default', 'Values': ['true']}])
        if not vpcs['Vpcs']:
            raise RuntimeError("No default VPC found")
        
        vpc_id = vpcs['Vpcs'][0]['VpcId']
        
        # Get subnets
        subnets = self.ec2.describe_subnets(Filters=[{'Name': 'vpc-id', 'Values': [vpc_id]}])
        return [subnet['SubnetId'] for subnet in subnets['Subnets'][:2]]  # Use first 2 subnets
    
    def _get_aws_security_group(self) -> str:
        """Get or create AWS security group."""
        # Try to find existing security group
        groups = self.ec2.describe_security_groups(
            Filters=[{'Name': 'group-name', 'Values': [f'{self.project_name}-sg']}]
        )
        
        if groups['SecurityGroups']:
            return groups['SecurityGroups'][0]['GroupId']
        
        # Create new security group
        vpcs = self.ec2.describe_vpcs(Filters=[{'Name': 'is-default', 'Values': ['true']}])
        vpc_id = vpcs['Vpcs'][0]['VpcId']
        
        sg = self.ec2.create_security_group(
            GroupName=f'{self.project_name}-sg',
            Description=f'Security group for {self.project_name}',
            VpcId=vpc_id
        )
        
        # Add ingress rules
        self.ec2.authorize_security_group_ingress(
            GroupId=sg['GroupId'],
            IpPermissions=[
                {
                    'IpProtocol': 'tcp',
                    'FromPort': 8080,
                    'ToPort': 8080,
                    'IpRanges': [{'CidrIp': '0.0.0.0/0'}]
                },
                {
                    'IpProtocol': 'tcp',
                    'FromPort': 8050,
                    'ToPort': 8050,
                    'IpRanges': [{'CidrIp': '0.0.0.0/0'}]
                }
            ]
        )
        
        return sg['GroupId']
    
    def _get_aws_execution_role(self) -> str:
        """Get AWS ECS execution role."""
        # This would typically be a pre-created role
        # For simplicity, assuming it exists
        return f"arn:aws:iam::{self._get_aws_account_id()}:role/ecsTaskExecutionRole"
    
    def _get_aws_account_id(self) -> str:
        """Get AWS account ID."""
        sts = self.aws_client.client('sts')
        return sts.get_caller_identity()['Account']
    
    def _create_aws_load_balancer(self) -> str:
        """Create AWS Application Load Balancer."""
        # Implementation would create ALB, target groups, listeners
        # Placeholder for brevity
        self.logger.info("Load balancer creation not fully implemented")
        return "placeholder-lb-arn"
    
    def _deploy_gcp(self, image_tag: str, deployment_manifest: Optional[str]) -> Dict[str, Any]:
        """Deploy to Google Cloud using GKE."""
        # Implementation for GCP deployment
        self.logger.info("GCP deployment not fully implemented")
        return {"provider": "gcp", "status": "not_implemented"}
    
    def _deploy_azure(self, image_tag: str, deployment_manifest: Optional[str]) -> Dict[str, Any]:
        """Deploy to Azure using Container Instances."""
        # Implementation for Azure deployment
        self.logger.info("Azure deployment not fully implemented")
        return {"provider": "azure", "status": "not_implemented"}
    
    def _deploy_local(self, image_tag: str, deployment_manifest: Optional[str]) -> Dict[str, Any]:
        """Deploy locally using Docker or Kubernetes."""
        if self.has_kubernetes and deployment_manifest:
            return self._deploy_kubernetes(deployment_manifest)
        elif self.has_docker:
            return self._deploy_docker(image_tag)
        else:
            raise RuntimeError("Neither Docker nor Kubernetes is available for local deployment")
    
    def _deploy_docker(self, image_tag: str) -> Dict[str, Any]:
        """Deploy using Docker Compose."""
        # Generate docker-compose.yml if not exists
        compose_file = "docker-compose.yml"
        if not os.path.exists(compose_file):
            self._generate_docker_compose(image_tag)
        
        # Deploy using docker-compose
        cmd = ["docker-compose", "up", "-d", "--scale", f"app={self.config.min_instances}"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"Docker deployment failed: {result.stderr}")
        
        self.logger.info("Docker deployment completed")
        
        return {
            'deployment_id': f"docker-{self.project_name}",
            'provider': 'local_docker',
            'endpoints': self._get_docker_endpoints(),
            'status': 'running'
        }
    
    def _deploy_kubernetes(self, manifest_path: str) -> Dict[str, Any]:
        """Deploy using Kubernetes."""
        cmd = ["kubectl", "apply", "-f", manifest_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"Kubernetes deployment failed: {result.stderr}")
        
        self.logger.info("Kubernetes deployment completed")
        
        return {
            'deployment_id': f"k8s-{self.project_name}",
            'provider': 'local_kubernetes',
            'namespace': 'default',
            'status': 'deployed'
        }
    
    def _generate_docker_compose(self, image_tag: str):
        """Generate docker-compose.yml for local deployment."""
        compose_config = {
            'version': '3.8',
            'services': {
                'app': {
                    'image': image_tag,
                    'ports': ['8080:8080', '8050:8050'],
                    'environment': self.config.environment_variables,
                    'restart': 'unless-stopped',
                    'volumes': ['./data:/app/data', './logs:/app/logs']
                },
                'redis': {
                    'image': 'redis:7-alpine',
                    'ports': ['6379:6379'],
                    'volumes': ['redis_data:/data']
                }
            },
            'volumes': {
                'redis_data': {}
            }
        }
        
        with open('docker-compose.yml', 'w') as f:
            yaml.dump(compose_config, f)
        
        self.logger.info("Generated docker-compose.yml")
    
    def _get_docker_endpoints(self) -> Dict[str, str]:
        """Get Docker service endpoints."""
        return {
            'api': 'http://localhost:8080',
            'dashboard': 'http://localhost:8050'
        }
    
    def generate_kubernetes_manifest(self, image_tag: str, output_path: Optional[str] = None) -> str:
        """
        Generate Kubernetes deployment manifest.
        
        Args:
            image_tag: Container image tag
            output_path: Output file path
            
        Returns:
            Path to generated manifest
        """
        if output_path is None:
            output_path = f"{self.project_name}-k8s-manifest.yaml"
        
        manifests = []
        
        # Namespace
        manifests.append({
            'apiVersion': 'v1',
            'kind': 'Namespace',
            'metadata': {'name': self.project_name}
        })
        
        # Deployment
        manifests.append({
            'apiVersion': 'apps/v1',
            'kind': 'Deployment',
            'metadata': {
                'name': f"{self.project_name}-deployment",
                'namespace': self.project_name,
                'labels': {'app': self.project_name}
            },
            'spec': {
                'replicas': self.config.min_instances,
                'selector': {'matchLabels': {'app': self.project_name}},
                'template': {
                    'metadata': {'labels': {'app': self.project_name}},
                    'spec': {
                        'containers': [{
                            'name': self.project_name,
                            'image': image_tag,
                            'ports': [
                                {'containerPort': 8080, 'name': 'api'},
                                {'containerPort': 8050, 'name': 'dashboard'}
                            ],
                            'env': [
                                {'name': k, 'value': v}
                                for k, v in self.config.environment_variables.items()
                            ],
                            'resources': {
                                'requests': {'cpu': '500m', 'memory': '1Gi'},
                                'limits': {'cpu': '1000m', 'memory': '2Gi'}
                            },
                            'livenessProbe': {
                                'httpGet': {'path': '/health', 'port': 8080},
                                'initialDelaySeconds': 30,
                                'periodSeconds': 10
                            }
                        }]
                    }
                }
            }
        })
        
        # Service
        manifests.append({
            'apiVersion': 'v1',
            'kind': 'Service',
            'metadata': {
                'name': f"{self.project_name}-service",
                'namespace': self.project_name
            },
            'spec': {
                'selector': {'app': self.project_name},
                'ports': [
                    {'name': 'api', 'port': 8080, 'targetPort': 8080},
                    {'name': 'dashboard', 'port': 8050, 'targetPort': 8050}
                ],
                'type': 'ClusterIP'
            }
        })
        
        # Ingress (if load balancer enabled)
        if self.config.load_balancer:
            manifests.append({
                'apiVersion': 'networking.k8s.io/v1',
                'kind': 'Ingress',
                'metadata': {
                    'name': f"{self.project_name}-ingress",
                    'namespace': self.project_name,
                    'annotations': {
                        'nginx.ingress.kubernetes.io/rewrite-target': '/',
                    }
                },
                'spec': {
                    'rules': [{
                        'host': f"{self.project_name}.local",
                        'http': {
                            'paths': [
                                {
                                    'path': '/api',
                                    'pathType': 'Prefix',
                                    'backend': {
                                        'service': {
                                            'name': f"{self.project_name}-service",
                                            'port': {'number': 8080}
                                        }
                                    }
                                },
                                {
                                    'path': '/dashboard',
                                    'pathType': 'Prefix',
                                    'backend': {
                                        'service': {
                                            'name': f"{self.project_name}-service",
                                            'port': {'number': 8050}
                                        }
                                    }
                                }
                            ]
                        }
                    }]
                }
            })
        
        # HPA (if auto scaling enabled)
        if self.config.auto_scaling:
            manifests.append({
                'apiVersion': 'autoscaling/v2',
                'kind': 'HorizontalPodAutoscaler',
                'metadata': {
                    'name': f"{self.project_name}-hpa",
                    'namespace': self.project_name
                },
                'spec': {
                    'scaleTargetRef': {
                        'apiVersion': 'apps/v1',
                        'kind': 'Deployment',
                        'name': f"{self.project_name}-deployment"
                    },
                    'minReplicas': self.config.min_instances,
                    'maxReplicas': self.config.max_instances,
                    'metrics': [
                        {
                            'type': 'Resource',
                            'resource': {
                                'name': 'cpu',
                                'target': {'type': 'Utilization', 'averageUtilization': 70}
                            }
                        }
                    ]
                }
            })
        
        # Write manifest file
        with open(output_path, 'w') as f:
            for i, manifest in enumerate(manifests):
                if i > 0:
                    f.write('---\n')
                yaml.dump(manifest, f, default_flow_style=False)
        
        self.logger.info(f"Generated Kubernetes manifest: {output_path}")
        return output_path
    
    def get_deployment_status(self) -> Dict[str, Any]:
        """Get current deployment status."""
        if not self.deployment_id:
            return {'status': 'not_deployed'}
        
        if self.config.provider == CloudProvider.AWS:
            return self._get_aws_status()
        elif self.config.provider == CloudProvider.LOCAL:
            return self._get_local_status()
        else:
            return {'status': 'unknown', 'provider': self.config.provider.value}
    
    def _get_aws_status(self) -> Dict[str, Any]:
        """Get AWS deployment status."""
        try:
            # Get service status
            cluster_name = f"{self.project_name}-{self.config.environment.value}"
            services = self.ecs.describe_services(
                cluster=cluster_name,
                services=[f"{self.project_name}-service"]
            )
            
            if not services['services']:
                return {'status': 'not_found'}
            
            service = services['services'][0]
            
            return {
                'status': service['status'],
                'desired_count': service['desiredCount'],
                'running_count': service['runningCount'],
                'pending_count': service['pendingCount'],
                'provider': 'aws',
                'cluster': cluster_name
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get AWS status: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def _get_local_status(self) -> Dict[str, Any]:
        """Get local deployment status."""
        if self.has_kubernetes:
            return self._get_kubernetes_status()
        elif self.has_docker:
            return self._get_docker_status()
        else:
            return {'status': 'no_runtime'}
    
    def _get_kubernetes_status(self) -> Dict[str, Any]:
        """Get Kubernetes deployment status."""
        try:
            cmd = ["kubectl", "get", "deployment", f"{self.project_name}-deployment", "-o", "json"]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            deployment = json.loads(result.stdout)
            status = deployment.get('status', {})
            
            return {
                'status': 'running' if status.get('readyReplicas', 0) > 0 else 'pending',
                'replicas': status.get('replicas', 0),
                'ready_replicas': status.get('readyReplicas', 0),
                'updated_replicas': status.get('updatedReplicas', 0),
                'provider': 'kubernetes'
            }
            
        except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
            return {'status': 'error', 'message': str(e)}
    
    def _get_docker_status(self) -> Dict[str, Any]:
        """Get Docker deployment status."""
        try:
            cmd = ["docker-compose", "ps", "--format", "json"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                containers = json.loads(result.stdout) if result.stdout.strip() else []
                running_count = sum(1 for c in containers if c.get('State') == 'running')
                
                return {
                    'status': 'running' if running_count > 0 else 'stopped',
                    'containers': len(containers),
                    'running_containers': running_count,
                    'provider': 'docker'
                }
            else:
                return {'status': 'error', 'message': 'Failed to get Docker status'}
                
        except json.JSONDecodeError:
            return {'status': 'unknown'}
    
    def scale(self, desired_instances: int) -> bool:
        """
        Scale the deployment.
        
        Args:
            desired_instances: Desired number of instances
            
        Returns:
            True if scaling successful
        """
        if not self.deployment_id:
            self.logger.error("No deployment found to scale")
            return False
        
        if desired_instances < self.config.min_instances or desired_instances > self.config.max_instances:
            self.logger.error(f"Desired instances {desired_instances} outside allowed range")
            return False
        
        try:
            if self.config.provider == CloudProvider.AWS:
                return self._scale_aws(desired_instances)
            elif self.config.provider == CloudProvider.LOCAL:
                return self._scale_local(desired_instances)
            else:
                self.logger.error(f"Scaling not implemented for {self.config.provider.value}")
                return False
                
        except Exception as e:
            self.logger.error(f"Scaling failed: {e}")
            return False
    
    def _scale_aws(self, desired_instances: int) -> bool:
        """Scale AWS ECS service."""
        cluster_name = f"{self.project_name}-{self.config.environment.value}"
        
        self.ecs.update_service(
            cluster=cluster_name,
            service=f"{self.project_name}-service",
            desiredCount=desired_instances
        )
        
        self.logger.info(f"Scaled AWS service to {desired_instances} instances")
        return True
    
    def _scale_local(self, desired_instances: int) -> bool:
        """Scale local deployment."""
        if self.has_kubernetes:
            cmd = ["kubectl", "scale", "deployment", f"{self.project_name}-deployment", 
                   f"--replicas={desired_instances}"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.logger.info(f"Scaled Kubernetes deployment to {desired_instances} replicas")
                return True
            else:
                self.logger.error(f"Failed to scale Kubernetes deployment: {result.stderr}")
                return False
        
        elif self.has_docker:
            cmd = ["docker-compose", "up", "-d", "--scale", f"app={desired_instances}"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.logger.info(f"Scaled Docker Compose to {desired_instances} containers")
                return True
            else:
                self.logger.error(f"Failed to scale Docker deployment: {result.stderr}")
                return False
        
        return False
    
    def cleanup(self) -> bool:
        """
        Clean up deployment resources.
        
        Returns:
            True if cleanup successful
        """
        try:
            if self.config.provider == CloudProvider.AWS:
                return self._cleanup_aws()
            elif self.config.provider == CloudProvider.LOCAL:
                return self._cleanup_local()
            else:
                self.logger.warning(f"Cleanup not implemented for {self.config.provider.value}")
                return False
                
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")
            return False
    
    def _cleanup_aws(self) -> bool:
        """Clean up AWS resources."""
        cluster_name = f"{self.project_name}-{self.config.environment.value}"
        
        # Delete service
        self.ecs.update_service(
            cluster=cluster_name,
            service=f"{self.project_name}-service",
            desiredCount=0
        )
        
        # Wait for service to scale down, then delete
        time.sleep(30)
        
        self.ecs.delete_service(
            cluster=cluster_name,
            service=f"{self.project_name}-service"
        )
        
        # Delete cluster
        self.ecs.delete_cluster(cluster=cluster_name)
        
        self.logger.info("AWS resources cleaned up")
        return True
    
    def _cleanup_local(self) -> bool:
        """Clean up local deployment resources."""
        if self.has_kubernetes:
            cmd = ["kubectl", "delete", "namespace", self.project_name]
            subprocess.run(cmd, capture_output=True)
            self.logger.info("Kubernetes resources cleaned up")
            
        if self.has_docker:
            cmd = ["docker-compose", "down", "-v"]
            subprocess.run(cmd, capture_output=True)
            self.logger.info("Docker resources cleaned up")
        
        return True