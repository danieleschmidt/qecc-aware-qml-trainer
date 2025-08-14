#!/usr/bin/env python3
"""
Production deployment script for QECC-Aware QML Trainer.
Handles environment setup, health checks, and deployment validation.
"""

import sys
import os
import subprocess
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional


class ProductionDeployer:
    """
    Production deployment manager for QECC-aware QML system.
    """
    
    def __init__(self, config_file: str = None):
        """Initialize deployment manager."""
        self.config_file = config_file or "deployment_config.json"
        self.config = self._load_config()
        self.deployment_log = []
        
    def _load_config(self) -> Dict[str, Any]:
        """Load deployment configuration."""
        default_config = {
            "environment": "production",
            "python_version": "3.9+",
            "required_packages": [
                "numpy>=1.24.0",
                "qiskit>=1.0.0",
                "qiskit-aer>=0.13.0"
            ],
            "optional_packages": [
                "torch>=2.0.0",
                "scikit-learn>=1.3.0",
                "matplotlib>=3.7.0"
            ],
            "system_requirements": {
                "min_memory_gb": 4,
                "min_disk_gb": 10,
                "min_cpu_cores": 2
            },
            "security": {
                "enable_input_validation": True,
                "enable_resource_limits": True,
                "enable_logging": True
            },
            "deployment_targets": ["local", "docker", "kubernetes"],
            "health_check_timeout": 30
        }
        
        if Path(self.config_file).exists():
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                # Merge with defaults
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                return config
            except Exception as e:
                print(f"⚠️  Failed to load config file: {e}")
                print("Using default configuration.")
        
        return default_config
    
    def _log(self, message: str, level: str = "INFO"):
        """Log deployment message."""
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] {level}: {message}"
        print(log_entry)
        self.deployment_log.append(log_entry)
    
    def check_system_requirements(self) -> bool:
        """Check system requirements for deployment."""
        self._log("🔍 Checking system requirements...")
        
        requirements_met = True
        
        # Check Python version
        python_version = sys.version_info
        min_version = (3, 9)
        if python_version >= min_version:
            self._log(f"✅ Python version: {python_version.major}.{python_version.minor}")
        else:
            self._log(f"❌ Python version {python_version.major}.{python_version.minor} < required {min_version[0]}.{min_version[1]}", "ERROR")
            requirements_met = False
        
        # Check memory (if psutil available)
        try:
            import psutil
            memory_gb = psutil.virtual_memory().total / (1024**3)
            min_memory = self.config["system_requirements"]["min_memory_gb"]
            if memory_gb >= min_memory:
                self._log(f"✅ Memory: {memory_gb:.1f} GB")
            else:
                self._log(f"⚠️  Memory: {memory_gb:.1f} GB < required {min_memory} GB", "WARNING")
        except ImportError:
            self._log("⚠️  Cannot check memory requirements (psutil not available)", "WARNING")
        
        # Check disk space
        try:
            disk_usage = os.statvfs('.')
            free_gb = (disk_usage.f_frsize * disk_usage.f_bavail) / (1024**3)
            min_disk = self.config["system_requirements"]["min_disk_gb"]
            if free_gb >= min_disk:
                self._log(f"✅ Disk space: {free_gb:.1f} GB available")
            else:
                self._log(f"⚠️  Disk space: {free_gb:.1f} GB < required {min_disk} GB", "WARNING")
        except Exception:
            self._log("⚠️  Cannot check disk space", "WARNING")
        
        return requirements_met
    
    def check_dependencies(self) -> bool:
        """Check required dependencies."""
        self._log("📦 Checking dependencies...")
        
        all_available = True
        
        # Core dependencies
        core_deps = {
            'numpy': 'numpy',
            'qiskit': 'qiskit',
        }
        
        for name, import_name in core_deps.items():
            try:
                __import__(import_name)
                self._log(f"✅ {name}")
            except ImportError:
                self._log(f"❌ {name} (required)", "ERROR")
                all_available = False
        
        # Optional dependencies
        optional_deps = {
            'torch': 'torch',
            'sklearn': 'sklearn',
            'matplotlib': 'matplotlib',
            'pandas': 'pandas',
            'tqdm': 'tqdm'
        }
        
        optional_count = 0
        for name, import_name in optional_deps.items():
            try:
                __import__(import_name)
                self._log(f"✅ {name} (optional)")
                optional_count += 1
            except ImportError:
                self._log(f"⚠️  {name} (optional, not available)")
        
        self._log(f"📊 Optional dependencies: {optional_count}/{len(optional_deps)} available")
        
        return all_available
    
    def run_health_checks(self) -> bool:
        """Run comprehensive health checks."""
        self._log("🏥 Running health checks...")
        
        health_status = True
        
        # Test basic imports
        try:
            from qecc_qml.core.quantum_nn import QECCAwareQNN
            from qecc_qml.training.basic_trainer_clean import BasicQECCTrainer
            self._log("✅ Core modules import successfully")
        except ImportError as e:
            self._log(f"❌ Core module import failed: {e}", "ERROR")
            health_status = False
        
        # Test basic functionality
        try:
            qnn = QECCAwareQNN(num_qubits=2, num_layers=1)
            trainer = BasicQECCTrainer(qnn, verbose=False)
            
            # Quick test training
            import numpy as np
            X_test = np.random.randn(4, 2).astype(np.float32)
            X_test = X_test / np.linalg.norm(X_test, axis=1, keepdims=True)
            y_test = np.array([0, 1, 0, 1])
            
            trainer.fit(X_test, y_test, epochs=1, batch_size=2, validation_split=0.0)
            predictions = trainer.predict(X_test[:2])
            
            if len(predictions) == 2:
                self._log("✅ Basic functionality test passed")
            else:
                self._log("❌ Basic functionality test failed", "ERROR")
                health_status = False
                
        except Exception as e:
            self._log(f"❌ Basic functionality test failed: {e}", "ERROR")
            health_status = False
        
        return health_status
    
    def create_deployment_package(self) -> str:
        """Create deployment package."""
        self._log("📦 Creating deployment package...")
        
        package_dir = Path("deployment_package")
        package_dir.mkdir(exist_ok=True)
        
        # Copy essential files
        essential_files = [
            "qecc_qml/",
            "examples/",
            "requirements.txt",
            "setup.py",
            "README.md",
            "LICENSE"
        ]
        
        import shutil
        
        for file_path in essential_files:
            src = Path(file_path)
            if src.exists():
                if src.is_dir():
                    dst = package_dir / src.name
                    if dst.exists():
                        shutil.rmtree(dst)
                    shutil.copytree(src, dst)
                else:
                    shutil.copy2(src, package_dir / src.name)
                self._log(f"✅ Packaged {file_path}")
            else:
                self._log(f"⚠️  {file_path} not found, skipping")
        
        # Create deployment manifest
        manifest = {
            "package_version": "1.0.0",
            "created_at": time.strftime('%Y-%m-%d %H:%M:%S'),
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
            "deployment_config": self.config,
            "files_included": essential_files
        }
        
        with open(package_dir / "deployment_manifest.json", 'w') as f:
            json.dump(manifest, f, indent=2)
        
        self._log(f"✅ Deployment package created at {package_dir}")
        return str(package_dir)
    
    def generate_docker_config(self) -> str:
        """Generate Docker configuration."""
        dockerfile_content = f'''# QECC-Aware QML Trainer - Production Dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY qecc_qml/ ./qecc_qml/
COPY examples/ ./examples/
COPY setup.py .
COPY README.md .

# Install the package
RUN pip install -e .

# Create non-root user
RUN useradd -m qecc && chown -R qecc:qecc /app
USER qecc

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD python -c "from qecc_qml.core.quantum_nn import QECCAwareQNN; print('OK')"

# Default command
CMD ["python", "-c", "print('QECC-Aware QML Trainer ready')"]
'''
        
        docker_compose_content = f'''version: '3.8'

services:
  qecc-qml-trainer:
    build: .
    container_name: qecc-qml-trainer
    environment:
      - PYTHONPATH=/app
      - LOG_LEVEL=INFO
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "python", "-c", "from qecc_qml.core.quantum_nn import QECCAwareQNN; print('OK')"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 10s
'''
        
        # Write Docker files
        with open("Dockerfile", 'w') as f:
            f.write(dockerfile_content)
        
        with open("docker-compose.yml", 'w') as f:
            f.write(docker_compose_content)
        
        self._log("✅ Docker configuration generated")
        return "Docker configuration created"
    
    def generate_kubernetes_config(self) -> str:
        """Generate Kubernetes configuration."""
        k8s_config = f'''apiVersion: apps/v1
kind: Deployment
metadata:
  name: qecc-qml-trainer
  labels:
    app: qecc-qml-trainer
spec:
  replicas: 2
  selector:
    matchLabels:
      app: qecc-qml-trainer
  template:
    metadata:
      labels:
        app: qecc-qml-trainer
    spec:
      containers:
      - name: qecc-qml-trainer
        image: qecc-qml-trainer:latest
        ports:
        - containerPort: 8080
        env:
        - name: PYTHONPATH
          value: "/app"
        - name: LOG_LEVEL
          value: "INFO"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
        livenessProbe:
          exec:
            command:
            - python
            - -c
            - "from qecc_qml.core.quantum_nn import QECCAwareQNN; print('OK')"
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          exec:
            command:
            - python
            - -c
            - "from qecc_qml.core.quantum_nn import QECCAwareQNN; print('OK')"
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: qecc-qml-trainer-service
spec:
  selector:
    app: qecc-qml-trainer
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8080
  type: LoadBalancer
'''
        
        with open("k8s-deployment.yaml", 'w') as f:
            f.write(k8s_config)
        
        self._log("✅ Kubernetes configuration generated")
        return "Kubernetes configuration created"
    
    def deploy(self, target: str = "local") -> bool:
        """Deploy to specified target."""
        self._log(f"🚀 Starting deployment to {target}...")
        
        # Run pre-deployment checks
        if not self.check_system_requirements():
            self._log("❌ System requirements not met", "ERROR")
            return False
        
        if not self.check_dependencies():
            self._log("❌ Required dependencies not available", "ERROR")
            return False
        
        if not self.run_health_checks():
            self._log("❌ Health checks failed", "ERROR")
            return False
        
        # Deploy based on target
        if target == "local":
            self._deploy_local()
        elif target == "docker":
            self._deploy_docker()
        elif target == "kubernetes":
            self._deploy_kubernetes()
        else:
            self._log(f"❌ Unknown deployment target: {target}", "ERROR")
            return False
        
        # Post-deployment validation
        if self._validate_deployment(target):
            self._log("✅ Deployment completed successfully")
            return True
        else:
            self._log("❌ Deployment validation failed", "ERROR")
            return False
    
    def _deploy_local(self):
        """Deploy locally."""
        self._log("📍 Deploying locally...")
        
        # Create necessary directories
        dirs = ["logs", "data", "checkpoints", "cache"]
        for dir_name in dirs:
            Path(dir_name).mkdir(exist_ok=True)
            self._log(f"✅ Created directory: {dir_name}")
        
        # Install package in development mode
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], 
                         check=True, capture_output=True)
            self._log("✅ Package installed in development mode")
        except subprocess.CalledProcessError as e:
            self._log(f"⚠️  Package installation warning: {e}")
    
    def _deploy_docker(self):
        """Deploy using Docker."""
        self._log("🐳 Deploying with Docker...")
        
        # Generate Docker configuration
        self.generate_docker_config()
        
        # Build Docker image
        try:
            subprocess.run(["docker", "build", "-t", "qecc-qml-trainer", "."], 
                         check=True)
            self._log("✅ Docker image built successfully")
        except subprocess.CalledProcessError as e:
            self._log(f"❌ Docker build failed: {e}", "ERROR")
            raise
        except FileNotFoundError:
            self._log("❌ Docker not found. Please install Docker.", "ERROR")
            raise
    
    def _deploy_kubernetes(self):
        """Deploy to Kubernetes."""
        self._log("☸️  Deploying to Kubernetes...")
        
        # Generate Kubernetes configuration
        self.generate_kubernetes_config()
        
        try:
            # Apply Kubernetes configuration
            subprocess.run(["kubectl", "apply", "-f", "k8s-deployment.yaml"], 
                         check=True)
            self._log("✅ Kubernetes deployment applied")
        except subprocess.CalledProcessError as e:
            self._log(f"❌ Kubernetes deployment failed: {e}", "ERROR")
            raise
        except FileNotFoundError:
            self._log("❌ kubectl not found. Please install kubectl.", "ERROR")
            raise
    
    def _validate_deployment(self, target: str) -> bool:
        """Validate deployment success."""
        self._log(f"✅ Validating {target} deployment...")
        
        # Basic validation - try importing and creating objects
        try:
            from qecc_qml.core.quantum_nn import QECCAwareQNN
            qnn = QECCAwareQNN(num_qubits=2, num_layers=1)
            self._log("✅ Post-deployment validation passed")
            return True
        except Exception as e:
            self._log(f"❌ Post-deployment validation failed: {e}", "ERROR")
            return False
    
    def generate_deployment_report(self) -> str:
        """Generate comprehensive deployment report."""
        report = {
            "deployment_timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "configuration": self.config,
            "deployment_log": self.deployment_log,
            "system_info": {
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                "platform": sys.platform,
                "working_directory": os.getcwd()
            },
            "files_generated": [
                "Dockerfile",
                "docker-compose.yml", 
                "k8s-deployment.yaml",
                "deployment_manifest.json"
            ]
        }
        
        report_file = "deployment_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        self._log(f"📊 Deployment report saved to {report_file}")
        return report_file


def main():
    """Main deployment function."""
    print("🚀 QECC-Aware QML Trainer - Production Deployment")
    print("=" * 60)
    
    deployer = ProductionDeployer()
    
    # Get deployment target from command line or default to local
    target = sys.argv[1] if len(sys.argv) > 1 else "local"
    
    if target not in ["local", "docker", "kubernetes"]:
        print(f"❌ Invalid target: {target}")
        print("Valid targets: local, docker, kubernetes")
        sys.exit(1)
    
    # Execute deployment
    try:
        success = deployer.deploy(target)
        
        # Generate report
        report_file = deployer.generate_deployment_report()
        
        if success:
            print(f"\n✅ Deployment to {target} completed successfully!")
            print(f"📊 Report saved to {report_file}")
            
            if target == "local":
                print(f"\n🎯 Next steps:")
                print(f"   • Run: python examples/basic_training_example.py")
                print(f"   • Run: python examples/robust_training_example.py")
                print(f"   • Check logs in ./logs/ directory")
            elif target == "docker":
                print(f"\n🎯 Next steps:")
                print(f"   • Run: docker-compose up -d")
                print(f"   • Check: docker logs qecc-qml-trainer")
            elif target == "kubernetes":
                print(f"\n🎯 Next steps:")
                print(f"   • Check: kubectl get pods")
                print(f"   • Monitor: kubectl logs -l app=qecc-qml-trainer")
                
            sys.exit(0)
        else:
            print(f"\n❌ Deployment to {target} failed!")
            print(f"📊 Check {report_file} for details")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n💥 Deployment error: {e}")
        deployer.generate_deployment_report()
        sys.exit(1)


if __name__ == "__main__":
    main()