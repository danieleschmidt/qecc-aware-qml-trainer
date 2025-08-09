"""
Container-based deployment for QECC-aware QML systems.

Provides automated Docker container generation, multi-stage builds,
and optimized container configurations for production deployment.
"""

import os
import json
import shutil
import tempfile
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
import logging
import subprocess


@dataclass
class ContainerConfig:
    """Configuration for container builds."""
    base_image: str = "python:3.11-slim"
    working_dir: str = "/app"
    user_uid: int = 1000
    user_gid: int = 1000
    expose_ports: List[int] = field(default_factory=lambda: [8080, 8050])
    environment_vars: Dict[str, str] = field(default_factory=dict)
    volumes: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    gpu_support: bool = False
    quantum_backends: List[str] = field(default_factory=lambda: ["qiskit", "cirq"])
    optimization_level: str = "production"  # development, testing, production
    security_scanning: bool = True
    multi_arch: bool = True  # Build for multiple architectures


class DockerBuilder:
    """
    Automated Docker container builder for QECC-QML applications.
    
    Generates optimized Docker images with proper dependency management,
    security configuration, and multi-stage builds.
    """
    
    def __init__(
        self,
        project_root: str,
        config: Optional[ContainerConfig] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize Docker builder.
        
        Args:
            project_root: Root directory of the project
            config: Container configuration
            logger: Optional logger instance
        """
        self.project_root = Path(project_root)
        self.config = config or ContainerConfig()
        self.logger = logger or logging.getLogger(__name__)
        
        # Build context
        self.build_context = None
        self.dockerfile_path = None
        
        # Validation
        self._validate_environment()
    
    def _validate_environment(self):
        """Validate Docker environment."""
        try:
            result = subprocess.run(
                ["docker", "--version"],
                capture_output=True,
                text=True,
                check=True
            )
            self.logger.info(f"Docker version: {result.stdout.strip()}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError("Docker is not installed or not accessible")
        
        # Check if project root exists
        if not self.project_root.exists():
            raise ValueError(f"Project root does not exist: {self.project_root}")
    
    def generate_dockerfile(self, output_path: Optional[str] = None) -> str:
        """
        Generate optimized Dockerfile.
        
        Args:
            output_path: Path to save Dockerfile (defaults to project root)
            
        Returns:
            Path to generated Dockerfile
        """
        if output_path is None:
            output_path = self.project_root / "Dockerfile"
        else:
            output_path = Path(output_path)
        
        dockerfile_content = self._build_dockerfile_content()
        
        with open(output_path, 'w') as f:
            f.write(dockerfile_content)
        
        self.dockerfile_path = str(output_path)
        self.logger.info(f"Generated Dockerfile: {output_path}")
        
        return str(output_path)
    
    def _build_dockerfile_content(self) -> str:
        """Build the Dockerfile content."""
        lines = []
        
        # Multi-stage build for optimization
        if self.config.optimization_level == "production":
            lines.extend(self._build_multistage_dockerfile())
        else:
            lines.extend(self._build_simple_dockerfile())
        
        return "\n".join(lines)
    
    def _build_multistage_dockerfile(self) -> List[str]:
        """Build multi-stage Dockerfile for production."""
        lines = []
        
        # Stage 1: Dependencies and build tools
        lines.extend([
            "# Multi-stage build for QECC-Aware QML",
            f"FROM {self.config.base_image} AS builder",
            "",
            "# Install build dependencies",
            "RUN apt-get update && apt-get install -y \\",
            "    gcc g++ \\",
            "    cmake \\",
            "    libopenblas-dev \\",
            "    libffi-dev \\",
            "    && rm -rf /var/lib/apt/lists/*",
            "",
            "# Create virtual environment",
            "RUN python -m venv /opt/venv",
            "ENV PATH=\"/opt/venv/bin:$PATH\"",
            "",
            "# Copy requirements and install Python dependencies",
            "COPY requirements.txt .",
            "RUN pip install --no-cache-dir --upgrade pip setuptools wheel",
            "RUN pip install --no-cache-dir -r requirements.txt",
            ""
        ])
        
        # Install quantum backends
        for backend in self.config.quantum_backends:
            if backend == "qiskit":
                lines.append("RUN pip install --no-cache-dir qiskit qiskit-aer qiskit-ibm-runtime")
            elif backend == "cirq":
                lines.append("RUN pip install --no-cache-dir cirq google-cloud-quantum-engine")
            elif backend == "braket":
                lines.append("RUN pip install --no-cache-dir amazon-braket-sdk")
        
        lines.extend([
            "",
            "# Copy application code and install",
            "COPY . /app/src",
            "WORKDIR /app/src",
            "RUN pip install --no-cache-dir -e .",
            ""
        ])
        
        # Stage 2: Runtime image
        lines.extend([
            "# Stage 2: Runtime image",
            f"FROM {self.config.base_image} AS runtime",
            "",
            "# Install runtime dependencies only",
            "RUN apt-get update && apt-get install -y \\",
            "    libopenblas0 \\",
            "    && rm -rf /var/lib/apt/lists/*",
            "",
            "# Create application user",
            f"RUN groupadd -g {self.config.user_gid} appuser && \\",
            f"    useradd -m -u {self.config.user_uid} -g appuser appuser",
            "",
            "# Copy virtual environment from builder",
            "COPY --from=builder /opt/venv /opt/venv",
            "ENV PATH=\"/opt/venv/bin:$PATH\"",
            "",
            "# Copy application code",
            f"COPY --from=builder /app/src {self.config.working_dir}",
            f"WORKDIR {self.config.working_dir}",
            "",
            "# Set ownership",
            f"RUN chown -R appuser:appuser {self.config.working_dir}",
            ""
        ])
        
        # Add GPU support if needed
        if self.config.gpu_support:
            lines.extend([
                "# GPU support",
                "RUN pip install --no-cache-dir cupy-cuda12x torch[cuda]",
                ""
            ])
        
        # Environment variables
        if self.config.environment_vars:
            lines.append("# Environment variables")
            for key, value in self.config.environment_vars.items():
                lines.append(f"ENV {key}={value}")
            lines.append("")
        
        # Expose ports
        if self.config.expose_ports:
            lines.extend([
                "# Expose ports",
                f"EXPOSE {' '.join(map(str, self.config.expose_ports))}",
                ""
            ])
        
        # Volumes
        if self.config.volumes:
            lines.extend([
                "# Volumes",
                f"VOLUME {json.dumps(self.config.volumes)}",
                ""
            ])
        
        # Health check
        lines.extend([
            "# Health check",
            "HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \\",
            "  CMD python -c \"import qecc_qml; print('OK')\" || exit 1",
            "",
            "# Switch to application user",
            "USER appuser",
            "",
            "# Default command",
            "CMD [\"python\", \"-m\", \"qecc_qml.cli\", \"--help\"]"
        ])
        
        return lines
    
    def _build_simple_dockerfile(self) -> List[str]:
        """Build simple Dockerfile for development."""
        lines = []
        
        lines.extend([
            f"FROM {self.config.base_image}",
            "",
            "# Install system dependencies",
            "RUN apt-get update && apt-get install -y \\",
            "    gcc g++ \\",
            "    cmake \\",
            "    libopenblas-dev \\",
            "    git \\",
            "    && rm -rf /var/lib/apt/lists/*",
            "",
            f"WORKDIR {self.config.working_dir}",
            "",
            "# Copy requirements and install dependencies",
            "COPY requirements.txt .",
            "RUN pip install --no-cache-dir --upgrade pip",
            "RUN pip install --no-cache-dir -r requirements.txt",
            ""
        ])
        
        # Install quantum backends
        for backend in self.config.quantum_backends:
            if backend == "qiskit":
                lines.append("RUN pip install --no-cache-dir qiskit qiskit-aer")
            elif backend == "cirq":
                lines.append("RUN pip install --no-cache-dir cirq")
        
        lines.extend([
            "",
            "# Copy application code",
            "COPY . .",
            "",
            "# Install application",
            "RUN pip install --no-cache-dir -e .",
            ""
        ])
        
        # Environment variables
        if self.config.environment_vars:
            for key, value in self.config.environment_vars.items():
                lines.append(f"ENV {key}={value}")
        
        # Expose ports
        if self.config.expose_ports:
            lines.append(f"EXPOSE {' '.join(map(str, self.config.expose_ports))}")
        
        lines.extend([
            "",
            "CMD [\"python\", \"-m\", \"qecc_qml.cli\"]"
        ])
        
        return lines
    
    def generate_dockerignore(self, output_path: Optional[str] = None) -> str:
        """
        Generate .dockerignore file.
        
        Args:
            output_path: Path to save .dockerignore (defaults to project root)
            
        Returns:
            Path to generated .dockerignore file
        """
        if output_path is None:
            output_path = self.project_root / ".dockerignore"
        else:
            output_path = Path(output_path)
        
        ignore_patterns = [
            "# Python",
            "__pycache__/",
            "*.pyc",
            "*.pyo",
            "*.pyd",
            ".Python",
            "build/",
            "develop-eggs/",
            "dist/",
            "downloads/",
            "eggs/",
            ".eggs/",
            "lib/",
            "lib64/",
            "parts/",
            "sdist/",
            "var/",
            "wheels/",
            "*.egg-info/",
            ".installed.cfg",
            "*.egg",
            "",
            "# Virtual environments",
            "venv/",
            "ENV/",
            "env/",
            ".venv/",
            "",
            "# IDEs",
            ".vscode/",
            ".idea/",
            "*.swp",
            "*.swo",
            "*~",
            "",
            "# OS",
            ".DS_Store",
            "Thumbs.db",
            "",
            "# Git",
            ".git/",
            ".gitignore",
            "",
            "# Documentation",
            "docs/_build/",
            "",
            "# Tests",
            ".pytest_cache/",
            ".coverage",
            "htmlcov/",
            "",
            "# Jupyter",
            ".ipynb_checkpoints/",
            "",
            "# Docker",
            "Dockerfile*",
            ".dockerignore",
            "",
            "# Kubernetes",
            "kubernetes/",
            "*.yaml",
            "*.yml",
            "",
            "# CI/CD",
            ".github/",
            ".gitlab-ci.yml",
            "",
            "# Logs",
            "*.log",
            "logs/",
            "",
            "# Temporary files",
            "tmp/",
            "temp/",
            "*.tmp",
            "",
            "# Large datasets (should be downloaded at runtime)",
            "data/",
            "datasets/",
            "*.h5",
            "*.hdf5",
            "*.npy",
            "*.npz"
        ]
        
        with open(output_path, 'w') as f:
            f.write("\n".join(ignore_patterns))
        
        self.logger.info(f"Generated .dockerignore: {output_path}")
        return str(output_path)
    
    def build_image(
        self,
        tag: str,
        context_path: Optional[str] = None,
        dockerfile_path: Optional[str] = None,
        build_args: Optional[Dict[str, str]] = None,
        no_cache: bool = False
    ) -> str:
        """
        Build Docker image.
        
        Args:
            tag: Image tag
            context_path: Build context path (defaults to project root)
            dockerfile_path: Path to Dockerfile (defaults to generated)
            build_args: Build arguments
            no_cache: Whether to disable cache
            
        Returns:
            Image ID
        """
        if context_path is None:
            context_path = str(self.project_root)
        
        if dockerfile_path is None:
            if self.dockerfile_path is None:
                self.generate_dockerfile()
            dockerfile_path = self.dockerfile_path
        
        # Build command
        cmd = ["docker", "build"]
        
        if no_cache:
            cmd.append("--no-cache")
        
        if build_args:
            for key, value in build_args.items():
                cmd.extend(["--build-arg", f"{key}={value}"])
        
        cmd.extend([
            "-t", tag,
            "-f", dockerfile_path,
            context_path
        ])
        
        # Multi-architecture build if enabled
        if self.config.multi_arch:
            cmd = ["docker", "buildx", "build", "--platform", "linux/amd64,linux/arm64"] + cmd[2:]
        
        self.logger.info(f"Building Docker image: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                cwd=context_path
            )
            
            self.logger.info("Docker image built successfully")
            
            # Extract image ID from output
            for line in result.stdout.split('\n'):
                if 'Successfully built' in line:
                    image_id = line.split()[-1]
                    return image_id
            
            return tag  # Return tag if image ID not found
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Docker build failed: {e}")
            self.logger.error(f"STDOUT: {e.stdout}")
            self.logger.error(f"STDERR: {e.stderr}")
            raise RuntimeError(f"Docker build failed: {e}")
    
    def push_image(self, tag: str, registry: Optional[str] = None) -> bool:
        """
        Push Docker image to registry.
        
        Args:
            tag: Image tag to push
            registry: Registry URL (optional)
            
        Returns:
            True if push successful
        """
        if registry:
            full_tag = f"{registry}/{tag}"
            
            # Tag for registry
            tag_cmd = ["docker", "tag", tag, full_tag]
            subprocess.run(tag_cmd, check=True)
            
            push_tag = full_tag
        else:
            push_tag = tag
        
        # Push image
        push_cmd = ["docker", "push", push_tag]
        
        try:
            subprocess.run(push_cmd, check=True)
            self.logger.info(f"Successfully pushed image: {push_tag}")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to push image: {e}")
            return False
    
    def scan_image(self, tag: str) -> Dict[str, Any]:
        """
        Scan image for security vulnerabilities.
        
        Args:
            tag: Image tag to scan
            
        Returns:
            Scan results
        """
        if not self.config.security_scanning:
            return {"status": "skipped"}
        
        # Try different security scanners
        scanners = [
            ("trivy", ["trivy", "image", "--format", "json", tag]),
            ("docker-scan", ["docker", "scan", tag]),
            ("grype", ["grype", tag, "-o", "json"])
        ]
        
        for scanner_name, cmd in scanners:
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=300
                )
                
                if scanner_name == "trivy" or scanner_name == "grype":
                    return {
                        "scanner": scanner_name,
                        "status": "completed",
                        "results": json.loads(result.stdout)
                    }
                else:
                    return {
                        "scanner": scanner_name,
                        "status": "completed",
                        "results": result.stdout
                    }
                    
            except (subprocess.CalledProcessError, json.JSONDecodeError, FileNotFoundError):
                continue
        
        self.logger.warning("No security scanner available")
        return {"status": "no_scanner"}
    
    def generate_docker_compose(self, services: Dict[str, Any], output_path: Optional[str] = None) -> str:
        """
        Generate docker-compose.yml file.
        
        Args:
            services: Service configurations
            output_path: Path to save compose file
            
        Returns:
            Path to generated compose file
        """
        if output_path is None:
            output_path = self.project_root / "docker-compose.yml"
        else:
            output_path = Path(output_path)
        
        compose_config = {
            "version": "3.8",
            "services": {},
            "volumes": {},
            "networks": {
                "qecc-network": {
                    "driver": "bridge"
                }
            }
        }
        
        # Default QECC-QML service
        compose_config["services"]["qecc-qml"] = {
            "build": {
                "context": ".",
                "dockerfile": "Dockerfile"
            },
            "ports": [f"{port}:{port}" for port in self.config.expose_ports],
            "environment": self.config.environment_vars,
            "volumes": [
                "./data:/app/data",
                "./results:/app/results"
            ],
            "networks": ["qecc-network"],
            "restart": "unless-stopped",
            "healthcheck": {
                "test": ["CMD", "python", "-c", "import qecc_qml; print('OK')"],
                "interval": "30s",
                "timeout": "10s",
                "retries": 3
            }
        }
        
        # Add monitoring dashboard service
        compose_config["services"]["dashboard"] = {
            "build": ".",
            "command": ["python", "-m", "qecc_qml.monitoring.dashboard"],
            "ports": ["8050:8050"],
            "environment": {
                "DASH_HOST": "0.0.0.0",
                "DASH_PORT": "8050"
            },
            "networks": ["qecc-network"],
            "depends_on": ["qecc-qml"]
        }
        
        # Add Redis for caching (if needed)
        compose_config["services"]["redis"] = {
            "image": "redis:7-alpine",
            "ports": ["6379:6379"],
            "networks": ["qecc-network"],
            "volumes": ["redis-data:/data"],
            "command": "redis-server --appendonly yes"
        }
        
        # Add volumes
        compose_config["volumes"] = {
            "redis-data": {},
            "qecc-data": {},
            "qecc-results": {}
        }
        
        # Merge with custom services
        for service_name, service_config in services.items():
            compose_config["services"][service_name] = service_config
        
        # Write compose file
        import yaml
        with open(output_path, 'w') as f:
            yaml.dump(compose_config, f, default_flow_style=False, indent=2)
        
        self.logger.info(f"Generated docker-compose.yml: {output_path}")
        return str(output_path)
    
    def create_build_context(self) -> str:
        """
        Create optimized build context.
        
        Returns:
            Path to build context directory
        """
        # Create temporary directory
        build_context = tempfile.mkdtemp(prefix="qecc_build_")
        build_context_path = Path(build_context)
        
        # Copy essential files
        essential_files = [
            "qecc_qml/",
            "requirements.txt",
            "setup.py",
            "pyproject.toml",
            "README.md",
            "LICENSE"
        ]
        
        for file_pattern in essential_files:
            source = self.project_root / file_pattern
            if source.exists():
                if source.is_dir():
                    shutil.copytree(source, build_context_path / file_pattern)
                else:
                    shutil.copy2(source, build_context_path / file_pattern)
        
        # Copy examples (optional)
        examples_dir = self.project_root / "examples"
        if examples_dir.exists():
            shutil.copytree(examples_dir, build_context_path / "examples")
        
        self.build_context = build_context
        self.logger.info(f"Created build context: {build_context}")
        
        return build_context
    
    def cleanup_build_context(self):
        """Clean up temporary build context."""
        if self.build_context and os.path.exists(self.build_context):
            shutil.rmtree(self.build_context)
            self.logger.info(f"Cleaned up build context: {self.build_context}")
            self.build_context = None
    
    def get_image_info(self, tag: str) -> Dict[str, Any]:
        """
        Get information about built image.
        
        Args:
            tag: Image tag
            
        Returns:
            Image information
        """
        try:
            # Get image details
            inspect_cmd = ["docker", "image", "inspect", tag]
            result = subprocess.run(inspect_cmd, capture_output=True, text=True, check=True)
            
            image_data = json.loads(result.stdout)[0]
            
            # Get image size
            size_mb = image_data.get("Size", 0) / (1024 * 1024)
            
            return {
                "id": image_data.get("Id", ""),
                "created": image_data.get("Created", ""),
                "size_mb": round(size_mb, 2),
                "architecture": image_data.get("Architecture", ""),
                "os": image_data.get("Os", ""),
                "layers": len(image_data.get("RootFS", {}).get("Layers", [])),
                "config": image_data.get("Config", {})
            }
            
        except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError) as e:
            self.logger.error(f"Failed to get image info: {e}")
            return {}
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup_build_context()