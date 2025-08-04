# Multi-stage Docker build for QECC-aware QML
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash qecc && \
    mkdir -p /app && \
    chown -R qecc:qecc /app

WORKDIR /app

# ================================
# Development stage
# ================================
FROM base as development

# Install development dependencies
RUN apt-get update && apt-get install -y \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install -r requirements.txt

# Install additional development tools
RUN pip install \
    jupyter \
    ipython \
    pytest \
    black \
    flake8 \
    mypy

# Copy source code
COPY --chown=qecc:qecc . .

# Install package in development mode
RUN pip install -e ".[dev]"

# Switch to non-root user
USER qecc

# Expose Jupyter port
EXPOSE 8888

# Default command for development
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

# ================================
# Production build stage
# ================================
FROM base as builder

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --user -r requirements.txt

# Copy source code
COPY . .

# Install package
RUN pip install --user .

# ================================
# Production runtime stage
# ================================
FROM python:3.11-slim as production

# Set environment variables for production
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/home/qecc/.local/bin:$PATH" \
    QECC_QML_LOG_LEVEL=INFO \
    QECC_QML_ENABLE_TELEMETRY=false

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash qecc

# Copy installed packages from builder
COPY --from=builder --chown=qecc:qecc /root/.local /home/qecc/.local

# Create application directory
RUN mkdir -p /app && chown -R qecc:qecc /app
WORKDIR /app

# Switch to non-root user
USER qecc

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD qecc-qml health --format json || exit 1

# Default command
CMD ["qecc-qml", "--help"]

# ================================
# Testing stage
# ================================
FROM development as testing

# Install test dependencies
RUN pip install \
    pytest-cov \
    pytest-xdist \
    pytest-benchmark

# Run tests
RUN python -m pytest tests/ -v --cov=qecc_qml --cov-report=html

# ================================
# Documentation stage
# ================================
FROM development as docs

# Install documentation dependencies
RUN pip install \
    sphinx \
    sphinx-rtd-theme \
    nbsphinx \
    pandoc

# Generate documentation
RUN cd docs && make html

# ================================
# Deployment stage with monitoring
# ================================
FROM production as monitoring

# Install monitoring tools
USER root
RUN pip install \
    prometheus-client \
    grafana-api

# Copy monitoring configuration
COPY docker/monitoring/ /etc/monitoring/

# Expose monitoring ports
EXPOSE 8000 9090

USER qecc

# Command with monitoring enabled
CMD ["qecc-qml", "train", "--enable-monitoring"]