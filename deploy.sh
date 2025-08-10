#!/bin/bash
set -e

# QECC-QML Production Deployment Script
# Fully automated deployment to Kubernetes with comprehensive validation

echo "🚀 QECC-QML PRODUCTION DEPLOYMENT"
echo "=================================="

# Configuration
NAMESPACE=${NAMESPACE:-qecc-qml}
IMAGE_TAG=${IMAGE_TAG:-latest}
DOCKER_REGISTRY=${DOCKER_REGISTRY:-}
ENVIRONMENT=${ENVIRONMENT:-production}
DRY_RUN=${DRY_RUN:-false}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check required tools
    local tools=("docker" "kubectl" "helm")
    for tool in "${tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            log_error "$tool is not installed"
            exit 1
        fi
        log_success "$tool found"
    done
    
    # Check Kubernetes connectivity
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    log_success "Kubernetes cluster accessible"
    
    # Check Docker daemon
    if ! docker info &> /dev/null; then
        log_error "Docker daemon not running"
        exit 1
    fi
    log_success "Docker daemon running"
}

run_quality_gates() {
    log_info "Running quality gates..."
    
    if [ -f "run_quality_gates_simple.py" ]; then
        if python3 run_quality_gates_simple.py; then
            log_success "Quality gates passed"
        else
            log_error "Quality gates failed"
            exit 1
        fi
    else
        log_warning "Quality gates script not found, skipping"
    fi
}

build_docker_image() {
    log_info "Building Docker image..."
    
    local image_name="qecc-qml:${IMAGE_TAG}"
    if [ -n "$DOCKER_REGISTRY" ]; then
        image_name="${DOCKER_REGISTRY}/qecc-qml:${IMAGE_TAG}"
    fi
    
    if [ "$DRY_RUN" = "true" ]; then
        log_info "DRY RUN: Would build $image_name"
        return 0
    fi
    
    if docker build -f docker/Dockerfile.production -t "$image_name" .; then
        log_success "Docker image built: $image_name"
        
        # Push to registry if specified
        if [ -n "$DOCKER_REGISTRY" ]; then
            if docker push "$image_name"; then
                log_success "Docker image pushed to registry"
            else
                log_error "Failed to push Docker image"
                exit 1
            fi
        fi
    else
        log_error "Docker image build failed"
        exit 1
    fi
}

create_namespace() {
    log_info "Creating namespace: $NAMESPACE"
    
    if [ "$DRY_RUN" = "true" ]; then
        log_info "DRY RUN: Would create namespace $NAMESPACE"
        return 0
    fi
    
    if kubectl get namespace "$NAMESPACE" &> /dev/null; then
        log_warning "Namespace $NAMESPACE already exists"
    else
        kubectl create namespace "$NAMESPACE"
        log_success "Namespace $NAMESPACE created"
    fi
    
    # Label the namespace
    kubectl label namespace "$NAMESPACE" name="$NAMESPACE" --overwrite
}

deploy_to_kubernetes() {
    log_info "Deploying to Kubernetes..."
    
    if [ "$DRY_RUN" = "true" ]; then
        log_info "DRY RUN: Would deploy to Kubernetes"
        kubectl apply --dry-run=client -f kubernetes/production-deployment.yaml
        return 0
    fi
    
    # Apply Kubernetes manifests
    if kubectl apply -f kubernetes/production-deployment.yaml; then
        log_success "Kubernetes manifests applied"
    else
        log_error "Failed to apply Kubernetes manifests"
        exit 1
    fi
    
    # Wait for deployment to be ready
    log_info "Waiting for deployment to be ready..."
    if kubectl rollout status deployment/qecc-qml-deployment -n "$NAMESPACE" --timeout=300s; then
        log_success "Deployment is ready"
    else
        log_error "Deployment failed to become ready"
        exit 1
    fi
}

run_health_checks() {
    log_info "Running health checks..."
    
    if [ "$DRY_RUN" = "true" ]; then
        log_info "DRY RUN: Would run health checks"
        return 0
    fi
    
    # Get service endpoint
    local service_ip
    service_ip=$(kubectl get service qecc-qml-service -n "$NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "")
    
    if [ -z "$service_ip" ]; then
        # Use port-forward for testing
        log_info "Using port-forward for health check"
        kubectl port-forward service/qecc-qml-service 8080:80 -n "$NAMESPACE" &
        local pf_pid=$!
        sleep 5
        
        if curl -f http://localhost:8080/health &> /dev/null; then
            log_success "Health check passed (via port-forward)"
        else
            log_error "Health check failed"
            kill $pf_pid 2>/dev/null || true
            exit 1
        fi
        
        kill $pf_pid 2>/dev/null || true
    else
        # Direct health check
        log_info "Testing service at $service_ip"
        local max_attempts=10
        local attempt=1
        
        while [ $attempt -le $max_attempts ]; do
            if curl -f "http://${service_ip}/health" &> /dev/null; then
                log_success "Health check passed"
                return 0
            fi
            
            log_info "Health check attempt $attempt/$max_attempts failed, retrying in 10s..."
            sleep 10
            ((attempt++))
        done
        
        log_error "Health check failed after $max_attempts attempts"
        exit 1
    fi
}

run_smoke_tests() {
    log_info "Running smoke tests..."
    
    if [ "$DRY_RUN" = "true" ]; then
        log_info "DRY RUN: Would run smoke tests"
        return 0
    fi
    
    # Create a test pod
    cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: qecc-qml-smoke-test
  namespace: $NAMESPACE
spec:
  restartPolicy: Never
  containers:
  - name: smoke-test
    image: curlimages/curl:latest
    command: ["/bin/sh"]
    args:
      - -c
      - |
        echo "Running smoke tests..."
        
        # Test health endpoint
        curl -f http://qecc-qml-service/health || exit 1
        echo "✅ Health endpoint working"
        
        # Test readiness endpoint
        curl -f http://qecc-qml-service/ready || exit 1
        echo "✅ Readiness endpoint working"
        
        echo "🎉 All smoke tests passed!"
EOF
    
    # Wait for test to complete
    if kubectl wait --for=condition=Ready pod/qecc-qml-smoke-test -n "$NAMESPACE" --timeout=60s; then
        # Get test results
        kubectl logs pod/qecc-qml-smoke-test -n "$NAMESPACE"
        
        # Check if test passed
        if kubectl get pod qecc-qml-smoke-test -n "$NAMESPACE" -o jsonpath='{.status.phase}' | grep -q "Succeeded"; then
            log_success "Smoke tests passed"
        else
            log_error "Smoke tests failed"
            kubectl logs pod/qecc-qml-smoke-test -n "$NAMESPACE"
            exit 1
        fi
    else
        log_error "Smoke test pod failed to start"
        exit 1
    fi
    
    # Cleanup test pod
    kubectl delete pod qecc-qml-smoke-test -n "$NAMESPACE" --ignore-not-found=true
}

show_deployment_info() {
    log_info "Deployment information:"
    
    echo ""
    echo "📊 DEPLOYMENT STATUS"
    echo "===================="
    
    if [ "$DRY_RUN" = "false" ]; then
        # Show deployment status
        kubectl get deployment qecc-qml-deployment -n "$NAMESPACE" -o wide
        
        echo ""
        echo "🔗 SERVICES"
        echo "==========="
        kubectl get services -n "$NAMESPACE"
        
        echo ""
        echo "📈 PODS"
        echo "======="
        kubectl get pods -n "$NAMESPACE" -o wide
        
        echo ""
        echo "🔍 SERVICE ENDPOINTS"
        echo "==================="
        local service_ip
        service_ip=$(kubectl get service qecc-qml-service -n "$NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "pending")
        echo "Load Balancer IP: $service_ip"
        echo "Health Check: http://$service_ip/health"
        echo "API Endpoint: http://$service_ip/api"
        
        echo ""
        echo "📋 AUTO-SCALING"
        echo "==============="
        kubectl get hpa qecc-qml-hpa -n "$NAMESPACE" 2>/dev/null || echo "HPA not found"
    else
        echo "DRY RUN: Deployment information not available"
    fi
    
    echo ""
    echo "🎉 DEPLOYMENT COMPLETED SUCCESSFULLY!"
    echo "=================================="
}

rollback_deployment() {
    log_warning "Rolling back deployment..."
    
    if [ "$DRY_RUN" = "true" ]; then
        log_info "DRY RUN: Would rollback deployment"
        return 0
    fi
    
    kubectl rollout undo deployment/qecc-qml-deployment -n "$NAMESPACE"
    log_info "Rollback initiated"
}

cleanup_on_failure() {
    log_error "Deployment failed, cleaning up..."
    
    if [ "$DRY_RUN" = "false" ]; then
        # Delete deployment resources
        kubectl delete -f kubernetes/production-deployment.yaml --ignore-not-found=true
        log_info "Cleanup completed"
    fi
}

# Main execution
main() {
    log_info "Starting deployment with configuration:"
    echo "  Namespace: $NAMESPACE"
    echo "  Image Tag: $IMAGE_TAG"
    echo "  Registry: ${DOCKER_REGISTRY:-local}"
    echo "  Environment: $ENVIRONMENT"
    echo "  Dry Run: $DRY_RUN"
    echo ""
    
    # Set up error handling
    trap cleanup_on_failure ERR
    
    # Execute deployment steps
    check_prerequisites
    run_quality_gates
    build_docker_image
    create_namespace
    deploy_to_kubernetes
    run_health_checks
    run_smoke_tests
    show_deployment_info
    
    log_success "🎉 DEPLOYMENT COMPLETED SUCCESSFULLY!"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        --image-tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        --registry)
            DOCKER_REGISTRY="$2"
            shift 2
            ;;
        --environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --rollback)
            rollback_deployment
            exit 0
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --namespace NAMESPACE     Kubernetes namespace (default: qecc-qml)"
            echo "  --image-tag TAG          Docker image tag (default: latest)"
            echo "  --registry REGISTRY      Docker registry URL"
            echo "  --environment ENV        Deployment environment (default: production)"
            echo "  --dry-run               Perform dry run without actual deployment"
            echo "  --rollback              Rollback to previous deployment"
            echo "  --help                  Show this help message"
            echo ""
            echo "Environment Variables:"
            echo "  NAMESPACE               Kubernetes namespace"
            echo "  IMAGE_TAG               Docker image tag"
            echo "  DOCKER_REGISTRY         Docker registry URL"
            echo "  ENVIRONMENT             Deployment environment"
            echo "  DRY_RUN                 Dry run mode (true/false)"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Run main function
main