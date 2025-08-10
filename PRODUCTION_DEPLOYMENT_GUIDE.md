# 🚀 QECC-QML Production Deployment Guide

This guide provides comprehensive instructions for deploying the QECC-QML framework to production environments.

## 📋 Prerequisites

### Required Tools
- Docker (v20.10+)
- Kubernetes cluster (v1.20+)
- kubectl CLI tool
- Helm (v3.8+) - optional but recommended

### System Requirements
- **CPU**: 4+ cores per node
- **Memory**: 8GB+ per node  
- **Storage**: 100GB+ available storage
- **Network**: High-speed networking for quantum backends

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────┐
│                Load Balancer                     │
└─────────────┬───────────────────────────────────┘
              │
┌─────────────▼───────────────────────────────────┐
│           QECC-QML Service                      │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐          │
│  │  Pod 1  │  │  Pod 2  │  │  Pod 3  │          │
│  └─────────┘  └─────────┘  └─────────┘          │
└─────────────────────────────────────────────────┘
              │
┌─────────────▼───────────────────────────────────┐
│              Data Layer                         │
│  ┌──────────┐  ┌───────────┐  ┌──────────────┐   │
│  │PostgreSQL│  │   Redis   │  │ Persistent   │   │
│  │          │  │  (Cache)  │  │   Storage    │   │
│  └──────────┘  └───────────┘  └──────────────┘   │
└─────────────────────────────────────────────────┘
```

## 🚀 Quick Deployment

### Option 1: Automated Script (Recommended)

```bash
# Make deployment script executable
chmod +x deploy.sh

# Deploy to production
./deploy.sh

# Deploy with custom settings
./deploy.sh --namespace my-qecc --image-tag v1.0.0 --environment production

# Dry run to validate configuration
./deploy.sh --dry-run
```

### Option 2: Manual Kubernetes Deployment

1. **Create namespace:**
```bash
kubectl create namespace qecc-qml
```

2. **Deploy application:**
```bash
kubectl apply -f kubernetes/production-deployment.yaml
```

3. **Verify deployment:**
```bash
kubectl get pods -n qecc-qml
kubectl get services -n qecc-qml
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LOG_LEVEL` | Logging level | `INFO` |
| `PROMETHEUS_ENABLED` | Enable metrics | `true` |
| `HEALTH_CHECK_ENABLED` | Enable health checks | `true` |
| `MAX_CIRCUIT_QUBITS` | Maximum qubits allowed | `100` |
| `MAX_CIRCUIT_DEPTH` | Maximum circuit depth | `1000` |
| `OPTIMIZATION_LEVEL` | Circuit optimization level | `2` |

### Secrets Configuration

Create Kubernetes secrets for sensitive data:

```bash
kubectl create secret generic qecc-qml-secrets \
  --from-literal=database-url="postgresql://user:password@postgres:5432/qecc_qml" \
  --from-literal=redis-url="redis://redis:6379/0" \
  --from-literal=api-secret-key="your-secret-key" \
  -n qecc-qml
```

## 🛡️ Security Configuration

### Network Policies
The deployment includes network policies that:
- Restrict ingress to authorized sources
- Limit egress to required services
- Isolate the application namespace

### Pod Security
- Runs as non-root user (UID 1000)
- Read-only root filesystem
- Security context constraints
- Resource limits enforced

### Secrets Management
- All sensitive data stored in Kubernetes secrets
- Secrets mounted as files, not environment variables
- Automatic secret rotation supported

## 📊 Monitoring & Observability

### Health Checks
- **Liveness Probe**: `GET /health` (30s interval)
- **Readiness Probe**: `GET /ready` (10s interval)  
- **Startup Probe**: `GET /health` (5s delay)

### Metrics Collection
Prometheus metrics available at `/metrics`:
- Request latency and throughput
- Quantum circuit execution metrics
- Resource utilization
- Error rates and patterns

### Logging
- Structured JSON logging
- Centralized log aggregation
- Log rotation and retention policies
- Debug logging available

## 🔄 Auto-Scaling

### Horizontal Pod Autoscaler (HPA)
- **CPU Target**: 70% utilization
- **Memory Target**: 80% utilization
- **Min Replicas**: 3
- **Max Replicas**: 20
- **Custom Metrics**: Quantum circuit execution rate

### Vertical Pod Autoscaler (VPA)
- Automatic resource recommendation
- Request/limit optimization
- Historical usage analysis

## 🗄️ Data Persistence

### Storage Classes
- **Data**: `fast-ssd` (100GB PVC)
- **Models**: `fast-ssd` (50GB PVC)
- **Logs**: `emptyDir` (ephemeral)

### Backup Strategy
- Automated daily backups
- Point-in-time recovery
- Cross-region replication
- Disaster recovery procedures

## 🔒 Production Checklist

### Pre-Deployment
- [ ] Kubernetes cluster operational
- [ ] Docker images built and pushed
- [ ] Secrets configured
- [ ] Network policies reviewed
- [ ] Resource quotas set
- [ ] Monitoring configured

### Post-Deployment
- [ ] Health checks passing
- [ ] Metrics collection working
- [ ] Logging operational
- [ ] Auto-scaling configured
- [ ] Backup procedures tested
- [ ] Security scan completed

## 🚨 Troubleshooting

### Common Issues

**Pods not starting:**
```bash
# Check pod status
kubectl describe pod <pod-name> -n qecc-qml

# View pod logs
kubectl logs <pod-name> -n qecc-qml

# Check events
kubectl get events -n qecc-qml --sort-by='.lastTimestamp'
```

**Service not accessible:**
```bash
# Check service endpoints
kubectl get endpoints qecc-qml-service -n qecc-qml

# Test service connectivity
kubectl port-forward service/qecc-qml-service 8080:80 -n qecc-qml
curl http://localhost:8080/health
```

**High resource usage:**
```bash
# Check resource usage
kubectl top pods -n qecc-qml
kubectl describe hpa qecc-qml-hpa -n qecc-qml

# Scale manually if needed
kubectl scale deployment qecc-qml-deployment --replicas=5 -n qecc-qml
```

### Emergency Procedures

**Rollback deployment:**
```bash
./deploy.sh --rollback
# Or manually:
kubectl rollout undo deployment/qecc-qml-deployment -n qecc-qml
```

**Emergency scale down:**
```bash
kubectl scale deployment qecc-qml-deployment --replicas=1 -n qecc-qml
```

**Drain problematic node:**
```bash
kubectl drain <node-name> --ignore-daemonsets --delete-emptydir-data
```

## 📈 Performance Tuning

### Resource Optimization
- Monitor actual resource usage
- Adjust requests/limits based on data
- Use VPA recommendations
- Optimize JVM settings if applicable

### Quantum Circuit Optimization
- Enable circuit compilation caching
- Tune optimization levels per workload
- Monitor quantum backend performance
- Implement circuit batching

### Database Performance
- Connection pooling configuration
- Query optimization
- Index management
- Read replica scaling

## 🔄 Updates & Maintenance

### Rolling Updates
```bash
# Update to new version
kubectl set image deployment/qecc-qml-deployment \
  qecc-qml=qecc-qml:v1.1.0 -n qecc-qml

# Monitor rollout
kubectl rollout status deployment/qecc-qml-deployment -n qecc-qml
```

### Blue-Green Deployment
Use the deployment script with blue-green strategy:
```bash
./deploy.sh --strategy=blue-green --image-tag=v1.1.0
```

### Canary Deployment
Deploy canary version for testing:
```bash
./deploy.sh --strategy=canary --image-tag=v1.1.0-beta
```

## 🎯 Best Practices

1. **Resource Management**
   - Set appropriate resource requests/limits
   - Use namespace resource quotas
   - Monitor resource utilization

2. **Security**
   - Regular security updates
   - Vulnerability scanning
   - Access control auditing

3. **Monitoring**
   - Comprehensive dashboards
   - Alerting on key metrics
   - Log analysis automation

4. **Testing**
   - Automated smoke tests
   - Load testing procedures
   - Disaster recovery drills

## 📞 Support

For deployment issues:
1. Check logs and events
2. Review troubleshooting guide
3. Contact operations team
4. Escalate to development team if needed

---

## 📝 Production Deployment Summary

The QECC-QML framework is now production-ready with:
- ✅ **Automated Deployment**: One-command deployment script
- ✅ **High Availability**: Multi-replica with auto-scaling
- ✅ **Security**: Network policies and pod security
- ✅ **Monitoring**: Comprehensive health checks and metrics
- ✅ **Scalability**: Horizontal and vertical auto-scaling
- ✅ **Reliability**: Rolling updates and rollback procedures

The deployment supports enterprise-grade requirements with production-hardened configurations.