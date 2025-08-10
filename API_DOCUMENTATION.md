# 🔌 QECC-QML API Documentation

Complete API reference for the QECC-Aware Quantum Machine Learning Framework.

## 📋 Table of Contents
- [Overview](#overview)
- [Authentication](#authentication)
- [Core API Endpoints](#core-api-endpoints)
- [Quantum Neural Networks](#quantum-neural-networks)
- [Error Correction](#error-correction)
- [Circuit Optimization](#circuit-optimization)
- [Monitoring & Health](#monitoring--health)
- [Examples](#examples)

## 🔍 Overview

The QECC-QML API provides RESTful endpoints for:
- Creating and managing quantum neural networks
- Applying error correction codes
- Optimizing quantum circuits
- Monitoring system health
- Running benchmarks and validation

**Base URL**: `http://localhost:8000/api/v1`

**Content Type**: `application/json`

## 🔐 Authentication

Currently supports API key authentication:

```http
Authorization: Bearer your-api-key-here
```

## 🎯 Core API Endpoints

### Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "timestamp": 1703123456.789,
  "uptime": 3600.5
}
```

### Readiness Check
```http
GET /ready
```

**Response:**
```json
{
  "status": "ready",
  "version": "0.1.0",
  "services": {
    "database": "connected",
    "quantum_backend": "available",
    "cache": "operational"
  }
}
```

### System Status
```http
GET /api/v1/status
```

**Response:**
```json
{
  "framework": {
    "version": "0.1.0",
    "components": ["qnn", "qecc", "optimization", "monitoring"]
  },
  "resources": {
    "cpu_usage": 45.2,
    "memory_usage": 62.1,
    "quantum_backends": 3
  },
  "metrics": {
    "circuits_executed": 1245,
    "errors_corrected": 89,
    "optimization_runs": 456
  }
}
```

## 🧠 Quantum Neural Networks

### Create QNN
```http
POST /api/v1/qnn
```

**Request:**
```json
{
  "name": "my_qnn",
  "num_qubits": 4,
  "num_layers": 3,
  "entanglement": "circular",
  "feature_map": "amplitude_encoding",
  "rotation_gates": ["rx", "ry", "rz"],
  "optimization_level": 2
}
```

**Response:**
```json
{
  "qnn_id": "qnn_abc123",
  "status": "created",
  "configuration": {
    "num_qubits": 4,
    "num_layers": 3,
    "circuit_depth": 12,
    "parameter_count": 36
  },
  "created_at": "2024-01-15T10:30:00Z"
}
```

### Get QNN Details
```http
GET /api/v1/qnn/{qnn_id}
```

**Response:**
```json
{
  "qnn_id": "qnn_abc123",
  "name": "my_qnn",
  "configuration": {
    "num_qubits": 4,
    "num_layers": 3,
    "entanglement": "circular",
    "feature_map": "amplitude_encoding"
  },
  "status": "ready",
  "metrics": {
    "total_executions": 150,
    "average_fidelity": 0.987,
    "last_execution": "2024-01-15T14:22:30Z"
  },
  "error_correction": {
    "enabled": true,
    "code_type": "surface_code",
    "distance": 3
  }
}
```

### Execute QNN
```http
POST /api/v1/qnn/{qnn_id}/execute
```

**Request:**
```json
{
  "input_data": [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]],
  "parameters": [1.57, 0.78, 2.35, 0.92],
  "shots": 1024,
  "backend": "qasm_simulator",
  "noise_model": {
    "gate_error_rate": 0.001,
    "readout_error_rate": 0.01
  }
}
```

**Response:**
```json
{
  "execution_id": "exec_xyz789",
  "results": {
    "outputs": [[0.23, 0.77], [0.65, 0.35]],
    "fidelity": 0.982,
    "execution_time": 2.45,
    "shots_completed": 1024
  },
  "quantum_metrics": {
    "circuit_depth": 12,
    "gate_count": 48,
    "errors_detected": 2,
    "errors_corrected": 2
  },
  "status": "completed"
}
```

### List QNNs
```http
GET /api/v1/qnn?limit=10&offset=0
```

**Response:**
```json
{
  "qnns": [
    {
      "qnn_id": "qnn_abc123",
      "name": "my_qnn",
      "num_qubits": 4,
      "status": "ready",
      "created_at": "2024-01-15T10:30:00Z"
    }
  ],
  "total": 1,
  "limit": 10,
  "offset": 0
}
```

## 🛡️ Error Correction

### Create Error Correction Scheme
```http
POST /api/v1/qecc
```

**Request:**
```json
{
  "name": "surface_code_3",
  "code_type": "surface_code",
  "distance": 3,
  "logical_qubits": 1,
  "decoder": "minimum_weight_matching",
  "syndrome_extraction_frequency": 2
}
```

**Response:**
```json
{
  "qecc_id": "qecc_def456",
  "status": "created",
  "configuration": {
    "code_type": "surface_code",
    "distance": 3,
    "physical_qubits": 17,
    "logical_qubits": 1,
    "threshold": 0.01
  },
  "decoder_config": {
    "type": "minimum_weight_matching",
    "performance": {
      "decoding_time_ms": 5.2,
      "success_rate": 0.995
    }
  }
}
```

### Apply Error Correction to QNN
```http
POST /api/v1/qnn/{qnn_id}/qecc
```

**Request:**
```json
{
  "qecc_id": "qecc_def456",
  "enable": true,
  "adaptive_threshold": 0.99,
  "real_time_correction": true
}
```

**Response:**
```json
{
  "status": "applied",
  "integration": {
    "physical_qubits_required": 68,
    "logical_qubits_protected": 4,
    "overhead_factor": 4.25
  },
  "performance_impact": {
    "execution_time_increase": "3.2x",
    "fidelity_improvement": "+0.025",
    "error_rate_reduction": "15x"
  }
}
```

### Get QECC Performance
```http
GET /api/v1/qecc/{qecc_id}/performance
```

**Response:**
```json
{
  "qecc_id": "qecc_def456",
  "performance_metrics": {
    "total_corrections": 1247,
    "correction_success_rate": 0.994,
    "average_correction_time": 4.8,
    "syndrome_extraction_efficiency": 0.987
  },
  "error_statistics": {
    "single_bit_errors": 892,
    "two_bit_errors": 284,
    "three_bit_errors": 71,
    "uncorrectable_errors": 3
  },
  "time_series": {
    "last_hour": {
      "corrections": 45,
      "success_rate": 0.996
    },
    "last_day": {
      "corrections": 1089,
      "success_rate": 0.994
    }
  }
}
```

## ⚡ Circuit Optimization

### Optimize Circuit
```http
POST /api/v1/optimize
```

**Request:**
```json
{
  "circuit": {
    "qasm": "OPENQASM 2.0; include \"qelib1.inc\"; qreg q[4]; creg c[4]; h q[0]; cx q[0],q[1]; measure q -> c;",
    "format": "qasm"
  },
  "optimization_level": 2,
  "preserve_qecc_structure": true,
  "target_backend": "ibm_lagos",
  "optimization_goals": ["minimize_depth", "minimize_gates", "maximize_fidelity"]
}
```

**Response:**
```json
{
  "optimization_id": "opt_ghi789",
  "status": "completed",
  "original_circuit": {
    "gates": 24,
    "depth": 8,
    "qubits": 4
  },
  "optimized_circuit": {
    "gates": 18,
    "depth": 6,
    "qubits": 4,
    "qasm": "OPENQASM 2.0; include \"qelib1.inc\"; ..."
  },
  "improvements": {
    "gate_reduction": 6,
    "depth_reduction": 2,
    "fidelity_preserved": 0.9995,
    "optimization_time": 1.23
  },
  "optimizations_applied": [
    "redundant_gate_removal",
    "rotation_gate_merging",
    "cnot_chain_optimization"
  ]
}
```

### Get Optimization History
```http
GET /api/v1/optimize?qnn_id={qnn_id}&limit=10
```

**Response:**
```json
{
  "optimizations": [
    {
      "optimization_id": "opt_ghi789",
      "qnn_id": "qnn_abc123",
      "timestamp": "2024-01-15T15:30:00Z",
      "gate_reduction": 6,
      "depth_reduction": 2,
      "optimization_time": 1.23
    }
  ],
  "total": 1,
  "average_improvement": {
    "gate_reduction": 4.2,
    "depth_reduction": 1.8,
    "fidelity_preserved": 0.9991
  }
}
```

## 🔧 Training API

### Start Training
```http
POST /api/v1/train
```

**Request:**
```json
{
  "qnn_id": "qnn_abc123",
  "dataset": {
    "training_data": "base64_encoded_data_or_url",
    "validation_data": "base64_encoded_data_or_url",
    "format": "numpy"
  },
  "training_config": {
    "epochs": 100,
    "batch_size": 32,
    "learning_rate": 0.01,
    "optimizer": "noise_aware_adam",
    "loss_function": "cross_entropy"
  },
  "noise_config": {
    "gate_error_rate": 0.001,
    "readout_error_rate": 0.01,
    "coherence_time": 50e-6
  },
  "callbacks": ["early_stopping", "learning_rate_scheduler"]
}
```

**Response:**
```json
{
  "training_id": "train_jkl012",
  "status": "started",
  "estimated_duration": 1800,
  "progress_url": "/api/v1/train/train_jkl012/progress",
  "configuration": {
    "total_parameters": 36,
    "training_samples": 1000,
    "validation_samples": 200,
    "expected_epochs": 100
  }
}
```

### Get Training Progress
```http
GET /api/v1/train/{training_id}/progress
```

**Response:**
```json
{
  "training_id": "train_jkl012",
  "status": "training",
  "progress": {
    "current_epoch": 25,
    "total_epochs": 100,
    "completion_percentage": 25.0,
    "estimated_time_remaining": 1350
  },
  "metrics": {
    "current_loss": 0.234,
    "best_loss": 0.198,
    "current_accuracy": 0.876,
    "best_accuracy": 0.892,
    "training_fidelity": 0.987,
    "validation_fidelity": 0.983
  },
  "recent_history": [
    {
      "epoch": 25,
      "loss": 0.234,
      "accuracy": 0.876,
      "val_loss": 0.251,
      "val_accuracy": 0.865
    }
  ]
}
```

## 📊 Monitoring & Health

### Get System Metrics
```http
GET /api/v1/metrics
```

**Response:**
```json
{
  "timestamp": "2024-01-15T16:00:00Z",
  "system_metrics": {
    "cpu_usage_percent": 45.2,
    "memory_usage_percent": 62.1,
    "disk_usage_percent": 23.8,
    "network_io_mbps": 15.3
  },
  "quantum_metrics": {
    "active_qnns": 5,
    "circuits_per_second": 12.4,
    "average_fidelity": 0.987,
    "error_correction_rate": 0.994
  },
  "backend_metrics": {
    "available_backends": 3,
    "queue_lengths": {
      "ibm_lagos": 2,
      "simulator": 0,
      "braket_sv1": 1
    },
    "average_execution_time": 2.34
  }
}
```

### Get Performance Analytics
```http
GET /api/v1/analytics?timeframe=24h&granularity=1h
```

**Response:**
```json
{
  "timeframe": "24h",
  "granularity": "1h",
  "data_points": 24,
  "metrics": {
    "circuit_executions": [45, 52, 38, 61, 47, ...],
    "average_fidelity": [0.985, 0.987, 0.983, 0.989, ...],
    "error_rates": [0.001, 0.0008, 0.0012, 0.0009, ...],
    "response_times": [2.1, 2.3, 1.9, 2.5, ...]
  },
  "summary": {
    "total_executions": 1203,
    "average_fidelity": 0.9865,
    "average_error_rate": 0.00095,
    "p95_response_time": 3.2
  }
}
```

## 🧪 Benchmarking

### Run Benchmark
```http
POST /api/v1/benchmark
```

**Request:**
```json
{
  "benchmark_type": "noise_resilience",
  "configuration": {
    "noise_levels": [0.001, 0.005, 0.01, 0.05, 0.1],
    "circuit_sizes": [4, 8, 12, 16],
    "shots": 1024,
    "repetitions": 5
  },
  "qnn_config": {
    "num_layers": 3,
    "entanglement": "circular",
    "error_correction": true
  }
}
```

**Response:**
```json
{
  "benchmark_id": "bench_mno345",
  "status": "started",
  "estimated_duration": 600,
  "configuration": {
    "total_experiments": 100,
    "expected_results": 5000
  },
  "progress_url": "/api/v1/benchmark/bench_mno345/results"
}
```

### Get Benchmark Results
```http
GET /api/v1/benchmark/{benchmark_id}/results
```

**Response:**
```json
{
  "benchmark_id": "bench_mno345",
  "status": "completed",
  "completion_time": "2024-01-15T16:45:00Z",
  "results": {
    "noise_resilience": {
      "4_qubits": {
        "0.001": {"accuracy": 0.95, "fidelity": 0.987},
        "0.005": {"accuracy": 0.91, "fidelity": 0.978},
        "0.01": {"accuracy": 0.85, "fidelity": 0.965},
        "0.05": {"accuracy": 0.72, "fidelity": 0.923},
        "0.1": {"accuracy": 0.58, "fidelity": 0.876}
      }
    }
  },
  "analysis": {
    "noise_threshold": 0.023,
    "quantum_advantage_maintained": true,
    "error_correction_effectiveness": 0.85
  },
  "report_url": "/api/v1/benchmark/bench_mno345/report.pdf"
}
```

## 📝 Examples

### Complete QNN Workflow
```python
import requests
import json

base_url = "http://localhost:8000/api/v1"
headers = {"Authorization": "Bearer your-api-key"}

# 1. Create QNN
qnn_config = {
    "name": "iris_classifier",
    "num_qubits": 4,
    "num_layers": 3,
    "entanglement": "circular"
}
response = requests.post(f"{base_url}/qnn", json=qnn_config, headers=headers)
qnn_id = response.json()["qnn_id"]

# 2. Create error correction
qecc_config = {
    "name": "surface_code_3",
    "code_type": "surface_code",
    "distance": 3
}
response = requests.post(f"{base_url}/qecc", json=qecc_config, headers=headers)
qecc_id = response.json()["qecc_id"]

# 3. Apply error correction to QNN
requests.post(f"{base_url}/qnn/{qnn_id}/qecc", 
              json={"qecc_id": qecc_id, "enable": True}, 
              headers=headers)

# 4. Execute QNN
execution_config = {
    "input_data": [[0.1, 0.2, 0.3, 0.4]],
    "shots": 1024,
    "backend": "qasm_simulator"
}
response = requests.post(f"{base_url}/qnn/{qnn_id}/execute", 
                        json=execution_config, 
                        headers=headers)

print("Execution results:", response.json())
```

### JavaScript/Node.js Example
```javascript
const axios = require('axios');

const baseURL = 'http://localhost:8000/api/v1';
const headers = { 'Authorization': 'Bearer your-api-key' };

async function createAndExecuteQNN() {
  try {
    // Create QNN
    const qnnResponse = await axios.post(`${baseURL}/qnn`, {
      name: 'demo_qnn',
      num_qubits: 4,
      num_layers: 2
    }, { headers });
    
    const qnnId = qnnResponse.data.qnn_id;
    console.log('Created QNN:', qnnId);
    
    // Execute QNN
    const executionResponse = await axios.post(`${baseURL}/qnn/${qnnId}/execute`, {
      input_data: [[0.5, 0.5, 0.5, 0.5]],
      shots: 1024
    }, { headers });
    
    console.log('Execution results:', executionResponse.data);
    
  } catch (error) {
    console.error('API Error:', error.response?.data || error.message);
  }
}

createAndExecuteQNN();
```

## 🔒 Error Codes

| Code | Description |
|------|-------------|
| 400 | Bad Request - Invalid input parameters |
| 401 | Unauthorized - Missing or invalid API key |
| 403 | Forbidden - Insufficient permissions |
| 404 | Not Found - Resource not found |
| 409 | Conflict - Resource already exists |
| 422 | Unprocessable Entity - Validation errors |
| 429 | Too Many Requests - Rate limit exceeded |
| 500 | Internal Server Error - Server error |
| 503 | Service Unavailable - System overloaded |

## 📊 Rate Limits

| Endpoint Type | Rate Limit | Window |
|---------------|------------|---------|
| Health checks | 100/min | 1 minute |
| QNN operations | 50/min | 1 minute |
| Training | 10/hour | 1 hour |
| Benchmarking | 5/hour | 1 hour |
| Heavy computations | 20/min | 1 minute |

---

This API documentation provides comprehensive coverage of all QECC-QML framework endpoints with detailed examples and error handling guidance.