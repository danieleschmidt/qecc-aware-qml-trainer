# QECC-Aware QML Trainer - Quick Start Guide

Get up and running with quantum error correction-aware machine learning in under 5 minutes!

## 🚀 Installation

### Prerequisites
- Python 3.9 or higher
- 4GB+ RAM (8GB recommended)
- 10GB+ free disk space

### Quick Install
```bash
# Clone the repository
git clone https://github.com/your-org/qecc-aware-qml-trainer.git
cd qecc-aware-qml-trainer

# Install core dependencies
pip install numpy qiskit qiskit-aer

# Install the package
pip install -e .
```

## 💡 Quick Examples

### Example 1: Basic Quantum ML Training
```python
from qecc_qml.core.quantum_nn import QECCAwareQNN
from qecc_qml.training.basic_trainer_clean import BasicQECCTrainer
import sys, os
sys.path.insert(0, os.path.join('.', 'qecc_qml', 'datasets'))
from simple_datasets import load_quantum_classification

# Create quantum neural network
qnn = QECCAwareQNN(num_qubits=4, num_layers=2)

# Load quantum dataset
X_train, y_train = load_quantum_classification(dataset='synthetic', n_samples=100)

# Create trainer
trainer = BasicQECCTrainer(qnn, learning_rate=0.05)

# Train the model
history = trainer.fit(X_train, y_train, epochs=10)

# Make predictions
X_test, y_test = load_quantum_classification(dataset='synthetic', n_samples=20, random_state=999)
predictions = trainer.predict(X_test)
results = trainer.evaluate(X_test, y_test)

print(f"Test Accuracy: {results['accuracy']:.3f}")
```

### Example 2: Error Correction-Enhanced Training
```python
from qecc_qml.codes.surface_code import SurfaceCode

# Create QNN with error correction
qnn = QECCAwareQNN(num_qubits=4, num_layers=2)

# Add surface code protection (if available)
try:
    surface_code = SurfaceCode(distance=3, logical_qubits=4)
    qnn.add_error_correction(surface_code)
    print("Error correction enabled!")
except:
    print("Continuing without error correction")

# Use robust trainer for enhanced reliability
from qecc_qml.training.robust_trainer import RobustQECCTrainer

trainer = RobustQECCTrainer(
    qnn, 
    enable_monitoring=True,
    validation_freq=3,
    checkpoint_freq=5
)

# Train with enhanced error handling
history = trainer.fit(X_train, y_train, epochs=15)
diagnostics = trainer.get_training_diagnostics()

print(f"Success Rate: {diagnostics['performance_metrics']['successful_epochs']}")
```

### Example 3: High-Performance Scalable Training
```python
from qecc_qml.training.scalable_trainer import ScalableQECCTrainer

# Generate large dataset
X_large, y_large = load_quantum_classification(dataset='synthetic', n_samples=1000)

# Create scalable trainer with optimizations
trainer = ScalableQECCTrainer(
    qnn,
    enable_optimization=True,    # Intelligent caching
    enable_auto_scaling=True,   # Dynamic resource scaling  
    enable_parallel=True,       # Parallel processing
    initial_batch_size=64,
    performance_target=1.0      # Target 1 second per epoch
)

# High-performance training
history = trainer.fit(X_large, y_large, epochs=20)

# Get optimization statistics
if hasattr(trainer, 'get_optimization_diagnostics'):
    opt_stats = trainer.get_optimization_diagnostics()
    print(f"Cache Hit Rate: {opt_stats.get('optimization_stats', {}).get('cache_efficiency', 0)*100:.1f}%")
```

## 🏃‍♂️ Run Complete Examples

Execute the provided example scripts:

```bash
# Basic functionality demo
python3 examples/basic_training_example.py

# Robust training with error handling  
python3 examples/robust_training_example.py

# High-performance scalable training
python3 examples/scalable_training_example.py
```

## 🧪 Test Your Installation

Run the comprehensive test suite:

```bash
# Run quality gates
python3 test_comprehensive_system.py

# Expected output:
# ✅ Quality Gates: PASSED
# Tests Run: 17, Success Rate: 94.1%
```

## 🚀 Production Deployment

Deploy to your preferred environment:

```bash
# Local deployment
python3 deploy.py local

# Docker deployment  
python3 deploy.py docker

# Kubernetes deployment
python3 deploy.py kubernetes
```

## 📚 Next Steps

### Learn More
- Read the full [Implementation Report](IMPLEMENTATION_REPORT.md)
- Explore [API Documentation](API_DOCUMENTATION.md)
- Check out [Research Examples](RESEARCH_FRAMEWORK_GUIDE.md)

### Customize and Extend
- Add custom error correction codes
- Implement custom optimizers
- Create custom datasets
- Develop application-specific modules

### Get Support
- Review [troubleshooting guide](TROUBLESHOOTING.md)
- Check [frequently asked questions](FAQ.md)
- Join the community discussions

## 🎯 Common Use Cases

### Research Applications
```python
# Quantum chemistry simulations
# Drug discovery optimization  
# Materials science modeling
# Cryptographic algorithm testing
```

### Industrial Applications
```python
# Financial risk modeling
# Supply chain optimization
# Machine learning model training
# Quantum advantage benchmarking
```

## ⚡ Performance Tips

1. **Start Small**: Begin with 2-4 qubits for testing
2. **Enable Caching**: Use Generation 3 trainer for repeated experiments
3. **Monitor Resources**: Check memory and CPU usage during training
4. **Batch Optimization**: Use adaptive batch sizing for best performance
5. **Error Correction**: Add QECC only when noise levels require it

## 🛠️ Troubleshooting

### Common Issues
- **Import Errors**: Ensure all dependencies are installed
- **Memory Issues**: Reduce batch size or number of qubits
- **Performance Issues**: Enable optimization in scalable trainer
- **Docker Issues**: Ensure Docker is installed and running

### Quick Fixes
```bash
# Reinstall package
pip uninstall qecc-aware-qml-trainer
pip install -e .

# Clear cache
rm -rf __pycache__ *.pyc
rm -rf checkpoint_*.json

# Reset environment  
pip install --upgrade qiskit numpy
```

---

**🚀 You're ready to explore quantum error correction-aware machine learning!**

Start with the basic example and progressively explore more advanced features as you become comfortable with the framework.

**Happy quantum machine learning! 🎉**