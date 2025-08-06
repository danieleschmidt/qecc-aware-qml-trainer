#!/usr/bin/env python3

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="qecc-aware-qml-trainer",
    version="0.1.0",
    author="Terragon Labs",
    author_email="contact@terragonlabs.com",
    description="Quantum Error Correction-Aware Machine Learning Trainer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/danieleschmidt/qecc-aware-qml-trainer",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "gpu": ["cupy-cuda12x>=12.0.0", "torch[cuda]>=2.0.0"],
        "dev": ["pytest>=7.0.0", "black>=22.0.0", "flake8>=4.0.0", "mypy>=0.950"],
        "docs": ["sphinx>=4.0.0", "sphinx-rtd-theme>=1.0.0"],
    },
    entry_points={
        "console_scripts": [
            "qecc-qml=qecc_qml.cli:main",
        ],
    },
)