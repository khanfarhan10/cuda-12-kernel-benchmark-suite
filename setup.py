"""Root-level setup.py — installs Python test/benchmark helpers only.
For the CUDA extension, use extension/setup.py instead."""
from setuptools import setup, find_packages

setup(
    name="cuda-benchmark-suite",
    version="0.1.0",
    packages=find_packages(exclude=["extension*"]),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.1.0",
        "numpy>=1.24.0",
        "tabulate>=0.9.0",
        "pytest>=7.4.0",
    ],
)
