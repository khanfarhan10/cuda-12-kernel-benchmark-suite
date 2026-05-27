#!/usr/bin/env bash
# Build the PyTorch C++/CUDA extension
set -e
cd extension
pip install -e . --no-build-isolation
echo "Extension built. Test with: python -c 'import cuda_kernels; print(cuda_kernels.__doc__)'"
