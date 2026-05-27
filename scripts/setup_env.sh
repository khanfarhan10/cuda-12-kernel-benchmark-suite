#!/usr/bin/env bash
# Setup Python environment and verify CUDA 12.3+ installation
set -e

echo "=== CUDA / NVCC verification ==="
nvcc --version
nvidia-smi

echo ""
echo "=== Python environment setup ==="
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "=== PyTorch CUDA check ==="
python -c "
import torch
print(f'PyTorch version : {torch.__version__}')
print(f'CUDA available  : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version    : {torch.version.cuda}')
    print(f'GPU             : {torch.cuda.get_device_name(0)}')
    print(f'cuDNN version   : {torch.backends.cudnn.version()}')
"
