#!/usr/bin/env bash
# Profile the C++ benchmark binary with Nsight Systems
# Produces a .nsys-rep file viewable in the Nsight Systems GUI.
set -e

BINARY=${1:-./build/cuda_bench}
OUTPUT=${2:-nsys_profile}

nsys profile \
    --trace=cuda,nvtx,osrt \
    --output="$OUTPUT" \
    --force-overwrite=true \
    "$BINARY"

echo ""
echo "Profile saved to ${OUTPUT}.nsys-rep"
echo "Open with: nsys-ui ${OUTPUT}.nsys-rep"
echo ""
echo "Key things to look for in Nsight Systems:"
echo "  - CUDA API timeline (cudaMemcpy, kernel launches)"
echo "  - Kernel concurrency / stream overlap"
echo "  - Memory transfer vs kernel overlap"
echo "  - CPU-GPU synchronization gaps"
