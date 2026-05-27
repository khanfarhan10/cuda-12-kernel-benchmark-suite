#!/usr/bin/env bash
# Profile a specific kernel with Nsight Compute for low-level GPU metrics.
# Produces a .ncu-rep file viewable in the Nsight Compute GUI.
set -e

BINARY=${1:-./build/cuda_bench}
KERNEL=${2:-vector_add_kernel}
OUTPUT=${3:-ncu_profile}

ncu \
    --target-processes all \
    --kernel-name "$KERNEL" \
    --set full \
    --output "$OUTPUT" \
    --force-overwrite \
    "$BINARY"

echo ""
echo "Profile saved to ${OUTPUT}.ncu-rep"
echo "Open with: ncu-ui ${OUTPUT}.ncu-rep"
echo ""
echo "Key metrics to examine:"
echo "  - sm__throughput.avg.pct_of_peak_sustained_elapsed   (SM utilization)"
echo "  - l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum       (global load bytes)"
echo "  - smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct (coalescing)"
echo "  - sm__warps_active.avg.pct_of_peak_sustained_active  (occupancy)"
echo "  - dram__bytes_read.sum / dram__bytes_write.sum        (DRAM bandwidth)"
