#!/usr/bin/env bash
# run_all.sh — Full benchmark pipeline
#
# Usage:
#   cd /home/wslarch/Documents/projects/soft-cuda
#   bash benchmarks/run_all.sh
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/../build_wsl"

echo ""
echo "          soft-cuda  Complete Benchmark Pipeline                  "
echo ""

echo ">>> Building (Debug)..."
mkdir -p "$BUILD_DIR"
cmake -S "$SCRIPT_DIR/.." -B "$BUILD_DIR" \
      -DCMAKE_BUILD_TYPE=Debug \
      -DCMAKE_CUDA_COMPILER=/opt/cuda/bin/nvcc
cmake --build "$BUILD_DIR" -j$(nproc 2>/dev/null || echo 4)
echo ""

echo ">>> Running AOT profiler..."
"$BUILD_DIR/soft_profiler"
echo ""

echo ">>> Running soft-cuda benchmarks..."
"$BUILD_DIR/benchmarks/bench_softcuda"
echo ""

echo ">>> Running Deep MLP Hybrid benchmark..."
"$BUILD_DIR/benchmarks/bench_deep_mlp"
echo ""

echo ">>> Running PyTorch benchmark..."
VENV="$SCRIPT_DIR/../.venv"
if [ -f "$VENV/bin/activate" ]; then
    source "$VENV/bin/activate"
fi
python3 "$SCRIPT_DIR/bench_pytorch.py"
echo ""

echo ">>> All benchmarks complete."
