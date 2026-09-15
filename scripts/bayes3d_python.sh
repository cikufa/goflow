#!/usr/bin/env bash
# Separate historical perception stack; never put it on Isaac's import path.
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PERCEPTION_PREFIX="$PROJECT_ROOT/.conda/envs/bayes3d-repro"
CUDA_LIBRARY_PATH="$PERCEPTION_PREFIX/lib"
for component in cudnn cublas cusolver cusparse cufft cuda_cupti cuda_runtime cuda_nvrtc nccl curand nvtx; do
  CUDA_LIBRARY_PATH="$CUDA_LIBRARY_PATH:$PERCEPTION_PREFIX/lib/python3.10/site-packages/nvidia/$component/lib"
done
export LD_LIBRARY_PATH="$CUDA_LIBRARY_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
CUDA_ASSEMBLER_ROOT="$PERCEPTION_PREFIX/lib/python3.10/site-packages/nvidia/cuda_nvcc"
export PATH="$CUDA_ASSEMBLER_ROOT/bin:$PATH"
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CUDA_ASSEMBLER_ROOT${XLA_FLAGS:+ $XLA_FLAGS}"
export GOFLOW_ENV_PREFIX="$PERCEPTION_PREFIX"
exec "$PROJECT_ROOT/scripts/project_python.sh" "$@"
