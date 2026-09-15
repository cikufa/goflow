#!/usr/bin/env bash
# Run project Python with writes restricted to the project directory.
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="${GOFLOW_ENV_PREFIX:-$PROJECT_ROOT/../.conda/envs/goflow-repro}"
ENV_PREFIX="$(cd -- "$ENV_PREFIX" && pwd)"
mkdir -p "$PROJECT_ROOT/.cache/"{tmp,pip,xdg,config,data,torch,numba,cuda,optix,gl}
exec bwrap --ro-bind / / --bind "$PROJECT_ROOT" "$PROJECT_ROOT" \
  --bind "$ENV_PREFIX" "$ENV_PREFIX" \
  --dev-bind /dev /dev --proc /proc --chdir "$PROJECT_ROOT" \
  --unsetenv PYTHONPATH --setenv PYTHONNOUSERSITE 1 \
  --setenv TMPDIR "$PROJECT_ROOT/.cache/tmp" \
  --setenv PIP_CACHE_DIR "$PROJECT_ROOT/.cache/pip" \
  --setenv XDG_CACHE_HOME "$PROJECT_ROOT/.cache/xdg" \
  --setenv XDG_CONFIG_HOME "$PROJECT_ROOT/.cache/config" \
  --setenv XDG_DATA_HOME "$PROJECT_ROOT/.cache/data" \
  --setenv TORCH_HOME "$PROJECT_ROOT/.cache/torch" \
  --setenv NUMBA_CACHE_DIR "$PROJECT_ROOT/.cache/numba" \
  --setenv CUDA_CACHE_PATH "$PROJECT_ROOT/.cache/cuda" \
  --setenv OPTIX_CACHE_PATH "$PROJECT_ROOT/.cache/optix" \
  --setenv __GL_SHADER_DISK_CACHE_PATH "$PROJECT_ROOT/.cache/gl" \
  -- "$ENV_PREFIX/bin/python" "$@"
