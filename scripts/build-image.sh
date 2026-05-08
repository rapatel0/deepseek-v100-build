#!/usr/bin/env bash
# Build the CUDA image for V100 / sm_70.
set -euo pipefail

if [[ -f "$(dirname "$0")/../.env" ]]; then
  # shellcheck disable=SC1091
  source "$(dirname "$0")/../.env"
fi

: "${IMAGE:=local/deepseek-v4-flash:cuda-sm70}"

cd "$(dirname "$0")/.."

docker build \
  -f docker/Dockerfile \
  --build-arg CUDA_VERSION="${CUDA_VERSION:-12.2.2}" \
  --build-arg CUDA_ARCH="${CUDA_ARCH:-70-real}" \
  -t "${IMAGE}" \
  .

echo "Built ${IMAGE}"
