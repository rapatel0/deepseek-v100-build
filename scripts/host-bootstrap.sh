#!/usr/bin/env bash
# Run this on a GPU host, or any host that can build CUDA images and reach
# your registry. It builds the image, pushes to the in-cluster
# registry, and downloads the GGUF into the local-PV path.
#
# Prereq on gpu-01: Docker (or microk8s + ctr), huggingface-cli or
# `pipx install huggingface_hub`, network access to huggingface.co.
#
# Usage:
#   bash scripts/host-bootstrap.sh
#
# Idempotent: re-running skips work already done.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -f .env ]]; then
  # shellcheck disable=SC1091
  source .env
fi

: "${IMAGE:=local/deepseek-v4-flash:cuda-sm70}"
: "${REGISTRY_IMAGE:=localhost:32000/deepseek-v4-flash:cuda-sm70}"
: "${MODEL_DIR:=/srv/models/deepseek-v4-flash}"
: "${HF_REPO:=antirez/deepseek-v4-gguf}"
: "${GGUF_FILE:=DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf}"

echo "==> 1. Build CUDA image (sm_70 / V100)"
if docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  echo "    ${IMAGE} already exists; skip (force: docker image rm ${IMAGE})"
else
  docker build -f docker/Dockerfile -t "${IMAGE}" .
fi

echo "==> 2. Push to in-cluster registry"
docker tag  "${IMAGE}" "${REGISTRY_IMAGE}"
docker push "${REGISTRY_IMAGE}"

echo "==> 3. Download GGUF (~86.7 GB)"
sudo mkdir -p "${MODEL_DIR}"
sudo chown "$(id -u):$(id -g)" "${MODEL_DIR}"
if [[ -f "${MODEL_DIR}/${GGUF_FILE}" ]]; then
  size=$(stat -c '%s' "${MODEL_DIR}/${GGUF_FILE}")
  echo "    ${GGUF_FILE} already present (${size} bytes); skip"
else
  if ! command -v huggingface-cli >/dev/null 2>&1; then
    echo "    huggingface-cli not found. Install: pipx install huggingface_hub" >&2
    exit 1
  fi
  huggingface-cli download "${HF_REPO}" "${GGUF_FILE}" \
    --local-dir "${MODEL_DIR}" \
    --local-dir-use-symlinks False
fi

echo
echo "==> Done. Verify on the host:"
echo "    docker images | grep deepseek-v4-flash"
echo "    ls -lh ${MODEL_DIR}/"
echo
echo "Then on the laptop with KUBECONFIG set:"
echo "    kubectl apply -f k8s/deployment.yaml"
echo "    kubectl -n deepseek logs -f deploy/deepseek-v4-flash"
