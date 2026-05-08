# deepseek-v100-build

CUDA build harness for serving DeepSeek-V4-Flash on 4x NVIDIA V100 32GB
GPUs via the `antirez/llama.cpp-deepseek-v4-flash` fork.

The fork includes CPU and Metal implementations for the custom DSv4
HyperConnection ops. This repo adds CUDA implementations for those HC ops and
patches the llama.cpp CUDA dispatcher so the model can run on Volta-era GPUs.

## What Is Included

```text
cuda-patches/
  ggml-cuda/
    dsv4-hc.cu
    dsv4-hc.cuh
  0001-wire-dsv4-hc-cuda.patch
docker/Dockerfile
scripts/
  build-image.sh
  download-model.sh
  host-bootstrap.sh
k8s/
  deployment.yaml
  service.yaml
  pvc.yaml
  namespace.yaml
```

The Dockerfile clones `antirez/llama.cpp-deepseek-v4-flash`, copies the CUDA
HC sources into `ggml/src/ggml-cuda/`, applies the patch, and builds
`llama-server` for `sm_70`.

## Requirements

- Linux/amd64 CUDA build host
- NVIDIA driver compatible with CUDA 12.2 runtime images
- Docker or another Dockerfile-compatible builder
- 4x V100 32GB for the target deployment
- The DSv4 GGUF from `antirez/deepseek-v4-gguf`

Defaults are intentionally local:

```bash
IMAGE=local/deepseek-v4-flash:cuda-sm70
REGISTRY_IMAGE=localhost:32000/deepseek-v4-flash:cuda-sm70
CUDA_VERSION=12.2.2
CUDA_ARCH=70-real
```

Override those in `.env` or the shell for your own registry.

## Build A v2-Compatible Image

```bash
cp .env.example .env

# Optional: edit REGISTRY_IMAGE, MODEL_DIR, NODE_NAME.

make build
make push
```

Or run the host bootstrap helper:

```bash
bash scripts/host-bootstrap.sh
```

That builds the CUDA image, pushes `REGISTRY_IMAGE`, and downloads the GGUF
into `MODEL_DIR`.

## Kubernetes Smoke Path

```bash
kubectl apply -k k8s/
kubectl -n deepseek rollout status deploy/deepseek-v4-flash --timeout=10m
kubectl -n deepseek logs deploy/deepseek-v4-flash | grep dsv4_hc
```

The `k8s/` manifests are a standalone example. In the homelab deployment, the
source is shipped through a build-source ConfigMap and the runtime manifests
live in the homelab repo.

## Verify The CUDA HC Patch

After a successful build, the shared CUDA library should contain the HC
symbols:

```bash
docker run --rm "$IMAGE" \
  sh -lc 'find /usr/local/lib -name "libggml-cuda.so*" -type f -print -quit | xargs nm -D | grep dsv4_hc'
```

At runtime, use scheduler debug logging to verify the HC ops land on CUDA:

```bash
GGML_SCHED_DEBUG=2 llama-server --log-verbose ...
```

## Notes

- The target architecture is `sm_70`; avoid Ampere+ features such as
  `cp.async`.
- HC kernels keep accumulators in FP32 for numerical stability.
- This repo intentionally does not include model weights, local `.env` files,
  logs, or cluster-specific secrets.
