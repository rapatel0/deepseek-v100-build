#!/usr/bin/env bash
# Download the DeepSeek-V4-Flash GGUF (86.7 GB) into MODEL_DIR.
# Idempotent — re-running only fetches missing/incomplete files.
set -euo pipefail

if [[ -f "$(dirname "$0")/../.env" ]]; then
  # shellcheck disable=SC1091
  source "$(dirname "$0")/../.env"
fi

: "${MODEL_DIR:?set MODEL_DIR in .env}"
: "${HF_REPO:?set HF_REPO in .env}"
: "${GGUF_FILE:?set GGUF_FILE in .env}"

mkdir -p "${MODEL_DIR}"

if [[ -f "${MODEL_DIR}/${GGUF_FILE}" ]]; then
  size=$(stat -c '%s' "${MODEL_DIR}/${GGUF_FILE}" 2>/dev/null || stat -f '%z' "${MODEL_DIR}/${GGUF_FILE}")
  echo "Model already present at ${MODEL_DIR}/${GGUF_FILE} (${size} bytes)."
  echo "Delete it to force a re-download."
  exit 0
fi

if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "huggingface-cli not found. Install with: pipx install huggingface_hub" >&2
  exit 1
fi

echo "Downloading ${GGUF_FILE} from ${HF_REPO} into ${MODEL_DIR}…"
echo "This is ~86.7 GB — go get a coffee."
huggingface-cli download "${HF_REPO}" "${GGUF_FILE}" \
  --local-dir "${MODEL_DIR}" \
  --local-dir-use-symlinks False

echo "Done."
ls -lh "${MODEL_DIR}/${GGUF_FILE}"
