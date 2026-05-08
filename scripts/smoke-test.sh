#!/usr/bin/env bash
# Hit the OpenAI-compatible endpoint and print the response.
# Usage:
#   ./smoke-test.sh                  # localhost:8080 (compose)
#   K8S=1 ./smoke-test.sh            # kubectl port-forward to the service
set -euo pipefail

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"

if [[ "${K8S:-0}" == "1" ]]; then
  echo "Port-forwarding svc/deepseek-v4-flash:8080…"
  kubectl -n deepseek port-forward svc/deepseek-v4-flash 8080:8080 >/dev/null &
  PF=$!
  trap 'kill ${PF} 2>/dev/null || true' EXIT
  sleep 2
fi

URL="http://${HOST}:${PORT}/v1/chat/completions"

echo "POST ${URL}"
curl --fail --silent --show-error "${URL}" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v4-flash",
    "messages": [
      {"role": "user", "content": "In one sentence, what is the capital of France?"}
    ],
    "temperature": 1.0,
    "top_p": 1.0,
    "max_tokens": 64
  }' | tee /dev/stderr | python3 -c 'import json,sys; print("\n---\n" + json.load(sys.stdin)["choices"][0]["message"]["content"])'
