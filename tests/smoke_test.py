"""Functional smoke test for the deployed DeepSeek-V4-Flash endpoint.

Run after the container/pod reports healthy:
    python3 tests/smoke_test.py
    BASE_URL=http://epyc-1:8080/v1 python3 tests/smoke_test.py
"""
import json
import os
import sys
import urllib.request


BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:8080/v1")


def post(path: str, payload: dict) -> dict:
    req = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read())


def get(path: str) -> dict:
    with urllib.request.urlopen(f"{BASE_URL}{path}", timeout=30) as resp:
        return json.loads(resp.read())


def main() -> int:
    print(f"Target: {BASE_URL}")

    models = get("/models")
    ids = [m["id"] for m in models.get("data", [])]
    print(f"  /v1/models -> {ids}")
    assert ids, "no models reported"

    out = post(
        "/chat/completions",
        {
            "model": ids[0],
            "messages": [
                {"role": "user", "content": "Reply with the single word: ok"}
            ],
            "temperature": 1.0,
            "top_p": 1.0,
            "max_tokens": 8,
        },
    )
    text = out["choices"][0]["message"]["content"].strip()
    print(f"  /v1/chat/completions -> {text!r}")
    assert text, "empty completion"

    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
