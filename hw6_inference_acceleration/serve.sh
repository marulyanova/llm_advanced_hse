#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--engine" ]]; then
    if [[ "${2:-}" != "sglang" ]]; then
        echo "Unsupported serving engine: ${2:-}" >&2
        exit 2
    fi
    shift 2
    exec /opt/sglang/bin/python -m sglang.launch_server \
        --model-path Qwen/Qwen3-8B --host 127.0.0.1 "$@"
fi

# Включаем API очистки кеша
export VLLM_SERVER_DEV_MODE=1
export PYTHONPATH="/opt/arctic${PYTHONPATH:+:$PYTHONPATH}"

exec vllm serve Qwen/Qwen3-8B --host 127.0.0.1 "$@"