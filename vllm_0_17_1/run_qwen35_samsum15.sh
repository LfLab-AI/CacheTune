#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${ROOT}/run_samsum.sh" \
  --model "${CACHETUNE_QWEN35_MODEL:-Qwen/Qwen3.5-9B}" \
  --recompute-ratio 0.15 --hybrid-policy block "$@"
