#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${CACHETUNE_PYTHON:-python}"
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PYTHON}" "${ROOT}/blend_samsum_freq.py" "$@"
