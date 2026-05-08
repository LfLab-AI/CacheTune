#!/usr/bin/env bash
set -euo pipefail

: "${VLLM_NCCL_SO_PATH:=/usr/lib/x86_64-linux-gnu/libnccl.so.2}"

DOCKER_BUILDKIT=1 docker build . \
  --target vllm-openai \
  --tag vllm-cachetune:test \
  --build-arg max_jobs=32 \
  --build-arg nvcc_threads=32 \
  --platform linux/amd64
