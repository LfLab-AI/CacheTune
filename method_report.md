# CacheTune Method Report

Author: lifei  
Date: February 2026  
Primary model: Mistral-7B-Instruct-v0.3  
Primary task: SAMSum dialogue summarization

## Abstract

CacheTune accelerates long-context LLM inference by reusing KV cache blocks while selectively recomputing the tokens that are most important for restoring cross-block attention. The method introduces two core mechanisms:

1. Frequency-domain token selection, which ranks context tokens by low-frequency value-state energy and recomputes the most globally informative tokens.
2. Hardware-aware recomputation scheduling, which estimates the best recomputation ratio from measured GPU compute cost and CPU-to-GPU transfer cost.

The implementation is integrated into a modified vLLM runtime with CPU pinned-memory offload and asynchronous layer-wise transfer. The goal is to reduce time to first token (TTFT) while preserving the quality of full prefill.

## 1. Problem Setting

For a query `Q` and document chunks `D_1, D_2, ..., D_n`, a cache reuse pipeline can precompute local KV cache blocks for each chunk and concatenate them before generation. This avoids a full prefill over the entire context, but each local KV cache was produced without cross-block attention. After concatenation, the reused KV cache differs from the KV cache that would have been produced by a true full-context prefill.

CacheTune addresses this mismatch by recomputing a selected subset of context tokens after the chunks are merged. The remaining tokens are reused from CPU memory and transferred to GPU memory only when needed.

For a context length `N` and recomputation ratio `r`:

- `r * N` tokens are recomputed by the model.
- `(1 - r) * N` tokens are reused from the cached KV tensors.

The main design questions are which tokens should be recomputed and how large `r` should be on the current hardware.

## 2. System Overview

CacheTune has an offline phase and an online inference phase.

```text
Offline phase
  1. Profile hardware to estimate C_gpu and C_pcie.
  2. Build per-chunk KV cache and offload it to CPU pinned memory.
  3. Run frequency-domain token scoring for each layer.
  4. Store precomputed recomputation indices in cache_fuse_metadata.

Online phase
  1. Prefetch reusable KV tensors for the current layer.
  2. Recompute the selected token subset on GPU.
  3. Merge recomputed KV with reused KV.
  4. Prefetch the next layer while the current layer computes attention.
```

## 3. Frequency-Domain Token Selection

The key observation is that low-frequency components in value states often represent broad, global semantic structure. Tokens with higher low-frequency energy are more likely to affect long-range attention and should be prioritized for recomputation.

For layer `l`, let the value tensor be:

```text
V_l in R^{N x H x D}
```

where `N` is sequence length, `H` is the number of attention heads, and `D` is head dimension.

CacheTune computes token importance as follows:

1. Apply a real FFT along the sequence dimension.
2. Keep only the low-frequency band controlled by `low_freq_pct`.
3. Reconstruct a smoothed value tensor with inverse FFT.
4. Score each token by the L2 norm of its reconstructed value state.
5. Select the top `K = floor(r * N)` context tokens.
6. Append suffix/query tokens, which must always be recomputed.

The result is stored as:

```python
cache_fuse_metadata["precomputed_indices"]
```

During online inference, the attention backend reads these indices directly, so frequency scoring does not add per-token online overhead.

## 4. Hardware-Aware Recompute Ratio

CacheTune estimates the per-token transfer and recomputation costs:

- `C_pcie`: CPU pinned-memory to GPU transfer cost per token.
- `C_gpu`: GPU recomputation cost per token per layer.

For context length `N` and recomputation ratio `r`:

```text
T_transfer(r)  = (1 - r) * N * C_pcie
T_recompute(r) = r * N * C_gpu
```

With layer-wise overlap, the exposed layer latency is approximated by:

```text
T_layer(r) = max(T_transfer(r), T_recompute(r))
```

Balancing the two terms gives:

```text
r* = C_pcie / (C_gpu + C_pcie)
```

To avoid under-recomputation, CacheTune applies a quality floor:

```text
r_final = max(r*, r_min)
```

The default examples use `r_min = 0.15`, with task-specific tuning available.

## 5. Implementation Notes

Important implementation paths:

| Path | Purpose |
| --- | --- |
| `example/blend_samsum_freq.py` | Main SAMSum driver with profiling, KV collection, frequency selection, and inference. |
| `example/*_freq_qwen.py` | Qwen/Qwen2.5 evaluation scripts. |
| `example/llama_freq_common.py` | Shared Llama-style CacheTune helpers. |
| `vllm_blend/vllm/model_executor/models/llama.py` | Layer state machine and asynchronous prefetch logic. |
| `vllm_blend/vllm/model_executor/models/qwen2.py` | Qwen2/Qwen2.5 model integration. |
| `vllm_blend/vllm/attention/backends/xformers.py` | Attention backend support for precomputed token indices and KV merging. |

The runtime uses a small state machine:

| Status | Meaning |
| --- | --- |
| `-1` | Decode path; normal decoding without cache reuse. |
| `0` | Full prefill path. |
| `1` | Selection/check layer; recompute selected queries. |
| `2` | Post-check layers; merge recomputed KV with reused KV. |

## 6. Experimental Setup

The default SAMSum setup uses:

- Model: Mistral-7B-Instruct-v0.3
- Engine: modified vLLM runtime
- Dataset: SAMSum
- Maximum context length: about 3400 tokens
- Metrics: TTFT and Rouge-L

Other scripts cover MuSiQue, WikiMQA, HotpotQA, and MultiNews.

## 7. Summary

CacheTune combines frequency-domain token selection with hardware-aware scheduling and asynchronous KV transfer. The system reduces the amount of KV data moved over PCIe, focuses recomputation on globally important tokens, and overlaps transfer with attention compute to reduce TTFT while keeping quality close to full prefill.
