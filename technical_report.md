# CacheTune Technical Report

Author: lifei  
Date: February 2026

## 1. Design Goals

CacheTune is designed for long-context inference where full prefill is expensive and naive KV cache reuse loses cross-block attention information. The system has four goals:

1. Reuse as much cached KV data as possible.
2. Recompute only the tokens that are most important for quality.
3. Move only the non-recomputed KV tensors across CPU-to-GPU links.
4. Hide transfer latency behind GPU attention computation.

The implementation is built in a modified vLLM runtime under `vllm_blend/`.

## 2. Cache Representation

Each document chunk is first processed independently. Its per-layer key and value tensors are captured before they are discarded by the normal inference path.

The collected tensors are immediately moved to CPU pinned memory:

```python
cpu_k = key.detach().to("cpu").pin_memory()
cpu_v = value.detach().to("cpu").pin_memory()
```

Pinned memory is used because it enables efficient non-blocking transfer back to GPU:

```python
gpu_k = cpu_k.to(device="cuda", non_blocking=True)
```

After all chunks are processed, their KV tensors are concatenated in global token order and stored in `cache_fuse_metadata["cpu_kv_cache"]`.

## 3. Token Selection

CacheTune selects recomputation tokens before online inference. The default selector analyzes the value tensor of each layer:

1. Convert value states to float32 for numerical stability.
2. Run `torch.fft.rfft` along the sequence dimension.
3. Zero high-frequency bins beyond the configured cutoff.
4. Run `torch.fft.irfft` to reconstruct low-frequency value states.
5. Score tokens by low-frequency L2 energy.
6. Select the highest scoring tokens under the recomputation budget.

The selected context indices are combined with suffix/query indices, then saved as per-layer lists.

## 4. Sparse Packing

After recomputation indices are known, CacheTune builds complementary reuse indices:

```text
reuse_indices = all_context_indices - recompute_indices
```

For post-check layers, only `reuse_indices` are transferred from CPU. Recomputed tokens are produced on GPU and scattered back into the correct global token positions. This makes the transfer volume proportional to `(1 - r) * N` rather than `N`.

## 5. Layer Pipeline

CacheTune uses a layer-wise pipeline:

```text
Layer i:
  wait for reusable KV of layer i
  launch prefetch for layer i + 1
  run attention and recomputation for layer i
```

The relevant metadata fields are:

| Field | Meaning |
| --- | --- |
| `check` | Enables CacheTune mode. |
| `status` | Current layer mode: full prefill, check layer, post-check layer, or decode. |
| `check_layer` | The first layer that performs partial recomputation. |
| `precomputed_indices` | Per-layer recomputation indices. |
| `reuse_indices` | Per-layer cache-transfer indices. |
| `cpu_kv_cache` | CPU pinned-memory KV tensors. |
| `pipeline_enabled` | Enables asynchronous prefetch. |
| `layer_counter` | Tracks which layer should read which precomputed index list. |

## 6. RoPE Handling

For Llama-style models, independently computed chunks can carry incompatible rotary positions if rotated keys are stored directly. CacheTune avoids this by storing raw pre-RoPE key tensors where possible and applying rotary embedding after the merged global positions are known.

The important invariant is that reused keys and recomputed keys must share the same global position system before attention reads them.

## 7. Hardware Profiling

The examples estimate two hardware constants:

```text
C_pcie = milliseconds per token for CPU-to-GPU KV transfer
C_gpu  = milliseconds per token for GPU recomputation
```

`C_pcie` is measured with pinned-memory tensor transfer. `C_gpu` is estimated with an attention-like compute proxy. The recomputation ratio is then chosen by:

```text
r* = C_pcie / (C_gpu + C_pcie)
r_final = max(r*, r_min)
```

This lets CacheTune adapt to machines with different PCIe bandwidth, GPU speed, or CPU memory behavior.

## 8. SSD Offload Variant

Some examples include an SSD offload mode where KV tensors can be stored on disk. In this setting, transfer cost is dominated by storage I/O, so the optimal recomputation ratio tends to be higher. The SSD script includes empirical search logic for calibrating a better ratio under non-linear disk latency.

## 9. Code Map

| Path | Role |
| --- | --- |
| `example/blend_samsum_freq.py` | Main frequency-selection SAMSum script. |
| `example/blend_samsum_freq_SSD.py` | SSD offload experiment. |
| `example/blend_freq_llama_common.py` | Shared helper code for Llama-style scripts. |
| `example/llama_freq_common.py` | Llama experiment orchestration and metrics. |
| `vllm_blend/vllm/model_executor/models/llama.py` | Llama model integration and prefetch pipeline. |
| `vllm_blend/vllm/model_executor/models/qwen2.py` | Qwen2/Qwen2.5 integration. |
| `vllm_blend/vllm/attention/backends/xformers.py` | Attention backend modifications. |
| `vllm_blend/vllm/worker/model_runner.py` | Worker-facing CacheTune control methods. |

## 10. Practical Notes

- Keep generated `*.out`, `*.log`, temporary text files, and `example/disk_offload_cache/` out of paper release artifacts.
- Keep third-party license files in `vllm_blend/`.
- Use the Qwen2.5 scripts when evaluating modern Qwen checkpoints because they automatically read model configuration fields such as `num_key_value_heads`.

## 11. Summary

CacheTune is a cache reuse and recomputation system that moves the expensive part of token selection offline, adapts recomputation to hardware, and overlaps sparse KV transfer with GPU compute. Its implementation keeps the runtime changes localized to model forward paths, attention backends, and worker control helpers.
