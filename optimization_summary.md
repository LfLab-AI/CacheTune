# CacheTune Optimization Summary

Author: lifei

This document summarizes the main optimizations used by CacheTune compared with a plain vLLM full-prefill path.

## 1. Frequency-Domain Token Selection

CacheTune uses frequency-domain analysis to identify context tokens that carry strong global semantic information. For each layer, the value tensor is transformed along the sequence dimension with RFFT. High-frequency bins are filtered out, the value tensor is reconstructed with inverse FFT, and each token is scored by the L2 norm of its low-frequency reconstruction.

The highest scoring tokens are selected for recomputation. Query and suffix tokens are always included because they must be computed in the merged context.

## 2. CPU Pinned-Memory KV Offload

Chunk-level KV tensors are offloaded to CPU pinned memory immediately after collection. This keeps GPU memory pressure low and allows later non-blocking DMA transfer. The merged CPU-side cache is arranged in global token order so that reuse indices can gather the correct token positions.

## 3. Sparse KV Transfer

Once recomputation tokens are selected, CacheTune transfers only the complementary reused tokens. If the recomputation ratio is `r`, the transfer volume is approximately `(1 - r) * N` tokens rather than the full context length `N`.

This is especially important when PCIe transfer is the TTFT bottleneck.

## 4. Hardware-Aware Ratio Scheduling

CacheTune profiles the machine to estimate:

```text
C_pcie: CPU-to-GPU transfer cost per token
C_gpu:  GPU recomputation cost per token
```

The ratio is chosen by balancing transfer and recomputation:

```text
r* = C_pcie / (C_gpu + C_pcie)
r_final = max(r*, r_min)
```

This rule increases recomputation when transfer is slow and decreases recomputation when GPU compute is the bottleneck.

## 5. Layer-Wise Pipeline Overlap

CacheTune uses an asynchronous CUDA stream for KV prefetch. While the current layer computes attention, the next layer's reusable KV tensors are transferred in the background. The exposed per-layer latency is therefore close to:

```text
max(T_transfer, T_recompute)
```

instead of their sum.

## 6. RoPE Position Correction

For rotary-position models, independently computed chunk keys can have incompatible local positions. CacheTune preserves or rebuilds raw key states and applies RoPE using the merged global positions before attention consumes the reused keys.

## 7. SSD Offload Mode

For experiments where KV tensors are placed on SSD, CacheTune treats disk I/O as a separate bottleneck. The SSD variant supports empirical search over recomputation ratios, which is useful because disk latency is often non-linear and depends on block size, queue depth, and storage hardware.

## 8. Summary

CacheTune combines offline frequency scoring, sparse transfer, hardware-aware recomputation, and asynchronous overlap. These optimizations reduce TTFT while keeping the recomputed token budget focused on positions that matter most for quality.
