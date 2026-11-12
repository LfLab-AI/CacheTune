# Correspondence to the CacheTune paper

This port follows the paper-aligned spectral implementation in `CacheTune/example/spectral_algorithms.py`. The old examples also retain a separate legacy path; its first-layer V-only selector is not the method implemented by default here.

## Independent offline cache and frequency scores

Each reusable chunk is encoded separately. For a chunk of length N, each cached attention layer supplies K and V with token axis 0. K is captured after any architecture-required K normalization and before RoPE; V is not position-rotated. Temporary boundary tokens used only to encode the chunk are excluded from the stored content slice. The online query/suffix never participates in offline scoring.

For each layer and each channel:

1. Compute the real FFT along the chunk's token axis. Its one-sided length is F = floor(N/2) + 1.
2. Keep bins k < floor(alpha * F), set the rest to zero, and apply inverse real FFT with the explicit original length N. The cutoff is literal and may be zero for a short chunk; no minimum-one-bin rule is introduced.
3. Compute separate L2 magnitudes for the low-pass K and V over all logical KV heads and head channels at each original token position.
4. Take their arithmetic mean: score_layer[i] = 0.5 * (norm(K_low[i]) + norm(V_low[i])).
5. Divide by the sum over the same chunk's positions plus epsilon = 1e-12; then average these normalized scores over all cached attention layers.

Chunks are never concatenated before the transform or before their layer normalization. The helper's TP-statistics reduction sums squared head norms and corrects replication before taking square roots, but the current runtime supports TP=1 only. Qwen3.5 contributes its full-attention layers to this score; recurrent Gated DeltaNet state is not treated as ordinary attention K/V.

## One selected set and its complement

For each chunk independently, choose its top floor(r * N) scores. Stable descending ordering resolves equal scores in favor of the smaller original token index. Add chunk offsets and sort the resulting positions in original sequence order. The selected and reused context sets are disjoint and cover the cached context. All online suffix positions are then appended to the selected set.

The same selection is used across layers. There are no forced sink tokens in the paper-aligned default. A global top-k over concatenated chunks, first-layer V-only scores, a max-one FFT cutoff, query-conditioned offline scores, or independent layerwise selections would define different algorithms.

## Dense Transformer repair

The port retains the published check_layers=[1] execution boundary, with zero-based decoder-layer indexing:

- Layer 0 runs the complete prompt normally.
- Layer 1 computes Q/K/V for all prompt positions. Only selected Q rows proceed through attention and the subsequent MLP; the complete current K/V is used at this boundary.
- From layer 2 onward, Q/K/V and MLP work follows the shared selected positions. Recomputed K/V is scattered into those original positions and combined with old complementary K/V. Reused K is rotated at its new global position before the attention read.

Thus the first two layers' Q/K/V projections are dense, while layer 1's attention queries and MLP are already sparse. The query suffix is always included. Sparse query causality is defined by key_position <= original_query_position, including any configured sliding-window constraint. It must not be derived from a compressed M-by-N triangular mask.

The repaired complete K/V is written into native vLLM cache storage so subsequent decode follows the ordinary engine path. The shared path maintains the identity of selected tokens across layers; it does not guarantee equality to full recomputation when selected queries still read mismatched reused K/V.

## Complement loading and overlap

`cpu_pipeline` stores the reused complement contiguously in pinned CPU memory, prefetches the next relevant layer on a transfer stream, and waits on its copy event before using it. The selected rows are not transferred from the old cache. `cpu_serial` uses the same selected/reused plan without transfer overlap; `gpu` holds the reused complement on the GPU.

Offline chunk encoding, FFT scoring and cache preparation must be distinguished from timed online layer loading, attention and recomputation. End-to-end wall time and engine TTFT should retain their respective definitions. This port's GPU/CPU modes do not establish SSD/HDD behavior.

## Qwen3.5 hybrid extension

Qwen3.5 interleaves full attention with Gated DeltaNet (GDN). Its recurrent operators require a complete ordered token stream. The hybrid extension therefore keeps N rows of hidden/residual state and executes every native GDN recurrent operator over all N tokens. It neither selects only r*N recurrent updates nor concatenates independent chunk recurrent states.

The default `block` policy applies selective repair to full-attention blocks from the first configured full-attention layer, inclusive:

- Full-attention queries and eligible projections use selected positions; reused complementary K/V supplies the remaining keys/values.
- Unselected attention/MLP branch outputs are restored from their independently encoded branch cache, and combined with the **current** residual stream. Cached old residual states must not replace that stream.
- MLPs in GDN layers remain complete-sequence under `block`; the native recurrent operators remain complete-sequence under every policy.

The `attention` policy leaves all MLP work dense. The experimental `mlp` policy also sparsifies MLPs in GDN layers. Both are optional and are not the default block extension.

This architecture-specific boundary differs from the legacy dense check layer: hybrid fusion starts at logical full-attention layer 0, whereas dense models preserve the published dense/check/fusion sequence above. Actual decoder-layer numbers are read from model configuration. Q/K normalization, attention gating, output projection, residual ordering and text MRoPE retain the native model's semantics.

Branch caches and the complete recurrent stream add storage, transfer and compute costs. The selected-token ratio describes the repaired positions in eligible branches, not the fraction of all model operations. At r=1, no old complementary branch or KV values should affect the result; this must be verified against native full inference, including recurrent-state endpoints.

## Deployment ratio

Fixed-ratio runs use r=0.15 by default. The generic request runner also exposes the published warm-started golden-section search. Positive recomputation/transfer costs provide the clipped prior ti/(tc+ti); measured complete-request TTFT at each candidate is the search objective. The warm probe pair is followed by reinitialized standard golden probes. The final bracket midpoint is explicitly measured rather than assigning another ratio's timing to it.

Calibration and test requests should be separated when reporting benchmark results, and timings should be collected without logits probes or duplicate correctness forwards. A first-layer timing estimate is a prior, not a substitute for the measured objective.

When explicit `--tc` and `--ti` measurements are omitted, the driver obtains a warm-start approximation from full-prefill TTFT per context token and a separate CUDA-event H2D profile per reused token. The latter includes eligible KV and hybrid branch sources. Full-prefill cost includes fixed suffix/recurrent work, so this estimate is only a search prior, particularly for hybrid models. It does not replace complete-request measurements or claim that all profiled work can be skipped. GPU-resident caches use fixed mode; the provided GSS driver calibrates host-transfer overlap.

## Checks before a correctness claim

Compare scores and selected indices directly with the published helper, including odd lengths, one-token chunks, zero scores and ties. Verify that chunk order changes only offsets, and that the suffix cannot leak into offline statistics. Check sparse attention against the same rows of a dense reference at noncontiguous original positions.

On the GPU, compare r=1 with native full recomputation and test A/B/A requests for state isolation. At r<1, test complementary-cache preservation, complete native-cache writes, finite logits and cache-mode consistency. Hybrid checks additionally trace GDN input length/order and compare ratio=1 convolution/SSM endpoints. These checks establish implementation properties; task quality and latency require separate runs.
