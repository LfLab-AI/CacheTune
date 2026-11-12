# CacheTune for vLLM 0.17.1

This directory contains a plugin and worker extension for **official vLLM 0.17.1**, ported from the server runtime and aligned with the published CacheTune spectral method. The older vLLM 0.4.1 fork remains in [`../CacheTune/vllm_blend/`](../CacheTune/vllm_blend/).

## Installation

Use a separate Linux/CUDA environment for this adapter and the legacy fork. Python 3.10 or newer is required. The package pins `vllm==0.17.1` and `flashinfer-python==0.6.4`; the GPU validation target is NVIDIA A100. Install a matching CUDA-enabled PyTorch/vLLM environment and expose any compiler needed by FlashInfer through your normal environment.

From the repository root:

```bash
python -m pip install -e './vllm_0_17_1[test]'
python -c 'import vllm, flashinfer; print(vllm.__version__, vllm.__file__); print(flashinfer.__version__)'
```

This package uses the `vllm.general_plugins` entry point and `cachetune.worker.CacheTuneWorkerExtension`; a second complete copy of upstream vLLM is not required. The plugin registers CacheTune model classes only when `CACHETUNE_ENABLE_ADAPTER=1` is set before importing vLLM. The example drivers set this switch themselves. Start a new process when changing between native and CacheTune model registration.

FlashInfer may JIT-compile kernels. If your system `nvcc` belongs to an older toolkit, explicitly select the installed toolkit and a fresh build workspace before running:

```bash
export CUDA_HOME=/path/to/your/cuda-toolkit
export PATH="$CUDA_HOME/bin:$PATH"
export FLASHINFER_NVCC="$CUDA_HOME/bin/nvcc"
export FLASHINFER_WORKSPACE_BASE="$PWD/vllm_0_17_1/.flashinfer_builds"
```

The validation environment used CUDA toolkit 12.8 with PyTorch's CUDA 12.9 runtime. A stale FlashInfer `build.ninja` can retain another compiler path; selecting a fresh workspace avoids reusing it.

## Code map

| Path | Purpose |
|---|---|
| `cachetune/spectral_algorithms.py` | Published chunk-local low-pass, K/V score aggregation, shared selection and golden-section search helpers. |
| `cachetune/state.py` | Independent-cache collection, all-attention-layer statistics, selected/reused plans and compact-cache transfer state. |
| `cachetune/models.py` | Dense Transformer model integration and layerwise sparse repair. |
| `cachetune/attention.py` | Sparse attention with original-position causality and native-cache integration. |
| `cachetune/worker.py` | vLLM worker-extension RPC interface. |
| `cachetune/hybrid.py`, `cachetune/hybrid_state.py` | **Qwen3.5 dense text hybrid-attention implementation**: full-attention repair, branch caches and complete-sequence Gated DeltaNet execution. |
| `example/run_spectral.py` | Generic explicit-chunk request runner; fixed or measured-calibration ratio. |
| `example/blend_samsum_freq.py` | SAMSum driver retaining the legacy raw-prompt/first-line evaluation protocol. |
| `tests/` | Numerical, state, protocol and GPU integration checks. |

## Run an explicit chunk request

Create `request.json` with independently reusable text chunks and an online query:

```json
{
  "chunks": [
    "The Aurora Observatory is located in Northport.",
    "Northport belongs to the coastal region of Belvar."
  ],
  "query": "\nWhich region contains the Aurora Observatory? Answer briefly.\n"
}
```

Run from the repository root:

```bash
python vllm_0_17_1/example/run_spectral.py \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --request request.json --ratio-mode fixed --recompute-ratio 0.15 \
  --output-json spectral_result.json
```

The query is not encoded into reusable offline chunks, scored, or loaded from their cache. It is appended online and recomputed in full. Default cutoff `alpha=0.5` is exposed as `--low-freq-pct` in the SAMSum driver. The paper-aligned selection uses both K and V, normalizes within each layer and chunk, averages all cached attention layers, and selects `floor(ratio * chunk_length)` positions separately per chunk. Equal scores prefer the lower original position. No sink positions are forced by default.

For deployment calibration, use `--ratio-mode gss` in the explicit-chunk runner. The warm-start prior comes from recomputation and transfer costs, while candidate ratios are compared using measured request TTFT. See `python vllm_0_17_1/example/run_spectral.py --help` for calibration parameters. Fixed-ratio evaluation uses `--ratio-mode fixed` and does not run the search.

## SAMSum

Supply your SAMSum JSON file explicitly. Model IDs and the Python executable can be overridden; no private server, CUDA toolkit or Hugging Face cache path is embedded in the wrappers.

```bash
bash vllm_0_17_1/run_samsum.sh \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --dataset /path/to/samsum.json --sample-limit 2 \
  --recompute-ratio 0.15 --output-json samsum_smoke.json
```

Remove `--sample-limit` to process the complete supplied file. Use `CACHETUNE_PYTHON=/path/to/python` to choose another installed interpreter. `--standard-only` runs native inference using the same dataset/tokenization protocol. `--verify-repeat` and `--verify-logits` are correctness diagnostics and should be disabled during timing measurements.

The SAMSum driver preserves the legacy few-shot/raw prompt and answer-first-line scoring protocol. It is not a native chat-template benchmark. Its query suffix remains online in every mode. Results from another prompt, full-summary scoring or a different output length should be reported separately.

For tokenizers without a BOS, this compatibility driver retains the historical EOS-boundary convention. The generic `run_spectral.py` follows the published spectral runner instead: it adds only a real BOS when one exists, never synthesizes an EOS, and never drops a content token to simulate a boundary.

## Qwen3.5 hybrid attention

Use the same installed package; the adapter detects the dense text model configuration. The implementation is located in `cachetune/hybrid.py` and `cachetune/hybrid_state.py`.

```bash
bash vllm_0_17_1/run_qwen35_samsum15.sh \
  --model Qwen/Qwen3.5-0.8B \
  --dataset /path/to/samsum.json --sample-limit 2 \
  --output-json qwen35_smoke.json
```

The wrapper defaults to `Qwen/Qwen3.5-9B`, ratio 0.15 and the `block` policy; explicit command-line options override the wrapper defaults. Under `block`, selected positions are recomputed in full-attention blocks, and cached branch outputs fill unselected positions while preserving the current residual stream. Every Gated DeltaNet recurrent operator still processes all tokens in order. Recurrent states from independently encoded chunks are not concatenated. The ratio controls eligible token positions, not total model FLOPs.

Logical full-attention layer 0 means the first full-attention layer identified from the model configuration, not decoder layer 0. Fusion starts at that first full-attention layer. `attention` and `mlp` policies remain optional experimental variants; the default port uses `block`. Details and the dense/hybrid layer-boundary distinction are in [METHOD.md](METHOD.md).

## Execution scope

- Dense decoder models supported by the adapter include Llama/Mistral, Qwen2/Qwen2.5 and Qwen3. Qwen3.5 support is for **dense text hybrid-attention models**.
- Single request, one sequence, one GPU: TP=PP=1; eager execution; automatic prefix caching, chunked prefill and asynchronous scheduling are disabled.
- Cache modes: `cpu_pipeline` (default), `cpu_serial` and `gpu`. The CPU path transfers only reused compact KV/branch rows; selected positions and the online suffix are recomputed.
- The dense model boundary matches the published `check_layers=[1]`: layer 0 is dense; layer 1 computes dense Q/K/V then selected-query attention/MLP; later layers fuse recomputed selected KV with reused complementary KV.
- Sparse attention uses original absolute token positions, never the compressed query-row index, for causality. Reused pre-RoPE K is rotated once at its new position.
- Ordinary decoding uses native vLLM after the repaired prefill cache is installed.

This implementation has not established continuous batching, multi-request concurrency, TP/PP, multimodal reuse, Qwen3.5 MoE, quantization, LoRA or speculative-decoding support. The new directory does not publish SSD/HDD cold-read performance claims. Historical server experiment tables are not validation of this revised spectral implementation.

## Validation

The A100 regression results and exact source hashes are in [VALIDATION.md](VALIDATION.md) and [`validation/summary.json`](validation/summary.json). The checks cover five real dense/hybrid models, both public runner modes, original-position attention, and the published numerical scoring contract.

Run CPU-capable numerical and protocol checks:

```bash
python -m pytest vllm_0_17_1/tests -q
```

Run the paper-method GPU integration check in an environment with the model available:

```bash
python vllm_0_17_1/tests/integration_paper.py \
  --model /path/to/model --output-json paper_integration.json
```

Validation should check published-score agreement, independent chunk boundaries, query exclusion, original-position causality, ratio=1 agreement with native full recomputation, and request-state isolation. Qwen3.5 additionally requires complete ordered recurrent execution and its ratio=1 convolution/SSM endpoint checks. Integration results must be associated with the actual model, environment and source revision; unit tests alone do not establish GPU correctness or a speedup.

## License

`LICENSE` retains the upstream Apache License 2.0 from the legacy vLLM fork. The adapter builds on [vLLM 0.17.1](https://github.com/vllm-project/vllm/tree/v0.17.1), and its spectral helpers follow the published [`CacheTune/example/spectral_algorithms.py`](../CacheTune/example/spectral_algorithms.py). Keep the upstream license and notices when redistributing source.

