# Validation of the vLLM 0.17.1 port

Date: 2026-11-13. These are implementation-correctness checks on public synthetic text, not task-quality or serving-performance claims.

## Environment and method reference

- NVIDIA A100-SXM4-40GB; single GPU, TP=PP=1; eager execution.
- Python 3.12.13, PyTorch 2.10.0+cu129, vLLM 0.17.1, Transformers 4.57.6 and FlashInfer 0.6.4.
- FlashInfer compilation used CUDA toolkit 12.8 and a separate build workspace.
- Method reference: the published `CacheTune/example/spectral_algorithms.py` at repository commit `0c58c7b3a4dac9e3bca7eb8292c8449fae10d35e`, checked against the V14 paper, Sections 4.1–4.3 and Appendices B/C/E.8.1.
- The spectral helper in this port is identical to that published helper. Comparing 1,386 server vLLM Python source files against the official v0.17.1 source archive found no modifications. The port does not depend on an undisclosed native-vLLM Python patch.

## Real-model integration

Every model below passed `tests/integration_paper.py` with its default zero logit-error tolerance.

| Model | Decoder layers | Full-attention / recurrent layers | Ratio=1 maximum first-logit error | Result |
|---|---:|---:|---:|---|
| Meta-Llama-3-8B | 32 | 32 / 0 | 0 | PASS |
| Mistral-7B-Instruct-v0.3 | 32 | 32 / 0 | 0 | PASS |
| Qwen2.5-0.5B-Instruct | 24 | 24 / 0 | 0 | PASS |
| Qwen3-0.6B | 28 | 28 / 0 | 0 | PASS |
| Qwen3.5-0.8B | 24 | 6 / 18 | 0 | PASS |

For each model, the script checked two ratio=1 requests against native full recomputation, including equal generated tokens and full-logit byte hashes. At ratio 0.15 it ran GPU, CPU-pipeline and CPU-serial modes twice each, checked equal outputs and first logits, traced every layer's QKV/query/MLP row counts, verified the native paged-cache contents, changed the online query without recollecting context, and ran A/B/A request-state isolation checks.

For Qwen3.5, ratio=1 also produced equal native convolution and SSM recurrent-state endpoints. At ratio 0.15, every recurrent layer received the complete ordered sequence, while the `block` policy sparsified full-attention branches and restored complementary branch outputs. This validates the small dense text hybrid configuration; it is not a new Qwen3.5-9B accuracy or latency measurement.

## Numerical, state and public-entry checks

**The clean public-source suite passed 219 tests and 40 subtests, with no failures.** The adapter wheel also built successfully.

The test suite includes complex-DFT projection comparisons for odd/even and one-token chunks; literal cutoff-zero cases; separate K/V norms; layer-sum normalization; all-layer aggregation; per-chunk budgets and stable ties; context-only collection; query exclusion; pinned-memory transfer and stream ordering; original-position causal/sliding-window attention; native registry isolation; and BOS/no-BOS public-runner construction.

The public generic runner was also run on the actual model in both fixed ratio and measured golden-section-search modes. Fixed mode completed at ratio 0.15 with repeat agreement. GSS completed six real objective evaluations and measured its returned ratio. The search used a deliberately small smoke-test bracket, not a benchmark calibration or a claimed deployment optimum. See `validation/summary.json` for the machine-readable record.

## Reproduce

Install the package as described in the README, select a suitable CUDA toolkit when JIT compilation is needed, then run:

```bash
python -m pytest vllm_0_17_1/tests -q
python vllm_0_17_1/tests/integration_paper.py \
  --model Qwen/Qwen3-0.6B --output-json dense_validation.json
python vllm_0_17_1/tests/integration_paper.py \
  --model Qwen/Qwen3.5-0.8B --output-json hybrid_validation.json
```

For 7B/8B models on a 40GB A100, the integration runs used `--gpu-memory-utilization 0.65`; the smaller models used 0.30. The script's input text is included in its source and no private benchmark dataset is required. The recorded source hashes bind these results to this implementation; they are not evidence of continuous batching, multi-GPU execution or task accuracy.
