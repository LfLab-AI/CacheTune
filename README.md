# CacheTune



CacheTune is a KV cache reuse system for efficient long-context LLM inference. It combines frequency-domain token selection, hardware-aware recomputation scheduling, CPU pinned-memory offload, and asynchronous layer-wise transfer to reduce time to first token while preserving generation quality.



## Highlights

- Frequency-domain token scoring selects the context tokens that should be recomputed.
- Hardware-aware scheduling balances PCIe transfer cost and GPU recomputation cost.
- Asynchronous pipeline overlap hides KV transfer behind attention computation.

## Installation
The implementation is based on [vLLM](https://github.com/vllm-project/vllm)
`Python >= 3.9` and `CUDA >= 12.1` are recommended. A GPU with at least 40 GB of memory is preferred for the largest examples.

```bash
cd CacheTune/vllm_blend
pip install -e .
cd ..
pip install -r requirements.txt
```

## Example Runs

Run the basic example:


Run SAMSum with frequency selection:

```bash
python example/blend_samsum_freq.py
```

Run Qwen on the supported benchmark scripts:

```bash
CUDA_VISIBLE_DEVICES=0,1 python example/blend_samsum_freq_qwen.py --model-path Qwen2.5-32B-Instruct --tensor-parallel-size 2 --gpu-memory-utilization 0.85 --max-model-len 8192 --enforce-eager
```

Running hardware-aware adaptive recomputation ratio analysis:

```bash
python example/blend_samsum_freq_SSD.py
```

## Repository Layout

- `CacheTune/example/`: legacy experiment drivers, benchmark scripts, and utility code.
- `CacheTune/example/inputs/`: legacy local benchmark input files.
- `CacheTune/vllm_blend/`: the modified vLLM 0.4.1 runtime.

## Notes

The `CacheTune/vllm_blend/` directory contains modified vLLM code and keeps its upstream license files and third-party notices. Generated logs, temporary files, pytest caches, and disk offload artifacts are ignored by default.

## vLLM Versions and Qwen3.5

The two implementations are stored side by side and use separate Python environments.

| Implementation | Directory | Main entry points |
|---|---|---|
| Legacy vLLM **0.4.1** fork | [`CacheTune/vllm_blend/`](CacheTune/vllm_blend/) | [`CacheTune/example/`](CacheTune/example/); use `--method spectral` for the paper-aligned K/V, all-layer spectral path. Original legacy drivers remain available. |
| vLLM **0.17.1** plugin and worker extension | [`vllm_0_17_1/`](vllm_0_17_1/) | [`vllm_0_17_1/example/run_spectral.py`](vllm_0_17_1/example/run_spectral.py) for explicit chunks and a query; [`run_samsum.sh`](vllm_0_17_1/run_samsum.sh) for SAMSum. |
| Qwen3.5 dense text hybrid attention | [`vllm_0_17_1/cachetune/hybrid.py`](vllm_0_17_1/cachetune/hybrid.py), [`hybrid_state.py`](vllm_0_17_1/cachetune/hybrid_state.py) | [`vllm_0_17_1/run_qwen35_samsum15.sh`](vllm_0_17_1/run_qwen35_samsum15.sh); default policy: `block`. |

Install the new adapter from the repository root in its own environment:

```bash
python -m pip install -e './vllm_0_17_1[test]'
```

The new implementation uses official vLLM 0.17.1 and FlashInfer 0.6.4; it does not vendor another complete upstream vLLM tree. It scores independently encoded chunks using pre-RoPE K and V from all cached attention layers, shares one selected set across layers, and always computes the query suffix online. Qwen3.5 keeps all Gated DeltaNet recurrent updates on the complete ordered sequence and applies selective repair to its full-attention blocks. Its token budget therefore does not describe total model FLOPs.

See the [new-version README](vllm_0_17_1/README.md) for installation, supported execution modes and validation, and [METHOD.md](vllm_0_17_1/METHOD.md) for the correspondence to the paper. This port is a single-request eager implementation (TP=PP=1); using the new vLLM version does not establish production continuous-batching support.
