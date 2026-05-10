# CacheTune

CacheTune is a KV cache reuse system for efficient long-context LLM inference. It combines frequency-domain token selection, hardware-aware recomputation scheduling, CPU pinned-memory offload, and asynchronous layer-wise transfer to reduce time to first token while preserving generation quality.

Author: lifei

## Highlights

- Frequency-domain token scoring selects the context tokens that should be recomputed.
- Hardware-aware scheduling balances PCIe transfer cost and GPU recomputation cost.
- Asynchronous pipeline overlap hides KV transfer behind attention computation.

## Installation
The implementation is based on [vLLM](https://github.com/vllm-project/vllm)
`Python >= 3.9` and `CUDA >= 12.1` are recommended. A GPU with at least 40 GB of memory is preferred for the largest examples.

```bash
cd vllm_blend
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
CUDA_VISIBLE_DEVICES=0,1 python blend_samsum_freq_qwen.py --model-path Qwen2.5-32B-Instruct --tensor-parallel-size 2 --gpu-memory-utilization 0.85 --max-model-len 8192 --enforce-eager
```

Running hardware-aware adaptive recomputation ratio analysis:

```bash
python blend_samsum_freq_SSD.py
```

## Repository Layout

- `example/`: experiment drivers, benchmark scripts, and utility code.
- `inputs/`: local benchmark input files.
- `vllm_blend/`: the modified vLLM runtime used by CacheTune.

## Notes

The `vllm_blend/` directory contains modified vLLM code and keeps its upstream license files and third-party notices. Generated logs, temporary files, pytest caches, and disk offload artifacts are ignored by default.
