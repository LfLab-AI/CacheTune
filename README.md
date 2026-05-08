# CacheTune

CacheTune is a KV cache reuse system for efficient long-context LLM inference. It combines frequency-domain token selection, hardware-aware recomputation scheduling, CPU pinned-memory offload, and asynchronous layer-wise transfer to reduce time to first token while preserving generation quality.

Author: lifei

## Highlights

- Frequency-domain token scoring selects the context tokens that should be recomputed.
- Hardware-aware scheduling balances PCIe transfer cost and GPU recomputation cost.
- CPU offload stores reusable KV tensors in pinned memory for efficient DMA transfer.
- Asynchronous pipeline overlap hides KV transfer behind attention computation.
- Qwen2/Qwen2.5, Llama, Mistral, SAMSum, MuSiQue, WikiMQA, HotpotQA, and MultiNews examples are included.

## Installation

`Python >= 3.9` and `CUDA >= 12.1` are recommended. A GPU with at least 40 GB of memory is preferred for the largest examples.

```bash
cd vllm_blend
pip install -e .
cd ..
pip install -r requirements.txt
```

## Example Runs

Run the basic example:

```bash
python example/blend.py
```

Run SAMSum with frequency selection:

```bash
python example/blend_samsum_freq.py
```

Run Qwen on the supported benchmark scripts:

```bash
python example/blend_samsum_freq_qwen.py
python example/blend_musique_freq_qwen.py
python example/blend_wikimqa_freq_qwen.py
python example/blend_hotpotqa_freq_qwen.py
python example/blend_multinews_freq_qwen.py
```

## Repository Layout

- `example/`: experiment drivers, benchmark scripts, and utility code.
- `inputs/`: local benchmark input files.
- `vllm_blend/`: the modified vLLM runtime used by CacheTune.
- `method_report.md`: concise method description.
- `technical_report.md`: implementation-level technical notes.
- `optimization_summary.md`: optimization overview.

## Notes

The `vllm_blend/` directory contains modified vLLM code and keeps its upstream license files and third-party notices. Generated logs, temporary files, pytest caches, and disk offload artifacts are ignored by default.
