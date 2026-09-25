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
- `example/inputs/`: local benchmark input files.
- `vllm_blend/`: the modified vLLM runtime used by CacheTune.

## Notes

The `vllm_blend/` directory contains modified vLLM code and keeps its upstream license files and third-party notices. Generated logs, temporary files, pytest caches, and disk offload artifacts are ignored by default.


## K/V spectral selection and latency calibration

All nine frequency examples support `--method spectral`. Omitting `--method`
keeps the existing behavior; `--method legacy` selects it explicitly. The
additional selector analyzes both K and V across every layer of each reusable
chunk. The original implementations remain in their existing files.

From `CacheTune/example`, select this checkout's built runtime if a different
editable vLLM installation is active:

```bash
export PYTHONPATH="$(pwd)/../vllm_blend${PYTHONPATH:+:$PYTHONPATH}"
python blend_samsum_freq.py --method spectral --help

# Fixed recomputation budget
python blend_samsum_freq.py --method spectral \
  --model-path /path/to/Mistral-7B-Instruct-v0.3 \
  --ratio-mode fixed --recomp-ratio 0.15 --enforce-eager

# Calibrate the budget with measured request latency
python blend_samsum_freq.py --method spectral \
  --model-path /path/to/Mistral-7B-Instruct-v0.3 \
  --ratio-mode gss --calibration-samples 10 --enforce-eager

# Qwen tensor parallel execution
CUDA_VISIBLE_DEVICES=0,1 python blend_hotpotqa_freq_qwen.py --method spectral \
  --model-path /path/to/Qwen2.5-32B-Instruct --tensor-parallel-size 2 \
  --dataset-path inputs/hotpotqa.json --enforce-eager
```

`blend_samsum_freq_SSD.py` defaults to `--storage disk` on this route. The other
eight examples default to `--storage cpu`; either setting is available from any
example. Set `--disk-root` to a directory on the intended storage medium.
Build/install this checkout's `vllm_blend` before running: the new worker and
attention helpers are required alongside the example modules.

### Selection algorithm

Each independently encoded chunk supplies pre-RoPE K and V. For each layer,
rFFT transforms the token axis, the first `floor(alpha*(N//2+1))` bins are kept,
and irFFT reconstructs exactly N positions. The token score is the mean of its
reconstructed K and V L2 norms. Each layer's scores are normalized by their
sum plus 1e-12, then averaged across layers. Tensor-parallel squared norms are
reduced before taking square roots, accounting for replicated KV heads.

Select the top `floor(r*N)` positions per chunk, with stable token-index tie
breaking. All layers share the selected positions and load their complement.
Changing r reuses the same ranking. Query tokens are neither scored nor loaded;
they are computed at runtime. Selected queries use absolute-position causal
attention. Its tensor bias uses O(selected tokens * total tokens) storage,
shared across heads and reused across layers. The legacy attention path remains
unchanged.

### Calibration

The default `--ratio-mode gss` initializes a bounded search from
`ti/(tc+ti)`. Compare the warm-start probes, reduce the interval, then initialize
two standard golden-section probes. Subsequent steps reuse one observation.
The final interval midpoint is measured explicitly before reporting its TTFT.
Each candidate replays the same calibration requests and cached rankings.

`tc` is estimated by amortizing median full-prefill TTFT over layers and tokens;
it includes fixed overhead. `ti` measures one layer's actual read and DMA path,
divided by reusable tokens. The slowest TP rank supplies the transfer prior.
These estimates initialize the search; measured mean TTFT is its objective.

| Option | Default or purpose |
| --- | --- |
| `--alpha` | Low-frequency bin fraction, 0.5. |
| `--ratio-mode fixed\|gss` | Fixed budget or latency calibration; gss. |
| `--recomp-ratio` | Fixed ratio, 0.15. |
| `--r-min`, `--r-max` | Search bounds, 0.15 and 1.0. |
| `--gss-tolerance` | Final interval width threshold, 0.01. |
| `--calibration-dataset` | Calibration task, samsum. |
| `--calibration-samples` | First 10 records of the calibration input. |
| `--dataset-path`, `--calibration-dataset-path` | Evaluation and calibration inputs. |
| `--dataset-target-path`, `--calibration-dataset-target-path` | Separate Multi-News target files. |
| `--warmup`, `--repeats`, `--profile-repeats` | 1, 1, and 3. |
| `--storage cpu\|disk`, `--disk-root` | Cache medium and disk directory. |
| `--output-json` | Configuration, measurements, selected indices, and results. |
| `--sample-limit` | Evaluation record limit, 200. |
| `--max-context-tokens`, `--max-tokens` | Context and generation limits. |

Use `--help` for model, tensor-parallel, memory, and context-window options.
The numerical defaults are configurable deployment choices. SAMSum evaluation
and calibration may use overlapping inputs; JSON records the token-hash
intersection. Supply separate files when disjoint validation is required.

TTFT runs from first scheduling to first output token. Offline FFT, cache
compaction/serialization and work-buffer allocation occur before this interval;
layer reads, transfers and attention occur inside it. Disk reads use the current
OS page cache. Immutable CPU snapshots are retained for replay, so this runner
does not measure an SSD-only host-memory footprint. Repeat calibration when
hardware, storage, or workload changes; noisy latency need not be unimodal.

QA evaluation reports normalized model-token F1 without special tokens.
Summarization reports ROUGE-L, plus ROUGE-1/2 for Multi-News. Use an official
benchmark evaluator for final cross-system quality comparisons.

Run focused tests from `CacheTune`:

```bash
python -m unittest discover -s tests -p 'test_spectral*.py' -v
```

CUDA worker-state tests skip on CPU-only machines.
