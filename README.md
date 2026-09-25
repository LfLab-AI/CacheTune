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

## Optional ICLR 2027 paper method

The commands above keep their existing behavior. All nine `blend_*freq*.py`
drivers also provide an explicit `--method paper` route. Omitting `--method`, or
passing `--method legacy`, runs the original implementation. The original code
inside each driver is retained. `run_paper_method()` in the same file delegates
the additional route to the shared `paper_runner.py`; `paper_dispatch.py` selects
the route before the original model imports.

From the directory containing the example scripts (`CacheTune/example` in the
server checkout), inspect the additional options without loading vLLM:

```bash
python blend_samsum_freq.py --method paper --help
```

Run a fixed 15% recomputation budget with the paper's frequency ranking:

```bash
python blend_samsum_freq.py --method paper \
  --model-path /path/to/Mistral-7B-Instruct-v0.3 \
  --dataset-path inputs/samsum.json \
  --ratio-mode fixed --recomp-ratio 0.15 --alpha 0.5 \
  --output-json paper_samsum_fixed.json
```

Calibrate on the first 10 SAMSum requests, then evaluate HotpotQA with Qwen:

```bash
CUDA_VISIBLE_DEVICES=0,1 python blend_hotpotqa_freq_qwen.py --method paper \
  --model-path /path/to/Qwen2.5-32B-Instruct \
  --tensor-parallel-size 2 --gpu-memory-utilization 0.85 \
  --max-model-len 8192 --enforce-eager \
  --dataset-path inputs/hotpotqa.json \
  --ratio-mode gss --calibration-dataset samsum \
  --calibration-dataset-path inputs/samsum.json --calibration-samples 10 \
  --output-json paper_hotpotqa_gss.json
```

Use the existing SSD driver for a disk cache pool, or select `--storage disk`
with any of the other drivers:

```bash
python blend_samsum_freq_SSD.py --method paper \
  --model-path /path/to/Mistral-7B-Instruct-v0.3 \
  --dataset-path inputs/samsum.json \
  --calibration-dataset-path inputs/samsum.json \
  --disk-root /path/on/the/target/ssd/cachetune-paper \
  --output-json paper_samsum_disk.json
```

All Qwen and non-Qwen drivers use the same additional method. The SSD driver
defaults to `--storage disk`; the other eight default to `--storage cpu`.
Dataset files and model weights remain deployment inputs.

The additional route also requires the matching runtime files in this checkout
(`worker/cachetune_paper.py` and `attention/paper_attention.py`, with their
additive hooks). Build/install this checkout's `vllm_blend` first. If another
editable vLLM checkout is installed, select this built source explicitly from
`CacheTune/example` before running the paper commands:

```bash
export PYTHONPATH="$(pwd)/../vllm_blend${PYTHONPATH:+:$PYTHONPATH}"
```

The new runner selects XFormers and checks that the installed worker provides
the paper methods. Query positions use an absolute-position causal mask; the
legacy bottom-right mask is retained only on the legacy route. The current
paper mask uses O(selected tokens * total tokens) memory, shared across heads
and reused across layers.

### Frequency ranking and calibration

The added ranking follows Section 4.1, equations (1)–(5), of the supplied
`CacheTune ICLR2027.pdf`. For each independently encoded chunk, it applies rFFT
along the token dimension to both pre-RoPE Keys and Values, retains the first
`floor(alpha * (N // 2 + 1))` frequency bins, and reconstructs exactly `N` token
positions using irFFT. A token receives the mean of its reconstructed Key and
Value L2 norms. Scores are normalized within each layer and averaged across all
layers. One resulting ranking supplies the same selected token positions to
every layer, and the complementary positions specify the reused KV entries.
Changing the recomputation ratio uses the stored ranking without rerunning FFT.

`--ratio-mode gss` is the new route's default. Calibration profiles computation
and the selected cache path, clips `ti / (tc + ti)` to the search interval, and
evaluates the warm-start probes against measured mean TTFT on the same
calibration requests. After the first interval reduction, the implementation
reinitializes both standard golden-section probes as specified in Appendix C,
Algorithm 1. Subsequent iterations reuse one probe measurement and return the
final interval midpoint. The paper's 30.9% and 36.4% ratios are hardware-specific
results, not preset answers for this implementation.

### Additional command-line options

| Option | Meaning or default on the paper route |
| --- | --- |
| `--dataset-path` | Override the evaluation dataset path. |
| `--dataset-target-path` | Reference-target file for a separate Multi-News source/target input. |
| `--ratio-mode fixed\|gss` | Fixed budget or measured calibration; default `gss`. |
| `--recomp-ratio` | Fixed recomputation ratio; default `0.15`. |
| `--alpha` | Fraction of low-frequency bins retained; default `0.5`. |
| `--r-min`, `--r-max` | GSS bounds; defaults `0.15`, `1.0`. |
| `--gss-tolerance` | Stop when the ratio interval is narrower than this value; default `0.01`. |
| `--calibration-dataset` | Calibration task; default `samsum`, independently of the evaluation driver. |
| `--calibration-dataset-path` | Override calibration input, including SAMSum calibration from another task's driver. |
| `--calibration-dataset-target-path` | Separate reference-target file when calibrating with Multi-News. |
| `--calibration-samples` | Number of calibration samples; default `10`. |
| `--warmup`, `--repeats` | Warmup and measured repetition counts; defaults `1`, `1`. |
| `--profile-repeats` | Hardware profile repetition count; default `3`. |
| `--storage cpu\|disk` | CPU or disk-backed reusable KV storage. |
| `--disk-root` | Directory on the intended storage medium for disk-backed caches. |
| `--output-json` | Save the run configuration, calibration measurements, and results. |
| `--sample-limit` | Limit evaluation samples for an initial deployment check. |
| `--model-path` | Model weights path or identifier. |
| `--tensor-parallel-size` | Number of tensor-parallel workers. |
| `--gpu-memory-utilization` | vLLM GPU memory allocation setting. |
| `--max-model-len` | Model context limit. |
| `--max-context-tokens` | Limit the input context used by the example. |
| `--max-tokens` | Maximum generated output tokens. |
| `--enforce-eager` | Use eager model execution. |

The paper specifies `alpha = 0.5`, a 15% quality-preserving lower bound, and the
first 10 SAMSum calibration samples. It does not specify a numeric GSS tolerance,
the exact upper bound below or equal to 1, profiling repetition counts, warmups,
or an integer rounding convention for `r * N`. The defaults exposed here for
these details are implementation choices and should be recorded with results.

For the disk route, the selected filesystem, host page cache, and warmup policy
affect measured access costs. A warmed OS page cache is not a cold-disk
measurement. Record the actual storage and cache state when reporting calibrated
ratios or latency, and rerun calibration after changing hardware or storage.

The computation prior uses median full-prefill TTFT divided by layers and
tokens as an amortized `tc` estimate; it includes fixed overhead rather than
identifying that overhead separately. `ti` measures one layer's actual cache
read and DMA path, divided by cached context tokens; the slowest TP rank supplies
the prior. GSS itself uses measured request TTFT, and the returned midpoint is
measured explicitly for reporting. Profiling choices are recorded in JSON.

TTFT is measured from first scheduling to first output token. Offline FFT,
per-ratio compaction or file preparation, and GPU buffer allocation occur before
that interval; actual layer reads, transfers and attention occur inside it.
Calibration retains immutable full CPU caches to replay the same requests.
Disk runs validate disk-backed layer reads while retaining those CPU snapshots;
they do not demonstrate SSD-only host-memory savings. Validate quality and
performance on the intended complete workload before reporting paper results.

The QA smoke evaluator reports normalized model-token F1 without special tokens,
which is tokenizer-dependent. Summarization reports ROUGE-L; Multi-News also
reports ROUGE-1/2. These are recorded in the JSON protocol. Use the benchmark's
official evaluator for a final cross-system quality comparison.

SAMSum calibration uses the first 10 records by default, as in the paper.
SAMSum evaluation starts at the first record too, so these input requests may
overlap. JSON records the overlap by token hash. Supply a separate calibration
file when a disjoint deployment validation split is required.

Run the focused tests from `CacheTune` with `python -m unittest discover -s tests
-p 'test_paper*.py' -v`. CUDA worker-state tests skip on CPU-only machines.
