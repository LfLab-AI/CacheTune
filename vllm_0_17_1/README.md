# CacheTune for vLLM 0.17.1

This directory contains a plugin and worker extension for **official vLLM 0.17.1**.

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



## Qwen3.5 hybrid attention

Use the same installed package; the adapter detects the dense text model configuration. The implementation is located in `cachetune/hybrid.py` and `cachetune/hybrid_state.py`.

```bash
bash vllm_0_17_1/run_qwen35_samsum15.sh \
  --model Qwen/Qwen3.5-0.8B \
  --dataset /path/to/samsum.json --sample-limit 2 \
  --output-json qwen35_smoke.json
```


## License

`LICENSE` retains the upstream Apache License 2.0 from the legacy vLLM fork. The adapter builds on [vLLM 0.17.1](https://github.com/vllm-project/vllm/tree/v0.17.1). Keep the upstream license and notices when redistributing source.

