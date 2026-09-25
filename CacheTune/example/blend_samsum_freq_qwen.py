"""
CacheTune for Qwen2 on SAMSum Dataset
Modified from blend_samsum_freq.py to support Qwen2 models
"""

# BEGIN CACHETUNE SPECTRAL DISPATCH
def run_spectral_method(argv=None):
    """Run K/V spectral selection; the original path remains the default."""
    from spectral_runner import main
    return main(dataset='samsum', default_storage='cpu',
                is_qwen=True, argv=argv)


if __name__ == "__main__":
    from spectral_dispatch import dispatch_if_requested as _dispatch_spectral_method
    _dispatch_spectral_method(run_spectral_method)
# END CACHETUNE SPECTRAL DISPATCH


from vllm import LLM, SamplingParams
import torch
import json
import numpy as np
import argparse
from transformers import AutoTokenizer, AutoConfig
from utils import load_dataset, normalize_question, build_fewshot_prompt, compute_rl
from pathlib import Path
from itertools import chain
import time


# --- Hardware Profiling: Measure PCIe bandwidth and GPU compute time per token per layer ---
def profile_hardware(num_kv_heads=8, head_dim=128, h=3584, p=2, num_trials=5):
    """
    Measure v_com (PCIe bandwidth, GB/s) and t_gpu_ms_per_token (GPU compute time per layer per token, ms)

    Note: Qwen2-7B uses head_dim=128, hidden_size=3584, num_kv_heads=8
    """
    print("\n[Hardware Profiling] Measuring v_com (PCIe) and t_gpu_ms_per_token (GPU compute)...")

    # --- Measure v_com (PCIe bandwidth, GB/s) ---
    test_N = 2048
    total_bytes = 2 * test_N * h * p  # K+V bytes (p=2 for bfloat16)
    test_tensor = torch.zeros(2 * test_N * h, dtype=torch.bfloat16).pin_memory()

    transfer_times = []
    for _ in range(num_trials):
        torch.cuda.synchronize()
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        start_ev.record()
        gpu_t = test_tensor.to("cuda", non_blocking=True)
        end_ev.record()
        torch.cuda.synchronize()
        transfer_times.append(start_ev.elapsed_time(end_ev))  # ms
        del gpu_t

    t_ms = np.median(transfer_times)
    v_com = (total_bytes / (t_ms / 1000.0)) / 1e9  # GB/s

    # --- Measure t_gpu_ms_per_token (GPU compute latency per token per layer, ms) ---
    # Use QKV projection as proxy: A(ref_N, h) @ W(h, h+2*kv_heads*head_dim)
    ref_N = 512
    A = torch.randn(ref_N, h, device="cuda", dtype=torch.bfloat16)
    W = torch.randn(h, h + 2 * num_kv_heads * head_dim, device="cuda", dtype=torch.bfloat16)
    for _ in range(3):
        _ = A @ W
    torch.cuda.synchronize()

    compute_times = []
    for _ in range(num_trials):
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        start_ev.record()
        _ = A @ W
        end_ev.record()
        torch.cuda.synchronize()
        compute_times.append(start_ev.elapsed_time(end_ev))  # ms

    t_ms_gpu = np.median(compute_times)
    t_gpu_ms_per_token = t_ms_gpu / ref_N  # ms per token per layer
    print(f"[Profiling Result] v_com={v_com:.2f} GB/s, t_gpu_ms_per_token={t_gpu_ms_per_token:.5f} ms")
    return v_com, t_gpu_ms_per_token


def optimal_l(N, num_kv_heads=8, head_dim=128, v_com=10.0, t_gpu_ms_per_token=0.001, p=2, r_min=0.15, r_max=0.50):
    """
    Use LP grid search to compute optimal recompute token count l (I/O-aware strategy).
    """
    kv_bytes_per_token = 2 * num_kv_heads * head_dim * p  # K+V bytes per token (bfloat16)
    v_com_Bps = v_com * 1e9  # bytes/s
    min_l = max(1, int(r_min * N))
    max_l = max(min_l, min(N, int(r_max * N)))  # 50% cap
    step = max(1, N // 100)

    def t_layer(l):
        t_compute = l * t_gpu_ms_per_token  # ms
        t_transfer = (N - l) * kv_bytes_per_token / v_com_Bps * 1000  # ms
        return max(t_compute, t_transfer)

    best_l, min_t = min_l, t_layer(min_l)
    for l in range(min_l + step, max_l + 1, step):
        t = t_layer(l)
        if t < min_t:
            min_t, best_l = t, l

    print(f"[optimal_l] N={N}, best_l={best_l} ({best_l/N:.1%}), "
          f"est. t_layer={min_t:.3f} ms, cap={int(r_max*100)}%")
    return best_l


def calculate_freq_indices(key, value, ratio, low_freq_pct=0.50, sink_size=0):
    """
    Perform frequency domain analysis on KV Cache, return indices of tokens to recompute.
    """
    v_float = value.to(torch.float32)
    seq_len = v_float.shape[0]

    total_budget = int(seq_len * ratio)
    sink_indices = torch.arange(sink_size, device=value.device)

    v_to_analyze = v_float[sink_size:]
    analyze_len = v_to_analyze.shape[0]

    if analyze_len == 0:
        return sink_indices

    v_freq = torch.fft.rfft(v_to_analyze, dim=0)

    cutoff = int(v_freq.shape[0] * low_freq_pct)

    if cutoff < v_freq.shape[0]:
        if v_freq.ndim == 3:
            v_freq[cutoff:, :, :] = 0.0
            norm_dims = (1, 2)
        elif v_freq.ndim == 2:
            v_freq[cutoff:, :] = 0.0
            norm_dims = (1,)
        else:
            sl = [slice(None)] * v_freq.ndim
            sl[0] = slice(cutoff, None)
            v_freq[tuple(sl)] = 0.0
            norm_dims = tuple(range(1, v_freq.ndim))

    v_low_reconstructed = torch.fft.irfft(v_freq, n=analyze_len, dim=0)
    imp_scores = torch.norm(v_low_reconstructed, p=2, dim=norm_dims)

    freq_budget = total_budget
    top_indices_local = torch.topk(imp_scores, k=freq_budget).indices
    top_indices_global = top_indices_local + sink_size

    final_indices = torch.cat([sink_indices, top_indices_global])
    final_indices, _ = torch.sort(final_indices)

    return final_indices.to(torch.int64).to(value.device)


def get_driver_model(llm_obj):
    """Return driver-side model object for both single-GPU and Ray TP modes."""
    executor = llm_obj.llm_engine.model_executor
    driver_worker = executor.driver_worker
    if hasattr(driver_worker, "model_runner"):
        return driver_worker.model_runner.model.model
    if hasattr(driver_worker, "worker") and driver_worker.worker is not None:
        return driver_worker.worker.model_runner.model.model
    raise RuntimeError("Unsupported driver worker type: cannot access model object.")


def run_on_all_workers(llm_obj, method, **kwargs):
    """Run a worker method on all TP ranks (driver + remote workers)."""
    executor = llm_obj.llm_engine.model_executor
    if hasattr(executor, "_run_workers"):
        return executor._run_workers(method, **kwargs)

    driver_worker = executor.driver_worker
    if hasattr(driver_worker, "execute_method"):
        return [driver_worker.execute_method(method, **kwargs)]
    return [getattr(driver_worker, method)(**kwargs)]


parser = argparse.ArgumentParser(description="CacheTune Qwen inference script")
parser.add_argument("--model-path", type=str,
                    default="/path/model/Qwen2.5-32B-Instruct",
                    help="Local path of Qwen model")
parser.add_argument("--tensor-parallel-size", type=int, default=2,
                    help="vLLM tensor parallel size (set 2 for dual-GPU)")
parser.add_argument("--gpu-memory-utilization", type=float, default=0.90,
                    help="Per-GPU memory utilization ratio for vLLM")
parser.add_argument("--max-model-len", type=int, default=4096,
                    help="vLLM max_model_len. Use 4096 for SAMSum on 32B to fit KV cache.")
parser.add_argument("--enforce-eager", action="store_true",
                    help="Disable CUDA graph capture to avoid long warmup / graph-memory overhead.")
parser.add_argument("--max-num-seqs", type=int, default=1,
                    help="vLLM max_num_seqs. Keep 1 for single-prompt script to reduce memory.")
args = parser.parse_args()

# Load dataset
eval_dataset = load_dataset("inputs/samsum.json")

# Initialize Qwen model
model_path = args.model_path
print(f"Loading model from {model_path}...")
print(f"Tensor parallel size: {args.tensor_parallel_size}")
print(f"GPU memory utilization: {args.gpu_memory_utilization}")
print(f"Max model len: {args.max_model_len}")
print(f"Enforce eager: {args.enforce_eager}")
print(f"Max num seqs: {args.max_num_seqs}")
llm = LLM(
    model=model_path,
    tensor_parallel_size=args.tensor_parallel_size,
    gpu_memory_utilization=args.gpu_memory_utilization,
    max_model_len=args.max_model_len,
    enforce_eager=args.enforce_eager,
    max_num_seqs=args.max_num_seqs,
    disable_custom_all_reduce=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_path)
llm.set_tokenizer(tokenizer)

prefix_prompt = "Summarize the dialogue into a few short sentences. The following are some examples.\n\n"

ttft_blend = []
ttft_full = []
rl_blend = []
rl_full = []

max_ctx_len = 3400

# [Hardware Profiling]: Only execute once before first sample
hardware_profiled = False
v_com = 10.0
t_gpu_ms_per_token = 0.001

# === AUTO-DETECT QWEN2.5 PARAMETERS ===
config = AutoConfig.from_pretrained(model_path)

qwen2_num_kv_heads = config.num_key_value_heads
qwen2_head_dim = config.hidden_size // config.num_attention_heads
qwen2_hidden_size = config.hidden_size
num_qwen2_layers = config.num_hidden_layers

# In tensor parallel, each rank holds a shard with local KV heads.
driver_model = get_driver_model(llm)
local_num_kv_heads = driver_model.layers[0].self_attn.num_kv_heads

print(f"\n[Auto-Detect] Qwen Model Config:")
print(f"  num_layers: {num_qwen2_layers}")
print(f"  hidden_size: {qwen2_hidden_size}")
print(f"  num_kv_heads: {qwen2_num_kv_heads}")
print(f"  local_num_kv_heads (per rank): {local_num_kv_heads}")
print(f"  head_dim: {qwen2_head_dim}\n")

for sample_idx, ex in enumerate(eval_dataset):
    # [Step 0]: Hardware Profiling (first execution)
    if not hardware_profiled:
        v_com, t_gpu_ms_per_token = profile_hardware(
            num_kv_heads=local_num_kv_heads,
            head_dim=qwen2_head_dim,
            h=qwen2_hidden_size,
            p=2
        )
        hardware_profiled = True
        print(f"\n{'='*60}")
        print(f"[Profiling] v_com={v_com:.2f} GB/s, t_gpu_ms_per_token={t_gpu_ms_per_token:.5f} ms")
        print(f"{'='*60}\n")

    answers = ex["answers"]
    doc_prompts, q_prompt = build_fewshot_prompt(ex)
    doc_chunk_ids = [tokenizer.encode(doc) for doc in doc_prompts]
    q_ids = tokenizer.encode(q_prompt)

    # Drop last few-shot examples if exceeding max_ctx_len
    while len(list(chain.from_iterable(doc_chunk_ids))) > max_ctx_len:
        del_idx = int(len(doc_chunk_ids) / 2)
        del doc_chunk_ids[del_idx]

    if len(doc_chunk_ids) == 0:
        continue

    sampling_params = SamplingParams(temperature=0, max_tokens=1)

    # Metadata setup (all ranks)
    run_on_all_workers(llm, "cachetune_reset_state")
    run_on_all_workers(
        llm,
        "cachetune_set_flags",
        updates={"collect": False, "check": False, "attn_bias": None},
        reset_pipeline=True,
    )

    s_start_full = tokenizer.encode(prefix_prompt)
    s_start_len = len(s_start_full)

    s_start = []
    s_start_1_len = len(s_start)

    s_end = []
    s_end_len = len(s_end)

    doc_chunk_ids = [s_start + chunk_ids for chunk_ids in doc_chunk_ids]
    doc_chunk_ids = [s_start_full] + doc_chunk_ids
    doc_chunk_ids = doc_chunk_ids + [s_start + q_ids + s_end]

    last_len = len(q_ids + s_end)

    run_on_all_workers(
        llm,
        "cachetune_set_flags",
        updates={"collect": True, "check": False},
        reset_pipeline=True,
    )

    # --- Step 1: Generate and concatenate KV Cache (CPU Offload) ---
    print(f"Sample {sample_idx}: Generating KV Cache with CPU Offload...")
    for i in range(len(doc_chunk_ids)):
        llm.generate(prompt_token_ids=[doc_chunk_ids[i]], sampling_params=sampling_params)
        run_on_all_workers(
            llm,
            "cachetune_append_chunk_kv",
            chunk_len=len(doc_chunk_ids[i]),
            s_start_len=s_start_len,
            s_start_1_len=s_start_1_len,
            is_first_chunk=(i == 0),
        )

    rank_infos = run_on_all_workers(llm, "cachetune_finalize_chunk_kv")
    llm_model = get_driver_model(llm)
    cache_fuse_metadata = llm_model.cache_fuse_metadata
    chunk_past_key_values = cache_fuse_metadata["cpu_kv_cache"]

    # --- Step 2: Frequency Domain Analysis ---
    print(f"Sample {sample_idx}: Analyzing Frequency Domain to find low-freq tokens...")
    precomputed_indices_list = []
    total_tokens_before_compact = chunk_past_key_values[0][0].shape[0]
    N_context = total_tokens_before_compact - last_len
    if N_context <= 0:
        print(f"[WARN] Sample {sample_idx}: invalid context length (N_context={N_context}, total={total_tokens_before_compact}, last_len={last_len}), skip CacheTune.")
        continue

    l_total = optimal_l(
        N_context,
        num_kv_heads=local_num_kv_heads,
        head_dim=qwen2_head_dim,
        v_com=v_com,
        t_gpu_ms_per_token=t_gpu_ms_per_token
    )
    l_total = max(1, min(N_context, l_total))
    recomp_ratio = 0.15##l_total / N_context
    sink_n = 0

    # Use one global index set (from layer 0) to keep status=1/status=2 semantics consistent.
    layer0_v = chunk_past_key_values[0][1].to(torch.device("cuda", torch.cuda.current_device()))
    layer0_k = chunk_past_key_values[0][0].to(torch.device("cuda", torch.cuda.current_device()))
    context_v = layer0_v[:-last_len]

    indices = calculate_freq_indices(
        layer0_k,
        context_v,
        ratio=recomp_ratio,
        low_freq_pct=0.50,
        sink_size=sink_n
    )

    total_len = layer0_v.shape[0]
    suffix_start = total_len - last_len
    if suffix_start < 0:
        raise ValueError(
            f"Invalid suffix range at sample {sample_idx}: "
            f"total_len={total_len}, last_len={last_len}")
    suffix_indices = torch.arange(suffix_start, total_len, device=indices.device)
    final_indices = torch.cat([indices, suffix_indices])
    final_indices = torch.unique(final_indices, sorted=True)

    if final_indices.numel() == 0:
        raise ValueError(f"Empty recompute indices at sample {sample_idx}")
    if final_indices.min().item() < 0 or final_indices.max().item() >= total_len:
        raise ValueError(
            f"Out-of-range recompute indices at sample {sample_idx}: "
            f"min={final_indices.min().item()}, max={final_indices.max().item()}, total_len={total_len}")

    precomputed_indices_list.append(final_indices)
    imp_indices_cpu = final_indices.to("cpu")

    run_on_all_workers(
        llm,
        "cachetune_prepare_from_indices",
        final_indices_cpu=imp_indices_cpu,
        last_len=last_len,
        recomp_ratio=float(recomp_ratio),
        check_layers=[1],
    )

    input_ids = []
    for i in range(len(doc_chunk_ids)):
        if i == 0:
            temp_ids = doc_chunk_ids[i]
        else:
            temp_ids = doc_chunk_ids[i][s_start_1_len:]
        input_ids += temp_ids

    input_prompt = tokenizer.decode(input_ids)

    # --- Step 3: Setup inference parameters ---
    sampling_params = SamplingParams(temperature=0, max_tokens=128)

    print(f"Sample idx: {sample_idx}")

    # Generate with CacheTune
    output = llm.generate(prompt_token_ids=[input_ids], sampling_params=sampling_params)
    res = output[0].outputs[0].text
    res = res.lstrip('\n').split('\n')[0]
    print(f"Cached generation (CacheTune): {res}")
    ttft = output[0].metrics.first_token_time - output[0].metrics.first_scheduled_time
    print(f"TTFT with cache: {ttft:.4f}")
    ttft_blend.append(ttft)
    rl = max([compute_rl(res, answer) for answer in answers])
    rl_blend.append(rl)

    # Reset pipeline on all ranks before baseline pass.
    run_on_all_workers(
        llm,
        "cachetune_set_flags",
        updates={"check": False, "collect": False, "pipeline_enabled": False},
        reset_pipeline=True,
    )

    # Generate baseline
    sampling_params = SamplingParams(temperature=0, max_tokens=128)
    output = llm.generate(prompt_token_ids=[input_ids], sampling_params=sampling_params)
    res = output[0].outputs[0].text
    res = res.lstrip('\n').split('\n')[0]
    print(f"Normal generation: {res}")
    ttft = output[0].metrics.first_token_time - output[0].metrics.first_scheduled_time
    print(f"TTFT with full prefill: {ttft}")
    ttft_full.append(ttft)
    rl = max([compute_rl(res, answer) for answer in answers])
    rl_full.append(rl)

    run_on_all_workers(
        llm,
        "cachetune_set_flags",
        updates={
            "pipeline_enabled": False,
            "cpu_kv_cache": None,
            "non_imp_indices": None,
            "precomputed_indices": None,
            "work_key": None,
            "work_val": None,
        },
        reset_pipeline=True,
    )

    print("------------")


print("---------------Result Summary---------------------")
print(f"[Profiling] v_com={v_com:.2f} GB/s, t_gpu_ms_per_token={t_gpu_ms_per_token:.5f} ms")
print(f"TTFT with cache: {np.mean(ttft_blend):.4f} seconds")
print(f"TTFT with full prefill: {np.mean(ttft_full):.4f} seconds")
print(f"Speedup: {np.mean(ttft_full) / np.mean(ttft_blend):.2f}x")
print(f"RL with cache: {np.mean(rl_blend):.4f}")
print(f"RL with full prefill: {np.mean(rl_full):.4f}")
print(f"Accuracy Preservation: {(np.mean(rl_blend) / np.mean(rl_full) * 100):.2f}%")
