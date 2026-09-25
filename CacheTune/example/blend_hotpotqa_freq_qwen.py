"""
CacheTune for Qwen2.5 on HotpotQA dataset (TP-aware).
"""

# BEGIN CACHETUNE SPECTRAL DISPATCH
def run_spectral_method(argv=None):
    """Run K/V spectral selection; the original path remains the default."""
    from spectral_runner import main
    return main(dataset='hotpotqa', default_storage='cpu',
                is_qwen=True, argv=argv)


if __name__ == "__main__":
    from spectral_dispatch import dispatch_if_requested as _dispatch_spectral_method
    _dispatch_spectral_method(run_spectral_method)
# END CACHETUNE SPECTRAL DISPATCH


from vllm import LLM, SamplingParams
import torch
import numpy as np
import argparse
from pathlib import Path
from transformers import AutoTokenizer, AutoConfig
from utils import load_dataset, build_qa_prompt, compute_f1


def profile_hardware(num_kv_heads=8, head_dim=128, h=3584, p=2, num_trials=5):
    print("\n[Hardware Profiling] Measuring v_com (PCIe) and t_gpu_ms_per_token (GPU compute)...")

    test_N = 2048
    total_bytes = 2 * test_N * h * p
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
        transfer_times.append(start_ev.elapsed_time(end_ev))
        del gpu_t

    t_ms = np.median(transfer_times)
    v_com = (total_bytes / (t_ms / 1000.0)) / 1e9

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
        compute_times.append(start_ev.elapsed_time(end_ev))

    t_ms_gpu = np.median(compute_times)
    t_gpu_ms_per_token = t_ms_gpu / ref_N
    print(f"[Profiling Result] v_com={v_com:.2f} GB/s, t_gpu_ms_per_token={t_gpu_ms_per_token:.5f} ms")
    return v_com, t_gpu_ms_per_token


def optimal_l(N, num_kv_heads=8, head_dim=128, v_com=10.0, t_gpu_ms_per_token=0.001, p=2, r_min=0.15, r_max=0.50):
    kv_bytes_per_token = 2 * num_kv_heads * head_dim * p
    v_com_Bps = v_com * 1e9
    min_l = max(1, int(r_min * N))
    max_l = max(min_l, min(N, int(r_max * N)))
    step = max(1, N // 100)

    def t_layer(l):
        t_compute = l * t_gpu_ms_per_token
        t_transfer = (N - l) * kv_bytes_per_token / v_com_Bps * 1000
        return max(t_compute, t_transfer)

    best_l, min_t = min_l, t_layer(min_l)
    for l in range(min_l + step, max_l + 1, step):
        t = t_layer(l)
        if t < min_t:
            min_t, best_l = t, l

    print(f"[optimal_l] N={N}, best_l={best_l} ({best_l / N:.1%}), est. t_layer={min_t:.3f} ms, cap={int(r_max * 100)}%")
    return best_l


def get_optimized_indices(raw_v, N, ratio, low_freq_pct=0.70, sink_size=0):
    v_float = raw_v.to(torch.float32)
    seq_len = v_float.shape[0]

    total_budget = max(1, int(seq_len * ratio))

    actual_sink = min(sink_size, N)
    sink_indices = torch.arange(0, actual_sink, device=raw_v.device)

    v_to_analyze = v_float[actual_sink:]
    analyze_len = v_to_analyze.shape[0]

    if analyze_len == 0:
        recomp = sink_indices.to(torch.int64)
        mask = torch.ones(N, dtype=torch.bool, device=raw_v.device)
        mask[recomp] = False
        reuse = torch.arange(N, device=raw_v.device)[mask]
        return recomp, reuse

    v_freq = torch.fft.rfft(v_to_analyze, dim=0)
    cutoff = int(v_freq.shape[0] * low_freq_pct)

    if v_freq.ndim == 3:
        norm_dims = (1, 2)
    elif v_freq.ndim == 2:
        norm_dims = (1,)
    else:
        norm_dims = tuple(range(1, v_freq.ndim))

    if cutoff < v_freq.shape[0]:
        sl = [slice(None)] * v_freq.ndim
        sl[0] = slice(cutoff, None)
        v_freq[tuple(sl)] = 0.0

    v_low = torch.fft.irfft(v_freq, n=analyze_len, dim=0)
    imp_scores = torch.norm(v_low, p=2, dim=norm_dims)

    freq_budget = max(1, total_budget - actual_sink)
    freq_budget = min(freq_budget, analyze_len)

    top_indices_local = torch.topk(imp_scores, k=freq_budget).indices
    top_indices_global = top_indices_local + actual_sink

    final_recomp = torch.cat([sink_indices, top_indices_global]).unique().sort().values

    mask = torch.ones(N, dtype=torch.bool, device=raw_v.device)
    mask[final_recomp] = False
    final_reuse = torch.arange(N, device=raw_v.device)[mask]

    return final_recomp.to(torch.int64), final_reuse.to(torch.int64)


def get_driver_model(llm_obj):
    executor = llm_obj.llm_engine.model_executor
    driver_worker = executor.driver_worker
    if hasattr(driver_worker, "model_runner"):
        return driver_worker.model_runner.model.model
    if hasattr(driver_worker, "worker") and driver_worker.worker is not None:
        return driver_worker.worker.model_runner.model.model
    raise RuntimeError("Unsupported driver worker type: cannot access model object.")


def run_on_all_workers(llm_obj, method, **kwargs):
    executor = llm_obj.llm_engine.model_executor
    if hasattr(executor, "_run_workers"):
        return executor._run_workers(method, **kwargs)

    driver_worker = executor.driver_worker
    if hasattr(driver_worker, "execute_method"):
        return [driver_worker.execute_method(method, **kwargs)]
    return [getattr(driver_worker, method)(**kwargs)]


def safe_ttft(req_output):
    metrics = getattr(req_output, "metrics", None)
    if metrics is None:
        return None
    first_token_time = getattr(metrics, "first_token_time", None)
    first_scheduled_time = getattr(metrics, "first_scheduled_time", None)
    if first_token_time is None or first_scheduled_time is None:
        return None
    return first_token_time - first_scheduled_time


def encode_like_reference(text, tokenizer):
    ids = tokenizer.encode(text)
    bos_id = tokenizer.bos_token_id
    if bos_id is not None and len(ids) > 0 and ids[0] == bos_id:
        return ids[1:]
    return ids


def normalize_pred_for_eval(text):
    if text is None:
        return ""
    s = text.strip().split("\n")[0].strip()
    markers = [
        "Answer the question directly based on the given passages",
        "Do NOT repeat the question",
        "Question:",
        "Answer:",
    ]
    for marker in markers:
        if marker in s:
            s = s.split(marker)[0].strip()
    return s


def resolve_hotpot_dataset_path():
    candidates = [
        Path("inputs/hotpot_dev_distractor_v1.json"),
        Path("/path/dataset/HotpotQA/raw/hotpot_dev_distractor_v1.json"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[-1])


parser = argparse.ArgumentParser(description="CacheTune HotpotQA Qwen inference script")
parser.add_argument("--model-path", type=str, default="/path/model/Qwen2.5-32B-Instruct")
parser.add_argument("--tensor-parallel-size", type=int, default=2)
parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
parser.add_argument("--max-model-len", type=int, default=4096)
parser.add_argument("--enforce-eager", action="store_true")
parser.add_argument("--max-num-seqs", type=int, default=1)
parser.add_argument("--sample-limit", type=int, default=200)
parser.add_argument("--low-freq-pct", type=float, default=0.70)
parser.add_argument("--recomp-ratio", type=float, default=0.15)
parser.add_argument("--sink-size", type=int, default=0)
parser.add_argument("--use-pyramid", action="store_true")
parser.add_argument("--pyramid-interval", type=int, default=8)
parser.add_argument("--shrink-ratio", type=float, default=1.0)
args = parser.parse_args()


dataset_path = resolve_hotpot_dataset_path()
print(f"Loading dataset from {dataset_path}")
eval_dataset_raw = load_dataset(dataset_path)
eval_dataset = eval_dataset_raw[:args.sample_limit]

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

prefix_prompt = "You will be asked a question after reading several passages. Please directly answer the question based on the given passages. Do NOT repeat the question. The answer should be within 5 words.\nPassages:\n"
query_prompt = "\n\nAnswer the question directly based on the given passages. Do NOT repeat the question. The answer should be within 5 words.\nQuestion: "

ttft_blend = []
ttft_full = []
f1_blend = []
f1_full = []

hardware_profiled = False
v_com = 10.0
t_gpu_ms_per_token = 0.001

config = AutoConfig.from_pretrained(model_path)
qwen_num_kv_heads = config.num_key_value_heads
qwen_head_dim = config.hidden_size // config.num_attention_heads
qwen_hidden_size = config.hidden_size
num_layers = config.num_hidden_layers

driver_model = get_driver_model(llm)
local_num_kv_heads = driver_model.layers[0].self_attn.num_kv_heads

print("\n[Auto-Detect] Qwen Model Config:")
print(f"  num_layers: {num_layers}")
print(f"  hidden_size: {qwen_hidden_size}")
print(f"  num_kv_heads: {qwen_num_kv_heads}")
print(f"  local_num_kv_heads (per rank): {local_num_kv_heads}")
print(f"  head_dim: {qwen_head_dim}\n")

for sample_idx, ex in enumerate(eval_dataset):
    if not hardware_profiled:
        v_com, t_gpu_ms_per_token = profile_hardware(
            num_kv_heads=local_num_kv_heads,
            head_dim=qwen_head_dim,
            h=qwen_hidden_size,
            p=2,
        )
        hardware_profiled = True
        print(f"\n{'=' * 60}")
        print(f"[Profiling] v_com={v_com:.2f} GB/s, t_gpu_ms_per_token={t_gpu_ms_per_token:.5f} ms")
        print(f"{'=' * 60}\n")

    answers = [ex["answer"]]
    ctxs = []
    for title, sentences in ex["context"]:
        ctxs.append({"title": title, "text": "".join(sentences)})
    ex_formatted = {"question": ex["question"], "ctxs": ctxs}

    doc_prompts, q_prompt = build_qa_prompt(ex_formatted, query_prompt)
    doc_chunk_ids = [encode_like_reference(doc, tokenizer) for doc in doc_prompts]
    q_ids = encode_like_reference(q_prompt, tokenizer)

    if len(doc_chunk_ids) == 0:
        continue

    sampling_params = SamplingParams(temperature=0, max_tokens=1)

    run_on_all_workers(llm, "cachetune_reset_state")
    run_on_all_workers(
        llm,
        "cachetune_set_flags",
        updates={"collect": False, "check": False, "attn_bias": None},
        reset_pipeline=True,
    )

    s_start_full = encode_like_reference(prefix_prompt, tokenizer)
    bos_pad = 1 if tokenizer.bos_token_id is not None else 0
    s_start_len = len(s_start_full) + bos_pad

    s_start = []
    s_start_1_len = len(s_start) + bos_pad
    s_end = []

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

    print(f"Sample {sample_idx}: Generating KV Cache with CPU Offload...")
    for i in range(len(doc_chunk_ids)):
        chunk_ids = doc_chunk_ids[i]
        chunk_eval_ids = [tokenizer.bos_token_id] + chunk_ids if tokenizer.bos_token_id is not None else chunk_ids
        llm.generate(prompt_token_ids=[chunk_eval_ids], sampling_params=sampling_params)
        run_on_all_workers(
            llm,
            "cachetune_append_chunk_kv",
            chunk_len=len(chunk_eval_ids),
            s_start_len=s_start_len,
            s_start_1_len=s_start_1_len,
            is_first_chunk=(i == 0),
        )

    run_on_all_workers(llm, "cachetune_finalize_chunk_kv")
    llm_model = get_driver_model(llm)
    cache_fuse_metadata = llm_model.cache_fuse_metadata
    chunk_past_key_values = cache_fuse_metadata["cpu_kv_cache"]

    print(f"Sample {sample_idx}: Analyzing Frequency Domain to find low-freq tokens...")
    total_tokens = chunk_past_key_values[0][0].shape[0]
    n_context = total_tokens - last_len
    if n_context <= 0:
        print(f"[WARN] Sample {sample_idx}: invalid context length (N_context={n_context}, total={total_tokens}, last_len={last_len}), skip CacheTune.")
        continue

    l_total = optimal_l(
        n_context,
        num_kv_heads=local_num_kv_heads,
        head_dim=qwen_head_dim,
        v_com=v_com,
        t_gpu_ms_per_token=t_gpu_ms_per_token,
    )
    l_total = max(1, min(n_context, l_total))
    recomp_ratio = args.recomp_ratio
    pyramid_interval = args.pyramid_interval
    shrink_ratio = args.shrink_ratio

    layer0_v = chunk_past_key_values[0][1].to(torch.device("cuda", torch.cuda.current_device()))
    context_v = layer0_v[:-last_len]

    if recomp_ratio >= 0.999:
        base_recomp_ctx = torch.arange(n_context, device=context_v.device, dtype=torch.int64)
    else:
        base_recomp_ctx, _ = get_optimized_indices(
            context_v,
            N=n_context,
            ratio=recomp_ratio,
            low_freq_pct=args.low_freq_pct,
            sink_size=args.sink_size,
        )

    total_len = layer0_v.shape[0]
    suffix_start = total_len - last_len
    if suffix_start < 0:
        raise ValueError(f"Invalid suffix range at sample {sample_idx}: total_len={total_len}, last_len={last_len}")
    suffix_indices = torch.arange(suffix_start, total_len, device=base_recomp_ctx.device)

    current_recomp_ctx = base_recomp_ctx.clone()
    use_pyramid = args.use_pyramid and recomp_ratio < 0.95
    if use_pyramid and len(current_recomp_ctx) > 1:
        for j in range(num_layers):
            if j > 1 and j % pyramid_interval == 0 and len(current_recomp_ctx) > 1:
                layer_v_j = chunk_past_key_values[j][1].to(torch.device("cuda", torch.cuda.current_device()))
                subset_v = layer_v_j[:-last_len][current_recomp_ctx]
                scores = torch.norm(subset_v.float(), p=2, dim=tuple(range(1, subset_v.ndim)))
                new_k = max(1, int(len(current_recomp_ctx) * shrink_ratio))
                top_sub = torch.topk(scores, k=new_k).indices
                current_recomp_ctx = current_recomp_ctx[top_sub].sort().values

    final_indices = torch.cat([current_recomp_ctx, suffix_indices])
    final_indices = torch.unique(final_indices, sorted=True)

    if final_indices.numel() == 0:
        raise ValueError(f"Empty recompute indices at sample {sample_idx}")
    if final_indices.min().item() < 0 or final_indices.max().item() >= total_len:
        raise ValueError(
            f"Out-of-range recompute indices at sample {sample_idx}: min={final_indices.min().item()}, "
            f"max={final_indices.max().item()}, total_len={total_len}")

    run_on_all_workers(
        llm,
        "cachetune_prepare_from_indices",
        final_indices_cpu=final_indices.to("cpu"),
        last_len=last_len,
        recomp_ratio=float(recomp_ratio),
        check_layers=[1],
    )

    input_ids = []
    for i in range(len(doc_chunk_ids)):
        if i == 0:
            temp_ids = doc_chunk_ids[i]
        else:
            temp_ids = doc_chunk_ids[i][s_start_1_len - bos_pad:]
        input_ids += temp_ids

    final_prompt_ids = [tokenizer.bos_token_id] + input_ids if tokenizer.bos_token_id is not None else input_ids

    if len(final_prompt_ids) != total_len:
        raise RuntimeError(
            f"Prompt/KV length mismatch at sample {sample_idx}: prompt={len(final_prompt_ids)} vs kv={total_len}")
    print(f"  -> Eval prompt length (shared by cache/full): {len(final_prompt_ids)}")

    sampling_params = SamplingParams(temperature=0, max_tokens=32)

    print(f"Sample idx: {sample_idx}")
    print(f"  -> Recompute Ratio: {recomp_ratio:.3f} ({recomp_ratio * 100:.1f}%)")
    print(f"  -> Context Tokens: {n_context}")
    print(f"  -> Recompute Tokens per Layer: ~{l_total}")
    print(f"  -> Initial recomp tokens (L1): {len(base_recomp_ctx)}")
    print(f"  -> Final recomp tokens: {len(current_recomp_ctx)}")
    print(f"  -> Pyramid enabled: {use_pyramid}")

    output = llm.generate(prompt_token_ids=[final_prompt_ids], sampling_params=sampling_params)
    if not output or not output[0].outputs:
        print(f"[WARN] Sample {sample_idx}: empty cached output, skip sample.")
        continue
    res_raw = output[0].outputs[0].text
    res = normalize_pred_for_eval(res_raw)
    print(f"Cached generation (CacheTune): {res}")

    ttft = safe_ttft(output[0])
    if ttft is None:
        print(f"[WARN] Sample {sample_idx}: missing cached TTFT metrics.")
        ttft_blend.append(np.nan)
    else:
        print(f"TTFT with cache: {ttft:.4f}")
        ttft_blend.append(ttft)

    f1 = max([compute_f1(res, ans, tokenizer) for ans in answers])
    f1_blend.append(f1)

    run_on_all_workers(llm, "cachetune_reset_state")

    output = llm.generate(prompt_token_ids=[final_prompt_ids], sampling_params=sampling_params)
    if not output or not output[0].outputs:
        print(f"[WARN] Sample {sample_idx}: empty full-prefill output.")
        ttft_full.append(np.nan)
        f1_full.append(0.0)
        continue
    res_raw = output[0].outputs[0].text
    res = normalize_pred_for_eval(res_raw)
    print(f"Normal generation: {res}")

    ttft = safe_ttft(output[0])
    if ttft is None:
        print(f"[WARN] Sample {sample_idx}: missing full-prefill TTFT metrics.")
        ttft_full.append(np.nan)
    else:
        print(f"TTFT with full prefill: {ttft}")
        ttft_full.append(ttft)

    f1 = max([compute_f1(res, ans, tokenizer) for ans in answers])
    f1_full.append(f1)

    run_on_all_workers(llm, "cachetune_reset_state")

    print("------------")

print("---------------Result Summary---------------------")
print(f"[Profiling] v_com={v_com:.2f} GB/s, t_gpu_ms_per_token={t_gpu_ms_per_token:.5f} ms")
ttft_blend_mean = np.nanmean(ttft_blend) if len(ttft_blend) > 0 else float("nan")
ttft_full_mean = np.nanmean(ttft_full) if len(ttft_full) > 0 else float("nan")
print(f"TTFT with cache: {ttft_blend_mean:.4f} seconds")
print(f"TTFT with full prefill: {ttft_full_mean:.4f} seconds")
if np.isfinite(ttft_blend_mean) and np.isfinite(ttft_full_mean) and ttft_blend_mean > 0:
    print(f"Speedup: {ttft_full_mean / ttft_blend_mean:.2f}x")
else:
    print("Speedup: N/A")
print(f"F1 with cache: {np.mean(f1_blend):.4f}")
print(f"F1 with full prefill: {np.mean(f1_full):.4f}")
if np.mean(f1_full) > 0:
    print(f"Accuracy Preservation: {(np.mean(f1_blend) / np.mean(f1_full) * 100):.2f}%")
else:
    print("Accuracy Preservation: N/A")
