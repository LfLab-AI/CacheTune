"""
CacheTune for Qwen2.5 on MuSiQue dataset (TP-aware).
"""

# BEGIN CACHETUNE PAPER DISPATCH
def run_paper_method(argv=None):
    """Run the opt-in ICLR 2027 method; the original path remains the default."""
    from paper_runner import main
    return main(dataset='musique', default_storage='cpu',
                is_qwen=True, argv=argv)


if __name__ == "__main__":
    from paper_dispatch import dispatch_if_requested as _dispatch_paper_method
    _dispatch_paper_method(run_paper_method)
# END CACHETUNE PAPER DISPATCH


from vllm import LLM, SamplingParams
import torch
import numpy as np
import argparse
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


def calculate_freq_indices(key, value, ratio, low_freq_pct=0.50, sink_size=0):
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

    top_indices_local = torch.topk(imp_scores, k=total_budget).indices
    top_indices_global = top_indices_local + sink_size

    final_indices = torch.cat([sink_indices, top_indices_global])
    final_indices, _ = torch.sort(final_indices)
    return final_indices.to(torch.int64).to(value.device)


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


def flatten_answer_texts(answer_obj):
    if isinstance(answer_obj, str):
        return [answer_obj]
    if isinstance(answer_obj, list):
        texts = []
        for x in answer_obj:
            texts.extend(flatten_answer_texts(x))
        return texts
    return [str(answer_obj)]


def safe_ttft(req_output):
    metrics = getattr(req_output, "metrics", None)
    if metrics is None:
        return None
    first_token_time = getattr(metrics, "first_token_time", None)
    first_scheduled_time = getattr(metrics, "first_scheduled_time", None)
    if first_token_time is None or first_scheduled_time is None:
        return None
    return first_token_time - first_scheduled_time


def encode_like_musique_reference(text, tokenizer):
    ids = tokenizer.encode(text)
    bos_id = tokenizer.bos_token_id
    if bos_id is not None and len(ids) > 0 and ids[0] == bos_id:
        return ids[1:]
    return ids


parser = argparse.ArgumentParser(description="CacheTune MuSiQue Qwen inference script")
parser.add_argument("--model-path", type=str, default="/path/model/Qwen2.5-32B-Instruct")
parser.add_argument("--tensor-parallel-size", type=int, default=2)
parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
parser.add_argument("--max-model-len", type=int, default=4096)
parser.add_argument("--enforce-eager", action="store_true")
parser.add_argument("--max-num-seqs", type=int, default=1)
parser.add_argument("--low-freq-pct", type=float, default=0.25)
args = parser.parse_args()

eval_dataset = load_dataset("inputs/musique_s.json")

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

prefix_prompt = "You will be asked a question after reading several passages. Please directly answer the question based on the given passages. Do NOT repeat the question. The answer should be within 5 words..\nPassages:\n"
query_prompt = "\n\nAnswer the question directly based on the given passages. Do NOT repeat the question. The answer should be within 5 words. \nQuestion:"

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

    answers = ex["answers"]
    doc_prompts, q_prompt = build_qa_prompt(ex, query_prompt)
    doc_chunk_ids = [encode_like_musique_reference(doc, tokenizer) for doc in doc_prompts]
    q_ids = encode_like_musique_reference(q_prompt, tokenizer)

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

    s_start_full = encode_like_musique_reference(prefix_prompt, tokenizer)
    bos_pad = 1 if tokenizer.bos_token_id is not None else 0
    s_start_len = len(s_start_full) + bos_pad

    s_start = []
    s_start_1_len = len(s_start) + bos_pad

    s_end = []

    doc_chunk_ids = [s_start + chunk_ids for chunk_ids in doc_chunk_ids]
    doc_chunk_ids = [s_start_full] + doc_chunk_ids
    doc_chunk_ids = doc_chunk_ids + [s_start + q_ids + s_end]

    last_len = len(q_ids + s_end)

    input_ids = []
    for i in range(len(doc_chunk_ids)):
        if i == 0:
            temp_ids = doc_chunk_ids[i]
        else:
            temp_ids = doc_chunk_ids[i][len(s_start):]
        input_ids += temp_ids

    if tokenizer.bos_token_id is not None:
        final_prompt_ids = [tokenizer.bos_token_id] + input_ids
    else:
        final_prompt_ids = input_ids

    run_on_all_workers(
        llm,
        "cachetune_set_flags",
        updates={"collect": True, "check": False},
        reset_pipeline=True,
    )

    print(f"Sample {sample_idx}: Generating KV Cache with CPU Offload...")
    for i in range(len(doc_chunk_ids)):
        chunk_eval_ids = doc_chunk_ids[i]
        if tokenizer.bos_token_id is not None:
            chunk_eval_ids = [tokenizer.bos_token_id] + chunk_eval_ids
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
    recomp_ratio = 0.15

    layer0_v = chunk_past_key_values[0][1].to(torch.device("cuda", torch.cuda.current_device()))
    layer0_k = chunk_past_key_values[0][0].to(torch.device("cuda", torch.cuda.current_device()))
    context_v = layer0_v[:-last_len]

    indices = calculate_freq_indices(
        layer0_k,
        context_v,
        ratio=recomp_ratio,
        low_freq_pct=args.low_freq_pct,
        sink_size=0,
    )

    total_len = layer0_v.shape[0]
    suffix_start = total_len - last_len
    if suffix_start < 0:
        raise ValueError(f"Invalid suffix range at sample {sample_idx}: total_len={total_len}, last_len={last_len}")
    suffix_indices = torch.arange(suffix_start, total_len, device=indices.device)
    final_indices = torch.cat([indices, suffix_indices])
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

    if len(final_prompt_ids) != total_len:
        raise RuntimeError(
            f"Prompt/KV length mismatch at sample {sample_idx}: prompt={len(final_prompt_ids)} vs kv={total_len}")
    if len(final_prompt_ids) > args.max_model_len:
        raise RuntimeError(
            f"Sample {sample_idx} prompt length {len(final_prompt_ids)} exceeds max_model_len={args.max_model_len}. "
            "Please increase --max-model-len (recommended: 8192 for MuSiQue) or shorten prompt context.")

    sampling_params = SamplingParams(temperature=0, max_tokens=32)

    print(f"Sample idx: {sample_idx}")
    print(f"  -> Recompute Ratio: {recomp_ratio:.3f} ({recomp_ratio * 100:.1f}%)")
    print(f"  -> Context Tokens: {n_context}")
    print(f"  -> Recompute Tokens per Layer: ~{l_total}")
    print(f"  -> Final prompt tokens: {len(final_prompt_ids)}")

    output = llm.generate(prompt_token_ids=[final_prompt_ids], sampling_params=sampling_params)
    if not output or not output[0].outputs:
        print(f"[WARN] Sample {sample_idx}: empty cached output, skip sample.")
        continue
    res = output[0].outputs[0].text
    res = res.strip().split("\n")[0]
    print(f"Cached generation (CacheTune): {res}")
    print(f"Cached output token length: {len(tokenizer.encode(res))}")

    ttft = safe_ttft(output[0])
    if ttft is None:
        print(f"[WARN] Sample {sample_idx}: missing cached TTFT metrics.")
        ttft_blend.append(np.nan)
    else:
        print(f"TTFT with cache: {ttft:.4f}")
        ttft_blend.append(ttft)

    gt_answers = []
    for answer in answers:
        gt_answers.extend(flatten_answer_texts(answer))
    f1 = max([compute_f1(res, ans, tokenizer) for ans in gt_answers])
    f1_blend.append(f1)

    run_on_all_workers(
        llm,
        "cachetune_set_flags",
        updates={
            "check": False,
            "collect": False,
            "pipeline_enabled": False,
            "attn_bias": None,
            "imp_indices": None,
            "non_imp_indices": None,
            "precomputed_indices": None,
            "work_key": None,
            "work_val": None,
            "org_pos": None,
            "org_seq_len": None,
            "suffix_len": None,
            "check_layers": [],
            "recomp_ratio": 0.0,
            "fast_attention": False,
            "layer_counter": 0,
            "cpu_kv_cache": None,
        },
        reset_pipeline=True,
    )

    output = llm.generate(prompt_token_ids=[final_prompt_ids], sampling_params=sampling_params)
    if not output or not output[0].outputs:
        print(f"[WARN] Sample {sample_idx}: empty full-prefill output.")
        ttft_full.append(np.nan)
        f1_full.append(0.0)
        continue
    res = output[0].outputs[0].text
    res = res.strip().split("\n")[0]
    print(f"Normal generation: {res}")
    print(f"Normal output token length: {len(tokenizer.encode(res))}")

    ttft = safe_ttft(output[0])
    if ttft is None:
        print(f"[WARN] Sample {sample_idx}: missing full-prefill TTFT metrics.")
        ttft_full.append(np.nan)
    else:
        print(f"TTFT with full prefill: {ttft}")
        ttft_full.append(ttft)

    f1 = max([compute_f1(res, ans, tokenizer) for ans in gt_answers])
    f1_full.append(f1)

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
print(f"Accuracy Preservation: {(np.mean(f1_blend) / np.mean(f1_full) * 100):.2f}%")
