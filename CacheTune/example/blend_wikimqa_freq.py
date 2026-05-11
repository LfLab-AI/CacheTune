from vllm import LLM, SamplingParams
import torch
import numpy as np
import argparse
import gc
from transformers import AutoTokenizer
from utils import load_dataset, build_qa_prompt, compute_f1


# --- Hardware Profiling: Measure PCIe bandwidth and GPU compute time per token per layer ---
def profile_hardware(num_kv_heads=8, head_dim=128, h=4096, p=2, num_trials=5):
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

    print(f"[optimal_l] N={N}, best_l={best_l} ({best_l/N:.1%}), est. t_layer={min_t:.3f} ms, cap={int(r_max*100)}%")
    return best_l


# --- V2.0 Frequency selection with optional attention sink ---
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


def flatten_answer_texts(answer_obj):
    if isinstance(answer_obj, str):
        return [answer_obj]
    if isinstance(answer_obj, list):
        texts = []
        for x in answer_obj:
            texts.extend(flatten_answer_texts(x))
        return texts
    return [str(answer_obj)]


parser = argparse.ArgumentParser(description="CacheTune WikiMQA script")
parser.add_argument("--model-path", type=str, default="/path/model/Mistral-7B-Instruct-v0.3")
parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
parser.add_argument("--recomp-ratio", type=float, default=0.15)
parser.add_argument("--low-freq-pct", type=float, default=0.70)
parser.add_argument("--sink-size", type=int, default=0)
parser.add_argument("--use-pyramid", action="store_true",
                    help="Enable layer-wise shrinking; disabled by default for fair ratio-vs-baseline comparison")
parser.add_argument("--pyramid-interval", type=int, default=8)
parser.add_argument("--shrink-ratio", type=float, default=1.0)
args = parser.parse_args()

eval_dataset = load_dataset("inputs/wikimqa_s.json")

model_path = args.model_path
llm = LLM(model=model_path, gpu_memory_utilization=args.gpu_memory_utilization)
tokenizer = AutoTokenizer.from_pretrained(model_path)
llm.set_tokenizer(tokenizer)

prefix_prompt = "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n"
query_prompt = "\n\nAnswer the question based on the given passages. Answer the question within 5 words. Do NOT repeat the question or output any other words. Question: "

ttft_blend = []
ttft_full = []
f1_blend = []
f1_full = []

hardware_profiled = False
v_com = 10.0
t_gpu_ms_per_token = 0.001

for sample_idx, ex in enumerate(eval_dataset):
    if not hardware_profiled:
        v_com, t_gpu_ms_per_token = profile_hardware()
        hardware_profiled = True
        print(f"\n{'='*60}")
        print(f"[Profiling] v_com={v_com:.2f} GB/s, t_gpu_ms_per_token={t_gpu_ms_per_token:.5f} ms")
        print(f"{'='*60}\n")

    answers = ex["answers"]
    doc_prompts, q_prompt = build_qa_prompt(ex, query_prompt)
    doc_chunk_ids = [tokenizer.encode(doc)[1:] for doc in doc_prompts]
    q_ids = tokenizer.encode(q_prompt)[1:]

    sampling_params = SamplingParams(temperature=0, max_tokens=1)

    cache_fuse_metadata = llm.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata
    cache_fuse_metadata["collect"] = False
    cache_fuse_metadata["check"] = False

    s_start_full = tokenizer.encode(prefix_prompt)[1:]
    s_start_len = len(s_start_full) + 1

    s_start = []
    s_start_1_len = len(s_start) + 1

    s_end = []

    doc_chunk_ids = [s_start + chunk_ids for chunk_ids in doc_chunk_ids]
    doc_chunk_ids = [s_start_full] + doc_chunk_ids
    doc_chunk_ids = doc_chunk_ids + [s_start + q_ids + s_end]

    last_len = len(q_ids + s_end)

    cache_fuse_metadata["collect"] = True
    cache_fuse_metadata["check"] = False
    cache_fuse_metadata["pipeline_enabled"] = False
    cache_fuse_metadata["layer_counter"] = 0

    num_layer = 32
    chunk_past_key_values = []

    print(f"Sample {sample_idx}: Generating KV Cache with CPU Offload...")
    for i in range(len(doc_chunk_ids)):
        chunk_ids = doc_chunk_ids[i]
        chunk_eval_ids = [tokenizer.bos_token_id] + chunk_ids
        llm.generate(prompt_token_ids=[chunk_eval_ids], sampling_params=sampling_params)

        llm_layers = llm.llm_engine.model_executor.driver_worker.model_runner.model.model.layers
        for j in range(num_layer):
            past_key_values = llm_layers[j].self_attn.hack_kv
            if i == 0:
                temp_k = past_key_values[0][:len(chunk_eval_ids)].clone()
                temp_v = past_key_values[1][:len(chunk_eval_ids)].clone()
            else:
                temp_k = past_key_values[0][1:len(chunk_ids) + 1].clone()
                temp_v = past_key_values[1][1:len(chunk_ids) + 1].clone()

            temp_k = temp_k.to("cpu").pin_memory()
            temp_v = temp_v.to("cpu").pin_memory()

            if i == 0:
                chunk_past_key_values.append([temp_k, temp_v])
            else:
                chunk_past_key_values[j][0] = torch.cat((chunk_past_key_values[j][0], temp_k), dim=0)
                chunk_past_key_values[j][1] = torch.cat((chunk_past_key_values[j][1], temp_v), dim=0)

            llm_layers[j].self_attn.hack_kv = None

    for j in range(num_layer):
        if not chunk_past_key_values[j][0].is_pinned():
            chunk_past_key_values[j][0] = chunk_past_key_values[j][0].contiguous().pin_memory()
        if not chunk_past_key_values[j][1].is_pinned():
            chunk_past_key_values[j][1] = chunk_past_key_values[j][1].contiguous().pin_memory()

    cache_fuse_metadata["cpu_kv_cache"] = chunk_past_key_values

    recomp_ratio = args.recomp_ratio
    pyramid_interval = args.pyramid_interval
    shrink_ratio = args.shrink_ratio

    layer_v0 = chunk_past_key_values[0][1].to("cuda")
    context_v0 = layer_v0[:-last_len]
    n_context = context_v0.shape[0]
    if n_context <= 0:
        print(f"[WARN] Sample {sample_idx}: invalid context length (N_context={n_context}), skip.")
        continue

    l_total = optimal_l(
        n_context,
        num_kv_heads=8,
        head_dim=128,
        v_com=v_com,
        t_gpu_ms_per_token=t_gpu_ms_per_token,
    )
    l_total = max(1, min(n_context, l_total))

    if recomp_ratio >= 0.999:
        base_recomp_ctx = torch.arange(n_context, device=context_v0.device, dtype=torch.int64)
    else:
        base_recomp_ctx, _ = get_optimized_indices(
            context_v0,
            N=n_context,
            ratio=recomp_ratio,
            low_freq_pct=args.low_freq_pct,
            sink_size=args.sink_size,
        )

    total_len = chunk_past_key_values[0][0].shape[0]
    suffix_indices = torch.arange(total_len - last_len, total_len, device=base_recomp_ctx.device)
    current_recomp_ctx = base_recomp_ctx.clone()
    cache_fuse_metadata["org_seq_len"] = total_len

    precomputed_indices_list = []
    use_pyramid = args.use_pyramid and recomp_ratio < 0.95
    for j in range(num_layer):
        if use_pyramid and j > 1 and j % pyramid_interval == 0 and len(current_recomp_ctx) > 1:
            layer_v_j = chunk_past_key_values[j][1].to("cuda")
            subset_v = layer_v_j[:-last_len][current_recomp_ctx]
            scores = torch.norm(subset_v.float(), p=2, dim=tuple(range(1, subset_v.ndim)))
            new_k = max(1, int(len(current_recomp_ctx) * shrink_ratio))
            top_sub = torch.topk(scores, k=new_k).indices
            current_recomp_ctx = current_recomp_ctx[top_sub].sort().values
        precomputed_indices_list.append(torch.cat([current_recomp_ctx, suffix_indices]))

    gpu_transfer_k = []
    gpu_transfer_v = []
    for j in range(num_layer):
        gpu_transfer_k.append(torch.empty_like(chunk_past_key_values[j][0], device="cuda"))
        gpu_transfer_v.append(torch.empty_like(chunk_past_key_values[j][1], device="cuda"))
    cache_fuse_metadata["gpu_transfer_k"] = gpu_transfer_k
    cache_fuse_metadata["gpu_transfer_v"] = gpu_transfer_v

    n_total = total_len
    num_kv_heads = chunk_past_key_values[0][0].shape[1] if chunk_past_key_values[0][0].ndim == 3 else 8
    head_dim_val = chunk_past_key_values[0][0].shape[2] if chunk_past_key_values[0][0].ndim == 3 else 128
    kv_dtype = chunk_past_key_values[0][0].dtype
    cache_fuse_metadata["work_key"] = torch.empty((n_total, num_kv_heads, head_dim_val), dtype=kv_dtype, device="cuda")
    cache_fuse_metadata["work_val"] = torch.empty((n_total, num_kv_heads, head_dim_val), dtype=kv_dtype, device="cuda")

    cache_fuse_metadata["precomputed_indices"] = precomputed_indices_list

    input_ids = []
    for i in range(len(doc_chunk_ids)):
        if i == 0:
            temp_ids = doc_chunk_ids[i]
        else:
            temp_ids = doc_chunk_ids[i][s_start_1_len - 1:]
        input_ids += temp_ids

    final_prompt_ids = [tokenizer.bos_token_id] + input_ids
    if len(final_prompt_ids) != total_len:
        raise RuntimeError(
            f"Prompt/KV length mismatch at sample {sample_idx}: prompt={len(final_prompt_ids)} vs kv={total_len}")

    sampling_params = SamplingParams(temperature=0, max_tokens=32)
    cache_fuse_metadata["check"] = True
    cache_fuse_metadata["collect"] = False
    cache_fuse_metadata["recomp_ratio"] = recomp_ratio
    cache_fuse_metadata["fast_attention"] = True
    cache_fuse_metadata["suffix_len"] = last_len
    cache_fuse_metadata["layer_counter"] = 0
    cache_fuse_metadata["pipeline_enabled"] = True

    print(f"Sample idx: {sample_idx}")
    print(f"  -> Recompute Ratio: {recomp_ratio:.3f} ({recomp_ratio * 100:.1f}%)")
    print(f"  -> Context Tokens: {n_context}")
    print(f"  -> Recompute Tokens per Layer (target): ~{l_total}")
    print(f"  -> Initial recomp tokens (L1): {len(precomputed_indices_list[0]) - last_len}")
    print(f"  -> Final recomp tokens (L{num_layer - 1}): {len(precomputed_indices_list[-1]) - last_len}")
    print(f"  -> Pyramid enabled: {use_pyramid}")

    output = llm.generate(prompt_token_ids=[final_prompt_ids], sampling_params=sampling_params)
    res = output[0].outputs[0].text
    res = res.strip().split("\n")[0]
    print(f"Cached generation (CacheTune): {res}")

    ttft = output[0].metrics.first_token_time - output[0].metrics.first_scheduled_time
    print(f"TTFT with cache: {ttft}")
    ttft_blend.append(ttft)

    gt_answers = []
    for answer in answers:
        gt_answers.extend(flatten_answer_texts(answer))
    f1 = max([compute_f1(res, ans, tokenizer) for ans in gt_answers])
    f1_blend.append(f1)

    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model.model
    llm_model._pipeline_initialized = False
    cache_fuse_metadata["pipeline_enabled"] = False
    cache_fuse_metadata["attn_bias"] = None
    cache_fuse_metadata["imp_indices"] = None
    cache_fuse_metadata["layer_counter"] = 0

    sampling_params = SamplingParams(temperature=0, max_tokens=32)
    cache_fuse_metadata["check"] = False
    cache_fuse_metadata["collect"] = False

    output = llm.generate(prompt_token_ids=[final_prompt_ids], sampling_params=sampling_params)

    res = output[0].outputs[0].text
    res = res.strip().split("\n")[0]
    print(f"Normal generation: {res}")
    ttft = output[0].metrics.first_token_time - output[0].metrics.first_scheduled_time
    print(f"TTFT with full prefill: {ttft}")
    ttft_full.append(ttft)

    f1 = max([compute_f1(res, ans, tokenizer) for ans in gt_answers])
    f1_full.append(f1)

    cache_fuse_metadata["pipeline_enabled"] = True
    llm_model._pipeline_initialized = False

    cache_fuse_metadata["cpu_kv_cache"] = None
    cache_fuse_metadata["gpu_transfer_k"] = None
    cache_fuse_metadata["gpu_transfer_v"] = None
    cache_fuse_metadata["precomputed_indices"] = None
    cache_fuse_metadata["work_key"] = None
    cache_fuse_metadata["work_val"] = None

    del output
    del chunk_past_key_values
    gc.collect()
    torch.cuda.empty_cache()

    print("------------")

print("---------------Result Summary---------------------")
print(f"[Profiling] v_com={v_com:.2f} GB/s, t_gpu_ms_per_token={t_gpu_ms_per_token:.5f} ms")
print(f"TTFT with cache: {np.mean(ttft_blend)}")
print(f"TTFT with full prefill: {np.mean(ttft_full)}")
print(f"F1 with cache: {np.mean(f1_blend)}")
print(f"F1 with full prefill: {np.mean(f1_full)}")
print(f"Accuracy Preservation: {(np.mean(f1_blend) / np.mean(f1_full) * 100):.2f}%")
