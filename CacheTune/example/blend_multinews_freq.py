import re
from itertools import chain
from pathlib import Path
import argparse

import numpy as np
import torch
from rouge_score import rouge_scorer
from transformers import AutoConfig, AutoTokenizer
from vllm import LLM, SamplingParams


def profile_hardware(num_kv_heads=8, head_dim=128, h=4096, p=2, num_trials=5):
    """Measure PCIe bandwidth and GPU compute time per token per layer."""
    print("\n[Hardware Profiling] Measuring v_com (PCIe) and "
          "t_gpu_ms_per_token (GPU compute)...")

    test_N = 2048
    total_bytes = 2 * test_N * h * p
    test_tensor = torch.zeros(2 * test_N * h,
                              dtype=torch.bfloat16).pin_memory()

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
    W = torch.randn(h,
                    h + 2 * num_kv_heads * head_dim,
                    device="cuda",
                    dtype=torch.bfloat16)
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
    print(f"[Profiling Result] v_com={v_com:.2f} GB/s, "
          f"t_gpu_ms_per_token={t_gpu_ms_per_token:.5f} ms")
    return v_com, t_gpu_ms_per_token


def optimal_l(N,
              num_kv_heads=8,
              head_dim=128,
              v_com=10.0,
              t_gpu_ms_per_token=0.001,
              p=2,
              r_min=0.15,
              r_max=0.50):
    """Grid-search the recompute token count."""
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

    print(f"[optimal_l] N={N}, best_l={best_l} ({best_l / N:.1%}), "
          f"est. t_layer={min_t:.3f} ms")
    return best_l


def get_optimized_indices(raw_v, N, ratio, low_freq_pct=0.70, sink_size=0):
    """Return recompute/reuse indices selected by low-frequency energy."""
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


def resolve_multinews_files():
    candidates = [
        (Path("/path/dataset/Multi-News/data/test.src.cleaned"),
         Path("/path/dataset/Multi-News/data/test.tgt")),
        (Path("inputs/test.src.cleaned"), Path("inputs/test.tgt")),
    ]
    for src_path, tgt_path in candidates:
        if src_path.exists() and tgt_path.exists():
            return src_path, tgt_path
    return candidates[0]


def load_multinews_dataset(limit=200):
    src_path, tgt_path = resolve_multinews_files()
    print(f"Loading dataset from {src_path} and {tgt_path}")

    eval_dataset = []
    with open(src_path, "r", encoding="utf-8") as f_src, open(
            tgt_path, "r", encoding="utf-8") as f_tgt:
        src_lines = f_src.readlines()[:limit]
        tgt_lines = f_tgt.readlines()[:limit]

    for src_line, tgt_line in zip(src_lines, tgt_lines):
        docs = src_line.replace("NEWLINE_CHAR", "\n").split("|||||")
        docs = [doc.strip() for doc in docs if doc.strip()]
        eval_dataset.append({
            "docs": docs,
            "answer": tgt_line.strip().replace("NEWLINE_CHAR", "\n"),
        })
    return eval_dataset


ROUGE_METRICS = ("rouge1", "rouge2", "rougeLsum")
ROUGE_SCORER = rouge_scorer.RougeScorer(ROUGE_METRICS, use_stemmer=True)


def normalize_summary_for_rouge_lsum(text):
    """Insert line breaks between sentences for rougeLsum."""
    text = text.strip()
    if not text:
        return ""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    return "\n".join(sentences) if sentences else text


def compute_rouge_dict(prediction, reference):
    """Return Rouge-1/2/Lsum F1 scores."""
    pred_norm = normalize_summary_for_rouge_lsum(prediction)
    ref_norm = normalize_summary_for_rouge_lsum(reference)
    scores = ROUGE_SCORER.score(ref_norm, pred_norm)
    return {metric: scores[metric].fmeasure for metric in ROUGE_METRICS}


def best_rouge_against_references(prediction, references):
    """Use the best-matching reference for each metric."""
    best_scores = {metric: 0.0 for metric in ROUGE_METRICS}
    for reference in references:
        scores = compute_rouge_dict(prediction, reference)
        for metric in ROUGE_METRICS:
            best_scores[metric] = max(best_scores[metric], scores[metric])
    return best_scores


def init_metric_store():
    return {metric: [] for metric in ROUGE_METRICS}


def append_metric_scores(store, scores):
    for metric in ROUGE_METRICS:
        store[metric].append(scores[metric])


def summarize_paired_scores(cache_scores, full_scores, num_bootstrap=5000):
    """Return paired diagnostics for cache-vs-full evaluation."""
    cache_arr = np.asarray(cache_scores, dtype=np.float64)
    full_arr = np.asarray(full_scores, dtype=np.float64)
    deltas = cache_arr - full_arr

    win_count = int(np.sum(deltas > 1e-9))
    loss_count = int(np.sum(deltas < -1e-9))
    tie_count = int(len(deltas) - win_count - loss_count)

    rng = np.random.default_rng(0)
    if len(deltas) == 0:
        ci_low = 0.0
        ci_high = 0.0
    else:
        bootstrap_means = []
        for _ in range(num_bootstrap):
            sample = rng.choice(deltas, size=len(deltas), replace=True)
            bootstrap_means.append(sample.mean())
        ci_low, ci_high = np.percentile(bootstrap_means, [2.5, 97.5])

    return {
        "mean_cache": float(cache_arr.mean()) if len(cache_arr) else 0.0,
        "mean_full": float(full_arr.mean()) if len(full_arr) else 0.0,
        "mean_delta": float(deltas.mean()) if len(deltas) else 0.0,
        "wins": win_count,
        "losses": loss_count,
        "ties": tie_count,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
    }


def build_prompt_chunks(tokenizer, docs, prefix_prompt, query_prompt):
    bos_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 1
    prefix_ids = tokenizer.encode(prefix_prompt)[1:]
    doc_ids = [
        tokenizer.encode(f"Document {idx + 1}:\n{doc}\n\n")[1:]
        for idx, doc in enumerate(docs)
    ]
    query_ids = tokenizer.encode(query_prompt)[1:]

    prompt_chunks = [[bos_token_id] + prefix_ids] + doc_ids + [query_ids]
    input_ids = list(chain.from_iterable(prompt_chunks))
    last_len = len(query_ids)
    return prompt_chunks, input_ids, last_len


def reset_cachetune_state(llm_model):
    cache_fuse_metadata = llm_model.cache_fuse_metadata
    cache_fuse_metadata["collect"] = False
    cache_fuse_metadata["check"] = False
    cache_fuse_metadata["attn_bias"] = None
    cache_fuse_metadata["imp_indices"] = None
    cache_fuse_metadata["pipeline_enabled"] = False
    cache_fuse_metadata["layer_counter"] = 0

    for key in [
            "cpu_kv_cache",
            "transfer_indices",
            "precomputed_indices",
            "non_imp_indices",
            "gpu_transfer_k",
            "gpu_transfer_v",
            "work_key",
            "work_val",
            "org_seq_len",
            "org_pos",
            "suffix_len",
            "kv_cache_dtype",
            "recomp_ratio",
            "fast_attention",
            "check_layer",
            "status2_layer_counter",
    ]:
        cache_fuse_metadata.pop(key, None)

    llm_model._pipeline_initialized = False
    llm_model._prefetch_buffers.clear()
    llm_model.old_kvs = [[None, None] for _ in range(len(llm_model.layers))]


parser = argparse.ArgumentParser(description="CacheTune MultiNews script")
parser.add_argument("--model-path", type=str, default="/path/model/Mistral-7B-Instruct-v0.3")
parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
parser.add_argument("--sample-limit", type=int, default=200)
parser.add_argument("--recomp-ratio", type=float, default=0.15)
parser.add_argument("--low-freq-pct", type=float, default=0.70)
parser.add_argument("--sink-size", type=int, default=0)
parser.add_argument("--use-pyramid", action="store_true")
parser.add_argument("--pyramid-interval", type=int, default=8)
parser.add_argument("--shrink-ratio", type=float, default=1.0)
args = parser.parse_args()


eval_dataset = load_multinews_dataset(limit=args.sample_limit)

model_path = args.model_path
print(f"Loading model from {model_path}...")
llm = LLM(model=model_path, gpu_memory_utilization=args.gpu_memory_utilization)
tokenizer = AutoTokenizer.from_pretrained(model_path)
llm.set_tokenizer(tokenizer)

model_config = AutoConfig.from_pretrained(model_path)
num_layers = model_config.num_hidden_layers
num_kv_heads = model_config.num_key_value_heads
head_dim = model_config.hidden_size // model_config.num_attention_heads
hidden_size = model_config.hidden_size

prefix_prompt = (
    "You are given several news passages. "
    "Write a one-page summary of all news.\n\nNews:\n")
query_prompt = "\n\nSummary:"

ttft_blend = []
ttft_full = []
quality_blend = init_metric_store()
quality_full = init_metric_store()
fidelity_vs_full = init_metric_store()
exact_match_vs_full = []

hardware_profiled = False
v_com = 10.0
t_gpu_ms_per_token = 0.001

llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model.model

for sample_idx, ex in enumerate(eval_dataset):
    if not hardware_profiled:
        v_com, t_gpu_ms_per_token = profile_hardware(
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            h=hidden_size,
            p=2,
        )
        hardware_profiled = True
        print(f"\n{'=' * 60}")
        print(f"[Profiling] v_com={v_com:.2f} GB/s, "
              f"t_gpu_ms_per_token={t_gpu_ms_per_token:.5f} ms")
        print(f"{'=' * 60}\n")

    prompt_chunks, input_ids, last_len = build_prompt_chunks(
        tokenizer, ex["docs"], prefix_prompt, query_prompt)
    total_len = len(input_ids)
    if total_len > 32000:
        print(f"Sample {sample_idx}: Input prompt ({total_len} tokens) "
              "exceeds limit of 32000, skipping...")
        continue

    answers = [ex["answer"]]

    reset_cachetune_state(llm_model)
    cache_fuse_metadata = llm_model.cache_fuse_metadata

    baseline_params = SamplingParams(temperature=0, max_tokens=550)
    output = llm.generate(prompt_token_ids=[input_ids],
                          sampling_params=baseline_params,
                          use_tqdm=False)
    baseline_res = output[0].outputs[0].text.strip()
    print(f"Sample {sample_idx}: Baseline generation:\n"
          f"{baseline_res[:150]}...")
    baseline_ttft = (output[0].metrics.first_token_time
                     - output[0].metrics.first_scheduled_time)
    print(f"TTFT with full prefill: {baseline_ttft:.4f}")
    ttft_full.append(baseline_ttft)
    append_metric_scores(quality_full,
                         best_rouge_against_references(baseline_res, answers))

    reset_cachetune_state(llm_model)
    cache_fuse_metadata = llm_model.cache_fuse_metadata
    cache_fuse_metadata["collect"] = True

    collect_params = SamplingParams(temperature=0, max_tokens=1)
    chunk_past_key_values = []

    print(f"Sample {sample_idx}: Generating KV Cache with CPU Offload...")
    for chunk_ids in prompt_chunks:
        llm.generate(prompt_token_ids=[chunk_ids],
                     sampling_params=collect_params,
                     use_tqdm=False)
        llm_layers = llm_model.layers
        for layer_idx in range(num_layers):
            past_key_values = llm_layers[layer_idx].self_attn.hack_kv
            temp_k = past_key_values[0][:len(chunk_ids)].clone()
            temp_v = past_key_values[1][:len(chunk_ids)].clone()

            temp_k = temp_k.to("cpu").pin_memory()
            temp_v = temp_v.to("cpu").pin_memory()

            if len(chunk_past_key_values) == 0:
                chunk_past_key_values.append([temp_k, temp_v])
            elif layer_idx >= len(chunk_past_key_values):
                chunk_past_key_values.append([temp_k, temp_v])
            else:
                chunk_past_key_values[layer_idx][0] = torch.cat(
                    (chunk_past_key_values[layer_idx][0], temp_k), dim=0)
                chunk_past_key_values[layer_idx][1] = torch.cat(
                    (chunk_past_key_values[layer_idx][1], temp_v), dim=0)

            llm_layers[layer_idx].self_attn.hack_kv = None

    for layer_idx in range(num_layers):
        if not chunk_past_key_values[layer_idx][0].is_pinned():
            chunk_past_key_values[layer_idx][0] = (
                chunk_past_key_values[layer_idx][0].contiguous().pin_memory())
        if not chunk_past_key_values[layer_idx][1].is_pinned():
            chunk_past_key_values[layer_idx][1] = (
                chunk_past_key_values[layer_idx][1].contiguous().pin_memory())

    print(f"Sample {sample_idx}: Analyzing Frequency Domain...")
    total_tokens_before_compact = chunk_past_key_values[0][0].shape[0]
    N_context = total_tokens_before_compact - last_len
    if N_context <= 0:
        print(f"[WARN] Sample {sample_idx}: invalid context length "
              f"(N_context={N_context}), skipping CacheTune.")
        continue

    l_total = optimal_l(
        N_context,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        v_com=v_com,
        t_gpu_ms_per_token=t_gpu_ms_per_token,
        r_min=0.15,
        r_max=0.50,
    )
    recomp_ratio = args.recomp_ratio
    pyramid_interval = args.pyramid_interval
    shrink_ratio = args.shrink_ratio

    layer0_v = chunk_past_key_values[0][1].to("cuda")
    layer0_k = chunk_past_key_values[0][0].to("cuda")
    context_v = layer0_v[:-last_len]

    if recomp_ratio >= 0.999:
        base_recomp_ctx = torch.arange(N_context, device=context_v.device, dtype=torch.int64)
    else:
        base_recomp_ctx, _ = get_optimized_indices(
        context_v,
        N=N_context,
        ratio=recomp_ratio,
        low_freq_pct=args.low_freq_pct,
        sink_size=args.sink_size,
        )

    total_kv_len = layer0_v.shape[0]
    suffix_start = total_kv_len - last_len
    suffix_indices = torch.arange(suffix_start,
                                  total_kv_len,
                                  device=base_recomp_ctx.device)

    current_recomp_ctx = base_recomp_ctx.clone()
    use_pyramid = args.use_pyramid and recomp_ratio < 0.95
    if use_pyramid and len(current_recomp_ctx) > 1:
        for layer_idx in range(num_layers):
            if layer_idx > 1 and layer_idx % pyramid_interval == 0 and len(current_recomp_ctx) > 1:
                layer_v_j = chunk_past_key_values[layer_idx][1].to("cuda")
                subset_v = layer_v_j[:-last_len][current_recomp_ctx]
                scores = torch.norm(subset_v.float(), p=2, dim=tuple(range(1, subset_v.ndim)))
                new_k = max(1, int(len(current_recomp_ctx) * shrink_ratio))
                top_sub = torch.topk(scores, k=new_k).indices
                current_recomp_ctx = current_recomp_ctx[top_sub].sort().values

    final_indices = torch.cat([current_recomp_ctx, suffix_indices])
    final_indices = torch.unique(final_indices, sorted=True)

    if final_indices.numel() == 0:
        raise ValueError(f"Empty recompute indices at sample {sample_idx}")
    if (final_indices.min().item() < 0
            or final_indices.max().item() >= total_kv_len):
        raise ValueError(
            f"Out-of-range recompute indices at sample {sample_idx}: "
            f"min={final_indices.min().item()}, "
            f"max={final_indices.max().item()}, total={total_kv_len}")

    imp_indices_cpu = final_indices.to("cpu")
    all_indices = torch.arange(total_kv_len)
    mask = torch.ones(total_kv_len, dtype=torch.bool)
    mask[imp_indices_cpu] = False
    non_imp_indices_cpu = all_indices[mask]

    cache_fuse_metadata["org_seq_len"] = total_kv_len
    cache_fuse_metadata["transfer_indices"] = non_imp_indices_cpu.to(
        "cuda", non_blocking=True)
    cache_fuse_metadata["precomputed_indices"] = [
        final_indices.clone() for _ in range(num_layers)
    ]

    for layer_idx in range(num_layers):
        k_compact = chunk_past_key_values[layer_idx][0][
            non_imp_indices_cpu].contiguous().pin_memory()
        v_compact = chunk_past_key_values[layer_idx][1][
            non_imp_indices_cpu].contiguous().pin_memory()
        chunk_past_key_values[layer_idx] = [k_compact, v_compact]

    cache_fuse_metadata["cpu_kv_cache"] = chunk_past_key_values

    gpu_transfer_k = []
    gpu_transfer_v = []
    for layer_idx in range(num_layers):
        gpu_transfer_k.append(
            torch.empty_like(chunk_past_key_values[layer_idx][0],
                             device="cuda"))
        gpu_transfer_v.append(
            torch.empty_like(chunk_past_key_values[layer_idx][1],
                             device="cuda"))
    cache_fuse_metadata["gpu_transfer_k"] = gpu_transfer_k
    cache_fuse_metadata["gpu_transfer_v"] = gpu_transfer_v

    kv_dtype = chunk_past_key_values[0][0].dtype
    cache_fuse_metadata["work_key"] = torch.empty(
        (total_kv_len, num_kv_heads, head_dim),
        dtype=kv_dtype,
        device="cuda",
    )
    cache_fuse_metadata["work_val"] = torch.empty(
        (total_kv_len, num_kv_heads, head_dim),
        dtype=kv_dtype,
        device="cuda",
    )

    cache_fuse_metadata["collect"] = False
    cache_fuse_metadata["check"] = True
    cache_fuse_metadata["recomp_ratio"] = recomp_ratio
    cache_fuse_metadata["fast_attention"] = True
    cache_fuse_metadata["suffix_len"] = last_len
    cache_fuse_metadata["layer_counter"] = 0
    cache_fuse_metadata["pipeline_enabled"] = True

    print(f"  -> Recompute Ratio: {recomp_ratio:.3f} "
          f"({recomp_ratio * 100:.1f}%)")
    print(f"  -> Context Tokens: {N_context}")
    print(f"  -> Recompute Tokens per Layer: {final_indices.numel() - last_len}")
    print(f"  -> Pyramid enabled: {use_pyramid}")

    cached_params = SamplingParams(temperature=0, max_tokens=550)
    output = llm.generate(prompt_token_ids=[input_ids],
                          sampling_params=cached_params,
                          use_tqdm=False)
    cached_res = output[0].outputs[0].text.strip()
    print(f"Cached generation (CacheTune):\n{cached_res[:150]}...")
    cached_ttft = (output[0].metrics.first_token_time
                   - output[0].metrics.first_scheduled_time)
    print(f"TTFT with cache: {cached_ttft:.4f}")
    ttft_blend.append(cached_ttft)
    append_metric_scores(quality_blend,
                         best_rouge_against_references(cached_res, answers))
    append_metric_scores(fidelity_vs_full,
                         compute_rouge_dict(cached_res, baseline_res))
    exact_match_vs_full.append(int(cached_res == baseline_res))

    reset_cachetune_state(llm_model)
    print("------------")

print("---------------Result Summary---------------------")
print(f"[Profiling] v_com={v_com:.2f} GB/s, "
      f"t_gpu_ms_per_token={t_gpu_ms_per_token:.5f} ms")
print(f"TTFT with cache: {np.mean(ttft_blend):.4f} seconds")
print(f"TTFT with full prefill: {np.mean(ttft_full):.4f} seconds")
print(f"Speedup: {np.mean(ttft_full) / np.mean(ttft_blend):.2f}x")
print("Quality vs reference:")
for metric in ROUGE_METRICS:
    paired_stats = summarize_paired_scores(quality_blend[metric],
                                           quality_full[metric])
    print(f"  {metric} with cache: {paired_stats['mean_cache']:.4f}")
    print(f"  {metric} with full prefill: {paired_stats['mean_full']:.4f}")
    print(f"  Mean {metric} delta (cache - full): "
          f"{paired_stats['mean_delta']:.4f}")
    print(f"  Paired win/loss/tie: {paired_stats['wins']}/"
          f"{paired_stats['losses']}/{paired_stats['ties']}")
    print(f"  Bootstrap 95% CI of mean delta: "
          f"[{paired_stats['ci_low']:.4f}, {paired_stats['ci_high']:.4f}]")
    print(f"  Accuracy Preservation: "
          f"{(paired_stats['mean_cache'] / paired_stats['mean_full'] * 100):.2f}%")

print("Fidelity vs full-prefill:")
for metric in ROUGE_METRICS:
    print(f"  {metric}: {np.mean(fidelity_vs_full[metric]):.4f}")
print(f"  Exact match rate: {np.mean(exact_match_vs_full) * 100:.2f}%")
