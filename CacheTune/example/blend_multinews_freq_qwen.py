import argparse
import re
from itertools import chain
from pathlib import Path

import numpy as np
import torch
from rouge_score import rouge_scorer
from transformers import AutoConfig, AutoTokenizer
from vllm import LLM, SamplingParams


def profile_hardware(num_kv_heads=8, head_dim=128, h=4096, p=2,
                     num_trials=5):
    print("\n[Hardware Profiling] Measuring v_com (PCIe) and "
          "t_gpu_ms_per_token (GPU compute)...")

    test_n = 2048
    total_bytes = 2 * test_n * h * p
    test_tensor = torch.zeros(2 * test_n * h,
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

    ref_n = 512
    a = torch.randn(ref_n, h, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(h,
                    h + 2 * num_kv_heads * head_dim,
                    device="cuda",
                    dtype=torch.bfloat16)
    for _ in range(3):
        _ = a @ w
    torch.cuda.synchronize()

    compute_times = []
    for _ in range(num_trials):
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        start_ev.record()
        _ = a @ w
        end_ev.record()
        torch.cuda.synchronize()
        compute_times.append(start_ev.elapsed_time(end_ev))

    t_ms_gpu = np.median(compute_times)
    t_gpu_ms_per_token = t_ms_gpu / ref_n
    print(f"[Profiling Result] v_com={v_com:.2f} GB/s, "
          f"t_gpu_ms_per_token={t_gpu_ms_per_token:.5f} ms")
    return v_com, t_gpu_ms_per_token


def optimal_l(n,
              num_kv_heads=8,
              head_dim=128,
              v_com=10.0,
              t_gpu_ms_per_token=0.001,
              p=2,
              r_min=0.15,
              r_max=0.50):
    kv_bytes_per_token = 2 * num_kv_heads * head_dim * p
    v_com_bps = v_com * 1e9
    min_l = max(1, int(r_min * n))
    max_l = max(min_l, min(n, int(r_max * n)))
    step = max(1, n // 100)

    def t_layer(l):
        t_compute = l * t_gpu_ms_per_token
        t_transfer = (n - l) * kv_bytes_per_token / v_com_bps * 1000
        return max(t_compute, t_transfer)

    best_l, min_t = min_l, t_layer(min_l)
    for l in range(min_l + step, max_l + 1, step):
        t = t_layer(l)
        if t < min_t:
            min_t, best_l = t, l

    print(f"[optimal_l] N={n}, best_l={best_l} ({best_l / n:.1%}), "
          f"est. t_layer={min_t:.3f} ms")
    return best_l


def get_optimized_indices(raw_v, n, ratio, low_freq_pct=0.70, sink_size=0):
    v_float = raw_v.to(torch.float32)
    seq_len = v_float.shape[0]
    total_budget = max(1, int(seq_len * ratio))

    actual_sink = min(sink_size, n)
    sink_indices = torch.arange(0, actual_sink, device=raw_v.device)

    v_to_analyze = v_float[actual_sink:]
    analyze_len = v_to_analyze.shape[0]

    if analyze_len == 0:
        recomp = sink_indices.to(torch.int64)
        mask = torch.ones(n, dtype=torch.bool, device=raw_v.device)
        mask[recomp] = False
        reuse = torch.arange(n, device=raw_v.device)[mask]
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

    final_recomp = torch.cat(
        [sink_indices, top_indices_global]).unique().sort().values

    mask = torch.ones(n, dtype=torch.bool, device=raw_v.device)
    mask[final_recomp] = False
    final_reuse = torch.arange(n, device=raw_v.device)[mask]

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
    text = text.strip()
    if not text:
        return ""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    return "\n".join(sentences) if sentences else text


def compute_rouge_dict(prediction, reference):
    pred_norm = normalize_summary_for_rouge_lsum(prediction)
    ref_norm = normalize_summary_for_rouge_lsum(reference)
    scores = ROUGE_SCORER.score(ref_norm, pred_norm)
    return {metric: scores[metric].fmeasure for metric in ROUGE_METRICS}


def best_rouge_against_references(prediction, references):
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


def get_driver_model(llm_obj):
    executor = llm_obj.llm_engine.model_executor
    driver_worker = executor.driver_worker
    if hasattr(driver_worker, "model_runner"):
        return driver_worker.model_runner.model.model
    if hasattr(driver_worker, "worker") and driver_worker.worker is not None:
        return driver_worker.worker.model_runner.model.model
    raise RuntimeError("Unsupported driver worker type: cannot access model.")


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


def split_leading_bos(tokenizer, text):
    ids = tokenizer.encode(text)
    bos_token_id = tokenizer.bos_token_id
    if bos_token_id is not None and ids and ids[0] == bos_token_id:
        return ids[1:], bos_token_id
    return ids, None


def encode_without_leading_bos(tokenizer, text):
    ids, _ = split_leading_bos(tokenizer, text)
    return ids


def build_prompt_chunks(tokenizer, docs, prefix_prompt, query_prompt):
    prefix_ids, prefix_bos_token_id = split_leading_bos(tokenizer,
                                                        prefix_prompt)
    doc_ids = [
        encode_without_leading_bos(
            tokenizer, f"Document {idx + 1}:\n{doc}\n\n")
        for idx, doc in enumerate(docs)
    ]
    query_ids = encode_without_leading_bos(tokenizer, query_prompt)

    first_chunk = ([prefix_bos_token_id] + prefix_ids
                   if prefix_bos_token_id is not None else prefix_ids)
    prompt_chunks = [first_chunk] + doc_ids + [query_ids]
    input_ids = list(chain.from_iterable(prompt_chunks))
    last_len = len(query_ids)
    return prompt_chunks, input_ids, last_len


parser = argparse.ArgumentParser(description="CacheTune MultiNews Qwen TP")
parser.add_argument("--model-path", type=str,
                    default="/path/model/Qwen2.5-32B-Instruct")
parser.add_argument("--tensor-parallel-size", type=int, default=2)
parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
parser.add_argument("--max-model-len", type=int, default=8192)
parser.add_argument("--enforce-eager", action="store_true")
parser.add_argument("--max-num-seqs", type=int, default=1)
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

model_config = AutoConfig.from_pretrained(model_path)
num_layers = model_config.num_hidden_layers
global_num_kv_heads = model_config.num_key_value_heads
head_dim = model_config.hidden_size // model_config.num_attention_heads
hidden_size = model_config.hidden_size

driver_model = get_driver_model(llm)
local_num_kv_heads = driver_model.layers[0].self_attn.num_kv_heads

print("\n[Auto-Detect] Qwen Model Config:")
print(f"  num_layers: {num_layers}")
print(f"  hidden_size: {hidden_size}")
print(f"  num_kv_heads: {global_num_kv_heads}")
print(f"  local_num_kv_heads (per rank): {local_num_kv_heads}")
print(f"  head_dim: {head_dim}\n")

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

for sample_idx, ex in enumerate(eval_dataset):
    if not hardware_profiled:
        v_com, t_gpu_ms_per_token = profile_hardware(
            num_kv_heads=local_num_kv_heads,
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
    total_input_len = len(input_ids)
    if total_input_len > args.max_model_len:
        print(f"Sample {sample_idx}: Input prompt ({total_input_len} tokens) "
              f"exceeds max_model_len={args.max_model_len}, skipping...")
        continue

    answers = [ex["answer"]]

    run_on_all_workers(llm, "cachetune_reset_state")

    baseline_params = SamplingParams(temperature=0, max_tokens=550)
    output = llm.generate(prompt_token_ids=[input_ids],
                          sampling_params=baseline_params,
                          use_tqdm=False)
    baseline_res = output[0].outputs[0].text.strip()
    print(f"Sample {sample_idx}: Baseline generation:\n"
          f"{baseline_res[:150]}...")
    baseline_ttft = safe_ttft(output[0])
    if baseline_ttft is None:
        print(f"[WARN] Sample {sample_idx}: missing full-prefill TTFT metrics.")
        ttft_full.append(np.nan)
    else:
        print(f"TTFT with full prefill: {baseline_ttft:.4f}")
        ttft_full.append(baseline_ttft)
    append_metric_scores(quality_full,
                         best_rouge_against_references(baseline_res, answers))

    run_on_all_workers(llm, "cachetune_reset_state")
    run_on_all_workers(
        llm,
        "cachetune_set_flags",
        updates={"collect": True, "check": False, "attn_bias": None},
        reset_pipeline=True,
    )

    collect_params = SamplingParams(temperature=0, max_tokens=1)
    print(f"Sample {sample_idx}: Generating KV Cache with CPU Offload...")
    for i, chunk_ids in enumerate(prompt_chunks):
        llm.generate(prompt_token_ids=[chunk_ids],
                     sampling_params=collect_params,
                     use_tqdm=False)
        run_on_all_workers(
            llm,
            "cachetune_append_chunk_kv",
            chunk_len=len(chunk_ids),
            s_start_len=len(prompt_chunks[0]),
            s_start_1_len=0,
            is_first_chunk=(i == 0),
        )

    run_on_all_workers(llm, "cachetune_finalize_chunk_kv")
    llm_model = get_driver_model(llm)
    cache_fuse_metadata = llm_model.cache_fuse_metadata
    chunk_past_key_values = cache_fuse_metadata["cpu_kv_cache"]

    print(f"Sample {sample_idx}: Analyzing Frequency Domain...")
    total_kv_len = chunk_past_key_values[0][0].shape[0]
    if total_kv_len != total_input_len:
        raise RuntimeError(
            f"Prompt/KV length mismatch at sample {sample_idx}: "
            f"prompt={total_input_len} vs kv={total_kv_len}")

    n_context = total_kv_len - last_len
    if n_context <= 0:
        print(f"[WARN] Sample {sample_idx}: invalid context length "
              f"(N_context={n_context}), skipping CacheTune.")
        continue

    _ = optimal_l(
        n_context,
        num_kv_heads=local_num_kv_heads,
        head_dim=head_dim,
        v_com=v_com,
        t_gpu_ms_per_token=t_gpu_ms_per_token,
        r_min=0.15,
        r_max=0.50,
    )
    recomp_ratio = args.recomp_ratio

    layer0_v = chunk_past_key_values[0][1].to(
        torch.device("cuda", torch.cuda.current_device()))
    context_v = layer0_v[:-last_len]

    if recomp_ratio >= 0.999:
        base_recomp_ctx = torch.arange(n_context,
                                       device=context_v.device,
                                       dtype=torch.int64)
    else:
        base_recomp_ctx, _ = get_optimized_indices(
            context_v,
            n=n_context,
            ratio=recomp_ratio,
            low_freq_pct=args.low_freq_pct,
            sink_size=args.sink_size,
        )

    suffix_start = total_kv_len - last_len
    suffix_indices = torch.arange(suffix_start,
                                  total_kv_len,
                                  device=base_recomp_ctx.device)

    current_recomp_ctx = base_recomp_ctx.clone()
    use_pyramid = args.use_pyramid and recomp_ratio < 0.95
    if use_pyramid and len(current_recomp_ctx) > 1:
        for layer_idx in range(num_layers):
            if (layer_idx > 1 and layer_idx % args.pyramid_interval == 0
                    and len(current_recomp_ctx) > 1):
                layer_v_j = chunk_past_key_values[layer_idx][1].to(
                    torch.device("cuda", torch.cuda.current_device()))
                subset_v = layer_v_j[:-last_len][current_recomp_ctx]
                scores = torch.norm(subset_v.float(),
                                    p=2,
                                    dim=tuple(range(1, subset_v.ndim)))
                new_k = max(1, int(len(current_recomp_ctx) * args.shrink_ratio))
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

    run_on_all_workers(
        llm,
        "cachetune_prepare_from_indices",
        final_indices_cpu=final_indices.to("cpu"),
        last_len=last_len,
        recomp_ratio=float(recomp_ratio),
        check_layers=[1],
    )

    print(f"  -> Recompute Ratio: {recomp_ratio:.3f} "
          f"({recomp_ratio * 100:.1f}%)")
    print(f"  -> Context Tokens: {n_context}")
    print(f"  -> Recompute Tokens per Layer: {final_indices.numel() - last_len}")
    print(f"  -> Pyramid enabled: {use_pyramid}")

    cached_params = SamplingParams(temperature=0, max_tokens=550)
    output = llm.generate(prompt_token_ids=[input_ids],
                          sampling_params=cached_params,
                          use_tqdm=False)
    cached_res = output[0].outputs[0].text.strip()
    print(f"Cached generation (CacheTune):\n{cached_res[:150]}...")
    cached_ttft = safe_ttft(output[0])
    if cached_ttft is None:
        print(f"[WARN] Sample {sample_idx}: missing cached TTFT metrics.")
        ttft_blend.append(np.nan)
    else:
        print(f"TTFT with cache: {cached_ttft:.4f}")
        ttft_blend.append(cached_ttft)
    append_metric_scores(quality_blend,
                         best_rouge_against_references(cached_res, answers))
    append_metric_scores(fidelity_vs_full,
                         compute_rouge_dict(cached_res, baseline_res))
    exact_match_vs_full.append(int(cached_res == baseline_res))

    run_on_all_workers(llm, "cachetune_reset_state")
    print("------------")

print("---------------Result Summary---------------------")
print(f"[Profiling] v_com={v_com:.2f} GB/s, "
      f"t_gpu_ms_per_token={t_gpu_ms_per_token:.5f} ms")
ttft_blend_mean = np.nanmean(ttft_blend) if len(ttft_blend) else float("nan")
ttft_full_mean = np.nanmean(ttft_full) if len(ttft_full) else float("nan")
print(f"TTFT with cache: {ttft_blend_mean:.4f} seconds")
print(f"TTFT with full prefill: {ttft_full_mean:.4f} seconds")
if (np.isfinite(ttft_blend_mean) and np.isfinite(ttft_full_mean)
        and ttft_blend_mean > 0):
    print(f"Speedup: {ttft_full_mean / ttft_blend_mean:.2f}x")
else:
    print("Speedup: N/A")
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
    if paired_stats["mean_full"] > 0:
        print(f"  Accuracy Preservation: "
              f"{(paired_stats['mean_cache'] / paired_stats['mean_full'] * 100):.2f}%")
    else:
        print("  Accuracy Preservation: N/A")

print("Fidelity vs full-prefill:")
for metric in ROUGE_METRICS:
    values = fidelity_vs_full[metric]
    print(f"  {metric}: {np.mean(values):.4f}" if values
          else f"  {metric}: N/A")
print(f"  Exact match rate: {np.mean(exact_match_vs_full) * 100:.2f}%"
      if exact_match_vs_full else "  Exact match rate: N/A")
