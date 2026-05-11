from vllm import LLM, SamplingParams
import argparse
import gc
import math
from pathlib import Path
from types import MethodType

import numpy as np
import torch
from transformers import AutoTokenizer

from utils import build_qa_prompt, compute_f1, load_dataset


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


def get_optimized_indices(raw_v, n, ratio, low_freq_pct=0.25, sink_size=0):
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

    final_recomp = torch.cat([sink_indices, top_indices_global]).unique().sort().values
    mask = torch.ones(n, dtype=torch.bool, device=raw_v.device)
    mask[final_recomp] = False
    final_reuse = torch.arange(n, device=raw_v.device)[mask]
    return final_recomp.to(torch.int64), final_reuse.to(torch.int64)


def flatten_answer_texts(answer_obj):
    if isinstance(answer_obj, str):
        return [answer_obj]
    if isinstance(answer_obj, list):
        texts = []
        for item in answer_obj:
            texts.extend(flatten_answer_texts(item))
        return texts
    return [str(answer_obj)]


def get_driver_model(llm):
    return llm.llm_engine.model_executor.driver_worker.model_runner.model.model


def install_disk_kv_prefetch(model):
    if getattr(model, "_disk_kv_prefetch_installed", False):
        return

    model._cpu_prefetch_layer = model._prefetch_layer
    model._cpu_rebuild_old_kv = model._rebuild_old_kv
    model._disk_prefetch_cpu_buffers = {}

    def _prefetch_layer_from_disk(self, layer_idx: int) -> None:
        meta = self.cache_fuse_metadata
        if not meta.get("use_disk_kv_cache", False):
            return self._cpu_prefetch_layer(layer_idx)

        disk_kv_cache = meta.get("disk_kv_cache")
        disk_kv_meta = meta.get("disk_kv_meta")
        if disk_kv_cache is None or disk_kv_meta is None:
            return self._cpu_prefetch_layer(layer_idx)
        if layer_idx >= len(self.layers) or layer_idx >= len(disk_kv_cache):
            return

        layer_meta = disk_kv_meta[layer_idx]
        k_shape = tuple(layer_meta["k_shape"])
        dtype = layer_meta["dtype"]
        n_transfer = k_shape[0]
        shape_rest = k_shape[1:]
        check_layers = meta.get("check_layers", [1])

        if layer_idx in check_layers:
            with torch.cuda.stream(self._transfer_stream):
                self._transfer_events[layer_idx].record(self._transfer_stream)
            self._prefetch_buffers[layer_idx] = ("zero", n_transfer, shape_rest, dtype)
            return

        k_path, v_path = disk_kv_cache[layer_idx]
        cpu_k = torch.load(k_path, map_location="cpu")
        cpu_v = torch.load(v_path, map_location="cpu")
        if not cpu_k.is_contiguous():
            cpu_k = cpu_k.contiguous()
        if not cpu_v.is_contiguous():
            cpu_v = cpu_v.contiguous()
        if not cpu_k.is_pinned():
            cpu_k = cpu_k.pin_memory()
        if not cpu_v.is_pinned():
            cpu_v = cpu_v.pin_memory()

        gpu_transfer_k_list = meta.get("gpu_transfer_k")
        gpu_transfer_v_list = meta.get("gpu_transfer_v")

        with torch.cuda.stream(self._transfer_stream):
            if gpu_transfer_k_list is not None and len(gpu_transfer_k_list) > layer_idx:
                k_gpu = gpu_transfer_k_list[layer_idx]
                v_gpu = gpu_transfer_v_list[layer_idx]
                k_gpu.copy_(cpu_k, non_blocking=True)
                v_gpu.copy_(cpu_v, non_blocking=True)
            else:
                k_gpu = cpu_k.to("cuda", non_blocking=True)
                v_gpu = cpu_v.to("cuda", non_blocking=True)
            self._transfer_events[layer_idx].record(self._transfer_stream)

        self._disk_prefetch_cpu_buffers[layer_idx] = (cpu_k, cpu_v)
        self._prefetch_buffers[layer_idx] = ("full", k_gpu, v_gpu)

    def _rebuild_old_kv_and_release_disk_cpu(self, layer_idx: int) -> None:
        self._cpu_rebuild_old_kv(layer_idx)
        self._disk_prefetch_cpu_buffers.pop(layer_idx, None)

    model._prefetch_layer = MethodType(_prefetch_layer_from_disk, model)
    model._rebuild_old_kv = MethodType(_rebuild_old_kv_and_release_disk_cpu, model)
    model._disk_kv_prefetch_installed = True


def save_sparse_kv_to_disk(chunk_past_key_values, transfer_indices, root_dir):
    root_dir = Path(root_dir)
    root_dir.mkdir(parents=True, exist_ok=True)

    disk_kv_cache = []
    disk_kv_meta = []
    for layer_idx, (k_cache, v_cache) in enumerate(chunk_past_key_values):
        k_sparse = k_cache[transfer_indices].contiguous()
        v_sparse = v_cache[transfer_indices].contiguous()
        path_k = root_dir / f"layer_{layer_idx}_k.pt"
        path_v = root_dir / f"layer_{layer_idx}_v.pt"
        torch.save(k_sparse, path_k)
        torch.save(v_sparse, path_v)
        disk_kv_cache.append([str(path_k), str(path_v)])
        disk_kv_meta.append({
            "k_shape": tuple(k_sparse.shape),
            "v_shape": tuple(v_sparse.shape),
            "dtype": k_sparse.dtype,
        })
        del k_sparse, v_sparse

    return disk_kv_cache, disk_kv_meta


def make_disk_gpu_buffers(disk_kv_meta):
    gpu_transfer_k = []
    gpu_transfer_v = []
    for layer_meta in disk_kv_meta:
        dtype = layer_meta["dtype"]
        gpu_transfer_k.append(torch.empty(tuple(layer_meta["k_shape"]), dtype=dtype, device="cuda"))
        gpu_transfer_v.append(torch.empty(tuple(layer_meta["v_shape"]), dtype=dtype, device="cuda"))
    return gpu_transfer_k, gpu_transfer_v


def configure_disk_kv_metadata(meta, disk_kv_cache, disk_kv_meta, transfer_indices, gpu_transfer_k, gpu_transfer_v):
    meta["use_disk_kv_cache"] = True
    meta["disk_kv_cache"] = disk_kv_cache
    meta["disk_kv_meta"] = disk_kv_meta
    meta["cpu_kv_cache"] = [None] * len(disk_kv_cache)
    meta["transfer_indices"] = transfer_indices
    meta["gpu_transfer_k"] = gpu_transfer_k
    meta["gpu_transfer_v"] = gpu_transfer_v


def disable_disk_kv_metadata(meta):
    meta["use_disk_kv_cache"] = False


def build_final_prompt_ids(tokenizer, doc_chunk_ids, s_start_1_len):
    input_ids = []
    for idx, chunk_ids in enumerate(doc_chunk_ids):
        if idx == 0:
            temp_ids = chunk_ids
        else:
            temp_ids = chunk_ids[s_start_1_len - 1:]
        input_ids += temp_ids
    return [tokenizer.bos_token_id] + input_ids


def collect_chunk_kv(llm, tokenizer, doc_chunk_ids, sampling_params, num_layer):
    cache_fuse_metadata = get_driver_model(llm).cache_fuse_metadata
    cache_fuse_metadata["collect"] = True
    cache_fuse_metadata["check"] = False
    cache_fuse_metadata["pipeline_enabled"] = False
    cache_fuse_metadata["layer_counter"] = 0

    chunk_past_key_values = []
    for idx, chunk_ids in enumerate(doc_chunk_ids):
        chunk_eval_ids = [tokenizer.bos_token_id] + chunk_ids
        llm.generate(prompt_token_ids=[chunk_eval_ids], sampling_params=sampling_params)

        llm_layers = get_driver_model(llm).layers
        for layer_idx in range(num_layer):
            past_key_values = llm_layers[layer_idx].self_attn.hack_kv
            if idx == 0:
                temp_k = past_key_values[0][:len(chunk_eval_ids)].clone()
                temp_v = past_key_values[1][:len(chunk_eval_ids)].clone()
            else:
                temp_k = past_key_values[0][1:len(chunk_ids) + 1].clone()
                temp_v = past_key_values[1][1:len(chunk_ids) + 1].clone()

            temp_k = temp_k.to("cpu").pin_memory()
            temp_v = temp_v.to("cpu").pin_memory()

            if idx == 0:
                chunk_past_key_values.append([temp_k, temp_v])
            else:
                chunk_past_key_values[layer_idx][0] = torch.cat(
                    (chunk_past_key_values[layer_idx][0], temp_k), dim=0)
                chunk_past_key_values[layer_idx][1] = torch.cat(
                    (chunk_past_key_values[layer_idx][1], temp_v), dim=0)

            llm_layers[layer_idx].self_attn.hack_kv = None

    for layer_idx in range(num_layer):
        if not chunk_past_key_values[layer_idx][0].is_pinned():
            chunk_past_key_values[layer_idx][0] = chunk_past_key_values[layer_idx][0].contiguous().pin_memory()
        if not chunk_past_key_values[layer_idx][1].is_pinned():
            chunk_past_key_values[layer_idx][1] = chunk_past_key_values[layer_idx][1].contiguous().pin_memory()

    return chunk_past_key_values


def build_recompute_plan(chunk_past_key_values, last_len, ratio, args, num_layer):
    layer_v0 = chunk_past_key_values[0][1].to("cuda")
    context_v0 = layer_v0[:-last_len]
    n_context = context_v0.shape[0]
    if n_context <= 0:
        raise ValueError(f"invalid context length: {n_context}")

    if ratio >= 0.999:
        base_recomp_ctx = torch.arange(n_context, device=context_v0.device, dtype=torch.int64)
    else:
        base_recomp_ctx, _ = get_optimized_indices(
            context_v0,
            n=n_context,
            ratio=ratio,
            low_freq_pct=args.low_freq_pct,
            sink_size=args.sink_size,
        )

    total_len = chunk_past_key_values[0][0].shape[0]
    suffix_indices = torch.arange(total_len - last_len, total_len, device=base_recomp_ctx.device)
    current_recomp_ctx = base_recomp_ctx.clone()
    precomputed_indices_list = []
    use_pyramid = args.use_pyramid and ratio < 0.95

    for layer_idx in range(num_layer):
        if (use_pyramid and layer_idx > 1 and layer_idx % args.pyramid_interval == 0
                and len(current_recomp_ctx) > 1):
            layer_v = chunk_past_key_values[layer_idx][1].to("cuda")
            subset_v = layer_v[:-last_len][current_recomp_ctx]
            scores = torch.norm(subset_v.float(), p=2, dim=tuple(range(1, subset_v.ndim)))
            new_k = max(1, int(len(current_recomp_ctx) * args.shrink_ratio))
            top_sub = torch.topk(scores, k=new_k).indices
            current_recomp_ctx = current_recomp_ctx[top_sub].sort().values
        precomputed_indices_list.append(torch.cat([current_recomp_ctx, suffix_indices]))

    all_context_indices = torch.arange(n_context, device=current_recomp_ctx.device)
    transfer_mask = torch.ones(n_context, dtype=torch.bool, device=current_recomp_ctx.device)
    transfer_mask[current_recomp_ctx] = False
    transfer_indices_cpu = all_context_indices[transfer_mask].to("cpu")
    transfer_indices_cuda = transfer_indices_cpu.to("cuda")

    return {
        "precomputed_indices": precomputed_indices_list,
        "transfer_indices_cpu": transfer_indices_cpu,
        "transfer_indices_cuda": transfer_indices_cuda,
        "total_len": total_len,
        "n_context": n_context,
        "use_pyramid": use_pyramid,
    }


def prepare_disk_cache_for_plan(chunk_past_key_values, plan, disk_dir):
    disk_kv_cache, disk_kv_meta = save_sparse_kv_to_disk(
        chunk_past_key_values, plan["transfer_indices_cpu"], disk_dir)
    gpu_transfer_k, gpu_transfer_v = make_disk_gpu_buffers(disk_kv_meta)
    return disk_kv_cache, disk_kv_meta, gpu_transfer_k, gpu_transfer_v


def configure_inference_metadata(meta, chunk_past_key_values, plan, disk_payload, recomp_ratio, last_len):
    disk_kv_cache, disk_kv_meta, gpu_transfer_k, gpu_transfer_v = disk_payload
    configure_disk_kv_metadata(
        meta,
        disk_kv_cache,
        disk_kv_meta,
        plan["transfer_indices_cuda"],
        gpu_transfer_k,
        gpu_transfer_v,
    )

    num_kv_heads = chunk_past_key_values[0][0].shape[1] if chunk_past_key_values[0][0].ndim == 3 else 8
    head_dim = chunk_past_key_values[0][0].shape[2] if chunk_past_key_values[0][0].ndim == 3 else 128
    kv_dtype = chunk_past_key_values[0][0].dtype
    total_len = plan["total_len"]

    meta["work_key"] = torch.empty((total_len, num_kv_heads, head_dim), dtype=kv_dtype, device="cuda")
    meta["work_val"] = torch.empty((total_len, num_kv_heads, head_dim), dtype=kv_dtype, device="cuda")
    meta["precomputed_indices"] = plan["precomputed_indices"]
    meta["check"] = True
    meta["collect"] = False
    meta["recomp_ratio"] = recomp_ratio
    meta["fast_attention"] = True
    meta["suffix_len"] = last_len
    meta["pipeline_enabled"] = True
    meta["layer_counter"] = 0


def cleanup_cachetune_state(meta, model):
    disable_disk_kv_metadata(meta)
    meta["pipeline_enabled"] = False
    meta["attn_bias"] = None
    meta["imp_indices"] = None
    meta["layer_counter"] = 0
    meta["cpu_kv_cache"] = None
    meta["gpu_transfer_k"] = None
    meta["gpu_transfer_v"] = None
    meta["precomputed_indices"] = None
    meta["work_key"] = None
    meta["work_val"] = None
    model._pipeline_initialized = False
    if hasattr(model, "_prefetch_buffers"):
        model._prefetch_buffers.clear()
    if hasattr(model, "_disk_prefetch_cpu_buffers"):
        model._disk_prefetch_cpu_buffers.clear()
    if hasattr(model, "_cached_sparse_positions"):
        model._cached_sparse_positions = None
    if hasattr(model, "_warmup_prefetched_layer"):
        model._warmup_prefetched_layer = None


parser = argparse.ArgumentParser(description="CacheTune MuSiQue SSD script")
parser.add_argument("--model-path", type=str, default="/path/model/Mistral-7B-Instruct-v0.3")
parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
parser.add_argument("--recomp-ratio", type=float, default=0.15)
parser.add_argument("--low-freq-pct", type=float, default=0.25)
parser.add_argument("--sink-size", type=int, default=0)
parser.add_argument("--use-pyramid", action="store_true")
parser.add_argument("--pyramid-interval", type=int, default=8)
parser.add_argument("--shrink-ratio", type=float, default=1.0)
parser.add_argument("--num-calib-samples", type=int, default=10)
parser.add_argument("--search-r-min", type=float, default=0.05)
parser.add_argument("--search-r-max", type=float, default=0.95)
parser.add_argument("--search-tol", type=float, default=0.05)
parser.add_argument("--disable-empirical-search", action="store_true")
parser.add_argument("--disk-cache-root", type=str, default="disk_offload_cache/musique")
parser.add_argument("--max-generation-tokens", type=int, default=32)
args = parser.parse_args()

eval_dataset = load_dataset("inputs/musique_s.json")
disk_cache_root = Path(args.disk_cache_root)

llm = LLM(model=args.model_path, gpu_memory_utilization=args.gpu_memory_utilization)
tokenizer = AutoTokenizer.from_pretrained(args.model_path)
llm.set_tokenizer(tokenizer)
llm_model = get_driver_model(llm)
install_disk_kv_prefetch(llm_model)

prefix_prompt = (
    "You will be asked a question after reading several passages. Please directly answer the question based on the "
    "given passages. Do NOT repeat the question. The answer should be within 5 words..\nPassages:\n"
)
query_prompt = (
    "\n\nAnswer the question directly based on the given passages. Do NOT repeat the question. "
    "The answer should be within 5 words. \nQuestion:"
)

ttft_blend = []
ttft_full = []
f1_blend = []
f1_full = []

hardware_profiled = False
v_com = 10.0
t_gpu_ms_per_token = 0.001
num_layer = 32
calibration_contexts = []
empirical_best_ratio = args.recomp_ratio if args.disable_empirical_search else None

sample_idx = 0
while sample_idx < len(eval_dataset):
    ex = eval_dataset[sample_idx]

    if not hardware_profiled:
        v_com, t_gpu_ms_per_token = profile_hardware()
        hardware_profiled = True
        print(f"\n{'=' * 60}")
        print(f"[Profiling] v_com={v_com:.2f} GB/s, t_gpu_ms_per_token={t_gpu_ms_per_token:.5f} ms")
        print(f"{'=' * 60}\n")

    answers = ex["answers"]
    doc_prompts, q_prompt = build_qa_prompt(ex, query_prompt)
    doc_chunk_ids = [tokenizer.encode(doc)[1:] for doc in doc_prompts]
    q_ids = tokenizer.encode(q_prompt)[1:]

    cache_fuse_metadata = llm_model.cache_fuse_metadata
    cleanup_cachetune_state(cache_fuse_metadata, llm_model)
    cache_fuse_metadata["check"] = False
    cache_fuse_metadata["collect"] = False

    s_start_full = tokenizer.encode(prefix_prompt)[1:]
    s_start_1_len = 1
    s_end = []
    doc_chunk_ids = [chunk_ids for chunk_ids in doc_chunk_ids]
    doc_chunk_ids = [s_start_full] + doc_chunk_ids
    doc_chunk_ids = doc_chunk_ids + [q_ids + s_end]
    last_len = len(q_ids + s_end)

    print(f"Sample {sample_idx}: Generating KV Cache with CPU Offload...")
    collect_params = SamplingParams(temperature=0, max_tokens=1)
    chunk_past_key_values = collect_chunk_kv(llm, tokenizer, doc_chunk_ids, collect_params, num_layer)
    cache_fuse_metadata["cpu_kv_cache"] = chunk_past_key_values

    final_prompt_ids = build_final_prompt_ids(tokenizer, doc_chunk_ids, s_start_1_len)
    total_len = chunk_past_key_values[0][0].shape[0]
    if len(final_prompt_ids) != total_len:
        raise RuntimeError(
            f"Prompt/KV length mismatch at sample {sample_idx}: prompt={len(final_prompt_ids)} vs kv={total_len}")

    if empirical_best_ratio is None:
        calibration_contexts.append({
            "chunk_past_key_values": chunk_past_key_values,
            "final_prompt_ids": final_prompt_ids,
            "last_len": last_len,
            "temp_N": total_len - last_len,
        })

        if (len(calibration_contexts) < args.num_calib_samples
                and sample_idx < len(eval_dataset) - 1):
            print(f"  [Calibration] Collected KV Cache for sample {sample_idx}. "
                  f"Need {args.num_calib_samples} for search...")
            sample_idx += 1
            continue

        print(f"\n[{'=' * 50}]")
        print(f"[Empirical Search] Starting SSD Golden Section Search with {len(calibration_contexts)} samples...")
        print(f"[{'=' * 50}]")
        search_sampling_params = SamplingParams(temperature=0, max_tokens=1)

        def evaluate_ttft_for_ratio(test_r):
            total_ttft = 0.0
            for ctx_idx, ctx in enumerate(calibration_contexts):
                ctx_kvs = ctx["chunk_past_key_values"]
                ctx_last_len = ctx["last_len"]
                plan = build_recompute_plan(ctx_kvs, ctx_last_len, test_r, args, num_layer)
                ratio_tag = int(round(test_r * 10000))
                disk_payload = prepare_disk_cache_for_plan(
                    ctx_kvs, plan, disk_cache_root / "search" / f"ctx_{ctx_idx}_r_{ratio_tag:05d}")
                configure_inference_metadata(
                    cache_fuse_metadata, ctx_kvs, plan, disk_payload, test_r, ctx_last_len)
                llm_model._pipeline_initialized = False

                out = llm.generate(
                    prompt_token_ids=[ctx["final_prompt_ids"]],
                    sampling_params=search_sampling_params)
                total_ttft += out[0].metrics.first_token_time - out[0].metrics.first_scheduled_time

                cleanup_cachetune_state(cache_fuse_metadata, llm_model)
                del disk_payload
                gc.collect()
                torch.cuda.empty_cache()

            avg_ttft = total_ttft / len(calibration_contexts)
            print(f"  [GSS Evaluation] Ratio: {test_r:.3f} | "
                  f"Avg SSD TTFT over {len(calibration_contexts)} samples: {avg_ttft:.4f} s")
            return avg_ttft

        first_ctx = calibration_contexts[0]
        ctx_n = first_ctx["temp_N"]
        ctx_n_total = first_ctx["chunk_past_key_values"][0][0].shape[0]

        cache_fuse_metadata["check"] = False
        cache_fuse_metadata["collect"] = False
        cache_fuse_metadata["pipeline_enabled"] = False
        disable_disk_kv_metadata(cache_fuse_metadata)
        _ = llm.generate(
            prompt_token_ids=[first_ctx["final_prompt_ids"]],
            sampling_params=SamplingParams(temperature=0, max_tokens=1))
        out_full = llm.generate(
            prompt_token_ids=[first_ctx["final_prompt_ids"]],
            sampling_params=SamplingParams(temperature=0, max_tokens=1))
        full_ttft = out_full[0].metrics.first_token_time - out_full[0].metrics.first_scheduled_time
        t_c = full_ttft * 1000 / (num_layer * ctx_n_total)
        print(f"  [Profile] full_ttft   = {full_ttft * 1000:7.2f} ms  =>  t_c = {t_c:.5f} ms/layer/token")

        r_min, r_max, tol = args.search_r_min, args.search_r_max, args.search_tol
        sparse_ttft = evaluate_ttft_for_ratio(r_min)
        t_i = sparse_ttft * 1000 / (num_layer * (1 - r_min) * ctx_n)
        t_i = max(t_i, 1e-6)
        print(f"  [Profile] sparse_ttft = {sparse_ttft * 1000:7.2f} ms  =>  t_i = {t_i:.5f} ms/layer/token")

        r0 = max(r_min, min(r_max, t_i / (t_c + t_i)))
        print(f"[Roofline Prior] r0 = t_i / (t_c + t_i) = {r0:.3f}")

        invphi = (math.sqrt(5) - 1) / 2
        a, b = r_min, r_max
        mid = (a + b) / 2
        if r0 <= mid:
            x1 = r0
            x2 = a + invphi * (b - a)
        else:
            x1 = b - invphi * (b - a)
            x2 = r0

        f1 = evaluate_ttft_for_ratio(x1)
        f2 = evaluate_ttft_for_ratio(x2)
        iteration = 1
        while abs(b - a) > tol:
            print(f"  => [GSS Converging] Iteration {iteration} narrowed to bracket: [{a:.3f}, {b:.3f}]")
            if f1 < f2:
                b = x2
                x2 = x1
                f2 = f1
                x1 = b - invphi * (b - a)
                f1 = evaluate_ttft_for_ratio(x1)
            else:
                a = x1
                x1 = x2
                f1 = f2
                x2 = a + invphi * (b - a)
                f2 = evaluate_ttft_for_ratio(x2)
            iteration += 1

        empirical_best_ratio = (a + b) / 2
        best_avg_ttft = min(f1, f2)
        print(f"[{'=' * 50}]")
        print("=> Golden Section Search Converged!")
        print(f"=> Selected Best Empirical Ratio on SSD KV: {empirical_best_ratio:.3f} "
              f"(Est Avg TTFT: {best_avg_ttft:.4f} s)")
        print(f"[{'=' * 50}]\n")

        del calibration_contexts
        calibration_contexts = []
        del chunk_past_key_values
        gc.collect()
        torch.cuda.empty_cache()

        print("\n[Restart] Restarting inference from Sample 0 with the found optimal ratio...")
        sample_idx = 0
        continue

    recomp_ratio = empirical_best_ratio
    plan = build_recompute_plan(chunk_past_key_values, last_len, recomp_ratio, args, num_layer)
    disk_payload = prepare_disk_cache_for_plan(
        chunk_past_key_values, plan, disk_cache_root / "actual" / f"sample_{sample_idx:05d}")
    configure_inference_metadata(cache_fuse_metadata, chunk_past_key_values, plan, disk_payload, recomp_ratio, last_len)

    sampling_params = SamplingParams(temperature=0, max_tokens=args.max_generation_tokens)
    print(f"Sample idx: {sample_idx}")
    print(f"  -> SSD Recompute Ratio: {recomp_ratio:.3f} ({recomp_ratio * 100:.1f}%)")
    print(f"  -> Context Tokens: {plan['n_context']}")
    print(f"  -> Initial recomp tokens (L1): {len(plan['precomputed_indices'][0]) - last_len}")
    print(f"  -> Final recomp tokens (L{num_layer - 1}): {len(plan['precomputed_indices'][-1]) - last_len}")
    print(f"  -> Pyramid enabled: {plan['use_pyramid']}")

    output = llm.generate(prompt_token_ids=[final_prompt_ids], sampling_params=sampling_params)
    res = output[0].outputs[0].text.strip().split("\n")[0]
    print(f"Cached generation (CacheTune SSD): {res}")
    ttft = output[0].metrics.first_token_time - output[0].metrics.first_scheduled_time
    print(f"TTFT with SSD cache: {ttft}")
    ttft_blend.append(ttft)

    gt_answers = []
    for answer in answers:
        gt_answers.extend(flatten_answer_texts(answer))
    f1_blend.append(max(compute_f1(res, ans, tokenizer) for ans in gt_answers))

    cleanup_cachetune_state(cache_fuse_metadata, llm_model)

    sampling_params = SamplingParams(temperature=0, max_tokens=args.max_generation_tokens)
    cache_fuse_metadata["check"] = False
    cache_fuse_metadata["collect"] = False
    output = llm.generate(prompt_token_ids=[final_prompt_ids], sampling_params=sampling_params)
    res = output[0].outputs[0].text.strip().split("\n")[0]
    print(f"Normal generation: {res}")
    ttft = output[0].metrics.first_token_time - output[0].metrics.first_scheduled_time
    print(f"TTFT with full prefill: {ttft}")
    ttft_full.append(ttft)
    f1_full.append(max(compute_f1(res, ans, tokenizer) for ans in gt_answers))

    del output
    del chunk_past_key_values
    del disk_payload
    gc.collect()
    torch.cuda.empty_cache()

    print("------------")
    sample_idx += 1


print("---------------Result Summary---------------------")
print(f"[Profiling] v_com={v_com:.2f} GB/s, t_gpu_ms_per_token={t_gpu_ms_per_token:.5f} ms")
print(f"TTFT with SSD cache: {np.mean(ttft_blend):.4f} seconds")
print(f"TTFT with full prefill: {np.mean(ttft_full):.4f} seconds")
if ttft_blend and ttft_full:
    print(f"Speedup: {np.mean(ttft_full) / np.mean(ttft_blend):.2f}x")
print(f"F1 with SSD cache: {np.mean(f1_blend):.4f}")
print(f"F1 with full prefill: {np.mean(f1_full):.4f}")
if f1_full and np.mean(f1_full) != 0:
    print(f"Accuracy Preservation: {(np.mean(f1_blend) / np.mean(f1_full) * 100):.2f}%")
