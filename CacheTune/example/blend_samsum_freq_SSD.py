
# BEGIN CACHETUNE PAPER DISPATCH
def run_paper_method(argv=None):
    """Run the opt-in ICLR 2027 method; the original path remains the default."""
    from paper_runner import main
    return main(dataset='samsum', default_storage='disk',
                is_qwen=False, argv=argv)


if __name__ == "__main__":
    from paper_dispatch import dispatch_if_requested as _dispatch_paper_method
    _dispatch_paper_method(run_paper_method)
# END CACHETUNE PAPER DISPATCH

from vllm import LLM, SamplingParams
import torch
import json
import numpy as np
from transformers import AutoTokenizer
from utils import load_dataset, normalize_question, build_fewshot_prompt, compute_rl
from pathlib import Path
from itertools import chain
import time
import gc
from types import MethodType


def profile_hardware(num_kv_heads=8, head_dim=128, h=4096, p=2, num_trials=5):
    """Measure PCIe transfer bandwidth and GPU compute latency."""
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
        transfer_times.append(start_ev.elapsed_time(end_ev))  # ms
        del gpu_t

    t_ms = np.median(transfer_times)
    v_com = (total_bytes / (t_ms / 1000.0)) / 1e9  # GB/s

    ref_N = 512
    A = torch.randn(ref_N, h, device="cuda", dtype=torch.bfloat16)
    W = torch.randn(h, h + 2 * num_kv_heads * head_dim, device="cuda", dtype=torch.bfloat16)
    # warmup
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


def optimal_l(N, num_kv_heads=8, head_dim=128, v_com=10.0, t_gpu_ms_per_token=0.001, p=2, r_min=0.15, r_max=0.0):
    """Choose the recomputation budget that balances compute and transfer time."""
    kv_bytes_per_token = 2 * num_kv_heads * head_dim * p  # K+V bytes per token (bfloat16)
    v_com_Bps = v_com * 1e9  # bytes/s
    min_l = max(1, int(r_min * N))
    max_l = max(min_l, min(N, int(r_max * N)))
    step  = max(1, N // 100)

    def t_layer(l):
        t_compute  = l * t_gpu_ms_per_token                            # ms
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


def get_optimized_indices(raw_v, N, ratio, low_freq_pct=0.50, sink_size=4):
    """Return recompute and reuse indices from low-frequency value energy."""
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


DISK_KV_ROOT = Path("disk_offload_cache")


def get_driver_model(llm):
    return llm.llm_engine.model_executor.driver_worker.model_runner.model.model


def install_disk_kv_prefetch(model):
    """Patch the current model instance so cache prefetch reads KV from SSD."""
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

        gpu_transfer_k_list = meta.get("gpu_transfer_k", None)
        gpu_transfer_v_list = meta.get("gpu_transfer_v", None)

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
        if hasattr(self, "_disk_prefetch_cpu_buffers"):
            self._disk_prefetch_cpu_buffers.pop(layer_idx, None)

    model._prefetch_layer = MethodType(_prefetch_layer_from_disk, model)
    model._rebuild_old_kv = MethodType(_rebuild_old_kv_and_release_disk_cpu, model)
    model._disk_kv_prefetch_installed = True


def save_sparse_kv_to_disk(chunk_past_key_values, transfer_indices, root_dir):
    """Persist sparse reusable KV to SSD and return paths plus shape metadata."""
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
        gpu_transfer_k.append(
            torch.empty(tuple(layer_meta["k_shape"]), dtype=dtype, device="cuda"))
        gpu_transfer_v.append(
            torch.empty(tuple(layer_meta["v_shape"]), dtype=dtype, device="cuda"))
    return gpu_transfer_k, gpu_transfer_v


def make_disk_cpu_placeholder(num_layer):
    # The model forward only checks that cpu_kv_cache is not None before
    # starting the pipeline; patched prefetch reads disk_kv_cache instead.
    return [None] * num_layer


def configure_disk_kv_metadata(cache_fuse_metadata,
                               disk_kv_cache,
                               disk_kv_meta,
                               transfer_indices,
                               gpu_transfer_k,
                               gpu_transfer_v):
    cache_fuse_metadata["use_disk_kv_cache"] = True
    cache_fuse_metadata["disk_kv_cache"] = disk_kv_cache
    cache_fuse_metadata["disk_kv_meta"] = disk_kv_meta
    cache_fuse_metadata["cpu_kv_cache"] = make_disk_cpu_placeholder(len(disk_kv_cache))
    cache_fuse_metadata["transfer_indices"] = transfer_indices
    cache_fuse_metadata["gpu_transfer_k"] = gpu_transfer_k
    cache_fuse_metadata["gpu_transfer_v"] = gpu_transfer_v


def disable_disk_kv_metadata(cache_fuse_metadata):
    cache_fuse_metadata["use_disk_kv_cache"] = False


# --------------------------------

eval_dataset = load_dataset("inputs/samsum.json")

llm = LLM(model="/path/model/Mistral-7B-Instruct-v0.3", gpu_memory_utilization=0.95)
tokenizer = AutoTokenizer.from_pretrained("/path/model/Mistral-7B-Instruct-v0.3")
llm.set_tokenizer(tokenizer)
llm_model = get_driver_model(llm)
install_disk_kv_prefetch(llm_model)

prefix_prompt = "Summarize the dialogue into a few short sentences. The following are some examples.\n\n"

ttft_blend = []
ttft_full = []
rl_blend = []
rl_full = []

max_ctx_len = 3400

empirical_best_ratio = None

hardware_profiled = False
v_com = 10.0               # GB/s
t_gpu_ms_per_token = 0.001  # ms/token/layer

sample_idx = 0
while sample_idx < len(eval_dataset):
    ex = eval_dataset[sample_idx]
    
    # [Step 0]: Profiling
    if not hardware_profiled:
        v_com, t_gpu_ms_per_token = profile_hardware()
        hardware_profiled = True
        print(f"\n{'='*60}")
        print(f"[Profiling] v_com={v_com:.2f} GB/s, t_gpu_ms_per_token={t_gpu_ms_per_token:.5f} ms")
        print(f"{'='*60}\n")
    
    answers = ex["answers"]
    doc_prompts, q_prompt = build_fewshot_prompt(ex)
    doc_chunk_ids = [tokenizer.encode(doc)[1:] for doc in doc_prompts]
    q_ids = tokenizer.encode(q_prompt)[1:]

    # drop last few-shot examples if exceeding max_ctx_len
    while len(list(chain.from_iterable(doc_chunk_ids))) > max_ctx_len:
        del_idx = int(len(doc_chunk_ids)/2)
        del doc_chunk_ids[del_idx]
    
    # skip if all ctxs are dropped
    if len(doc_chunk_ids)==0:
        sample_idx += 1
        continue
                
    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0, max_tokens=1)

    # Metadata setup
    cache_fuse_metadata = llm.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata
    cache_fuse_metadata['collect'] = False
    cache_fuse_metadata['check'] = False
    cache_fuse_metadata['attn_bias'] = None
    disable_disk_kv_metadata(cache_fuse_metadata)

    s_start_full = tokenizer.encode(prefix_prompt)[1:]
    s_start_len = len(s_start_full) + 1

    s_start = []
    s_start_1_len = len(s_start) + 1

    s_end = []
    s_end_len = len(s_end)

    doc_chunk_ids = [s_start+chunk_ids for chunk_ids in doc_chunk_ids]
    doc_chunk_ids = [s_start_full] + doc_chunk_ids
    doc_chunk_ids = doc_chunk_ids + [s_start+q_ids+s_end]

    last_len = len(q_ids+s_end)

    cache_fuse_metadata['collect'] = True
    cache_fuse_metadata["check"] = False
    num_layer = 32
    chunk_past_key_values = []
    shift = 0
    
    print(f"Sample {sample_idx}: Generating KV Cache with CPU Offload...")
    for i in range(len(doc_chunk_ids)):
        prompts = [tokenizer.decode(doc_chunk_ids[i])]
        llm.generate(prompts, sampling_params)

        llm_layers = llm.llm_engine.model_executor.driver_worker.model_runner.model.model.layers
        for j in range(num_layer):
            past_key_values = llm_layers[j].self_attn.hack_kv
            if i == 0:
                temp_k = past_key_values[0][:s_start_len].clone()
                temp_v = past_key_values[1][:s_start_len].clone()
            else:
                temp_k = past_key_values[0][s_start_1_len:len(doc_chunk_ids[i])+1].clone()
                temp_v = past_key_values[1][s_start_1_len:len(doc_chunk_ids[i])+1].clone()
            
            temp_k = temp_k.to("cpu").pin_memory()
            temp_v = temp_v.to("cpu").pin_memory()

            if i == 0:
                chunk_past_key_values.append([temp_k, temp_v])
            else:
                chunk_past_key_values[j][0] = torch.cat((chunk_past_key_values[j][0],temp_k), dim=0)
                chunk_past_key_values[j][1] = torch.cat((chunk_past_key_values[j][1],temp_v), dim=0)

            llm_layers[j].self_attn.hack_kv = None

    for j in range(num_layer):
        if not chunk_past_key_values[j][0].is_pinned():
            chunk_past_key_values[j][0] = chunk_past_key_values[j][0].contiguous().pin_memory()
        if not chunk_past_key_values[j][1].is_pinned():
            chunk_past_key_values[j][1] = chunk_past_key_values[j][1].contiguous().pin_memory()
    cache_fuse_metadata['cpu_kv_cache'] = chunk_past_key_values

    input_ids = []
    for i in range(len(doc_chunk_ids)):
        if i == 0:
            temp_ids = doc_chunk_ids[i]
        else:
            temp_ids = doc_chunk_ids[i][s_start_1_len-1:]
        input_ids += temp_ids
        
    input_prompt = tokenizer.decode(input_ids)

    # =========================================================================
    # [NEW] Multi-Sample Empirical Offline Search (Done on first 10 samples)
    # =========================================================================
    USE_DISK_OFFLOAD = True
    temp_N = chunk_past_key_values[0][1].shape[0] - last_len

    if 'calibration_contexts' not in locals():
        calibration_contexts = []

    if empirical_best_ratio is None:
        calibration_contexts.append({
            'chunk_past_key_values': chunk_past_key_values,
            'input_prompt': input_prompt,
            'last_len': last_len,
            'temp_N': temp_N
        })
        
        NUM_CALIB_SAMPLES = 10
        if len(calibration_contexts) < NUM_CALIB_SAMPLES and sample_idx < len(eval_dataset) - 1:
            print(f"  [Calibration] Collected KV Cache for sample {sample_idx}. Need {NUM_CALIB_SAMPLES} for search...")
            continue  # Skip to next sample until we have enough for a robust search
            
        import math
        print(f"\n[{'='*50}]")
        print(f"[{'='*50}]")
        
        search_sampling_params = SamplingParams(temperature=0, max_tokens=1)
        
        def evaluate_ttft_for_ratio(test_r):
            total_ttft = 0.0
            
            for ctx_idx, ctx in enumerate(calibration_contexts):
                ctx_kvs = ctx['chunk_past_key_values']
                ctx_last_len = ctx['last_len']
                ctx_temp_N = ctx['temp_N']
                ctx_prompt = ctx['input_prompt']
                
                test_precomputed_indices = []
                
                for j in range(num_layer):
                    layer_v = ctx_kvs[j][1].to("cuda")
                    context_v = layer_v[:-ctx_last_len]
                    
                    recomp_idx, reuse_idx = get_optimized_indices(
                        context_v, N=context_v.shape[0], ratio=test_r, low_freq_pct=0.70, sink_size=4
                    )
                    
                    total_len = layer_v.shape[0]
                    suffix_indices = torch.arange(total_len - ctx_last_len, total_len, device=recomp_idx.device)
                    final_recomp = torch.cat([recomp_idx, suffix_indices])
                    test_precomputed_indices.append(final_recomp)
                    
                    if j == 0:
                        test_transfer_indices = reuse_idx.to("cpu")
                        test_cuda_transfer_indices = reuse_idx.to("cuda")
                
                ratio_tag = int(round(test_r * 10000))
                test_disk_dir = DISK_KV_ROOT / "search" / f"ctx_{ctx_idx}_r_{ratio_tag:05d}"
                test_disk_kv_cache, test_disk_kv_meta = save_sparse_kv_to_disk(
                    ctx_kvs, test_transfer_indices, test_disk_dir)
                test_gpu_transfer_k, test_gpu_transfer_v = make_disk_gpu_buffers(
                    test_disk_kv_meta)

                configure_disk_kv_metadata(
                    cache_fuse_metadata,
                    test_disk_kv_cache,
                    test_disk_kv_meta,
                    test_cuda_transfer_indices,
                    test_gpu_transfer_k,
                    test_gpu_transfer_v)
                
                kv_dtype = ctx_kvs[0][0].dtype
                cache_fuse_metadata['work_key'] = torch.empty((total_len, 8, 128), dtype=kv_dtype, device='cuda')
                cache_fuse_metadata['work_val'] = torch.empty((total_len, 8, 128), dtype=kv_dtype, device='cuda')
                cache_fuse_metadata['precomputed_indices'] = test_precomputed_indices
                cache_fuse_metadata["check"] = True
                cache_fuse_metadata['collect'] = False
                cache_fuse_metadata['recomp_ratio'] = test_r
                cache_fuse_metadata['fast_attention'] = True
                cache_fuse_metadata['suffix_len'] = ctx_last_len
                cache_fuse_metadata['pipeline_enabled'] = True
                cache_fuse_metadata['layer_counter'] = 0
                
                llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model.model
                llm_model._pipeline_initialized = False
                
                out = llm.generate([ctx_prompt], search_sampling_params)
                ttft = out[0].metrics.first_token_time - out[0].metrics.first_scheduled_time
                total_ttft += ttft
                
                # Memory cleanup for next sample
                disable_disk_kv_metadata(cache_fuse_metadata)
                del test_disk_kv_cache
                del test_disk_kv_meta
                del test_gpu_transfer_k
                del test_gpu_transfer_v
                del cache_fuse_metadata['work_key']
                del cache_fuse_metadata['work_val']
                gc.collect()
                torch.cuda.empty_cache()
                llm_model._pipeline_initialized = False
            
            avg_ttft = total_ttft / len(calibration_contexts)
            print(f"  [GSS Evaluation] Ratio: {test_r:.3f} | Avg TTFT over {len(calibration_contexts)} samples: {avg_ttft:.4f} s")
            return avg_ttft

        # ==================================================================
        # ------------------------------------------------------------------
        # Instead of synthetic micro-benchmarks (which miss scatter, RoPE,
        # kernel-launch, and DMA-startup overheads), we measure t_c and t_i
        # directly on the actual CacheTune inference pipeline using the first
        # calibration context. This yields numbers that reflect every real
        # runtime cost on the target hardware + model + code path.
        #
        # ==================================================================
        r_min, r_max = 0.05, 0.95
        tol = 0.05   # Stop when the uncertainty bracket is smaller than 5%

        print(f"\n[End-to-End Profiling] Measuring t_c and t_i on actual CacheTune pipeline...")
        first_ctx   = calibration_contexts[0]
        ctx_N       = first_ctx['temp_N']                                     # context tokens (no suffix)
        ctx_N_total = first_ctx['chunk_past_key_values'][0][0].shape[0]       # full seq tokens

        # --- Measure t_c from a full-prefill baseline (no KV reuse) ---
        cache_fuse_metadata["check"]            = False
        cache_fuse_metadata["collect"]          = False
        cache_fuse_metadata["pipeline_enabled"] = False
        disable_disk_kv_metadata(cache_fuse_metadata)
        _ = llm.generate([first_ctx['input_prompt']],
                         SamplingParams(temperature=0, max_tokens=1))         # warm-up to avoid first-call jitter
        out_full  = llm.generate([first_ctx['input_prompt']],
                                 SamplingParams(temperature=0, max_tokens=1))
        full_ttft = out_full[0].metrics.first_token_time - out_full[0].metrics.first_scheduled_time
        t_c = full_ttft * 1000 / (num_layer * ctx_N_total)                    # ms/layer/token
        print(f"  [Profile] full_ttft   = {full_ttft*1000:7.2f} ms  =>  t_c = {t_c:.5f} ms/layer/token")

        # --- Measure t_i from an I/O-bound CacheTune run at r = r_min ---
        sparse_ttft = evaluate_ttft_for_ratio(r_min)
        t_i = sparse_ttft * 1000 / (num_layer * (1 - r_min) * ctx_N)          # ms/layer/token
        t_i = max(t_i, 1e-6)                                                   # floor for safety
        print(f"  [Profile] sparse_ttft = {sparse_ttft*1000:7.2f} ms  =>  t_i = {t_i:.5f} ms/layer/token")

        # ==================================================================
        # Step B: Roofline-Warmstart Golden Section Search  (Algorithm 1)
        # ------------------------------------------------------------------
        #   Step B.1: Compute roofline prior r0 = t_i / (t_c + t_i)
        #   Step B.2: Anchor the first GSS probe at r0; place the companion
        #             probe by the standard golden-section formula on the
        #             opposite half of [r_min, r_max]. This guarantees
        #             x1 < x2 and preserves the geometric invariants for
        #             single-evaluation reuse in subsequent iterations.
        #   Step B.3: Standard GSS iterations (unchanged from textbook).
        # ==================================================================

        # ---- Step B.1: Roofline prior ----
        r0 = t_i / (t_c + t_i)
        r0 = max(r_min, min(r_max, r0))                                        # clip to semantic bounds
        print(f"[Roofline Prior] r0 = t_i / (t_c + t_i) = {r0:.3f}")

        # ---- Step B.2: Warm-start initialization ----
        invphi = (math.sqrt(5) - 1) / 2
        a, b   = r_min, r_max
        mid    = (a + b) / 2

        if r0 <= mid:
            # r0 lies in the left half: anchor it as the left probe x1.
            x1 = r0
            x2 = a + invphi * (b - a)                                          # standard right probe
        else:
            # r0 lies in the right half: anchor it as the right probe x2.
            x1 = b - invphi * (b - a)                                          # standard left probe
            x2 = r0

        f1 = evaluate_ttft_for_ratio(x1)
        f2 = evaluate_ttft_for_ratio(x2)

        # ---- Step B.3: Standard GSS refinement ----
        iteration = 1
        while abs(b - a) > tol:
            print(f"  => [GSS Converging] Iteration {iteration} narrowed to bracket: [{a:.3f}, {b:.3f}]")
            if f1 < f2:
                # Minimum lies in [a, x2]; shrink right bound and reuse f1
                b  = x2
                x2 = x1
                f2 = f1
                x1 = b - invphi * (b - a)
                f1 = evaluate_ttft_for_ratio(x1)
            else:
                # Minimum lies in [x1, b]; shrink left bound and reuse f2
                a  = x1
                x1 = x2
                f1 = f2
                x2 = a + invphi * (b - a)
                f2 = evaluate_ttft_for_ratio(x2)
            iteration += 1

        empirical_best_ratio = (a + b) / 2
        best_avg_ttft        = min(f1, f2)
        
        print(f"[{'='*50}]")
        print(f"=> Golden Section Search Converged!")
        print(f"=> Selected Best Empirical Ratio based on multiple samples: {empirical_best_ratio:.3f} (Est Avg TTFT: {best_avg_ttft:.4f} s)")
        print(f"[{'='*50}]\n")
        
        # Free up stored calibration contexts to save CPU memory
        del calibration_contexts
        gc.collect()
        
        print("\n[Restart] Restarting inference from Sample 0 with the found optimal ratio...")
        sample_idx = 0
        continue

    # --- Step 2: Frequency Analysis (Actual Run) ---
    print(f"Sample {sample_idx}: V2.0 Frequency Domain Analysis with found optimal ratio {empirical_best_ratio:.2f}...")
    freq_ratio = empirical_best_ratio
    precomputed_indices_list = []
    recomp_ratio = freq_ratio
    sink_n = 4
    
    cpu_sparse_k = []
    gpu_transfer_k = []
    gpu_transfer_v = []
    
    for j in range(num_layer):
        layer_v = chunk_past_key_values[j][1].to("cuda")
        context_v = layer_v[:-last_len]
        N_context = context_v.shape[0]

        recomp_idx, reuse_idx = get_optimized_indices(
            context_v,
            N=N_context,
            ratio=recomp_ratio,
            low_freq_pct=0.70,
            sink_size=sink_n
        )

        total_len = layer_v.shape[0]
        suffix_indices = torch.arange(total_len - last_len, total_len, device=recomp_idx.device)
        final_recomp = torch.cat([recomp_idx, suffix_indices])
        precomputed_indices_list.append(final_recomp)

        if j == 0:
            transfer_indices = reuse_idx.to("cpu")
            cuda_transfer_indices = reuse_idx.to("cuda")

    if USE_DISK_OFFLOAD:
        disk_dir = DISK_KV_ROOT / "actual" / f"sample_{sample_idx:05d}"
        disk_kv_cache, disk_kv_meta = save_sparse_kv_to_disk(
            chunk_past_key_values, transfer_indices, disk_dir)
        gpu_transfer_k, gpu_transfer_v = make_disk_gpu_buffers(disk_kv_meta)
        configure_disk_kv_metadata(
            cache_fuse_metadata,
            disk_kv_cache,
            disk_kv_meta,
            cuda_transfer_indices,
            gpu_transfer_k,
            gpu_transfer_v)
    else:
        disable_disk_kv_metadata(cache_fuse_metadata)
        for j in range(num_layer):
            k_sparse = chunk_past_key_values[j][0][transfer_indices].pin_memory()
            v_sparse = chunk_past_key_values[j][1][transfer_indices].pin_memory()
            cpu_sparse_k.append([k_sparse, v_sparse])

            gpu_transfer_k.append(torch.empty_like(k_sparse, device='cuda'))
            gpu_transfer_v.append(torch.empty_like(v_sparse, device='cuda'))

        cache_fuse_metadata['cpu_kv_cache'] = cpu_sparse_k
        cache_fuse_metadata['transfer_indices'] = cuda_transfer_indices
        cache_fuse_metadata['gpu_transfer_k'] = gpu_transfer_k
        cache_fuse_metadata['gpu_transfer_v'] = gpu_transfer_v
    
    N_total = total_len
    kv_dtype = chunk_past_key_values[0][0].dtype
    cache_fuse_metadata['work_key'] = torch.empty((N_total, 8, 128), dtype=kv_dtype, device='cuda')
    cache_fuse_metadata['work_val'] = torch.empty((N_total, 8, 128), dtype=kv_dtype, device='cuda')

    cache_fuse_metadata['precomputed_indices'] = precomputed_indices_list

    sampling_params = SamplingParams(temperature=0, max_tokens=128)
    cache_fuse_metadata["check"] = True
    cache_fuse_metadata['collect'] = False
    cache_fuse_metadata['recomp_ratio'] = recomp_ratio
    cache_fuse_metadata['fast_attention'] = True
    cache_fuse_metadata['fast_attention'] = True
    cache_fuse_metadata['suffix_len'] = last_len
    
    cache_fuse_metadata['pipeline_enabled'] = True
    
    cache_fuse_metadata['layer_counter'] = 0 

    print(f"Sample idx: {sample_idx}")
    output = llm.generate([input_prompt], sampling_params)
    res = output[0].outputs[0].text
    res = res.lstrip('\n').split('\n')[0]
    print(f"Cached generation (CacheTune): {res}")
    ttft = output[0].metrics.first_token_time-output[0].metrics.first_scheduled_time
    print(f"TTFT with cache: {ttft:.4f}")
    ttft_blend.append(ttft)
    rl = max([compute_rl(res, answer) for answer in answers]) 
    rl_blend.append(rl)
    
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model.model
    llm_model._pipeline_initialized = False
    
    if cache_fuse_metadata.get('pipeline_enabled', False):
        cache_fuse_metadata['pipeline_enabled'] = False
    disable_disk_kv_metadata(cache_fuse_metadata)

    
    sampling_params = SamplingParams(temperature=0, max_tokens=128)
    cache_fuse_metadata["check"] = False
    cache_fuse_metadata['collect'] = False
    output = llm.generate([input_prompt], sampling_params)
    res = output[0].outputs[0].text
    res = res.lstrip('\n').split('\n')[0]
    print(f"Normal generation: {res}")
    ttft = output[0].metrics.first_token_time-output[0].metrics.first_scheduled_time
    print(f"TTFT with full prefill: {ttft}")
    ttft_full.append(ttft)
    rl = max([compute_rl(res, answer) for answer in answers])
    rl_full.append(rl)
    
    cache_fuse_metadata['pipeline_enabled'] = True
    llm_model._pipeline_initialized = False
    
    print("------------")
    sample_idx += 1
    

print("---------------Result Summary---------------------")
print(f"[Profiling] v_com={v_com:.2f} GB/s, t_gpu_ms_per_token={t_gpu_ms_per_token:.5f} ms")
print(f"TTFT with cache: {np.mean(ttft_blend):.4f} seconds")
print(f"TTFT with full prefill: {np.mean(ttft_full):.4f} seconds")
print(f"Speedup: {np.mean(ttft_full) / np.mean(ttft_blend):.2f}x")
print(f"rl with cache: {np.mean(rl_blend):.4f}")
print(f"rl with full prefill: {np.mean(rl_full):.4f}")
print(f"Accuracy Preservation: {(np.mean(rl_blend) / np.mean(rl_full) * 100):.2f}%")





# def calculate_freq_indices(key, value, ratio=0.15, low_freq_pct=0.25):
#     v_float = value.to(torch.float32)
    
#     seq_len = v_float.shape[0]
    
#     v_freq = torch.fft.rfft(v_float, dim=0)
#     cutoff = int(v_freq.shape[0] * low_freq_pct)
#     if cutoff < v_freq.shape[0]:
#         v_freq[cutoff:, ...] = 0.0 
#     v_low = torch.fft.irfft(v_freq, n=seq_len, dim=0)
    
#     k_freq = torch.fft.rfft(k_float, dim=0)
#     if cutoff < k_freq.shape[0]:
#     k_low = torch.fft.irfft(k_freq, n=seq_len, dim=0)

#     score_v = torch.norm(v_low, p=2, dim=(1, 2))
#     score_k = torch.norm(k_low, p=2, dim=(1, 2))
    
#     total_score = score_v + score_k 
    
#     topk_num = int(seq_len * ratio)
#     if topk_num < 1: topk_num = 1
    
#     top_indices = torch.topk(total_score, k=topk_num).indices
#     top_indices, _ = torch.sort(top_indices)
    
#     return top_indices.to(torch.int64).to(value.device)

# def calculate_freq_indices(key, value, ratio, layer_idx, low_freq_pct=0.25, sink_size=0):
#     """
    
#     """
    
#     if layer_idx == 0:
#         return torch.tensor([], dtype=torch.int64, device=value.device)
#     # -------------------------------

#     v_float = value.to(torch.float32)
#     seq_len = v_float.shape[0]
    
#     total_budget = int(seq_len * ratio)
    
#     sink_indices = torch.arange(sink_size, device=value.device)
    
#     v_to_analyze = v_float[sink_size:] 
#     analyze_len = v_to_analyze.shape[0]
    
#     if analyze_len == 0:
#         return sink_indices

#     v_freq = torch.fft.rfft(v_to_analyze, dim=0)
    
#     cutoff = int(v_freq.shape[0] * low_freq_pct)
    
#     if cutoff < v_freq.shape[0]:
#         if v_freq.ndim == 3:
#             v_freq[cutoff:, :, :] = 0.0
#             norm_dims = (1, 2)
#         elif v_freq.ndim == 2:
#             v_freq[cutoff:, :] = 0.0
#             norm_dims = (1,)
#         else:
#             sl = [slice(None)] * v_freq.ndim
#             sl[0] = slice(cutoff, None)
#             v_freq[tuple(sl)] = 0.0
#             norm_dims = tuple(range(1, v_freq.ndim))
            
#     v_low_reconstructed = torch.fft.irfft(v_freq, n=analyze_len, dim=0)
    
#     imp_scores = torch.norm(v_low_reconstructed, p=2, dim=norm_dims)
    
#     freq_budget = total_budget 
    
#     top_indices_local = torch.topk(imp_scores, k=freq_budget).indices
    
#     top_indices_global = top_indices_local + sink_size
    
#     final_indices = torch.cat([sink_indices, top_indices_global])
#     final_indices, _ = torch.sort(final_indices)
    
#     return final_indices.to(torch.int64).to(value.device)
