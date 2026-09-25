
def run_spectral_method(argv=None):
    from spectral_runner import main
    return main(dataset='samsum', default_storage='cpu',
                is_qwen=False, argv=argv)


if __name__ == "__main__":
    from spectral_dispatch import dispatch_if_requested as _dispatch_spectral_method
    _dispatch_spectral_method(run_spectral_method)

from vllm import LLM, SamplingParams
import torch
import json
import numpy as np
from transformers import AutoTokenizer
from utils import load_dataset, normalize_question, build_fewshot_prompt, compute_rl
from pathlib import Path
from itertools import chain
import time


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


def get_optimized_indices(raw_v, N, ratio, low_freq_pct=0.50, sink_size=0):
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



# --------------------------------

eval_dataset = load_dataset("inputs/samsum.json")

llm = LLM(model="/path/model/Mistral-7B-Instruct-v0.3", gpu_memory_utilization=0.85)
tokenizer = AutoTokenizer.from_pretrained("/path/model/Mistral-7B-Instruct-v0.3")
llm.set_tokenizer(tokenizer)

prefix_prompt = "Summarize the dialogue into a few short sentences. The following are some examples.\n\n"

ttft_blend = []
ttft_full = []
rl_blend = []
rl_full = []

max_ctx_len = 3400

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
    doc_prompts, q_prompt = build_fewshot_prompt(ex)
    doc_chunk_ids = [tokenizer.encode(doc)[1:] for doc in doc_prompts]
    q_ids = tokenizer.encode(q_prompt)[1:]

    # drop last few-shot examples if exceeding max_ctx_len
    while len(list(chain.from_iterable(doc_chunk_ids))) > max_ctx_len:
        del_idx = int(len(doc_chunk_ids)/2)
        del doc_chunk_ids[del_idx]
    
    # skip if all ctxs are dropped
    if len(doc_chunk_ids)==0:
        continue
                
    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0, max_tokens=1)

    # Metadata setup
    cache_fuse_metadata = llm.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata
    cache_fuse_metadata['collect'] = False
    cache_fuse_metadata['check'] = False
    cache_fuse_metadata['attn_bias'] = None

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
        shift += len(doc_chunk_ids[i])
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
    
    N_context = chunk_past_key_values[0][0].shape[0] - last_len
    total_len = chunk_past_key_values[0][0].shape[0]
    suffix_indices = torch.arange(total_len - last_len, total_len, device="cuda")
    
    freq_ratio = 0.15
    
    l_total = optimal_l(N_context, num_kv_heads=8, head_dim=128, v_com=v_com, t_gpu_ms_per_token=t_gpu_ms_per_token)
    
    layer_v0 = chunk_past_key_values[0][1].to("cuda")
    context_v0 = layer_v0[:-last_len]
    base_recomp_ctx, base_reuse_ctx = get_optimized_indices(
        context_v0, N=N_context, ratio=freq_ratio, low_freq_pct=0.50, sink_size=0
    )
    
    num_base_recomp = len(base_recomp_ctx)
    num_extra_recomp = 0.0##max(0, l_total - num_base_recomp)
    
    if num_extra_recomp > 0:
        actual_extra = min(num_extra_recomp, len(base_reuse_ctx))
        step = len(base_reuse_ctx) / actual_extra
        extra_indices_idx = [int(i * step) for i in range(actual_extra)]
        hw_recomp_ctx = base_reuse_ctx[extra_indices_idx]
        
        mask = torch.ones(len(base_reuse_ctx), dtype=torch.bool, device=base_reuse_ctx.device)
        mask[extra_indices_idx] = False
        final_transfer_ctx = base_reuse_ctx[mask]
        
        total_recomp_ctx = torch.cat([base_recomp_ctx, hw_recomp_ctx]).sort().values
    else:
        total_recomp_ctx = base_recomp_ctx
        final_transfer_ctx = base_reuse_ctx

    cache_fuse_metadata['org_seq_len'] = total_len
    
    precomputed_indices_list = []
    for j in range(num_layer):
        precomputed_indices_list.append(torch.cat([total_recomp_ctx, suffix_indices]))
    
    transfer_indices = torch.cat([final_transfer_ctx, suffix_indices]).cpu()
    cuda_transfer_indices = transfer_indices.to('cuda')
    
    cpu_sparse_k = []
    cpu_sparse_v = []
    gpu_transfer_k = []
    gpu_transfer_v = []
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

    input_ids = []
    for i in range(len(doc_chunk_ids)):
        if i == 0:
            temp_ids = doc_chunk_ids[i]
        else:
            temp_ids = doc_chunk_ids[i][s_start_1_len-1:]
        input_ids += temp_ids
        
    input_prompt = tokenizer.decode(input_ids)
    
    sampling_params = SamplingParams(temperature=0, max_tokens=128)
    cache_fuse_metadata["check"] = True
    cache_fuse_metadata['collect'] = False
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

