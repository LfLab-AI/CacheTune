"""Dense Transformer adapters retaining the native vLLM V1 worker and decoder.

Weight loading, normal full recompute, and every decode step remain upstream.
Only independent-chunk collection and the explicitly armed sparse prefill use
the loop below. All linear/norm/MLP/RoPE and paged-cache operators are vLLM's.
"""
from __future__ import annotations

import os
from types import MethodType

import torch
from vllm.model_executor.models.llama import LlamaForCausalLM
from vllm.model_executor.models.mistral import MistralForCausalLM
from vllm.model_executor.models.qwen2 import Qwen2ForCausalLM
from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM
from vllm.model_executor.layers.attention.attention import (
    get_attention_context, unified_kv_cache_update,
)

from . import register_models
from .attention import SparseAttentionPlan
from .state import CacheTuneState


def _validate_paged(attention, key, value):
    """Opt-in integration assertion; excluded from performance runs."""
    _, _, cache, slots = get_attention_context(attention.layer_name)
    valid = slots >= 0
    slots = slots[valid]
    block_size = cache.shape[2]
    k_read = cache[0][slots // block_size, slots % block_size]
    v_read = cache[1][slots // block_size, slots % block_size]
    if not torch.equal(k_read, key[valid]) or not torch.equal(v_read, value[valid]):
        raise AssertionError("Native paged KV differs from the full merged prompt")


def _sparse_attention(attn, q, k, v, plan, validate):
    native = attn.attn
    metadata, _, cache, slots = get_attention_context(native.layer_name)
    n = plan.full_length
    if (metadata is None or slots is None or slots.numel() != n
            or metadata.num_actual_tokens != n
            or metadata.query_start_loc.numel() != 2
            or metadata.max_query_len != n or metadata.max_seq_len != n):
        raise ValueError("CacheTune requires one complete unshared V1 prefill")
    if native.kv_cache_dtype not in ("auto", "bfloat16", "float16"):
        raise ValueError("CacheTune requires unquantized KV cache")
    if native.attn_backend.forward_includes_kv_cache_update:
        raise ValueError("Use the FLASH_ATTN backend for this CacheTune adapter")
    k = k.reshape(n, attn.num_kv_heads, attn.head_dim)
    v = v.reshape(n, attn.num_kv_heads, attn.head_dim)
    q_heads = q.reshape(-1, attn.num_heads, attn.head_dim)
    window = getattr(native, "sliding_window", None)
    full_query = plan.num_queries == n
    force_flashinfer = os.environ.get("CACHETUNE_FORCE_FLASHINFER", "0") == "1"
    compare = full_query and os.environ.get("CACHETUNE_COMPARE_DENSE", "0") == "1"
    plan.last_comparison = None
    if full_query and not force_flashinfer:
        # With all positions selected, the native dense causal kernel is exact
        # for the requested shape. Retain native floating-point behavior here.
        out = native(q, k, v)
        plan.last_backend = "Native FlashAttention full-query path"
        if validate:
            _validate_paged(native, k, v)
        if compare:
            from .diagnostics import compare_attention
            other = plan.run(q_heads, k, v, scale=attn.scaling, window=window)
            plan.last_comparison = compare_attention(
                q_heads, k, v, out.reshape_as(q_heads), other, attn.scaling, window)
        return out
    # Full N rows go to full N slots. The native op must never receive M rows
    # against N slot entries, even though the attention query has only M rows.
    unified_kv_cache_update(k, v, native.layer_name)
    if validate:
        _validate_paged(native, k, v)
    out = plan.run(q_heads, k, v,
                   scale=attn.scaling, window=window)
    plan.last_backend = "FlashInfer custom original-position mask"
    if compare:
        from .diagnostics import compare_attention
        reference = native(q, k, v).reshape_as(q_heads)
        plan.last_comparison = compare_attention(
            q_heads, k, v, reference, out, attn.scaling, window)
    return out.reshape(q.shape[0], attn.q_size)


def _adapter_forward(model, input_ids, positions, intermediate_tensors=None,
                     inputs_embeds=None, **kwargs):
    state = model._cachetune_state
    sparse = state.note_input(input_ids, positions)
    collecting = state.collecting and state._capture_armed
    debugger = getattr(model, "_cachetune_debug", None)
    debug_prefill = debugger is not None and getattr(model, "_cachetune_debug_armed", False)
    if debug_prefill:
        if positions.ndim != 1 or not torch.equal(
                positions, torch.arange(positions.numel(), device=positions.device,
                                        dtype=positions.dtype)):
            raise ValueError("Diagnostic export requires one complete prefill at position zero")
    if not sparse and not collecting and not debug_prefill:
        if state.prepared:
            state._diag["native_forward_calls"] = state._diag.get("native_forward_calls", 0) + 1
        return model._cachetune_native_forward(
            input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs)
    if intermediate_tensors is not None or inputs_embeds is not None:
        raise ValueError("CacheTune collection/repair accepts single-rank text inputs only")
    if getattr(model, "aux_hidden_state_layers", ()):
        raise ValueError("Speculative auxiliary hidden states are unsupported in CacheTune")
    attentions = [layer.self_attn for layer in model.layers]
    if sparse:
        state.begin_prefill(positions, attentions)
        selected = state.selected_indices
        selected_positions = positions.index_select(0, selected)
        plan = SparseAttentionPlan(selected, state.prompt_len)
        if (plan.num_queries != plan.full_length
                or os.environ.get("CACHETUNE_FORCE_FLASHINFER", "0") == "1"
                or os.environ.get("CACHETUNE_COMPARE_DENSE", "0") == "1"):
            for attention in attentions:
                plan.prepare(positions.device, getattr(attention.attn, "sliding_window", None))
    else:
        selected = selected_positions = plan = None
    hidden = model.embed_input_ids(input_ids)
    residual = None
    validate = os.environ.get("CACHETUNE_VALIDATE_PAGED", "0") == "1"
    for i, layer in enumerate(model.layers):
        is_sparse_layer = sparse and i >= state.check_layer
        is_fusion_layer = sparse and i > state.check_layer
        if is_fusion_layer:
            state.enter_layer(i)
        if residual is None:
            residual = hidden
            hidden = layer.input_layernorm(hidden)
        else:
            hidden, residual = layer.input_layernorm(hidden, residual)
        qkv_rows = hidden.shape[0]
        attention = layer.self_attn
        qkv, _ = attention.qkv_proj(hidden)
        q, k, v = qkv.split([attention.q_size, attention.kv_size, attention.kv_size], dim=-1)
        # Qwen3 caches the normalized K, still before RoPE. Its per-head norm
        # cannot be omitted or applied again when cached K changes position.
        if hasattr(attention, "q_norm"):
            q = attention.q_norm(q.reshape(-1, attention.num_heads, attention.head_dim)).reshape(q.shape)
            k = attention.k_norm(k.reshape(-1, attention.num_kv_heads, attention.head_dim)).reshape(k.shape)
        state.capture(i, k, v)
        layer_positions = selected_positions if is_fusion_layer else positions
        q, k = attention.rotary_emb(layer_positions, q, k)
        if getattr(attention, "do_llama_4_scaling", False):
            q = (q * attention._get_llama_4_attn_scale(layer_positions)).to(q.dtype)
        if debug_prefill:
            debugger.capture_projected_qkv(i, q, k, v)
        if sparse and i == state.check_layer:
            q = q.index_select(0, selected)
            residual = residual.index_select(0, selected)
        if is_fusion_layer:
            k, v = state.merge(i, k, v)
        if debug_prefill:
            debugger.capture_after_rope(
                i, q, k, selected_positions if is_sparse_layer else positions,
                int(positions.numel()), attention.num_heads, attention.num_kv_heads,
                attention.head_dim, attention.scaling)
        if is_sparse_layer:
            out = _sparse_attention(attention, q, k, v, plan, validate)
        else:
            out = attention.attn(q, k, v)
        hidden, _ = attention.o_proj(out)
        hidden, residual = layer.post_attention_layernorm(hidden, residual)
        hidden = layer.mlp(hidden)
        if sparse:
            state.record_layer(i, qkv_rows=int(qkv_rows), query_rows=int(q.shape[0]),
                               kv_rows=int(k.shape[0]), mlp_rows=int(hidden.shape[0]),
                               paged_verified=bool(validate and is_sparse_layer),
                               backend=type(attention.attn.impl).__name__,
                               sparse_attention=plan.last_backend if is_sparse_layer else None,
                               attention_comparison=plan.last_comparison if is_sparse_layer else None)
    hidden, _ = model.norm(hidden, residual)
    if sparse:
        # V1 samples hidden_states[logits_indices] in original prompt coordinates.
        # Only the final boundary is expanded; all intermediate MLPs stay sparse.
        expanded = hidden.new_zeros((state.prompt_len, hidden.shape[-1]))
        expanded.index_copy_(0, selected, hidden)
        state.finish_prefill()
        if debug_prefill:
            model._cachetune_debug_armed = False
        return expanded
    if debug_prefill:
        model._cachetune_debug_armed = False
    return hidden


def _attach(outer, config):
    model = outer.model
    if not config.model_config.enforce_eager or not getattr(model, "do_not_compile", False):
        raise ValueError("CacheTune requires enforce_eager=True with compilation disabled")
    parallel = config.parallel_config
    if parallel.tensor_parallel_size != 1 or parallel.pipeline_parallel_size != 1:
        raise ValueError("CacheTune currently supports TP=PP=1")
    if config.model_config.quantization is not None:
        raise ValueError("CacheTune migration currently supports unquantized weights")
    if config.speculative_config is not None:
        raise ValueError("CacheTune does not support speculative decoding")
    if config.lora_config is not None or config.kv_transfer_config is not None:
        raise ValueError("CacheTune's current runner does not combine LoRA or external KV connectors")
    if not getattr(config.model_config.hf_text_config, "is_causal", True):
        raise ValueError("CacheTune requires decoder causal attention")
    for layer in model.layers:
        if getattr(layer, "ada_rms_norm_t_cond", None) is not None:
            raise ValueError("Conditional Mistral layers require a separate CacheTune adapter")
        if getattr(layer.self_attn, "dual_chunk_attention_config", None):
            raise ValueError("Dual-chunk attention is outside the verified CacheTune path")
    model._cachetune_state = CacheTuneState(len(model.layers), check_layer=1)
    model._cachetune_native_forward = model.forward
    model.forward = MethodType(_adapter_forward, model)
    outer._cachetune_native_compute_logits = outer.compute_logits
    outer.compute_logits = MethodType(_probe_logits, outer)


def _probe_logits(outer, hidden_states):
    logits = outer._cachetune_native_compute_logits(hidden_states)
    state = outer.model._cachetune_state
    if getattr(state, "_probe_next", False):
        state._probe_logits = logits.detach().float().cpu().clone()
        state._probe_next = False
    return logits


class CacheTuneLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, *, vllm_config, prefix=""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        _attach(self, vllm_config)


class CacheTuneMistralForCausalLM(MistralForCausalLM):
    def __init__(self, *, vllm_config, prefix=""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        _attach(self, vllm_config)


class CacheTuneQwen3ForCausalLM(Qwen3ForCausalLM):
    def __init__(self, *, vllm_config, prefix=""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        _attach(self, vllm_config)


class CacheTuneQwen2ForCausalLM(Qwen2ForCausalLM):
    """Native Qwen2/Qwen2.5 dense decoder with CacheTune sparse prefill."""
    def __init__(self, *, vllm_config, prefix=""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        _attach(self, vllm_config)
