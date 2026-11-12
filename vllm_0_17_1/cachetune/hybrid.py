"""Text-only Qwen3.5 hybrid repair on native vLLM 0.17.1 kernels.

Every Gated DeltaNet consumes the complete ordered sequence and writes its own
native convolution/SSM state. The paper-default ``block`` policy recomputes
selected QKV, attention and MLP rows from the first full-attention layer. Only
branch *outputs* are reused; independently encoded residuals are never replayed.
"""
from __future__ import annotations

import os
from types import MethodType
from types import SimpleNamespace

import torch
from vllm.forward_context import get_forward_context
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ForCausalLM, Qwen3_5ForConditionalGeneration,
)

from .attention import SparseAttentionPlan
from .models import _sparse_attention, _probe_logits
from .hybrid_state import HybridCacheTuneState


def text_positions(positions):
    if positions.ndim == 1:
        return positions
    if (positions.ndim == 2 and positions.shape[0] == 3
            and torch.equal(positions[0], positions[1])
            and torch.equal(positions[0], positions[2])):
        return positions[0]
    raise ValueError("Hybrid CacheTune supports text positions only, not image/video MRoPE")


def split_gated_qkv(attention, qkv):
    if attention.attn_output_gate:
        qg, k, v = qkv.split(
            [attention.q_size * 2, attention.kv_size, attention.kv_size], dim=-1)
        q, gate = qg.view(qg.shape[0], attention.num_heads, -1).chunk(2, dim=-1)
        q, gate = q.reshape(qg.shape[0], -1), gate.reshape(qg.shape[0], -1)
    else:
        q, k, v = qkv.split(
            [attention.q_size, attention.kv_size, attention.kv_size], dim=-1)
        gate = None
    q = attention.q_norm(q.reshape(-1, attention.num_heads, attention.head_dim)).reshape(-1, attention.q_size)
    k = attention.k_norm(k.reshape(-1, attention.num_kv_heads, attention.head_dim)).reshape(-1, attention.kv_size)
    return q, k, v, gate


def _scale_branch(layer, name, tensor):
    if layer.layer_scale:
        scale = getattr(layer, name + "_layer_scale").to(tensor.dtype)
        tensor = tensor * (scale[0] + 1)
    return tensor


def _snapshot_recurrent(model):
    state = model._cachetune_state
    if not getattr(state, "_recurrent_probe_next", False):
        return
    context = get_forward_context()
    if not isinstance(context.attn_metadata, dict):
        return
    snapshot = {}
    for i, layer in enumerate(model.layers):
        if layer.layer_type != "linear_attention":
            continue
        gdn = layer.linear_attn
        metadata = context.attn_metadata[gdn.prefix]
        if metadata.num_prefills != 1 or metadata.num_decodes:
            raise ValueError("Recurrent probe requires one full prefill")
        indices = metadata.non_spec_state_indices_tensor.to(torch.long)
        conv, ssm = gdn.kv_cache[context.virtual_engine]
        snapshot[i] = {"conv": conv.index_select(0, indices).detach().cpu().clone(),
                       "ssm": ssm.index_select(0, indices).detach().cpu().clone()}
    state._recurrent_snapshot = snapshot
    state._recurrent_probe_next = False


def _hybrid_forward(model, input_ids, positions, intermediate_tensors=None,
                    inputs_embeds=None, **kwargs):
    state = model._cachetune_state
    if not state.collecting and not state.enabled:
        if state.prepared:
            state._diag["native_forward_calls"] = state._diag.get("native_forward_calls", 0) + 1
        out = model._cachetune_native_forward(
            input_ids, positions, intermediate_tensors, inputs_embeds)
        _snapshot_recurrent(model)
        return out
    flat_positions = text_positions(positions)
    sparse = state.note_input(input_ids, flat_positions)
    collecting = state.collecting and state._capture_armed
    if not sparse and not collecting:
        return model._cachetune_native_forward(
            input_ids, positions, intermediate_tensors, inputs_embeds)
    if intermediate_tensors is not None or inputs_embeds is not None:
        raise ValueError("Hybrid CacheTune accepts text token IDs, TP=PP=1 only")
    full_ids = state.full_attention_layer_ids
    dense_map = {actual: logical for logical, actual in enumerate(full_ids)}
    if sparse:
        attentions = [model.layers[i].self_attn for i in full_ids]
        # Preserve native MRoPE kernel dispatch also for repositioned cached K.
        cache_views = attentions
        if positions.ndim == 2:
            cache_views = [SimpleNamespace(
                q_size=a.q_size,
                rotary_emb=lambda p, q, k, a=a: a.rotary_emb(p.expand(3, -1), q, k))
                for a in attentions]
        state.begin_prefill(flat_positions, cache_views)
        state.begin_hybrid()
        selected = state.selected_indices
        selected_positions = positions.index_select(-1, selected)
        # The validated host snapshot avoids synchronizing the just-started H2D
        # copies merely to read the same selected positions back from the GPU.
        plan = SparseAttentionPlan(state._selected_cpu, state.prompt_len)
        if selected.numel() != state.prompt_len:
            plan.prepare(positions.device, None)
        boundary = full_ids[state.check_layer]
    else:
        selected = selected_positions = plan = None
        boundary = len(model.layers)
    hidden, residual = model.embed_input_ids(input_ids), None
    validate = os.environ.get("CACHETUNE_VALIDATE_PAGED", "0") == "1"
    n = input_ids.numel()
    for i, layer in enumerate(model.layers):
        dense = layer.layer_type == "full_attention"
        sparse_attention = sparse and dense and i >= boundary
        sparse_mlp = sparse and i >= boundary and (
            state.hybrid_policy == "mlp" or (dense and state.hybrid_policy == "block"))
        if sparse and (sparse_attention or sparse_mlp):
            state.enter_branch_layer(i)
        logical = dense_map.get(i)
        # Offline selection is already available at the first full-attention
        # layer. Its selected K/V must therefore be fused too: computing dense
        # QKV here would both waste projection work and silently update U.
        fusion = sparse_attention and logical >= state.check_layer
        if fusion:
            state.enter_layer(logical)
        if residual is None:
            residual = hidden
            hidden = layer.input_layernorm(hidden)
        else:
            hidden, residual = layer.input_layernorm(hidden, residual)
        # residual remains N rows, carrying the repaired upstream stream.
        if dense:
            qkv_input = hidden.index_select(0, selected) if fusion else hidden
            qkv, _ = layer.self_attn.qkv_proj(qkv_input)
            q, k, v, gate = split_gated_qkv(layer.self_attn, qkv)
            if collecting:
                state.capture(logical, k, v)
            q, k = layer.self_attn.rotary_emb(
                selected_positions if fusion else positions, q, k)
            if sparse_attention and not fusion:
                q = q.index_select(0, selected)
                if gate is not None:
                    gate = gate.index_select(0, selected)
            if fusion:
                k, v = state.merge(logical, k, v)
            if sparse_attention:
                out = _sparse_attention(layer.self_attn, q, k, v, plan, validate)
            else:
                out = layer.self_attn.attn(q, k, v)
            if gate is not None:
                out = out * torch.sigmoid(gate)
            attention_branch, _ = layer.self_attn.o_proj(out)
            attention_branch = _scale_branch(layer, "attn", attention_branch)
            if collecting:
                state.capture_branch(i, "attention", attention_branch)
            if sparse_attention:
                attention_branch = state.merge_branch(i, "attention", attention_branch)
        else:
            # Native kernels process every token in order and update both caches.
            attention_branch = torch.empty_like(hidden)
            layer.linear_attn(hidden_states=hidden, output=attention_branch)
            attention_branch = _scale_branch(layer, "attn", attention_branch)
        hidden, residual = layer.post_attention_layernorm(attention_branch, residual)
        mlp_input = hidden.index_select(0, selected) if sparse_mlp else hidden
        hidden = _scale_branch(layer, "ffn", layer.mlp(mlp_input))
        if collecting:
            state.capture_branch(i, "mlp", hidden)
        if sparse_mlp:
            hidden = state.merge_branch(i, "mlp", hidden)
        if sparse:
            state.record_layer(i, layer_type=layer.layer_type,
                               qkv_rows=int(qkv_input.shape[0]) if dense else int(n),
                               query_rows=int(q.shape[0]) if dense else int(n),
                               kv_rows=int(k.shape[0]) if dense else None,
                               mlp_rows=int(mlp_input.shape[0]), output_rows=int(hidden.shape[0]),
                               recurrent_rows=int(n) if not dense else None,
                               paged_verified=bool(validate and sparse_attention),
                               sparse_attention=plan.last_backend if sparse_attention else None)
    hidden, _ = model.norm(hidden, residual)
    _snapshot_recurrent(model)
    if sparse:
        state.finish_hybrid()
        state.finish_prefill()
    return hidden


def _attach_hybrid(outer, config):
    target = getattr(outer, "language_model", outer)
    model = target.model
    if not config.model_config.enforce_eager or not getattr(model, "do_not_compile", False):
        raise ValueError("Hybrid CacheTune requires eager mode without compilation")
    if config.parallel_config.tensor_parallel_size != 1 or config.parallel_config.pipeline_parallel_size != 1:
        raise ValueError("Hybrid CacheTune requires TP=PP=1")
    if config.model_config.hf_text_config.model_type != "qwen3_5_text":
        raise ValueError("Hybrid CacheTune currently supports dense Qwen3.5 text, not MoE")
    if any(x is not None for x in (config.model_config.quantization,
                                   config.speculative_config, config.lora_config,
                                   config.kv_transfer_config)):
        raise ValueError("Hybrid CacheTune requires unquantized weights without speculative decoding, LoRA or external connectors")
    ids = [i for i, layer in enumerate(model.layers) if layer.layer_type == "full_attention"]
    model._cachetune_state = HybridCacheTuneState(ids, num_actual_layers=len(model.layers))
    model._cachetune_native_forward = model.forward
    model.forward = MethodType(_hybrid_forward, model)
    target._cachetune_native_compute_logits = target.compute_logits
    target.compute_logits = MethodType(_probe_logits, target)


class CacheTuneQwen3_5ForConditionalGeneration(Qwen3_5ForConditionalGeneration):
    def __init__(self, *, vllm_config, prefix="model"):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        _attach_hybrid(self, vllm_config)


class CacheTuneQwen3_5ForCausalLM(Qwen3_5ForCausalLM):
    def __init__(self, *, vllm_config, prefix=""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        _attach_hybrid(self, vllm_config)
