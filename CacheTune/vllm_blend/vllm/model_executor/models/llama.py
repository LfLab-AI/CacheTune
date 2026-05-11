# coding=utf-8
# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only LLaMA model compatible with HuggingFace weights."""
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn
from transformers import LlamaConfig

from vllm.attention import Attention, AttentionMetadata
from vllm.config import LoRAConfig
from vllm.distributed import (get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size)
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (LinearMethodBase,
                                               MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, kv_cache_scales_loader)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import SamplerOutput
from vllm.utils import is_hip


class LlamaMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        linear_method: Optional[LinearMethodBase] = None,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size, [intermediate_size] * 2,
            bias=False,
            linear_method=linear_method)
        self.down_proj = RowParallelLinear(intermediate_size,
                                           hidden_size,
                                           bias=False,
                                           linear_method=linear_method)
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class LlamaAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        linear_method: Optional[LinearMethodBase] = None,
        bias: bool = False,
        sliding_window: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        # This will be overwritten by model initialization if we are using it.
        # N.B. currently we only support per tensor scalar scaling factors
        # & only applicable to ROCm (AMD GPU).
        # The scaling factor convention we are assuming is
        # quantized_value * scaling_factor ~= true_value
        # which is consistent with the practice of setting
        # scaling_factor = tensor_amax / FPtype_max
        self.kv_scale = 1.0

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=bias,
            linear_method=linear_method,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=bias,
            linear_method=linear_method,
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = Attention(self.num_heads,
                              self.head_dim,
                              self.scaling,
                              num_kv_heads=self.num_kv_heads,
                              sliding_window=sliding_window)
        
        self.hack_kv = []

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        
        status,
        cache_fuse_metadata,
        old_kv,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        
        # [V2.0] Collect Pre-RoPE Raw KV (before any rotation)
        if cache_fuse_metadata['collect']:
            self.hack_kv = [k.clone(), v.clone()]
        
        if status == 2:
            # ========== V2.0 Core: Scatter + Background-RoPE-Optimized Assembly ==========
            imp_indices = cache_fuse_metadata["imp_indices"]
            transfer_indices = cache_fuse_metadata.get('transfer_indices')
            if transfer_indices is None:
                transfer_indices = cache_fuse_metadata.get('non_imp_indices')
            N_total = cache_fuse_metadata['org_seq_len']
            
            k_3d = k.view(-1, self.num_kv_heads, self.head_dim)
            v_3d = v.view(-1, self.num_kv_heads, self.head_dim)
            
            # 1. Get work buffers for full KV assembly
            if 'work_key' in cache_fuse_metadata:
                full_k = cache_fuse_metadata['work_key']
                full_v = cache_fuse_metadata['work_val']
            else:
                full_k = torch.empty(
                    (N_total, self.num_kv_heads, self.head_dim),
                    dtype=k.dtype, device=k.device)
                full_v = torch.empty(
                    (N_total, self.num_kv_heads, self.head_dim),
                    dtype=v.dtype, device=v.device)

            # [Optim D] Precompute the 3D view of the transferred KV once to avoid
            # redundant stride/metadata construction per scatter call.
            old_k_3d = old_kv[0].view(-1, self.num_kv_heads, self.head_dim)
            old_v_3d = old_kv[1].view(-1, self.num_kv_heads, self.head_dim)

            # 2. Copy full KV from CPU DMA transfer
            #    (Note: old_kv[0] is ALREADY RoPE-rotated by the background stream!)
            if transfer_indices is not None:
                full_k.index_copy_(0, transfer_indices, old_k_3d)
                full_v.index_copy_(0, transfer_indices, old_v_3d)
            else:
                full_k[:] = old_k_3d
                full_v[:] = old_v_3d

            # 3. RoPE on query and newly computed keys (ONLY for imp_indices!)
            q_rotated, k_rotated = self.rotary_emb(positions, q, k)
            k_rotated_3d = k_rotated.view(-1, self.num_kv_heads, self.head_dim)
            
            # 4. Scatter recompute part (rotated newly computed k, v)
            full_k.index_copy_(0, imp_indices, k_rotated_3d)
            full_v.index_copy_(0, imp_indices, v_3d)
            
            # 5. Pass assembled rotated K and raw V to attention
            k_flat = full_k.reshape(N_total, -1)
            v_flat = full_v.reshape(N_total, -1)
            attn_output = self.attn(
                q_rotated, k_flat, v_flat,
                kv_cache, attn_metadata,
                status, cache_fuse_metadata, None,
                self.kv_scale)
        else:
            # Status 0, 1, -1: Normal RoPE to q, k
            q, k = self.rotary_emb(positions, q, k)
            attn_output = self.attn(
                q, k, v, kv_cache, attn_metadata,
                status, cache_fuse_metadata, old_kv,
                self.kv_scale)
        
        output, _ = self.o_proj(attn_output)
        return output


class LlamaDecoderLayer(nn.Module):

    def __init__(
        self,
        config: LlamaConfig,
        linear_method: Optional[LinearMethodBase] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings",
                                          8192)
        sliding_window = getattr(config, "sliding_window", None)
        # Support abacusai/Smaug-72B-v0.1 with attention_bias
        # Support internlm/internlm-7b with bias
        attention_bias = getattr(config, "attention_bias", False) or getattr(
            config, "bias", False)
        self.self_attn = LlamaAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=getattr(config, "num_key_value_heads",
                                 config.num_attention_heads),
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            linear_method=linear_method,
            bias=attention_bias,
            sliding_window=sliding_window,
        )
        self.mlp = LlamaMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            linear_method=linear_method,
        )
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
        
        status: int,
        cache_fuse_metadata: dict,
        old_kv,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
            
            # CacheTune integration start
            status=status,
            cache_fuse_metadata=cache_fuse_metadata,
            old_kv=old_kv,
            # CacheTune integration end
        )

        if status == 1:
            residual = residual[cache_fuse_metadata["imp_indices"]]
        
        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class LlamaModel(nn.Module):

    def __init__(
        self,
        config: LlamaConfig,
        linear_method: Optional[LinearMethodBase] = None,
        lora_config: Optional[LoRAConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        lora_vocab = (lora_config.lora_extra_vocab_size *
                      (lora_config.max_loras or 1)) if lora_config else 0
        self.vocab_size = config.vocab_size + lora_vocab
        self.org_vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
        )
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(config, linear_method)
            for _ in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.cache_fuse_metadata = {"check_layers":[1],
                                    "check": False,
                                    "recomp_ratios":[0.16],
                                    "recomp_ratio":0.16,
                                    "original_slot_mapping":None,
                                    "our_slot_mapping":None,
                                    "kv_cache_dtype": None,
                                    "attn_bias": None,
                                    "imp_indices": None,
                                    "org_seq_len": None,
                                    "collect": False}
        
        self.old_kvs = [[None, None] for _ in range(len(self.layers))]
        
        self._transfer_stream = torch.cuda.Stream()
        self._transfer_events = [torch.cuda.Event() for _ in range(len(self.layers))]
        
        # [Pipeline v2]: Asynchronous GPU Recomputation Stream and Events
        self._recompute_stream = torch.cuda.Stream()
        self._recompute_events = [torch.cuda.Event() for _ in range(len(self.layers))]
        
        self._pipeline_initialized = False
        self._prefetch_buffers = {}

        # [Optim F] Tracks the fusion layer that was pre-DMA'd during warmup.
        # Consumed once at iteration i == _warmup_prefetched_layer - 1 of the
        # main loop, to avoid re-issuing a redundant DMA for that layer.
        self._warmup_prefetched_layer = None

        # [Optim B] Persistent fake_q buffer reused across all background RoPE calls
        # (vLLM's rotary_emb needs both q and k; we only care about k, so q is a dummy).
        # Lazily allocated on first use to avoid hard-coding num_kv_heads/head_dim here.
        self._persistent_fake_q = None

        # [Optim C] Cache of sparse_positions (= org_pos[transfer_indices]).
        # All fusion layers of the same request share identical sparse_positions,
        # so we gather once per request rather than 31 times.
        self._cached_sparse_positions = None
        self._cached_sparse_positions_version = 0
        
    def _prefetch_layer(self, layer_idx: int) -> None:
        """Asynchronously prefetch reusable KV for one fusion layer."""
        cpu_kv_cache = self.cache_fuse_metadata.get('cpu_kv_cache', None)
        if cpu_kv_cache is None or layer_idx >= len(self.layers):
            return

        N_total    = cpu_kv_cache[layer_idx][0].shape[0]
        shape_rest = cpu_kv_cache[layer_idx][0].shape[1:]
        dtype      = cpu_kv_cache[layer_idx][0].dtype
        check_layers = self.cache_fuse_metadata.get('check_layers', [1])

        if layer_idx in check_layers:
            with torch.cuda.stream(self._transfer_stream):
                self._transfer_events[layer_idx].record(self._transfer_stream)
            self._prefetch_buffers[layer_idx] = ('zero', N_total, shape_rest, dtype)
            return

        cpu_k = cpu_kv_cache[layer_idx][0]
        cpu_v = cpu_kv_cache[layer_idx][1]
        
        gpu_transfer_k_list = self.cache_fuse_metadata.get('gpu_transfer_k', None)
        gpu_transfer_v_list = self.cache_fuse_metadata.get('gpu_transfer_v', None)
        
        with torch.cuda.stream(self._transfer_stream):
            if gpu_transfer_k_list is not None and len(gpu_transfer_k_list) > layer_idx:
                k_gpu = gpu_transfer_k_list[layer_idx]
                v_gpu = gpu_transfer_v_list[layer_idx]
                k_gpu.copy_(cpu_k, non_blocking=True)
                v_gpu.copy_(cpu_v, non_blocking=True)
            else:
                k_gpu = cpu_k.to('cuda', non_blocking=True)
                v_gpu = cpu_v.to('cuda', non_blocking=True)
            self._transfer_events[layer_idx].record(self._transfer_stream)
        self._prefetch_buffers[layer_idx] = ('full', k_gpu, v_gpu)

    def _background_rope(self, layer_idx: int) -> None:
        """[V2.0 Pipeline] Apply RoPE to the fetched old_kv in the background stream."""
        if layer_idx not in self._prefetch_buffers:
            return
        buf = self._prefetch_buffers[layer_idx]
        tag = buf[0]
        if tag == 'full':
            _, k_gpu, v_gpu = buf
            attn = self.layers[layer_idx].self_attn
            full_positions = self.cache_fuse_metadata.get('org_pos')
            transfer_indices = self.cache_fuse_metadata.get('transfer_indices')
            if transfer_indices is None:
                transfer_indices = self.cache_fuse_metadata.get('non_imp_indices')
            if full_positions is not None:
                N_transfer = k_gpu.shape[0]

                # [Optim C] Reuse sparse_positions across layers within a request
                sparse_positions = self._cached_sparse_positions
                if sparse_positions is None or sparse_positions.shape[0] != N_transfer:
                    sparse_positions = (full_positions if transfer_indices is None
                                        else full_positions[transfer_indices])
                    self._cached_sparse_positions = sparse_positions

                # [Optim B] Reuse fake_q across layers; lazily (re)allocate only when
                # shape/dtype/device changes (typically once per request)
                need_alloc = (
                    self._persistent_fake_q is None
                    or self._persistent_fake_q.shape[0] < N_transfer
                    or self._persistent_fake_q.shape[1] != attn.q_size
                    or self._persistent_fake_q.dtype != k_gpu.dtype
                    or self._persistent_fake_q.device != k_gpu.device
                )
                if need_alloc:
                    self._persistent_fake_q = torch.empty(
                        (N_transfer, attn.q_size), dtype=k_gpu.dtype, device=k_gpu.device)
                fake_q = self._persistent_fake_q[:N_transfer]

                raw_k_flat = k_gpu.reshape(N_transfer, -1)

                # Apply RoPE on the sparse CPU KV
                _, k_rotated_flat = attn.rotary_emb(sparse_positions, fake_q, raw_k_flat)
                k_rotated_3d = k_rotated_flat.view(N_transfer, attn.num_kv_heads, attn.head_dim)
                self._prefetch_buffers[layer_idx] = ('full', k_rotated_3d, v_gpu)

    def _rebuild_old_kv(self, layer_idx: int) -> None:
        """Rebuild the old KV handle from the prefetch buffer."""
        if layer_idx not in self._prefetch_buffers:
            return
        buf = self._prefetch_buffers.pop(layer_idx)
        tag = buf[0]

        if tag == 'zero':
            # [Optim A] Check layer does not read old_kv: status=1 in LlamaAttention.forward
            # skips the old_kv branch, and xformers status=1 path never dereferences old_kv.
            self.old_kvs[layer_idx] = [None, None]
        else:  # 'full'
            _, k_gpu, v_gpu = buf
            self.old_kvs[layer_idx] = [k_gpu, v_gpu]

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.get_input_embeddings(input_ids)
        
        if attn_metadata.prefill_metadata:
            temp_status = 0 # full prefill
            if self.cache_fuse_metadata["check"]:
                self.cache_fuse_metadata["org_seq_len"] = input_ids.shape[0] 
                check_layer_idx = 0
                self.cache_fuse_metadata["fake_q"] = None  
                self.cache_fuse_metadata["attn_bias"] = None
                self.cache_fuse_metadata["imp_indices"] = None
                self.cache_fuse_metadata["original_slot_mapping"] = None
                self.cache_fuse_metadata["our_slot_mapping"] = None
                self.cache_fuse_metadata['org_pos'] = positions[:]

                # [Optim C] Invalidate cross-layer caches for the new request
                self._cached_sparse_positions = None

                # [Pipeline v2]: Initialize the pipeline by triggering transfer and recompute for the first layer
                if self.cache_fuse_metadata.get('pipeline_enabled', False):
                    cpu_kv_cache = self.cache_fuse_metadata.get('cpu_kv_cache', None)
                    if cpu_kv_cache is not None and not self._pipeline_initialized:
                        self.old_kvs = [[None, None] for _ in range(len(self.layers))]
                        self._warmup_prefetched_layer = None
                        first_layer = self.cache_fuse_metadata["check_layers"][0]
                        self._prefetch_layer(first_layer)
                        with torch.cuda.stream(self._recompute_stream):
                            self._recompute_stream.wait_stream(self._transfer_stream)
                            self._background_rope(first_layer)
                            self._recompute_events[first_layer].record(self._recompute_stream)

                        # [Optim F] The first_layer prefetch is a zero-case no-op (check layer
                        # never reads transferred KV). To start the real DMA as early as
                        # possible, we additionally prefetch the first fusion layer here so
                        # that its transfer can overlap with layer 0's full-prefill compute,
                        # rather than only with the sparse-query check layer.
                        second_layer = first_layer + 1
                        if second_layer < len(self.layers):
                            self._prefetch_layer(second_layer)
                            with torch.cuda.stream(self._recompute_stream):
                                self._recompute_stream.wait_stream(self._transfer_stream)
                                self._background_rope(second_layer)
                                self._recompute_events[second_layer].record(self._recompute_stream)
                            self._warmup_prefetched_layer = second_layer
                        else:
                            self._warmup_prefetched_layer = None

                        self._pipeline_initialized = True
            #FIXME(Author): fix this clone for faster time
            #self.cache_fuse_metadata["our_slot_mapping"] = input_metadata.slot_mapping.clone()
        else:
            temp_status = -1 # decode
        residual = None
        
        
        for i in range(len(self.layers)):
            
            if self.cache_fuse_metadata["check"]:
                if i in self.cache_fuse_metadata["check_layers"]:
                    temp_status = 1 # check this layer
                    self.cache_fuse_metadata["check_layer"] = self.cache_fuse_metadata["check_layers"][check_layer_idx]
                    check_layer_idx += 1
                elif i > self.cache_fuse_metadata["check_layers"][0]:
                    temp_status = 2 # after check
            
            # [Pipeline v2]: Wait for current layer's KV transfer and recompute to finish
            if self.cache_fuse_metadata.get('pipeline_enabled', False) and temp_status in [1, 2]:
                self._transfer_events[i].wait()
                self._recompute_events[i].wait()
                self._rebuild_old_kv(i)

                # Asynchronously trigger next layer's KV transfer and recompute
                if i + 1 < len(self.layers):
                    # [Optim F] Skip if warmup has already prefetched this layer
                    if getattr(self, '_warmup_prefetched_layer', None) == i + 1:
                        # Consume the one-shot flag; layer (i+1)'s DMA + RoPE are already
                        # in-flight and their events were recorded at warmup time.
                        self._warmup_prefetched_layer = None
                    else:
                        self._prefetch_layer(i + 1)
                        with torch.cuda.stream(self._recompute_stream):
                            self._recompute_stream.wait_stream(self._transfer_stream)
                            self._background_rope(i + 1)
                            self._recompute_events[i+1].record(self._recompute_stream)
            
            if temp_status == 2 and 'precomputed_indices' in self.cache_fuse_metadata:
                old_imp = self.cache_fuse_metadata.get("imp_indices")
                precomputed_indices = self.cache_fuse_metadata['precomputed_indices']
                if len(precomputed_indices) == 0:
                    new_imp = old_imp
                elif i < len(precomputed_indices):
                    new_imp = precomputed_indices[i]
                else:
                    new_imp = precomputed_indices[-1]
                if old_imp is not None and new_imp is not None and len(new_imp) < len(old_imp):
                    keep_mask = torch.isin(old_imp, new_imp)
                    hidden_states = hidden_states[keep_mask]
                    if residual is not None:
                        residual = residual[keep_mask]
                    positions = positions[keep_mask]
                    self.cache_fuse_metadata["imp_indices"] = new_imp
            
            old_kv = self.old_kvs[i]
            
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                kv_caches[i],
                attn_metadata,
                residual,
                
                status = temp_status,
                cache_fuse_metadata=self.cache_fuse_metadata,
                old_kv=old_kv
            )
            
            if temp_status==1:
                #import pdb
                #pdb.set_trace()
                positions = positions[self.cache_fuse_metadata["imp_indices"]]
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class LlamaForCausalLM(nn.Module):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    # LoRA specific attributes
    supported_lora_modules = [
        "qkv_proj",
        "o_proj",
        "gate_up_proj",
        "down_proj",
        "embed_tokens",
        "lm_head",
    ]
    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }
    embedding_padding_modules = ["lm_head"]

    def __init__(
        self,
        config: LlamaConfig,
        linear_method: Optional[LinearMethodBase] = None,
        lora_config: Optional[LoRAConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.linear_method = linear_method
        self.model = LlamaModel(config, linear_method, lora_config=lora_config)
        self.unpadded_vocab_size = config.vocab_size
        if lora_config:
            self.unpadded_vocab_size += lora_config.lora_extra_vocab_size
        self.lm_head = ParallelLMHead(
            self.unpadded_vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            padding_size=DEFAULT_VOCAB_PADDING_SIZE
            # We need bigger padding if using lora for kernel
            # compatibility
            if not lora_config else lora_config.lora_vocab_padding_size,
        )

        logit_scale = getattr(config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                config.vocab_size, logit_scale)
        self.sampler = Sampler()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, kv_caches,
                                   attn_metadata)
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head.weight, hidden_states,
                                       sampling_metadata)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if ("rotary_emb.cos_cached" in name
                    or "rotary_emb.sin_cached" in name):
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)

    # If this function is called, it should always initialize KV cache scale
    # factors (or else raise an exception). Thus, handled exceptions should
    # make sure to leave KV cache scale factors in a known good (dummy) state
    def load_kv_cache_scales(self, quantization_param_path: str) -> None:
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        for layer_idx, scaling_factor in kv_cache_scales_loader(
                quantization_param_path, tp_rank, tp_size,
                self.config.num_hidden_layers,
                self.config.__class__.model_type):
            layer_self_attn = self.model.layers[layer_idx].self_attn

            if is_hip():
                # The scaling factor convention we are assuming is
                # quantized_value * scaling_factor ~= true_value
                # which is consistent with the practice of setting
                # scaling_factor = tensor_amax / FPtype_max
                scaling_factor *= 2
            if hasattr(layer_self_attn, "kv_scale"):
                layer_self_attn.kv_scale = scaling_factor
            else:
                raise RuntimeError("Self attention has no KV cache scaling "
                                   "factor attribute!")






# # coding=utf-8
# # Adapted from
# # https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
# # Copyright 2023 The vLLM team.
# # Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
# #
# # This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# # and OPT implementations in this library. It has been modified from its
# # original forms to accommodate minor architectural differences compared
# # to GPT-NeoX and OPT used by the Meta AI team that trained the model.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
# """Inference-only LLaMA model compatible with HuggingFace weights."""
# from typing import Any, Dict, Iterable, List, Optional, Tuple

# import torch
# from torch import nn
# from transformers import LlamaConfig

# from vllm.attention import Attention, AttentionMetadata
# from vllm.config import LoRAConfig
# from vllm.distributed import (get_tensor_model_parallel_rank,
#                               get_tensor_model_parallel_world_size)
# from vllm.model_executor.layers.activation import SiluAndMul
# from vllm.model_executor.layers.layernorm import RMSNorm
# from vllm.model_executor.layers.linear import (LinearMethodBase,
#                                                MergedColumnParallelLinear,
#                                                QKVParallelLinear,
#                                                RowParallelLinear)
# from vllm.model_executor.layers.logits_processor import LogitsProcessor
# from vllm.model_executor.layers.rotary_embedding import get_rope
# from vllm.model_executor.layers.sampler import Sampler
# from vllm.model_executor.layers.vocab_parallel_embedding import (
#     DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding)
# from vllm.model_executor.model_loader.weight_utils import (
#     default_weight_loader, kv_cache_scales_loader)
# from vllm.model_executor.sampling_metadata import SamplingMetadata
# from vllm.sequence import SamplerOutput
# from vllm.utils import is_hip


# class LlamaMLP(nn.Module):

#     def __init__(
#         self,
#         hidden_size: int,
#         intermediate_size: int,
#         hidden_act: str,
#         linear_method: Optional[LinearMethodBase] = None,
#     ) -> None:
#         super().__init__()
#         self.gate_up_proj = MergedColumnParallelLinear(
#             hidden_size, [intermediate_size] * 2,
#             bias=False,
#             linear_method=linear_method)
#         self.down_proj = RowParallelLinear(intermediate_size,
#                                            hidden_size,
#                                            bias=False,
#                                            linear_method=linear_method)
#         if hidden_act != "silu":
#             raise ValueError(f"Unsupported activation: {hidden_act}. "
#                              "Only silu is supported for now.")
#         self.act_fn = SiluAndMul()

#     def forward(self, x):
#         gate_up, _ = self.gate_up_proj(x)
#         x = self.act_fn(gate_up)
#         x, _ = self.down_proj(x)
#         return x


# class LlamaAttention(nn.Module):

#     def __init__(
#         self,
#         hidden_size: int,
#         num_heads: int,
#         num_kv_heads: int,
#         rope_theta: float = 10000,
#         rope_scaling: Optional[Dict[str, Any]] = None,
#         max_position_embeddings: int = 8192,
#         linear_method: Optional[LinearMethodBase] = None,
#         bias: bool = False,
#         sliding_window: Optional[int] = None,
#     ) -> None:
#         super().__init__()
#         self.hidden_size = hidden_size
#         tp_size = get_tensor_model_parallel_world_size()
#         self.total_num_heads = num_heads
#         assert self.total_num_heads % tp_size == 0
#         self.num_heads = self.total_num_heads // tp_size
#         self.total_num_kv_heads = num_kv_heads
#         if self.total_num_kv_heads >= tp_size:
#             # Number of KV heads is greater than TP size, so we partition
#             # the KV heads across multiple tensor parallel GPUs.
#             assert self.total_num_kv_heads % tp_size == 0
#         else:
#             # Number of KV heads is less than TP size, so we replicate
#             # the KV heads across multiple tensor parallel GPUs.
#             assert tp_size % self.total_num_kv_heads == 0
#         self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
#         self.head_dim = hidden_size // self.total_num_heads
#         self.q_size = self.num_heads * self.head_dim
#         self.kv_size = self.num_kv_heads * self.head_dim
#         self.scaling = self.head_dim**-0.5
#         self.rope_theta = rope_theta
#         self.max_position_embeddings = max_position_embeddings

#         # This will be overwritten by model initialization if we are using it.
#         # N.B. currently we only support per tensor scalar scaling factors
#         # & only applicable to ROCm (AMD GPU).
#         # The scaling factor convention we are assuming is
#         # quantized_value * scaling_factor ~= true_value
#         # which is consistent with the practice of setting
#         # scaling_factor = tensor_amax / FPtype_max
#         self.kv_scale = 1.0

#         self.qkv_proj = QKVParallelLinear(
#             hidden_size,
#             self.head_dim,
#             self.total_num_heads,
#             self.total_num_kv_heads,
#             bias=bias,
#             linear_method=linear_method,
#         )
#         self.o_proj = RowParallelLinear(
#             self.total_num_heads * self.head_dim,
#             hidden_size,
#             bias=bias,
#             linear_method=linear_method,
#         )

#         self.rotary_emb = get_rope(
#             self.head_dim,
#             rotary_dim=self.head_dim,
#             max_position=max_position_embeddings,
#             base=rope_theta,
#             rope_scaling=rope_scaling,
#         )
#         self.attn = Attention(self.num_heads,
#                               self.head_dim,
#                               self.scaling,
#                               num_kv_heads=self.num_kv_heads,
#                               sliding_window=sliding_window)
        
#         self.hack_kv = []

#     def forward(
#         self,
#         positions: torch.Tensor,
#         hidden_states: torch.Tensor,
#         kv_cache: torch.Tensor,
#         attn_metadata: AttentionMetadata,
        
#         status,
#         cache_fuse_metadata,
#         old_kv,
#     ) -> torch.Tensor:
#         qkv, _ = self.qkv_proj(hidden_states)
#         q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        
#         # CacheTune note: Rotate the old K 
#         # Need to modify the kernel to only take K as input 
#         if status in [1,2]:
#             if cache_fuse_metadata["fake_q"] is None:
#                 cache_fuse_metadata['fake_q'] = torch.rand_like(q)
#             _, old_kv[0] = self.rotary_emb(cache_fuse_metadata['org_pos'],
#                                         cache_fuse_metadata['fake_q'],
#                                         old_kv[0])
            
#         if cache_fuse_metadata['collect']:
#             self.hack_kv = [k.clone(), v.clone()]
#         q, k = self.rotary_emb(positions, q, k)
#         attn_output = self.attn(q, k, v, kv_cache, attn_metadata,
#                                 status, cache_fuse_metadata, old_kv,
#                                 self.kv_scale)
#         # attn_output = self.attn(q, k, v, 
#         #                         kv_cache=kv_cache, 
#         #                         attn_metadata=attn_metadata,
#         #                         status=status, 
#         #                         cache_fuse_metadata=cache_fuse_metadata, 
#         #                         old_kv=old_kv,
#         #                         self.kv_scale)        
#         output, _ = self.o_proj(attn_output)
#         return output


# class LlamaDecoderLayer(nn.Module):

#     def __init__(
#         self,
#         config: LlamaConfig,
#         linear_method: Optional[LinearMethodBase] = None,
#     ) -> None:
#         super().__init__()
#         self.hidden_size = config.hidden_size
#         rope_theta = getattr(config, "rope_theta", 10000)
#         rope_scaling = getattr(config, "rope_scaling", None)
#         max_position_embeddings = getattr(config, "max_position_embeddings",
#                                           8192)
#         sliding_window = getattr(config, "sliding_window", None)
#         # Support abacusai/Smaug-72B-v0.1 with attention_bias
#         # Support internlm/internlm-7b with bias
#         attention_bias = getattr(config, "attention_bias", False) or getattr(
#             config, "bias", False)
#         self.self_attn = LlamaAttention(
#             hidden_size=self.hidden_size,
#             num_heads=config.num_attention_heads,
#             num_kv_heads=getattr(config, "num_key_value_heads",
#                                  config.num_attention_heads),
#             rope_theta=rope_theta,
#             rope_scaling=rope_scaling,
#             max_position_embeddings=max_position_embeddings,
#             linear_method=linear_method,
#             bias=attention_bias,
#             sliding_window=sliding_window,
#         )
#         self.mlp = LlamaMLP(
#             hidden_size=self.hidden_size,
#             intermediate_size=config.intermediate_size,
#             hidden_act=config.hidden_act,
#             linear_method=linear_method,
#         )
#         self.input_layernorm = RMSNorm(config.hidden_size,
#                                        eps=config.rms_norm_eps)
#         self.post_attention_layernorm = RMSNorm(config.hidden_size,
#                                                 eps=config.rms_norm_eps)

#     def forward(
#         self,
#         positions: torch.Tensor,
#         hidden_states: torch.Tensor,
#         kv_cache: torch.Tensor,
#         attn_metadata: AttentionMetadata,
#         residual: Optional[torch.Tensor],
        
#         status: int,
#         cache_fuse_metadata: dict,
#         old_kv,
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         # Self Attention
#         if residual is None:
#             residual = hidden_states
#             hidden_states = self.input_layernorm(hidden_states)
#         else:
#             hidden_states, residual = self.input_layernorm(
#                 hidden_states, residual)
#         hidden_states = self.self_attn(
#             positions=positions,
#             hidden_states=hidden_states,
#             kv_cache=kv_cache,
#             attn_metadata=attn_metadata,
            
#             # CacheTune integration start
#             status=status,
#             cache_fuse_metadata=cache_fuse_metadata,
#             old_kv=old_kv,
#             # CacheTune integration end
#         )

#         if status == 1:
#             residual = residual[cache_fuse_metadata["imp_indices"]]
        
#         # Fully Connected
#         hidden_states, residual = self.post_attention_layernorm(
#             hidden_states, residual)
#         hidden_states = self.mlp(hidden_states)
#         return hidden_states, residual


# class LlamaModel(nn.Module):

#     def __init__(
#         self,
#         config: LlamaConfig,
#         linear_method: Optional[LinearMethodBase] = None,
#         lora_config: Optional[LoRAConfig] = None,
#     ) -> None:
#         super().__init__()
#         self.config = config
#         self.padding_idx = config.pad_token_id
#         lora_vocab = (lora_config.lora_extra_vocab_size *
#                       (lora_config.max_loras or 1)) if lora_config else 0
#         self.vocab_size = config.vocab_size + lora_vocab
#         self.org_vocab_size = config.vocab_size
#         self.embed_tokens = VocabParallelEmbedding(
#             self.vocab_size,
#             config.hidden_size,
#             org_num_embeddings=config.vocab_size,
#         )
#         self.layers = nn.ModuleList([
#             LlamaDecoderLayer(config, linear_method)
#             for _ in range(config.num_hidden_layers)
#         ])
#         self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

#         self.cache_fuse_metadata = {"check_layers":[1],
#                                     "check": False,
#                                     "recomp_ratios":[0.16],
#                                     "recomp_ratio":0.16,
#                                     "original_slot_mapping":None,
#                                     "our_slot_mapping":None,
#                                     "kv_cache_dtype": None,
#                                     "attn_bias": None,
#                                     "imp_indices": None,
#                                     "org_seq_len": None,
#                                     "collect": False}
        
#         self.old_kvs_buffers = [
#             [[None, None] for _ in range(len(self.layers))],  # Buffer A
#             [[None, None] for _ in range(len(self.layers))]   # Buffer B
#         ]
        
#         self.old_kvs = self.old_kvs_buffers[0]
        
#         self._transfer_stream = torch.cuda.Stream()
#         self._transfer_events = [
#             [torch.cuda.Event() for _ in range(len(self.layers))],  # Buffer A events
#             [torch.cuda.Event() for _ in range(len(self.layers))]   # Buffer B events
#         ]
#         self._pipeline_initialized = False
        
#     def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
#         return self.embed_tokens(input_ids)

#     def forward(
#         self,
#         input_ids: Optional[torch.Tensor],
#         positions: torch.Tensor,
#         kv_caches: List[torch.Tensor],
#         attn_metadata: AttentionMetadata,
#         inputs_embeds: Optional[torch.Tensor] = None,
#     ) -> torch.Tensor:
#         if inputs_embeds is not None:
#             hidden_states = inputs_embeds
#         else:
#             hidden_states = self.get_input_embeddings(input_ids)
        
#         if attn_metadata.prefill_metadata:
#             temp_status = 0 # full prefill
#             if self.cache_fuse_metadata["check"]:
#                 self.cache_fuse_metadata["org_seq_len"] = input_ids.shape[0] 
#                 check_layer_idx = 0
#                 self.cache_fuse_metadata["fake_q"] = None  
#                 self.cache_fuse_metadata["attn_bias"] = None
#                 self.cache_fuse_metadata["imp_indices"] = None
#                 self.cache_fuse_metadata["original_slot_mapping"] = None
#                 self.cache_fuse_metadata["our_slot_mapping"] = None
#                 self.cache_fuse_metadata['org_pos'] = positions[:]
                
#                 if self.cache_fuse_metadata.get('pipeline_enabled', False):
#                     cpu_kv_cache = self.cache_fuse_metadata.get('cpu_kv_cache', None)
#                     if cpu_kv_cache is not None and not self._pipeline_initialized:
#                         self.current_buffer_idx = 0
#                         self.old_kvs = self.old_kvs_buffers[self.current_buffer_idx]
#                         current_events = self._transfer_events[self.current_buffer_idx]
                        
#                         first_layer = self.cache_fuse_metadata["check_layers"][0]
#                         with torch.cuda.stream(self._transfer_stream):
#                             k_gpu = cpu_kv_cache[first_layer][0].to("cuda", non_blocking=True)
#                             v_gpu = cpu_kv_cache[first_layer][1].to("cuda", non_blocking=True)
#                             self.old_kvs[first_layer] = [k_gpu, v_gpu]
#                             current_events[first_layer].record(self._transfer_stream)
#                         self._pipeline_initialized = True
#             #FIXME(Author): fix this clone for faster time (Is this still needed?)
#             #self.cache_fuse_metadata["our_slot_mapping"] = input_metadata.slot_mapping.clone()
#         else:
#             temp_status = -1 # decode
#         residual = None
        
        
#         for i in range(len(self.layers)):
            
#             if self.cache_fuse_metadata["check"]:
#                 if i in self.cache_fuse_metadata["check_layers"]:
#                     temp_status = 1 # check this layer
#                     self.cache_fuse_metadata["check_layer"] = self.cache_fuse_metadata["check_layers"][check_layer_idx]
#                     check_layer_idx += 1
#                 elif i > self.cache_fuse_metadata["check_layers"][0]:
#                     temp_status = 2 # after check
            
#             if self.cache_fuse_metadata.get('pipeline_enabled', False) and temp_status in [1, 2]:
#                 cpu_kv_cache = self.cache_fuse_metadata.get('cpu_kv_cache', None)
#                 current_events = self._transfer_events[self.current_buffer_idx]
                
#                 current_events[i].wait()
                
#                 if cpu_kv_cache is not None and i + 1 < len(self.layers):
#                     with torch.cuda.stream(self._transfer_stream):
#                         k_gpu = cpu_kv_cache[i+1][0].to("cuda", non_blocking=True)
#                         v_gpu = cpu_kv_cache[i+1][1].to("cuda", non_blocking=True)
#                         self.old_kvs[i+1] = [k_gpu, v_gpu]
#                         current_events[i+1].record(self._transfer_stream)
                
            
#             old_kv = self.old_kvs[i]
            
#             layer = self.layers[i]
#             hidden_states, residual = layer(
#                 positions,
#                 hidden_states,
#                 kv_caches[i],
#                 attn_metadata,
#                 residual,
                
#                 status = temp_status,
#                 cache_fuse_metadata=self.cache_fuse_metadata,
#                 old_kv=old_kv
#             )
            
#             if temp_status==1:
#                 #import pdb
#                 #pdb.set_trace()
#                 positions = positions[self.cache_fuse_metadata["imp_indices"]]
#         hidden_states, _ = self.norm(hidden_states, residual)
#         return hidden_states


# class LlamaForCausalLM(nn.Module):
#     packed_modules_mapping = {
#         "qkv_proj": [
#             "q_proj",
#             "k_proj",
#             "v_proj",
#         ],
#         "gate_up_proj": [
#             "gate_proj",
#             "up_proj",
#         ],
#     }

#     # LoRA specific attributes
#     supported_lora_modules = [
#         "qkv_proj",
#         "o_proj",
#         "gate_up_proj",
#         "down_proj",
#         "embed_tokens",
#         "lm_head",
#     ]
#     embedding_modules = {
#         "embed_tokens": "input_embeddings",
#         "lm_head": "output_embeddings",
#     }
#     embedding_padding_modules = ["lm_head"]

#     def __init__(
#         self,
#         config: LlamaConfig,
#         linear_method: Optional[LinearMethodBase] = None,
#         lora_config: Optional[LoRAConfig] = None,
#     ) -> None:
#         super().__init__()
#         self.config = config
#         self.linear_method = linear_method
#         self.model = LlamaModel(config, linear_method, lora_config=lora_config)
#         self.unpadded_vocab_size = config.vocab_size
#         if lora_config:
#             self.unpadded_vocab_size += lora_config.lora_extra_vocab_size
#         self.lm_head = ParallelLMHead(
#             self.unpadded_vocab_size,
#             config.hidden_size,
#             org_num_embeddings=config.vocab_size,
#             padding_size=DEFAULT_VOCAB_PADDING_SIZE
#             # We need bigger padding if using lora for kernel
#             # compatibility
#             if not lora_config else lora_config.lora_vocab_padding_size,
#         )

#         logit_scale = getattr(config, "logit_scale", 1.0)
#         self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
#                                                 config.vocab_size, logit_scale)
#         self.sampler = Sampler()

#     def forward(
#         self,
#         input_ids: torch.Tensor,
#         positions: torch.Tensor,
#         kv_caches: List[torch.Tensor],
#         attn_metadata: AttentionMetadata,
#     ) -> torch.Tensor:
#         hidden_states = self.model(input_ids, positions, kv_caches,
#                                    attn_metadata)
#         return hidden_states

#     def compute_logits(self, hidden_states: torch.Tensor,
#                        sampling_metadata: SamplingMetadata) -> torch.Tensor:
#         logits = self.logits_processor(self.lm_head.weight, hidden_states,
#                                        sampling_metadata)
#         return logits

#     def sample(
#         self,
#         logits: torch.Tensor,
#         sampling_metadata: SamplingMetadata,
#     ) -> Optional[SamplerOutput]:
#         next_tokens = self.sampler(logits, sampling_metadata)
#         return next_tokens

#     def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
#         stacked_params_mapping = [
#             # (param_name, shard_name, shard_id)
#             ("qkv_proj", "q_proj", "q"),
#             ("qkv_proj", "k_proj", "k"),
#             ("qkv_proj", "v_proj", "v"),
#             ("gate_up_proj", "gate_proj", 0),
#             ("gate_up_proj", "up_proj", 1),
#         ]
#         params_dict = dict(self.named_parameters())
#         for name, loaded_weight in weights:
#             if "rotary_emb.inv_freq" in name:
#                 continue
#             if ("rotary_emb.cos_cached" in name
#                     or "rotary_emb.sin_cached" in name):
#                 # Models trained using ColossalAI may include these tensors in
#                 # the checkpoint. Skip them.
#                 continue
#             for (param_name, weight_name, shard_id) in stacked_params_mapping:
#                 if weight_name not in name:
#                     continue
#                 name = name.replace(weight_name, param_name)
#                 # Skip loading extra bias for GPTQ models.
#                 if name.endswith(".bias") and name not in params_dict:
#                     continue
#                 param = params_dict[name]
#                 weight_loader = param.weight_loader
#                 weight_loader(param, loaded_weight, shard_id)
#                 break
#             else:
#                 # Skip loading extra bias for GPTQ models.
#                 if name.endswith(".bias") and name not in params_dict:
#                     continue
#                 param = params_dict[name]
#                 weight_loader = getattr(param, "weight_loader",
#                                         default_weight_loader)
#                 weight_loader(param, loaded_weight)

#     # If this function is called, it should always initialize KV cache scale
#     # factors (or else raise an exception). Thus, handled exceptions should
#     # make sure to leave KV cache scale factors in a known good (dummy) state
#     def load_kv_cache_scales(self, quantization_param_path: str) -> None:
#         tp_size = get_tensor_model_parallel_world_size()
#         tp_rank = get_tensor_model_parallel_rank()
#         for layer_idx, scaling_factor in kv_cache_scales_loader(
#                 quantization_param_path, tp_rank, tp_size,
#                 self.config.num_hidden_layers,
#                 self.config.__class__.model_type):
#             layer_self_attn = self.model.layers[layer_idx].self_attn

#             if is_hip():
#                 # The scaling factor convention we are assuming is
#                 # quantized_value * scaling_factor ~= true_value
#                 # which is consistent with the practice of setting
#                 # scaling_factor = tensor_amax / FPtype_max
#                 scaling_factor *= 2
#             if hasattr(layer_self_attn, "kv_scale"):
#                 layer_self_attn.kv_scale = scaling_factor
#             else:
#                 raise RuntimeError("Self attention has no KV cache scaling "
#                                    "factor attribute!")
