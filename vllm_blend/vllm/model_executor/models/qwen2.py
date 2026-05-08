# coding=utf-8
# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/qwen2/modeling_qwen2.py
# Copyright 2024 The Qwen team.
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
"""Inference-only Qwen2 model compatible with HuggingFace weights."""
from typing import Iterable, List, Optional, Tuple

import torch
from torch import nn
from transformers import Qwen2Config

from vllm.attention import Attention, AttentionMetadata
from vllm.config import LoRAConfig
from vllm.distributed import get_tensor_model_parallel_world_size
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
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import SamplerOutput


class Qwen2MLP(nn.Module):

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


class Qwen2Attention(nn.Module):

    def __init__(self,
                 hidden_size: int,
                 num_heads: int,
                 num_kv_heads: int,
                 max_position: int = 4096 * 32,
                 rope_theta: float = 10000,
                 use_sliding_window: bool = False,
                 linear_method: Optional[LinearMethodBase] = None,
                 sliding_window: Optional[int] = None) -> None:
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
        self.sliding_window = sliding_window if use_sliding_window else None

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=True,
            linear_method=linear_method,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            linear_method=linear_method,
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=self.rope_theta,
        )
        self.attn = Attention(self.num_heads,
                              self.head_dim,
                              self.scaling,
                              num_kv_heads=self.num_kv_heads,
                              sliding_window=self.sliding_window)

        # CacheTune: Hook for KV capture during offline phase
        self.hack_kv = []

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        # CacheTune parameters
        status: int = 0,
        cache_fuse_metadata: Optional[dict] = None,
        old_kv: Optional[list] = None,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # CacheTune: Rotate old_kv with the positions that match the
        # transferred KV layout. Compact KV stores only non-recompute tokens.
        if cache_fuse_metadata is not None and status in [1, 2]:
            fake_q = torch.empty((old_kv[0].shape[0], self.q_size),
                                 dtype=q.dtype,
                                 device=q.device)
            if "non_imp_indices" in cache_fuse_metadata:
                compact_pos = cache_fuse_metadata["org_pos"][
                    cache_fuse_metadata["non_imp_indices"]]
            else:
                compact_pos = cache_fuse_metadata.get("org_pos", positions)
            _, old_kv[0] = self.rotary_emb(compact_pos, fake_q, old_kv[0])

        # CacheTune: Hook KV for offline collection phase
        if cache_fuse_metadata is not None and cache_fuse_metadata.get('collect', False):
            self.hack_kv = [k.clone(), v.clone()]

        q, k = self.rotary_emb(positions, q, k)

        # Pass CacheTune parameters to attention
        if cache_fuse_metadata is not None:
            attn_output = self.attn(q, k, v, kv_cache, attn_metadata,
                                    status=status,
                                    cache_fuse_metadata=cache_fuse_metadata,
                                    old_kv=old_kv)
        else:
            attn_output = self.attn(q, k, v, kv_cache, attn_metadata)

        output, _ = self.o_proj(attn_output)
        return output


class Qwen2DecoderLayer(nn.Module):

    def __init__(
        self,
        config: Qwen2Config,
        layer_idx: int,
        linear_method: Optional[LinearMethodBase] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        # Requires transformers > 4.32.0
        rope_theta = getattr(config, "rope_theta", 1000000)
        use_sliding_window = (config.use_sliding_window
                              and layer_idx < config.max_window_layers)
        self.self_attn = Qwen2Attention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=config.max_position_embeddings,
            num_kv_heads=config.num_key_value_heads,
            rope_theta=rope_theta,
            use_sliding_window=use_sliding_window,
            linear_method=linear_method,
            sliding_window=config.sliding_window)
        self.mlp = Qwen2MLP(
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
        # CacheTune parameters
        status: int = 0,
        cache_fuse_metadata: Optional[dict] = None,
        old_kv: Optional[list] = None,
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
            status=status,
            cache_fuse_metadata=cache_fuse_metadata,
            old_kv=old_kv,
        )

        # CacheTune: Slice residual for status=1 (check layer)
        if status == 1 and cache_fuse_metadata is not None:
            residual = residual[cache_fuse_metadata["imp_indices"]]

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen2Model(nn.Module):

    def __init__(
        self,
        config: Qwen2Config,
        linear_method: Optional[LinearMethodBase] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )
        self.layers = nn.ModuleList([
            Qwen2DecoderLayer(config, layer_idx, linear_method)
            for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # CacheTune: Initialize metadata
        self.cache_fuse_metadata = {
            "check_layers": [1],
            "check": False,
            "recomp_ratio": 0.16,
            "collect": False,
            "attn_bias": None,
            "imp_indices": None,
            "org_seq_len": None,
            "kv_cache_dtype": None,
        }

        # CacheTune: Pipeline state
        self.old_kvs = [[None, None] for _ in range(len(self.layers))]
        self._transfer_stream = torch.cuda.Stream()
        self._transfer_events = [torch.cuda.Event() for _ in range(len(self.layers))]
        self._pipeline_initialized = False
        self._prefetch_buffers = {}

    def _prefetch_layer(self, layer_idx: int) -> None:
        """Asynchronously transfer layer_idx KV from CPU pinned memory to GPU."""
        cpu_kv_cache = self.cache_fuse_metadata.get('cpu_kv_cache', None)
        if cpu_kv_cache is None or layer_idx >= len(self.layers):
            return

        N_total = cpu_kv_cache[layer_idx][0].shape[0]
        shape_rest = cpu_kv_cache[layer_idx][0].shape[1:]
        dtype = cpu_kv_cache[layer_idx][0].dtype
        check_layers = self.cache_fuse_metadata.get('check_layers', [1])

        # Check layer: no transfer needed, mark as zero
        if layer_idx in check_layers:
            with torch.cuda.stream(self._transfer_stream):
                self._transfer_events[layer_idx].record(self._transfer_stream)
            self._prefetch_buffers[layer_idx] = ('zero', N_total, shape_rest, dtype)
            return

        # Status=2 layers: transfer full KV from CPU pinned memory
        cpu_k = cpu_kv_cache[layer_idx][0]
        cpu_v = cpu_kv_cache[layer_idx][1]

        with torch.cuda.stream(self._transfer_stream):
            k_gpu = cpu_k.to('cuda', non_blocking=True)
            v_gpu = cpu_v.to('cuda', non_blocking=True)
            self._transfer_events[layer_idx].record(self._transfer_stream)
        self._prefetch_buffers[layer_idx] = ('full', k_gpu, v_gpu)

    def _rebuild_old_kv(self, layer_idx: int) -> None:
        """Rebuild old_kv from prefetch buffer."""
        if layer_idx not in self._prefetch_buffers:
            return
        buf = self._prefetch_buffers.pop(layer_idx)
        tag = buf[0]

        if tag == 'zero':
            _, N_total, shape_rest, dtype = buf
            self.old_kvs[layer_idx] = [
                torch.zeros(N_total, *shape_rest, device='cuda', dtype=dtype),
                torch.zeros(N_total, *shape_rest, device='cuda', dtype=dtype),
            ]
        else:  # 'full'
            _, k_gpu, v_gpu = buf
            self.old_kvs[layer_idx] = [k_gpu, v_gpu]

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None

        # CacheTune: Determine status for all layers
        temp_status = 0  # default: full prefill
        if attn_metadata.prefill_metadata and self.cache_fuse_metadata["check"]:
            self.cache_fuse_metadata["org_seq_len"] = input_ids.shape[0]
            self.cache_fuse_metadata["fake_q"] = None
            self.cache_fuse_metadata["attn_bias"] = None
            self.cache_fuse_metadata["imp_indices"] = None
            self.cache_fuse_metadata['org_pos'] = positions[:]

            # CacheTune: Pipeline warmup if enabled
            if self.cache_fuse_metadata.get('pipeline_enabled', False):
                cpu_kv_cache = self.cache_fuse_metadata.get('cpu_kv_cache', None)
                if cpu_kv_cache is not None and not self._pipeline_initialized:
                    first_layer = self.cache_fuse_metadata["check_layers"][0]
                    self._prefetch_layer(first_layer)
                    self._pipeline_initialized = True
        elif not attn_metadata.prefill_metadata:
            temp_status = -1  # decode phase

        for i in range(len(self.layers)):
            # CacheTune: Update status
            if self.cache_fuse_metadata["check"]:
                if i in self.cache_fuse_metadata["check_layers"]:
                    temp_status = 1  # check layer
                elif i > self.cache_fuse_metadata["check_layers"][0]:
                    temp_status = 2  # fusion layers

            # CacheTune: Pipeline I/O with compute
            if self.cache_fuse_metadata.get('pipeline_enabled', False) and temp_status in [1, 2]:
                self._transfer_events[i].wait()
                self._rebuild_old_kv(i)
                if i + 1 < len(self.layers):
                    self._prefetch_layer(i + 1)

            old_kv = self.old_kvs[i]

            layer = self.layers[i]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                kv_caches[i],
                attn_metadata,
                residual,
                status=temp_status,
                cache_fuse_metadata=self.cache_fuse_metadata,
                old_kv=old_kv,
            )

            # CacheTune: Slice positions for status=1
            if temp_status == 1:
                positions = positions[self.cache_fuse_metadata["imp_indices"]]

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen2ForCausalLM(nn.Module):
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
    ]
    embedding_modules = {}
    embedding_padding_modules = []

    def __init__(
        self,
        config: Qwen2Config,
        linear_method: Optional[LinearMethodBase] = None,
        lora_config: Optional[LoRAConfig] = None,
    ) -> None:
        del lora_config
        super().__init__()
        self.config = config
        self.linear_method = linear_method
        self.model = Qwen2Model(config, linear_method)

        if config.tie_word_embeddings:
            self.lm_head_weight = self.model.embed_tokens.weight
        else:
            self.lm_head = ParallelLMHead(config.vocab_size,
                                          config.hidden_size)
            self.lm_head_weight = self.lm_head.weight

        self.logits_processor = LogitsProcessor(config.vocab_size)
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
        logits = self.logits_processor(self.lm_head_weight, hidden_states,
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
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if self.config.tie_word_embeddings and "lm_head.weight" in name:
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
