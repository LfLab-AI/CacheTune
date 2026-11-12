"""Request-scoped CacheTune state for native vLLM dense Transformer models.

This module owns independent pre-RoPE KVs and their optional CPU transfer, not
model execution or vLLM's paged cache. The model adapter must write the merged
full KVs to the original slots and use the original selected query positions.
"""
from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Any, Sequence

import torch

from .scoring import context_chunk_spans, select_context_indices


def _token_ids(values: Sequence[int] | torch.Tensor) -> list[int]:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().tolist()
    return [int(value) for value in values]


def _cache_bytes(pairs) -> int:
    return sum(t.numel() * t.element_size() for pair in (pairs or []) for t in pair)


def _external_context_indices(values, context_len: int, budget: int,
                              sink_size: int) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        if values.ndim != 1 or values.dtype not in (torch.int32, torch.int64):
            raise ValueError("External context indices must be a 1D int32/int64 tensor")
        values = values.detach().cpu().tolist()
    else:
        try:
            values = list(values)
        except TypeError as exc:
            raise ValueError("External context indices must be an integer sequence") from exc
    if any(isinstance(i, bool) or not isinstance(i, Integral) for i in values):
        raise ValueError("External context indices must be integers")
    values = [int(i) for i in values]
    if (len(values) != budget or len(set(values)) != len(values)
            or any(i < 0 or i >= context_len for i in values)):
        raise ValueError("External context indices must be unique context positions with the exact ratio budget")
    if not set(range(sink_size)).issubset(values):
        raise ValueError("External context indices must include every forced sink position")
    return torch.tensor(sorted(values), dtype=torch.long)


class CacheTuneState:
    """Single-request, complete-prefill state; repeat reuse is explicit.

    Model integration:
      note_input(ids, positions) -> begin_prefill(positions, attentions)
      check layer: select Q/residual with selected_indices; keep dense K/V
      subsequent fusion layers: merge(layer_idx, selected_K, selected_V)
      finish_prefill() -> subsequent decode uses the native model path

    ``capture`` is called after any K normalization and before RoPE. During
    collection, ``note_input`` arms capture only once per chunk generation.
    """

    def __init__(self, num_layers: int, check_layer: int = 1, *,
                 fuse_check_layer: bool = False) -> None:
        if num_layers < 2 or not 0 <= check_layer < num_layers:
            raise ValueError("CacheTune requires at least two layers and a valid check layer")
        self.num_layers = int(num_layers)
        self.check_layer = int(check_layer)
        if not isinstance(fuse_check_layer, bool):
            raise ValueError("fuse_check_layer must be boolean")
        self.fuse_check_layer = fuse_check_layer
        self.collecting = False
        self.enabled = False
        self.active = False
        self.prompt_len = 0
        self.device: torch.device | None = None
        self.selected_indices: torch.Tensor | None = None
        self.reused_indices: torch.Tensor | None = None
        self._selected_cpu: torch.Tensor | None = None
        self._reused_cpu: torch.Tensor | None = None
        self._prompt_ids: list[int] | None = None
        self._capture_ids: list[int] | None = None
        self._capture_armed = False
        self._captures: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._pieces: list[list[tuple[torch.Tensor, torch.Tensor]]] = [
            [] for _ in range(self.num_layers)
        ]
        self._collected_ids: list[int] = []
        self._chunk_count = 0
        self._chunk_lengths: list[int] = []
        self._full_cache: list[tuple[torch.Tensor, torch.Tensor]] | None = None
        self.cpu_kv_cache: list[tuple[torch.Tensor, torch.Tensor]] | None = None
        self.gpu_kv_cache: list[tuple[torch.Tensor, torch.Tensor]] | None = None
        self.cache_mode = "cpu_pipeline"
        self._next_layer = self.first_fusion_layer
        self._selection: dict[str, Any] | None = None
        self._attentions: list[Any] = []
        self._positions: torch.Tensor | None = None
        self._reuse_positions: torch.Tensor | None = None
        self._transfer_stream: torch.cuda.Stream | None = None
        self._rope_stream: torch.cuda.Stream | None = None
        self._stream_device: torch.device | None = None
        self._copy_events: list[Any] = []
        self._ready_events: list[Any] = []
        self._prefetched: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._ready_pairs: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._transfer_buffers: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._work_buffers: dict[tuple[tuple[int, ...], torch.dtype],
                                 tuple[torch.Tensor, torch.Tensor]] = {}
        self._fake_queries: dict[tuple[int, torch.dtype], torch.Tensor] = {}
        self._input_verified = False
        self._diag: dict[str, Any] = {"prefill_count": 0, "layers": []}

    @property
    def first_fusion_layer(self) -> int:
        """Dense models fuse after the check layer; hybrid full-attention includes it."""
        return self.check_layer + (0 if self.fuse_check_layer else 1)

    @property
    def prepared(self) -> bool:
        return self.cpu_kv_cache is not None or self.gpu_kv_cache is not None

    def _storage_diagnostics(self) -> dict[str, Any]:
        return {
            "cache_mode": self.cache_mode,
            "retained_full_cache": self._full_cache is not None,
            "retained_full_cache_cpu_bytes": _cache_bytes(self._full_cache),
            "compact_cpu_cache_bytes": _cache_bytes(self.cpu_kv_cache),
            "compact_gpu_cache_bytes": _cache_bytes(self.gpu_kv_cache),
            "gpu_transfer_buffer_bytes": _cache_bytes(self._transfer_buffers.values()),
            "gpu_work_buffer_bytes": _cache_bytes(self._work_buffers.values()),
            "gpu_fake_query_bytes": sum(t.numel() * t.element_size()
                                        for t in self._fake_queries.values()),
        }

    def _drain(self) -> None:
        if self.device is not None and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _clear_request(self, release_buffers: bool = False) -> None:
        self.enabled = self.active = self._input_verified = False
        self._positions = self._reuse_positions = None
        self._prefetched.clear()
        self._ready_pairs.clear()
        self._fake_queries.clear()
        self._attentions.clear()
        self._next_layer = self.first_fusion_layer
        if release_buffers:
            self._transfer_buffers.clear()
            self._work_buffers.clear()
            self.selected_indices = self.reused_indices = None

    def reset(self, clear_cache: bool = True) -> dict[str, Any]:
        self._drain()
        self._clear_request(release_buffers=clear_cache)
        self.collecting = self._capture_armed = False
        self._capture_ids = None
        self._captures.clear()
        if clear_cache:
            self._probe_next = False
            self._probe_logits = self._probe_reference = None
            self._pieces = [[] for _ in range(self.num_layers)]
            self._collected_ids = []
            self._chunk_count = 0
            self._chunk_lengths = []
            self._full_cache = self.cpu_kv_cache = self.gpu_kv_cache = None
            self.cache_mode = "cpu_pipeline"
            self._selection = self._prompt_ids = None
            self._selected_cpu = self._reused_cpu = None
            self.prompt_len = 0
            self._diag = {"prefill_count": 0, "layers": []}
        return {"cleared_cache": bool(clear_cache)}

    def start_collect(self) -> dict[str, Any]:
        self.reset(clear_cache=True)
        self.collecting = True
        return {"collecting": True, "num_layers": self.num_layers}

    def note_input(self, input_ids: torch.Tensor | None,
                   positions: torch.Tensor) -> bool:
        """Called at model entry. Return True for the armed sparse prefill."""
        if not self.collecting and not self.enabled:
            return False
        if input_ids is None or input_ids.ndim != 1 or positions.ndim != 1:
            raise ValueError("CacheTune supports text token IDs and 1D positions only")
        if self.collecting:
            if self._capture_ids is not None:
                # A generation may decode after the capture. Do not replace it.
                self._capture_armed = False
                return False
            if input_ids.numel() != positions.numel() or not torch.equal(
                positions, torch.arange(positions.numel(), device=positions.device,
                                        dtype=positions.dtype)
            ):
                raise ValueError("Chunk collection requires one complete prefill starting at position zero")
            self._capture_ids = _token_ids(input_ids)
            self._capture_armed = True
            self.device = input_ids.device
            return False
        if self.active:
            raise RuntimeError("A CacheTune prefill is already active")
        if _token_ids(input_ids) != self._prompt_ids:
            raise ValueError("The scheduled prompt token IDs do not match the prepared cache")
        if positions.numel() != self.prompt_len or not torch.equal(
            positions, torch.arange(self.prompt_len, device=positions.device,
                                    dtype=positions.dtype)
        ):
            raise ValueError("CacheTune requires a complete, uncached prefill starting at zero")
        self._input_verified = True
        return True

    def capture(self, layer_idx: int, key: torch.Tensor,
                value: torch.Tensor) -> None:
        if not self.collecting or not self._capture_armed:
            return
        if not 0 <= layer_idx < self.num_layers or layer_idx in self._captures:
            raise ValueError("Duplicate or invalid layer in independent KV collection")
        expected = len(self._capture_ids or [])
        if (key.ndim < 2 or value.ndim < 2 or key.shape != value.shape
                or key.shape[0] != expected or key.device != value.device
                or key.dtype != value.dtype):
            raise ValueError("Collected dense K/V shapes, dtype, device, or token count disagree")
        self._captures[layer_idx] = (key.detach().clone(), value.detach().clone())
        if len(self._captures) == self.num_layers:
            self._capture_armed = False

    def append_chunk(self, expected_ids: Sequence[int] | torch.Tensor,
                     drop_bos: bool = False) -> dict[str, Any]:
        ids = _token_ids(expected_ids)
        if not self.collecting or self._capture_ids != ids:
            raise ValueError("Chunk token IDs do not match the captured model prefill")
        if len(self._captures) != self.num_layers:
            raise ValueError("Every dense model layer must have a captured K/V pair")
        if drop_bos not in (True, False, 0, 1):
            raise ValueError("drop_bos must be a boolean")
        offset = int(bool(drop_bos))
        if not ids or offset >= len(ids):
            raise ValueError("Independent chunks must retain at least one token")
        for layer_idx in range(self.num_layers):
            pair = self._captures[layer_idx]
            copied = tuple(t[offset:].detach().to("cpu").contiguous().clone()
                           for t in pair)
            self._pieces[layer_idx].append(copied)
        self._collected_ids.extend(ids[offset:])
        self._chunk_count += 1
        self._chunk_lengths.append(len(ids) - offset)
        self._captures.clear()
        self._capture_ids = None
        self._capture_armed = False
        return {"chunks": self._chunk_count, "retained_chunk_tokens": len(ids) - offset,
                "collected_tokens": len(self._collected_ids)}

    def prepare(self, prompt_ids: Sequence[int] | torch.Tensor, suffix_len: int,
                ratio: float, *, alpha: float = 0.5, sink_size: int = 0,
                selected_context_indices=None, check_layer: int = 1,
                fuse_check_layer: bool | None = None,
                cache_mode: str = "cpu_pipeline",
                retain_full_cache: bool = False) -> dict[str, Any]:
        """Prepare immutable compact sources and one shared selection.

        Each independent context chunk receives floor(ratio * tokens) rows.
        K/V norms are combined and normalized per layer before layer averaging.
        Optional sinks consume their chunk budget; they are not a paper default.
        External indices contain context rows only; the suffix is appended.
        The full CPU cache is released by default. Re-prepare requires either
        a fresh collection or retain_full_cache=True on the preceding prepare.
        """
        ids = _token_ids(prompt_ids)
        if (isinstance(ratio, bool) or not isinstance(ratio, Real)
                or not math.isfinite(ratio) or not 0 <= ratio <= 1):
            raise ValueError("recompute ratio must lie in [0, 1]")
        if (isinstance(suffix_len, bool) or not isinstance(suffix_len, Integral)
                or not 0 < suffix_len < len(ids)):
            raise ValueError("Require a nonempty context and a nonempty query suffix")
        suffix_len = int(suffix_len)
        n, n_context = len(ids), len(ids) - suffix_len
        spans = context_chunk_spans(self._chunk_lengths or [n], n_context, ratio)
        budget = sum(count for _, _, count in spans)
        if (isinstance(alpha, bool) or not isinstance(alpha, Real)
                or not math.isfinite(alpha) or not 0 <= alpha <= 1):
            raise ValueError("alpha must lie in [0, 1]")
        if (isinstance(sink_size, bool) or not isinstance(sink_size, Integral)
                or not 0 <= sink_size <= budget):
            raise ValueError("sink_size must be a nonnegative integer within the context ratio budget")
        sink_size = int(sink_size)
        if any(max(0, min(end, sink_size) - start) > count
               for start, end, count in spans):
            raise ValueError("Forced sink positions exceed their independent chunk budget")
        if (isinstance(check_layer, bool) or not isinstance(check_layer, Integral)
                or not 0 <= check_layer < self.num_layers):
            raise ValueError("check_layer must be a valid zero-based layer index")
        if fuse_check_layer is None:
            fuse_check_layer = self.fuse_check_layer
        if not isinstance(fuse_check_layer, bool):
            raise ValueError("fuse_check_layer must be boolean")
        if cache_mode not in ("cpu_pipeline", "cpu_serial", "gpu"):
            raise ValueError("cache_mode must be cpu_pipeline, cpu_serial, or gpu")
        if not isinstance(retain_full_cache, bool):
            raise ValueError("retain_full_cache must be boolean")
        selected_context = None
        if selected_context_indices is not None:
            selected_context = _external_context_indices(
                selected_context_indices, n_context, budget, sink_size)
        if self._captures or self._capture_ids is not None:
            raise RuntimeError("Append the final collected chunk before preparing reuse")
        if self._collected_ids not in (ids[:n_context], ids):
            raise ValueError("Concatenated chunk IDs must match the prompt context (or legacy full prompt)")
        source_tokens = len(self._collected_ids)
        if self.device is None or self.device.type != "cuda":
            raise RuntimeError("CacheTune preparation requires captures from the CUDA model")
        if self._full_cache is None and any(not pieces for pieces in self._pieces):
            raise ValueError("No full independent cache remains; collect again or retain_full_cache=True on the preceding prepare")
        self._drain()
        self._clear_request(release_buffers=True)
        if self._full_cache is None:
            self._full_cache = [tuple(torch.cat([piece[j] for piece in pieces], dim=0)
                                     .contiguous().pin_memory() for j in (0, 1))
                                for pieces in self._pieces]
            self._pieces = [[] for _ in range(self.num_layers)]
        if any(k.shape != v.shape or k.shape[0] != source_tokens for k, v in self._full_cache):
            raise ValueError("Independent CPU K/V layers do not cover the collected context")
        chunk_details = [{"start": start, "end": end, "tokens": end-start,
                          "budget": count} for start, end, count in spans]
        if selected_context is None:
            selected_context, chunk_details = select_context_indices(
                self._full_cache, self._chunk_lengths, n_context,
                ratio, alpha, sink_size)
        suffix = torch.arange(n_context, n, dtype=torch.long)
        selected = torch.cat((selected_context, suffix)).to(torch.long)
        mask = torch.ones(n, dtype=torch.bool)
        mask[selected] = False
        reused = torch.arange(n, dtype=torch.long)[mask]
        if (selected.unique().numel() != selected.numel()
                or not torch.equal(torch.cat((selected, reused)).sort().values,
                                   torch.arange(n, dtype=torch.long))
                or not torch.equal(selected[-suffix_len:], suffix)):
            raise AssertionError("Recompute and reuse indices must partition the prompt, including its suffix")
        self.cpu_kv_cache = self.gpu_kv_cache = None
        compact = [tuple(t.index_select(0, reused).contiguous() for t in pair)
                   for pair in self._full_cache]
        self.cache_mode, self.check_layer = cache_mode, int(check_layer)
        self.fuse_check_layer = fuse_check_layer
        if cache_mode == "gpu":
            self.gpu_kv_cache = [tuple(t.to(self.device) for t in pair) for pair in compact]
            sources = self.gpu_kv_cache
        else:
            self.cpu_kv_cache = [tuple(t.pin_memory() for t in pair) for pair in compact]
            sources = self.cpu_kv_cache
        if not retain_full_cache:
            self._full_cache = None
        self._prompt_ids, self.prompt_len = ids, n
        self._selected_cpu, self._reused_cpu = selected, reused
        # Match the old runner's preparation boundary: allocate transfer and
        # reconstruction storage before the timed generation starts.
        self.selected_indices = selected.to(self.device)
        self.reused_indices = reused.to(self.device)
        for layer_idx in range(self.first_fusion_layer, self.num_layers):
            source_k, source_v = sources[layer_idx]
            self._transfer_buffers[layer_idx] = (
                torch.empty_like(source_k, device=self.device),
                torch.empty_like(source_v, device=self.device))
            work_id = (tuple(source_k.shape[1:]), source_k.dtype)
            if work_id not in self._work_buffers:
                work_shape = (n,) + tuple(source_k.shape[1:])
                self._work_buffers[work_id] = (
                    torch.empty(work_shape, dtype=source_k.dtype, device=self.device),
                    torch.empty(work_shape, dtype=source_v.dtype, device=self.device))
        self.collecting = False
        self._selection = {
            "prompt_tokens": n, "context_tokens": n_context, "suffix_tokens": int(suffix_len),
            "collected_tokens": source_tokens, "suffix_cached": source_tokens > n_context,
            "selected_context_tokens": budget, "selected_total_tokens": selected.numel(),
            "reused_tokens": reused.numel(), "selected_indices": selected.tolist(),
            "reused_indices": reused.tolist(), "num_layers": self.num_layers,
            "check_layers": [self.check_layer],
            "score_layer": None,
            "score_layers": list(range(self.num_layers)) if selected_context_indices is None else [],
            "signal": "K/V-all-layers" if selected_context_indices is None else "External-indices",
            "score_normalization": "per-chunk layer token sum + 1e-12; mean across layers",
            "chunk_selections": chunk_details,
            "chunk_context_lengths": [end-start for start, end, _ in spans],
            "chunk_budgets": [count for _, _, count in spans],
            "chunk_lengths": list(self._chunk_lengths),
            "budget_rounding": "floor per independent context chunk",
            "first_recompute_layer": self.check_layer,
            "first_fusion_layer": self.first_fusion_layer,
            "fuse_check_layer": self.fuse_check_layer,
            "selection_source": "frequency" if selected_context_indices is None else "external_context_indices",
            "alpha": float(alpha), "sink_size": sink_size,
            "recompute_ratio": float(ratio), "chunks": self._chunk_count,
            "cache_mode": self.cache_mode, "retain_full_cache": retain_full_cache,
        }
        return dict(self._selection)

    def enable(self, prompt_ids: Sequence[int] | torch.Tensor) -> dict[str, Any]:
        if not self.prepared or _token_ids(prompt_ids) != self._prompt_ids:
            raise ValueError("Prepare a cache with matching prompt IDs before enabling reuse")
        self._drain()
        self._clear_request()
        self.collecting = False
        self.enabled = True
        return {"enabled": True, "prompt_tokens": self.prompt_len}

    def disable(self) -> dict[str, Any]:
        self._drain()
        self._clear_request()
        self.collecting = self._capture_armed = False
        self._diag = {"mode": "native_full", "layers": [],
                      "native_forward_calls": 0, "transferred_bytes": 0}
        return {"enabled": False, "cache_prepared": self.prepared}

    def begin_prefill(self, positions: torch.Tensor,
                      attention_layers: Sequence[Any]) -> None:
        if not self.enabled or not self._input_verified or self.active:
            raise RuntimeError("Verify and enable this full prompt before beginning sparse prefill")
        if len(attention_layers) != self.num_layers or not self.prepared:
            raise ValueError("The adapter and compact cache must contain every dense attention layer")
        if positions.device.type != "cuda" or positions.numel() != self.prompt_len:
            raise ValueError("Sparse prefill positions must cover the full prompt on CUDA")
        if positions.device != self.device:
            raise ValueError("Prepared cache and prefill must use the same CUDA device")
        self.device = positions.device
        if self.selected_indices is None or self.selected_indices.device != self.device:
            self.selected_indices = self._selected_cpu.to(self.device)
            self.reused_indices = self._reused_cpu.to(self.device)
        self._positions = positions
        self._reuse_positions = positions.index_select(0, self.reused_indices)
        self._attentions = list(attention_layers)
        if self.cache_mode == "cpu_pipeline":
            if self._transfer_stream is None or self._stream_device != self.device:
                self._transfer_stream = torch.cuda.Stream(device=self.device)
                self._rope_stream = torch.cuda.Stream(device=self.device)
                self._copy_events = [torch.cuda.Event() for _ in range(self.num_layers)]
                self._ready_events = [torch.cuda.Event() for _ in range(self.num_layers)]
                self._stream_device = self.device
            main = torch.cuda.current_stream(self.device)
            main.wait_stream(self._transfer_stream)
            main.wait_stream(self._rope_stream)
            self._transfer_stream.wait_stream(main)
            self._rope_stream.wait_stream(main)
        self._prefetched.clear()
        self._ready_pairs.clear()
        self._diag = {"prefill_count": self._diag.get("prefill_count", 0) + 1,
                      "mode": "cachetune", "native_forward_calls": 0,
                      "layers": [], "transferred_bytes": 0, "device_copy_bytes": 0,
                      "merged_layers": [],
                      "prompt_tokens": self.prompt_len,
                      "selected_tokens": self.selected_indices.numel(),
                      "reused_tokens": self.reused_indices.numel()}
        self._diag["allocated_transfer_bytes"] = sum(
            t.numel() * t.element_size() for pair in self._transfer_buffers.values() for t in pair)
        self._next_layer = self.first_fusion_layer
        self.active = True
        if self.cache_mode == "cpu_pipeline" and self.first_fusion_layer < self.num_layers:
            self._prefetch(self.first_fusion_layer)

    def _rotate_reused_key(self, layer_idx: int, key: torch.Tensor) -> torch.Tensor:
        if not key.shape[0]:
            return key
        attention = self._attentions[layer_idx]
        q_size = getattr(attention, "q_size", None)
        if q_size is None:
            q_size = int(attention.num_heads) * int(attention.head_dim)
        fake_key = (int(q_size), key.dtype)
        if fake_key not in self._fake_queries:
            self._fake_queries[fake_key] = torch.zeros(
                (key.shape[0], int(q_size)), device=self.device, dtype=key.dtype)
        _, rotated = attention.rotary_emb(
            self._reuse_positions, self._fake_queries[fake_key],
            key.reshape(key.shape[0], -1))
        return rotated.reshape(key.shape)

    def _prefetch(self, layer_idx: int) -> None:
        if self.cache_mode != "cpu_pipeline":
            raise RuntimeError("Only cpu_pipeline may prefetch on a separate stream")
        cpu_k, cpu_v = self.cpu_kv_cache[layer_idx]
        if any(t.device.type != "cpu" or (t.numel() and not t.is_pinned())
               for t in (cpu_k, cpu_v)):
            raise ValueError("Pipeline inputs must be immutable pinned CPU K/V")
        if layer_idx not in self._transfer_buffers:
            self._transfer_buffers[layer_idx] = (
                torch.empty_like(cpu_k, device=self.device),
                torch.empty_like(cpu_v, device=self.device))
        k_gpu, v_gpu = self._transfer_buffers[layer_idx]
        main = torch.cuda.current_stream(self.device)
        # Buffers allocated on main must be initialized before other streams use them.
        self._transfer_stream.wait_stream(main)
        with torch.cuda.stream(self._transfer_stream):
            k_gpu.copy_(cpu_k, non_blocking=True)
            v_gpu.copy_(cpu_v, non_blocking=True)
            self._copy_events[layer_idx].record(self._transfer_stream)
        with torch.cuda.stream(self._rope_stream):
            self._rope_stream.wait_event(self._copy_events[layer_idx])
            k_gpu = self._rotate_reused_key(layer_idx, k_gpu)
            self._ready_events[layer_idx].record(self._rope_stream)
        k_gpu.record_stream(self._rope_stream)
        self._prefetched[layer_idx] = (k_gpu, v_gpu)
        self._diag["transferred_bytes"] += sum(t.numel() * t.element_size() for t in (cpu_k, cpu_v))

    def _load_serial(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Copy and rotate only this layer, on the current compute stream."""
        sources = self.gpu_kv_cache if self.cache_mode == "gpu" else self.cpu_kv_cache
        source_k, source_v = sources[layer_idx]
        if self.cache_mode == "cpu_serial":
            if any(t.device.type != "cpu" or (t.numel() and not t.is_pinned())
                   for t in (source_k, source_v)):
                raise ValueError("cpu_serial inputs must be immutable pinned CPU K/V")
        elif self.cache_mode != "gpu" or any(t.device != self.device for t in (source_k, source_v)):
            raise ValueError("gpu inputs must be immutable compact K/V on the execution device")
        key, value = self._transfer_buffers[layer_idx]
        # non_blocking changes host waiting, not stream ordering: copy, RoPE,
        # then this layer's computation execute sequentially on the same stream.
        key.copy_(source_k, non_blocking=True)
        value.copy_(source_v, non_blocking=True)
        key = self._rotate_reused_key(layer_idx, key)
        counter = "device_copy_bytes" if self.cache_mode == "gpu" else "transferred_bytes"
        self._diag[counter] += _cache_bytes([(source_k, source_v)])
        return key, value

    def enter_layer(self, layer_idx: int) -> None:
        """Optional pre-QKV hook: overlap the next transfer with this layer.

        Calling this before the fusion layer forward preserves the original
        pipeline's overlap window. ``merge`` calls it as a fallback if omitted.
        """
        if not self.active or not self.first_fusion_layer <= layer_idx < self.num_layers:
            raise ValueError("Only layers at or after the first recompute layer consume cached K/V")
        if layer_idx in self._ready_pairs:
            return
        if layer_idx != self._next_layer:
            raise RuntimeError("Fusion layers must be merged once, in increasing layer order")
        if self.cache_mode != "cpu_pipeline":
            self._ready_pairs[layer_idx] = self._load_serial(layer_idx)
            return
        if layer_idx not in self._prefetched:
            raise RuntimeError("Missing prefetched layer; fusion layers must execute in order")
        main = torch.cuda.current_stream(self.device)
        main.wait_event(self._ready_events[layer_idx])
        old_k, old_v = self._prefetched.pop(layer_idx)
        for tensor in (old_k, old_v):
            tensor.record_stream(main)
        self._ready_pairs[layer_idx] = (old_k, old_v)
        if layer_idx + 1 < self.num_layers:
            self._prefetch(layer_idx + 1)

    def merge(self, layer_idx: int, key: torch.Tensor,
              value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Wait for reused global-RoPE K/V and merge every original token row."""
        if not self.active or not self.first_fusion_layer <= layer_idx < self.num_layers:
            raise ValueError("Only layers at or after the first recompute layer may merge cached K/V")
        if (key.shape != value.shape or key.shape[0] != self.selected_indices.numel()
                or key.device != self.device or value.device != key.device
                or value.dtype != key.dtype):
            raise ValueError("Sparse K/V must match selected rows, device, and dtype")
        self.enter_layer(layer_idx)
        old_k, old_v = self._ready_pairs.pop(layer_idx)
        features = key.shape[1:]
        old_k, old_v = old_k.reshape((-1,) + features), old_v.reshape((-1,) + features)
        if old_k.shape[0] != self.reused_indices.numel() or old_k.dtype != key.dtype:
            raise ValueError("Reused cache shape or dtype does not match sparse K/V")
        shape = (self.prompt_len,) + features
        # Work buffers are shared across layers with the same feature dimensions.
        work_id = (tuple(features), key.dtype)
        if work_id not in self._work_buffers:
            self._work_buffers[work_id] = (torch.empty(shape, dtype=key.dtype, device=key.device),
                                           torch.empty(shape, dtype=value.dtype, device=value.device))
        full_k, full_v = self._work_buffers[work_id]
        full_k.index_copy_(0, self.reused_indices, old_k)
        full_v.index_copy_(0, self.reused_indices, old_v)
        full_k.index_copy_(0, self.selected_indices, key)
        full_v.index_copy_(0, self.selected_indices, value)
        self._diag["merged_layers"].append(int(layer_idx))
        self._next_layer += 1
        return full_k, full_v

    def finish_prefill(self) -> None:
        if not self.active:
            raise RuntimeError("No active CacheTune prefill to finish")
        expected = list(range(self.first_fusion_layer, self.num_layers))
        if (self._diag.get("merged_layers") != expected or self._prefetched
                or self._ready_pairs):
            raise RuntimeError("The prefill did not consume every fusion layer")
        self.active = self.enabled = self._input_verified = False
        # CUDA tensors remain alive until disable/reset synchronizes. This also
        # lets the adapter scatter final hidden rows after finish_prefill().

    def record_layer(self, layer_idx: int, **row_counts: Any) -> None:
        self._diag.setdefault("layers", []).append({"layer": int(layer_idx), **row_counts})

    def diagnostics(self) -> dict[str, Any]:
        return {"collecting": self.collecting, "enabled": self.enabled,
                "active": self.active, "prepared": self.prepared,
                "chunks": self._chunk_count, "selection": self._selection,
                **self._diag, **self._storage_diagnostics()}
