"""Branch-output storage for dense-attention/GDN hybrid CacheTune prefill.

The inherited KV state uses logical full-attention layer IDs. Branch caches
use actual decoder layer IDs. They hold branch *outputs*, never residuals or
recurrent states: the adapter must add them to the current full residual.
"""
from __future__ import annotations

from numbers import Integral

import torch

from .state import CacheTuneState, _token_ids


def _bytes(mapping):
    return sum(t.numel() * t.element_size() for row in mapping.values() for t in row.values())


class HybridCacheTuneState(CacheTuneState):
    def __init__(self, full_attention_layer_ids, num_actual_layers=None, check_layer=0):
        ids = list(full_attention_layer_ids)
        if (len(ids) < 2 or any(isinstance(i, bool) or not isinstance(i, Integral) or i < 0 for i in ids)
                or ids != sorted(set(ids))):
            raise ValueError("Full-attention layer IDs must be increasing unique nonnegative integers (at least two)")
        self.full_attention_layer_ids = tuple(int(i) for i in ids)
        if num_actual_layers is None:
            num_actual_layers = ids[-1] + 1
        if (isinstance(num_actual_layers, bool) or not isinstance(num_actual_layers, Integral)
                or num_actual_layers <= ids[-1]):
            raise ValueError("num_actual_layers must cover every full-attention layer")
        self.num_actual_layers = int(num_actual_layers)
        self.hybrid_policy = "block"
        self.hybrid_boundary = None
        self.required_branches = {}
        self._branch_captures = {}
        self._branch_pieces = {}
        self._branch_full = None
        self._branch_sources = {}
        self._branch_transfer_buffers = {}
        self._branch_work_buffers = {}
        self._branch_stream = None
        self._branch_stream_device = None
        self._branch_events = {}
        self._branch_diag = {}
        self._clear_branch_request()
        super().__init__(len(ids), check_layer=check_layer)

    def _clear_branch_request(self, release_buffers=False):
        self._branch_active = False
        self._branch_order = []
        self._branch_cursor = 0
        self._branch_current = None
        self._branch_prefetched = {}
        self._branch_ready = {}
        self._branch_used = set()
        self._branch_diag = {}
        if release_buffers:
            self._branch_transfer_buffers.clear()
            self._branch_work_buffers.clear()

    def _clear_request(self, release_buffers=False):
        super()._clear_request(release_buffers=release_buffers)
        self._clear_branch_request(release_buffers=release_buffers)

    def reset(self, clear_cache=True):
        result = super().reset(clear_cache=clear_cache)
        self._branch_captures.clear()
        if clear_cache:
            # Recurrent probes are request-specific, unlike enable/disable
            # which intentionally preserves a same-prompt comparison reference.
            self._recurrent_probe_next = False
            self._recurrent_snapshot = None
            self._recurrent_reference = None
            self._branch_pieces.clear()
            self._branch_full = None
            self._branch_sources.clear()
            self.required_branches = {}
            self.hybrid_boundary = None
            self.hybrid_policy = "block"
        return result

    def capture_branch(self, layer, name, tensor):
        """Capture post-projection/scaling attention or post-scaling MLP output.

        Call only inside the adapter's collection-prefill loop. The final KV
        capture disarms the base collector before that layer's branches exist,
        so this method deliberately does not inspect ``_capture_armed``.
        """
        if not self.collecting or self._capture_ids is None:
            return
        if (isinstance(layer, bool) or not isinstance(layer, Integral)
                or not 0 <= layer < self.num_actual_layers):
            raise ValueError("Invalid actual decoder layer for branch capture")
        if name not in ("attention", "mlp"):
            raise ValueError("Branch name must be attention or mlp")
        layer = int(layer)
        # GDN's attention/recurrent output is always recomputed under all policies.
        if name == "attention" and layer not in self.full_attention_layer_ids:
            return
        if name in self._branch_captures.get(layer, {}):
            raise ValueError("Duplicate independent branch capture")
        if (not isinstance(tensor, torch.Tensor) or tensor.ndim != 2
                or not tensor.is_floating_point()
                or tensor.shape[0] != len(self._capture_ids)
                or tensor.device != self.device):
            raise ValueError("Branch capture must contain all chunk rows on the collection device")
        self._branch_captures.setdefault(layer, {})[name] = tensor.detach().clone()

    def append_chunk(self, expected_ids, drop_bos=False):
        ids = _token_ids(expected_ids)
        if not self.collecting or self._capture_ids != ids:
            raise ValueError("Chunk IDs do not match the branch collection")
        if drop_bos not in (True, False, 0, 1) or not ids or int(bool(drop_bos)) >= len(ids):
            raise ValueError("Invalid chunk BOS drop")
        expected = {(i, "mlp") for i in range(self.num_actual_layers)}
        expected.update((i, "attention") for i in self.full_attention_layer_ids)
        actual = {(i, name) for i, row in self._branch_captures.items() for name in row}
        if actual != expected:
            raise ValueError(f"Incomplete branch collection; missing={sorted(expected - actual)}")
        offset = int(bool(drop_bos))
        copies = {i: {name: t[offset:].detach().cpu().contiguous().clone()
                      for name, t in row.items()}
                  for i, row in self._branch_captures.items()}
        result = super().append_chunk(ids, drop_bos=drop_bos)
        for i, row in copies.items():
            for name, tensor in row.items():
                self._branch_pieces.setdefault(i, {}).setdefault(name, []).append(tensor)
        self._branch_captures.clear()
        return {**result, "branch_layers": self.num_actual_layers}

    def _required(self, policy, check_layer):
        """Return branch reuse beginning at a logical full-attention boundary.

        ``block`` with check_layer=0 is the paper path. ``attention`` and
        ``mlp`` are retained as experimental ablations. The latter additionally
        reuses MLP branches in recurrent layers, while still updating every
        Gated DeltaNet state from the full ordered sequence.
        """
        if policy not in ("attention", "block", "mlp"):
            raise ValueError("hybrid_policy must be attention, block, or mlp")
        if (isinstance(check_layer, bool) or not isinstance(check_layer, Integral)
                or not 0 <= check_layer < self.num_layers):
            raise ValueError("Hybrid check_layer is a logical full-attention index")
        boundary = self.full_attention_layer_ids[check_layer]
        required = {}
        for i in range(boundary, self.num_actual_layers):
            names = []
            if i in self.full_attention_layer_ids:
                names.append("attention")
            if policy == "mlp" or (policy == "block" and i in self.full_attention_layer_ids):
                names.append("mlp")
            if names:
                required[i] = tuple(names)
        return boundary, required

    def prepare(self, prompt_ids, suffix_len, ratio, *, hybrid_policy="block",
                check_layer=0, retain_full_cache=False, **options):
        if "fuse_check_layer" in options:
            raise ValueError("Hybrid CacheTune always fuses its check layer; do not override fuse_check_layer")
        boundary, required = self._required(hybrid_policy, check_layer)
        ids = _token_ids(prompt_ids)
        if self._branch_captures:
            raise RuntimeError("Append branch captures before preparing")
        # Validate branch availability before the base prepare mutates KV state.
        full = self._branch_full
        if full is None:
            if not self._branch_pieces:
                raise ValueError("No full branch cache remains; recollect or retain_full_cache=True")
            full = {i: {name: torch.cat(parts, dim=0).contiguous()
                        for name, parts in row.items()
                        if retain_full_cache or name in required.get(i, ())}
                    for i, row in self._branch_pieces.items()
                    if retain_full_cache or i in required}
        for i, names in required.items():
            for name in names:
                source = full.get(i, {}).get(name)
                if (source is None or source.ndim != 2
                        or source.shape[0] != len(self._collected_ids)):
                    raise ValueError(f"Missing full ordered branch rows at layer {i}, {name}")
        selection = super().prepare(ids, suffix_len, ratio, check_layer=check_layer,
                                    fuse_check_layer=True,
                                    retain_full_cache=retain_full_cache, **options)
        compact = {i: {name: full[i][name].index_select(0, self._reused_cpu).contiguous()
                       for name in names} for i, names in required.items()}
        self._branch_sources = {
            i: {name: (t.to(self.device) if self.cache_mode == "gpu" else t.pin_memory())
                for name, t in row.items()}
            for i, row in compact.items()
        }
        self._branch_full = full if retain_full_cache else None
        self._branch_pieces.clear()
        self.hybrid_policy, self.hybrid_boundary = hybrid_policy, boundary
        self.required_branches = required
        # Allocate request buffers before measured generation. GPU-resident
        # branches need no scratch copy: merge only reads their immutable source.
        if self.cache_mode != "gpu":
            self._branch_transfer_buffers = {
                i: {name: torch.empty_like(t, device=self.device) for name, t in row.items()}
                for i, row in self._branch_sources.items()
            }
        for row in self._branch_sources.values():
            for name, t in row.items():
                key = (name, tuple(t.shape[1:]), t.dtype)
                if t.shape[0] and key not in self._branch_work_buffers:
                    self._branch_work_buffers[key] = torch.empty(
                        (self.prompt_len,) + tuple(t.shape[1:]), dtype=t.dtype, device=self.device)
        logical_scores = selection.get("score_layers", [])
        actual_scores = [self.full_attention_layer_ids[i] for i in logical_scores]
        self._selection.update({
            "hybrid_policy": hybrid_policy,
            "full_attention_layer_ids": list(self.full_attention_layer_ids),
            "num_actual_layers": self.num_actual_layers,
            "hybrid_boundary": boundary,
            # Base scoring combines K/V from every full-attention layer; the
            # logical-to-actual map must not label it first-layer Value-only.
            "score_actual_layer": None,
            "score_actual_layers": actual_scores,
            "scored_layer_indices": actual_scores,
            "check_actual_layers": [boundary],
            "hybrid_paper_default": hybrid_policy == "block" and check_layer == 0,
            "hybrid_experimental": hybrid_policy != "block" or check_layer != 0,
        })
        return dict(self._selection)

    def begin_hybrid(self):
        """Call after inherited begin_prefill; branch sources remain immutable."""
        if not self.active or not self.required_branches or self._branch_active:
            raise RuntimeError("Begin a prepared KV prefill before beginning hybrid branches")
        self._clear_branch_request()
        self._branch_active = True
        self._branch_order = sorted(self.required_branches)
        self._branch_diag = {"branch_transferred_bytes": 0, "branch_device_copy_bytes": 0,
                             "branch_prefetched_layers": [], "branch_merged": []}
        if self.cache_mode == "cpu_pipeline":
            if self._branch_stream is None or self._branch_stream_device != self.device:
                self._branch_stream = torch.cuda.Stream(device=self.device)
                self._branch_stream_device = self.device
                self._branch_events = {}
            for i in self._branch_order:
                if i not in self._branch_events:
                    self._branch_events[i] = torch.cuda.Event()
            main = torch.cuda.current_stream(self.device)
            main.wait_stream(self._branch_stream)
            self._branch_stream.wait_stream(main)
            self._prefetch_branch(self._branch_order[0])

    def _prefetch_branch(self, layer):
        source = self._branch_sources[layer]
        if any(t.device.type != "cpu" or (t.numel() and not t.is_pinned()) for t in source.values()):
            raise ValueError("Branch pipeline sources must be pinned CPU snapshots")
        targets = self._branch_transfer_buffers[layer]
        with torch.cuda.stream(self._branch_stream):
            for name, t in source.items():
                targets[name].copy_(t, non_blocking=True)
            self._branch_events[layer].record(self._branch_stream)
        self._branch_prefetched[layer] = targets
        self._branch_diag["branch_transferred_bytes"] += _bytes({layer: source})
        self._branch_diag["branch_prefetched_layers"].append(layer)

    def enter_branch_layer(self, layer):
        """Wait for this layer, then overlap the next needed layer's copy."""
        if not self._branch_active:
            raise RuntimeError("No active hybrid branch prefill")
        if self._branch_current == layer:
            return
        if (self._branch_current is not None or self._branch_cursor >= len(self._branch_order)
                or self._branch_order[self._branch_cursor] != layer):
            raise RuntimeError("Branch layers must be fully consumed in actual decoder order")
        if self.cache_mode == "cpu_pipeline":
            if layer not in self._branch_prefetched:
                raise RuntimeError("Missing prefetched hybrid branch layer")
            main = torch.cuda.current_stream(self.device)
            main.wait_event(self._branch_events[layer])
            ready = self._branch_prefetched.pop(layer)
            for t in ready.values():
                t.record_stream(main)
            if self._branch_cursor + 1 < len(self._branch_order):
                self._prefetch_branch(self._branch_order[self._branch_cursor + 1])
        elif self.cache_mode == "cpu_serial":
            ready = self._branch_transfer_buffers[layer]
            source = self._branch_sources[layer]
            for name, t in source.items():
                if t.device.type != "cpu" or (t.numel() and not t.is_pinned()):
                    raise ValueError("Serial branch sources must be pinned CPU snapshots")
                ready[name].copy_(t, non_blocking=True)
            self._branch_diag["branch_transferred_bytes"] += _bytes({layer: source})
        elif self.cache_mode == "gpu":
            ready = self._branch_sources[layer]
            if any(t.device != self.device for t in ready.values()):
                raise ValueError("GPU branch snapshots must reside on the execution device")
        else:
            raise RuntimeError("Unknown hybrid cache mode")
        # Own the mapping; consuming a name must never delete its source/buffer.
        self._branch_ready[layer] = dict(ready)
        self._branch_current = layer

    def merge_branch(self, layer, name, selected_tensor):
        """Reconstruct a full branch; the caller retains the current residual."""
        if name not in self.required_branches.get(layer, ()):
            raise ValueError("This layer/branch is not reused under the prepared hybrid policy")
        if (layer, name) in self._branch_used:
            raise RuntimeError("A hybrid branch may be merged only once per prefill")
        self.enter_branch_layer(layer)
        old = self._branch_ready[layer][name]
        if (not isinstance(selected_tensor, torch.Tensor) or selected_tensor.ndim != old.ndim
                or selected_tensor.shape[0] != self.selected_indices.numel()
                or selected_tensor.shape[1:] != old.shape[1:]
                or selected_tensor.device != self.device or selected_tensor.dtype != old.dtype):
            raise ValueError("Updated branch rows must match the selected positions and cached features")
        if old.shape[0] != self.reused_indices.numel():
            raise ValueError("Cached branch rows must match the complementary positions")
        if not old.shape[0]:
            # M=N: no reused branch contributes and no arithmetic is introduced.
            output = selected_tensor
        else:
            key = (name, tuple(old.shape[1:]), old.dtype)
            output = self._branch_work_buffers[key]
            output.index_copy_(0, self.reused_indices, old)
            output.index_copy_(0, self.selected_indices, selected_tensor)
        self._branch_ready[layer].pop(name)
        self._branch_used.add((layer, name))
        self._branch_diag["branch_merged"].append({"layer": layer, "name": name,
                                                  "selected_rows": int(selected_tensor.shape[0]),
                                                  "full_rows": self.prompt_len})
        if not self._branch_ready[layer]:
            del self._branch_ready[layer]
            self._branch_current = None
            self._branch_cursor += 1
        return output

    def finish_hybrid(self):
        expected = {(i, name) for i, names in self.required_branches.items() for name in names}
        if (not self._branch_active or self._branch_used != expected
                or self._branch_cursor != len(self._branch_order)
                or self._branch_prefetched or self._branch_ready):
            raise RuntimeError("The hybrid prefill did not consume every required branch")
        self._branch_active = False

    def _storage_diagnostics(self):
        work_bytes = sum(t.numel() * t.element_size() for t in self._branch_work_buffers.values())
        return {**super()._storage_diagnostics(),
                "retained_full_branch_cpu_bytes": _bytes(self._branch_full or {}),
                "compact_branch_cpu_bytes": _bytes(self._branch_sources) if self.cache_mode != "gpu" else 0,
                "compact_branch_gpu_bytes": _bytes(self._branch_sources) if self.cache_mode == "gpu" else 0,
                "branch_gpu_transfer_buffer_bytes": _bytes(self._branch_transfer_buffers),
                "branch_gpu_work_buffer_bytes": work_bytes}

    def diagnostics(self):
        result = super().diagnostics()
        result.update(self._branch_diag)
        result.update({"hybrid_policy": self.hybrid_policy, "hybrid_boundary": self.hybrid_boundary,
                       "full_attention_layer_ids": list(self.full_attention_layer_ids),
                       "num_actual_layers": self.num_actual_layers,
                       "scored_layer_indices": (self._selection or {}).get("scored_layer_indices", []),
                       "hybrid_paper_default": (self._selection or {}).get("hybrid_paper_default", False),
                       "hybrid_experimental": (self._selection or {}).get("hybrid_experimental", False),
                       "required_branches": {i: list(names) for i, names in self.required_branches.items()}})
        result["total_transferred_bytes"] = (result.get("transferred_bytes", 0)
                                              + result.get("branch_transferred_bytes", 0))
        return result
