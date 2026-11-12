"""Opt-in, single-request snapshots matching the legacy debug semantics.

These CPU copies and the attention proxy are diagnostic work, not part of the
timed/default inference path. The proxy is deliberately unmasked and must not
be interpreted as the attention probabilities used by the serving backend.
"""

from __future__ import annotations

from copy import deepcopy
import math
from numbers import Integral, Real
from typing import Iterable

import torch


def _integer(value, name, *, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _tensor(value, name):
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim not in (2, 3) or not value.is_floating_point():
        raise ValueError(f"{name} must be a floating 2D or 3D tensor")
    return value


def _cpu_copy(value):
    # clone is essential even when the caller already supplies a CPU tensor.
    return value.detach().to(device="cpu").clone()


class CacheTuneDebugCapture:
    """Collect the first snapshot of each requested kind at each layer.

    ``raw_qkv`` preserves the old ``plot_attn`` capture point: after RoPE but
    before query selection and cached-KV merging. It does not mean pre-RoPE.
    ``attention_proxy`` preserves the old ``save_attn`` head-zero unmasked
    softmax, with suffix queries located by their original token positions.
    Call ``reset`` between requests; later decode calls cannot overwrite an
    already captured layer. ``export`` and ``finish`` return independent CPU
    dictionaries suitable for ``torch.save`` and do not clear the capture.
    """

    def __init__(
        self,
        layers: Iterable[int] | None = None,
        raw_qkv: bool = False,
        attention_proxy: bool = False,
        suffix_len: int = 1,
    ):
        if not isinstance(raw_qkv, bool) or not isinstance(attention_proxy, bool):
            raise TypeError("raw_qkv and attention_proxy must be bool")
        self.layers = (
            None if layers is None
            else frozenset(_integer(layer, "layer") for layer in layers)
        )
        self.raw_qkv = raw_qkv
        self.attention_proxy = attention_proxy
        self.suffix_len = _integer(suffix_len, "suffix_len")
        self.reset()

    def reset(self):
        """Clear snapshots while retaining the configured capture options."""
        self._captures = {}

    def enabled_for(self, layer):
        layer = _integer(layer, "layer")
        return ((self.raw_qkv or self.attention_proxy)
                and (self.layers is None or layer in self.layers))

    def capture_projected_qkv(self, layer, q, k, v):
        """Snapshot post-RoPE Q/K/V before Q selection and cached-KV merging."""
        layer = _integer(layer, "layer")
        if not self.raw_qkv or not self.enabled_for(layer):
            return
        existing = self._captures.get(layer, {})
        if "projected_qkv" in existing:
            return
        q, k, v = (_tensor(t, name) for t, name in ((q, "q"), (k, "k"), (v, "v")))
        if k.shape != v.shape:
            raise ValueError("projected K and V must have matching shapes")
        snapshot = {
            "q": _cpu_copy(q), "k": _cpu_copy(k), "v": _cpu_copy(v),
            "metadata": {
                "stage": "post_rope_pre_query_selection_pre_kv_merge",
                "legacy_option": "plot_attn",
                "query_rows": int(q.shape[0]),
                "key_rows": int(k.shape[0]),
                "input_shapes": {"q": list(q.shape), "k": list(k.shape), "v": list(v.shape)},
            },
        }
        self._captures.setdefault(layer, {})["projected_qkv"] = snapshot

    def capture_after_rope(
        self, layer, q, k, query_positions, full_length,
        num_heads, num_kv_heads, head_dim, scale,
    ):
        """Capture the suffix/head-zero unmasked proxy using merged full K.

        Q may contain all rows or selected rows; ``query_positions[i]`` must
        describe Q row i in the original prompt. K must contain all original
        prompt positions, in order. All suffix positions must be present in Q.
        Both flattened [rows, heads * dim] and [rows, heads, dim] are accepted.
        """
        layer = _integer(layer, "layer")
        if not self.attention_proxy or not self.enabled_for(layer):
            return
        if "attention_proxy" in self._captures.get(layer, {}):
            return
        n = _integer(full_length, "full_length", minimum=1)
        hq = _integer(num_heads, "num_heads", minimum=1)
        hkv = _integer(num_kv_heads, "num_kv_heads", minimum=1)
        dim = _integer(head_dim, "head_dim", minimum=1)
        if hq % hkv:
            raise ValueError("num_heads must be divisible by num_kv_heads for GQA")
        if self.suffix_len > n:
            raise ValueError("suffix_len exceeds full_length")
        if (isinstance(scale, bool) or not isinstance(scale, Real)
                or not math.isfinite(scale) or scale <= 0):
            raise ValueError("scale must be a positive finite number")
        q, k = _tensor(q, "q"), _tensor(k, "k")
        m = int(q.shape[0])
        for name, tensor, rows, heads in (("q", q, m, hq), ("k", k, n, hkv)):
            expected = (rows, heads * dim) if tensor.ndim == 2 else (rows, heads, dim)
            if tuple(tensor.shape) != expected:
                raise ValueError(f"{name} shape must be {expected}; got {tuple(tensor.shape)}")
        if not isinstance(query_positions, torch.Tensor):
            raise TypeError("query_positions must be a torch.Tensor")
        if query_positions.ndim != 1 or query_positions.numel() != m:
            raise ValueError("query_positions must contain one original position per Q row")
        if query_positions.dtype not in (torch.int32, torch.int64):
            raise TypeError("query_positions must have int32 or int64 dtype")
        positions = _cpu_copy(query_positions).to(dtype=torch.int64)
        if m > n or (positions.numel() and (positions.min() < 0 or positions.max() >= n)):
            raise ValueError("query_positions must be within the full prompt")
        if positions.unique().numel() != m:
            raise ValueError("query_positions must be unique")
        suffix_rows = torch.nonzero(positions >= n - self.suffix_len).flatten()
        suffix_rows = suffix_rows[torch.argsort(positions[suffix_rows])]
        suffix_positions = positions[suffix_rows]
        expected_suffix = torch.arange(n - self.suffix_len, n, dtype=torch.int64)
        if not torch.equal(suffix_positions, expected_suffix):
            raise ValueError("Q must contain every original suffix position")

        # For grouped-query attention, query head h maps to KV head h // group.
        # We preserve the legacy head-0 proxy rather than averaging all heads.
        query_head = 0
        kv_head = query_head // (hq // hkv)
        q_head = q.reshape(m, hq, dim)[:, query_head, :]
        q_suffix = _cpu_copy(q_head.index_select(0, suffix_rows.to(q.device))).float()
        k_all = _cpu_copy(k.reshape(n, hkv, dim)[:, kv_head, :]).float()
        with torch.no_grad():
            weights = torch.softmax(q_suffix @ k_all.transpose(0, 1) * float(scale), dim=-1)
        self._captures.setdefault(layer, {})["attention_proxy"] = {
            "weights": weights,
            "query_positions": suffix_positions.clone(),
            "key_positions": torch.arange(n, dtype=torch.int64),
            "metadata": {
                "stage": "post_rope_post_query_selection_post_kv_merge",
                "legacy_option": "save_attn",
                "definition": "softmax(Q_suffix_head0 @ K_all_matching_GQA_head.T * scale)",
                "is_actual_attention": False,
                "causal_mask_applied": False,
                "sliding_window_mask_applied": False,
                "compute_dtype": "float32",
                "query_head": query_head,
                "kv_head": kv_head,
                "full_length": n,
                "query_rows": m,
                "query_mode": "full" if m == n else "sparse",
                "suffix_len": self.suffix_len,
                "num_heads": hq,
                "num_kv_heads": hkv,
                "head_dim": dim,
                "scale": float(scale),
            },
        }

    def export(self):
        """Return an independent, CPU-only snapshot without clearing it."""
        return deepcopy({
            "schema_version": 1,
            "configuration": {
                "layers": None if self.layers is None else sorted(self.layers),
                "raw_qkv": self.raw_qkv,
                "attention_proxy": self.attention_proxy,
                "suffix_len": self.suffix_len,
            },
            "metadata": {
                "capture_policy": "first_capture_per_layer_until_reset",
                "attention_proxy_is_actual_attention": False,
                "attention_proxy_note": "Unmasked head-zero diagnostic proxy, not backend attention probabilities.",
            },
            "layers": self._captures,
        })

    def finish(self):
        """Alias for export; callers decide when to reset or disable capture."""
        return self.export()
