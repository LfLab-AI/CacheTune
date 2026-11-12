"""Original-position sparse prefill using FlashInfer 0.6.4.

The caller supplies selected Q rows and complete, merged K/V in token order.
This module neither applies RoPE nor writes vLLM's paged cache. Masks belong to
one request and can be reused across its layers. Construct/prepare plans outside
CUDA graph capture; this implementation is intended for eager prefill.

API reference (Apache-2.0 upstream):
https://github.com/flashinfer-ai/flashinfer/blob/v0.6.4/flashinfer/prefill.py
https://github.com/flashinfer-ai/flashinfer/blob/v0.6.4/flashinfer/quantization.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real

import torch


_MASK_CHUNK_BYTES = 8 * 1024 * 1024


def _normalize_window(window: int | None) -> int | None:
    if window is None:
        return None
    if isinstance(window, bool) or not isinstance(window, Integral):
        raise TypeError("window must be a positive integer, -1, or None")
    if window == -1:
        return None
    if window <= 0:
        raise ValueError("window counts visible tokens including self and must be positive")
    return int(window)


@dataclass
class _PackedMask:
    tensor: torch.Tensor
    ready: torch.cuda.Event


class SparseAttentionPlan:
    """Request-local exact causal mask for non-contiguous selected queries.

    ``selected_positions`` must be a non-empty, strictly increasing 1D integer
    tensor in ``[0, full_length)``. Its values are snapshotted on CPU once; CUDA
    input therefore synchronizes during construction. ``window`` counts the
    maximum visible tokens including the query's own position. None/-1 means
    unbounded. A run-time None uses this default; -1 explicitly overrides it
    with unbounded attention.

    Q/K must already have the desired positional encoding. The plan owns only
    metadata and packed masks, never model K/V. Do not reuse it for a different
    request or mutate its private state. Calls are intended to be serialized
    by the worker; concurrent host calls on the same plan are unsupported.
    """

    def __init__(
        self,
        selected_positions: torch.Tensor,
        full_length: int,
        window: int | None = None,
    ) -> None:
        if isinstance(full_length, bool) or not isinstance(full_length, Integral):
            raise TypeError("full_length must be a positive integer")
        if full_length <= 0:
            raise ValueError("full_length must be positive")
        if not isinstance(selected_positions, torch.Tensor):
            raise TypeError("selected_positions must be a torch.Tensor")
        if selected_positions.ndim != 1:
            raise ValueError("selected_positions must be one-dimensional")
        if selected_positions.dtype not in (torch.int32, torch.int64):
            raise TypeError("selected_positions must have int32 or int64 dtype")
        if selected_positions.numel() == 0:
            raise ValueError("at least one query must be selected")
        positions = selected_positions.detach().to(device="cpu", dtype=torch.int64).clone()
        if positions[0].item() < 0 or positions[-1].item() >= full_length:
            raise ValueError("selected_positions must lie in [0, full_length)")
        if torch.any(positions[1:] <= positions[:-1]).item():
            raise ValueError("selected_positions must be strictly increasing and unique")
        self._positions_cpu = positions
        self._full_length = int(full_length)
        self._default_window = _normalize_window(window)
        self._masks: dict[tuple[torch.device, int | None], _PackedMask] = {}

    @property
    def full_length(self) -> int:
        return self._full_length

    @property
    def num_queries(self) -> int:
        return self._positions_cpu.numel()

    @property
    def selected_positions(self) -> torch.Tensor:
        """A CPU copy, so caller mutation cannot invalidate cached masks."""
        return self._positions_cpu.clone()

    @property
    def cached_mask_count(self) -> int:
        return len(self._masks)

    def _window(self, window: int | None) -> int | None:
        return self._default_window if window is None else _normalize_window(window)

    def prepare(
        self, device: torch.device | str, window: int | None = None
    ) -> None:
        """Build the packed mask once for this device/window, before layer use.

        Lazy preparation in ``run`` is also supported. Mask construction and
        packing are real request work and must be counted in end-to-end TTFT.
        """
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("FlashInfer sparse prefill requires a CUDA device")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        resolved_window = self._window(window)
        cache_key = (device, resolved_window)
        if cache_key in self._masks:
            return
        # Import lazily: CPU metadata tests and importing the package do not
        # require a working CUDA/FlashInfer installation. Errors propagate.
        from flashinfer.quantization import packbits

        with torch.cuda.device(device):
            positions = self._positions_cpu.to(device)
            columns = torch.arange(self.full_length, device=device)
            packed = torch.empty(
                (self.num_queries * self.full_length + 7) // 8,
                device=device,
                dtype=torch.uint8,
            )
            # Non-final chunks contain a multiple of 8 rows. Their flattened
            # bit counts are consequently byte-aligned even when N % 8 != 0.
            # This avoids a full M*N temporary without padding between rows.
            chunk_rows = max(8, (_MASK_CHUNK_BYTES // self.full_length // 8) * 8)
            for start in range(0, self.num_queries, chunk_rows):
                stop = min(start + chunk_rows, self.num_queries)
                row_positions = positions[start:stop, None]
                visible = columns[None, :] <= row_positions
                if resolved_window is not None:
                    visible &= columns[None, :] > row_positions - resolved_window
                chunk = packbits(visible.flatten(), bitorder="little")
                byte_start = start * self.full_length // 8
                packed[byte_start : byte_start + chunk.numel()].copy_(chunk)
            ready = torch.cuda.Event()
            ready.record(torch.cuda.current_stream(device))
        self._masks[cache_key] = _PackedMask(packed, ready)

    def run(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        scale: float | None = None,
        window: int | None = None,
    ) -> torch.Tensor:
        """Return [M,Hq,Dv] attention over complete [N,Hkv,D] merged K/V.

        BF16 and FP16 are supported, including GQA. Non-contiguous inputs are
        materialized when needed. Inputs are never modified. Sliding windows
        are included in the original-position custom mask, not FlashInfer's
        suffix-aligned built-in window. There is no silent fallback kernel.
        """
        self._validate_inputs(q, k, v, scale)
        self.prepare(q.device, window)
        entry = self._masks[(q.device, self._window(window))]
        from flashinfer import single_prefill_with_kv_cache

        with torch.cuda.device(q.device):
            stream = torch.cuda.current_stream(q.device)
            stream.wait_event(entry.ready)
            entry.tensor.record_stream(stream)
            return single_prefill_with_kv_cache(
                q.contiguous(),
                k.contiguous(),
                v.contiguous(),
                packed_custom_mask=entry.tensor,
                causal=False,
                kv_layout="NHD",
                pos_encoding_mode="NONE",
                sm_scale=None if scale is None else float(scale),
                window_left=-1,
                backend="fa2",
                return_lse=False,
            )

    def _validate_inputs(self, q, k, v, scale) -> None:
        if any(not isinstance(x, torch.Tensor) for x in (q, k, v)):
            raise TypeError("q, k, and v must be torch.Tensor instances")
        if any(x.ndim != 3 for x in (q, k, v)):
            raise ValueError("q, k, and v must have shape [tokens, heads, head_dim]")
        if q.shape[0] != self.num_queries:
            raise ValueError("Q row count must equal the number of selected positions")
        if k.shape[0] != self.full_length or v.shape[0] != self.full_length:
            raise ValueError("K/V must cover the entire original prompt")
        if min(q.shape[1:]) <= 0 or min(k.shape[1:]) <= 0 or min(v.shape[1:]) <= 0:
            raise ValueError("head counts and dimensions must be positive")
        if k.shape[1] != v.shape[1] or q.shape[2] != k.shape[2]:
            raise ValueError("K/V head counts and Q/K head dimensions must match")
        if q.shape[1] % k.shape[1]:
            raise ValueError("Q head count must be a multiple of KV head count")
        if q.dtype not in (torch.float16, torch.bfloat16):
            raise TypeError("sparse prefill currently supports only FP16/BF16")
        if any(x.dtype != q.dtype or x.device != q.device for x in (k, v)):
            raise ValueError("Q/K/V must share the same dtype and device")
        if q.device.type != "cuda":
            raise ValueError("FlashInfer sparse prefill requires CUDA Q/K/V")
        if scale is not None:
            if isinstance(scale, bool) or not isinstance(scale, Real):
                raise TypeError("scale must be a finite positive real number")
            if not math.isfinite(scale) or scale <= 0:
                raise ValueError("scale must be a finite positive real number")
