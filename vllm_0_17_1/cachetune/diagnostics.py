"""Debug-only attention comparisons against an independent FP32 reference.

``native`` and ``flashinfer`` are already computed outputs, not callables.
All tensors use NHD layout; Q contains every original prompt position. This
module synchronizes for diagnostic scalars and is not a performance path.
"""
from __future__ import annotations

import math
from contextlib import contextmanager
from numbers import Integral, Real

import torch


_METRIC_ROWS = 256


def _json_number(value: float) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def _finite_summary(value: torch.Tensor) -> dict:
    finite = nan = positive_inf = negative_inf = 0
    for chunk in value.split(_METRIC_ROWS, dim=0):
        finite += int(torch.isfinite(chunk).sum().item())
        nan += int(torch.isnan(chunk).sum().item())
        positive_inf += int(torch.isposinf(chunk).sum().item())
        negative_inf += int(torch.isneginf(chunk).sum().item())
    return {
        "all_finite": finite == value.numel(),
        "finite_count": finite,
        "element_count": value.numel(),
        "nan_count": nan,
        "positive_inf_count": positive_inf,
        "negative_inf_count": negative_inf,
    }


def _metrics(actual: torch.Tensor, reference: torch.Tensor) -> dict:
    """FP32 differences, FP64 scalar reductions, bounded temporary storage."""
    actual_finite = _finite_summary(actual)
    reference_finite = _finite_summary(reference)
    result = {"actual": actual_finite, "reference": reference_finite}
    names = ("relative_l2", "absolute_l2", "reference_l2", "actual_l2",
             "max_absolute_error", "mean_absolute_error", "rmse")
    if not actual_finite["all_finite"] or not reference_finite["all_finite"]:
        result.update({name: None for name in names})
        return result
    error_sq = reference_sq = actual_sq = absolute_sum = maximum = 0.0
    for start in range(0, actual.shape[0], _METRIC_ROWS):
        a = actual[start:start + _METRIC_ROWS].float()
        b = reference[start:start + _METRIC_ROWS].float()
        difference = a - b
        # Convert before squaring to avoid FP32 overflow in the diagnostics.
        error_sq += float(difference.double().square().sum().item())
        reference_sq += float(b.double().square().sum().item())
        actual_sq += float(a.double().square().sum().item())
        absolute = difference.abs()
        absolute_sum += float(absolute.sum(dtype=torch.float64).item())
        maximum = max(maximum, float(absolute.max().item()))
    error_l2, reference_l2 = math.sqrt(error_sq), math.sqrt(reference_sq)
    values = {
        "relative_l2": error_l2 / max(reference_l2, 1e-12),
        "absolute_l2": error_l2,
        "reference_l2": reference_l2,
        "actual_l2": math.sqrt(actual_sq),
        "max_absolute_error": maximum,
        "mean_absolute_error": absolute_sum / actual.numel(),
        "rmse": math.sqrt(error_sq / actual.numel()),
    }
    result.update({name: _json_number(value) for name, value in values.items()})
    return result


@contextmanager
def _fp32_matmul_without_tf32():
    # Both settings are process-global. The caller must serialize diagnostics;
    # restore even if reference construction raises (e.g. CUDA OOM).
    previous_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    previous_precision = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_allow_tf32
        # Preserve "medium" as well as "high"/"highest", rather than only its
        # allow_tf32 boolean projection. No cuDNN operations are used here.
        torch.set_float32_matmul_precision(previous_precision)


def _dense_reference(q, k, v, positions, scale, window):
    """Selected original rows; each KV head expands to its contiguous Q group."""
    selected_q = q.index_select(0, positions).float()
    num_queries, num_heads, _ = selected_q.shape
    groups = num_heads // k.shape[1]
    columns = torch.arange(k.shape[0], device=q.device)
    visible = columns[None, :] <= positions[:, None]
    if window is not None:
        visible &= columns[None, :] > positions[:, None] - window
    reference = torch.empty((num_queries, num_heads, v.shape[2]),
                            dtype=torch.float32, device=q.device)
    with _fp32_matmul_without_tf32():
        for kv_head in range(k.shape[1]):
            start, stop = kv_head * groups, (kv_head + 1) * groups
            queries = selected_q[:, start:stop, :].transpose(0, 1)
            # A stride-zero expansion implements repeat_interleave's head
            # mapping without materializing full N*Hq*D FP32 K/V tensors.
            keys = k[:, kv_head, :].float().unsqueeze(0).expand(groups, -1, -1)
            values = v[:, kv_head, :].float().unsqueeze(0).expand(groups, -1, -1)
            scores = torch.bmm(queries, keys.transpose(1, 2)) * scale
            scores.masked_fill_(~visible.unsqueeze(0), float("-inf"))
            probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32)
            reference[:, start:stop, :] = torch.bmm(probabilities, values).transpose(0, 1)
    return reference


@torch.no_grad()
def compare_attention(q, k, v, native, flashinfer, scale, window=None) -> dict:
    """Compare two full-row outputs and eight fixed rows against FP32 math.

    Q: [N,Hq,D], K: [N,Hkv,D], V: [N,Hkv,Dv]; outputs: [N,Hq,Dv].
    The same post-RoPE Q/K and V must have produced both supplied outputs.
    ``window`` counts visible positions including self; None/-1 is unbounded.
    Returned dictionaries are serializable with ``json.dumps(..., allow_nan=False)``.
    Relative L2 uses the named reference's norm, clamped below at 1e-12.
    """
    tensors = {"q": q, "k": k, "v": v,
               "native": native, "flashinfer": flashinfer}
    for name, value in tensors.items():
        if not isinstance(value, torch.Tensor) or value.ndim != 3:
            raise ValueError(f"{name} must be a three-dimensional Tensor")
        if not value.is_floating_point() or min(value.shape) <= 0:
            raise ValueError(f"{name} must be nonempty and floating point")
    if any(value.device != q.device for value in tensors.values()):
        raise ValueError("All input and output tensors must share one device")
    n, heads, dim = q.shape
    if (k.shape[0] != n or v.shape[0] != n or k.shape[1] != v.shape[1]
            or k.shape[2] != dim or heads % k.shape[1]):
        raise ValueError("Q/K/V dimensions must describe full-row MHA or GQA")
    expected = (n, heads, v.shape[2])
    if tuple(native.shape) != expected or tuple(flashinfer.shape) != expected:
        raise ValueError(f"Both output shapes must be {expected}")
    if isinstance(scale, bool) or not isinstance(scale, Real) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be a finite positive number")
    if window is not None:
        if isinstance(window, bool) or not isinstance(window, Integral):
            raise ValueError("window must be a positive integer, -1, or None")
        if window == -1:
            window = None
        elif window <= 0:
            raise ValueError("window must count at least the current position")
        else:
            window = int(window)
    count = min(8, n)
    selected = ([0] if count == 1 else
                [i * (n - 1) // (count - 1) for i in range(count)])
    positions = torch.tensor(selected, device=q.device, dtype=torch.long)
    reference = _dense_reference(q, k, v, positions, float(scale), window)
    selected_native = native.index_select(0, positions)
    selected_flashinfer = flashinfer.index_select(0, positions)
    return {
        "schema_version": 1,
        "scale": float(scale),
        "window_including_self": window,
        "gqa_group_size": heads // k.shape[1],
        "query_positions": selected,
        "tensors": {name: {"shape": list(value.shape), "dtype": str(value.dtype),
                           **_finite_summary(value)} for name, value in tensors.items()},
        "flashinfer_vs_native": _metrics(flashinfer, native),
        "selected_reference": {
            "dtype": "torch.float32",
            "tf32_enabled": False,
            "causal_by_original_position": True,
            **_finite_summary(reference),
            "native_vs_reference": _metrics(selected_native, reference),
            "flashinfer_vs_reference": _metrics(selected_flashinfer, reference),
            "per_query": [
                {"position": position,
                 "native_vs_reference": _metrics(selected_native[i:i + 1], reference[i:i + 1]),
                 "flashinfer_vs_reference": _metrics(selected_flashinfer[i:i + 1], reference[i:i + 1])}
                for i, position in enumerate(selected)
            ],
        },
    }
