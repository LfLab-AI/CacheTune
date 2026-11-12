"""Offline, chunk-local frequency scoring from CacheTune Section 4.1.

Each independently encoded chunk is transformed along its token axis. Keys
are supplied before RoPE (after any model-specific K normalization). No query,
answer, or adjacent chunk participates in a cached chunk's frequency score.
"""
from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Sequence

import torch


from .spectral_algorithms import (
    lowpass_reconstruct, spectral_statistics, spectral_token_scores,
)

SCORE_EPSILON = 1e-12


def _heads(values):
    if not isinstance(values, torch.Tensor) or values.ndim < 2:
        raise ValueError("KV must have token and feature axes")
    return values.reshape(values.shape[0], 1, -1) if values.ndim == 2 else values


def lowpass_reconstruction(values, alpha=0.5):
    shape = values.shape
    return lowpass_reconstruct(_heads(values), alpha).reshape(shape)


def layer_token_scores(key, value, alpha=0.5):
    stats = spectral_statistics([_heads(key)], [_heads(value)], alpha)
    return 0.5 * (stats["k_sq"][0].sqrt() + stats["v_sq"][0].sqrt())


def chunk_scores(layer_pairs, alpha=0.5, epsilon=SCORE_EPSILON):
    pairs = list(layer_pairs)
    return spectral_token_scores([_heads(k) for k, _ in pairs],
                                 [_heads(v) for _, v in pairs], alpha, epsilon)


def context_chunk_spans(chunk_lengths: Sequence[int], context_len: int,
                        ratio: float) -> list[tuple[int, int, int]]:
    """Map independent chunk boundaries to context, excluding the suffix.

    The paper leaves fractional token rounding unspecified. This adapter uses
    ``floor(r * chunk_context_tokens)`` independently per cached chunk, so the
    offline index list for a chunk does not depend on its request neighbors.
    """
    if (isinstance(context_len, bool) or not isinstance(context_len, Integral)
            or context_len < 0):
        raise ValueError("context_len must be a nonnegative integer")
    if (isinstance(ratio, bool) or not isinstance(ratio, Real)
            or not math.isfinite(ratio) or not 0 <= ratio <= 1):
        raise ValueError("recompute ratio must lie in [0, 1]")
    start, spans = 0, []
    for length in chunk_lengths:
        if isinstance(length, bool) or not isinstance(length, Integral) or length <= 0:
            raise ValueError("Chunk lengths must be positive integers")
        stop = start + int(length)
        end = min(stop, int(context_len))
        if start < end:
            spans.append((start, end, int((end - start) * float(ratio))))
        start = stop
    if start < context_len:
        raise ValueError("Independent chunks do not cover the context")
    return spans


def select_context_indices(layer_pairs: Sequence[tuple[torch.Tensor, torch.Tensor]],
                           chunk_lengths: Sequence[int], context_len: int,
                           ratio: float, alpha: float = 0.5,
                           sink_size: int = 0) -> tuple[torch.Tensor, list[dict]]:
    """Select each cached chunk independently, returning global CPU indices.

    A nonzero ``sink_size`` is an explicit compatibility option, not part of
    the paper's default scorer. Forced sink rows consume their chunk's budget;
    all rows still participate in that chunk's low-pass reconstruction.
    """
    spans = context_chunk_spans(chunk_lengths, context_len, ratio)
    if (isinstance(sink_size, bool) or not isinstance(sink_size, Integral)
            or not 0 <= sink_size <= context_len):
        raise ValueError("sink_size must lie within the context")
    if not layer_pairs:
        raise ValueError("At least one KV layer is required")
    chosen, details = [], []
    for start, end, budget in spans:
        sinks = max(0, min(end, int(sink_size)) - start)
        if sinks > budget:
            raise ValueError("Forced sink positions exceed their independent chunk budget")
        scores = chunk_scores([(key[start:end], value[start:end])
                               for key, value in layer_pairs], alpha)
        local = torch.arange(sinks, dtype=torch.long, device=scores.device)
        if budget > sinks:
            # Stable ties make the offline plan reproducible on equal-score
            # chunks; otherwise this is the same Top-k ordering as the paper.
            ranked = torch.argsort(scores[sinks:], descending=True, stable=True)
            local = torch.cat((local, ranked[:budget - sinks] + sinks))
        chosen.append((local + start).to(device="cpu", dtype=torch.long))
        details.append({"start": start, "end": end, "tokens": end - start,
                        "budget": budget, "forced_sink_tokens": sinks})
    indices = torch.cat(chosen).sort().values if chosen else torch.empty(0, dtype=torch.long)
    return indices, details
