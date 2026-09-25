"""Torch-only causal-bias helper for the opt-in CacheTune spectral path."""

import torch


@torch.no_grad()
def make_spectral_causal_bias(
    query: torch.Tensor,
    query_positions: torch.Tensor,
    num_key_tokens: int,
) -> torch.Tensor:
    """Make an xFormers tensor bias for selected absolute query positions.

    ``query`` is BMHD or BMGHD, with B=1. Keys occupy their complete original
    positions 0..N-1. A query at position p may attend only to keys k <= p,
    regardless of the number or spacing of other selected queries. The returned
    bias is [1,H,M,N] or [1,G,H,M,N], with 0 for visible and -inf for future
    keys. Heads share storage; the row stride is padded to a multiple of eight
    elements for xFormers tensor-bias kernels, including when N is unaligned.
    """
    if not isinstance(query, torch.Tensor) or query.ndim not in (4, 5):
        raise ValueError("query must have xFormers BMHD or BMGHD shape")
    if query.shape[0] != 1 or not query.is_floating_point():
        raise ValueError("spectral selected-query attention requires one floating query batch")
    if any(size <= 0 for size in query.shape):
        raise ValueError("query axes must be nonempty")
    if (isinstance(num_key_tokens, bool) or not isinstance(num_key_tokens, int)
            or num_key_tokens <= 0):
        raise ValueError("num_key_tokens must be a positive integer")
    if (not isinstance(query_positions, torch.Tensor)
            or query_positions.ndim != 1
            or query_positions.dtype not in (torch.int32, torch.int64)):
        raise ValueError("query_positions must be a one-dimensional integer tensor")
    if query_positions.numel() != query.shape[1]:
        raise ValueError("one absolute position is required for every selected query")
    positions = query_positions.to(device=query.device, dtype=torch.long)
    if bool(((positions < 0) | (positions >= num_key_tokens)).any()):
        raise ValueError("selected query positions must lie within the complete KV sequence")

    padded_keys = (num_key_tokens + 7) // 8 * 8
    storage = torch.zeros(
        (query.shape[1], padded_keys), device=query.device, dtype=query.dtype
    )
    rows = storage[:, :num_key_tokens]
    key_positions = torch.arange(num_key_tokens, device=query.device)
    rows.masked_fill_(key_positions[None, :] > positions[:, None], float("-inf"))
    head_shape = tuple(query.shape[2:-1])
    view_shape = (1,) + (1,) * len(head_shape) + tuple(rows.shape)
    target_shape = (1,) + head_shape + tuple(rows.shape)
    return rows.view(view_shape).expand(target_shape)
