"""Pure numerical helpers for the optional manuscript-compatible example path.

The legacy examples do not import this module unless the paper path is selected.
Frequency filtering follows Section 4.1 / Appendix A; calibration follows
Algorithm 1 in the supplied CacheTune ICLR 2027 manuscript. No vLLM is required.
"""

from dataclasses import dataclass
import math
from numbers import Real
from typing import Callable, Dict, Iterable, Mapping, Sequence, Tuple

import torch


def _finite_number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("{} must be a finite real number".format(name))
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("{} must be a finite real number".format(name))
    return value


def _real_tensor(value, name: str, ndim: int) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise ValueError("{} must be a torch.Tensor".format(name))
    if value.ndim != ndim or not value.is_floating_point():
        raise ValueError("{} must be a {}-D real floating tensor".format(name, ndim))
    if not bool(torch.isfinite(value).all()):
        raise ValueError("{} contains non-finite values".format(name))
    return value


@torch.no_grad()
def lowpass_reconstruct(x: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
    """Low-pass one cached chunk, with token axis 0 and shape [N, H, D].

    Retain bins ``k < floor(alpha * (floor(N / 2) + 1))`` from the one-sided
    real FFT, and reconstruct with the explicit original length N. Chunks must
    be passed separately: a transform across concatenated chunks is different.
    Half/bfloat16 inputs are promoted to float32 for the transform; float64 is
    preserved. The cutoff floor, including a possible zero cutoff, is literal.
    """
    alpha = _finite_number(alpha, "alpha")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    x = _real_tensor(x, "x", 3)
    if any(size == 0 for size in x.shape):
        raise ValueError("x must have nonempty token, head and feature axes")
    work = x if x.dtype in (torch.float32, torch.float64) else x.float()
    coefficients = torch.fft.rfft(work, dim=0)
    cutoff = math.floor(alpha * coefficients.shape[0])
    coefficients[cutoff:] = 0
    return torch.fft.irfft(coefficients, n=x.shape[0], dim=0)


@torch.no_grad()
def spectral_statistics(
    key_layers: Sequence[torch.Tensor],
    value_layers: Sequence[torch.Tensor],
    alpha: float = 0.5,
) -> Dict[str, object]:
    """Compute one chunk's per-layer squared K/V norms before TP reduction.

    Each layer is [N, H, D]; [L, N, H, D] tensors are accepted as sequences.
    Results are CPU [L, N] tensors. Squared statistics, rather than head-local
    norms, permit correct combination of tensor-parallel KV head partitions.
    """
    keys, values = list(key_layers), list(value_layers)
    if not keys or len(keys) != len(values):
        raise ValueError("K and V must have the same nonzero number of layers")
    key_sq, value_sq = [], []
    tokens = None
    for layer, (key, value) in enumerate(zip(keys, values)):
        key = _real_tensor(key, "K[{}]".format(layer), 3)
        value = _real_tensor(value, "V[{}]".format(layer), 3)
        if key.shape != value.shape:
            raise ValueError("paired K and V layer shapes must match")
        if tokens is None:
            tokens = key.shape[0]
        elif tokens != key.shape[0]:
            raise ValueError("every layer must contain the same chunk tokens")
        key_low = lowpass_reconstruct(key, alpha)
        value_low = lowpass_reconstruct(value, alpha)
        # Float64 accumulation avoids overflow from squaring large fp16/fp32 KVs.
        key_sq.append(key_low.double().square().sum(dim=(1, 2)).cpu())
        value_sq.append(value_low.double().square().sum(dim=(1, 2)).cpu())
    return {
        "k_sq": torch.stack(key_sq),
        "v_sq": torch.stack(value_sq),
        "replication_factor": 1,
    }


@torch.no_grad()
def aggregate_spectral_statistics(
    rank_stats: Iterable[Mapping[str, object]], epsilon: float = 1e-12
) -> torch.Tensor:
    """Reduce TP squared norms, normalize per layer, then average layers.

    ``replication_factor`` is the number of copies of each logical KV head
    across all ranks (1 for ordinary head sharding), either a scalar or one
    integer per layer. It must be identical on each rank. Head squares are
    summed and divided by this factor *before*
    taking the K and V square roots. The score is their arithmetic mean, then
    normalized by that layer's token sum plus epsilon. Returns CPU [N].

    Epsilon=1e-12 is an explicit numerical engineering choice, not a measured
    parameter of the manuscript.
    """
    epsilon = _finite_number(epsilon, "epsilon")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    stats = list(rank_stats)
    if not stats:
        raise ValueError("rank_stats must contain at least one rank")
    summed_k = summed_v = None
    replication = None
    for rank, stat in enumerate(stats):
        if not isinstance(stat, Mapping) or "k_sq" not in stat or "v_sq" not in stat:
            raise ValueError("each rank must provide k_sq and v_sq")
        k_sq = _real_tensor(stat["k_sq"], "rank {} k_sq".format(rank), 2).double().cpu()
        v_sq = _real_tensor(stat["v_sq"], "rank {} v_sq".format(rank), 2).double().cpu()
        if k_sq.shape != v_sq.shape or any(size == 0 for size in k_sq.shape):
            raise ValueError("rank statistics must have matching nonempty [L, N] shapes")
        if bool((k_sq < 0).any()) or bool((v_sq < 0).any()):
            raise ValueError("squared norms must be nonnegative")
        factor = stat.get("replication_factor", 1)
        if isinstance(factor, (list, tuple)):
            if len(factor) != k_sq.shape[0]:
                raise ValueError("replication_factor must have one entry per layer")
            copies = tuple(_finite_number(item, "replication_factor") for item in factor)
        else:
            copies = (_finite_number(factor, "replication_factor"),) * k_sq.shape[0]
        if any(item < 1 or not item.is_integer() for item in copies):
            raise ValueError("replication_factor must contain positive integers")
        if replication is None:
            replication = copies
        elif copies != replication:
            raise ValueError("all ranks must report the same replication_factor")
        if summed_k is None:
            summed_k, summed_v = k_sq.clone(), v_sq.clone()
        else:
            if k_sq.shape != summed_k.shape:
                raise ValueError("all ranks must contain the same layers and tokens")
            summed_k.add_(k_sq)
            summed_v.add_(v_sq)
    replication_tensor = torch.tensor(replication, dtype=torch.float64)[:, None]
    layer_scores = 0.5 * (
        torch.sqrt(summed_k / replication_tensor) + torch.sqrt(summed_v / replication_tensor)
    )
    if not bool(torch.isfinite(layer_scores).all()):
        raise ValueError("aggregated spectral norms overflowed")
    normalizers = layer_scores.sum(dim=1, keepdim=True) + epsilon
    if not bool(torch.isfinite(normalizers).all()):
        raise ValueError("aggregated spectral score normalization overflowed")
    return (layer_scores / normalizers).mean(dim=0)


def spectral_token_scores(
    key_layers: Sequence[torch.Tensor],
    value_layers: Sequence[torch.Tensor],
    alpha: float = 0.5,
    epsilon: float = 1e-12,
) -> torch.Tensor:
    """Convenience function for one chunk with all logical KV heads present."""
    return aggregate_spectral_statistics(
        [spectral_statistics(key_layers, value_layers, alpha)], epsilon
    )


@torch.no_grad()
def select_shared_tokens(
    chunk_scores: Sequence[torch.Tensor], ratio: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Select each chunk's top floor(ratio*N) tokens and its reuse complement.

    Returns disjoint, globally offset, position-sorted CPU int64 tensors shared
    by all layers. Ties prefer the smaller original token index. Flooring the
    fractional token budget is an explicit engineering choice. A ratio of 0
    selects none, while 1 selects all; empty chunks are allowed.
    """
    ratio = _finite_number(ratio, "ratio")
    if not 0.0 <= ratio <= 1.0:
        raise ValueError("ratio must be in [0, 1]")
    selected, reused = [], []
    offset = 0
    for chunk, scores in enumerate(chunk_scores):
        scores = _real_tensor(scores, "chunk_scores[{}]".format(chunk), 1).cpu()
        if bool((scores < 0).any()):
            raise ValueError("spectral token scores must be nonnegative")
        size = scores.numel()
        count = math.floor(ratio * size)
        order = torch.argsort(scores, descending=True, stable=True)
        mask = torch.zeros(size, dtype=torch.bool)
        mask[order[:count]] = True
        positions = torch.arange(size, dtype=torch.long) + offset
        selected.append(positions[mask])
        reused.append(positions[~mask])
        offset += size
    empty = torch.empty(0, dtype=torch.long)
    return (
        torch.cat(selected) if selected else empty.clone(),
        torch.cat(reused) if reused else empty.clone(),
    )


@dataclass(frozen=True)
class GSSTrace:
    """An evaluated pair within a search bracket, before its next update."""

    phase: str
    a: float
    b: float
    x1: float
    x2: float
    f1: float
    f2: float


@dataclass(frozen=True)
class GSSResult:
    ratio: float
    value: float
    prior: float
    interval: Tuple[float, float]
    evaluations: Tuple[Tuple[float, float], ...]
    iterations: int
    trace: Tuple[GSSTrace, ...]

    @property
    def cache(self) -> Dict[float, float]:
        """Actual measured ratios and their objective values (no interpolation)."""
        return dict(self.evaluations)


def golden_section_search(
    objective: Callable[[float], float],
    tc: float,
    ti: float,
    r_max: float = 1.0,
    tolerance: float = 0.01,
    r_min: float = 0.15,
) -> GSSResult:
    """Warm-started GSS exactly matching manuscript Algorithm 1.

    ``objective`` must measure mean complete-request TTFT on the same fixed
    calibration set and offline rankings at every ratio. tc and ti are positive
    per-unit recomputation/transfer costs. They determine the clipped roofline
    prior; the measured objective determines the search. r_max=1 and tolerance
    0.01 are configurable engineering defaults. The generic helper accepts
    bounds in [0, 1]; the paper experiment uses r_min=0.15.

    As a reporting extension, the returned interval midpoint is explicitly
    evaluated, so ``value`` always belongs to the returned deployment ratio.
    Identical floating-point ratios reuse their already measured value.
    """
    if not callable(objective):
        raise ValueError("objective must be callable")
    tc, ti = _finite_number(tc, "tc"), _finite_number(ti, "ti")
    a, b = _finite_number(r_min, "r_min"), _finite_number(r_max, "r_max")
    tolerance = _finite_number(tolerance, "tolerance")
    if tc <= 0 or ti <= 0:
        raise ValueError("tc and ti must be positive")
    if not 0.0 <= a <= b <= 1.0:
        raise ValueError("bounds must satisfy 0 <= r_min <= r_max <= 1")
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    if a < b and tolerance <= math.ulp(b):
        raise ValueError("tolerance is below floating-point resolution at r_max")
    # Equivalent to ti/(tc+ti), without overflowing the sum of finite costs.
    scale = max(tc, ti)
    prior = min(b, max(a, (ti / scale) / (tc / scale + ti / scale)))
    cache: Dict[float, float] = {}
    evaluations = []
    trace = []

    def evaluate(ratio: float) -> float:
        if ratio not in cache:
            value = _finite_number(objective(ratio), "objective result")
            cache[ratio] = value
            evaluations.append((ratio, value))
        return cache[ratio]

    if a == b:
        value = evaluate(a)
        return GSSResult(a, value, prior, (a, b), tuple(evaluations), 0, ())

    phi = (math.sqrt(5.0) - 1.0) / 2.0
    if prior <= (a + b) / 2.0:
        x1, x2 = prior, a + phi * (b - a)
    else:
        x1, x2 = b - phi * (b - a), prior
    f1, f2 = evaluate(x1), evaluate(x2)
    trace.append(GSSTrace("warm", a, b, x1, x2, f1, f2))
    if f1 <= f2:  # Algorithm 1, line 5 uses a non-strict comparison.
        b = x2
    else:
        a = x1

    # The warm pair is not a standard golden pair. Reinitialize BOTH probes.
    x1, x2 = b - phi * (b - a), a + phi * (b - a)
    f1, f2 = evaluate(x1), evaluate(x2)
    trace.append(GSSTrace("reinitialize", a, b, x1, x2, f1, f2))
    iterations = 0
    while b - a >= tolerance:
        if f1 < f2:  # Algorithm 1, line 9 uses a strict comparison.
            b, x2, f2 = x2, x1, f1
            x1 = b - phi * (b - a)
            f1 = evaluate(x1)
        else:
            a, x1, f1 = x1, x2, f2
            x2 = a + phi * (b - a)
            f2 = evaluate(x2)
        iterations += 1
        trace.append(GSSTrace("refine", a, b, x1, x2, f1, f2))

    ratio = (a + b) / 2.0
    value = evaluate(ratio)
    return GSSResult(
        ratio, value, prior, (a, b), tuple(evaluations), iterations, tuple(trace)
    )
