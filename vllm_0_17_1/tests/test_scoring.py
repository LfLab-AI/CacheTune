"""CPU tests for the published chunk-local scoring/selection contract."""
import math

import pytest
import torch

from cachetune.scoring import (
    chunk_scores, context_chunk_spans, layer_token_scores,
    lowpass_reconstruction, select_context_indices,
)


def _pairs(n, layers=3, seed=8, dtype=torch.float64):
    rng = torch.Generator().manual_seed(seed)
    return [(torch.randn(n, 2, 4, generator=rng, dtype=dtype),
             torch.randn(n, 2, 4, generator=rng, dtype=dtype))
            for _ in range(layers)]


def _full_fft_projection(x, alpha):
    """Independent complex-DFT oracle with conjugate frequency pairs."""
    n = len(x)
    cut = math.floor(alpha * (n // 2 + 1))
    mask = torch.zeros(n, dtype=torch.bool)
    if cut:
        mask[:cut] = True
        if cut > 1:
            mask[-(cut-1):] = True
    return torch.fft.ifft(torch.fft.fft(x, dim=0) * mask[:, None, None], dim=0).real


@pytest.mark.parametrize("n", [1, 2, 7, 8, 17])
@pytest.mark.parametrize("alpha", [0, .25, .5, 1])
def test_real_fft_matches_complete_frequency_projection(n, alpha):
    key = _pairs(n)[0][0]
    result = lowpass_reconstruction(key, alpha)
    assert result.shape == key.shape
    assert result.dtype == torch.float64
    torch.testing.assert_close(result, _full_fft_projection(key, alpha), atol=1e-12, rtol=1e-12)


def test_k_and_v_norms_are_separate_then_averaged():
    key = torch.tensor([[[3., 4.]], [[0., 0.]]], dtype=torch.float64)
    value = torch.tensor([[[0., 0.]], [[0., 12.]]], dtype=torch.float64)
    actual = layer_token_scores(key, value, alpha=1)
    torch.testing.assert_close(actual, torch.tensor([2.5, 6.], dtype=torch.float64))
    # One concatenated norm would yield 5 and 12, not the implemented K/V mean.
    assert not torch.equal(actual, torch.tensor([5., 12.], dtype=torch.float64))


def test_normalization_is_per_layer_before_layer_mean():
    pairs = _pairs(9)
    expected = torch.zeros(9, dtype=torch.float64)
    for key, value in pairs:
        k = lowpass_reconstruction(key, .5)
        v = lowpass_reconstruction(value, .5)
        raw = .5 * (k.double().square().sum((1, 2)).sqrt()
                     + v.double().square().sum((1, 2)).sqrt())
        expected += raw / (raw.sum() + 1e-12) / len(pairs)
    torch.testing.assert_close(chunk_scores(pairs), expected, atol=1e-15, rtol=1e-15)
    scaled = [(k * (1000 if i == 1 else 1), v * (1000 if i == 1 else 1))
              for i, (k, v) in enumerate(pairs)]
    torch.testing.assert_close(chunk_scores(pairs), chunk_scores(scaled), atol=1e-13, rtol=1e-13)


def test_every_kv_layer_participates_in_ranking():
    zeros = torch.zeros(6, 1, 2, dtype=torch.float64)
    early = zeros.clone(); early[0, 0, 0] = 1
    late = zeros.clone(); late[5, 0, 1] = 10
    score = chunk_scores([(early, zeros), (late, zeros)], alpha=1)
    assert score[0] > 0 and score[5] > 0
    early_scores = chunk_scores([(early, zeros)], alpha=1)
    assert early_scores[0] > 1e10 * early_scores[1:].abs().max()


def test_chunk_local_selection_is_invariant_to_neighbor_changes():
    first, second = _pairs(9, seed=1), _pairs(8, seed=4)
    joined = [(torch.cat((a[0], b[0])), torch.cat((a[1], b[1])))
              for a, b in zip(first, second)]
    selected, meta = select_context_indices(joined, [9, 8], 17, .5)
    separate, _ = select_context_indices(first, [9], 9, .5)
    assert selected[selected < 9].tolist() == separate.tolist()
    changed = [(torch.cat((a[0], b[0] * 100 + 777)),
                torch.cat((a[1], b[1] * -80 - 200)))
               for a, b in zip(first, second)]
    changed_indices, _ = select_context_indices(changed, [9, 8], 17, .5)
    assert changed_indices[changed_indices < 9].tolist() == separate.tolist()
    assert [row["budget"] for row in meta] == [4, 4]


def test_context_only_budget_excludes_suffix_and_floors_each_chunk():
    pairs = _pairs(20)
    selected, meta = select_context_indices(pairs, [5, 5, 10], 10, .15)
    assert selected.numel() == 0  # floor(.15*5) + floor(.15*5), not floor(.15*10).
    assert [(x["start"], x["end"]) for x in meta] == [(0, 5), (5, 10)]
    assert context_chunk_spans([5, 15], 9, .5) == [(0, 5, 2), (5, 9, 2)]
    changed = [(k.clone(), v.clone()) for k, v in pairs]
    for k, v in changed:
        k[10:] = 1e12; v[10:] = -1e12
    before, _ = select_context_indices(pairs, [5, 5, 10], 10, .6)
    after, _ = select_context_indices(changed, [5, 5, 10], 10, .6)
    assert before.tolist() == after.tolist()


def test_zero_cutoff_and_ties_prefer_smaller_original_indices():
    pairs = _pairs(8)
    chosen, _ = select_context_indices(pairs, [4, 4], 8, .5, alpha=0)
    assert chosen.tolist() == [0, 1, 4, 5]
    # N=1, alpha=.5 has cutoff=floor(.5*1)=0, not an implicit DC keep.
    assert torch.count_nonzero(lowpass_reconstruction(torch.ones(1, 1, 2), .5)) == 0


def test_large_scores_use_float64_norm_accumulation_and_do_not_mutate_sources():
    key = torch.full((8, 1, 2), 1e20, dtype=torch.float32)
    value = key.clone()
    snapshots = (key.clone(), value.clone())
    score = chunk_scores([(key, value)], alpha=1)
    assert torch.isfinite(score).all()
    torch.testing.assert_close(score, torch.full((8,), .125, dtype=torch.float64))
    assert torch.equal(key, snapshots[0]) and torch.equal(value, snapshots[1])


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_cache_values_are_rejected(bad):
    key, value = _pairs(7)[0]
    key[2, 0, 1] = bad
    with pytest.raises(ValueError, match="non-finite"):
        chunk_scores([(key, value)])


def test_recompute_none_and_all_are_exact_context_partitions():
    pairs = _pairs(17)
    assert select_context_indices(pairs, [9, 8], 17, 0)[0].numel() == 0
    assert select_context_indices(pairs, [9, 8], 17, 1)[0].tolist() == list(range(17))
