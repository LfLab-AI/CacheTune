"""CUDA state/transfer tests, independent of a vLLM worker or FlashInfer.

The rotary probe deliberately writes Keys in place, exercising ownership and
stream ordering. Attention/model correctness is covered by separate tests.
"""
from __future__ import annotations

import pytest
import torch

from cachetune.state import CacheTuneState
from cachetune.scoring import select_context_indices


@pytest.fixture
def gpu():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for state/transfer tests")
    return torch.device("cuda", torch.cuda.current_device())


def _rotate(key, positions):
    shape = key.shape
    heads = key.reshape(key.shape[0], 2, 4).float()
    frequency = torch.tensor([1.0, 0.01], device=key.device)
    angle = positions.float()[:, None, None] * frequency[None, None, :]
    left, right = heads[..., :2], heads[..., 2:]
    return torch.cat((left * angle.cos() - right * angle.sin(),
                      right * angle.cos() + left * angle.sin()), dim=-1).to(key.dtype).reshape(shape)


class RotaryProbe:
    q_size, num_heads, head_dim = 16, 4, 4

    def rotary_emb(self, positions, query, key):
        key.copy_(_rotate(key, positions))
        return query, key


def _collected(device, layers=4, n=17):
    generator = torch.Generator(device=device).manual_seed(2468)
    raw = [(torch.randn(n, 8, dtype=torch.bfloat16, device=device, generator=generator),
            torch.randn(n, 8, dtype=torch.bfloat16, device=device, generator=generator))
           for _ in range(layers)]
    ids = list(range(101, 101 + n))
    state = CacheTuneState(layers)
    state.start_collect()
    assert not state.note_input(torch.tensor(ids, device=device), torch.arange(n, device=device))
    for layer, (key, value) in enumerate(raw):
        state.capture(layer, key, value)
    state.append_chunk(ids)
    return state, ids, raw


def _paper_indices(raw, context_length, ratio, alpha, sink_size):
    pairs = [(k[:context_length], v[:context_length]) for k, v in raw]
    return select_context_indices(pairs, [context_length], context_length,
                                  ratio, alpha, sink_size)[0]


def _drive(state, ids, raw):
    device, n = raw[0][0].device, len(ids)
    sources = state.gpu_kv_cache if state.cache_mode == "gpu" else state.cpu_kv_cache
    snapshots = [(key.clone(), value.clone()) for key, value in sources]
    state.enable(ids)
    positions = torch.arange(n, device=device)
    assert state.note_input(torch.tensor(ids, device=device), positions)
    state.begin_prefill(positions, [RotaryProbe() for _ in raw])
    fusion_layers = list(range(state.first_fusion_layer, state.num_layers))
    expected_pending = fusion_layers[:1] if state.cache_mode == "cpu_pipeline" else []
    assert sorted(state._prefetched) == expected_pending
    outputs = []
    for layer in fusion_layers:
        key = _rotate(raw[layer][0].index_select(0, state.selected_indices),
                      positions.index_select(0, state.selected_indices))
        key = key + 3
        value = raw[layer][1].index_select(0, state.selected_indices) - 5
        if layer % 2 == 0:
            state.enter_layer(layer)
            state.enter_layer(layer)  # Pre-QKV hook is idempotent before merge.
        merged_key, merged_value = state.merge(layer, key, value)
        expected_key = _rotate(raw[layer][0], positions)
        expected_value = raw[layer][1].clone()
        expected_key.index_copy_(0, state.selected_indices, key)
        expected_value.index_copy_(0, state.selected_indices, value)
        torch.testing.assert_close(merged_key, expected_key, atol=0, rtol=0)
        torch.testing.assert_close(merged_value, expected_value, atol=0, rtol=0)
        outputs.append((merged_key.clone(), merged_value.clone()))
        if state.cache_mode != "cpu_pipeline":
            assert not state._prefetched
    state.finish_prefill()
    torch.cuda.synchronize(device)
    for source_pair, snapshot_pair in zip(sources, snapshots):
        for source, snapshot in zip(source_pair, snapshot_pair):
            assert torch.equal(source, snapshot)
    assert not state.active and not state.enabled
    return outputs, state.diagnostics()


@pytest.mark.parametrize("mode", ["cpu_pipeline", "cpu_serial", "gpu"])
@pytest.mark.parametrize("check_layer", [0, 1, 3])
def test_modes_merge_global_positions_repeat_and_preserve_sources(gpu, mode, check_layer):
    state, ids, raw = _collected(gpu)
    info = state.prepare(ids, 3, 0.5, alpha=0.25, sink_size=2,
                         check_layer=check_layer, cache_mode=mode)
    assert info["selected_indices"][-3:] == [14, 15, 16]
    assert info["selected_indices"][:2] == [0, 1]
    assert state._full_cache is None
    first, diagnostics = _drive(state, ids, raw)
    second, repeated = _drive(state, ids, raw)
    for left, right in zip(first, second):
        for a, b in zip(left, right):
            assert torch.equal(a, b)
    source_bytes = (17 - 10) * 8 * 2 * 2 * 4
    consumed_bytes = source_bytes // 4 * (4 - state.first_fusion_layer)
    assert diagnostics["cache_mode"] == mode
    assert diagnostics["retained_full_cache_cpu_bytes"] == 0
    assert diagnostics["compact_cpu_cache_bytes"] == (0 if mode == "gpu" else source_bytes)
    assert diagnostics["compact_gpu_cache_bytes"] == (source_bytes if mode == "gpu" else 0)
    assert diagnostics["transferred_bytes"] == (0 if mode == "gpu" else consumed_bytes)
    assert diagnostics["device_copy_bytes"] == (consumed_bytes if mode == "gpu" else 0)
    assert repeated["merged_layers"] == list(range(state.first_fusion_layer, 4))
    if mode != "cpu_pipeline":
        assert state._transfer_stream is None and state._rope_stream is None
    state.reset()
    assert not state.prepared
    assert state.diagnostics()["compact_cpu_cache_bytes"] == 0
    assert state.diagnostics()["compact_gpu_cache_bytes"] == 0


def test_all_modes_produce_identical_selection_and_merged_kv(gpu):
    results, selections = [], []
    for mode in ("cpu_pipeline", "cpu_serial", "gpu"):
        state, ids, raw = _collected(gpu)
        selections.append(state.prepare(ids, 3, 0.5, cache_mode=mode)["selected_indices"])
        result, _ = _drive(state, ids, raw)
        results.append([(k.cpu(), v.cpu()) for k, v in result])
        state.reset()
    assert selections[0] == selections[1] == selections[2]
    for candidate in results[1:]:
        for baseline_pair, actual_pair in zip(results[0], candidate):
            for baseline, actual in zip(baseline_pair, actual_pair):
                assert torch.equal(baseline, actual)


@pytest.mark.parametrize("ratio,alpha,sink", [
    (0.25, 0.5, 0), (0.5, 0.25, 2), (0.5, 1.0, 1),
    (0.5, 0.5, 7), (0.0, 0.5, 0), (1.0, 0.5, 14),
])
def test_selection_matches_published_all_layer_kv_scorer(gpu, ratio, alpha, sink):
    state, ids, raw = _collected(gpu)
    expected = _paper_indices(raw, 14, ratio, alpha, sink).tolist()
    info = state.prepare(ids, 3, ratio, alpha=alpha, sink_size=sink)
    assert info["selected_indices"] == expected + [14, 15, 16]
    assert info["selected_context_tokens"] == int(14 * ratio)
    assert info["alpha"] == alpha and info["sink_size"] == sink
    assert info["score_layer"] is None
    assert info["score_layers"] == list(range(4))
    assert info["signal"] == "K/V-all-layers"
    assert info["chunk_context_lengths"] == [14]
    assert info["chunk_budgets"] == [int(14 * ratio)]
    state.reset()


@pytest.mark.parametrize("mode", ["cpu_pipeline", "cpu_serial", "gpu"])
def test_external_unsorted_indices_append_suffix_and_preserve_budget(gpu, mode):
    state, ids, raw = _collected(gpu)
    external = torch.tensor([11, 0, 8, 2, 5, 4, 6], device=gpu, dtype=torch.int32)
    before = external.clone()
    info = state.prepare(ids, 3, 0.5, sink_size=1,
                         selected_context_indices=external, check_layer=0, cache_mode=mode)
    assert torch.equal(external, before)
    assert info["selected_indices"] == [0, 2, 4, 5, 6, 8, 11, 14, 15, 16]
    assert info["selection_source"] == "external_context_indices"
    assert info["score_layer"] is None
    assert info["signal"] == "External-indices"
    _drive(state, ids, raw)
    state.reset()


def test_reprepare_requires_explicit_full_cache_retention_and_mode_switch(gpu):
    state, ids, raw = _collected(gpu)
    state.prepare(ids, 3, 0.5, retain_full_cache=True)
    full_snapshot = [(k.clone(), v.clone()) for k, v in state._full_cache]
    assert state.diagnostics()["retained_full_cache_cpu_bytes"] == 4 * 17 * 8 * 2 * 2
    _drive(state, ids, raw)
    state.prepare(ids, 3, 0.25, cache_mode="gpu", retain_full_cache=True)
    assert state.cpu_kv_cache is None
    _drive(state, ids, raw)
    for pair, snapshot in zip(state._full_cache, full_snapshot):
        for a, b in zip(pair, snapshot):
            assert torch.equal(a, b)
    state.prepare(ids, 3, 0.5, cache_mode="cpu_serial")
    assert state.gpu_kv_cache is None and state._full_cache is None
    saved_selection = state.selected_indices.clone()
    with pytest.raises(ValueError, match="No full independent cache remains"):
        state.prepare(ids, 3, 0.25)
    assert state.prepared and torch.equal(saved_selection, state.selected_indices)
    _drive(state, ids, raw)
    state.reset()


@pytest.mark.parametrize("kwargs", [
    {"alpha": -0.1}, {"alpha": 1.1}, {"alpha": float("nan")}, {"alpha": True},
    {"sink_size": -1}, {"sink_size": 8}, {"sink_size": True},
    {"check_layer": -1}, {"check_layer": 4}, {"check_layer": 1.5},
    {"cache_mode": "disk"}, {"retain_full_cache": 1},
    {"selected_context_indices": [0, 1]},
    {"selected_context_indices": [0, 1, 2, 3, 4, 5, 5]},
    {"selected_context_indices": [0, 1, 2, 3, 4, 5, 14]},
    {"selected_context_indices": [-1, 1, 2, 3, 4, 5, 6]},
    {"selected_context_indices": [0, 1, 2, 3, 4, 5, 6.0]},
    {"selected_context_indices": [False, 1, 2, 3, 4, 5, 6]},
    {"selected_context_indices": [1, 2, 3, 4, 5, 6, 7], "sink_size": 1},
    {"selected_context_indices": torch.arange(7).float()},
    {"selected_context_indices": torch.arange(7).reshape(1, 7)},
])
def test_invalid_options_are_rejected_before_cuda_or_collection(kwargs):
    with pytest.raises(ValueError):
        CacheTuneState(4).prepare(list(range(17)), 3, 0.5, **kwargs)


@pytest.mark.parametrize("mode", ["cpu_pipeline", "cpu_serial", "gpu"])
@pytest.mark.parametrize("ratio", [0.0, 1.0])
def test_empty_reuse_or_context_selection_still_merges_and_repeats(gpu, mode, ratio):
    state, ids, raw = _collected(gpu)
    state.prepare(ids, 3, ratio, cache_mode=mode)
    _drive(state, ids, raw)
    _drive(state, ids, raw)
    state.reset()


def test_incomplete_or_out_of_order_prefill_is_rejected(gpu):
    state, ids, raw = _collected(gpu)
    state.prepare(ids, 3, 0.5, cache_mode="cpu_serial", check_layer=0)
    state.enable(ids)
    positions = torch.arange(len(ids), device=gpu)
    state.note_input(torch.tensor(ids, device=gpu), positions)
    state.begin_prefill(positions, [RotaryProbe() for _ in raw])
    with pytest.raises(RuntimeError, match="increasing layer order"):
        state.enter_layer(2)
    with pytest.raises(RuntimeError, match="did not consume"):
        state.finish_prefill()
    state.reset()


@pytest.mark.parametrize("mode", ["cpu_pipeline", "cpu_serial", "gpu"])
def test_query_suffix_is_not_required_in_offline_cache(gpu, mode):
    state, context_ids, raw = _collected(gpu, n=14)
    prompt = context_ids + [901, 902, 903]
    info = state.prepare(prompt, 3, .5, cache_mode=mode, retain_full_cache=True)
    assert info["collected_tokens"] == 14 and info["suffix_cached"] is False
    assert info["selected_indices"][-3:] == [14, 15, 16]
    assert all(k.shape[0] == 14 for k, _ in state._full_cache)
    # Reuse can be re-prepared for a different online query of the same length.
    second_prompt = context_ids + [907, 908, 909]
    second = state.prepare(second_prompt, 3, .5, cache_mode=mode)
    assert info["selected_indices"] == second["selected_indices"]
    assert second["suffix_cached"] is False
    state.reset()


def test_chunk_budgets_remain_independent_after_collection(gpu):
    state = CacheTuneState(3)
    state.start_collect()
    ids = []
    for first, n in ((100, 5), (200, 5)):
        chunk_ids = list(range(first, first+n))
        assert not state.note_input(torch.tensor(chunk_ids, device=gpu),
                                    torch.arange(n, device=gpu))
        for layer in range(3):
            state.capture(layer, torch.ones(n, 8, device=gpu),
                          torch.ones(n, 8, device=gpu))
        state.append_chunk(chunk_ids)
        ids.extend(chunk_ids)
    info = state.prepare(ids + [300], 1, .15)
    assert info["chunk_context_lengths"] == [5, 5]
    assert info["chunk_budgets"] == [0, 0]
    assert info["selected_context_tokens"] == 0
    assert info["selected_indices"] == [10]
    state.reset()
