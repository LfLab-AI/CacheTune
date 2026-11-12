"""Hybrid branch contract checks; GPU cases exercise real CUDA copy streams."""
import pytest
import torch

from cachetune.hybrid_state import HybridCacheTuneState


def _collect(state, ids, *, drop=False, device="cpu"):
    tokens = torch.tensor(ids, dtype=torch.long, device=device)
    state.note_input(tokens, torch.arange(len(ids), device=device))
    expected = {}
    for layer in range(state.num_actual_layers):
        if layer in state.full_attention_layer_ids:
            logical = state.full_attention_layer_ids.index(layer)
            kv = tokens[:, None].float().expand(-1, 2).clone() + logical
            state.capture(logical, kv, kv + 1)
        for name in ("attention", "mlp"):
            tensor = tokens[:, None].float().expand(-1, 3).clone() + layer * 100 + (50 if name == "mlp" else 0)
            state.capture_branch(layer, name, tensor)
            expected[(layer, name)] = tensor.detach().cpu().clone()[int(drop):]
            tensor.zero_()  # Later projection-buffer mutation must not alter capture.
    state.append_chunk(ids, drop_bos=drop)
    return expected


def test_collection_drop_bos_after_final_kv_and_immutability():
    state = HybridCacheTuneState([1, 3], num_actual_layers=5)
    state.start_collect()
    first = _collect(state, [10, 11, 12])
    second = _collect(state, [99, 13, 14], drop=True)
    assert state._collected_ids == [10, 11, 12, 13, 14]
    # Actual layer 4 follows the final KV-producing layer 3.
    for layer, names in state._branch_pieces.items():
        for name, pieces in names.items():
            torch.testing.assert_close(torch.cat(pieces), torch.cat((first[(layer, name)], second[(layer, name)])))
    assert "attention" not in state._branch_pieces[0]
    assert 4 in state._branch_pieces
    state.reset()
    assert not state._branch_sources and not state._branch_pieces


def test_policy_sets_and_actual_boundary():
    state = HybridCacheTuneState([3, 7, 11], num_actual_layers=13)
    assert state.hybrid_policy == "block" and state.check_layer == 0
    assert state._required("block", 0) == (
        3, {3: ("attention", "mlp"), 7: ("attention", "mlp"), 11: ("attention", "mlp")})
    assert state._required("attention", 1) == (7, {7: ("attention",), 11: ("attention",)})
    assert state._required("block", 1) == (7, {7: ("attention", "mlp"), 11: ("attention", "mlp")})
    _, required = state._required("mlp", 1)
    assert sorted(required) == list(range(7, 13))
    assert required[8] == ("mlp",) and required[11] == ("attention", "mlp")


def test_hybrid_fusion_boundary_cannot_be_overridden():
    state = HybridCacheTuneState([1, 3], num_actual_layers=5)
    with pytest.raises(ValueError, match="always fuses"):
        state.prepare([1, 2, 3], 1, 0.5, fuse_check_layer=False)


@pytest.mark.parametrize("new_request", ["reset", "start_collect"])
def test_recurrent_probe_reference_is_same_request_only(new_request):
    state = HybridCacheTuneState([1, 3], num_actual_layers=5)
    reference = {0: {"conv": torch.ones(1, 2), "ssm": torch.ones(1, 2, 2)}}
    state._recurrent_reference = reference
    state._recurrent_snapshot = reference
    state._recurrent_probe_next = True
    # Full/Cached comparison within one request must retain its reference.
    state.disable()
    assert state._recurrent_reference is reference
    # A new collection or explicit cache reset must cancel an unconsumed probe
    # and remove both snapshots, even when the next sample has identical shapes.
    getattr(state, new_request)()
    assert state._recurrent_probe_next is False
    assert state._recurrent_snapshot is None
    assert state._recurrent_reference is None


@pytest.mark.parametrize("ids,total", [([3, 1], None), ([1, 1], None), ([1], None), ([True, 3], None), ([1, 3], 3)])
def test_invalid_layer_maps(ids, total):
    with pytest.raises(ValueError):
        HybridCacheTuneState(ids, num_actual_layers=total)


class _IdentityRope:
    q_size = 2

    def rotary_emb(self, positions, q, k):
        return q, k


@pytest.fixture
def gpu():
    if not torch.cuda.is_available():
        pytest.skip("Real CUDA streams are required")
    return torch.device("cuda", torch.cuda.current_device())


@pytest.mark.parametrize("mode", ["cpu_pipeline", "cpu_serial", "gpu"])
@pytest.mark.parametrize("policy", ["attention", "block", "mlp"])
@pytest.mark.parametrize("ratio", [0.0, 0.4, 1.0])
def test_gpu_branch_reconstruction_repeat_and_accounting(gpu, mode, policy, ratio):
    state = HybridCacheTuneState([1, 3], num_actual_layers=5)
    ids = [10, 11, 12, 13, 14, 15]
    state.start_collect()
    expected = _collect(state, ids, device=gpu)
    selection = state.prepare(ids, 1, ratio, hybrid_policy=policy, cache_mode=mode)
    assert selection["score_actual_layer"] is None
    assert selection["score_actual_layers"] == [1, 3]
    assert selection["scored_layer_indices"] == [1, 3]
    assert selection["first_fusion_layer"] == 0
    assert selection["hybrid_paper_default"] is (policy == "block")
    assert selection["hybrid_experimental"] is (policy != "block")
    assert state._branch_full is None and not state._branch_pieces
    snapshots = {i: {name: t.detach().cpu().clone() for name, t in row.items()}
                 for i, row in state._branch_sources.items()}
    for repeat in range(2):
        state.enable(ids)
        positions = torch.arange(len(ids), device=gpu)
        assert state.note_input(torch.tensor(ids, device=gpu), positions)
        state.begin_prefill(positions, [_IdentityRope(), _IdentityRope()])
        state.begin_hybrid()
        for layer, names in state.required_branches.items():
            state.enter_branch_layer(layer)
            state.enter_branch_layer(layer)  # Idempotent before a layer's merges.
            for name in names:
                updated = torch.full((state.selected_indices.numel(), 3), 1000 + repeat,
                                     device=gpu, dtype=torch.float32)
                actual = state.merge_branch(layer, name, updated)
                reference = expected[(layer, name)].clone()
                reference[state._selected_cpu] = 1000 + repeat
                torch.testing.assert_close(actual.cpu(), reference, rtol=0, atol=0)
                if ratio == 1:
                    assert actual is updated
        state.finish_hybrid()
        dense = torch.zeros(state.selected_indices.numel(), 2, device=gpu)
        for logical in range(len(state.full_attention_layer_ids)):
            state.merge(logical, dense, dense)
        state.finish_prefill()
        diag = state.diagnostics()
        source_bytes = sum(t.numel() * t.element_size() for row in snapshots.values() for t in row.values())
        assert diag["branch_transferred_bytes"] == (0 if mode == "gpu" else source_bytes)
        assert diag["total_transferred_bytes"] == diag["transferred_bytes"] + diag["branch_transferred_bytes"]
        for i, row in state._branch_sources.items():
            for name, t in row.items():
                torch.testing.assert_close(t.cpu(), snapshots[i][name], rtol=0, atol=0)
    state.reset()
    assert state._branch_full is None and not state._branch_sources


def test_gpu_reprepare_requires_retention_and_preserves_policy_options(gpu):
    state = HybridCacheTuneState([1, 3], num_actual_layers=5)
    ids = [10, 11, 12, 13]
    state.start_collect()
    _collect(state, ids, device=gpu)
    state.prepare(ids, 1, 0.5, hybrid_policy="attention", retain_full_cache=True)
    assert state._branch_full is not None
    state.prepare(ids, 1, 1.0, hybrid_policy="mlp", retain_full_cache=False)
    assert state._branch_full is None and 4 in state.required_branches
    with pytest.raises(ValueError, match="No full branch"):
        state.prepare(ids, 1, 0.5)


def test_gpu_context_only_collection_and_new_query_do_not_change_ranking(gpu):
    state = HybridCacheTuneState([1, 3], num_actual_layers=5)
    context = [10, 11, 12, 13, 14, 15]
    state.start_collect()
    expected = _collect(state, context, device=gpu)
    first = state.prepare(context + [80, 81], 2, 0.4,
                          cache_mode="gpu", retain_full_cache=True)
    assert first["collected_tokens"] == len(context)
    assert first["suffix_cached"] is False
    context_indices = first["selected_indices"][:-2]
    prompt = context + [90, 91, 92]
    second = state.prepare(prompt, 3, 0.4,
                           cache_mode="gpu", retain_full_cache=True)
    assert second["selected_indices"][:-3] == context_indices
    assert second["selected_indices"][-3:] == [6, 7, 8]
    assert second["scored_layer_indices"] == [1, 3]
    state.enable(prompt)
    positions = torch.arange(len(prompt), device=gpu)
    state.note_input(torch.tensor(prompt, device=gpu), positions)
    state.begin_prefill(positions, [_IdentityRope(), _IdentityRope()])
    state.begin_hybrid()
    current_residual = torch.arange(len(prompt), device=gpu)[:, None].expand(-1, 3) + 10000
    for layer, names in state.required_branches.items():
        for name in names:
            updated = torch.full((state.selected_indices.numel(), 3), 777.0, device=gpu)
            actual = state.merge_branch(layer, name, updated)
            reference = torch.cat((expected[(layer, name)], torch.zeros(3, 3)))
            reference[state._selected_cpu] = 777.0
            torch.testing.assert_close(actual.cpu(), reference, rtol=0, atol=0)
            # A cached branch is added to this request's current residual;
            # the cache never supplies an independently encoded residual.
            torch.testing.assert_close((actual + current_residual).cpu(),
                                       reference + current_residual.cpu(), rtol=0, atol=0)
    state.finish_hybrid()
    fresh = torch.zeros(state.selected_indices.numel(), 2, device=gpu)
    for logical in range(len(state.full_attention_layer_ids)):
        k, v = state.merge(logical, fresh, fresh)
        assert k.shape[0] == v.shape[0] == len(prompt)
    state.finish_prefill()


def test_gpu_branch_order_and_missing_use_are_rejected(gpu):
    state = HybridCacheTuneState([1, 3], num_actual_layers=5)
    ids = [10, 11, 12, 13]
    state.start_collect()
    _collect(state, ids, device=gpu)
    state.prepare(ids, 1, 0.5)
    state.enable(ids)
    p = torch.arange(len(ids), device=gpu)
    state.note_input(torch.tensor(ids, device=gpu), p)
    state.begin_prefill(p, [_IdentityRope(), _IdentityRope()])
    state.begin_hybrid()
    with pytest.raises(RuntimeError, match="order"):
        state.enter_branch_layer(3)
    state.enter_branch_layer(1)
    selected = torch.zeros(state.selected_indices.numel(), 3, device=gpu)
    state.merge_branch(1, "attention", selected)
    with pytest.raises(RuntimeError, match="every required branch"):
        state.finish_hybrid()
    with pytest.raises(RuntimeError, match="order"):
        state.enter_branch_layer(3)
    with pytest.raises(RuntimeError, match="only once"):
        state.merge_branch(1, "attention", selected)
    state.reset()
