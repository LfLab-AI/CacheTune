"""CPU-only checks of legacy debug semantics; no serving engine is needed."""

import io

import pytest
import torch

from cachetune.debug import CacheTuneDebugCapture


def _tensors():
    generator = torch.Generator().manual_seed(79)
    return (torch.randn(6, 4, 3, generator=generator),
            torch.randn(6, 2, 3, generator=generator),
            torch.randn(6, 2, 3, generator=generator))


def _capture_proxy(capture, q, k, positions=None, **kwargs):
    options = dict(full_length=6, num_heads=4, num_kv_heads=2, head_dim=3, scale=0.4)
    options.update(kwargs)
    capture.capture_after_rope(3, q, k, torch.arange(len(q)) if positions is None else positions, **options)
    return capture.export()["layers"][3]["attention_proxy"]


def test_proxy_matches_gqa_head_zero_reference_and_is_unmasked():
    q, k, _ = _tensors()
    # Other query/KV heads must not leak into the head-zero diagnostic.
    q[:, 1:] = 1000
    k[:, 1:] = -1000
    proxy = _capture_proxy(CacheTuneDebugCapture(attention_proxy=True, suffix_len=2), q, k)
    expanded_k = k.repeat_interleave(2, dim=1)
    logits = torch.einsum("qhd,khd->hqk", q[-2:], expanded_k) * 0.4
    expected = logits[0].softmax(-1)
    torch.testing.assert_close(proxy["weights"], expected)
    # Original query 4 is allowed to attend to key 5 in this legacy proxy.
    assert proxy["weights"][0, 5] > 0
    assert proxy["metadata"]["is_actual_attention"] is False
    assert proxy["metadata"]["causal_mask_applied"] is False
    assert proxy["metadata"]["kv_head"] == 0


@pytest.mark.parametrize("flatten", [False, True])
def test_sparse_suffix_uses_original_positions_not_last_query_rows(flatten):
    q, k, _ = _tensors()
    full = _capture_proxy(CacheTuneDebugCapture(attention_proxy=True, suffix_len=2), q, k)
    # The suffix occurs at rows 0 and 2 here, not at the final two Q rows.
    positions = torch.tensor([5, 0, 4, 2], dtype=torch.int32)
    selected_q = q.index_select(0, positions.long())
    if flatten:
        selected_q, k = selected_q.flatten(1), k.flatten(1)
    sparse = _capture_proxy(CacheTuneDebugCapture(attention_proxy=True, suffix_len=2), selected_q, k, positions)
    torch.testing.assert_close(sparse["weights"], full["weights"], rtol=0, atol=0)
    assert sparse["query_positions"].tolist() == [4, 5]
    assert sparse["key_positions"].tolist() == list(range(6))
    assert sparse["metadata"]["query_mode"] == "sparse"


def test_snapshots_are_detached_immutable_exportable_and_resettable():
    q, k, v = (t.requires_grad_() for t in _tensors())
    positions = torch.arange(6)
    capture = CacheTuneDebugCapture(raw_qkv=True, attention_proxy=True, suffix_len=2)
    capture.capture_projected_qkv(3, q, k, v)
    _capture_proxy(capture, q, k, positions)
    before = capture.finish()
    with torch.no_grad():
        q.fill_(9); k.zero_(); v.fill_(7)
    positions.zero_()
    # Subsequent decode-like calls cannot overwrite the first prefill capture.
    capture.capture_projected_qkv(3, q[:1], k[:1], v[:1])
    capture.capture_after_rope(3, q[:1], k, torch.tensor([5]), 6, 4, 2, 3, 0.4)
    after = capture.finish()
    for name in ("q", "k", "v"):
        tensor = after["layers"][3]["projected_qkv"][name]
        torch.testing.assert_close(tensor, before["layers"][3]["projected_qkv"][name], rtol=0, atol=0)
        assert tensor.device.type == "cpu" and not tensor.requires_grad
        tensor.zero_()  # Export must not expose internal tensor storage either.
    after["layers"][3]["attention_proxy"]["weights"].zero_()
    fresh = capture.export()
    torch.testing.assert_close(fresh["layers"][3]["projected_qkv"]["q"], before["layers"][3]["projected_qkv"]["q"])
    torch.testing.assert_close(fresh["layers"][3]["attention_proxy"]["weights"], before["layers"][3]["attention_proxy"]["weights"])
    assert fresh["layers"][3]["projected_qkv"]["metadata"]["stage"] == "post_rope_pre_query_selection_pre_kv_merge"
    stream = io.BytesIO()
    torch.save(fresh, stream)
    stream.seek(0)
    restored = torch.load(stream, weights_only=True)
    torch.testing.assert_close(restored["layers"][3]["attention_proxy"]["weights"], before["layers"][3]["attention_proxy"]["weights"])
    capture.reset()
    assert capture.export()["layers"] == {}
    capture.capture_projected_qkv(3, q, k, v)
    assert torch.equal(capture.export()["layers"][3]["projected_qkv"]["q"], q.detach())


def test_disabled_and_layer_filters_skip_capture():
    q, k, v = _tensors()
    for capture in (CacheTuneDebugCapture(), CacheTuneDebugCapture(layers=[], raw_qkv=True),
                    CacheTuneDebugCapture(layers=[31], raw_qkv=True, attention_proxy=True)):
        capture.capture_projected_qkv(3, q, k, v)
        capture.capture_after_rope(3, q, k, torch.arange(6), 6, 4, 2, 3, 0.4)
        assert capture.export()["layers"] == {}
    capture = CacheTuneDebugCapture(layers=[31], raw_qkv=True)
    capture.capture_projected_qkv(31, q, k, v)
    assert list(capture.export()["layers"]) == [31]


@pytest.mark.parametrize("positions", [torch.tensor([0, 1, 5]), torch.tensor([0, 4, 4]),
                                       torch.tensor([-1, 4, 5]), torch.tensor([0, 4, 6])])
def test_invalid_or_incomplete_original_positions_are_rejected(positions):
    q, k, _ = _tensors()
    with pytest.raises(ValueError):
        _capture_proxy(CacheTuneDebugCapture(attention_proxy=True, suffix_len=2), q[:3], k, positions)


@pytest.mark.parametrize("positions", [torch.arange(6).float(), torch.arange(5), torch.arange(6).reshape(2, 3)])
def test_position_dtype_and_row_alignment_are_checked(positions):
    q, k, _ = _tensors()
    with pytest.raises((TypeError, ValueError)):
        _capture_proxy(CacheTuneDebugCapture(attention_proxy=True), q, k, positions)


def test_incomplete_key_cache_and_invalid_head_mapping_are_rejected():
    q, k, _ = _tensors()
    for keys, overrides in ((k[:5], {}), (k, {"num_heads": 3}), (k, {"scale": float("nan")})):
        with pytest.raises(ValueError):
            _capture_proxy(CacheTuneDebugCapture(attention_proxy=True), q, keys, **overrides)


def test_zero_suffix_is_empty_not_entire_query():
    q, k, _ = _tensors()
    proxy = _capture_proxy(CacheTuneDebugCapture(attention_proxy=True, suffix_len=0), q, k)
    assert tuple(proxy["weights"].shape) == (0, 6)
    assert proxy["query_positions"].numel() == 0
