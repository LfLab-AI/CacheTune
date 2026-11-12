"""Run with: python -m pytest tests/test_attention.py -q

GPU cases require CUDA, BF16 support, and the port's FlashInfer 0.6.4 runtime.
They exercise the real FlashInfer kernel, never a replacement/mock kernel.
"""

import math

import pytest
import torch

from cachetune.attention import SparseAttentionPlan


@pytest.fixture
def gpu():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the real FlashInfer tests")
    if not torch.cuda.is_bf16_supported():
        pytest.skip("BF16-capable CUDA device is required")
    pytest.importorskip("flashinfer", reason="FlashInfer is not installed")
    return torch.device("cuda", torch.cuda.current_device())


def _data(device, n=129, hq=8, hkv=2, dim=64, seed=731):
    gen = torch.Generator(device=device).manual_seed(seed)
    return tuple(
        torch.randn(n, heads, dim, device=device, dtype=torch.bfloat16, generator=gen)
        for heads in (hq, hkv, hkv)
    )


def _dense_reference(q_full, k, v, positions, scale, window=None):
    """Independent dense full-sequence causal attention, then select rows."""
    n, hq, _ = q_full.shape
    hkv = k.shape[1]
    k_heads = k.float().repeat_interleave(hq // hkv, dim=1)
    v_heads = v.float().repeat_interleave(hq // hkv, dim=1)
    scores = torch.einsum("ihd,jhd->hij", q_full.float(), k_heads) * scale
    rows = torch.arange(n, device=q_full.device)[:, None]
    columns = torch.arange(n, device=q_full.device)[None, :]
    mask = columns <= rows
    if window is not None:
        mask &= columns > rows - window
    scores.masked_fill_(~mask[None], float("-inf"))
    out = torch.einsum("hij,jhd->ihd", scores.softmax(-1), v_heads)
    return out.index_select(0, positions.to(q_full.device))


@pytest.mark.parametrize(
    "positions,full_length,error",
    [
        (torch.tensor([], dtype=torch.long), 5, ValueError),
        (torch.tensor([1, 1]), 5, ValueError),
        (torch.tensor([2, 1]), 5, ValueError),
        (torch.tensor([-1, 2]), 5, ValueError),
        (torch.tensor([0, 5]), 5, ValueError),
        (torch.tensor([[0, 1]]), 5, ValueError),
        (torch.tensor([0.0, 1.0]), 5, TypeError),
        (torch.tensor([0]), 0, ValueError),
        (torch.tensor([0]), True, TypeError),
    ],
)
def test_invalid_plan(positions, full_length, error):
    with pytest.raises(error):
        SparseAttentionPlan(positions, full_length)


@pytest.mark.parametrize("window", [0, -2, True, 1.5])
def test_invalid_window(window):
    with pytest.raises((TypeError, ValueError)):
        SparseAttentionPlan(torch.tensor([0]), 5, window=window)


def test_plan_snapshots_positions():
    positions = torch.tensor([0, 3, 7], dtype=torch.int32)
    plan = SparseAttentionPlan(positions, 9)
    positions[1] = 2
    exposed = plan.selected_positions
    exposed[2] = 4
    assert plan.selected_positions.tolist() == [0, 3, 7]
    assert plan.num_queries == 3
    assert plan.full_length == 9
    assert plan.cached_mask_count == 0


def test_cpu_execution_fails_explicitly():
    plan = SparseAttentionPlan(torch.tensor([0, 2]), 3)
    with pytest.raises(ValueError, match="CUDA"):
        plan.run(
            torch.zeros(2, 4, 64, dtype=torch.bfloat16),
            torch.zeros(3, 2, 64, dtype=torch.bfloat16),
            torch.zeros(3, 2, 64, dtype=torch.bfloat16),
        )


@pytest.mark.parametrize(
    "hq,hkv,dim,window",
    [(8, 8, 64, None), (8, 2, 128, None), (12, 3, 128, 17), (8, 2, 64, 1)],
)
def test_bf16_against_dense_original_position_causal(gpu, hq, hkv, dim, window):
    q_full, k, v = _data(gpu, hq=hq, hkv=hkv, dim=dim)
    positions = torch.tensor([0, 1, 6, 17, 64, 67, 95, 128])
    plan = SparseAttentionPlan(positions, len(k), window=window)
    q = q_full.index_select(0, positions.to(gpu))
    before = [x.clone() for x in (q, k, v)]
    scale = 0.37 / math.sqrt(dim)
    actual = plan.run(q, k, v, scale=scale)
    expected = _dense_reference(q_full, k, v, positions, scale, window)
    assert actual.shape == q.shape
    assert actual.dtype == torch.bfloat16
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.float(), expected, atol=1e-2, rtol=1e-2)
    for source, snapshot in zip((q, k, v), before):
        torch.testing.assert_close(source, snapshot, atol=0, rtol=0)


def test_future_kv_cannot_change_past_selected_outputs(gpu):
    q_full, k, v = _data(gpu)
    positions = torch.tensor([0, 7, 29, 64, 128])
    plan = SparseAttentionPlan(positions, len(k))
    q = q_full.index_select(0, positions.to(gpu))
    original = plan.run(q, k, v)
    changed_k, changed_v = k.clone(), v.clone()
    changed_k[30:] = -changed_k[30:] * 3
    changed_v[30:] += 20
    changed = plan.run(q, changed_k, changed_v)
    torch.testing.assert_close(changed[:3], original[:3], atol=0, rtol=0)
    assert not torch.allclose(changed[3:].float(), original[3:].float())


def test_window_excludes_old_keys_at_original_position(gpu):
    q_full, k, v = _data(gpu)
    plan = SparseAttentionPlan(torch.tensor([64]), len(k), window=9)
    q = q_full[64:65]
    original = plan.run(q, k, v)
    changed_k, changed_v = k.clone(), v.clone()
    # Only original positions [56, 64] can contribute to this query.
    changed_k[:56] *= -7
    changed_v[:56] += 30
    changed_k[65:] *= -7
    changed_v[65:] += 30
    changed = plan.run(q, changed_k, changed_v)
    torch.testing.assert_close(changed, original, atol=0, rtol=0)


def test_packed_mask_reused_across_layers_and_window_switch(gpu, monkeypatch):
    import flashinfer.quantization as quantization

    count = 0
    real_packbits = quantization.packbits

    def counted_packbits(*args, **kwargs):
        nonlocal count
        count += 1
        return real_packbits(*args, **kwargs)

    monkeypatch.setattr(quantization, "packbits", counted_packbits)
    positions = torch.tensor([0, 3, 7, 11, 18])
    plan = SparseAttentionPlan(positions, 19, window=4)
    plan.prepare(gpu)
    initial_count = count
    assert initial_count > 0
    for seed in (123, 456):
        q_full, k, v = _data(gpu, n=19, seed=seed)
        q = q_full.index_select(0, positions.to(gpu))
        actual = plan.run(q, k, v)
        expected = _dense_reference(q_full, k, v, positions, 1 / 8, window=4)
        torch.testing.assert_close(actual.float(), expected, atol=1e-2, rtol=1e-2)
    assert count == initial_count
    assert plan.cached_mask_count == 1
    actual = plan.run(q, k, v, window=-1)
    expected = _dense_reference(q_full, k, v, positions, 1 / 8)
    torch.testing.assert_close(actual.float(), expected, atol=1e-2, rtol=1e-2)
    assert plan.cached_mask_count == 2
    assert count > initial_count
    after_second_mask = count
    plan.run(q, k, v)
    assert count == after_second_mask


def test_chunked_mask_packing_has_no_intermediate_padding(gpu, monkeypatch):
    import cachetune.attention as attention

    monkeypatch.setattr(attention, "_MASK_CHUNK_BYTES", 128)
    q_full, k, v = _data(gpu, n=29)
    positions = torch.tensor([0, 1, 3, 4, 7, 11, 15, 19, 20, 22, 25, 27, 28])
    plan = SparseAttentionPlan(positions, 29)
    actual = plan.run(q_full.index_select(0, positions.to(gpu)), k, v)
    expected = _dense_reference(q_full, k, v, positions, 1 / 8)
    torch.testing.assert_close(actual.float(), expected, atol=1e-2, rtol=1e-2)


def test_cross_stream_preparation(gpu):
    q_full, k, v = _data(gpu, n=29)
    positions = torch.tensor([0, 4, 11, 28])
    plan = SparseAttentionPlan(positions, 29)
    with torch.cuda.stream(torch.cuda.Stream(device=gpu)):
        plan.prepare(gpu)
    actual = plan.run(q_full.index_select(0, positions.to(gpu)), k, v)
    expected = _dense_reference(q_full, k, v, positions, 1 / 8)
    torch.testing.assert_close(actual.float(), expected, atol=1e-2, rtol=1e-2)


def test_wrong_query_count_fails_before_kernel(gpu):
    q_full, k, v = _data(gpu, n=29)
    plan = SparseAttentionPlan(torch.tensor([0, 4, 28]), 29)
    with pytest.raises(ValueError, match="Q row count"):
        plan.run(q_full[:2], k, v)


@pytest.mark.parametrize("scale", [0, -1, float("nan"), float("inf")])
def test_invalid_scale_fails_before_kernel(gpu, scale):
    q_full, k, v = _data(gpu, n=29)
    plan = SparseAttentionPlan(torch.tensor([0, 4, 28]), 29)
    with pytest.raises(ValueError, match="scale"):
        plan.run(q_full[[0, 4, 28]], k, v, scale=scale)
