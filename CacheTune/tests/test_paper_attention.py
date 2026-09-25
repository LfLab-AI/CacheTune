"""CPU reference checks for selected-query causal attention, without vLLM."""

import ast
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch


REPO = Path(__file__).resolve().parents[1]
HELPER = REPO / "vllm_blend/vllm/attention/paper_attention.py"
BACKEND = REPO / "vllm_blend/vllm/attention/backends/xformers.py"
SPEC = importlib.util.spec_from_file_location("paper_attention_helper", HELPER)
paper_attention = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(paper_attention)
make_paper_causal_bias = paper_attention.make_paper_causal_bias


class PaperAttentionTests(unittest.TestCase):
    def test_noncontiguous_queries_use_absolute_causality(self):
        positions = torch.tensor([0, 3, 8, 9, 10])
        query = torch.zeros(1, len(positions), 4, 8)
        bias = make_paper_causal_bias(query, positions, 11)
        self.assertEqual(bias.shape, (1, 4, 5, 11))
        dense = torch.zeros(11, 11).masked_fill(
            torch.arange(11)[None, :] > torch.arange(11)[:, None], float('-inf'))
        torch.testing.assert_close(bias[0, 0], dense[positions])
        self.assertTrue(torch.isneginf(bias[0, 0, 0, 1:]).all())
        self.assertEqual(bias.stride(-1), 1)
        self.assertEqual(bias.stride(-2) % 8, 0)
        self.assertEqual(bias.stride(1), 0)  # All heads share the padded 2-D storage.

    def test_mha_and_gqa_outputs_match_dense_selected_rows(self):
        generator = torch.Generator().manual_seed(120)
        n, groups, heads_per_group, d = 13, 2, 3, 8
        positions = torch.tensor([0, 2, 5, 10, 11, 12])
        queries = torch.randn(n, groups, heads_per_group, d,
                              generator=generator, dtype=torch.float64)
        keys = torch.randn(n, groups, d, generator=generator, dtype=torch.float64)
        values = torch.randn(n, groups, d, generator=generator, dtype=torch.float64)
        flat_q = queries.reshape(n, groups * heads_per_group, d)
        flat_k = keys[:, :, None].expand_as(queries).reshape_as(flat_q)
        flat_v = values[:, :, None].expand_as(queries).reshape_as(flat_q)
        dense_logits = torch.einsum('qhd,khd->hqk', flat_q, flat_k) / d**0.5
        future = torch.arange(n)[None, :] > torch.arange(n)[:, None]
        dense_logits.masked_fill_(future[None], float('-inf'))
        expected = torch.einsum('hqk,khd->qhd', dense_logits.softmax(-1), flat_v)[positions]

        for gqa in (False, True):
            with self.subTest(gqa=gqa):
                selected_q = queries[positions] if gqa else flat_q[positions]
                bias = make_paper_causal_bias(selected_q[None], positions, n)
                if gqa:
                    self.assertEqual(bias.shape, (1, groups, heads_per_group, len(positions), n))
                else:
                    self.assertEqual(bias.shape, (1, groups * heads_per_group, len(positions), n))
                selected_logits = torch.einsum('qhd,khd->hqk', flat_q[positions], flat_k) / d**0.5
                selected_logits += bias.reshape(groups * heads_per_group, len(positions), n)
                observed = torch.einsum('hqk,khd->qhd', selected_logits.softmax(-1), flat_v)
                torch.testing.assert_close(observed, expected)

    def test_complete_sequence_suffix_and_dtypes(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            for positions in (torch.arange(16), torch.arange(12, 16), torch.tensor([15])):
                with self.subTest(dtype=dtype, positions=positions.tolist()):
                    bias = make_paper_causal_bias(torch.zeros(1, len(positions), 2, 8, dtype=dtype), positions, 16)
                    self.assertEqual(bias.dtype, dtype)
                    for i, position in enumerate(positions):
                        self.assertTrue((bias[0, 0, i, :position + 1] == 0).all())
                        self.assertTrue(torch.isneginf(bias[0, 0, i, position + 1:]).all())

    def test_invalid_positions_and_query_shapes_fail(self):
        valid_query = torch.zeros(1, 2, 3, 8)
        for positions in (torch.tensor([-1, 2]), torch.tensor([0, 11]),
                          torch.tensor([1]), torch.tensor([0., 1.]), torch.tensor([[0, 1]])):
            with self.assertRaises(ValueError):
                make_paper_causal_bias(valid_query, positions, 11)
        for query in (torch.zeros(2, 2, 3, 8), torch.zeros(2, 3, 8),
                      valid_query.long(), torch.zeros(1, 0, 3, 8)):
            with self.assertRaises(ValueError):
                make_paper_causal_bias(query, torch.tensor([0, 1]), 11)

    def test_backend_gate_preserves_legacy_and_reuses_paper_bias(self):
        # Extract the real method body to test its dispatch without importing the
        # GPU-only vLLM package. xFormers is mocked only at its external call.
        syntax = ast.parse(BACKEND.read_text(encoding='utf-8'))
        backend_class = next(node for node in syntax.body if isinstance(node, ast.ClassDef)
                             and node.name == 'XFormersImpl')
        method = next(node for node in backend_class.body if isinstance(node, ast.FunctionDef)
                      and node.name == '_run_memory_efficient_xformers_forward')
        module = ast.Module(body=[method], type_ignores=[])
        recorded = []

        def attention(query, key, value, **kwargs):
            recorded.append(kwargs['attn_bias'])
            return torch.zeros_like(query)

        namespace = dict(torch=torch, XFormersMetadata=object,
                         xops=SimpleNamespace(memory_efficient_attention_forward=attention))
        exec(compile(module, str(BACKEND), 'exec'), namespace)
        run = namespace['_run_memory_efficient_xformers_forward']
        impl = SimpleNamespace(num_kv_heads=2, num_heads=2, num_queries_per_kv=1,
                               alibi_slopes=None, scale=.25)
        metadata = SimpleNamespace(prompt_lens=[11], attn_bias=object())
        query, key = torch.zeros(3, 2, 8), torch.zeros(11, 2, 8)
        legacy_bias = object()
        state = dict(attn_bias=legacy_bias, imp_indices=torch.tensor([0, 5, 10]))
        run(impl, query, key, key, metadata, 1, state)
        self.assertIs(recorded[-1], legacy_bias)
        state['paper_causal_mask'] = True
        with patch.dict(sys.modules, {'vllm.attention.paper_attention': paper_attention}):
            run(impl, query, key, key, metadata, 1, state)
            paper_bias = recorded[-1]
            self.assertIsInstance(paper_bias, torch.Tensor)
            self.assertTrue(torch.isneginf(paper_bias[0, 0, 0, 1:]).all())
            run(impl, query, key, key, metadata, 2, state)
            self.assertIs(recorded[-1], paper_bias)
            # A fresh status=1 rebuilds it for the next request's positions.
            state['imp_indices'] = torch.tensor([2, 4, 10])
            run(impl, query, key, key, metadata, 1, state)
            self.assertIsNot(recorded[-1], paper_bias)
            self.assertEqual(recorded[-1][0, 0, 0, 2].item(), 0)


if __name__ == '__main__':
    unittest.main()
