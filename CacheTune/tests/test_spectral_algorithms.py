"""CPU-only checks for the optional spectral implementation; no vLLM required."""

import math
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "example"))
from spectral_algorithms import (  # noqa: E402
    aggregate_spectral_statistics,
    golden_section_search,
    lowpass_reconstruct,
    select_shared_tokens,
    spectral_statistics,
    spectral_token_scores,
)


def full_fft_reference(array, alpha):
    """Independent complex FFT reference, retaining symmetric real-FFT bins."""
    n = array.shape[0]
    cutoff = math.floor(alpha * (n // 2 + 1))
    transformed = np.fft.fft(array, axis=0)
    keep = np.zeros(n, dtype=bool)
    if cutoff:
        keep[:cutoff] = True
        if cutoff > 1:
            keep[-(cutoff - 1):] = True
    transformed[~keep] = 0
    return np.fft.ifft(transformed, axis=0).real


class SpectralTests(unittest.TestCase):
    def test_even_odd_and_single_token_against_full_fft(self):
        rng = np.random.default_rng(81)
        for n in (1, 2, 7, 8, 13):
            source = rng.normal(size=(n, 2, 3))
            for alpha in (0.0, 0.25, 0.5, 1.0):
                with self.subTest(n=n, alpha=alpha):
                    result = lowpass_reconstruct(torch.from_numpy(source), alpha)
                    np.testing.assert_allclose(
                        result.numpy(), full_fft_reference(source, alpha), atol=1e-12
                    )
                    self.assertEqual(result.shape, source.shape)

    def test_known_frequencies_and_half_promotion(self):
        t = torch.arange(16, dtype=torch.float64)
        low = 2 + torch.sin(2 * torch.pi * t / 16)
        high = torch.cos(2 * torch.pi * 6 * t / 16)
        reconstructed = lowpass_reconstruct((low + high)[:, None, None])
        torch.testing.assert_close(reconstructed[:, 0, 0], low)
        promoted = lowpass_reconstruct(torch.ones(16, 2, 3, dtype=torch.float16))
        self.assertEqual(promoted.dtype, torch.float32)
        torch.testing.assert_close(promoted, torch.ones(16, 2, 3))

    def test_independent_chunks_do_not_blend_boundaries(self):
        left = torch.zeros(8, 1, 1, dtype=torch.float64)
        right = torch.full((8, 1, 1), 7.0, dtype=torch.float64)
        separately = torch.cat([lowpass_reconstruct(left), lowpass_reconstruct(right)])
        jointly = lowpass_reconstruct(torch.cat([left, right]))
        torch.testing.assert_close(separately, torch.cat([left, right]))
        self.assertFalse(torch.allclose(separately, jointly))

    def test_score_uses_both_k_and_v_all_layers_after_normalization(self):
        k = torch.tensor([[[[9.]], [[1.]]], [[[1.]], [[1.]]]], dtype=torch.float64)
        v = torch.tensor([[[[1.]], [[1.]]], [[[1.]], [[9.]]]], dtype=torch.float64)
        scores = spectral_token_scores(k, v, alpha=1.0)
        expected = 0.5 * (torch.tensor([5., 1.]) / (6. + 1e-12)
                          + torch.tensor([1., 5.]) / (6. + 1e-12))
        torch.testing.assert_close(scores, expected.double())
        changed = k.clone()
        changed[0, 0] = 1
        self.assertFalse(torch.allclose(scores, spectral_token_scores(changed, v, alpha=1)))
        # Scaling just one layer must not weight that layer more heavily.
        scaled_k, scaled_v = k.clone(), v.clone()
        scaled_k[0] *= 100
        scaled_v[0] *= 100
        torch.testing.assert_close(scores, spectral_token_scores(scaled_k, scaled_v, alpha=1))

    def test_tensor_parallel_reduction_and_replication(self):
        generator = torch.Generator().manual_seed(18)
        k = torch.randn(3, 9, 4, 2, generator=generator, dtype=torch.float64)
        v = torch.randn(3, 9, 4, 2, generator=generator, dtype=torch.float64)
        expected = spectral_token_scores(k, v)
        shards = [spectral_statistics(k[:, :, :2], v[:, :, :2]),
                  spectral_statistics(k[:, :, 2:], v[:, :, 2:])]
        torch.testing.assert_close(aggregate_spectral_statistics(shards), expected)
        replicated = [dict(stat, replication_factor=2) for stat in shards for _ in range(2)]
        torch.testing.assert_close(aggregate_spectral_statistics(replicated), expected)
        for stat in replicated:
            stat['replication_factor'] = [2, 2, 2]
        torch.testing.assert_close(aggregate_spectral_statistics(replicated), expected)

    def test_zero_scores_are_finite(self):
        zeros = torch.zeros(2, 8, 1, 2)
        result = spectral_token_scores(zeros, zeros)
        torch.testing.assert_close(result, torch.zeros(8, dtype=torch.float64))

    def test_selection_floor_chunk_offsets_ties_and_complement(self):
        scores = [torch.tensor([1., 3., 3., 2., 0.]), torch.tensor([4., 9., 1.])]
        chosen, reused = select_shared_tokens(scores, 0.5)
        self.assertEqual(chosen.tolist(), [1, 2, 6])
        self.assertEqual(reused.tolist(), [0, 3, 4, 5, 7])
        self.assertEqual(sorted(chosen.tolist() + reused.tolist()), list(range(8)))
        for ratio in (0., 1.):
            chosen, reused = select_shared_tokens(scores, ratio)
            self.assertEqual(len(chosen), int(8 * ratio))
            self.assertEqual(len(reused), int(8 * (1 - ratio)))
        chosen, reused = select_shared_tokens([], 0.5)
        self.assertEqual(chosen.dtype, torch.long)
        self.assertEqual(len(chosen) + len(reused), 0)

    def test_invalid_spectral_inputs(self):
        valid = torch.ones(8, 1, 1)
        for alpha in (-0.1, 1.1, math.inf, math.nan, True):
            with self.subTest(alpha=alpha), self.assertRaises(ValueError):
                lowpass_reconstruct(valid, alpha)
        for source in (valid.long(), valid[:, :, 0], valid[:0], valid * math.nan):
            with self.assertRaises(ValueError):
                lowpass_reconstruct(source)
        with self.assertRaises(ValueError):
            spectral_statistics([], [])
        with self.assertRaises(ValueError):
            spectral_statistics([valid], [valid[:3]])
        stat = spectral_statistics([valid], [valid])
        for invalid in ([], [dict(stat, k_sq=-stat['k_sq'])],
                        [stat, dict(stat, replication_factor=2)]):
            with self.assertRaises(ValueError):
                aggregate_spectral_statistics(invalid)
        for ratio in (-.1, 1.1, math.nan):
            with self.assertRaises(ValueError):
                select_shared_tokens([torch.ones(3)], ratio)


class GSSTests(unittest.TestCase):
    def test_convex_optimum_and_measured_midpoint(self):
        calls = []

        def objective(r):
            calls.append(r)
            return 0.12 + (r - 0.364) ** 2

        result = golden_section_search(objective, tc=2.0, ti=1.0, tolerance=1e-5)
        self.assertLess(abs(result.ratio - 0.364), 1e-5)
        self.assertEqual(result.ratio, sum(result.interval) / 2)
        self.assertEqual(result.value, 0.12 + (result.ratio - 0.364) ** 2)
        self.assertEqual(calls[-1], result.ratio)
        self.assertIn(result.ratio, result.cache)
        self.assertEqual(len(calls), len(set(calls)))
        self.assertLess(result.interval[1] - result.interval[0], 1e-5)

    def test_exact_warm_formula_and_reinitialized_golden_trace(self):
        phi = (math.sqrt(5) - 1) / 2
        for tc, ti in ((3., 1.), (1., 3.)):
            result = golden_section_search(lambda r: (r - .61) ** 2, tc, ti)
            warm = result.trace[0]
            self.assertEqual(warm.phase, 'warm')
            if result.prior <= .575:
                self.assertEqual(warm.x1, result.prior)
                self.assertEqual(warm.x2, .15 + phi * .85)
            else:
                self.assertEqual(warm.x1, 1 - phi * .85)
                self.assertEqual(warm.x2, result.prior)
            initial = result.trace[1]
            self.assertEqual(initial.phase, 'reinitialize')
            if warm.f1 <= warm.f2:
                self.assertEqual((initial.a, initial.b), (warm.a, warm.x2))
            else:
                self.assertEqual((initial.a, initial.b), (warm.x1, warm.b))
            for entry in result.trace[1:]:
                self.assertLess(entry.a, entry.x1)
                self.assertLess(entry.x1, entry.x2)
                self.assertLess(entry.x2, entry.b)
                self.assertAlmostEqual(entry.x1, entry.b - phi * (entry.b - entry.a))
                self.assertAlmostEqual(entry.x2, entry.a + phi * (entry.b - entry.a))
            for before, after in zip(result.trace[1:], result.trace[2:]):
                self.assertAlmostEqual(after.b - after.a, phi * (before.b - before.a))
                self.assertGreaterEqual(after.a, before.a)
                self.assertLessEqual(after.b, before.b)
                reused = set((before.x1, before.x2)) & set((after.x1, after.x2))
                self.assertEqual(len(reused), 1)

    def test_spectral_tie_comparisons_differ_in_warm_and_standard_steps(self):
        result = golden_section_search(lambda r: 1.0, 1., 1., tolerance=.1)
        warm, standard, first_refine = result.trace[:3]
        self.assertEqual(standard.b, warm.x2)  # <= keeps left during warm start.
        self.assertEqual(first_refine.a, standard.x1)  # < sends a tie right.

    def test_clipped_prior_boundary_optima_and_equal_bounds(self):
        low = golden_section_search(lambda r: r, tc=1000, ti=1, tolerance=1e-6)
        high = golden_section_search(lambda r: -r, tc=1, ti=1000,
                                     r_max=.8, tolerance=1e-6)
        self.assertEqual(low.prior, .15)
        self.assertEqual(high.prior, .8)
        self.assertLess(abs(low.ratio - .15), 1e-6)
        self.assertLess(abs(high.ratio - .8), 1e-6)
        point = golden_section_search(lambda r: r + 2, 1., 1., r_min=.4, r_max=.4)
        self.assertEqual(point.ratio, .4)
        self.assertEqual(point.value, 2.4)
        self.assertEqual(len(point.evaluations), 1)
        self.assertEqual(point.iterations, 0)

    def test_large_finite_costs_do_not_overflow_prior(self):
        result = golden_section_search(lambda r: (r - .5)**2, 1e308, 1e308)
        self.assertEqual(result.prior, .5)

    def test_invalid_parameters_and_objectives(self):
        for params in ({'tc': 0}, {'ti': -1}, {'tc': math.nan}, {'ti': math.inf},
                       {'r_min': -.1}, {'r_max': 1.1}, {'r_min': .9, 'r_max': .5},
                       {'tolerance': 0}, {'tolerance': math.nan}, {'tolerance': 1e-30}):
            with self.subTest(params=params), self.assertRaises(ValueError):
                args = dict(tc=1., ti=1.)
                args.update(params)
                golden_section_search(lambda r: r, **args)
        for invalid in (math.nan, math.inf, -math.inf, 'bad', True):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                golden_section_search(lambda r: invalid, 1., 1.)


if __name__ == '__main__':
    unittest.main()
