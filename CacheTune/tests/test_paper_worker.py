"""Paper worker math and state regression tests (no vLLM/model weights needed)."""
import importlib.util
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch


MODULE_PATH = (Path(__file__).resolve().parents[1] / "vllm_blend" / "vllm"
               / "worker" / "cachetune_paper.py")
SPEC = importlib.util.spec_from_file_location("paper_worker_under_test", MODULE_PATH)
paper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(paper)


class SpectralTests(unittest.TestCase):
    def test_low_band_has_exact_floor_cutoff(self):
        t = torch.arange(8, dtype=torch.float32)
        signal = 3 + torch.cos(2 * math.pi * t / 8) + 4 * torch.cos(4 * math.pi * t / 8)
        # F=5, floor(.5*F)=2: retain DC and frequency 1, remove frequency 2.
        expected = (3 + torch.cos(2 * math.pi * t / 8)).square()
        torch.testing.assert_close(paper.low_band_squared_norm(signal[:, None]), expected,
                                   rtol=1e-5, atol=1e-5)

    def test_tp_squared_statistics_reconstruct_full_head_norm(self):
        torch.manual_seed(18)
        raw_k, raw_v = torch.randn(17, 4, 8), torch.randn(17, 4, 8)
        full = 0.5 * (paper.low_band_squared_norm(raw_k).sqrt()
                      + paper.low_band_squared_norm(raw_v).sqrt())
        shards_k = sum(paper.low_band_squared_norm(x) for x in raw_k.chunk(2, dim=1))
        shards_v = sum(paper.low_band_squared_norm(x) for x in raw_v.chunk(2, dim=1))
        merged = 0.5 * (shards_k.sqrt() + shards_v.sqrt())
        torch.testing.assert_close(full, merged)
        # Replicated KV heads contribute twice and must be divided before sqrt.
        torch.testing.assert_close((2 * shards_k / 2).sqrt(), shards_k.sqrt())

    def test_chunk_boundaries_and_short_chunk(self):
        first, second = torch.ones(8, 2), torch.full((8, 2), 10.0)
        separate = torch.cat([paper.low_band_squared_norm(first),
                              paper.low_band_squared_norm(second)])
        combined = paper.low_band_squared_norm(torch.cat([first, second]))
        self.assertFalse(torch.allclose(separate, combined))
        # T=1 gives F=1, floor(.5*F)=0, not an invented minimum of one bin.
        self.assertEqual(paper.low_band_squared_norm(torch.ones(1, 2)).item(), 0)


class FakeModel:
    def __init__(self):
        self.cache_fuse_metadata = {}
        self.layers = [SimpleNamespace(self_attn=SimpleNamespace(
            hack_kv=None, num_kv_heads=2, total_num_kv_heads=4, head_dim=2))
            for _ in range(3)]

    def _prefetch_layer(self, layer_idx):
        pass

    def _rebuild_old_kv(self, layer_idx):
        pass


class FakeWorker(paper.CacheTunePaperWorkerMixin):
    def __init__(self):
        self.rank = 0
        self.parallel_config = SimpleNamespace(tensor_parallel_size=2)
        self.model_runner = SimpleNamespace(model=SimpleNamespace(model=FakeModel()))


@unittest.skipUnless(torch.cuda.is_available(), "state installation needs CUDA")
class WorkerStateTests(unittest.TestCase):
    def setUp(self):
        self.worker = FakeWorker()

    def tearDown(self):
        self.worker.cachetune_paper_release()

    def make_context(self, context_id="one", offset=0):
        self.worker.cachetune_paper_begin(context_id)
        for i, layer in enumerate(self.worker._paper_model().layers):
            k = torch.arange(28, dtype=torch.float32, device="cuda").reshape(7, 4)
            layer.self_attn.hack_kv = [k + offset + i, k * 2 + offset + i]
        stats = self.worker.cachetune_paper_append_chunk(1, 7)
        self.assertEqual(tuple(stats["k_sq"].shape), (3, 6))
        self.assertEqual(stats["replication_factor"], [1, 1, 1])
        info = self.worker.cachetune_paper_finalize(context_id, last_len=2)
        self.assertEqual(info["total_len"], 8)
        self.assertEqual(info["chunk_lengths"], [6])
        return self.worker._paper_contexts[context_id]["full_kv"]

    def prepare(self, indices, context_id="one", **kwargs):
        return self.worker.cachetune_paper_prepare(
            torch.tensor(indices), last_len=2, recomp_ratio=0.5,
            context_id=context_id, **kwargs)

    def test_suffix_and_immutable_snapshots_across_probes_and_contexts(self):
        source = self.make_context()
        expected = source[0][0].clone()
        self.assertEqual(source[0][0][-2:].abs().sum().item(), 0)
        self.prepare([0, 2, 6, 7])
        meta = self.worker._paper_model().cache_fuse_metadata
        self.assertTrue(meta["paper_causal_mask"])
        torch.testing.assert_close(meta["cpu_kv_cache"][0][0], expected[[1, 3, 4, 5]])
        self.prepare([1, 3, 6, 7])
        torch.testing.assert_close(meta["cpu_kv_cache"][0][0], expected[[0, 2, 4, 5]])
        torch.testing.assert_close(source[0][0], expected)
        self.make_context("two", offset=100)
        self.prepare([0, 2, 6, 7], "one")
        torch.testing.assert_close(meta["cpu_kv_cache"][0][0], expected[[1, 3, 4, 5]])
        self.assertEqual(self.prepare(list(range(8)))["transfer_tokens"], 0)
        self.assertFalse(meta["check"])
        self.assertFalse(meta["paper_causal_mask"])

    def test_bad_indices_are_rejected(self):
        self.make_context()
        for indices in ([0, 2, 6], [0, 2, 7], [0, 2, 6, 6, 7], [2, 0, 6, 7], [-1, 6, 7], [6, 7, 8]):
            with self.assertRaises(ValueError):
                self.prepare(indices)

    def test_disk_sparse_payload_and_cpu_switch(self):
        source = self.make_context()
        with tempfile.TemporaryDirectory() as tmp:
            self.prepare([0, 2, 6, 7], storage="disk", disk_root=tmp)
            meta = self.worker._paper_model().cache_fuse_metadata
            path = meta["paper_disk_paths"][0][0]
            torch.testing.assert_close(paper._load_tensor(path), source[0][0][[1, 3, 4, 5]])
            self.assertTrue(meta["paper_disk_enabled"])
            self.prepare([1, 3, 6, 7], storage="cpu")
            self.assertFalse(meta["paper_disk_enabled"])
            self.worker.cachetune_paper_release("one")
            self.assertFalse(Path(path).exists())

    def test_transfer_profile_reports_one_layer_without_suffix(self):
        self.make_context()
        with tempfile.TemporaryDirectory() as tmp:
            for storage in ("cpu", "disk"):
                profile = self.worker.cachetune_paper_profile_transfer(
                    "one", storage=storage, disk_root=tmp, trials=2)
                self.assertEqual(profile["bytes"], 6 * 4 * 4 * 2)
                self.assertEqual(profile["context_tokens"], 6)
                self.assertEqual(profile["measured_layers"], 1)
                self.assertGreater(profile["bytes_per_second"], 0)
                self.assertEqual(profile["seconds_per_token_layer"], profile["seconds"] / 6)
                self.assertEqual(list(Path(tmp).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
