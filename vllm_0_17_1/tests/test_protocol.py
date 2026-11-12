"""CPU-only checks for the unchanged prompt contract and worker lifecycle."""

from pathlib import Path
import json
import sys
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "example"))
import blend_samsum_freq as runner
from utils import build_fewshot_prompt, scored_first_line


class Tokenizer:
    bos_token_id = 1
    eos_token_id = 2

    def encode(self, text, add_special_tokens):
        if add_special_tokens:
            raise AssertionError("Protocol must disable implicit special tokens")
        return [ord(char) + 10 for char in text]

    def convert_ids_to_tokens(self, token_id):
        return f"special-{token_id}"


class FakeWorkerLLM:
    def __init__(self):
        self.events = []
        self.mode = "full"
        self.parts = []
        self.last_prompt = None

    def collective_rpc(self, method, args, kwargs):
        self.events.append(method)
        if method == "cachetune_reset":
            self.mode = "full"
            self.parts = []
        elif method == "cachetune_start_collect":
            self.mode = "collect"
        elif method == "cachetune_append_chunk":
            assert self.mode == "collect"
            assert kwargs["expected_ids"] == self.last_prompt
            self.parts.append(self.last_prompt[int(kwargs["drop_bos"]):])
        elif method == "cachetune_prepare":
            self.prepare_kwargs = dict(kwargs)
            prompt = kwargs["prompt_ids"]
            assert [t for part in self.parts for t in part] == prompt[:len(prompt) - kwargs["suffix_len"]]
            context = len(prompt) - kwargs["suffix_len"]
            lengths = [len(part) for part in self.parts]
            budget = sum(int(n * kwargs["ratio"]) for n in lengths)
            context_indices = kwargs.get("selected_context_indices")
            selected = (list(range(budget)) if context_indices is None else sorted(context_indices)) + list(range(context, len(prompt)))
            return [{"selected_indices": selected,
                     "reused_indices": [i for i in range(context) if i not in selected],
                     "score_layer": None,
                     "chunk_context_lengths": lengths,
                     "signal": "K/V-all-layers" if context_indices is None else "External-indices",
                     "selection_source": "frequency" if context_indices is None else "external_context_indices"}]
        elif method == "cachetune_enable":
            self.mode = "cached"
        elif method == "cachetune_disable":
            self.mode = "full"
        elif method == "cachetune_diagnostics":
            return [{"mode": self.mode}]
        return [None]

    @staticmethod
    def generate(llm, ids, max_tokens):
        llm.last_prompt = list(ids)
        llm.events.append(("generate", llm.mode, max_tokens))
        return {"token_ids": [7], "text": "summary", "scored_text": "summary"}


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.example = {"ctxs": [{"text": "one"}, {"text": "two"}],
                        "question": "Dialogue\nSummary:", "answers": ["answer"]}

    def test_original_prompt_and_chunk_boundary_match(self):
        tokenizer = Tokenizer()
        tokens = runner.build_sample_tokens(self.example, tokenizer)
        expected = [1] + [ord(c) + 10 for c in runner.PREFIX + "onetwo\n\nDialogue\nSummary:"]
        self.assertEqual(tokens["prompt_ids"], expected)
        self.assertEqual(tokens["chunks"][0] + [t for c in tokens["chunks"][1:] for t in c[1:]], expected)
        self.assertFalse(tokens["tokenization"]["eos_fallback_used"])

    def test_qwen_boundary_fallback_keeps_all_content_tokens(self):
        tokenizer = Tokenizer()
        tokenizer.bos_token_id = None
        tokens = runner.build_sample_tokens(self.example, tokenizer)
        self.assertEqual(tokens["prompt_ids"][0], 2)
        self.assertTrue(tokens["tokenization"]["eos_fallback_used"])
        self.assertEqual(tokens["chunks"][1], [2] + tokenizer.encode("one", False))
        self.assertEqual(tokens["chunks"][2], [2] + tokenizer.encode("two", False))

    def test_missing_boundary_is_rejected(self):
        tokenizer = Tokenizer()
        tokenizer.bos_token_id = tokenizer.eos_token_id = None
        with self.assertRaises(ValueError):
            runner.boundary_token(tokenizer)

    def test_qwen35_raw_config_guard_does_not_require_autoconfig(self):
        config = {"model_type": "qwen3_5", "architectures": ["Qwen3_5ForConditionalGeneration"],
                  "text_config": {"model_type": "qwen3_5_text"}}
        with TemporaryDirectory() as folder:
            (Path(folder) / "config.json").write_text(json.dumps(config), encoding="utf-8")
            loaded = runner.read_model_config(folder)
        self.assertTrue(runner.validate_model_mode(loaded, False)["is_qwen35"])
        self.assertTrue(runner.validate_model_mode(loaded, True)["is_qwen35"])
        self.assertFalse(runner.validate_model_mode({"model_type": "qwen3"}, False)["is_qwen35"])
        with self.assertRaisesRegex(ValueError, "MoE"):
            runner.validate_model_mode({"text_config": {"model_type": "qwen3_5_moe_text"}}, False)

    def test_sample_indices_preserve_original_ids_and_aba_order(self):
        selected = runner.plan_samples(["A", "B", "C"], sample_indices=runner.parse_sample_indices("0, 1,0"))
        self.assertEqual(selected, [(0, "A"), (1, "B"), (0, "A")])
        self.assertEqual(runner.plan_samples(["A", "B", "C"], sample_limit=2), [(0, "A"), (1, "B")])
        with self.assertRaisesRegex(ValueError, "out of range"):
            runner.plan_samples(["A"], sample_indices=[1])
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            runner.plan_samples(["A"], sample_limit=1, sample_indices=[0])

    def test_pruning_drops_middle_documents_without_truncating_them(self):
        example = {"ctxs": [{"text": "aaa"}, {"text": "bbbb"}, {"text": "ccccc"}],
                   "question": "query"}
        tokens = runner.build_sample_tokens(example, Tokenizer(), context_budget=7)
        self.assertEqual(tokens["retained_document_count"], 1)
        self.assertEqual(tokens["chunks"][1][1:], Tokenizer().encode("aaa", False))
        self.assertEqual(tokens["suffix_len"], len("\n\nquery"))

    def test_worker_lifecycle_preserves_full_baseline(self):
        tokens = runner.build_sample_tokens(self.example, Tokenizer())
        llm = FakeWorkerLLM()
        indices, cached, full, repeat, diagnostics = runner.run_cached_pair(
            llm, tokens, 0.25, 128, True, FakeWorkerLLM.generate)
        self.assertEqual(cached, repeat)
        self.assertEqual(cached, full)
        self.assertEqual(diagnostics["full"]["mode"], "full")
        generations = [event for event in llm.events if isinstance(event, tuple)]
        self.assertEqual(generations[-3:], [("generate", "cached", 128),
                                          ("generate", "cached", 128),
                                          ("generate", "full", 128)])
        self.assertTrue(all(e == ("generate", "collect", 1) for e in generations[:-3]))
        self.assertEqual(indices["selected_context_tokens"], sum(int(n * 0.25) for n in indices["chunk_context_lengths"]))
        self.assertEqual({key: llm.prepare_kwargs[key] for key in
                          ("alpha", "sink_size", "check_layer", "cache_mode", "selected_context_indices")},
                         {"alpha": 0.5, "sink_size": 0, "check_layer": 1,
                          "cache_mode": "cpu_pipeline", "selected_context_indices": None})

    def test_nondefault_prepare_parameters_and_external_indices_reach_worker(self):
        tokens = {"chunks": [[1, 10, 11, 12, 13], [1, 14, 15]],
                  "prompt_ids": [1, 10, 11, 12, 13, 14, 15], "suffix_len": 2}
        for mode in ("cpu_pipeline", "cpu_serial", "gpu"):
            with self.subTest(mode=mode):
                llm = FakeWorkerLLM()
                indices, *_ = runner.run_cached_pair(
                    llm, tokens, 0.4, 2, generate=FakeWorkerLLM.generate,
                    alpha=0.25, sink_size=1, check_layer=3, cache_mode=mode,
                    selected_context_indices=[3, 0])
                self.assertEqual(llm.prepare_kwargs["selected_context_indices"], [3, 0])
                self.assertEqual(llm.prepare_kwargs["alpha"], 0.25)
                self.assertEqual(llm.prepare_kwargs["sink_size"], 1)
                self.assertEqual(llm.prepare_kwargs["check_layer"], 3)
                self.assertEqual(llm.prepare_kwargs["cache_mode"], mode)
                self.assertEqual(indices["selected_indices"], [0, 3, 5, 6])
                self.assertEqual(indices["reused_indices"], [1, 2, 4])
                self.assertEqual(indices["check_layers"], [3])

    def test_selection_contract_rejects_missing_sink_wrong_override_and_budget(self):
        valid = {"selected_indices": [0, 3, 5, 6], "reused_indices": [1, 2, 4]}
        no_sink = {"selected_indices": [1, 3, 5, 6], "reused_indices": [0, 2, 4]}
        with self.assertRaisesRegex(AssertionError, "forced sink"):
            runner.validate_selection(no_sink, 7, 2, 0.4, sink_size=1)
        with self.assertRaisesRegex(AssertionError, "explicit context"):
            runner.validate_selection(valid, 7, 2, 0.4, selected_context_indices=[0, 2])
        with self.assertRaisesRegex(AssertionError, "budget"):
            runner.validate_selection(valid, 7, 2, 0.6)
        with self.assertRaisesRegex(AssertionError, "check_layers"):
            runner.validate_selection({**valid, "check_layers": [1]}, 7, 2, 0.4, check_layer=3)

    def test_external_selection_metadata_matches_worker_and_rejects_frequency_labels(self):
        info = {"selected_indices": [0, 3, 5, 6], "reused_indices": [1, 2, 4],
                "score_layer": None, "signal": "External-indices",
                "selection_source": "external_context_indices"}
        result = runner.validate_selection(info, 7, 2, 0.4, selected_context_indices=[3, 0])
        self.assertIsNone(result["score_layer"])
        self.assertEqual(result["signal"], "External-indices")
        self.assertEqual(result["selection_source"], "external_context_indices")
        for key, wrong in (("score_layer", 0), ("signal", "Value-only"),
                           ("selection_source", "frequency")):
            with self.subTest(key=key), self.assertRaisesRegex(AssertionError, key):
                runner.validate_selection({**info, key: wrong}, 7, 2, 0.4,
                                          selected_context_indices=[3, 0])

    def test_indices_json_preserves_original_index_mapping_and_rejects_missing_keys(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "indices.json"
            path.write_text('{"0": [3, 0], "1": [2, 0]}', encoding="utf-8")
            self.assertEqual(runner.load_context_indices(path, [0, 1, 0]),
                             {0: [3, 0], 1: [2, 0]})
            with self.assertRaisesRegex(ValueError, "dataset index 2"):
                runner.load_context_indices(path, [2])
            for invalid in ('{"0": [true]}', '{"0": [2, 2]}', '{"00": [1]}',
                            '{"0": [1], "0": [2]}'):
                path.write_text(invalid, encoding="utf-8")
                with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                    runner.load_context_indices(path, [0])

    def test_extended_cli_preserves_selection_defaults_and_dtype_auto(self):
        defaults = runner.parse_args([])
        self.assertEqual((defaults.low_freq_pct, defaults.sink_size, defaults.check_layer,
                          defaults.cache_mode, defaults.indices_json, defaults.dtype),
                         (0.5, 0, None, "cpu_pipeline", None, "auto"))
        custom = runner.parse_args(["--low-freq-pct", "0.25", "--sink-size", "1",
                                    "--check-layer", "3", "--cache-mode", "gpu"])
        self.assertEqual((custom.low_freq_pct, custom.sink_size, custom.check_layer,
                          custom.cache_mode), (0.25, 1, 3, "gpu"))

    def test_zero_and_full_ratio_partition(self):
        tokens = runner.build_sample_tokens(self.example, Tokenizer())
        for ratio in (0, 1):
            indices, *_ = runner.run_cached_pair(FakeWorkerLLM(), tokens, ratio, 2, generate=FakeWorkerLLM.generate)
            self.assertEqual(indices["selected_context_tokens"], int(indices["context_tokens"] * ratio))
            if ratio == 1:
                self.assertEqual(indices["reused_indices"], [])

    def test_wrong_selection_suffix_rejected(self):
        with self.assertRaises(AssertionError):
            runner.validate_selection({"selected_indices": [0, 1], "reused_indices": [2, 3]}, 4, 1, 1 / 3)

    def test_multiworker_result_is_rejected(self):
        class MultiWorker:
            def collective_rpc(self, *args, **kwargs):
                return [None, None]
        with self.assertRaises(RuntimeError):
            runner.rpc(MultiWorker(), "cachetune_reset")

    def test_generation_sends_exact_token_ids_and_preserves_missing_metrics(self):
        fake_vllm = ModuleType("vllm")
        fake_vllm.SamplingParams = lambda **kwargs: SimpleNamespace(**kwargs)

        class TokenOnlyLLM:
            def generate(self, prompts, sampling_params, use_tqdm):
                self.prompts = prompts
                self.params = sampling_params
                self.use_tqdm = use_tqdm
                completion = SimpleNamespace(token_ids=[9], text="\nsummary\nextra", finish_reason="length")
                return [SimpleNamespace(prompt_token_ids=prompts[0]["prompt_token_ids"],
                                        outputs=[completion], metrics=None)]

        llm = TokenOnlyLLM()
        with patch.dict(sys.modules, {"vllm": fake_vllm}):
            result = runner.generate_ids(llm, [2, 41, 42], 128)
        self.assertEqual(llm.prompts, [{"prompt_token_ids": [2, 41, 42]}])
        self.assertEqual(llm.params.temperature, 0)
        self.assertEqual(llm.params.max_tokens, 128)
        self.assertFalse(llm.use_tqdm)
        self.assertEqual(result["scored_text"], "summary")
        self.assertIsNone(result["engine_ttft_seconds"])
        self.assertIsNone(result["engine_arrival_ttft_seconds"])

    def test_v1_timing_never_subtracts_wall_clock_from_monotonic(self):
        metrics = SimpleNamespace(first_token_ts=1234.75, scheduled_ts=1234.5,
                                  first_token_latency=0.6, arrival_time=1700000000.0)
        result = runner.request_timing(metrics)
        self.assertEqual(result["engine_ttft_seconds"], 0.25)
        self.assertEqual(result["engine_arrival_ttft_seconds"], 0.6)
        self.assertEqual(result["engine_timing_source"], "vllm_v1_request_state_stats")

    def test_legacy_timing_uses_legacy_fields_only(self):
        metrics = SimpleNamespace(first_token_time=1700000000.75,
                                  first_scheduled_time=1700000000.5,
                                  arrival_time=1700000000.0)
        result = runner.request_timing(metrics)
        self.assertEqual(result["engine_ttft_seconds"], 0.25)
        self.assertEqual(result["engine_arrival_ttft_seconds"], 0.75)
        self.assertEqual(result["engine_timing_source"], "vllm_legacy_request_metrics")

    def test_uninitialized_and_invalid_timing_remains_null(self):
        for metrics in (None,
                        SimpleNamespace(first_token_ts=0, scheduled_ts=0, first_token_latency=0),
                        SimpleNamespace(first_token_ts=3, scheduled_ts=4, first_token_latency=float("nan"))):
            with self.subTest(metrics=metrics):
                result = runner.request_timing(metrics)
                self.assertIsNone(result["engine_ttft_seconds"])
                self.assertIsNone(result["engine_arrival_ttft_seconds"])

    def test_runtime_version_guard_rejects_old_environment(self):
        for version in ("0.17.1", "0.17.1+cu129", "0.17.1.dev0+local"):
            runner.validate_runtime_version(version)
        for version in ("0.4.1", "0.17.10", "0.18.0"):
            with self.subTest(version=version), self.assertRaisesRegex(RuntimeError, "requires vLLM 0.17.1"):
                runner.validate_runtime_version(version)

    def test_fewshot_and_scoring_text_match_original(self):
        self.assertEqual(build_fewshot_prompt(self.example), (["one", "two"], "\n\nDialogue\nSummary:"))
        self.assertEqual(scored_first_line("\n\nFirst sentence\nSecond sentence"), "First sentence")
        self.assertEqual(scored_first_line("\n  \nignored"), "  ")

    def test_standard_only_summary_has_no_cached_result(self):
        summary = runner.summarize([{"full": {"rougeL": 0.5, "engine_ttft_seconds": None}, "cached": None}])
        self.assertEqual(summary["cached_samples"], 0)
        self.assertIsNone(summary["mean_cached_rougeL"])
        self.assertEqual(summary["mean_full_rougeL"], 0.5)

    def test_aba_revisit_checks_tokens_and_selection(self):
        def row(run_index, sample_index, cached_tokens, selected):
            return {"run_index": run_index, "sample_index": sample_index,
                    "cached": {"token_ids": cached_tokens}, "full": {"token_ids": [10]},
                    "indices": {"selected_indices": selected, "reused_indices": [0]}}

        seen = {}
        self.assertEqual(runner.record_revisit(row(0, 0, [9], [1, 2]), seen), [])
        self.assertEqual(runner.record_revisit(row(1, 1, [11], [2, 3]), seen), [])
        repeated = row(2, 0, [9], [1, 2])
        self.assertEqual(runner.record_revisit(repeated, seen), [])
        self.assertEqual(repeated["revisit_reference_run_index"], 0)
        self.assertTrue(repeated["revisit_cached_equals_reference"])
        changed = row(3, 0, [8], [2, 3])
        self.assertEqual(runner.record_revisit(changed, seen),
                         ["revisit_cached_equals_reference", "revisit_selection_equals_reference"])

    def test_standard_only_revisit_checks_full_tokens(self):
        seen = {}
        first = {"run_index": 0, "sample_index": 3, "cached": None, "full": {"token_ids": [7]}}
        changed = {"run_index": 1, "sample_index": 3, "cached": None, "full": {"token_ids": [8]}}
        self.assertEqual(runner.record_revisit(first, seen), [])
        self.assertEqual(runner.record_revisit(changed, seen), ["revisit_full_equals_reference"])
        self.assertNotIn("revisit_cached_equals_reference", changed)


if __name__ == "__main__":
    unittest.main()
