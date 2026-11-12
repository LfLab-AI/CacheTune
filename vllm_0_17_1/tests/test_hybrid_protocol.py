"""CPU protocol checks; tensor-only worker checks additionally require PyTorch."""
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "example"))
import blend_samsum_freq as runner


QWEN35_CONFIG = {"model_type": "qwen3_5",
                 "architectures": ["Qwen3_5ForConditionalGeneration"],
                 "text_config": {"model_type": "qwen3_5_text"}}


class ProtocolLLM:
    def __init__(self):
        self.events = []

    def collective_rpc(self, method, args, kwargs):
        self.events.append((method, kwargs))
        if method == "cachetune_prepare":
            self.prepare_kwargs = kwargs
            return [{"selected_indices": [0, 3, 5, 6], "reused_indices": [1, 2, 4]}]
        if method == "cachetune_diagnostics":
            return [{}]
        return [None]

    @staticmethod
    def generate(llm, ids, max_tokens):
        return {"token_ids": [9], "text": "summary"}


class AnswerExtractionTests(unittest.TestCase):
    def test_leading_reasoning_and_unclosed_blocks(self):
        cases = (
            ("<think></think>\n\nAnswer.\nSecond.", "Answer."),
            (" \n<think>reason\nmore</think>\n Answer.\nSecond.", "Answer."),
            ("<think>reasoning without end", ""),
            ("<think></think>", ""),
            ("Answer mentions <think>literally</think>.", "Answer mentions <think>literally</think>."),
            ("</think>\nAnswer.", "</think>"),
            ("\nAnswer.\nSecond.", "Answer."),
        )
        for text, expected in cases:
            with self.subTest(text=text):
                self.assertEqual(runner.extract_qwen35_answer(text), expected)

    def test_runtime_detection_and_generation_preserve_raw_output(self):
        class FakeLLM:
            def __init__(self, model_config):
                self.llm_engine = SimpleNamespace(model_config=model_config)

            def generate(self, prompts, sampling_params, use_tqdm):
                completion = SimpleNamespace(token_ids=[7, 8], text="<think>reason</think>\nAnswer.",
                                             finish_reason="stop")
                return [SimpleNamespace(prompt_token_ids=prompts[0]["prompt_token_ids"],
                                        outputs=[completion], metrics=None)]

        module = ModuleType("vllm")
        module.SamplingParams = lambda **kwargs: kwargs
        configurations = (
            SimpleNamespace(hf_text_config=SimpleNamespace(model_type="qwen3_5_text")),
            SimpleNamespace(hf_config=SimpleNamespace(
                text_config=SimpleNamespace(model_type="qwen3_5_text"))),
            SimpleNamespace(hf_config={"text_config": {"model_type": "qwen3_5_text"}}),
        )
        with patch.dict(sys.modules, {"vllm": module}):
            for config in configurations:
                with self.subTest(config=config):
                    # generate_ids is shared by both native Full and Cached execution.
                    full = runner.generate_ids(FakeLLM(config), [1, 2], 8)
                    cached = runner.generate_ids(FakeLLM(config), [1, 2], 8)
                    self.assertEqual(full["scored_text"], "Answer.")
                    self.assertEqual(full["scored_text"], cached["scored_text"])
                    self.assertEqual(full["text"], "<think>reason</think>\nAnswer.")
                    self.assertEqual(full["token_ids"], [7, 8])
                    self.assertEqual(full["answer_extraction"]["policy"], runner.QWEN35_ANSWER_EXTRACTION)
            dense = runner.generate_ids(FakeLLM(SimpleNamespace(
                hf_text_config=SimpleNamespace(model_type="qwen3"))), [1, 2], 8)
            self.assertEqual(dense["scored_text"], "<think>reason</think>")
            self.assertNotIn("answer_extraction", dense)
        self.assertFalse(runner.is_qwen35_runtime(SimpleNamespace()))


class HybridRunnerTests(unittest.TestCase):
    def test_dense_qwen35_preserves_conditional_architecture(self):
        metadata = runner.validate_model_mode(QWEN35_CONFIG, False)
        self.assertTrue(metadata["is_qwen35"])
        self.assertFalse(metadata["is_moe"])
        self.assertEqual(metadata["architectures"], ["Qwen3_5ForConditionalGeneration"])
        causal = {"model_type": "qwen3_5_text", "architectures": ["Qwen3_5ForCausalLM"]}
        self.assertTrue(runner.validate_model_mode(causal, False)["is_qwen35"])

    def test_moe_rejected_for_cached_but_native_standard_remains_available(self):
        variants = [
            {"model_type": "qwen3_5_moe", "text_config": {"model_type": "qwen3_5_moe_text"}},
            {"text_config": {"model_type": "qwen3_5_moe_text"}},
            {**QWEN35_CONFIG, "text_config": {"model_type": "qwen3_5_text", "num_experts": 256}},
        ]
        for config in variants:
            with self.subTest(config=config):
                with self.assertRaisesRegex(ValueError, "MoE.*standard-only"):
                    runner.validate_model_mode(config, False)
                self.assertTrue(runner.validate_model_mode(config, True)["is_qwen35"])

    def test_unknown_qwen35_variant_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported Qwen3.5 variant"):
            runner.validate_model_mode({"model_type": "qwen3_5_unknown"}, False)

    def test_model_specific_defaults_are_config_driven(self):
        for config, check, policy in ((QWEN35_CONFIG, 0, "block"),
                                      ({"model_type": "qwen3"}, 1, None),
                                      ({"model_type": "mistral"}, 1, None)):
            with self.subTest(config=config):
                args = runner.parse_args([])
                self.assertIsNone(args.check_layer)
                metadata = runner.validate_model_mode(config, False)
                self.assertEqual(runner.resolve_model_options(args, metadata), policy)
                self.assertEqual(args.check_layer, check)
        args = runner.parse_args(["--check-layer", "2", "--hybrid-policy", "attention"])
        runner.resolve_model_options(args, runner.validate_model_mode(QWEN35_CONFIG, False))
        self.assertEqual(args.check_layer, 2)

    def test_hybrid_policy_on_dense_model_is_not_silently_ignored(self):
        args = runner.parse_args(["--hybrid-policy", "mlp"])
        with self.assertRaisesRegex(ValueError, "only to Qwen3.5"):
            runner.resolve_model_options(args, runner.validate_model_mode({"model_type": "qwen3"}, False))

    def test_engine_options_enforce_text_only_without_architecture_override(self):
        args = runner.parse_args([])
        metadata = runner.validate_model_mode(QWEN35_CONFIG, False)
        options = runner.build_engine_options(args, metadata)
        self.assertTrue(options["language_model_only"])
        self.assertEqual(options["limit_mm_per_prompt"], {"image": 0, "video": 0})
        self.assertFalse(options["enable_mm_embeds"])
        self.assertNotIn("hf_overrides", options)
        self.assertEqual(options["attention_config"], {"backend": "FLASH_ATTN"})
        self.assertEqual(options["worker_extension_cls"], runner.WORKER_EXTENSION)

    def test_standard_engine_and_dense_engine_remain_separate(self):
        for config in (QWEN35_CONFIG, {"model_type": "qwen3"}):
            args = runner.parse_args(["--standard-only"])
            metadata = runner.validate_model_mode(config, True)
            self.assertIsNone(runner.resolve_model_options(args, metadata))
            options = runner.build_engine_options(args, metadata)
            self.assertTrue(options["language_model_only"])
            self.assertNotIn("worker_extension_cls", options)
            self.assertNotIn("attention_config", options)
            self.assertNotIn("hf_overrides", options)
        options = runner.build_engine_options(runner.parse_args([]), {"is_qwen35": False})
        self.assertNotIn("language_model_only", options)
        self.assertNotIn("limit_mm_per_prompt", options)

    def test_hybrid_policies_reach_prepare_and_selection_report(self):
        tokens = {"chunks": [[1, 10, 11, 12, 13], [1, 14, 15]],
                  "prompt_ids": [1, 10, 11, 12, 13, 14, 15], "suffix_len": 2}
        for policy in (None, "attention", "block", "mlp"):
            with self.subTest(policy=policy):
                llm = ProtocolLLM()
                selection, *_ = runner.run_cached_pair(
                    llm, tokens, 0.4, 2, generate=ProtocolLLM.generate,
                    check_layer=0 if policy else 1,
                    selected_context_indices=[3, 0], hybrid_policy=policy)
                if policy is None:
                    self.assertNotIn("hybrid_policy", llm.prepare_kwargs)
                    self.assertNotIn("hybrid_policy", selection)
                else:
                    self.assertEqual(llm.prepare_kwargs["hybrid_policy"], policy)
                    self.assertEqual(selection["hybrid_policy"], policy)
                    self.assertEqual(selection["check_layers"], [0])
                self.assertEqual(llm.prepare_kwargs["selected_context_indices"], [3, 0])
                self.assertEqual(llm.events[-2][0], "cachetune_disable")

    def test_worker_cannot_mislabel_hybrid_policy(self):
        selection = {"selected_indices": [0, 3, 5, 6], "reused_indices": [1, 2, 4],
                     "hybrid_policy": "attention"}
        with self.assertRaisesRegex(AssertionError, "hybrid_policy"):
            runner.validate_selection(selection, 7, 2, 0.4, hybrid_policy="block")

    def test_lazy_registration_is_explicit_opt_in(self):
        from cachetune import register_models
        registered = {}
        fake_vllm = ModuleType("vllm")
        fake_vllm.ModelRegistry = SimpleNamespace(register_model=lambda name, path: registered.update({name: path}))
        with patch.dict(sys.modules, {"vllm": fake_vllm}):
            with patch.dict(os.environ, {"CACHETUNE_ENABLE_ADAPTER": "0"}):
                register_models()
                self.assertEqual(registered, {})
            with patch.dict(os.environ, {"CACHETUNE_ENABLE_ADAPTER": "1"}):
                register_models()
        self.assertEqual(len(registered), 6)
        self.assertEqual(registered["Qwen2ForCausalLM"], "cachetune.models:CacheTuneQwen2ForCausalLM")
        for name in ("Qwen3_5ForCausalLM", "Qwen3_5ForConditionalGeneration"):
            self.assertEqual(registered[name], f"cachetune.hybrid:CacheTune{name}")
        self.assertEqual(registered["Qwen3ForCausalLM"], "cachetune.models:CacheTuneQwen3ForCausalLM")


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "PyTorch required for worker tensor checks")
class HybridWorkerTests(unittest.TestCase):
    def setUp(self):
        import torch
        from cachetune.state import CacheTuneState
        from cachetune.worker import CacheTuneWorkerExtension
        self.torch = torch
        self.state = CacheTuneState(2)
        self.model = SimpleNamespace(_cachetune_state=self.state,
                                     layers=[SimpleNamespace(layer_type="linear_attention"),
                                             SimpleNamespace(layer_type="full_attention")])
        self.worker = CacheTuneWorkerExtension()
        self.worker.model_runner = SimpleNamespace(model=SimpleNamespace(
            language_model=SimpleNamespace(model=self.model)))

    def snapshot(self, delta=0.0):
        return {0: {"conv": self.torch.tensor([[1.0, 2.0 + delta]]),
                    "ssm": self.torch.tensor([[[3.0, 4.0]]])}}

    def complete(self, snapshot):
        self.state._recurrent_snapshot = snapshot
        self.state._recurrent_probe_next = False

    def test_conditional_and_causal_unwrap_target_the_same_state(self):
        self.assertIs(self.worker._cachetune_get_model(), self.model)
        self.assertIs(self.worker._cachetune_get_state(), self.state)
        self.worker.model_runner.model = SimpleNamespace(model=self.model)
        self.assertIs(self.worker._cachetune_get_state(), self.state)

    def test_hybrid_debug_is_rejected_without_arming(self):
        with self.assertRaisesRegex(NotImplementedError, "Hybrid"):
            self.worker.cachetune_debug_start(raw_qkv=True)
        self.assertFalse(getattr(self.model, "_cachetune_debug_armed", False))

    def test_incomplete_probe_cannot_return_stale_snapshot(self):
        self.complete(self.snapshot())
        self.worker.cachetune_recurrent_probe_start()
        self.assertIsNone(self.state._recurrent_snapshot)
        with self.assertRaisesRegex(RuntimeError, "No completed"):
            self.worker.cachetune_recurrent_probe_finish(set_reference=True)

    def test_recurrent_reference_is_immutable_and_comparison_is_json(self):
        snapshot = self.snapshot()
        self.complete(snapshot)
        result = self.worker.cachetune_recurrent_probe_finish(set_reference=True)
        self.assertTrue(result["finite"])
        self.assertTrue(result["reference_saved"])
        snapshot[0]["conv"][0, 0] = 99
        self.assertEqual(float(self.state._recurrent_reference[0]["conv"][0, 0]), 1.0)
        self.complete(self.snapshot())
        result = self.worker.cachetune_recurrent_probe_finish()
        self.assertTrue(result["all_equal"])
        self.assertEqual(result["layers"]["0"]["conv"]["relative_l2"], 0.0)
        self.complete(self.snapshot(delta=0.5))
        result = self.worker.cachetune_recurrent_probe_finish()
        self.assertFalse(result["all_equal"])
        self.assertEqual(result["layers"]["0"]["conv"]["max_absolute_error"], 0.5)
        self.assertTrue(result["layers"]["0"]["ssm"]["equal"])
        json.dumps(result, allow_nan=False)

    def test_nonfinite_values_are_explicit_and_json_safe(self):
        self.complete(self.snapshot())
        self.worker.cachetune_recurrent_probe_finish(set_reference=True)
        current = self.snapshot()
        current[0]["ssm"][0, 0, 0] = float("nan")
        self.complete(current)
        result = self.worker.cachetune_recurrent_probe_finish()
        self.assertFalse(result["finite"])
        self.assertIsNone(result["layers"]["0"]["ssm"]["relative_l2"])
        json.dumps(result, allow_nan=False)

    def test_missing_reference_or_layer_shape_is_rejected(self):
        self.complete(self.snapshot())
        with self.assertRaisesRegex(RuntimeError, "matching recurrent-state"):
            self.worker.cachetune_recurrent_probe_finish()
        self.worker.cachetune_recurrent_probe_finish(set_reference=True)
        current = self.snapshot()
        current[0]["conv"] = current[0]["conv"][:, :1]
        self.complete(current)
        with self.assertRaisesRegex(RuntimeError, "shape/dtype mismatch"):
            self.worker.cachetune_recurrent_probe_finish()

    def test_dense_model_recurrent_probe_is_rejected(self):
        self.model.layers = [SimpleNamespace(), SimpleNamespace()]
        with self.assertRaisesRegex(ValueError, "hybrid model"):
            self.worker.cachetune_recurrent_probe_start()


if __name__ == "__main__":
    unittest.main()
