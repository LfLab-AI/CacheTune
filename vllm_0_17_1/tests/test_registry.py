"""Fresh-process plugin isolation checks; never instantiate or load a GPU model."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
ARCHITECTURES = ("LlamaForCausalLM", "MistralForCausalLM", "Qwen2ForCausalLM", "Qwen3ForCausalLM")
MARKER = "CACHETUNE_REGISTRY_JSON="


def run_probe(code, enabled):
    env = {**os.environ, "PYTHONPATH": str(ROOT), "CUDA_VISIBLE_DEVICES": "",
           "VLLM_PLUGINS": "cachetune"}
    if enabled is None:
        env.pop("CACHETUNE_ENABLE_ADAPTER", None)
    else:
        env["CACHETUNE_ENABLE_ADAPTER"] = enabled
    process = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env,
                             text=True, capture_output=True, timeout=60)
    if process.returncode:
        raise AssertionError(f"Registry probe exited {process.returncode}\n{process.stdout}\n{process.stderr}")
    lines = [line[len(MARKER):] for line in process.stdout.splitlines() if line.startswith(MARKER)]
    if len(lines) != 1:
        raise AssertionError(f"Missing registry probe result\n{process.stdout}\n{process.stderr}")
    return json.loads(lines[0])


class RegistrationGateTests(unittest.TestCase):
    def test_disabled_plugin_does_not_even_import_vllm(self):
        code = r'''
import builtins, json, sys
from cachetune import register_models
original_import = builtins.__import__
def checked_import(name, *args, **kwargs):
    if name == "vllm" or name.startswith("vllm."):
        raise AssertionError("Disabled CacheTune plugin imported vLLM")
    return original_import(name, *args, **kwargs)
builtins.__import__ = checked_import
register_models()
assert "cachetune.models" not in sys.modules
print("CACHETUNE_REGISTRY_JSON=" + json.dumps({"disabled": True}))
'''
        for enabled in (None, "0"):
            with self.subTest(enabled=enabled):
                self.assertEqual(run_probe(code, enabled), {"disabled": True})


@unittest.skipUnless(importlib.util.find_spec("vllm") is not None,
                     "Run these native registry probes in CacheBlend_new/.venv")
class NativeRegistryTests(unittest.TestCase):
    PROBE = r'''
import json, sys
import torch
from vllm import ModelRegistry
from vllm.plugins import load_general_plugins
load_general_plugins()
architectures = ("LlamaForCausalLM", "MistralForCausalLM", "Qwen2ForCausalLM", "Qwen3ForCausalLM")
result = {}
for architecture in architectures:
    descriptor = ModelRegistry.models[architecture]
    result[architecture] = descriptor.module_name + ":" + descriptor.class_name
assert "cachetune.models" not in sys.modules, "Plugin eagerly imported adapter models"
assert not torch.cuda.is_initialized(), "Registry probe initialized CUDA"
print("CACHETUNE_REGISTRY_JSON=" + json.dumps(result))
'''

    def assert_native(self, enabled):
        result = run_probe(self.PROBE, enabled)
        modules = {"LlamaForCausalLM": "llama", "MistralForCausalLM": "mistral", "Qwen2ForCausalLM": "qwen2",
                   "Qwen3ForCausalLM": "qwen3"}
        self.assertEqual(result, {architecture: f"vllm.model_executor.models.{modules[architecture]}:{architecture}"
                                  for architecture in ARCHITECTURES})

    def test_default_registry_is_native(self):
        self.assert_native(None)

    def test_disabled_registry_is_native(self):
        self.assert_native("0")

    def test_enabled_registry_uses_lazy_cachetune_adapters(self):
        self.assertEqual(run_probe(self.PROBE, "1"),
                         {architecture: f"cachetune.models:CacheTune{architecture}"
                          for architecture in ARCHITECTURES})


if __name__ == "__main__":
    unittest.main()
