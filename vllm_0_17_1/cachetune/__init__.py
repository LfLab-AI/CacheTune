"""CacheTune's single-request adapter for vLLM 0.17.1."""

import os

__version__ = "0.2.0"


def register_models():
    """Opt in per process; ordinary vLLM keeps its native architecture registry."""
    if os.environ.get("CACHETUNE_ENABLE_ADAPTER", "0") != "1":
        return
    # Register by import path so plugin discovery never loads model classes.
    from vllm import ModelRegistry
    for architecture in ("LlamaForCausalLM", "MistralForCausalLM", "Qwen2ForCausalLM", "Qwen3ForCausalLM"):
        ModelRegistry.register_model(architecture, f"cachetune.models:CacheTune{architecture}")
    for architecture in ("Qwen3_5ForConditionalGeneration", "Qwen3_5ForCausalLM"):
        ModelRegistry.register_model(architecture, f"cachetune.hybrid:CacheTune{architecture}")

