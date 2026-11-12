"""Single-request SAMSum protocol on vLLM 0.17.1 worker RPCs.

CacheTune uses chunk-local K/V spectral scores, normalized within each layer
and averaged over every KV layer, alpha=0.5, with no forced sink tokens.
Dense models preserve the published check_layers=[1] pipeline boundary. Qwen3.5 uses logical
full-attention layer 0 and the block hybrid policy. --standard-only performs ordinary
inference using the same tokenization/scoring protocol, with no KV reuse.
"""

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

# Absolute .venv/bin/python invocation must also expose its JIT build tools.
os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")

if __package__:
    from .utils import build_fewshot_prompt, compute_rl, load_dataset, scored_first_line
else:
    from utils import build_fewshot_prompt, compute_rl, load_dataset, scored_first_line


PREFIX = "Summarize the dialogue into a few short sentences. The following are some examples.\n\n"
WORKER_EXTENSION = "cachetune.worker.CacheTuneWorkerExtension"


def read_model_config(model):
    """Read raw config without requiring Transformers to register its model type."""
    model_path = Path(model).expanduser()
    if model_path.is_dir():
        config_path = model_path / "config.json"
    else:
        from transformers.utils.hub import cached_file

        config_path = Path(cached_file(model, "config.json"))
    with config_path.open(encoding="utf-8") as stream:
        config = json.load(stream)
    if not isinstance(config, dict):
        raise ValueError("The model config.json must contain a JSON object")
    return config


def validate_model_mode(config, standard_only):
    configs = [config]
    if isinstance(config.get("text_config"), dict):
        configs.append(config["text_config"])
    names = [str(c.get("model_type", "")) for c in configs]
    names.extend(str(name) for c in configs for name in (c.get("architectures") or []))
    is_qwen35 = any("qwen3_5" in name.lower().replace(".", "_") for name in names)
    is_moe = any("moe" in name.lower() for name in names) or any(
        isinstance(c.get("num_experts"), int) and c["num_experts"] > 0 for c in configs)
    if is_qwen35 and not standard_only:
        if is_moe:
            raise ValueError("Qwen3.5 MoE requires --standard-only; hybrid CacheTune supports dense text models only")
        text_config = configs[-1]
        if text_config.get("model_type") not in ("qwen3_5", "qwen3_5_text"):
            raise ValueError("Unsupported Qwen3.5 variant; hybrid CacheTune requires a dense qwen3_5_text model")
    return {"model_type": config.get("model_type"),
            "text_model_type": config.get("text_config", {}).get("model_type")
            if isinstance(config.get("text_config"), dict) else None,
            "architectures": config.get("architectures", []), "is_qwen35": is_qwen35,
            "is_moe": is_moe}


def resolve_model_options(args, model_metadata):
    """Resolve layer coordinates after reading config, without guessing from a path."""
    if args.check_layer is None:
        args.check_layer = 0 if model_metadata["is_qwen35"] else 1
    if not model_metadata["is_qwen35"] and args.hybrid_policy != "block":
        raise ValueError("--hybrid-policy applies only to Qwen3.5 hybrid CacheTune")
    return args.hybrid_policy if model_metadata["is_qwen35"] and not args.standard_only else None


def build_engine_options(args, model_metadata):
    options = {
        "model": args.model, "dtype": args.dtype,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len, "max_num_seqs": 1,
        "tensor_parallel_size": 1, "pipeline_parallel_size": 1,
        "enforce_eager": args.enforce_eager, "enable_prefix_caching": False,
        "enable_chunked_prefill": False, "max_num_batched_tokens": args.max_model_len,
        "async_scheduling": False, "disable_log_stats": False,
    }
    if args.standard_only or model_metadata["is_qwen35"]:
        options["language_model_only"] = True
    if model_metadata["is_qwen35"] and not args.standard_only:
        # Retain the conditional wrapper and its native weight loader. EngineArgs
        # accepts both options through LLM(**kwargs) in vLLM 0.17.1.
        options["limit_mm_per_prompt"] = {"image": 0, "video": 0}
        options["enable_mm_embeds"] = False
    if not args.standard_only:
        options["worker_extension_cls"] = WORKER_EXTENSION
        options["attention_config"] = {"backend": "FLASH_ATTN"}
    return options


def parse_sample_indices(value):
    try:
        indices = [int(item.strip()) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use comma-separated nonnegative indices, e.g. 0,1,0") from exc
    if not indices or any(i < 0 for i in indices):
        raise argparse.ArgumentTypeError("Sample indices must be nonnegative")
    return indices


def plan_samples(dataset, sample_limit=None, sample_indices=None):
    if sample_limit is not None and sample_indices is not None:
        raise ValueError("--sample-limit and --sample-indices are mutually exclusive")
    indices = sample_indices if sample_indices is not None else list(range(len(dataset)))[:sample_limit]
    invalid = [i for i in indices if i < 0 or i >= len(dataset)]
    if invalid:
        raise ValueError(f"Sample index {invalid[0]} is out of range for {len(dataset)} dataset examples")
    return [(index, dataset[index]) for index in indices]


def load_context_indices(path, sample_indices):
    """Read explicit context positions; model-dependent validity belongs to state."""
    if path is None:
        return None

    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate sample key in --indices-json: {key}")
            result[key] = value
        return result

    with Path(path).open(encoding="utf-8") as stream:
        data = json.load(stream, object_pairs_hook=unique_keys)
    if not isinstance(data, dict):
        raise ValueError("--indices-json must be an object mapping dataset indices to context token lists")
    result = {}
    for key, indices in data.items():
        if not key.isascii() or not key.isdecimal() or str(int(key)) != key:
            raise ValueError(f"--indices-json sample key must be a canonical nonnegative integer: {key!r}")
        if (not isinstance(indices, list)
                or any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in indices)
                or len(set(indices)) != len(indices)):
            raise ValueError(f"--indices-json sample {key} requires unique nonnegative integer context positions")
        result[int(key)] = indices
    missing = sorted(set(sample_indices) - result.keys())
    if missing:
        raise ValueError(f"--indices-json has no selection for requested dataset index {missing[0]}")
    return result


def boundary_token(tokenizer):
    """Retain the original BOS convention, explicitly extending it for Qwen3."""
    token_id = tokenizer.bos_token_id
    source = "bos_token_id"
    if token_id is None:
        token_id = tokenizer.eos_token_id
        source = "eos_token_id_fallback"
    if token_id is None or isinstance(token_id, bool) or not isinstance(token_id, int):
        raise ValueError("Independent chunks require a scalar BOS or EOS boundary token")
    if token_id < 0:
        raise ValueError("The boundary token ID must be nonnegative")
    return token_id, {
        "boundary_token_id": token_id,
        "boundary_token_source": source,
        "boundary_token_text": tokenizer.convert_ids_to_tokens(token_id),
        "eos_fallback_used": source == "eos_token_id_fallback",
        "add_special_tokens": False,
        "chat_template_applied": False,
        "boundary_policy": "Prepend one boundary token to each independent chunk; retain it only for the first chunk in the assembled prompt.",
    }


def build_sample_tokens(example, tokenizer, context_budget=3400):
    """Preserve old document pruning and raw prompt formatting token for token."""
    marker, tokenization = boundary_token(tokenizer)
    encode = lambda text: list(tokenizer.encode(text, add_special_tokens=False))
    documents, query = build_fewshot_prompt(example)
    document_ids = [encode(text) for text in documents]
    original_count = len(document_ids)
    while sum(map(len, document_ids)) > context_budget:
        del document_ids[len(document_ids) // 2]
    if not document_ids:
        return None
    query_ids = encode(query)
    if not query_ids:
        raise ValueError("The query suffix must have at least one token")
    parts = [encode(PREFIX)] + document_ids + [query_ids]
    chunks = [[marker] + part for part in parts]
    prompt_ids = [marker] + [token for part in parts for token in part]
    assembled = chunks[0] + [token for chunk in chunks[1:] for token in chunk[1:]]
    if assembled != prompt_ids:
        raise AssertionError("Chunk concatenation differs from the full prompt")
    return {
        "chunks": chunks,
        "prompt_ids": prompt_ids,
        "suffix_len": len(query_ids),
        "original_document_count": original_count,
        "retained_document_count": len(document_ids),
        "retained_document_tokens": sum(map(len, document_ids)),
        "tokenization": tokenization,
    }


def rpc(llm, method, *args, **kwargs):
    """This runner deliberately supports exactly one worker/rank."""
    responses = llm.collective_rpc(method, args=args, kwargs=kwargs)
    if not isinstance(responses, (list, tuple)) or len(responses) != 1:
        raise RuntimeError(f"{method}: expected exactly one worker response, got {responses!r}")
    return responses[0]


def request_timing(metrics):
    """Read native per-request timing without mixing monotonic and wall clocks."""
    def finite_number(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)

    def interval(start, end):
        if finite_number(start) and finite_number(end) and 0 < start <= end:
            return end - start
        return None

    ttft = arrival_ttft = None
    source = "unavailable"
    if metrics is not None and any(hasattr(metrics, name) for name in
                                   ("first_token_ts", "scheduled_ts", "first_token_latency")):
        source = "vllm_v1_request_state_stats"
        first_token = getattr(metrics, "first_token_ts", None)
        scheduled = getattr(metrics, "scheduled_ts", None)
        ttft = interval(scheduled, first_token)
        # V1 arrival_time is frontend wall-clock; first_token_ts is engine-core
        # monotonic. The native latency was already computed within one clock.
        latency = getattr(metrics, "first_token_latency", None)
        if (finite_number(first_token) and first_token > 0
                and finite_number(latency) and latency >= 0):
            arrival_ttft = latency
    elif metrics is not None and hasattr(metrics, "first_token_time"):
        source = "vllm_legacy_request_metrics"
        first_token = getattr(metrics, "first_token_time", None)
        ttft = interval(getattr(metrics, "first_scheduled_time", None), first_token)
        arrival_ttft = interval(getattr(metrics, "arrival_time", None), first_token)
    return {"engine_ttft_seconds": ttft,
            "engine_arrival_ttft_seconds": arrival_ttft,
            "engine_timing_source": source}


def validate_runtime_version(version):
    if not (version == "0.17.1" or version.startswith(("0.17.1+", "0.17.1.dev"))):
        raise RuntimeError(f"This CacheTune port requires vLLM 0.17.1, but imported {version!r}; use CacheBlend_new/.venv and remove the old vllm_blend from PYTHONPATH")


QWEN35_ANSWER_EXTRACTION = "qwen35_leading_think_then_first_line_v1"


def extract_qwen35_answer(text):
    """Score the answer after a leading completed think block, never its reasoning."""
    leading = text.lstrip()
    if not leading.startswith("<think>"):
        return scored_first_line(text)
    end = leading.find("</think>", len("<think>"))
    if end < 0:
        return ""
    return scored_first_line(leading[end + len("</think>"):].lstrip())


def is_qwen35_runtime(llm):
    """Read the runtime model identity without requiring real-engine test doubles."""
    def field(config, name):
        return config.get(name) if isinstance(config, dict) else getattr(config, name, None)

    engine = getattr(llm, "llm_engine", None)
    model_config = getattr(engine, "model_config", None)
    config = field(model_config, "hf_text_config")
    if config is None:
        config = field(model_config, "hf_config")
        config = field(config, "text_config") or config
    model_type = field(config, "model_type") or ""
    return model_type.startswith("qwen3_5")


def generate_ids(llm, ids, max_tokens):
    from vllm import SamplingParams

    params = SamplingParams(temperature=0, max_tokens=max_tokens)
    start = time.perf_counter()
    # V1 accepts TokensPrompt; never decode/re-tokenize cache coordinates.
    outputs = llm.generate([{"prompt_token_ids": list(ids)}],
                           sampling_params=params, use_tqdm=False)
    wall = time.perf_counter() - start
    if len(outputs) != 1 or list(outputs[0].prompt_token_ids) != list(ids):
        raise AssertionError("vLLM prompt token IDs differ from supplied cache positions")
    output = outputs[0]
    if len(output.outputs) != 1:
        raise AssertionError("Expected one greedy completion")
    timing = request_timing(getattr(output, "metrics", None))
    completion = output.outputs[0]
    result = {
        "token_ids": list(completion.token_ids),
        "text": completion.text,
        "scored_text": scored_first_line(completion.text),
        "finish_reason": completion.finish_reason,
        **timing,
        "generate_wall_seconds": wall,
    }
    if is_qwen35_runtime(llm):
        result["scored_text"] = extract_qwen35_answer(completion.text)
        result["answer_extraction"] = {"policy": QWEN35_ANSWER_EXTRACTION}
    return result


def validate_selection(info, prompt_len, suffix_len, ratio, *, alpha=0.5,
                       sink_size=0, check_layer=1, cache_mode="cpu_pipeline",
                       selected_context_indices=None, hybrid_policy=None, chunk_context_lengths=None):
    """Check worker selection against the preserved experiment contract."""
    if not isinstance(info, dict):
        raise AssertionError("cachetune_prepare must return a selection dictionary")
    context_len = prompt_len - suffix_len
    if context_len <= 0 or suffix_len <= 0:
        raise ValueError("Require nonempty context and query suffix")
    selected = info["selected_indices"]
    reused = info["reused_indices"]
    if any(isinstance(i, bool) or not isinstance(i, int) for i in selected + reused):
        raise AssertionError("Selection indices must be integer token positions")
    if selected != sorted(set(selected)) or reused != sorted(set(reused)):
        raise AssertionError("Selection indices must be sorted and unique")
    if sorted(selected + reused) != list(range(prompt_len)):
        raise AssertionError("Selected and reused indices must partition the prompt")
    if selected[-suffix_len:] != list(range(context_len, prompt_len)):
        raise AssertionError("The entire query suffix must be recomputed")
    if chunk_context_lengths is None:
        chunk_context_lengths = info.get("chunk_context_lengths", [context_len])
    if (any(isinstance(n, bool) or not isinstance(n, int) or n <= 0
            for n in chunk_context_lengths) or sum(chunk_context_lengths) != context_len):
        raise AssertionError("Chunk lengths must partition the reusable context")
    expected_context = sum(int(n * ratio) for n in chunk_context_lengths)
    if len(selected) - suffix_len != expected_context:
        raise AssertionError("Context selection budget differs from sum of per-chunk floor(ratio * tokens)")
    selected_context = selected[:len(selected) - suffix_len]
    if not 0 <= sink_size <= expected_context:
        raise AssertionError("sink_size must fit the total context recomputation budget")
    if not set(range(sink_size)).issubset(selected_context):
        raise AssertionError("The forced sink prefix must be part of the selected context")
    if selected_context_indices is not None and selected_context != sorted(selected_context_indices):
        raise AssertionError("Worker selection differs from the explicit context indices")
    expected = {
        "prompt_tokens": prompt_len,
        "context_tokens": context_len,
        "suffix_tokens": suffix_len,
        "selected_context_tokens": expected_context,
        "selected_total_tokens": len(selected),
        "reused_tokens": len(reused),
        "score_layer": None,
        "signal": "K/V-all-layers" if selected_context_indices is None else "External-indices",
        "selection_source": "frequency" if selected_context_indices is None else "external_context_indices",
        "check_layers": [check_layer],
        "alpha": alpha,
        "sink_size": sink_size,
        "cache_mode": cache_mode,
    }
    if hybrid_policy is not None:
        expected["hybrid_policy"] = hybrid_policy
    for key, value in expected.items():
        if key in info and info[key] != value:
            raise AssertionError(f"Worker selection metadata mismatch: {key}")
    return {**info, **expected}


def run_cached_pair(llm, tokens, ratio, max_tokens, verify_repeat=False, generate=None,
                    verify_logits=False, *, alpha=0.5, sink_size=0, check_layer=1,
                    cache_mode="cpu_pipeline", selected_context_indices=None,
                    hybrid_policy=None):
    if hybrid_policy is not None and hybrid_policy not in ("attention", "block", "mlp"):
        raise ValueError("Unknown hybrid policy")
    generate = generate or generate_ids
    prompt_ids = tokens["prompt_ids"]
    drop_flags = tokens.get("drop_bos_flags")
    if drop_flags is not None and (
            not isinstance(drop_flags, (list, tuple))
            or len(drop_flags) != len(tokens["chunks"])
            or any(not isinstance(flag, bool) for flag in drop_flags)):
        raise ValueError("drop_bos_flags must contain one boolean per independent chunk")
    rpc(llm, "cachetune_reset", clear_cache=True)
    rpc(llm, "cachetune_start_collect")
    # Query suffix is new request input: never independently encode/cache it.
    context_remaining = len(prompt_ids) - tokens["suffix_len"]
    chunk_context_lengths = []
    for i, chunk in enumerate(tokens["chunks"]):
        if not context_remaining:
            break
        drop = int(i > 0) if drop_flags is None else int(drop_flags[i])
        keep = min(len(chunk) - drop, context_remaining)
        if keep <= 0:
            continue
        context_chunk = chunk[:keep + drop]
        generate(llm, context_chunk, 1)
        rpc(llm, "cachetune_append_chunk", expected_ids=context_chunk, drop_bos=bool(drop))
        chunk_context_lengths.append(keep)
        context_remaining -= keep
    if context_remaining:
        raise ValueError("Independent chunks do not cover the reusable context")
    hybrid_options = {} if hybrid_policy is None else {"hybrid_policy": hybrid_policy}
    indices = rpc(llm, "cachetune_prepare", prompt_ids=prompt_ids,
                  suffix_len=tokens["suffix_len"], ratio=ratio, alpha=alpha,
                  sink_size=sink_size, check_layer=check_layer, cache_mode=cache_mode,
                  selected_context_indices=selected_context_indices, **hybrid_options)
    indices = validate_selection(indices, len(prompt_ids), tokens["suffix_len"], ratio,
                                 alpha=alpha, sink_size=sink_size, check_layer=check_layer,
                                 cache_mode=cache_mode, selected_context_indices=selected_context_indices,
                                 hybrid_policy=hybrid_policy,
                                 chunk_context_lengths=chunk_context_lengths)
    rpc(llm, "cachetune_enable", prompt_ids=prompt_ids)
    if verify_logits:
        rpc(llm, "cachetune_probe_start")
    cached = generate(llm, prompt_ids, max_tokens)
    if verify_logits:
        cached["first_logits"] = rpc(llm, "cachetune_probe_finish", set_reference=True)
    diagnostics = {"cached": rpc(llm, "cachetune_diagnostics")}
    repeat = None
    if verify_repeat:
        rpc(llm, "cachetune_enable", prompt_ids=prompt_ids)
        repeat = generate(llm, prompt_ids, max_tokens)
        diagnostics["repeat"] = rpc(llm, "cachetune_diagnostics")
    rpc(llm, "cachetune_disable")
    if verify_logits:
        rpc(llm, "cachetune_probe_start")
    full = generate(llm, prompt_ids, max_tokens)
    if verify_logits:
        full["first_logits"] = rpc(llm, "cachetune_probe_finish", set_reference=False)
    diagnostics["full"] = rpc(llm, "cachetune_diagnostics")
    return indices, cached, full, repeat, diagnostics


def average(values):
    values = [x for x in values if x is not None]
    return sum(values) / len(values) if values else None


def summarize(samples):
    cached_rows = [x for x in samples if x.get("cached") is not None]
    return {
        "samples": len(samples),
        "cached_samples": len(cached_rows),
        "mean_cached_rougeL": average([x["cached"]["rougeL"] for x in cached_rows]),
        "mean_full_rougeL": average([x["full"]["rougeL"] for x in samples]),
        "mean_cached_engine_ttft_seconds": average([x["cached"]["engine_ttft_seconds"] for x in cached_rows]),
        "mean_full_engine_ttft_seconds": average([x["full"]["engine_ttft_seconds"] for x in samples]),
        "mean_cached_engine_arrival_ttft_seconds": average([x["cached"].get("engine_arrival_ttft_seconds") for x in cached_rows]),
        "mean_full_engine_arrival_ttft_seconds": average([x["full"].get("engine_arrival_ttft_seconds") for x in samples]),
        "cached_ttft_samples": sum(x["cached"]["engine_ttft_seconds"] is not None for x in cached_rows),
        "full_ttft_samples": sum(x["full"]["engine_ttft_seconds"] is not None for x in samples),
        "exact_full_matches": sum(x["cached_equals_full"] for x in cached_rows),
        "repeat_checks": sum("repeat_equals_cached" in x for x in samples),
        "repeat_matches": sum(x.get("repeat_equals_cached", False) for x in samples),
        "revisit_checks": sum("revisit_reference_run_index" in x for x in samples),
        "revisit_full_matches": sum(x.get("revisit_full_equals_reference", False) for x in samples),
        "revisit_cached_checks": sum("revisit_cached_equals_reference" in x for x in samples),
        "revisit_cached_matches": sum(x.get("revisit_cached_equals_reference", False) for x in samples),
        "revisit_selection_matches": sum(x.get("revisit_selection_equals_reference", False) for x in samples),
    }


def record_revisit(row, first_occurrences):
    """Compare an A/B/A revisit with A before B, without comparing timings."""
    previous = first_occurrences.get(row["sample_index"])
    if previous is None:
        first_occurrences[row["sample_index"]] = row
        return []
    row["revisit_reference_run_index"] = previous["run_index"]
    row["revisit_full_equals_reference"] = row["full"]["token_ids"] == previous["full"]["token_ids"]
    checks = ["revisit_full_equals_reference"]
    if row["cached"] is not None:
        row["revisit_cached_equals_reference"] = row["cached"]["token_ids"] == previous["cached"]["token_ids"]
        row["revisit_selection_equals_reference"] = all(
            row["indices"][key] == previous["indices"][key]
            for key in ("selected_indices", "reused_indices")
        )
        checks.extend(["revisit_cached_equals_reference", "revisit_selection_equals_reference"])
    return [key for key in checks if not row[key]]


def save_report(path, report):
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(path.name + ".tmp")
        temp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(path)


def parse_args(argv=None):
    script_dir = Path(__file__).resolve().parent
    candidates = [script_dir / "inputs" / "samsum.json",
                  script_dir.parent.parent / "CacheTune" / "example" / "inputs" / "samsum.json",
                  script_dir.parent / "inputs" / "samsum.json"]
    dataset = next((p for p in candidates if p.is_file()), candidates[0])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--dataset", type=Path, default=dataset)
    sample_group = parser.add_mutually_exclusive_group()
    sample_group.add_argument("--sample-limit", type=int)
    sample_group.add_argument("--sample-indices", type=parse_sample_indices,
                              help="Run original dataset indices in this order, including repeats (e.g. 0,1,0).")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--recompute-ratio", type=float, default=0.15)
    parser.add_argument("--low-freq-pct", type=float, default=0.5,
                        help="Fraction of rFFT bins retained separately in every reusable chunk (0, 1].")
    parser.add_argument("--sink-size", type=int, default=0,
                        help="Force this many leading context tokens within the total recomputation budget.")
    parser.add_argument("--check-layer", type=int, default=None,
                        help="Dense decoder layer (default 1), or Qwen3.5 logical full-attention layer (default 0).")
    parser.add_argument("--hybrid-policy", choices=["attention", "block", "mlp"], default="block",
                        help="Qwen3.5 text-only hybrid branch reuse policy; default block.")
    parser.add_argument("--cache-mode", choices=["cpu_pipeline", "cpu_serial", "gpu"],
                        default="cpu_pipeline")
    parser.add_argument("--indices-json", type=Path,
                        help="JSON object mapping original dataset indices to explicit context token positions.")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--dtype", default="auto", choices=["bfloat16", "float16", "auto"])
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--verify-repeat", action="store_true")
    parser.add_argument("--verify-logits", action="store_true",
                        help="Capture first-token logits for correctness; do not use for timings")
    parser.add_argument("--standard-only", action="store_true", help="Run ordinary inference without any CacheTune RPC or selective recomputation.")
    args = parser.parse_args(argv)
    if args.sample_limit is not None and args.sample_limit < 1:
        parser.error("--sample-limit must be positive (omit it for all samples)")
    if not 0 <= args.recompute_ratio <= 1:
        parser.error("--recompute-ratio must be in [0, 1]")
    if not 0 < args.low_freq_pct <= 1:
        parser.error("--low-freq-pct must be in (0, 1]")
    if args.sink_size < 0 or (args.check_layer is not None and args.check_layer < 0):
        parser.error("--sink-size and --check-layer must be nonnegative")
    if args.max_tokens < 1 or args.max_model_len < args.max_tokens + 1:
        parser.error("Require positive --max-tokens below --max-model-len")
    if not 0 < args.gpu_memory_utilization < 1:
        parser.error("--gpu-memory-utilization must be in (0, 1)")
    if not args.enforce_eager and not args.standard_only:
        parser.error("The CacheTune port currently requires --enforce-eager")
    if args.verify_repeat and args.standard_only:
        parser.error("--verify-repeat checks cached reuse and cannot be used with --standard-only")
    if args.verify_logits and args.standard_only:
        parser.error("--verify-logits compares CacheTune with Full and requires CacheTune mode")
    if args.standard_only and (args.indices_json is not None or args.low_freq_pct != 0.5
                               or args.sink_size != 0 or args.check_layer not in (None, 1)
                               or args.cache_mode != "cpu_pipeline" or args.hybrid_policy != "block"):
        parser.error("Selection overrides and --cache-mode apply only to CacheTune, not --standard-only")
    args.dataset = args.dataset.expanduser().resolve()
    if args.indices_json is not None:
        args.indices_json = args.indices_json.expanduser().resolve()
    if args.output_json is not None:
        args.output_json = args.output_json.expanduser().resolve()
    return args


def main():
    args = parse_args()
    # Set before importing vLLM or resolving model configuration. Spawn workers
    # inherit the same choice; standard inference must keep native model classes.
    os.environ["CACHETUNE_ENABLE_ADAPTER"] = "0" if args.standard_only else "1"
    config = read_model_config(args.model)
    model_metadata = validate_model_mode(config, args.standard_only)
    hybrid_policy = resolve_model_options(args, model_metadata)
    from transformers import AutoTokenizer
    import vllm

    validate_runtime_version(vllm.__version__)
    LLM = vllm.LLM

    if not args.standard_only:
        from cachetune import register_models

        register_models()
    dataset = load_dataset(str(args.dataset))
    samples_to_run = plan_samples(dataset, args.sample_limit, args.sample_indices)
    provided_indices = load_context_indices(args.indices_json, [index for index, _ in samples_to_run])
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    _, tokenization = boundary_token(tokenizer)
    engine_options = build_engine_options(args, model_metadata)
    report = {
        "configuration": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "engine_configuration": engine_options,
        "runtime_version": {"vllm": vllm.__version__, "vllm_file": vllm.__file__},
        "cachetune_adapter_enabled": os.environ["CACHETUNE_ENABLE_ADAPTER"] == "1",
        "model_metadata": model_metadata,
        "tokenization": tokenization,
        "selection": None if args.standard_only else {
            "source": "frequency" if provided_indices is None else "external_context_indices",
            "signal": "K/V-all-layers" if provided_indices is None else "External-indices",
            "score_layer": None,
            "alpha": args.low_freq_pct, "sink_size": args.sink_size,
            "check_layers": [args.check_layer], "cache_mode": args.cache_mode,
            **({"hybrid_policy": hybrid_policy,
                "layer_index_coordinate": "full_attention_layer_index"}
               if hybrid_policy is not None else {}),
        },
        "execution_mode": "standard_inference_only" if args.standard_only else "cachetune",
        "timing_clock_definition": {
            "engine_ttft_seconds": "vLLM V1 first_token_ts - scheduled_ts; both are engine-core monotonic timestamps. This is the scheduled-to-first-token interval (native prefill_time), not arrival-based TTFT.",
            "engine_arrival_ttft_seconds": "vLLM V1 first_token_latency; frontend wall-clock arrival-to-first-output interval, including queueing and frontend/core delivery. Never subtract arrival_time from first_token_ts.",
            "legacy_fallback": "Only legacy RequestMetrics use first_token_time - first_scheduled_time and first_token_time - arrival_time within their wall-clock domain.",
            "generate_wall_seconds": "Parent-process perf_counter duration of this same synchronous generate call, including decode; not a TTFT substitute.",
        },
        "timing_note": "Native per-request statistics are enabled with disable_log_stats=False. TTFT is obtained from this generation's final RequestOutput.metrics without an additional generation. Missing, uninitialized, or invalid metrics remain null. Independent-chunk collection and FFT preparation are excluded. Correctness probes add overhead; this serial protocol does not establish concurrent serving performance.",
        "samples": [], "skipped_samples": [], "status": "running",
        "requested_samples": len(samples_to_run),
        "requested_sample_indices": [index for index, _ in samples_to_run],
    }
    save_report(args.output_json, report)
    llm = None
    failed = False
    first_occurrences = {}
    try:
        llm = LLM(**engine_options)
        actual_dtype = getattr(llm.llm_engine.model_config, "dtype", None)
        report["resolved_model_dtype"] = str(actual_dtype) if actual_dtype is not None else None
        save_report(args.output_json, report)
        for run_idx, (sample_idx, example) in enumerate(samples_to_run):
            tokens = build_sample_tokens(example, tokenizer)
            if tokens is None:
                report["skipped_samples"].append({"run_index": run_idx, "sample_index": sample_idx, "reason": "no few-shot context remains after pruning"})
                save_report(args.output_json, report)
                continue
            prompt_ids = tokens["prompt_ids"]
            if len(prompt_ids) + args.max_tokens > args.max_model_len:
                raise ValueError(f"Sample {sample_idx} needs {len(prompt_ids)} prompt + {args.max_tokens} output tokens, exceeding max_model_len={args.max_model_len}")
            if args.standard_only:
                full = generate_ids(llm, prompt_ids, args.max_tokens)
                indices, cached, repeat, diagnostics = None, None, None, None
            else:
                indices, cached, full, repeat, diagnostics = run_cached_pair(
                    llm, tokens, args.recompute_ratio, args.max_tokens, args.verify_repeat,
                    verify_logits=args.verify_logits, alpha=args.low_freq_pct,
                    sink_size=args.sink_size, check_layer=args.check_layer,
                    cache_mode=args.cache_mode,
                    hybrid_policy=hybrid_policy,
                    selected_context_indices=None if provided_indices is None else provided_indices[sample_idx])
            answers = example["answers"]
            if isinstance(answers, str):
                answers = [answers]
            if not answers:
                raise ValueError(f"Sample {sample_idx} has no reference answer")
            for result in (cached, full):
                if result is not None:
                    result["rougeL"] = max(compute_rl(result["scored_text"], answer) for answer in answers)
            row = {"run_index": run_idx, "sample_index": sample_idx, "prompt_token_ids": prompt_ids,
                   "indices": indices, "cached": cached, "full": full,
                   "runtime_diagnostics": diagnostics,
                   "original_document_count": tokens["original_document_count"],
                   "retained_document_count": tokens["retained_document_count"]}
            if cached is not None:
                row["cached_equals_full"] = cached["token_ids"] == full["token_ids"]
            if repeat is not None:
                row["repeat"] = repeat
                row["repeat_equals_cached"] = repeat["token_ids"] == cached["token_ids"]
            revisit_failures = record_revisit(row, first_occurrences)
            report["samples"].append(row)
            report["summary"] = summarize(report["samples"])
            save_report(args.output_json, report)
            print(json.dumps({"run_index": run_idx, "sample_index": sample_idx, "prompt_tokens": len(prompt_ids),
                              "selected_context_tokens": indices["selected_context_tokens"] if indices else None,
                              "cached_equals_full": row.get("cached_equals_full"),
                              "cached_rougeL": cached["rougeL"] if cached else None,
                              "full_rougeL": full["rougeL"]}, ensure_ascii=False), flush=True)
            if repeat is not None and not row["repeat_equals_cached"]:
                raise AssertionError(f"Sample {sample_idx}: repeated cache reuse changed generated tokens")
            if args.verify_logits and cached is not None:
                print(json.dumps({"sample_index":sample_idx, "first_logits_comparison":full["first_logits"]}), flush=True)
                if not cached["first_logits"]["finite"] or not full["first_logits"]["finite"]:
                    raise AssertionError("Non-finite first-token logits")
                if args.recompute_ratio == 1.0 and full["first_logits"]["relative_l2"] >= 0.01:
                    raise AssertionError("100% repair first-logit relative L2 error exceeds 1%")
            if cached is not None and args.recompute_ratio == 1.0 and not row["cached_equals_full"]:
                raise AssertionError(f"Sample {sample_idx}: 100% recomputation differs from Full")
            if revisit_failures:
                raise AssertionError(f"Run {run_idx}, sample {sample_idx}: revisit mismatch: {', '.join(revisit_failures)}")
            if not args.standard_only:
                rpc(llm, "cachetune_reset", clear_cache=True)
        report["status"] = "complete"
        report["summary"] = summarize(report["samples"])
        report["summary"]["requested_samples"] = len(samples_to_run)
        report["summary"]["skipped_samples"] = len(report["skipped_samples"])
        print(json.dumps(report["summary"], ensure_ascii=False, indent=2), flush=True)
    except Exception as exc:
        failed = True
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        save_report(args.output_json, report)
        if llm is not None and not args.standard_only:
            try:
                rpc(llm, "cachetune_reset", clear_cache=True)
            except Exception as exc:
                report["cleanup_error"] = f"{type(exc).__name__}: {exc}"
                report["status"] = "failed"
                save_report(args.output_json, report)
                if not failed:
                    raise


if __name__ == "__main__":
    main()
