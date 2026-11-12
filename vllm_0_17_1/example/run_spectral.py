"""Run paper-aligned CacheTune on explicit independently reusable text chunks.

Input JSON: {"chunks": ["document one", "document two"], "query": "question"}.
An array of requests is also accepted. Prompt formatting is explicit input;
this driver never inserts a model-specific chat template behind the caller.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from example import blend_samsum_freq as entry


def build_request(row, tokenizer):
    if (not isinstance(row, dict) or not isinstance(row.get("chunks"), list)
            or not row["chunks"] or not all(isinstance(x, str) and x for x in row["chunks"])
            or not isinstance(row.get("query"), str) or not row["query"]):
        raise ValueError("Each request needs nonempty string chunks and a query")
    docs = [list(tokenizer.encode(text, add_special_tokens=False)) for text in row["chunks"]]
    suffix = list(tokenizer.encode(row["query"], add_special_tokens=False))
    if not suffix or any(not ids for ids in docs):
        raise ValueError("Chunks and query must encode to nonempty token sequences")
    bos = tokenizer.bos_token_id
    if bos is not None and (isinstance(bos, bool) or not isinstance(bos, int) or bos < 0):
        raise ValueError("BOS must be a nonnegative integer token ID or None")
    # Match the public spectral protocol: a real BOS, if present, is retained
    # only on the first chunk. Later independent encodings may use a temporary
    # BOS, which is dropped from their stored KV. No synthetic EOS is added.
    leading = [] if bos is None else [bos]
    chunks = [leading + ids for ids in docs]
    drop_flags = [False] + [bos is not None] * (len(docs) - 1)
    context = leading + [token for ids in docs for token in ids]
    return {"chunks": chunks, "prompt_ids": context + suffix,
            "suffix_len": len(suffix), "drop_bos_flags": drop_flags,
            "tokenization": {"bos_token_id": bos, "eos_fallback_used": False,
                             "chat_template_applied": False}}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--ratio-mode", choices=("fixed", "gss"), default="gss")
    parser.add_argument("--recompute-ratio", type=float, default=0.15)
    parser.add_argument("--low-freq-pct", type=float, default=0.5)
    parser.add_argument("--cache-mode", choices=("cpu_pipeline", "cpu_serial", "gpu"), default="cpu_pipeline")
    parser.add_argument("--hybrid-policy", choices=("block", "attention", "mlp"), default="block")
    parser.add_argument("--r-min", type=float, default=0.15)
    parser.add_argument("--r-max", type=float, default=1.0)
    parser.add_argument("--gss-tolerance", type=float, default=0.01)
    parser.add_argument("--calibration-request", type=Path,
                        help="Fixed calibration requests; defaults to --request")
    parser.add_argument("--calibration-samples", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--tc", type=float, help="Optional measured recomputation seconds per context-token unit")
    parser.add_argument("--ti", type=float, help="Optional measured transfer seconds per same unit")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.65)
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16"), default="auto")
    parser.add_argument("--verify-repeat", action="store_true")
    args = parser.parse_args(argv)
    if not 0 <= args.recompute_ratio <= 1 or not 0 < args.low_freq_pct <= 1:
        parser.error("Require ratio in [0,1] and low-freq-pct in (0,1]")
    if not 0.15 <= args.r_min < args.r_max <= 1 or not 0 < args.gss_tolerance < args.r_max - args.r_min:
        parser.error("GSS requires .15 <= r-min < r-max <= 1 and positive smaller tolerance")
    if min(args.calibration_samples, args.repeats, args.max_tokens) < 1 or args.warmup < 0:
        parser.error("Counts must be positive; warmup may be zero")
    if args.max_model_len <= args.max_tokens or not 0 < args.gpu_memory_utilization < 1:
        parser.error("Invalid model length or GPU memory fraction")
    if (args.tc is None) != (args.ti is None):
        parser.error("Supply both --tc and --ti, or neither")
    if args.tc is not None and any(not math.isfinite(x) or x <= 0 for x in (args.tc, args.ti)):
        parser.error("Measured --tc/--ti must be positive finite seconds")
    if args.ratio_mode == "gss" and args.cache_mode == "gpu":
        parser.error("Use fixed ratio for GPU-resident cache; GSS calibrates host-transfer overlap")
    return args


def load_requests(path, tokenizer):
    rows = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(rows, dict):
        rows = [rows]
    if not isinstance(rows, list) or not rows:
        raise ValueError("Request file must contain one object or a nonempty array")
    return [build_request(row, tokenizer) for row in rows]


def main(argv=None):
    args = parse_args(argv)
    os.environ["CACHETUNE_ENABLE_ADAPTER"] = "1"
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    config = entry.read_model_config(args.model)
    metadata = entry.validate_model_mode(config, False)
    args.standard_only, args.enforce_eager, args.check_layer = False, True, None
    policy = entry.resolve_model_options(args, metadata)
    from transformers import AutoTokenizer
    import vllm
    from cachetune import register_models
    from cachetune.spectral_algorithms import golden_section_search
    entry.validate_runtime_version(vllm.__version__)
    register_models()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    requests = load_requests(args.request, tokenizer)
    calibration = load_requests(args.calibration_request or args.request, tokenizer)[:args.calibration_samples]
    for request in requests + calibration:
        if len(request["prompt_ids"]) + args.max_tokens > args.max_model_len:
            raise ValueError("Request exceeds --max-model-len; no hidden truncation is applied")
    report = {"status": "running", "configuration": {k: str(v) if isinstance(v, Path) else v
              for k, v in vars(args).items()}, "model_metadata": metadata, "calibration": None,
              "samples": [], "timing_scope": "Scheduled-to-first-token V1 metrics; independent chunk collection, scoring and preparation excluded."}
    entry.save_report(args.output_json, report)
    llm = None
    try:
        llm = vllm.LLM(**entry.build_engine_options(args, metadata))

        def run(request, ratio, max_tokens=1, repeat=False):
            return entry.run_cached_pair(llm, request, ratio, max_tokens,
                verify_repeat=repeat, alpha=args.low_freq_pct,
                check_layer=args.check_layer, cache_mode=args.cache_mode,
                hybrid_policy=policy)

        ratio = args.recompute_ratio
        if args.ratio_mode == "gss":
            # Profile only the roofline warm start. The search objective always
            # measures complete-request TTFT, including all online layer loads.
            for _ in range(args.warmup):
                for request in calibration:
                    run(request, args.r_min)
            cost_samples = []
            if args.tc is None:
                for request in calibration:
                    _, _, full, _, _ = run(request, args.r_min)
                    seconds = full["engine_ttft_seconds"]
                    if seconds is None or not math.isfinite(seconds) or seconds <= 0:
                        raise RuntimeError("Native full-prefill TTFT is unavailable for warm-start profiling")
                    transfer = entry.rpc(llm, "cachetune_profile_transfer", repeats=args.repeats)
                    context = len(request["prompt_ids"]) - request["suffix_len"]
                    cost_samples.append({"tc": seconds/context,
                                         "ti": transfer["seconds_per_reused_token"],
                                         "transfer": transfer})
                tc = statistics.mean(x["tc"] for x in cost_samples)
                ti = statistics.mean(x["ti"] for x in cost_samples)
            else:
                tc, ti = args.tc, args.ti

            def objective(candidate):
                values = []
                for _ in range(args.repeats):
                    for request in calibration:
                        _, cached, _, _, _ = run(request, candidate)
                        value = cached["engine_ttft_seconds"]
                        if value is None or not math.isfinite(value) or value <= 0:
                            raise RuntimeError("Complete-request TTFT is missing or invalid")
                        values.append(value)
                return statistics.mean(values)

            result = golden_section_search(objective, tc=tc, ti=ti,
                r_min=args.r_min, r_max=args.r_max, tolerance=args.gss_tolerance)
            ratio = result.ratio
            report["calibration"] = {**asdict(result), "tc": tc, "ti": ti,
                "profile_samples": cost_samples, "request_count": len(calibration),
                "cost_definition": "Warm-start tc: measured full-prefill seconds/context token; ti: serial H2D of eligible compact KV and hybrid branches/reused token. Both use all active layers. The optimized objective is measured complete-request TTFT."}
            entry.save_report(args.output_json, report)
        for index, request in enumerate(requests):
            selection, cached, full, repeat, diagnostics = run(
                request, ratio, args.max_tokens, args.verify_repeat)
            if repeat is not None and cached["token_ids"] != repeat["token_ids"]:
                raise AssertionError("Repeated reuse produced different tokens")
            report["samples"].append({"sample_index": index, "selection": selection,
                "cached": cached, "full": full, "repeat": repeat,
                "runtime_diagnostics": diagnostics})
            entry.save_report(args.output_json, report)
        report.update(status="complete", deployment_ratio=ratio)
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        entry.save_report(args.output_json, report)
        if llm is not None:
            entry.rpc(llm, "cachetune_reset", clear_cache=True)
    print(json.dumps({"status": report["status"], "ratio": ratio,
                      "samples": len(report["samples"]), "output": str(args.output_json)}))


if __name__ == "__main__":
    main()
