"""Opt-in implementation of the supplied ICLR paper (legacy scripts stay default).

Model imports are deliberately lazy so every existing entry point exposes paper
help without CUDA. The selector and the measured deployment search are separate.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import time


def make_parser(dataset, default_storage, is_qwen):
    parser = argparse.ArgumentParser(description="CacheTune paper frequency selection and Algorithm 1 GSS")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--dataset-target-path", type=Path, help="MultiNews .tgt file when input is .src")
    parser.add_argument("--tensor-parallel-size", type=int, default=2 if is_qwen else 1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--max-context-tokens", type=int, default=3400)
    parser.add_argument("--max-tokens", type=int, default=550 if dataset == "multinews" else 128 if dataset == "samsum" else 32)
    parser.add_argument("--sample-limit", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--ratio-mode", choices=("fixed", "gss"), default="gss")
    parser.add_argument("--recomp-ratio", type=float, default=0.15)
    parser.add_argument("--r-min", type=float, default=0.15)
    parser.add_argument("--r-max", type=float, default=1.0)
    parser.add_argument("--gss-tolerance", type=float, default=0.01)
    parser.add_argument("--calibration-samples", type=int, default=10)
    parser.add_argument("--calibration-dataset", choices=("samsum", "musique", "wikimqa", "hotpotqa", "multinews"), default="samsum")
    parser.add_argument("--calibration-dataset-path", type=Path)
    parser.add_argument("--calibration-dataset-target-path", type=Path)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--profile-repeats", type=int, default=3)
    parser.add_argument("--storage", choices=("cpu", "disk"), default=default_storage)
    parser.add_argument("--disk-root", type=Path, default=Path("disk_offload_cache/paper"))
    parser.add_argument("--output-json", type=Path, default=Path("paper_results") / (dataset + ".json"))
    return parser


def validate_args(args):
    if not 0 < args.alpha <= 1:
        raise ValueError("alpha must lie in (0, 1]")
    if not 0.15 <= args.r_min < args.r_max <= 1:
        raise ValueError("paper search bounds require 0.15 <= r-min < r-max <= 1")
    if not 0.15 <= args.recomp_ratio <= 1:
        raise ValueError("paper fixed ratio must lie in [0.15, 1]")
    if not 0 < args.gss_tolerance < args.r_max - args.r_min:
        raise ValueError("gss-tolerance must be positive and smaller than the search interval")
    for name in ("sample_limit", "calibration_samples", "repeats", "profile_repeats", "max_tokens", "max_context_tokens", "tensor_parallel_size"):
        if getattr(args, name) < 1:
            raise ValueError(name + " must be positive")
    if args.warmup < 0 or args.max_num_seqs != 1:
        raise ValueError("warmup must be nonnegative; paper runner requires max-num-seqs=1")
    if args.max_context_tokens + args.max_tokens >= args.max_model_len:
        raise ValueError("max-model-len must leave room for context, query, and output")


def dataset_file(dataset, explicit=None):
    if explicit is not None:
        if not explicit.is_file():
            raise FileNotFoundError(explicit)
        return explicit.resolve()
    names = {"samsum": "samsum.json", "musique": "musique_s.json", "wikimqa": "wikimqa_s.json",
             "hotpotqa": "hotpot_dev_distractor_v1.json", "multinews": "test.src.cleaned"}
    candidates = [Path(__file__).resolve().parent / "inputs" / names[dataset], Path("inputs") / names[dataset]]
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError("Dataset absent; pass --dataset-path (or --calibration-dataset-path): " + names[dataset])


def load_rows(dataset, path, target=None):
    if dataset == "multinews" and path.suffix != ".json":
        target = target or path.with_name("test.tgt")
        source_lines = path.read_text(encoding="utf-8").splitlines()
        target_lines = target.read_text(encoding="utf-8").splitlines()
        if len(source_lines) != len(target_lines):
            raise ValueError("MultiNews source and target line counts differ")
        return [{"docs": [d.strip() for d in src.replace("NEWLINE_CHAR", "\n").split("|||||") if d.strip()],
                 "answers": [tgt.replace("NEWLINE_CHAR", "\n").strip()]}
                for src, tgt in zip(source_lines, target_lines)]
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Expected a JSON array of benchmark records")
    return data


def answer_texts(value):
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [text for item in value.values() for text in answer_texts(item)]
    if isinstance(value, list):
        return [text for item in value for text in answer_texts(item)]
    return []


def build_request(row, dataset, tokenizer, args, sample_id):
    if dataset == "samsum":
        prefix = "Summarize the dialogue into a few short sentences. The following are some examples.\n\n"
        docs = [ctx["text"] for ctx in row["ctxs"]]
        query = "\n\n" + row["question"]
    elif dataset == "multinews":
        prefix = "Summarize the following news articles into a coherent summary.\n\n"
        docs = row.get("docs") or [ctx["text"] for ctx in row["ctxs"]]
        query = "\n\nSummary:"
    else:
        prefix = "You will be asked a question after reading several passages. Please directly answer the question based on the given passages. Do NOT repeat the question. The answer should be within 5 words.\nPassages:\n"
        if dataset == "hotpotqa" and "context" in row:
            docs = [title + "\n\n" + "".join(sentences) + "\n\n" for title, sentences in row["context"]]
        else:
            docs = [ctx.get("title", "") + "\n\n" + ctx["text"] + "\n\n" for ctx in row["ctxs"]]
        question = row["question"]
        if not question.endswith("?"):
            question += "?"
        query = "\n\nAnswer the question directly based on the given passages. Do NOT repeat the question. The answer should be within 5 words.\nQuestion: " + question + "\nAnswer:"
    # The assembled request gets a single leading BOS. Independent document
    # encoding uses its own BOS, removed before storing the reusable chunk.
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=True)
    docs_ids = [tokenizer.encode(doc, add_special_tokens=False) for doc in docs]
    suffix = tokenizer.encode(query, add_special_tokens=False)
    budget = min(args.max_context_tokens, args.max_model_len - len(suffix) - args.max_tokens)
    if budget <= len(prefix_ids) or not suffix:
        raise ValueError("No room for reusable context/query at sample " + str(sample_id))
    chunks = [prefix_ids]
    remaining = budget - len(prefix_ids)
    for ids in docs_ids:
        if remaining <= 0:
            break
        ids = ids[:remaining]
        if ids:
            chunks.append(ids)
            remaining -= len(ids)
    if len(chunks) < 2:
        raise ValueError("No reusable document tokens at sample " + str(sample_id))
    tokens = [token for chunk in chunks for token in chunk] + suffix
    digest = hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()
    answers = answer_texts(row.get("answers", row.get("answer", row.get("summary"))))
    if not answers:
        raise ValueError("Missing reference answers at sample " + str(sample_id))
    return {"id": str(sample_id), "chunks": chunks, "suffix": suffix, "tokens": tokens, "token_sha256": digest, "answers": answers}


def run_on_workers(llm, method, **kwargs):
    executor = llm.llm_engine.model_executor
    if hasattr(executor, "_run_workers"):
        return executor._run_workers(method, **kwargs)
    worker = executor.driver_worker
    if hasattr(worker, "execute_method"):
        return [worker.execute_method(method, **kwargs)]
    return [getattr(worker, method)(**kwargs)]


def ttft(output):
    metrics = output[0].metrics
    if metrics.first_token_time is None or metrics.first_scheduled_time is None:
        raise RuntimeError("vLLM did not provide first-token/scheduled timestamps")
    value = metrics.first_token_time - metrics.first_scheduled_time
    if not math.isfinite(value) or value <= 0:
        raise RuntimeError("Invalid TTFT: " + str(value))
    return float(value)


def save_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def quality_score(dataset, text, answers, tokenizer):
    from utils import compute_rl, normalize_answer
    if dataset == "multinews":
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        scores = [scorer.score(answer, text) for answer in answers]
        return {name: max(score[name].fmeasure for score in scores) for name in ("rouge1", "rouge2", "rougeL")}
    if dataset == "samsum":
        return {"rougeL": max(compute_rl(text.lstrip("\n").split("\n")[0], answer) for answer in answers)}
    from collections import Counter
    predicted_text = text.lstrip("\n").split("\n")[0]
    predicted = tokenizer.encode(normalize_answer(predicted_text), add_special_tokens=False)
    values = []
    for answer in answers:
        gold = tokenizer.encode(normalize_answer(answer), add_special_tokens=False)
        if not predicted or not gold:
            values.append(float(predicted == gold))
        else:
            common = sum((Counter(predicted) & Counter(gold)).values())
            values.append(2.0 * common / (len(predicted) + len(gold)))
    return {"f1": max(values)}


def main(dataset, default_storage="cpu", is_qwen=False, argv=None):
    args = make_parser(dataset, default_storage, is_qwen).parse_args(argv)
    validate_args(args)
    evaluation_path = dataset_file(dataset, args.dataset_path)
    rows = load_rows(dataset, evaluation_path, args.dataset_target_path)[:args.sample_limit]
    if not rows:
        raise ValueError("Evaluation dataset is empty")
    calibration_path = None
    calibration_rows = []
    if args.ratio_mode == "gss":
        calibration_path = dataset_file(args.calibration_dataset, args.calibration_dataset_path)
        calibration_rows = load_rows(args.calibration_dataset, calibration_path, args.calibration_dataset_target_path)[:args.calibration_samples]
        if len(calibration_rows) != args.calibration_samples:
            raise ValueError("Calibration dataset has fewer records than --calibration-samples")
    import torch
    import os
    os.environ["VLLM_ATTENTION_BACKEND"] = "XFORMERS"
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.worker.worker import Worker
    if not hasattr(Worker, "cachetune_paper_begin"):
        raise RuntimeError("Installed vLLM is a different checkout. Install this repository's CacheTune/vllm_blend with pip install -e .")
    from paper_algorithms import aggregate_spectral_statistics, select_shared_tokens, golden_section_search
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    llm = LLM(model=args.model_path, tensor_parallel_size=args.tensor_parallel_size,
              gpu_memory_utilization=args.gpu_memory_utilization, max_model_len=args.max_model_len,
              enforce_eager=args.enforce_eager, max_num_seqs=1, disable_custom_all_reduce=True)
    llm.set_tokenizer(tokenizer)
    # Probe the new RPC before collecting anything. An unrelated editable vLLM
    # installation must not silently run an older implementation.
    run_on_workers(llm, "cachetune_paper_release")
    run_tag = time.strftime("%Y%m%d_%H%M%S") + "_" + str(time.time_ns())
    disk_root = str((args.disk_root / run_tag).resolve())
    report = {"method": "paper", "status": "running", "dataset": dataset,
              "dataset_path": str(evaluation_path), "configuration": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "protocol": {"fft_axis": "token", "alpha_semantics": "floor(alpha*(N//2+1)) retained rFFT bins",
                           "attention_backend": "XFORMERS; absolute-position causal mask for selected queries",
                           "quality_metric": "QA: normalized model-token F1 without special tokens; summarization: ROUGE-L (MultiNews also ROUGE-1/2)",
                           "score": "mean_layers(normalize_tokens(0.5*(L2(K_low)+L2(V_low))))",
                           "selection": "floor(r*N) independently per chunk; stable token-index tie break",
                           "query": "not scored or loaded; always computed", "ttft": "first_token_time - first_scheduled_time",
                           "ttft_exclusions": "offline FFT, per-ratio cache compaction/serialization and GPU buffer allocation; actual layer reads/DMA/attention are timed",
                           "disk_cache": "buffered OS reads; no cold-storage claim; full CPU snapshots retained for replay, not an SSD-only memory-footprint benchmark", "disk_root": disk_root},
              "calibration": None, "samples": []}
    save_report(args.output_json, report)
    one_token = SamplingParams(temperature=0, max_tokens=1)

    def generate(request, max_tokens=1):
        params = one_token if max_tokens == 1 else SamplingParams(temperature=0, max_tokens=max_tokens)
        return llm.generate(prompt_token_ids=[request["tokens"]], sampling_params=params, use_tqdm=False)

    def collect(request, context_id):
        run_on_workers(llm, "cachetune_paper_begin", context_id=context_id)
        scores = []
        for index, chunk in enumerate(request["chunks"]):
            bos = tokenizer.bos_token_id
            leading = [bos] if index > 0 and bos is not None else []
            independent_tokens = leading + chunk
            llm.generate(prompt_token_ids=[independent_tokens], sampling_params=one_token, use_tqdm=False)
            statistics_by_rank = run_on_workers(llm, "cachetune_paper_append_chunk", start=len(leading), end=len(independent_tokens), score=True, alpha=args.alpha)
            scores.append(aggregate_spectral_statistics(statistics_by_rank))
        infos = run_on_workers(llm, "cachetune_paper_finalize", context_id=context_id, last_len=len(request["suffix"]))
        if any(info["total_len"] != len(request["tokens"]) for info in infos):
            raise RuntimeError("Collected KV/token layout differs across ranks or from assembled request")
        request.update(context_id=context_id, scores=scores, num_layers=infos[0]["num_layers"])

    def prepare(request, ratio):
        selected, reused = select_shared_tokens(request["scores"], ratio)
        context_len = sum(len(chunk) for chunk in request["chunks"])
        suffix_indices = torch.arange(context_len, len(request["tokens"]), dtype=torch.long)
        selected = torch.cat((selected.cpu(), suffix_indices))
        run_on_workers(llm, "cachetune_paper_prepare", context_id=request["context_id"],
                              final_indices_cpu=selected, last_len=len(request["suffix"]), recomp_ratio=ratio,
                              storage=args.storage, disk_root=disk_root, check_layers=[1])
        return selected, reused

    ratio = args.recomp_ratio
    try:
        if args.ratio_mode == "gss":
            calibration = [build_request(row, args.calibration_dataset, tokenizer, args, i) for i, row in enumerate(calibration_rows)]
            for i, request in enumerate(calibration):
                print("[paper] collecting calibration request", i, flush=True)
                collect(request, "calibration_" + str(i))
            objective_trace = []

            def objective(candidate):
                measurements = []
                for request in calibration:
                    for repeat in range(args.warmup + args.repeats):
                        prepare(request, candidate)
                        value = ttft(generate(request))
                        if repeat >= args.warmup:
                            measurements.append(value)
                mean = statistics.mean(measurements)
                objective_trace.append({"ratio": candidate, "mean_ttft_s": mean, "ttft_s": measurements})
                print("[paper GSS] ratio={:.6f} mean TTFT={:.6f}s".format(candidate, mean), flush=True)
                return mean

            # Profiling is only a warm-start prior, never the search objective.
            # A full-prefill TTFT amortized by tokens/layers estimates tc. The
            # storage adapter measures the selected medium directly for ti.
            first = calibration[0]
            run_on_workers(llm, "cachetune_paper_disable")
            dense_values = []
            for repeat in range(args.warmup + args.profile_repeats):
                value = ttft(generate(first))
                if repeat >= args.warmup:
                    dense_values.append(value)
            tc = statistics.median(dense_values) / (first["num_layers"] * len(first["tokens"]))
            transfer = run_on_workers(llm, "cachetune_paper_profile_transfer", context_id=first["context_id"],
                                      storage=args.storage, disk_root=disk_root, trials=args.profile_repeats)
            ti = max(item["seconds_per_token_layer"] for item in transfer)
            result = golden_section_search(objective, tc=tc, ti=ti, r_min=args.r_min, r_max=args.r_max, tolerance=args.gss_tolerance)
            ratio = result.ratio
            report["calibration"] = {"dataset": args.calibration_dataset, "dataset_path": str(calibration_path),
                "request_token_sha256": [request["token_sha256"] for request in calibration],
                "request_ids": [request["id"] for request in calibration], "tc_seconds": tc, "ti_seconds": ti,
                "profiling": {"tc_estimator": "median full-prefill TTFT / (layers * total tokens); amortized proxy includes fixed overhead",
                              "fixed_overhead": "not separately identified; included in amortized tc prior; measured GSS objective includes complete TTFT", "full_ttft_s": dense_values, "transfer_by_rank": transfer},
                "prior": result.prior, "ratio": ratio, "final_interval": list(result.interval), "midpoint_mean_ttft_s": result.value,
                "iterations": result.iterations, "trace": objective_trace}
            run_on_workers(llm, "cachetune_paper_release")
            save_report(args.output_json, report)
        report["selected_ratio"] = ratio
        for index, row in enumerate(rows):
            request = build_request(row, dataset, tokenizer, args, index)
            collect(request, "evaluation")
            selected, reused = prepare(request, ratio)
            cached = generate(request, args.max_tokens)
            cached_text, cached_time = cached[0].outputs[0].text, ttft(cached)
            run_on_workers(llm, "cachetune_paper_disable")
            full = generate(request, args.max_tokens)
            full_text, full_time = full[0].outputs[0].text, ttft(full)
            record = {"sample_id": index, "token_sha256": request["token_sha256"], "chunk_lengths": [len(chunk) for chunk in request["chunks"]],
                      "context_tokens": len(request["tokens"]) - len(request["suffix"]), "query_tokens": len(request["suffix"]),
                      "recompute_indices": selected.tolist(), "transfer_indices": reused.tolist(),
                      "cached_ttft_s": cached_time, "full_ttft_s": full_time,
                      "cached_text": cached_text, "full_text": full_text,
                      "cached_quality": quality_score(dataset, cached_text, request["answers"], tokenizer),
                      "full_quality": quality_score(dataset, full_text, request["answers"], tokenizer)}
            report["samples"].append(record)
            print("[paper] sample={} r={:.5f} cached={:.5f}s full={:.5f}s".format(index, ratio, cached_time, full_time), flush=True)
            run_on_workers(llm, "cachetune_paper_release", context_id="evaluation")
            save_report(args.output_json, report)
        report["status"] = "complete"
        calibration_hashes = set((report["calibration"] or {}).get("request_token_sha256", []))
        report["calibration_evaluation_overlap_token_sha256"] = sorted(
            calibration_hashes.intersection(sample["token_sha256"] for sample in report["samples"]))
        report["summary"] = {"samples": len(report["samples"]),
            "mean_cached_ttft_s": statistics.mean(s["cached_ttft_s"] for s in report["samples"]),
            "mean_full_ttft_s": statistics.mean(s["full_ttft_s"] for s in report["samples"])}
        save_report(args.output_json, report)
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = type(exc).__name__ + ": " + str(exc)
        save_report(args.output_json, report)
        raise
    finally:
        run_on_workers(llm, "cachetune_paper_release")
    print("[paper] complete:", args.output_json, flush=True)
    return 0
