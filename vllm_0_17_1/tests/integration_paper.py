"""Single-GPU paper-path correctness checks using public, synthetic text.

Examples:
    python tests/integration_paper.py --model Qwen/Qwen3-0.6B --output-json results/dense.json
    python tests/integration_paper.py --model Qwen/Qwen3.5-0.8B --output-json results/hybrid.json

This is a functional regression, not a latency, throughput, or accuracy benchmark.
Only three context chunks are collected offline. Every question is supplied online.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Public model ID or local model directory")
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.30)
    parser.add_argument("--dtype", default="auto", choices=("auto", "float16", "bfloat16", "float32"))
    parser.add_argument("--logit-atol", type=float, default=0.0,
                        help="Explicit ratio-1 maximum-logit-error allowance; default requires exact equality")
    opts = parser.parse_args()
    if not 128 <= opts.max_model_len <= 2048:
        parser.error("--max-model-len must be in [128, 2048]")
    if not 2 <= opts.max_new_tokens <= 4:
        parser.error("--max-new-tokens must be in [2, 4] so a real decode step is checked")
    if not 0 < opts.gpu_memory_utilization < 1:
        parser.error("--gpu-memory-utilization must lie in (0, 1)")
    if not math.isfinite(opts.logit_atol) or opts.logit_atol < 0:
        parser.error("--logit-atol must be finite and nonnegative")
    return opts


def digest_ids(ids):
    return hashlib.sha256(json.dumps(list(ids), separators=(",", ":")).encode()).hexdigest()


def worker_fingerprints(worker):
    """Callable native collective_rpc: return hashes, never tensor payloads.

    Hashes let the JSON record distinguish equal full logit vectors and native
    recurrent states from completions that merely share an argmax.
    """
    import torch

    outer = worker.model_runner.model
    target = getattr(outer, "language_model", outer)
    state = target.model._cachetune_state

    def fingerprint(tensor):
        value = tensor.detach().cpu().contiguous()
        return {"shape": list(value.shape), "dtype": str(value.dtype),
                "sha256": hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()}

    logits = getattr(state, "_probe_logits", None)
    if logits is None:
        raise RuntimeError("No captured first-token logits for fingerprinting")
    result = {"logits": fingerprint(logits)}
    recurrent = getattr(state, "_recurrent_snapshot", None)
    if recurrent:
        result["recurrent"] = {
            str(layer): {name: fingerprint(tensor) for name, tensor in sorted(row.items())}
            for layer, row in sorted(recurrent.items())}
    return result


def make_samples(tokenizer, boundary_token):
    marker, marker_info = boundary_token(tokenizer)
    examples = (
        {
            "name": "A",
            "context": [
                "Read the records below. Answer the final question briefly using only these records.\n",
                "Record North: The river station is named Alder. Its code is 731. The keeper is Mira.\n",
                "Record South: The hill station is named Birch. Its code is 246. The keeper is Tomas.\n",
            ],
            "question": "Question: What is the code of the river station?\nAnswer:",
            "alternative_question": "Question: Who keeps the hill station, and what is its name?\nAnswer:",
        },
        {
            "name": "B",
            "context": [
                "Read the inventory below. Answer the final question briefly using only these records.\n",
                "Record East: The blue crate contains brass keys. It weighs 17 kilograms and belongs to Lena.\n",
                "Record West: The red crate contains silver bells. It weighs 29 kilograms and belongs to Omar.\n",
            ],
            "question": "Question: Who owns the red crate?\nAnswer:",
            "alternative_question": "Question: What is inside the blue crate?\nAnswer:",
        },
    )
    samples = []
    for example in examples:
        encoded = [list(tokenizer.encode(text, add_special_tokens=False))
                   for text in example["context"]]
        if not all(encoded):
            raise AssertionError("Every public context chunk must contain tokens")
        chunks = [[marker] + ids for ids in encoded]
        context_ids = chunks[0] + [token for chunk in chunks[1:] for token in chunk[1:]]
        question = list(tokenizer.encode(example["question"], add_special_tokens=False))
        alternative = list(tokenizer.encode(example["alternative_question"], add_special_tokens=False))
        if not question or not alternative or question == alternative:
            raise AssertionError("The two online questions must be distinct and nonempty")
        lengths = [len(chunk) - int(i > 0) for i, chunk in enumerate(chunks)]
        assert sum(lengths) == len(context_ids)
        samples.append({**example, "chunks": chunks, "context_ids": context_ids,
                        "chunk_lengths": lengths, "question_ids": question,
                        "alternative_question_ids": alternative,
                        "tokenization": marker_info})
    return samples


class PaperRegression:
    def __init__(self, llm, entry, opts, layer_types, report):
        self.llm, self.entry, self.opts = llm, entry, opts
        self.layer_types = layer_types
        self.hybrid = "linear_attention" in layer_types
        self.full_ids = [i for i, kind in enumerate(layer_types) if kind == "full_attention"]
        self.check = 0 if self.hybrid else 1
        self.report = report
        self.collection_serial = 0

    def rpc(self, method, *args, **kwargs):
        return self.entry.rpc(self.llm, method, *args, **kwargs)

    def collect(self, sample):
        self.rpc("cachetune_start_collect")
        self.collection_serial += 1
        record = {"sample": sample["name"], "serial": self.collection_serial,
                  "context_only": True, "chunk_records": []}
        for index, chunk in enumerate(sample["chunks"]):
            # Deliberately never call generate on either online question here.
            self.entry.generate_ids(self.llm, chunk, 1)
            appended = self.rpc("cachetune_append_chunk", chunk, drop_bos=index > 0)
            record["chunk_records"].append({"index": index, "encoded_tokens": len(chunk),
                                             "sha256": digest_ids(chunk), "append": appended})
        assert appended["collected_tokens"] == len(sample["context_ids"]), appended
        assert appended["chunks"] == 3, appended
        self.report["collections"].append(record)

    def prepare(self, sample, ratio, mode="gpu", alternative=False):
        suffix = sample["alternative_question_ids"] if alternative else sample["question_ids"]
        ids = sample["context_ids"] + suffix
        kwargs = dict(alpha=0.5, sink_size=0, check_layer=self.check,
                      cache_mode=mode, retain_full_cache=True)
        if self.hybrid:
            kwargs["hybrid_policy"] = "block"
        selection = self.rpc("cachetune_prepare", ids, len(suffix), ratio, **kwargs)
        context = len(sample["context_ids"])
        selected, reused = selection["selected_indices"], selection["reused_indices"]
        assert selected == sorted(set(selected)) and reused == sorted(set(reused))
        assert sorted(selected + reused) == list(range(len(ids)))
        assert selected[-len(suffix):] == list(range(context, len(ids)))
        budgets = [int(length * ratio) for length in sample["chunk_lengths"]]
        assert selection["chunk_context_lengths"] == sample["chunk_lengths"], selection
        assert selection["chunk_budgets"] == budgets, selection
        assert len(selected) == sum(budgets) + len(suffix), selection
        start = 0
        for length, budget in zip(sample["chunk_lengths"], budgets):
            assert sum(start <= i < start + length for i in selected) == budget
            start += length
        assert selection["collected_tokens"] == context
        assert selection["suffix_cached"] is False
        assert selection["score_layer"] is None
        assert selection["score_layers"] == list(range(len(self.full_ids)))
        assert selection["signal"] == "K/V-all-layers"
        assert selection["selection_source"] == "frequency"
        if self.hybrid:
            assert selection["score_actual_layers"] == self.full_ids
            assert selection["scored_layer_indices"] == self.full_ids
            assert selection["hybrid_paper_default"] is True
            assert selection["hybrid_experimental"] is False
            assert selection["first_fusion_layer"] == 0
        else:
            assert selection["first_fusion_layer"] == self.check + 1
        return ids, selection

    def run(self, ids, *, cached, reference):
        from vllm import SamplingParams

        if cached:
            self.rpc("cachetune_enable", ids)
        else:
            self.rpc("cachetune_disable")
        self.rpc("cachetune_probe_start")
        if self.hybrid:
            self.rpc("cachetune_recurrent_probe_start")
        params = SamplingParams(temperature=0, min_tokens=2, max_tokens=self.opts.max_new_tokens)
        outputs = self.llm.generate([{"prompt_token_ids": list(ids)}], params, use_tqdm=False)
        assert len(outputs) == 1 and list(outputs[0].prompt_token_ids) == ids
        completion = outputs[0].outputs[0]
        assert 2 <= len(completion.token_ids) <= self.opts.max_new_tokens
        assert completion.finish_reason in ("stop", "length")
        logits = self.rpc("cachetune_probe_finish", set_reference=reference)
        assert logits["finite"], logits
        if not reference:
            assert math.isfinite(logits["relative_l2"]), logits
            assert math.isfinite(logits["max_absolute_error"]), logits
        result = {"token_ids": list(completion.token_ids), "text": completion.text,
                  "finish_reason": completion.finish_reason, "logits": logits}
        if self.hybrid:
            recurrent = self.rpc("cachetune_recurrent_probe_finish", set_reference=reference)
            assert recurrent["finite"] and recurrent["endpoint"] == "prefill_end", recurrent
            expected = {str(i) for i, kind in enumerate(self.layer_types) if kind == "linear_attention"}
            assert set(recurrent["layers"]) == expected
            assert all(set(row) == {"conv", "ssm"} for row in recurrent["layers"].values())
            result["recurrent"] = recurrent
        result["fingerprints"] = self.rpc("cachetune_fingerprints")
        return result

    def rows(self, selection):
        diagnostics = self.rpc("cachetune_diagnostics")
        rows = diagnostics["layers"]
        n, m = selection["prompt_tokens"], len(selection["selected_indices"])
        assert len(rows) == len(self.layer_types), diagnostics
        assert diagnostics["native_forward_calls"] >= 1, diagnostics
        for actual, (kind, row) in enumerate(zip(self.layer_types, rows)):
            assert row["layer"] == actual, row
            if self.hybrid:
                full = kind == "full_attention"
                expected_rows = m if full else n
                assert row["layer_type"] == kind, row
                assert row["qkv_rows"] == row["query_rows"] == row["mlp_rows"] == expected_rows, row
                assert row["output_rows"] == n, row
                assert row["kv_rows"] == (n if full else None), row
                assert row["recurrent_rows"] == (None if full else n), row
                assert row["paged_verified"] is full, row
            else:
                assert row["qkv_rows"] == (m if actual > self.check else n), row
                assert row["query_rows"] == (m if actual >= self.check else n), row
                assert row["kv_rows"] == n, row
                assert row["mlp_rows"] == (m if actual >= self.check else n), row
                assert row["paged_verified"] is (actual >= self.check), row
        expected_merges = list(range(0 if self.hybrid else self.check + 1, len(self.full_ids)))
        assert diagnostics["merged_layers"] == expected_merges, diagnostics
        return diagnostics

    def equal(self, expected, current, *, exact=True):
        assert current["token_ids"] == expected["token_ids"], (expected, current)
        error = current["logits"]["max_absolute_error"]
        assert error <= (0.0 if exact else self.opts.logit_atol), current["logits"]
        if exact or self.opts.logit_atol == 0:
            assert current["fingerprints"]["logits"] == expected["fingerprints"]["logits"]
        if self.hybrid:
            assert current["recurrent"]["all_equal"] is True, current["recurrent"]
            assert current["fingerprints"]["recurrent"] == expected["fingerprints"]["recurrent"]

    def exercise(self, samples):
        # 100% selected must recover the model's native full-context execution.
        for sample in samples:
            self.collect(sample)
            ids, selection = self.prepare(sample, 1.0)
            full = self.run(ids, cached=False, reference=True)
            cached = self.run(ids, cached=True, reference=False)
            diagnostics = self.rows(selection)
            self.report["ratio1"].append({"sample": sample["name"], "selection": selection,
                                           "full": full, "cached": cached, "diagnostics": diagnostics})
            self.equal(full, cached, exact=False)
            self.rpc("cachetune_reset")

        sample = samples[0]
        self.collect(sample)
        anchor = None
        original_selection = None
        for mode in ("gpu", "cpu_pipeline", "cpu_serial"):
            ids, selection = self.prepare(sample, 0.15, mode)
            for repeat in range(2):
                output = self.run(ids, cached=True, reference=anchor is None)
                diagnostics = self.rows(selection)
                if mode == "gpu":
                    assert diagnostics["transferred_bytes"] == 0, diagnostics
                    if self.hybrid:
                        assert diagnostics["branch_transferred_bytes"] == 0, diagnostics
                else:
                    assert diagnostics["transferred_bytes"] > 0, diagnostics
                self.report["cache_modes"].append({"mode": mode, "repeat": repeat,
                                                    "selection": selection, "output": output,
                                                    "diagnostics": diagnostics})
                if anchor is None:
                    anchor, original_selection = output, selection
                else:
                    self.equal(anchor, output)
                    assert selection["selected_indices"] == original_selection["selected_indices"]

        serial = self.collection_serial
        ids, alternate = self.prepare(sample, 0.15, alternative=True)
        original_context = original_selection["selected_indices"][:-len(sample["question_ids"])]
        alternate_context = alternate["selected_indices"][:-len(sample["alternative_question_ids"])]
        assert alternate_context == original_context
        assert self.collection_serial == serial, "An online question must not trigger offline collection"
        changed = self.run(ids, cached=True, reference=True)
        repeated = self.run(ids, cached=True, reference=False)
        self.equal(changed, repeated)
        self.report["new_query"] = {"collection_serial": serial,
                                     "context_selection": original_context,
                                     "original_selection": original_selection,
                                     "alternative_selection": alternate,
                                     "output": changed, "repeat": repeated,
                                     "diagnostics": self.rows(alternate)}
        self.rpc("cachetune_reset")

        # Cache and recurrent/probe state for B must not contaminate the next A.
        for index in (0, 1, 0):
            sample = samples[index]
            self.collect(sample)
            ids, selection = self.prepare(sample, 0.15)
            output = self.run(ids, cached=True, reference=True)
            diagnostics = self.rows(selection)
            self.report["aba"].append({"sample": sample["name"], "selection": selection,
                                       "output": output, "diagnostics": diagnostics})
            self.rpc("cachetune_reset")
        before, after = self.report["aba"][0], self.report["aba"][2]
        assert before["selection"]["selected_indices"] == after["selection"]["selected_indices"]
        assert before["output"]["token_ids"] == after["output"]["token_ids"]
        assert before["output"]["fingerprints"] == after["output"]["fingerprints"]


def main():
    opts = arguments()
    # Must be set before vLLM imports/worker startup.
    os.environ["CACHETUNE_ENABLE_ADAPTER"] = "1"
    os.environ["CACHETUNE_VALIDATE_PAGED"] = "1"
    os.environ["CACHETUNE_FORCE_FLASHINFER"] = "0"
    os.environ["CACHETUNE_COMPARE_DENSE"] = "0"
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "example"))
    report = {"schema_version": 1, "status": "RUNNING", "model": opts.model,
              "scope": "Single-GPU functional correctness; not a performance or task-quality benchmark",
              "protocol": {"context_chunks": 3, "context_only_collection": True,
                           "alpha": 0.5, "sparse_ratio": 0.15, "sink_size": 0,
                           "budget": "floor per independent context chunk",
                           "online_suffix": "all query tokens recomputed; never collected offline",
                           "hybrid_policy": "block", "ratio1_logit_atol": opts.logit_atol},
              "collections": [], "ratio1": [], "cache_modes": [], "aba": []}
    llm = None
    try:
        import torch
        import vllm
        from transformers import AutoTokenizer
        from vllm import LLM
        from cachetune import register_models
        import blend_samsum_freq as entry

        entry.validate_runtime_version(vllm.__version__)
        configuration = entry.read_model_config(opts.model)
        metadata = entry.validate_model_mode(configuration, False)
        text_config = configuration.get("text_config", configuration)
        hybrid = metadata["is_qwen35"]
        num_layers = text_config["num_hidden_layers"]
        if hybrid:
            layer_types = text_config.get("layer_types")
            if layer_types is None:
                interval = text_config.get("full_attention_interval", 4)
                layer_types = ["full_attention" if (i+1) % interval == 0 else "linear_attention"
                               for i in range(num_layers)]
            layer_types = list(layer_types)
        else:
            layer_types = ["full_attention"] * num_layers
        assert set(layer_types) <= {"linear_attention", "full_attention"}
        tokenizer = AutoTokenizer.from_pretrained(opts.model)
        samples = make_samples(tokenizer, entry.boundary_token)
        for sample in samples:
            for suffix in (sample["question_ids"], sample["alternative_question_ids"]):
                assert len(sample["context_ids"]) + len(suffix) + opts.max_new_tokens <= opts.max_model_len
        report["inputs"] = samples
        report["layer_types"] = layer_types
        report["versions"] = {"python": sys.version, "torch": torch.__version__,
                              "vllm": vllm.__version__, "cuda": torch.version.cuda}
        register_models()
        engine = dict(model=opts.model, dtype=opts.dtype, max_model_len=opts.max_model_len,
                      gpu_memory_utilization=opts.gpu_memory_utilization, max_num_seqs=1,
                      max_num_batched_tokens=opts.max_model_len,
                      tensor_parallel_size=1, pipeline_parallel_size=1,
                      enforce_eager=True, enable_prefix_caching=False,
                      enable_chunked_prefill=False, async_scheduling=False,
                      seed=0, worker_extension_cls=entry.WORKER_EXTENSION,
                      attention_config={"backend": "FLASH_ATTN"})
        if hybrid:
            engine.update(language_model_only=True, limit_mm_per_prompt={"image": 0, "video": 0},
                          enable_mm_embeds=False)
        report["engine_options"] = engine
        llm = LLM(**engine)
        PaperRegression(llm, entry, opts, layer_types, report).exercise(samples)
        report["status"] = "PASS"
    except Exception:
        report.update(status="FAIL", traceback=traceback.format_exc())
        raise
    finally:
        if llm is not None:
            try:
                entry.rpc(llm, "cachetune_reset")
            except Exception:
                report["cleanup_traceback"] = traceback.format_exc()
        opts.output_json.parent.mkdir(parents=True, exist_ok=True)
        opts.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(opts.output_json)}))


if __name__ == "__main__":
    main()
