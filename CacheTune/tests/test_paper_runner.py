"""CPU-only contracts for paper-route inputs, dispatch, and request assembly."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

EXAMPLES = Path(__file__).resolve().parents[1] / "example"
SPEC = importlib.util.spec_from_file_location("cachetune_paper_runner_under_test", EXAMPLES / "paper_runner.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class Tokenizer:
    bos_token_id = 1

    def encode(self, text, add_special_tokens=True):
        return ([self.bos_token_id] if add_special_tokens else []) + [ord(c) + 10 for c in text]


def args_for(dataset="samsum", extra=None):
    return runner.make_parser(dataset, "cpu", False).parse_args(["--model-path", "test-model"] + (extra or []))


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def test_json_array_and_nested_answers(self):
        path = self.root / "rows.json"
        rows = [{"ctxs": [{"text": "hello"}], "question": "summarize", "answers": ["summary"]}]
        path.write_text(json.dumps(rows), encoding="utf-8")
        self.assertEqual(runner.load_rows("samsum", path), rows)
        self.assertEqual(runner.answer_texts({"aliases": ["a", {"answer": "b"}], "unused": None}), ["a", "b"])
        path.write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "JSON array"):
            runner.load_rows("samsum", path)

    def test_multinews_separate_source_and_target(self):
        source, target = self.root / "test.src.cleaned", self.root / "test.tgt"
        source.write_text("firstNEWLINE_CHARline ||||| second\nthird\n", encoding="utf-8")
        target.write_text("summary one\nsummaryNEWLINE_CHARtwo\n", encoding="utf-8")
        rows = runner.load_rows("multinews", source)
        self.assertEqual(rows[0], {"docs": ["first\nline", "second"], "answers": ["summary one"]})
        self.assertEqual(rows[1]["answers"], ["summary\ntwo"])
        target.write_text("only one\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "line counts"):
            runner.load_rows("multinews", source, target)

    def test_dataset_override_is_resolved_and_missing_fails(self):
        path = self.root / "custom.json"
        path.write_text("[]", encoding="utf-8")
        self.assertEqual(runner.dataset_file("samsum", path), path.resolve())
        with self.assertRaises(FileNotFoundError):
            runner.dataset_file("samsum", self.root / "missing.json")


class QualityTests(unittest.TestCase):
    def test_qa_keeps_first_token_and_names_starting_no(self):
        # Qwen tokenizers need not prepend BOS; removing index 0 changes F1.
        utils = types.ModuleType("utils")
        utils.compute_rl = lambda prediction, answer: 0.0
        utils.normalize_answer = lambda text: text.lower().strip()
        with patch.dict(sys.modules, {"utils": utils}):
            self.assertEqual(runner.quality_score("wikimqa", "Norway", ["Norway"], Tokenizer()), {"f1": 1.0})
            self.assertAlmostEqual(runner.quality_score("musique", "abc", ["zbc"], Tokenizer())["f1"], 2 / 3)


class RequestTests(unittest.TestCase):
    def assert_layout(self, request):
        chunks = request["chunks"]
        joined = [token for chunk in chunks for token in chunk]
        self.assertEqual(joined + request["suffix"], request["tokens"])
        offset = 0
        for chunk in chunks:
            self.assertEqual(request["tokens"][offset:offset + len(chunk)], chunk)
            offset += len(chunk)
        self.assertEqual(request["tokens"][offset:], request["suffix"])
        self.assertEqual(request["tokens"].count(Tokenizer.bos_token_id), 1)

    def test_samsum_query_never_enters_reusable_chunks(self):
        row = {"ctxs": [{"text": "offline dialogue"}], "question": "UNIQUE_QUERY_A", "answers": ["gold"]}
        original = runner.build_request(row, "samsum", Tokenizer(), args_for(), 2)
        changed = runner.build_request(dict(row, question="UNIQUE_QUERY_B"), "samsum", Tokenizer(), args_for(), 2)
        self.assert_layout(original)
        self.assertEqual(original["chunks"], changed["chunks"])
        self.assertNotEqual(original["suffix"], changed["suffix"])
        self.assertNotEqual(original["token_sha256"], changed["token_sha256"])
        self.assertEqual(original["chunks"][1], Tokenizer().encode("offline dialogue", add_special_tokens=False))

    def test_context_truncation_keeps_suffix_complete(self):
        row = {"ctxs": [{"text": "x" * 500}, {"text": "must not enter"}], "question": "full query", "answers": ["gold"]}
        args = args_for(extra=["--max-context-tokens", "120"])
        request = runner.build_request(row, "samsum", Tokenizer(), args, "truncated")
        self.assert_layout(request)
        self.assertEqual(sum(map(len, request["chunks"])), 120)
        self.assertEqual(len(request["chunks"]), 2)
        self.assertEqual(request["suffix"], Tokenizer().encode("\n\nfull query", add_special_tokens=False))

    def test_all_question_answering_formats(self):
        for dataset in ("musique", "wikimqa", "hotpotqa"):
            with self.subTest(dataset=dataset):
                row = {"ctxs": [{"title": "A", "text": "fact one"}, {"title": "B", "text": "fact two"}], "question": "Which fact", "answers": ["one"]}
                request = runner.build_request(row, dataset, Tokenizer(), args_for(dataset), 0)
                self.assert_layout(request)
                self.assertEqual(len(request["chunks"]), 3)
                suffix = "".join(chr(token - 10) for token in request["suffix"])
                self.assertIn("Question: Which fact?\nAnswer:", suffix)
        row = {"context": [["title", ["sentence 1.", "sentence 2."]]], "question": "Q?", "answer": "A"}
        request = runner.build_request(row, "hotpotqa", Tokenizer(), args_for("hotpotqa"), 1)
        self.assertEqual(request["answers"], ["A"])
        self.assertEqual(request["chunks"][1], Tokenizer().encode("title\n\nsentence 1.sentence 2.\n\n", add_special_tokens=False))
        self.assert_layout(request)

    def test_multinews_json_docs_and_ctxs_formats(self):
        for row in ({"docs": ["first", "second"], "summary": "gold"},
                    {"ctxs": [{"text": "first"}, {"text": "second"}], "answers": ["gold"]}):
            request = runner.build_request(row, "multinews", Tokenizer(), args_for("multinews"), 0)
            self.assert_layout(request)
            self.assertEqual(len(request["chunks"]), 3)
            self.assertEqual(request["answers"], ["gold"])

    def test_missing_answers_or_no_context_are_rejected(self):
        base = {"ctxs": [{"text": "hello"}], "question": "summarize"}
        with self.assertRaisesRegex(ValueError, "Missing reference"):
            runner.build_request(base, "samsum", Tokenizer(), args_for(), 0)
        with self.assertRaisesRegex(ValueError, "No reusable"):
            runner.build_request(dict(base, ctxs=[], answers=["gold"]), "samsum", Tokenizer(), args_for(), 0)


class ParserTests(unittest.TestCase):
    def test_paper_defaults_and_explicit_fixed_route(self):
        args = args_for()
        runner.validate_args(args)
        self.assertEqual((args.alpha, args.recomp_ratio, args.r_min), (0.5, 0.15, 0.15))
        self.assertEqual((args.ratio_mode, args.calibration_dataset, args.calibration_samples), ("gss", "samsum", 10))
        qwen = runner.make_parser("hotpotqa", "disk", True).parse_args(["--model-path", "test", "--ratio-mode", "fixed"])
        self.assertEqual((qwen.tensor_parallel_size, qwen.storage, qwen.ratio_mode), (2, "disk", "fixed"))

    def test_invalid_values_fail_before_model_imports(self):
        cases = {"alpha": [0, 1.1, float("nan")], "r_min": [0.1, 1.0], "r_max": [0.1, 1.1],
                 "recomp_ratio": [0.1, 1.1], "gss_tolerance": [0, 2], "warmup": [-1],
                 "repeats": [0], "profile_repeats": [0], "calibration_samples": [0],
                 "sample_limit": [0], "tensor_parallel_size": [0], "max_num_seqs": [2]}
        for field, values in cases.items():
            for value in values:
                with self.subTest(field=field, value=value):
                    args = args_for()
                    setattr(args, field, value)
                    with self.assertRaises(ValueError):
                        runner.validate_args(args)

    def test_help_for_all_nine_same_file_routes_without_model_imports(self):
        paths = sorted(EXAMPLES.glob("blend*freq*.py"))
        self.assertEqual(len(paths), 9)
        # An import blocker makes this meaningful even on hosts with PyTorch.
        code = (
            "import builtins,runpy,sys; original=builtins.__import__; "
            "blocked={'torch','vllm','transformers','rouge_score'}; "
            "builtins.__import__=lambda name,*a,**k: (_ for _ in ()).throw(AssertionError('heavy import: '+name)) if name.split('.')[0] in blocked else original(name,*a,**k); "
            "path=sys.argv[1];sys.path.insert(0,str(__import__('pathlib').Path(path).parent)); "
            "sys.argv=[path,'--method=paper','--help'];runpy.run_path(path,run_name='__main__')"
        )
        for path in paths:
            with self.subTest(driver=path.name):
                result = subprocess.run([sys.executable, "-c", code, str(path)], capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr)
                for option in ("--ratio-mode", "--calibration-dataset-path", "--alpha", "--storage", "--profile-repeats"):
                    self.assertIn(option, result.stdout)


class MeasurementTests(unittest.TestCase):
    def test_ttft_is_first_token_minus_first_scheduled(self):
        output = [types.SimpleNamespace(metrics=types.SimpleNamespace(first_token_time=5.5, first_scheduled_time=4.0))]
        self.assertEqual(runner.ttft(output), 1.5)
        for value in (None, 4.0, float("inf")):
            output[0].metrics.first_token_time = value
            with self.assertRaises(RuntimeError):
                runner.ttft(output)

    def test_json_report_cannot_silently_persist_nonfinite_measurements(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "results" / "run.json"
            runner.save_report(path, {"status": "complete", "value": 0.1})
            self.assertEqual(json.loads(path.read_text())["status"], "complete")
            with self.assertRaises(ValueError):
                runner.save_report(path, {"value": float("nan")})
            self.assertEqual(json.loads(path.read_text())["status"], "complete")


if __name__ == "__main__":
    unittest.main()
