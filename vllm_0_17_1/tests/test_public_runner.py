"""CPU-only public-runner tokenization and offline/query separation checks."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from example import run_spectral


class Tokenizer:
    bos_token_id = 1

    def encode(self, text, add_special_tokens=False):
        if add_special_tokens:
            raise AssertionError("Public requests explicitly control special tokens")
        return [ord(char) + 10 for char in text]


class NoBOSTokenizer(Tokenizer):
    bos_token_id = None

    @property
    def eos_token_id(self):
        raise AssertionError("Generic CacheTune must not use EOS as a BOS fallback")


class Worker:
    """Minimal request lifecycle probe; no model or timing emulation."""
    def __init__(self):
        self.events, self.parts, self.collected_inputs = [], [], []
        self.mode = None

    def collective_rpc(self, method, args, kwargs):
        self.events.append(method)
        if method == "cachetune_reset":
            self.parts = []
        elif method == "cachetune_start_collect":
            self.mode = "collect"
        elif method == "cachetune_append_chunk":
            ids = kwargs["expected_ids"]
            assert ids == self.last_generated
            self.parts.append(ids[int(kwargs["drop_bos"]):])
        elif method == "cachetune_prepare":
            n = len(kwargs["prompt_ids"])
            end = n - kwargs["suffix_len"]
            assert [t for part in self.parts for t in part] == kwargs["prompt_ids"][:end]
            lengths = [len(part) for part in self.parts]
            selected, start = [], 0
            for length in lengths:
                selected.extend(range(start, start + int(length * kwargs["ratio"])))
                start += length
            selected.extend(range(end, n))
            self.selection = {
                "selected_indices": selected,
                "reused_indices": [i for i in range(end) if i not in selected],
                "chunk_context_lengths": lengths,
                "score_layer": None, "signal": "K/V-all-layers",
                "selection_source": "frequency",
            }
            return [self.selection]
        elif method == "cachetune_enable":
            self.mode = "cached"
        elif method == "cachetune_disable":
            self.mode = "full"
        elif method == "cachetune_diagnostics":
            return [{"mode": self.mode}]
        return [None]

    @staticmethod
    def generate(worker, ids, max_tokens):
        worker.last_generated = list(ids)
        if worker.mode == "collect":
            worker.collected_inputs.append(list(ids))
        return {"token_ids": [12], "text": "answer"}


class PublicRunnerTests(unittest.TestCase):
    def test_import_uses_real_example_module_not_root_cli_wrapper(self):
        self.assertTrue(callable(run_spectral.entry.read_model_config))
        self.assertEqual(Path(run_spectral.entry.__file__).parent.name, "example")

    def test_bos_retained_once_and_dropped_from_later_independent_chunks(self):
        request = run_spectral.build_request(
            {"chunks": ["abc", "def"], "query": "Q?"}, Tokenizer())
        encode = Tokenizer().encode
        self.assertEqual(request["chunks"], [[1] + encode("abc"), [1] + encode("def")])
        self.assertEqual(request["drop_bos_flags"], [False, True])
        self.assertEqual(request["prompt_ids"], [1] + encode("abcdefQ?"))
        self.assertEqual(request["prompt_ids"].count(1), 1)

    def test_no_bos_means_no_inserted_eos_and_no_dropped_document_tokens(self):
        request = run_spectral.build_request(
            {"chunks": ["abc", "def"], "query": "Q?"}, NoBOSTokenizer())
        encode = NoBOSTokenizer().encode
        self.assertEqual(request["chunks"], [encode("abc"), encode("def")])
        self.assertEqual(request["drop_bos_flags"], [False, False])
        self.assertEqual(request["prompt_ids"], encode("abcdefQ?"))
        self.assertFalse(request["tokenization"]["eos_fallback_used"])

    def test_suffix_is_never_an_independent_collection_input(self):
        for tokenizer in (Tokenizer(), NoBOSTokenizer()):
            with self.subTest(bos=tokenizer.bos_token_id):
                request = run_spectral.build_request(
                    {"chunks": ["abc", "def"], "query": "QUESTION"}, tokenizer)
                worker = Worker()
                info, *_ = run_spectral.entry.run_cached_pair(
                    worker, request, .5, 2, generate=Worker.generate)
                self.assertEqual(worker.collected_inputs, request["chunks"])
                end = len(request["prompt_ids"]) - request["suffix_len"]
                self.assertEqual([t for part in worker.parts for t in part], request["prompt_ids"][:end])
                self.assertEqual(info["selected_indices"][-request["suffix_len"]:],
                                 list(range(end, len(request["prompt_ids"]))))

    def test_each_chunk_receives_its_own_floor_budget(self):
        request = run_spectral.build_request(
            {"chunks": ["abcde", "fghij"], "query": "Q"}, NoBOSTokenizer())
        info, *_ = run_spectral.entry.run_cached_pair(
            Worker(), request, .15, 2, generate=Worker.generate)
        self.assertEqual(info["selected_context_tokens"], 0)
        self.assertEqual(info["selected_indices"], [10])

    def test_explicit_drop_flags_are_validated_before_worker_mutation(self):
        request = run_spectral.build_request(
            {"chunks": ["abc", "def"], "query": "Q"}, NoBOSTokenizer())
        for flags in ([False], [0, 0], "false"):
            request["drop_bos_flags"] = flags
            worker = Worker()
            with self.assertRaisesRegex(ValueError, "drop_bos_flags"):
                run_spectral.entry.run_cached_pair(worker, request, .5, 2,
                                                   generate=Worker.generate)
            self.assertEqual(worker.events, [])

    def test_empty_or_invalid_request_is_rejected(self):
        for request in ({"chunks": [], "query": "Q"},
                        {"chunks": [""], "query": "Q"},
                        {"chunks": ["abc"], "query": ""}):
            with self.assertRaises(ValueError):
                run_spectral.build_request(request, NoBOSTokenizer())


if __name__ == "__main__":
    unittest.main()
