"""Dataset and scoring helpers for the original SAMSum experiment protocol."""

import json
from functools import lru_cache


def load_dataset(dataset_path):
    print("Loading dataset:", dataset_path)
    with open(dataset_path, encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, list):
        raise ValueError("SAMSum input must be a JSON list of examples")
    return data


def build_fewshot_prompt(example):
    # Preserve the old formatting exactly, including the query's two newlines.
    return [f"{ctx['text']}" for ctx in example["ctxs"]], "\n\n" + example["question"]


def scored_first_line(text):
    """The legacy SAMSum script scores the first non-leading-newline line."""
    return text.lstrip("\n").split("\n")[0]


@lru_cache(maxsize=1)
def _rouge_scorer():
    from rouge_score import rouge_scorer

    return rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def compute_rl(pred, gold):
    return _rouge_scorer().score(gold, pred)["rougeL"].fmeasure
