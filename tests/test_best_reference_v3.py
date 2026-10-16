"""Offline tests: best-reference policy and frozen-source v3 batch driver.

Run: python -m unittest discover -s tests -q
No GPU or Hugging Face model weights are needed.
"""
import csv
import io
import json
import math
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import benchmark_celerity_longbench_v2 as v2
import benchmark_celerity_longbench_v3 as v3
import longbench_perplexity as lb
import verify_longbench_results as auditor


class FakeDataset:
    def __init__(self, examples):
        self.examples = examples

    def __getitem__(self, index):
        return self.examples[index]

    def __len__(self):
        return len(self.examples)

    def __iter__(self):
        return iter(self.examples)

    def select(self, indices):
        return FakeDataset([self.examples[i] for i in indices])


class FakeTokenizer:
    bos_token_id = 1

    def __call__(self, text, add_special_tokens=False, **kwargs):
        return {"input_ids": [1] * len(text)}


class FakeModel:
    config = SimpleNamespace(max_position_embeddings=128)


class BestReferenceTests(unittest.TestCase):
    def test_cli_best_scores_all_distinct_and_uses_minimum_per_token_nll(self):
        dataset = FakeDataset([{
            "_id": "x", "context": "context", "input": "question",
            "answers": ["A", "BBBB", "BBBB", "CCCCCCCC"],
        }])
        scores = {"A": (5.0, 1), "BBBB": (8.0, 4), "CCCCCCCC": (12.0, 8)}
        scored = []

        def fake_score(**kw):
            scored.append(kw["answer"])
            return scores[kw["answer"]]

        with tempfile.TemporaryDirectory() as td:
            output = Path(td) / "best_summary.jsonl"
            argv = [
                "longbench_perplexity.py", "--model", "tiny", "--mode", "answer",
                "--tasks", "narrativeqa", "--max-length", "64", "--stride", "8",
                "--reference-policy", "best", "--output-jsonl", str(output),
            ]
            with (patch.object(sys, "argv", argv),
                  patch.object(lb, "load_dataset", return_value=dataset),
                  patch.object(lb, "load_model_and_tokenizer", return_value=(FakeModel(), FakeTokenizer())),
                  patch.object(lb, "answer_conditional_ppl", side_effect=fake_score),
                  patch.object(lb, "tqdm", side_effect=lambda x: x),
                  redirect_stdout(io.StringIO())):
                lb.main()
            self.assertEqual(scored, ["A", "BBBB", "CCCCCCCC"])
            trace = json.loads(output.with_name("best_summary_examples.jsonl").read_text().strip())
            self.assertEqual(trace["reference_policy"], "best")
            self.assertEqual(trace["reference_count_scored"], 3)
            self.assertEqual(trace["selected_reference_index"], 3)
            self.assertEqual(trace["tokens_scored"], 8)
            self.assertEqual(trace["nll_sum"], 12.0)
            self.assertAlmostEqual(trace["avg_nll"], 1.5)
            self.assertAlmostEqual(trace["reference_macro_avg_nll"], 1.5)
            result = auditor.verify_gold(output)
            self.assertEqual(result["reference_policy"], "best")
            self.assertEqual(result["tokens"], 8)
            self.assertAlmostEqual(result["ppl"], math.exp(1.5))

    def test_v3_score_one_selects_best_mean_and_records_candidates(self):
        case = {
            "task": "narrativeqa", "example_index": 5, "example_id": "e",
            "length_group": "4k", "prompt": "Q", "answer": "A",
            "answers": ["A", "BBBB", "BBBB", "CCCCCCCC"],
        }
        scores = {"A": (5.0, 1), "BBBB": (8.0, 4), "CCCCCCCC": (12.0, 8)}
        with patch.object(v3, "answer_conditional_ppl", side_effect=lambda **kw: scores[kw["answer"]]):
            result = v3.score_one(None, FakeTokenizer(), case, 4096, 128)
        self.assertEqual(result["reference_policy"], "best")
        self.assertEqual(result["reference_index"], 3)
        self.assertEqual(result["reference_count_available"], 4)
        self.assertEqual(result["reference_count_scored"], 3)
        self.assertEqual(result["tokens_scored"], 8)
        self.assertEqual(result["nll_sum"], 12.0)
        self.assertEqual(sum(x["selected"] for x in result["references"]), 1)
        self.assertEqual(result["references"][-1]["reference_index"], 3)

    def test_single_reference_matches_v2_score_exactly(self):
        # On single-reference examples, best must be identical to the proven
        # original first-reference policy, not just statistically similar.
        case = {
            "task": "narrativeqa", "example_index": 0, "example_id": "e",
            "length_group": "4k", "prompt": "Question", "answer": "Paris",
            "answers": ["Paris"],
        }
        scorer = lambda **kw: (12.5, 5)
        with patch.object(v2, "answer_conditional_ppl", side_effect=scorer):
            first = v2.score_one(None, FakeTokenizer(), case, 4096, 128)
        with patch.object(v3, "answer_conditional_ppl", side_effect=scorer):
            best = v3.score_one(None, FakeTokenizer(), case, 4096, 128)
        for key in ("nll_sum", "tokens_scored", "avg_nll", "perplexity",
                    "prompt_tokens_original", "prompt_tokens_retained_first_chunk"):
            self.assertEqual(first[key], best[key], key)
        self.assertEqual(best["reference_count_scored"], 1)

    def test_v3_best_stable_tie_and_rejects_bad_score(self):
        case = {
            "task": "narrativeqa", "example_index": 0, "example_id": "e",
            "length_group": "4k", "prompt": "Q", "answer": "AA",
            "answers": ["AA", "BBB"],
        }
        with patch.object(v3, "answer_conditional_ppl",
                          side_effect=lambda **kw: (4.0, 2) if kw["answer"] == "AA" else (6.0, 3)):
            self.assertEqual(v3.score_one(None, FakeTokenizer(), case, 4096, 128)["reference_index"], 0)
        with patch.object(v3, "answer_conditional_ppl", return_value=(float("nan"), 2)):
            with self.assertRaisesRegex(RuntimeError, "Invalid reference"):
                v3.score_one(None, FakeTokenizer(), case, 4096, 128)

    def test_v3_frozen_models_and_samples_and_resume_safety(self):
        args = SimpleNamespace(
            order="recent", model_offset=0, limit_models=0,
            tasks=["narrativeqa"], examples_per_task=1, seed=42,
            sampling="random", context_mode="common", common_context=4096,
            answer_stride=128, dtype="bfloat16", source_manifest=None,
            min_trained_context=0,
        )
        m4 = dict(
            version=2, settings=v2.settings(args),
            models=[
                dict(model=f"adpretko/celerity-271m-{x}k-ad0p0",
                     revision=(str(x) * 40)[:40], trained_context_tokens=x * 1024)
                for x in (4, 8, 16)
            ],
            total_celerity_repositories=3, excluded_models=[], longbench_revision="frozen",
            sample_indices={"narrativeqa": [0]},
            sample_id_digest={"narrativeqa": "a" * 64},
        )
        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / "source.json"
            target = Path(td) / "best.json"
            v2.atomic_json(source, m4)
            args.source_manifest = source
            args.min_trained_context = 8192
            args.common_context = 8192
            with patch.object(v3, "discover", side_effect=AssertionError("must not discover")):
                frozen = v3.freeze_manifest(target, None, args)
            self.assertEqual(frozen["version"], 3)
            self.assertEqual(frozen["settings"]["reference_policy"], "best")
            self.assertEqual(frozen["settings"]["best_reference_selection"],
                             "minimum_mean_nll_per_token_v1")
            self.assertEqual(len(frozen["models"]), 2)
            self.assertEqual(frozen["sample_indices"], m4["sample_indices"])
            self.assertEqual(frozen["sample_id_digest"], m4["sample_id_digest"])
            self.assertEqual(frozen["longbench_revision"], "frozen")
            self.assertEqual(v3.freeze_manifest(target, None, args)["models"], frozen["models"])
            m4["sample_indices"]["narrativeqa"] = [1]
            v2.atomic_json(source, m4)
            with self.assertRaises(ValueError):
                v3.freeze_manifest(target, None, args)

    def test_v3_select_cases_uses_all_references_and_frozen_indices(self):
        row = {"_id": "ex", "context": "c", "input": "q",
               "answers": ["Paris", "Paris, France", "Paris"], "length": 1024}
        manifest = {
            "settings": {"tasks": ["narrativeqa"], "source_manifest_sha256": "abc"},
            "longbench_revision": "pin", "sample_indices": {"narrativeqa": [0]},
            "sample_id_digest": {"narrativeqa": v2.sha_json([("ex", "Paris")])},
        }
        with tempfile.TemporaryDirectory() as td:
            with patch.object(v3, "load_dataset", return_value=FakeDataset([row])):
                cases = v3.select_cases(manifest, Path(td) / "manifest.json")
            self.assertEqual(len(cases), 1)
            self.assertEqual(cases[0]["answers"], row["answers"])
            self.assertEqual(cases[0]["answer"], row["answers"][0])

    def test_v3_results_save_and_resume(self):
        args = SimpleNamespace(context_mode="common", common_context=4096,
                               answer_stride=128, device="cpu", dtype="float32", progress_every=10,
                               ephemeral_model_cache=True)
        model = dict(model="adpretko/celerity-271m-4k-ad0p0", revision="a" * 40,
                     trained_context_tokens=4096)
        case = dict(task="narrativeqa", example_index=0, example_id="e",
                    length_group="4k", prompt="Q", answer="A", answers=["A", "BBBB"])
        with tempfile.TemporaryDirectory() as td:
            args.model_cache_parent = Path(td) / "cache"
            out = Path(td) / "output"
            with (patch.object(v3, "load_model", return_value=(object(), FakeTokenizer(), "tokdigest")),
                  patch.object(v3, "answer_conditional_ppl", side_effect=lambda **kw: (4.0, 1) if kw["answer"] == "A" else (8.0, 4))):
                result = v3.run_one(model, [case], args, out)
            self.assertEqual(result["status"], "ok", result)
            self.assertEqual(result["reference_policy"], "best")
            self.assertEqual(result["scored_tokens"], 4)
            self.assertEqual(result["example_count"], 1)
            v3.save_summary(out, result)
            self.assertEqual(v3.completed_models(out, {"models": [model]}), {model["model"]})
            with (out / "summary.csv").open(newline="") as f:
                row = next(csv.DictReader(f))
            self.assertEqual(row["reference_policy"], "best")
            examples = [json.loads(line) for line in (out / "examples" / "celerity-271m-4k-ad0p0.jsonl").read_text().splitlines()]
            self.assertEqual(examples[0]["reference_index"], 1)
            task = json.loads((out / "tasks" / "celerity-271m-4k-ad0p0.jsonl").read_text().splitlines()[0])
            self.assertEqual(task["reference_policy"], "best")
            self.assertEqual(list(args.model_cache_parent.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
