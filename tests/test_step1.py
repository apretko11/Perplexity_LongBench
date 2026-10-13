"""Offline regression tests for Step 1 gold-answer LongBench evaluation."""

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

# In the real repository the current module is imported directly. The tests
# intentionally patch model/dataset loading and never download any HF weights.
import longbench_perplexity as lb
from longbench_utils import middle_truncate_tokens


class FakeDataset:
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __iter__(self):
        return iter(self.examples)

    def select(self, indices):
        return FakeDataset([self.examples[i] for i in indices])


class FakeTokenizer:
    bos_token_id = 1

    def __call__(self, text, add_special_tokens=False, **kwargs):
        return {"input_ids": [2 + ord(c) % 20 for c in text]}


class UniformModel(torch.nn.Module):
    def __init__(self, vocab=32):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.vocab = vocab
        self.config = SimpleNamespace(max_position_embeddings=32)

    def forward(self, input_ids):
        return SimpleNamespace(logits=torch.zeros(
            (*input_ids.shape, self.vocab), device=input_ids.device,
        ))


class StepOneTests(unittest.TestCase):
    def test_middle_truncation_boundary_budgets(self):
        source = list(range(12))
        self.assertEqual(middle_truncate_tokens(source, 0), [])
        self.assertEqual(middle_truncate_tokens(source, 1), [0])
        self.assertEqual(middle_truncate_tokens(source, 4), [0, 1, 10, 11])
        self.assertEqual(middle_truncate_tokens(source, 5), [0, 1, 2, 10, 11])
        self.assertEqual(middle_truncate_tokens(source, 12), source)
        self.assertEqual(middle_truncate_tokens(source, 20), source)
        with self.assertRaises(ValueError):
            middle_truncate_tokens(source, -1)

    def test_causal_shift_and_prompt_mask(self):
        # Three prompt tokens precede two answer tokens. Only the two
        # answer tokens are scored; uniform logits => log(vocab) per token.
        model = UniformModel(vocab=32)
        ids = torch.tensor([[1, 2, 3, 4, 5]])
        labels = torch.tensor([[-100, -100, -100, 4, 5]])
        nll, tokens = lb.nll_sum_for_labels(model, ids, labels)
        self.assertEqual(tokens, 2)
        self.assertAlmostEqual(nll, 2 * math.log(32), places=5)

    def test_long_answer_scores_every_token_once(self):
        model = UniformModel(vocab=32)
        nll, tokens = lb.answer_conditional_ppl(
            model=model,
            tokenizer=FakeTokenizer(),
            prompt="a" * 45,
            answer="abcde" * 5,
            max_length=12,
            stride=4,
            use_chat_template=False,
        )
        self.assertEqual(tokens, 25)
        self.assertAlmostEqual(nll, tokens * math.log(32), places=4)

    def test_long_answer_large_stride_preserves_first_target(self):
        # The scorer must retain a predecessor for the first target in each
        # window, even when each chunk is almost max_length tokens long.
        model = UniformModel(vocab=32)
        nll, tokens = lb.answer_conditional_ppl(
            model=model,
            tokenizer=FakeTokenizer(),
            prompt="a" * 40,
            answer="abcdefg" * 6,
            max_length=12,
            stride=11,
            use_chat_template=False,
        )
        self.assertEqual(tokens, 42)
        self.assertAlmostEqual(nll, tokens * math.log(32), places=4)

    def test_full_mode_keeps_task_summary_and_example_trace(self):
        dataset = FakeDataset([{
            "_id": "full-1", "context": "Doc text", "input": "Question",
        }])
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "full.jsonl"
            args = [
                "longbench_perplexity.py", "--model", "tiny",
                "--mode", "full", "--tasks", "narrativeqa",
                "--max-length", "16", "--output-jsonl", str(out),
            ]
            with (patch.object(sys, "argv", args),
                  patch.object(lb, "load_dataset", return_value=dataset),
                  patch.object(lb, "load_model_and_tokenizer", return_value=(UniformModel(), FakeTokenizer())),
                  patch.object(lb, "full_text_ppl", return_value=(4.0, 2)),
                  patch.object(lb, "tqdm", side_effect=lambda x: x)):
                lb.main()
            summary = [json.loads(x) for x in out.read_text().splitlines()]
            trace = [json.loads(x) for x in out.with_name("full_examples.jsonl").read_text().splitlines()]
            self.assertEqual(len(summary), 2)
            self.assertEqual(len(trace), 1)
            self.assertEqual(trace[0]["mode"], "full")
            self.assertAlmostEqual(summary[-1]["perplexity"], math.exp(2.0))

    def test_reference_policies_and_example_trace(self):
        # "Paris" versus a longer accepted answer: neither requires
        # generating any text. Reference policy controls teacher forcing.
        dataset = FakeDataset([{
            "_id": "toy-1",
            "context": "a long document",
            "input": "What is the capital of France?",
            "answers": ["Paris", "Paris, France", "Paris"],
        }])
        scores = {"Paris": (1.0, 1), "Paris, France": (8.0, 4)}
        call_args = []

        def fake_score(**kwargs):
            call_args.append(kwargs["answer"])
            return scores[kwargs["answer"]]

        with tempfile.TemporaryDirectory() as td:
            for policy in ("first", "all"):
                call_args.clear()
                out = Path(td) / f"{policy}.jsonl"
                args = [
                    "longbench_perplexity.py", "--model", "tiny",
                    "--mode", "answer", "--tasks", "narrativeqa",
                    "--max-length", "16", "--stride", "8",
                    "--reference-policy", policy,
                    "--output-jsonl", str(out),
                ]
                with (patch.object(sys, "argv", args),
                      patch.object(lb, "load_dataset", return_value=dataset),
                      patch.object(lb, "load_model_and_tokenizer", return_value=(UniformModel(), FakeTokenizer())),
                      patch.object(lb, "answer_conditional_ppl", side_effect=fake_score),
                      patch.object(lb, "tqdm", side_effect=lambda x: x)):
                    lb.main()

                summary = [json.loads(x) for x in out.read_text().splitlines()]
                trace_path = out.with_name(out.stem + "_examples.jsonl")
                trace = [json.loads(x) for x in trace_path.read_text().splitlines()]
                self.assertEqual(len(summary), 2)  # task + OVERALL
                self.assertEqual(len(trace), 1)
                self.assertEqual(trace[0]["example_id"], "toy-1")
                self.assertEqual(trace[0]["reference_policy"], policy)
                self.assertEqual(summary[-1]["reference_policy"], policy)
                if policy == "first":
                    self.assertEqual(call_args, ["Paris"])
                    self.assertEqual(trace[0]["tokens_scored"], 1)
                    self.assertEqual(trace[0]["reference_count_scored"], 1)
                    self.assertAlmostEqual(summary[-1]["avg_nll"], 1.0)
                else:
                    self.assertEqual(call_args, ["Paris", "Paris, France"])
                    self.assertEqual(trace[0]["tokens_scored"], 5)
                    self.assertEqual(trace[0]["reference_count_scored"], 2)
                    # Micro tokens: (1+8)/(1+4)=1.8.
                    self.assertAlmostEqual(summary[-1]["avg_nll"], 1.8)
                    # Equal references within an example: (1/1+8/4)/2=1.5.
                    self.assertAlmostEqual(summary[-1]["example_macro_avg_nll"], 1.5)


if __name__ == "__main__":
    unittest.main()
