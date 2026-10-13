"""Step 2 offline regressions for middle-truncated LongBench generation.

Uses fake model/tokenizer; no model downloads or GPU needed.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

import generate_longbench as gen
import generated_perplexity as scorer
from longbench_utils import (
    build_generation_context,
    reconstruct_generation_context,
    token_ids_sha256,
)


class FakeTokenizer:
    bos_token_id = 1
    eos_token_id = 999
    pad_token_id = 999

    def __call__(self, text, add_special_tokens=False, **kwargs):
        return {"input_ids": [10 + (ord(c) % 70) for c in text]}

    def apply_chat_template(self, msgs, tokenize, add_generation_prompt):
        assert tokenize and add_generation_prompt
        return [211] + self(msgs[0]["content"])["input_ids"] + [212]

    def decode(self, tokens, skip_special_tokens=True):
        return ",".join(map(str, tokens))


class Recorder(torch.nn.Module):
    """Generate deterministic IDs, recording exact tensor inputs."""

    def __init__(self, stop_on_call=None):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.calls = []
        self.stop_on_call = stop_on_call

    def generate(self, *, input_ids, attention_mask, max_new_tokens,
                 do_sample, num_beams, pad_token_id, eos_token_id):
        self.calls.append({
            "input_ids": input_ids[0].tolist(),
            "attention_mask": attention_mask[0].tolist(),
            "max_new_tokens": max_new_tokens,
        })
        assert not do_sample and num_beams == 1
        if self.stop_on_call == len(self.calls):
            continuation = [999]
        else:
            continuation = [100 + len(self.calls)] * max_new_tokens
        return torch.tensor([input_ids[0].tolist() + continuation], dtype=torch.long)


class StepTwoTests(unittest.TestCase):
    def test_long_prompt_head_tail_exact_budget(self):
        prompt = list(range(20))
        context, plan = build_generation_context(prompt, [], 8)
        self.assertEqual(context, [0, 1, 2, 3, 16, 17, 18, 19])
        self.assertEqual((plan["prompt_head_tokens"], plan["prompt_tail_tokens"]), (4, 4))
        self.assertTrue(plan["prompt_middle_truncated"])
        self.assertEqual(reconstruct_generation_context(prompt, [], plan), context)
        self.assertEqual(plan["input_context_sha256"], token_ids_sha256(context))

    def test_generated_answer_history_preserved_and_trace_checked(self):
        prompt = list(range(20))
        generated = [90, 91, 92]
        context, plan = build_generation_context(prompt, generated, 8)
        self.assertEqual(context, [0, 1, 2, 18, 19, 90, 91, 92])
        self.assertEqual(plan["generated_history_start"], 0)
        self.assertEqual(reconstruct_generation_context(prompt, generated, plan), context)
        with self.assertRaisesRegex(ValueError, "checksum"):
            reconstruct_generation_context(prompt[:-1] + [77], generated, plan)

    def test_fully_generated_window_never_discards_newest_history(self):
        prompt = list(range(20))
        generated = list(range(30, 50))
        context, plan = build_generation_context(prompt, generated, 8)
        self.assertEqual(context, list(range(42, 50)))
        self.assertEqual((plan["prompt_head_tokens"], plan["prompt_tail_tokens"]), (0, 0))
        self.assertEqual(reconstruct_generation_context(prompt, generated, plan), context)

    def test_short_prompt_and_bad_budgets(self):
        context, plan = build_generation_context([2, 3, 4], [80, 81], 8)
        self.assertEqual(context, [2, 3, 4, 80, 81])
        self.assertEqual((plan["prompt_head_tokens"], plan["prompt_tail_tokens"]), (3, 0))
        self.assertFalse(plan["prompt_middle_truncated"])
        with self.assertRaises(ValueError):
            build_generation_context([1], [], 0)
        with self.assertRaises(ValueError):
            build_generation_context([], [], 2)

    def test_generate_answer_chunks_match_exact_recorded_contexts(self):
        model, tokenizer = Recorder(), FakeTokenizer()
        trace = {}
        prompt = "abcdefghijklmnopqrstuvwxyz"  # longer than tiny context
        prediction, raw, content, prompt_len = gen.generate_answer(
            model=model, tokenizer=tokenizer, prompt=prompt,
            max_length=12, max_new_tokens=5, generation_chunk_size=2,
            use_chat_template=False, generation_trace=trace,
        )
        self.assertEqual(raw, [101, 101, 102, 102, 103])
        self.assertEqual(content, raw)
        self.assertEqual(prediction, tokenizer.decode(raw))
        self.assertEqual(prompt_len, 27)  # 26 letters + BOS
        self.assertEqual(len(trace["chunks"]), 3)
        self.assertEqual(trace["format"], "head_tail_prompt_recent_generation_v1")
        prompt_ids = gen.tokenize_prompt(tokenizer, prompt, use_chat_template=False)
        self.assertEqual(trace["prompt_token_ids_sha256"], token_ids_sha256(prompt_ids))
        for i, (call, rec) in enumerate(zip(model.calls, trace["chunks"])):
            reconstructed = reconstruct_generation_context(prompt_ids, raw, rec)
            self.assertEqual(reconstructed, call["input_ids"])
            self.assertEqual(len(call["attention_mask"]), len(call["input_ids"]))
            self.assertEqual(rec["generated_history_end"], rec["generated_start"])
            self.assertLessEqual(len(reconstructed) + call["max_new_tokens"], 12)
            self.assertEqual(rec["generated_end"] - rec["generated_start"], call["max_new_tokens"])
            if i == 0:
                self.assertEqual(reconstructed[:5], prompt_ids[:5])
                self.assertEqual(reconstructed[-5:], prompt_ids[-5:])
            else:
                self.assertEqual(reconstructed[-rec["generated_start"]:], raw[:rec["generated_start"]])

    def test_early_eos_keeps_raw_ids_for_trace_and_removes_from_content(self):
        model, tokenizer = Recorder(stop_on_call=2), FakeTokenizer()
        trace = {}
        _, raw, content, _ = gen.generate_answer(
            model=model, tokenizer=tokenizer, prompt="abc" * 8,
            max_length=12, max_new_tokens=10, generation_chunk_size=3,
            use_chat_template=True, generation_trace=trace,
        )
        self.assertEqual(raw, [101, 101, 101, 999])
        self.assertEqual(content, [101, 101, 101])
        self.assertEqual(len(trace["chunks"]), 2)
        self.assertEqual(trace["chunks"][-1]["generated_end"], 4)
        prompt_ids = gen.tokenize_prompt(tokenizer, "abc" * 8, use_chat_template=True)
        self.assertEqual(trace["prompt_token_ids_sha256"], token_ids_sha256(prompt_ids))
        self.assertEqual(trace["chunks"][0]["prompt_tail_tokens"] > 0, True)
        for call, plan in zip(model.calls, trace["chunks"]):
            self.assertEqual(call["input_ids"], reconstruct_generation_context(prompt_ids, raw, plan))

    def test_backward_compatible_four_value_return(self):
        model, tokenizer = Recorder(), FakeTokenizer()
        result = gen.generate_answer(
            model=model, tokenizer=tokenizer, prompt="example",
            max_length=16, max_new_tokens=2, generation_chunk_size=2,
            use_chat_template=False,
        )
        self.assertEqual(len(result), 4)

    def test_traced_rows_route_to_verified_scorer(self):
        # Step 3 must accept a valid trace format and dispatch it to the
        # verified scorer, never the legacy tail-only scorer. Detailed trace
        # integrity is tested independently in test_step3.py.
        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / "new_generation.jsonl"
            output = Path(td) / "score.jsonl"
            source.write_text(json.dumps({
                "model": "fake",
                "task": "narrativeqa",
                "example_index": 0,
                "max_length": 16,
                "generation_chunk_size": 4,
                "use_chat_template": False,
                "generated_token_ids": [101],
                "generated_content_token_ids": [101],
                "generated_tokens": 1,
                "generated_tokens_raw": 1,
                "generation_trace": {
                    "format": "head_tail_prompt_recent_generation_v1",
                    "chunks": [],  # Mocked scorer validates dispatch, not trace internals.
                },
            }) + "\n", encoding="utf-8")
            argv = [
                "generated_perplexity.py", "--model", "fake",
                "--input-jsonl", str(source),
                "--output-jsonl", str(output),
            ]
            fake_dataset = [{"context": "A document", "input": "A question"}]
            with (patch.object(sys, "argv", argv),
                  patch.object(scorer, "load_model_and_tokenizer", return_value=(object(), object())),
                  patch.object(scorer, "load_dataset", return_value=fake_dataset),
                  patch.object(scorer, "score_traced_generation", return_value=(2.0, 1, 1)) as verified,
                  patch.object(scorer, "generated_answer_conditional_ppl",
                               side_effect=AssertionError("Legacy scoring must not run")),
                  patch.object(scorer, "tqdm", side_effect=lambda values, **kw: values)):
                scorer.main()
            verified.assert_called_once()
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(rows[0]["scoring_context_policy"],
                             "head_tail_prompt_recent_generation_v1")
            self.assertEqual(rows[0]["verified_generation_chunks"], 1)
            self.assertEqual(rows[0]["tokens_scored"], 1)
            self.assertEqual(rows[-1]["tokens_scored"], 1)


if __name__ == "__main__":
    unittest.main()
