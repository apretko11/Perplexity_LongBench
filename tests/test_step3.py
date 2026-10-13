"""Step 3 offline tests: reconstruct and score precisely the Step 2 contexts.

No model downloads or GPU required. Tests exercise the real torch NLL shift
and independently inspect every teacher-forced input/label tensor.
"""

import copy
import math
import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from longbench_traced_scoring import TRACE_FORMAT, score_traced_generation
from longbench_utils import build_generation_context, token_ids_sha256


class ToyTokenizer:
    bos_token_id = 1
    eos_token_id = 999
    pad_token_id = 999


def toy_tokenize_prompt(*, tokenizer, prompt, use_chat_template):
    core = [10 + ord(ch) % 60 for ch in prompt]
    return [211] + core + [212] if use_chat_template else [1] + core


class UniformModel(torch.nn.Module):
    def __init__(self, vocab=1024):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.vocab = vocab

    def forward(self, input_ids):
        return SimpleNamespace(logits=torch.zeros(
            (*input_ids.shape, self.vocab), dtype=torch.float32,
            device=input_ids.device,
        ))


def real_nll_fn(*, model, input_ids, labels):
    logits = model(input_ids).logits
    n_tokens = int((labels[:, 1:] != -100).sum().item())
    nll = F.cross_entropy(
        logits[:, :-1, :].reshape(-1, logits.shape[-1]),
        labels[:, 1:].reshape(-1), ignore_index=-100, reduction="sum",
    )
    return float(nll.item()), n_tokens


def make_row(raw, *, prompt="abcdefghijklmnopqrstuvwxyz", context_limit=14,
             chunk_size=3, content=None, chat=False):
    prompt_ids = toy_tokenize_prompt(tokenizer=ToyTokenizer(), prompt=prompt,
                                     use_chat_template=chat)
    chunks = []
    start = 0
    while start < len(raw):
        end = min(start + chunk_size, len(raw))
        nnew = end - start
        context, plan = build_generation_context(
            prompt_ids, raw[:start], context_limit - nnew,
        )
        chunks.append({
            "chunk_index": len(chunks),
            "requested_new_tokens": nnew,
            "generated_start": start,
            "generated_end": end,
            **plan,
        })
        start = end
    content = list(raw) if content is None else list(content)
    return {
        "max_length": context_limit,
        "generated_token_ids": list(raw),
        "generated_content_token_ids": content,
        "generated_tokens": len(content),
        "generated_tokens_raw": len(raw),
        "prediction": "A decoded string we must never re-tokenize",
        "generation_trace": {
            "format": TRACE_FORMAT,
            "prompt_token_ids_sha256": token_ids_sha256(prompt_ids),
            "chunks": chunks,
        },
    }


def execute(row, *, prompt="abcdefghijklmnopqrstuvwxyz", chat=False,
            context_limit=14, nll_fn=real_nll_fn):
    return score_traced_generation(
        model=UniformModel(), tokenizer=ToyTokenizer(), prompt=prompt,
        row=row, max_length=context_limit, use_chat_template=chat,
        nll_fn=nll_fn, tokenize_prompt_fn=toy_tokenize_prompt,
    )


class StepThreeTests(unittest.TestCase):
    def test_all_content_tokens_scored_exactly_once_across_chunks(self):
        raw = [300, 301, 302, 303, 304, 305, 306]
        row = make_row(raw)
        seen = []
        def spy_nll(*, model, input_ids, labels):
            ids, labs = input_ids.tolist()[0], labels.tolist()[0]
            self.assertEqual(len(ids), len(labs))
            self.assertGreater(len(ids), 1)
            self.assertEqual(labs[:len(ids) - sum(x != -100 for x in labs)],
                             [-100] * (len(ids) - sum(x != -100 for x in labs)))
            scored = [x for x in labs if x != -100]
            seen.extend(scored)
            return real_nll_fn(model=model, input_ids=input_ids, labels=labels)
        nll, tokens, chunks = execute(row, nll_fn=spy_nll)
        self.assertEqual(seen, raw)
        self.assertEqual((tokens, chunks), (7, 3))
        self.assertAlmostEqual(nll, 7 * math.log(1024), places=3)

    def test_context_inputs_are_exact_step2_head_tail_and_history(self):
        raw = [300, 301, 302, 303, 304, 305, 306]
        row = make_row(raw)
        calls = []
        def spy_nll(*, model, input_ids, labels):
            calls.append((input_ids.tolist()[0], labels.tolist()[0]))
            return real_nll_fn(model=model, input_ids=input_ids, labels=labels)
        execute(row, nll_fn=spy_nll)
        self.assertEqual(len(calls), 3)
        prompt_ids = toy_tokenize_prompt(tokenizer=ToyTokenizer(),
                                         prompt="abcdefghijklmnopqrstuvwxyz",
                                         use_chat_template=False)
        for i, (ids, labels) in enumerate(calls):
            span = row["generation_trace"]["chunks"][i]
            start, end = span["generated_start"], span["generated_end"]
            expected_input, _ = build_generation_context(
                prompt_ids, raw[:start], 14 - (end - start),
            )
            self.assertEqual(ids, expected_input + raw[start:end])
            self.assertEqual(labels, [-100] * len(expected_input) + raw[start:end])

    def test_eos_excluded_but_raw_trace_covered(self):
        raw = [300, 301, 302, 999]
        row = make_row(raw, content=raw[:-1])
        nll, tokens, chunks = execute(row)
        self.assertEqual((tokens, chunks), (3, 2))
        self.assertAlmostEqual(nll, 3 * math.log(1024), places=3)

    def test_immediate_eos_scores_zero_but_verifies_context(self):
        row = make_row([999], content=[])
        def should_not_run(**kwargs):
            raise AssertionError("NLL should not run when no content tokens exist")
        self.assertEqual(execute(row, nll_fn=should_not_run), (0.0, 0, 1))

    def test_unexpected_removed_suffix_is_rejected(self):
        row = make_row([300, 301], content=[300])
        with self.assertRaisesRegex(ValueError, "EOS/pad"):
            execute(row)

    def test_prompt_checksum_detects_wrong_dataset_or_chat_template(self):
        row = make_row([300, 301])
        with self.assertRaisesRegex(ValueError, "Prompt token checksum"):
            execute(row, prompt="abCdefghijklmnopqrstuvwxyz")
        with self.assertRaisesRegex(ValueError, "Prompt token checksum"):
            execute(row, chat=True)

    def test_chat_template_replays_correctly(self):
        row = make_row([300, 301, 302], chat=True)
        self.assertEqual(execute(row, chat=True)[1:], (3, 1))

    def test_tampered_chunk_checksum_is_detected(self):
        row = make_row([300, 301, 302, 303])
        row["generation_trace"]["chunks"][1]["input_context_sha256"] = "corrupt"
        with self.assertRaisesRegex(ValueError, "checksum"):
            execute(row)

    def test_overlap_missing_chunk_or_bad_history_is_rejected(self):
        row = make_row([300, 301, 302, 303, 304])
        row["generation_trace"]["chunks"][1]["generated_start"] = 2
        with self.assertRaisesRegex(ValueError, "spans"):
            execute(row)

        row = make_row([300, 301, 302, 303, 304])
        row["generation_trace"]["chunks"].pop()
        with self.assertRaisesRegex(ValueError, "cover"):
            execute(row)

        row = make_row([300, 301, 302, 303, 304])
        row["generation_trace"]["chunks"][1]["generated_history_end"] = 2
        with self.assertRaisesRegex(ValueError, "history"):
            execute(row)

    def test_context_limit_cannot_change(self):
        row = make_row([300, 301, 302])
        with self.assertRaisesRegex(ValueError, "max_length"):
            execute(row, context_limit=13)

    def test_content_ids_cannot_change_or_be_retokenized(self):
        row = make_row([300, 301, 302])
        row["generated_content_token_ids"] = [300, 303, 302]
        with self.assertRaisesRegex(ValueError, "unchanged prefix"):
            execute(row)

    def test_all_generated_history_occupies_window_after_long_answer(self):
        row = make_row(list(range(300, 325)), chunk_size=3)
        # As generated history grows, the original prompt is displaced.
        last = row["generation_trace"]["chunks"][-1]
        self.assertEqual((last["prompt_head_tokens"], last["prompt_tail_tokens"]), (0, 0))
        nll, tokens, chunks = execute(row)
        self.assertEqual(tokens, 25)
        self.assertEqual(chunks, 9)
        self.assertAlmostEqual(nll, tokens * math.log(1024), places=3)

    def test_empty_generation_trace_can_correspond_to_zero_output(self):
        row = make_row([])
        self.assertEqual(execute(row), (0.0, 0, 0))

    def test_new_format_is_required(self):
        row = make_row([300])
        row["generation_trace"]["format"] = "unknown"
        with self.assertRaisesRegex(ValueError, "unsupported"):
            execute(row)


if __name__ == "__main__":
    unittest.main()
