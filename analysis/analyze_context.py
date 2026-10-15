#!/usr/bin/env python3
"""Stage 1: Matched-model, matched-example LongBench context analysis."""

import csv
import json
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS = ROOT / "experiments"
OUTPUT = ROOT / "results" / "context_analysis"

CONTEXTS = {
    4096: "4k",
    8192: "8k",
    16384: "16k",
}

COMPARISONS = [
    (4096, 8192),
    (8192, 16384),
    (4096, 16384),
]


def save_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_experiments():
    manifests = {}
    summaries = {}
    examples = {}

    for context, label in CONTEXTS.items():
        folder = EXPERIMENTS / f"celerity_all_{label}_bf16"

        manifests[context] = json.loads(
            (folder / "manifest.json").read_text()
        )

        with (folder / "summary.csv").open() as f:
            rows = list(csv.DictReader(f))

        expected = {
            m["model"]: m
            for m in manifests[context]["models"]
        }

        assert len(rows) == len(expected)

        for row in rows:
            model = row["model"]

            assert row["status"] == "ok", model
            assert row["model_revision"] == expected[model]["revision"]
            assert int(row["evaluation_context_tokens"]) == context

            summaries[model, context] = row

            name = model.split("/", 1)[1]
            path = folder / "examples" / f"{name}.jsonl"

            with path.open() as f:
                model_examples = [
                    json.loads(line)
                    for line in f
                    if line.strip()
                ]

            assert len(model_examples) == 80, model

            for ex in model_examples:
                key = (
                    model,
                    context,
                    ex["task"],
                    ex["example_index"],
                )

                assert key not in examples
                assert ex["model_revision"] == row["model_revision"]

                examples[key] = ex

    # Confirm all experiments use the same frozen LongBench examples.
    baseline = manifests[4096]

    for context in (8192, 16384):
        current = manifests[context]

        for field in (
            "longbench_revision",
            "sample_indices",
            "sample_id_digest",
        ):
            assert current[field] == baseline[field], field

    assert len(examples) == 30240

    return summaries, examples


def analyze(summaries, examples):
    OUTPUT.mkdir(parents=True, exist_ok=True)

    model_results = []
    example_results = []

    for shorter, longer in COMPARISONS:

        models = sorted(
            {m for m, c in summaries if c == shorter}
            &
            {m for m, c in summaries if c == longer}
        )

        changes = []

        for model in models:
            before = summaries[model, shorter]
            after = summaries[model, longer]

            assert before["model_revision"] == after["model_revision"]

            ppl_before = float(before["perplexity"])
            ppl_after = float(after["perplexity"])

            pct_change = 100 * (ppl_after / ppl_before - 1)

            changes.append(pct_change)

            model_results.append({
                "model": model,
                "trained_context": before["trained_context_tokens"],
                "from_context": shorter,
                "to_context": longer,
                "ppl_from": ppl_before,
                "ppl_to": ppl_after,
                "ppl_change_pct": pct_change,
                "avg_nll_change": (
                    float(after["avg_nll"])
                    - float(before["avg_nll"])
                ),
            })

            # Match individual LongBench examples.
            first = {
                (task, idx): ex
                for (m, c, task, idx), ex in examples.items()
                if m == model and c == shorter
            }

            second = {
                (task, idx): ex
                for (m, c, task, idx), ex in examples.items()
                if m == model and c == longer
            }

            assert first.keys() == second.keys()
            assert len(first) == 80

            for (task, idx), a in first.items():
                b = second[task, idx]

                assert a["example_id"] == b["example_id"]
                assert a["answer_tokens"] == b["answer_tokens"]
                assert (
                    a["prompt_tokens_original"]
                    == b["prompt_tokens_original"]
                )

                truncated_before = a["prompt_was_truncated_first_chunk"]
                truncated_after = b["prompt_was_truncated_first_chunk"]

                if truncated_before and truncated_after:
                    category = "both_truncated"
                elif truncated_before:
                    category = "became_untruncated"
                else:
                    assert not truncated_after
                    category = "both_untruncated"

                example_results.append({
                    "model": model,
                    "trained_context": before["trained_context_tokens"],
                    "from_context": shorter,
                    "to_context": longer,
                    "task": task,
                    "example_index": idx,
                    "example_id": a["example_id"],
                    "answer_tokens": a["answer_tokens"],
                    "avg_nll_from": a["avg_nll"],
                    "avg_nll_to": b["avg_nll"],
                    "delta_avg_nll": (
                        float(b["avg_nll"]) - float(a["avg_nll"])
                    ),
                    "prompt_original_tokens": a["prompt_tokens_original"],
                    "prompt_retained_from": a["prompt_tokens_retained_first_chunk"],
                    "prompt_retained_to": b["prompt_tokens_retained_first_chunk"],
                    "truncation_category": category,
                })

        improved = sum(x < 0 for x in changes)
        worsened = sum(x > 0 for x in changes)

        print(f"\n{shorter} -> {longer} tokens")
        print(f"  Matched models: {len(models)}")
        print(f"  Improved: {improved}")
        print(f"  Worsened: {worsened}")
        print(f"  Median PPL change: {median(changes):+.2f}%")

    save_csv(OUTPUT / "paired_models.csv", model_results)
    save_csv(OUTPUT / "paired_examples.csv", example_results)

    print("\nAnalysis complete.")
    print(f"Model comparisons: {len(model_results)}")
    print(f"Example comparisons: {len(example_results)}")
    print(f"Results saved to: {OUTPUT}")


if __name__ == "__main__":
    summaries, examples = load_experiments()
    analyze(summaries, examples)
