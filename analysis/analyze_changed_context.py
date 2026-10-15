#!/usr/bin/env python3
"""Context-length analysis restricted to examples receiving more prompt tokens."""

import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "results/context_analysis/paired_examples.csv"

COMPARISONS = [
    (4096, 8192),
    (8192, 16384),
    (4096, 16384),
]

with INPUT.open(newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))


def summarize(shorter, longer, single_chunk=False):
    groups = defaultdict(list)

    for row in rows:
        if (
            int(row["from_context"]) != shorter
            or int(row["to_context"]) != longer
        ):
            continue

        # Include only examples where the longer context
        # actually retains more original prompt tokens.
        if (
            int(row["prompt_retained_to"])
            <= int(row["prompt_retained_from"])
        ):
            continue

        # Optional stricter subset: the entire reference answer
        # is scored in one chunk (stride = 128).
        if single_chunk and int(row["answer_tokens"]) > 128:
            continue

        groups[row["model"]].append(row)

    model_changes = []
    example_count = 0

    for model, examples in groups.items():
        tokens = sum(int(r["answer_tokens"]) for r in examples)

        # Recover the change in total NLL from the
        # per-token NLL differences.
        delta_nll = sum(
            float(r["delta_avg_nll"]) * int(r["answer_tokens"])
            for r in examples
        )

        # Ratio of pooled perplexities for matched examples.
        pct_change = 100 * (math.exp(delta_nll / tokens) - 1)

        model_changes.append(pct_change)
        example_count += len(examples)

    improved = sum(x < 0 for x in model_changes)
    worsened = sum(x > 0 for x in model_changes)

    label = "single-chunk only" if single_chunk else "all changed-input examples"

    print(f"\n{shorter} -> {longer} ({label})")
    print(f"  Models: {len(model_changes)}")
    print(f"  Examples: {example_count}")
    print(f"  Improved models: {improved}")
    print(f"  Worsened models: {worsened}")
    print(f"  Median PPL change: {median(model_changes):+.2f}%")


for shorter, longer in COMPARISONS:
    summarize(shorter, longer)
    summarize(shorter, longer, single_chunk=True)
