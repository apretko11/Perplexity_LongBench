#!/usr/bin/env python3
"""Audit examples that must have identical scoring inputs."""

import csv
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "results/context_analysis/paired_examples.csv"

with INPUT.open(newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

comparisons = [
    (4096, 8192),
    (8192, 16384),
    (4096, 16384),
]

for shorter, longer in comparisons:
    matched = [
        r for r in rows
        if int(r["from_context"]) == shorter
        and int(r["to_context"]) == longer
    ]

    # The entire prompt and reference answer fit within
    # the smaller context. All scoring windows should therefore
    # contain identical token sequences in both evaluations.
    controls = [
        r for r in matched
        if (
            int(r["prompt_original_tokens"])
            + int(r["answer_tokens"])
            <= shorter
        )
    ]

    assert all(
        int(r["prompt_retained_from"])
        == int(r["prompt_original_tokens"])
        for r in controls
    )

    def delta(row):
        return float(row["delta_avg_nll"])

    total_tokens = sum(int(r["answer_tokens"]) for r in controls)

    weighted_delta = (
        sum(delta(r) * int(r["answer_tokens"]) for r in controls)
        / total_tokens
    )

    outliers = [
        r for r in controls
        if abs(delta(r)) > 0.01
    ]

    print(f"\n{'=' * 60}")
    print(f"{shorter} -> {longer}")
    print(f"Identical-input controls: {len(controls)}")
    print(f"Cases with |delta NLL/token| > 0.01: {len(outliers)}")
    print(f"Maximum absolute delta: {max(abs(delta(r)) for r in controls):.6f}")
    print(f"Token-weighted delta: {weighted_delta:+.6f}")
    print(f"Equivalent PPL change: {100 * (math.exp(weighted_delta) - 1):+.3f}%")

    print("\nModels with the most outliers:")
    for model, count in Counter(r["model"] for r in outliers).most_common(5):
        print(f"  {count:3d}  {model}")

    print("\nLargest individual discrepancies:")
    for r in sorted(controls, key=lambda r: abs(delta(r)), reverse=True)[:5]:
        print(
            f"  {delta(r):+.6f}  "
            f"{r['model']}  "
            f"{r['task']}#{r['example_index']}"
        )
