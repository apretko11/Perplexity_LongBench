#!/usr/bin/env python3
"""Analyze paired context-length effects by prompt truncation category."""

import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parent.parent

INPUT = ROOT / "results/context_analysis/paired_examples.csv"

groups = defaultdict(list)

with INPUT.open(newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        key = (
            int(row["from_context"]),
            int(row["to_context"]),
            row["truncation_category"],
        )

        groups[key].append(row)


for shorter, longer in [
    (4096, 8192),
    (8192, 16384),
    (4096, 16384),
]:
    print(f"\n{'=' * 65}")
    print(f"CONTEXT: {shorter} -> {longer}")
    print(f"{'=' * 65}")

    for category in [
        "both_untruncated",
        "became_untruncated",
        "both_truncated",
    ]:
        rows = groups.get((shorter, longer, category), [])

        if not rows:
            continue

        deltas = [
            float(row["delta_avg_nll"])
            for row in rows
        ]

        tokens = [
            int(row["answer_tokens"])
            for row in rows
        ]

        # Token-weighted change in log perplexity.
        weighted_delta = (
            sum(d * t for d, t in zip(deltas, tokens))
            / sum(tokens)
        )

        # Equivalent relative change in pooled perplexity.
        pct_change = 100 * (math.exp(weighted_delta) - 1)

        improved = sum(d < 0 for d in deltas)
        worsened = sum(d > 0 for d in deltas)

        print(f"\n{category}")
        print(f"  Examples: {len(rows)}")
        print(f"  Improved: {improved}")
        print(f"  Worsened: {worsened}")
        print(f"  Median delta NLL/token: {median(deltas):+.5f}")
        print(f"  Token-weighted delta NLL: {weighted_delta:+.5f}")
        print(f"  Pooled perplexity change: {pct_change:+.2f}%")

        # Sanity check: if the complete prompt and answer fit
        # within both windows, changing the maximum context
        # should not materially change the score.
        if category == "both_untruncated":
            single_chunk = [
                abs(float(r["delta_avg_nll"]))
                for r in rows
                if int(r["answer_tokens"]) <= 128
            ]

            if single_chunk:
                print(
                    "  Max |delta NLL| for single-chunk answers:",
                    f"{max(single_chunk):.6f}",
                )

print("\nDone.")
