#!/usr/bin/env python3
"""Analyze paired LongBench perplexity changes by task."""

import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parent.parent

INPUT = ROOT / "results/context_analysis/paired_examples.csv"
OUTPUT = ROOT / "results/context_analysis"

# Group the five matched examples belonging to each
# model, context comparison, and LongBench task.
groups = defaultdict(list)

with INPUT.open(newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        key = (
            int(row["from_context"]),
            int(row["to_context"]),
            row["model"],
            row["task"],
        )
        groups[key].append(row)


model_task_results = []
task_groups = defaultdict(list)

for (shorter, longer, model, task), examples in groups.items():

    assert len(examples) == 5, (model, task)

    tokens = sum(
        int(r["answer_tokens"])
        for r in examples
    )

    # Token-weighted average NLL within this task.
    nll_before = sum(
        float(r["avg_nll_from"]) * int(r["answer_tokens"])
        for r in examples
    )

    nll_after = sum(
        float(r["avg_nll_to"]) * int(r["answer_tokens"])
        for r in examples
    )

    avg_nll_before = nll_before / tokens
    avg_nll_after = nll_after / tokens

    delta_nll = avg_nll_after - avg_nll_before

    # Equivalent percentage change in task-level PPL.
    pct_change = 100 * math.expm1(delta_nll)

    # Number of examples retaining additional prompt tokens.
    changed_inputs = sum(
        int(r["prompt_retained_to"])
        > int(r["prompt_retained_from"])
        for r in examples
    )

    result = {
        "model": model,
        "trained_context": int(examples[0]["trained_context"]),
        "from_context": shorter,
        "to_context": longer,
        "task": task,
        "examples": len(examples),
        "answer_tokens": tokens,
        "avg_nll_from": avg_nll_before,
        "avg_nll_to": avg_nll_after,
        "delta_avg_nll": delta_nll,
        "ppl_change_pct": pct_change,
        "examples_with_more_context": changed_inputs,
    }

    model_task_results.append(result)

    task_groups[shorter, longer, task].append(result)


# Aggregate the model-level task changes.
task_summary = []

for (shorter, longer, task), rows in sorted(task_groups.items()):

    changes = [r["ppl_change_pct"] for r in rows]
    nll_changes = [r["delta_avg_nll"] for r in rows]

    improved = sum(x < 0 for x in changes)
    worsened = sum(x > 0 for x in changes)

    task_summary.append({
        "from_context": shorter,
        "to_context": longer,
        "task": task,
        "models": len(rows),
        "models_improved": improved,
        "models_worsened": worsened,
        "median_ppl_change_pct": median(changes),
        "mean_ppl_change_pct": sum(changes) / len(changes),
        "median_delta_nll": median(nll_changes),
        "examples_with_more_context_pct": (
            100 * sum(r["examples_with_more_context"] for r in rows)
            / (5 * len(rows))
        ),
    })


def save_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


OUTPUT.mkdir(parents=True, exist_ok=True)

save_csv(
    OUTPUT / "paired_model_tasks.csv",
    model_task_results,
)

save_csv(
    OUTPUT / "task_summary.csv",
    task_summary,
)


# Print task rankings for each context comparison.
for shorter, longer in [
    (4096, 8192),
    (8192, 16384),
    (4096, 16384),
]:

    selected = [
        r for r in task_summary
        if r["from_context"] == shorter
        and r["to_context"] == longer
    ]

    selected.sort(key=lambda r: r["median_ppl_change_pct"])

    print(f"\n{'=' * 75}")
    print(f"CONTEXT: {shorter} -> {longer}")
    print(f"{'=' * 75}")

    print(
        f"{'Task':25s} "
        f"{'Improved':>10s} "
        f"{'Median PPL Δ':>14s} "
        f"{'More Context':>14s}"
    )

    for r in selected:
        improved = f"{r['models_improved']}/{r['models']}"

        print(
            f"{r['task']:25s} "
            f"{improved:>10s} "
            f"{r['median_ppl_change_pct']:>+13.2f}% "
            f"{r['examples_with_more_context_pct']:>13.1f}%"
        )


print("\nSaved:")
print("  results/context_analysis/paired_model_tasks.csv")
print("  results/context_analysis/task_summary.csv")
