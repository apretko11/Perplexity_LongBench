#!/usr/bin/env python3
"""Matched Celerity dropout comparisons for research discussion."""

import csv
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "dropout_meeting"
OUT.mkdir(parents=True, exist_ok=True)

EXPERIMENTS = {
    4096: "celerity_all_4k_a6000_bf16",
    8192: "celerity_all_8k_bf16",
    16384: "celerity_all_16k_bf16",
}

# Only the canonical attention-dropout checkpoint families.
# Excludes residual-dropout, learning-rate, batch-size,
# and other ablation variants.
PATTERN = re.compile(
    r"^celerity-(\d+)m-(\d+)k-ad(0p\d+)(-ild)?$"
)

models = {}

for eval_context, directory in EXPERIMENTS.items():
    folder = ROOT / "experiments" / directory

    with (folder / "summary.csv").open() as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        name = row["model"].split("/")[-1]
        match = PATTERN.fullmatch(name)

        if not match:
            continue

        size = int(match.group(1))
        trained_context = int(match.group(2)) * 1024
        dropout = match.group(3)
        ild = bool(match.group(4))

        assert row["status"] == "ok"
        assert int(row["trained_context_tokens"]) == trained_context
        assert int(row["evaluation_context_tokens"]) == eval_context

        # Compute an additional metric giving each LongBench task
        # equal weight, rather than weighting by answer length.
        task_path = folder / "tasks" / f"{name}.jsonl"

        tasks = [
            json.loads(line)
            for line in task_path.read_text().splitlines()
            if line.strip()
        ]

        assert len(tasks) == 16

        balanced_ppl = math.exp(
            sum(float(t["avg_nll"]) for t in tasks) / len(tasks)
        )

        key = (
            size,
            trained_context,
            eval_context,
            dropout,
            ild,
        )

        assert key not in models

        models[key] = {
            "model": row["model"],
            "size_m": size,
            "trained_context": trained_context,
            "evaluation_context": eval_context,
            "dropout": dropout,
            "ild": ild,
            "ppl": float(row["perplexity"]),
            "task_balanced_ppl": balanced_ppl,
        }


def compare(control, treatment, label):
    return {
        "comparison": label,
        "size_m": control["size_m"],
        "trained_context": control["trained_context"],
        "evaluation_context": control["evaluation_context"],
        "control_model": control["model"],
        "treatment_model": treatment["model"],
        "control_ppl": control["ppl"],
        "treatment_ppl": treatment["ppl"],
        "ppl_change_pct": 100 * (
            treatment["ppl"] / control["ppl"] - 1
        ),
        "task_balanced_change_pct": 100 * (
            treatment["task_balanced_ppl"]
            / control["task_balanced_ppl"] - 1
        ),
    }


attention_results = []
ild_results = []

for key, model in sorted(models.items()):
    size, train, evaluation, dropout, ild = key

    # Attention dropout vs. the zero-dropout control.
    if not ild and dropout != "0p0":
        baseline = models.get(
            (size, train, evaluation, "0p0", False)
        )

        if baseline:
            attention_results.append(
                compare(baseline, model, f"AD {dropout} vs AD 0p0")
            )

    # ILD vs. standard dropout at the same attention rate.
    if ild:
        baseline = models.get(
            (size, train, evaluation, dropout, False)
        )

        if baseline:
            ild_results.append(
                compare(baseline, model, f"ILD vs standard AD {dropout}")
            )


def write_csv(filename, rows):
    if not rows:
        raise RuntimeError(f"No matched comparisons for {filename}")

    with (OUT / filename).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


write_csv("attention_dropout.csv", attention_results)
write_csv("ild_comparisons.csv", ild_results)


# Meeting-focused subsets:
# - 8K-trained models evaluated at 8K.
# - 16K-trained models evaluated at 4K, 8K, and 16K.
SELECTED = [
    (8192, 8192),
    (16384, 4096),
    (16384, 8192),
    (16384, 16384),
]

for train, evaluation in SELECTED:
    print(f"\n{'=' * 72}")
    print(f"TRAINED: {train} | EVALUATED: {evaluation}")
    print(f"{'=' * 72}")

    print("\nATTENTION DROPOUT vs. NO DROPOUT")

    selected = [
        r for r in attention_results
        if r["trained_context"] == train
        and r["evaluation_context"] == evaluation
    ]

    for r in sorted(
        selected,
        key=lambda x: (x["size_m"], x["comparison"])
    ):
        print(
            f"{r['size_m']:4d}M  "
            f"{r['comparison']:22s} "
            f"PPL {r['ppl_change_pct']:+8.2f}%  "
            f"Task-balanced {r['task_balanced_change_pct']:+8.2f}%"
        )

    print("\nILD vs. STANDARD DROPOUT")

    selected = [
        r for r in ild_results
        if r["trained_context"] == train
        and r["evaluation_context"] == evaluation
    ]

    for r in sorted(
        selected,
        key=lambda x: (x["size_m"], x["comparison"])
    ):
        print(
            f"{r['size_m']:4d}M  "
            f"{r['comparison']:25s} "
            f"PPL {r['ppl_change_pct']:+8.2f}%  "
            f"Task-balanced {r['task_balanced_change_pct']:+8.2f}%"
        )


print("\nSaved results:")
print(f"  {OUT / 'attention_dropout.csv'}")
print(f"  {OUT / 'ild_comparisons.csv'}")
