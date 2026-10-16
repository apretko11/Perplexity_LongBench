#!/usr/bin/env python3
"""Task-level effects of AD 0.2 vs. no dropout at 16K."""

import json
import math
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "experiments/celerity_all_16k_bf16/tasks"
OUT = ROOT / "results/dropout_meeting"

def load(size, dropout):
    path = TASKS / f"celerity-{size}m-16k-ad{dropout}.jsonl"

    return {
        row["task"]: row
        for row in (
            json.loads(line)
            for line in path.read_text().splitlines()
            if line.strip()
        )
    }

results = []

for size in (271, 504, 906):

    baseline = load(size, "0p0")
    dropout = load(size, "0p2")

    assert baseline.keys() == dropout.keys()
    assert len(baseline) == 16

    family = []

    for task in baseline:

        a = baseline[task]
        b = dropout[task]

        assert a["examples"] == b["examples"] == 5
        assert a["tokens_scored"] == b["tokens_scored"]

        delta_nll = b["avg_nll"] - a["avg_nll"]
        change = 100 * math.expm1(delta_nll)

        row = {
            "size_m": size,
            "task": task,
            "answer_tokens": a["tokens_scored"],
            "baseline_ppl": a["perplexity"],
            "dropout_ppl": b["perplexity"],
            "ppl_change_pct": change,
            "delta_nll": delta_nll,
        }

        family.append(row)
        results.append(row)

    family.sort(key=lambda x: x["ppl_change_pct"])

    print(f"\n{'=' * 60}")
    print(f"{size}M | AD 0.2 vs. AD 0.0 | 16K")
    print(f"{'=' * 60}")

    print("\nFive tasks benefiting most:")
    for row in family[:5]:
        print(f"  {row['task']:25s} {row['ppl_change_pct']:+8.2f}%")

    print("\nFive tasks benefiting least:")
    for row in family[-5:]:
        print(f"  {row['task']:25s} {row['ppl_change_pct']:+8.2f}%")

path = OUT / "dropout_task_breakdown_16k.csv"

with path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(results[0]))
    writer.writeheader()
    writer.writerows(results)

print(f"\nSaved: {path}")
