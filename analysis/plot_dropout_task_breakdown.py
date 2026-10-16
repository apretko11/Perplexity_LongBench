#!/usr/bin/env python3
"""Meeting report and figures for AD 0.2 vs AD 0.0 at 16K."""

import csv
import math
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results" / "dropout_meeting"
FIGS = BASE / "figures"
SIZES = (271, 504, 906)

# Capture the original analysis script's raw terminal output.
raw = subprocess.run(
    [sys.executable, str(ROOT / "analysis" / "dropout_task_breakdown.py")],
    cwd=ROOT,
    text=True,
    capture_output=True,
    check=True,
).stdout

print(raw, end="")

with (BASE / "dropout_task_breakdown_16k.csv").open(newline="") as f:
    rows = list(csv.DictReader(f))

data = {}

for r in rows:
    key = (int(r["size_m"]), r["task"])
    assert key not in data

    data[key] = {
        "pct": float(r["ppl_change_pct"]),
        "dnll": float(r["delta_nll"]),
        "tokens": int(r["answer_tokens"]),
    }

assert len(data) == 48

TASKS = sorted(
    {task for size, task in data},
    key=lambda task: sum(data[size, task]["dnll"] for size in SIZES),
)

assert len(TASKS) == 16
assert all((size, task) in data for size in SIZES for task in TASKS)

FIGS.mkdir(exist_ok=True)


def save(fig, name):
    for ext in ("png", "pdf"):
        path = FIGS / f"{name}.{ext}"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        print("Saved:", path)

    plt.close(fig)


def contributions(size, balanced):
    total_tokens = sum(data[size, t]["tokens"] for t in TASKS)

    return [
        data[size, t]["dnll"] * (
            1 / len(TASKS)
            if balanced
            else data[size, t]["tokens"] / total_tokens
        )
        for t in TASKS
    ]


def ppl_change(size, balanced):
    return 100 * math.expm1(sum(contributions(size, balanced)))


# ============================================================
# FIGURE 4: TASK-LEVEL PPL CHANGES
# ============================================================

matrix = [
    [data[s, t]["dnll"] for s in SIZES]
    for t in TASKS
]

cap = max(abs(value) for row in matrix for value in row)

fig, ax = plt.subplots(figsize=(11, 9.5))

im = ax.imshow(
    matrix,
    aspect="auto",
    cmap="RdBu_r",
    norm=TwoSlopeNorm(vmin=-cap, vcenter=0, vmax=cap),
)

ax.set_xticks(range(3), [f"{s}M" for s in SIZES])
ax.set_yticks(range(16), TASKS)
ax.set_xlabel("Model size — trained and evaluated at 16K")

for i, task in enumerate(TASKS):
    for j, size in enumerate(SIZES):
        value = data[size, task]

        ax.text(
            j, i,
            f'{value["pct"]:+.1f}%',
            ha="center",
            va="center",
            fontsize=9,
            color=(
                "white"
                if abs(value["dnll"]) > 0.55 * cap
                else "black"
            ),
        )

fig.colorbar(
    im,
    ax=ax,
    shrink=0.79,
    label="Change in log PPL (dropout − no dropout)",
)

fig.suptitle(
    "Attention Dropout 0.2 vs. No Dropout: LongBench Tasks",
    fontsize=16,
)

fig.text(
    0.5, 0.015,
    "Labels are PPL changes (%); negative = dropout helps. "
    "Five examples per task.",
    ha="center",
    fontsize=10,
)

fig.tight_layout(rect=(0, 0.045, 1, 0.95))

save(fig, "04_ad0p2_vs_ad0p0_task_heatmap_16k")


# ============================================================
# FIGURE 5: POOLED VS TASK-BALANCED CONTRIBUTIONS
# ============================================================

fig, axes = plt.subplots(
    2, 3,
    figsize=(17, 12),
    sharey=True,
)

for row, balanced in enumerate((False, True)):
    values_by_size = {
        s: contributions(s, balanced)
        for s in SIZES
    }

    limit = max(
        abs(x)
        for values in values_by_size.values()
        for x in values
    ) * 1.15

    for col, size in enumerate(SIZES):
        ax = axes[row, col]
        values = values_by_size[size]

        ax.barh(
            range(16),
            values,
            color=[
                "#4477AA" if x < 0 else "#CC6677"
                for x in values
            ],
        )

        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlim(-limit, limit)
        ax.set_ylim(15.5, -0.5)
        ax.set_yticks(range(16), TASKS)
        ax.tick_params(axis="y", labelleft=(col == 0))
        ax.grid(axis="x", alpha=0.2)
        ax.set_axisbelow(True)

        label = "Task-balanced" if balanced else "Pooled"

        ax.set_title(
            f"{size}M | {label}: "
            f"{ppl_change(size, balanced):+.2f}%"
        )

        ax.set_xlabel("Contribution to Δ log PPL")

fig.suptitle(
    "Why Pooled and Task-Balanced Perplexity Disagree",
    fontsize=17,
)

fig.text(
    0.5, 0.015,
    "Blue = dropout helps; red = no dropout helps. "
    "Contributions sum in log-PPL space.",
    ha="center",
    fontsize=10,
)

fig.tight_layout(rect=(0, 0.04, 1, 0.96))

save(fig, "05_ad0p2_vs_ad0p0_task_weighting_16k")


# ============================================================
# SAVE DISTINCT TEXT SUMMARY + ORIGINAL RAW OUTPUT
# ============================================================

existing = next(
    (
        p for p in (
            BASE / "Attention_Dropout_Perplexity.txt",
            FIGS / "Attention_Dropout_Perplexity.txt",
        )
        if p.exists()
    ),
    BASE / "Attention_Dropout_Perplexity.txt",
)

summary_path = existing.with_name(
    "Attention_Dropout_Task_Breakdown_16K.txt"
)


def pct(size, task):
    return f'{data[size, task]["pct"]:+.2f}%'


summary = (
    "ATTENTION DROPOUT: TASK-LEVEL LONGBENCH RESULTS (16K)\n\n"
    "SUMMARY\n\n"

    f"For 906M, attention dropout 0.2 vs. no dropout improves "
    f"pooled gold-answer PPL ({ppl_change(906, False):+.2f}%) "
    f"and task-balanced PPL ({ppl_change(906, True):+.2f}%). "
    f"Large task-level gains include Passage Retrieval "
    f"({pct(906, 'passage_retrieval_en')}), "
    f"TriviaQA ({pct(906, 'triviaqa')}), "
    f"and Musique ({pct(906, 'musique')}). "

    f"Smaller models show mixed results: "
    f"271M pooled {ppl_change(271, False):+.2f}% versus "
    f"task-balanced {ppl_change(271, True):+.2f}%, "
    f"and 504M pooled {ppl_change(504, False):+.2f}% versus "
    f"task-balanced {ppl_change(504, True):+.2f}%. "
    f"In particular, TREC worsens for 271M "
    f"({pct(271, 'trec')}) and 504M ({pct(504, 'trec')}), "
    f"while Passage Count worsens strongly for 271M "
    f"({pct(271, 'passage_count')}).\n\n"

    "Pooled perplexity weights each task by its gold-answer "
    "token count; task-balanced perplexity assigns equal weight "
    "to all 16 tasks in per-token negative log-likelihood space. "
    "Short-answer tasks can therefore matter much more to the "
    "balanced metric. These are descriptive results from five "
    "examples per task, evaluating the first reference answer "
    "rather than free-generation accuracy.\n\n"

    "================================================================\n"
    "RAW TERMINAL OUTPUT — analysis/dropout_task_breakdown.py\n"
    "================================================================\n\n"

    + raw.rstrip("\n") + "\n"
)

summary_path.write_text(summary, encoding="utf-8")

print("Saved:", summary_path)
print("Done: two figures (PNG + PDF) and a distinct summary with raw output.")
