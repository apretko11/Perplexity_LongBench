#!/usr/bin/env python3
"""Plot matched Celerity LongBench dropout comparisons."""

import csv
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "dropout_meeting"
OUT = DATA / "figures"

RATES = (0.1, 0.2, 0.4)
SIZES = (271, 504, 906)

METRICS = (
    ("ppl_change_pct", "Pooled answer PPL change (%)"),
    ("task_balanced_change_pct", "Task-balanced PPL change (%)"),
)


def load_csv(filename):
    path = DATA / filename

    if not path.is_file():
        raise SystemExit(
            f"Missing {path}. Run analysis/dropout_meeting.py first."
        )

    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise RuntimeError(f"Empty CSV: {path}")

    return rows


def prepare(rows, pattern):
    """Index comparisons by size, training context, eval context, rate."""

    index = {}

    for row in rows:
        match = re.fullmatch(pattern, row["comparison"])

        if not match:
            raise ValueError(
                f"Unexpected comparison: {row['comparison']}"
            )

        rate = float(match.group(1).replace("p", "."))

        key = (
            int(row["size_m"]),
            int(row["trained_context"]),
            int(row["evaluation_context"]),
            rate,
        )

        if key in index:
            raise ValueError(f"Duplicate comparison: {key}")

        for metric, _ in METRICS:
            row[metric] = float(row[metric])

            if not math.isfinite(row[metric]):
                raise ValueError(f"Nonfinite {metric}: {key}")

        index[key] = row

    return index


def value(index, size, train, evaluation, rate, metric):
    key = (size, train, evaluation, rate)

    if key not in index:
        raise ValueError(f"Missing comparison: {key}")

    return index[key][metric]


def decorate(ax, horizontal=True):
    if horizontal:
        ax.axhline(0, linestyle="--", linewidth=1, alpha=0.65)
        ax.grid(axis="y", alpha=0.18)
    else:
        ax.axvline(0, linestyle="--", linewidth=1, alpha=0.65)
        ax.grid(axis="x", alpha=0.18)

    ax.set_axisbelow(True)


def save(fig, filename):
    OUT.mkdir(parents=True, exist_ok=True)

    for extension in ("png", "pdf"):
        path = OUT / f"{filename}.{extension}"

        fig.savefig(
            path,
            dpi=220,
            bbox_inches="tight",
        )

        print(f"Saved: {path}")

    plt.close(fig)


# ============================================================
# FIGURE 1: ATTENTION DROPOUT
# ============================================================

def figure_attention_dropout(attention):

    fig, axes = plt.subplots(
        2, 2,
        figsize=(15, 9),
        sharex="col",
        sharey="row",
    )

    for col, context in enumerate((8192, 16384)):

        for row, (metric, label) in enumerate(METRICS):

            ax = axes[row, col]

            for size in SIZES:

                # Include the zero-dropout reference point.
                x = [0.0, *RATES]

                y = [0.0] + [
                    value(
                        attention,
                        size,
                        context,
                        context,
                        rate,
                        metric,
                    )
                    for rate in RATES
                ]

                ax.plot(
                    x,
                    y,
                    marker="o",
                    linewidth=2,
                    label=f"{size}M",
                )

            decorate(ax)

            ax.set_xticks([0.0, *RATES])

            ax.set_title(
                f"Trained {context // 1024}K | "
                f"Evaluated {context // 1024}K"
            )

            if col == 0:
                ax.set_ylabel(label)

            if row == 1:
                ax.set_xlabel("Attention dropout probability")

    handles, labels = axes[0, 0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.045),
    )

    fig.suptitle(
        "Attention Dropout vs. No Dropout",
        fontsize=17,
        y=0.985,
    )

    fig.text(
        0.5, 0.012,
        "Negative = lower gold-answer perplexity (better). "
        "Comparisons use matched model size and training context.",
        ha="center",
        fontsize=10,
    )

    fig.tight_layout(rect=(0, 0.09, 1, 0.95))

    save(fig, "01_attention_dropout")


# ============================================================
# FIGURE 2: ILD VS. STANDARD ATTENTION DROPOUT
# ============================================================

def figure_ild(ild):

    fig, axes = plt.subplots(
        2, 2,
        figsize=(15, 10),
        sharex="row",
    )

    combinations = [
        (size, rate)
        for size in SIZES
        for rate in (0.2, 0.4)
    ]

    positions = [0, 0.8, 2, 2.8, 4, 4.8]

    labels = [
        f"{size}M | AD {rate:.1f}"
        for size, rate in combinations
    ]

    for row, (metric, metric_label) in enumerate(METRICS):

        # Use consistent horizontal scales across the two
        # training-context panels for each metric.
        all_values = [
            value(ild, size, ctx, ctx, rate, metric)
            for ctx in (8192, 16384)
            for size, rate in combinations
        ]

        bound = max(
            2.0,
            max(abs(x) for x in all_values) * 1.38,
        )

        for col, context in enumerate((8192, 16384)):

            ax = axes[row, col]

            values = [
                value(
                    ild,
                    size,
                    context,
                    context,
                    rate,
                    metric,
                )
                for size, rate in combinations
            ]

            for position, (size, rate), result in zip(
                positions, combinations, values
            ):

                ax.barh(
                    position,
                    result,
                    height=0.67,
                    hatch="//" if rate == 0.4 else None,
                )

                padding = 0.025 * bound

                ax.text(
                    result + (
                        padding if result >= 0 else -padding
                    ),
                    position,
                    f"{result:+.1f}%",
                    va="center",
                    ha="left" if result >= 0 else "right",
                    fontsize=9,
                )

            decorate(ax, horizontal=False)

            ax.set_yticks(positions, labels)
            ax.invert_yaxis()
            ax.set_xlim(-bound, bound)

            ax.set_title(
                f"Trained {context // 1024}K | "
                f"Evaluated {context // 1024}K"
            )

            ax.set_xlabel(metric_label)

    fig.suptitle(
        "Input-Layer Dropout vs. Standard Attention Dropout",
        fontsize=17,
        y=0.985,
    )

    fig.text(
        0.5, 0.012,
        "Negative = ILD performs better. "
        "Solid bars: AD 0.2 | Hatched bars: AD 0.4.",
        ha="center",
        fontsize=10,
    )

    fig.tight_layout(rect=(0, 0.045, 1, 0.95))

    save(fig, "02_ild_comparison")


# ============================================================
# FIGURE 3: DROPOUT EFFECT VS. CONTEXT LENGTH
# ============================================================

def figure_context_sensitivity(attention):

    fig, axes = plt.subplots(
        2, 3,
        figsize=(17, 8.7),
        sharex="col",
        sharey="row",
    )

    contexts = (4096, 8192, 16384)

    for col, size in enumerate(SIZES):

        for row, (metric, metric_label) in enumerate(METRICS):

            ax = axes[row, col]

            for rate in RATES:

                x = [4, 8, 16]

                y = [
                    value(
                        attention,
                        size,
                        16384,
                        context,
                        rate,
                        metric,
                    )
                    for context in contexts
                ]

                ax.plot(
                    x,
                    y,
                    marker="o",
                    linewidth=2,
                    label=f"AD {rate:.1f}",
                )

            decorate(ax)

            ax.set_xticks([4, 8, 16])

            ax.set_title(
                f"{size}M | Trained at 16K"
            )

            if col == 0:
                ax.set_ylabel(metric_label)

            if row == 1:
                ax.set_xlabel(
                    "Evaluation context (K tokens)"
                )

    handles, labels = axes[0, 0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.045),
    )

    fig.suptitle(
        "How Does Attention Dropout Behave at Longer Contexts?",
        fontsize=17,
        y=0.985,
    )

    fig.text(
        0.5, 0.012,
        "All models trained at 16K. "
        "Each point is relative to its matched zero-dropout model "
        "at the same evaluation context.",
        ha="center",
        fontsize=10,
    )

    fig.tight_layout(rect=(0, 0.09, 1, 0.95))

    save(fig, "03_dropout_context_sensitivity")


# ============================================================
# MAIN
# ============================================================

def main():

    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })

    attention = prepare(
        load_csv("attention_dropout.csv"),
        r"AD (0p\d+) vs AD 0p0",
    )

    ild = prepare(
        load_csv("ild_comparisons.csv"),
        r"ILD vs standard AD (0p\d+)",
    )

    figure_attention_dropout(attention)
    figure_ild(ild)
    figure_context_sensitivity(attention)

    print("\nAll figures generated.")
    print(f"Output directory: {OUT}")


if __name__ == "__main__":
    main()
