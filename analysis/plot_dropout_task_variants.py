#!/usr/bin/env python3
"""16K task-level plots for AD0.1, AD0.4 and ILD0.2/0.4/0.8 vs AD0.0.

Run from repo with: python analysis/plot_dropout_task_variants.py
Uses archived 16K task JSONL only; never modifies existing results or notes.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[1]
TASK_DIR = ROOT / "experiments/celerity_all_16k_bf16/tasks"
OUT = ROOT / "results/dropout_meeting"
FIG = OUT / "figures"
SIZES = (271, 504, 906)

# (filename slug, figure title, treatment suffix, reference suffix)
COMPARISONS = (
    ("ad0p1_vs_ad0p0", "Attention dropout 0.1 vs. no dropout", "ad0p1", "ad0p0"),
    ("ad0p4_vs_ad0p0", "Attention dropout 0.4 vs. no dropout", "ad0p4", "ad0p0"),
    ("ad0p2_ild_vs_ad0p0", "AD 0.2 + ILD vs. no dropout", "ad0p2-ild", "ad0p0"),
    ("ad0p4_ild_vs_ad0p0", "AD 0.4 + ILD vs. no dropout", "ad0p4-ild", "ad0p0"),
    ("ad0p8_ild_vs_ad0p0", "AD 0.8 + ILD vs. no dropout", "ad0p8-ild", "ad0p0"),
)


def read_task_file(size: int, suffix: str) -> dict[str, dict]:
    path = TASK_DIR / f"celerity-{size}m-16k-{suffix}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Missing model task file: {path}")
    values = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    tasks = {row["task"]: row for row in values}
    if len(values) != len(tasks) or len(tasks) != 16:
        raise ValueError(f"Expected 16 unique tasks in {path}")
    for row in values:
        if int(row["examples"]) != 5 or int(row["tokens_scored"]) <= 0:
            raise ValueError(f"Bad example or token count: {path}: {row['task']}")
        if not math.isfinite(float(row["avg_nll"])):
            raise ValueError(f"Nonfinite loss: {path}: {row['task']}")
    return tasks


def build_comparison(slug, title, treatment, control):
    data = {}
    for size in SIZES:
        a = read_task_file(size, control)
        b = read_task_file(size, treatment)
        if set(a) != set(b):
            raise ValueError(f"Mismatched task sets for {size}M {slug}")
        for task in a:
            tok_a, tok_b = int(a[task]["tokens_scored"]), int(b[task]["tokens_scored"])
            if tok_a != tok_b:
                raise ValueError(f"Reference tokens differ for {size}M {task} {slug}")
            delta = float(b[task]["avg_nll"]) - float(a[task]["avg_nll"])
            data[size, task] = {
                "delta_nll": delta,
                "ppl_pct": 100 * math.expm1(delta),
                "tokens": tok_a,
            }
    return {"slug": slug, "title": title, "treatment": treatment,
            "control": control, "data": data}


def save_figure(fig, stem: str):
    for ext in ("png", "pdf"):
        dest = FIG / f"{stem}.{ext}"
        fig.savefig(dest, bbox_inches="tight", dpi=220)
        print("Saved:", dest)
    plt.close(fig)


def contribution(comp, size, task, task_balanced):
    d = comp["data"]
    if task_balanced:
        weight = 1 / len(ORDER)
    else:
        weight = d[size, task]["tokens"] / sum(d[size, t]["tokens"] for t in ORDER)
    return d[size, task]["delta_nll"] * weight


def aggregate_pct(comp, size, task_balanced):
    return 100 * math.expm1(sum(contribution(comp, size, t, task_balanced) for t in ORDER))


def heatmap(comp):
    d = comp["data"]
    grid = [[d[size, task]["delta_nll"] for size in SIZES] for task in ORDER]
    largest = max(0.001, max(abs(value) for row in grid for value in row))
    fig, ax = plt.subplots(figsize=(11, 9.5))
    im = ax.imshow(grid, aspect="auto", cmap="RdBu_r",
                   norm=TwoSlopeNorm(vmin=-largest, vcenter=0, vmax=largest))
    ax.set_xticks(range(len(SIZES)), [f"{size}M" for size in SIZES])
    ax.set_yticks(range(len(ORDER)), ORDER)
    ax.set_xlabel("Model size — trained and evaluated at 16K")
    for i, task in enumerate(ORDER):
        for j, size in enumerate(SIZES):
            cell = d[size, task]
            ax.text(j, i, f"{cell['ppl_pct']:+.1f}%", ha="center", va="center",
                    fontsize=9, color="white" if abs(cell["delta_nll"]) > 0.55 * largest else "black")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Change in log PPL (treatment − control)")
    fig.suptitle(f"{comp['title']} — LongBench tasks (16K)", fontsize=15)
    fig.text(0.5, 0.012, "Labels: task PPL change (%). Negative = treatment beats AD 0.0; 5 examples per task.",
             ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.045, 1, 0.95))
    save_figure(fig, f"04_{comp['slug']}_task_heatmap_16k")


def weighting(comp):
    fig, axes = plt.subplots(2, 3, figsize=(17, 12), sharey=True)
    for r, balanced in enumerate((False, True)):
        contribs = {size: [contribution(comp, size, task, balanced) for task in ORDER] for size in SIZES}
        bound = max(0.005, 1.15 * max(abs(x) for vals in contribs.values() for x in vals))
        for c, size in enumerate(SIZES):
            ax = axes[r, c]
            values = contribs[size]
            ax.barh(range(len(ORDER)), values,
                    color=["#4477AA" if v < 0 else "#CC6677" for v in values])
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_xlim(-bound, bound)
            ax.set_ylim(len(ORDER) - 0.5, -0.5)
            ax.set_yticks(range(len(ORDER)), ORDER)
            ax.tick_params(axis="y", labelleft=(c == 0))
            ax.grid(axis="x", alpha=0.2)
            ax.set_axisbelow(True)
            label = "Task-balanced" if balanced else "Pooled"
            ax.set_title(f"{size}M | {label}: {aggregate_pct(comp, size, balanced):+.2f}%")
            ax.set_xlabel("Contribution to Δ log PPL")
    fig.suptitle(f"{comp['title']} — Why weighting matters", fontsize=16)
    fig.text(0.5, 0.015, "Blue: treatment beats AD 0.0; red: AD 0.0 wins. Contributions sum in log-PPL space.",
             ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    save_figure(fig, f"05_{comp['slug']}_task_weighting_16k")


def write_details(comparisons):
    rows = []
    lines = [
        "LONGBENCH TASK BREAKDOWN: AD RATES AND ILD VS ZERO-DROPOUT BASELINE (16K)",
        "Every treatment is compared with the same AD 0.0 baseline at matched size, train context, and evaluation context.",
        "ILD comparisons include an attention-dropout-rate change AND ILD; they do not isolate the incremental ILD effect.",
        "Negative change means the treatment has lower gold-answer perplexity.",
        "Per-task PPL values use 5 examples each; results are descriptive.", "",
    ]
    for comp in comparisons:
        lines.append("=" * 75)
        lines.append(comp["title"].upper())
        lines.append("=" * 75)
        for size in SIZES:
            lines.append(f"\n{size}M | pooled PPL: {aggregate_pct(comp, size, False):+.2f}% | "
                         f"task-balanced PPL: {aggregate_pct(comp, size, True):+.2f}%")
            for task in ORDER:
                d = comp["data"][size, task]
                lines.append(f"  {task:26s} {d['ppl_pct']:+9.2f}%  "
                             f"delta NLL/token: {d['delta_nll']:+.6f}  answer tokens: {d['tokens']}")
                rows.append({"comparison": comp["slug"], "control": comp["control"],
                             "treatment": comp["treatment"], "size_m": size, "task": task,
                             "ppl_change_pct": d["ppl_pct"], "delta_nll": d["delta_nll"],
                             "answer_tokens": d["tokens"]})
        lines.append("")
    text_path = OUT / "Attention_Dropout_Rates_ILD_vs_Baseline_16K.txt"
    csv_path = OUT / "dropout_rates_ild_vs_baseline_16k.csv"
    # Distinct names: do not modify the existing AD0.2 meeting summary.
    text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("Saved:", text_path)
    print("Saved:", csv_path)


def main():
    global ORDER
    if not TASK_DIR.is_dir():
        raise SystemExit(f"Cannot find frozen 16K results: {TASK_DIR}")
    FIG.mkdir(parents=True, exist_ok=True)
    comparisons = [build_comparison(*t) for t in COMPARISONS]
    # Use the same task order as the existing AD0.2 heatmap for visual comparability.
    reference = build_comparison("ad0p2_vs_ad0p0", "AD 0.2 vs no dropout", "ad0p2", "ad0p0")
    ORDER = sorted({task for _, task in reference["data"]},
                   key=lambda task: sum(reference["data"][size, task]["delta_nll"] for size in SIZES))
    if len(ORDER) != 16:
        raise ValueError("Expected 16 tasks")
    for comp in comparisons:
        print("\n", comp["title"])
        heatmap(comp)
        weighting(comp)
    write_details(comparisons)
    print("\nDone. Five comparisons, ten figures (each saved in PNG and PDF).")


if __name__ == "__main__":
    main()

