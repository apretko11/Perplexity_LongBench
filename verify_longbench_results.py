#!/usr/bin/env python3
"""Audit aggregate PPL against individual JSONL records; no models required.

Usage:
  python verify_longbench_results.py gold --summary results/gpt2_step1_first.jsonl
  python verify_longbench_results.py generated --results results/gpt2_step3_selfppl.jsonl

Raises a nonzero exit code for missing/duplicated examples, miscounted tokens,
miscomputed NLL/PPL, or disagreeing aggregate rows.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def read_rows(path: Path) -> list[dict]:
    if not path.is_file():
        raise ValueError(f"File does not exist: {path}")
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except Exception as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc
    if not rows:
        raise ValueError(f"No rows: {path}")
    return rows


def close(found, expected, name: str, tolerance=5e-6) -> None:
    if found is None or not math.isfinite(float(found)) or not math.isclose(
        float(found), float(expected), rel_tol=tolerance, abs_tol=tolerance
    ):
        raise ValueError(f"{name}: found {found!r}, expected {expected!r}")


def validate_score(row: dict, label: str) -> None:
    tokens = int(row["tokens_scored"])
    nll = float(row["nll_sum"])
    if tokens < 0 or nll < 0 or not math.isfinite(nll):
        raise ValueError(f"{label}: invalid tokens or NLL")
    if tokens:
        close(row["avg_nll"], nll / tokens, label + ".avg_nll")
        close(row["perplexity"], math.exp(nll / tokens), label + ".perplexity")
    elif row.get("avg_nll") is not None or row.get("perplexity") is not None:
        raise ValueError(f"{label}: zero-token score must have null PPL")


def check_aggregate(rows: list[dict], summary: dict, label: str) -> None:
    tokens = sum(int(r["tokens_scored"]) for r in rows)
    nll = sum(float(r["nll_sum"]) for r in rows)
    if int(summary["examples"]) != len(rows) or int(summary["tokens_scored"]) != tokens:
        raise ValueError(f"{label}: example/token count mismatch")
    close(summary["avg_nll"], nll / tokens, label + ".avg_nll")
    close(summary["perplexity"], math.exp(nll / tokens), label + ".perplexity")
    if "nll_sum" in summary:
        close(summary["nll_sum"], nll, label + ".nll_sum")


def verify_gold(summary_path: Path, examples_path: Path | None = None) -> dict:
    if examples_path is None:
        examples_path = summary_path.with_name(summary_path.stem + "_examples.jsonl")
    summaries = read_rows(summary_path)
    examples = read_rows(examples_path)
    per_task = defaultdict(list)
    seen = set()
    for r in examples:
        if r.get("row_type") != "example":
            raise ValueError("Unexpected row type in gold examples")
        key = (r["task"], r["example_index"])
        if key in seen:
            raise ValueError(f"Duplicate example: {key}")
        seen.add(key)
        validate_score(r, f"gold {key}")
        if r.get("mode") == "answer":
            refs = r["references"]
            if not refs or len(refs) != int(r["reference_count_scored"]):
                raise ValueError(f"{key}: missing reference records")
            for i, ref in enumerate(refs):
                validate_score(ref, f"reference {key}:{i}")
            close(r["nll_sum"], sum(float(x["nll_sum"]) for x in refs), str(key) + ".reference_nll")
            if int(r["tokens_scored"]) != sum(int(x["tokens_scored"]) for x in refs):
                raise ValueError(f"{key}: reference token count mismatch")
            ref_macro = sum(float(x["avg_nll"]) for x in refs) / len(refs)
            close(r["reference_macro_avg_nll"], ref_macro, str(key) + ".reference_macro")
        per_task[r["task"]].append(r)
    expected_tasks = set(per_task)
    summaries_by_task = {r["task"]: r for r in summaries}
    if len(summaries_by_task) != len(summaries) or set(summaries_by_task) != expected_tasks | {"OVERALL"}:
        raise ValueError("Gold summary task labels/duplicates do not match example tasks")
    for task, rows in per_task.items():
        s = summaries_by_task[task]
        check_aggregate(rows, s, f"task:{task}")
        example_macro = sum(float(x["reference_macro_avg_nll"]) for x in rows) / len(rows)
        close(s["example_macro_avg_nll"], example_macro, f"task:{task}.example_macro")
    check_aggregate(examples, summaries_by_task["OVERALL"], "gold:OVERALL")
    overall = summaries_by_task["OVERALL"]
    macro = sum(float(summaries_by_task[t]["avg_nll"]) for t in per_task) / len(per_task)
    close(overall["macro_avg_nll"], macro, "gold:overall.task_macro")
    example_macro = sum(float(x["reference_macro_avg_nll"]) for x in examples) / len(examples)
    close(overall["example_macro_avg_nll"], example_macro, "gold:overall.example_macro")
    return {"type": "gold", "examples": len(examples), "tokens": overall["tokens_scored"],
            "ppl": overall["perplexity"], "tasks": len(per_task), "reference_policy": overall.get("reference_policy")}


def verify_generated(path: Path) -> dict:
    records = read_rows(path)
    examples = [r for r in records if r.get("row_type") == "example"]
    task_summaries = [r for r in records if r.get("row_type") == "task_summary"]
    overalls = [r for r in records if r.get("row_type") == "overall_summary"]
    if not examples or len(overalls) != 1 or len(task_summaries) == 0:
        raise ValueError("Missing generated examples, task summaries, or overall row")
    seen = set()
    per_task = defaultdict(list)
    policies = set()
    for r in examples:
        key = (r["task"], r["example_index"])
        if key in seen:
            raise ValueError(f"Duplicate generated example: {key}")
        seen.add(key)
        validate_score(r, f"generated {key}")
        if int(r["tokens_scored"]) != int(r["generated_tokens"]):
            raise ValueError(f"{key}: generated content token count != scored count")
        if r.get("scoring_context_policy") is not None:
            policies.add(r["scoring_context_policy"])
            if r["scoring_context_policy"] == "head_tail_prompt_recent_generation_v1":
                if not isinstance(r.get("verified_generation_chunks"), int):
                    raise ValueError(f"{key}: traced scoring missing verified chunk count")
        per_task[r["task"]].append(r)
    if len(policies) > 1:
        raise ValueError("Mixed generated scoring context policies")
    summary_by_task = {r["task"]: r for r in task_summaries}
    if len(task_summaries) != len(summary_by_task) or set(summary_by_task) != set(per_task):
        raise ValueError("Generated task summaries mismatch")
    # Zero-output examples are emitted but excluded from task/overall aggregates.
    scored = [r for r in examples if int(r["tokens_scored"]) > 0]
    for task in per_task:
        selected = [r for r in per_task[task] if int(r["tokens_scored"]) > 0]
        if not selected:
            raise ValueError(f"{task}: no scored examples for its summary")
        check_aggregate(selected, summary_by_task[task], f"generated:{task}")
    check_aggregate(scored, overalls[0], "generated:OVERALL")
    if policies and overalls[0].get("scoring_context_policy") != next(iter(policies)):
        raise ValueError("Overall scoring policy mismatch")
    return {"type": "generated", "examples": len(examples), "scored_examples": len(scored),
            "tokens": overalls[0]["tokens_scored"], "ppl": overalls[0]["perplexity"], "tasks": len(per_task),
            "scoring_context_policy": overalls[0].get("scoring_context_policy")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="kind", required=True)
    gold = sub.add_parser("gold")
    gold.add_argument("--summary", required=True, type=Path)
    gold.add_argument("--examples", type=Path)
    generated = sub.add_parser("generated")
    generated.add_argument("--results", required=True, type=Path)
    args = parser.parse_args()
    result = (verify_gold(args.summary, args.examples) if args.kind == "gold"
              else verify_generated(args.results))
    print("PASS: " + json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
