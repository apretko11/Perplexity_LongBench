#!/usr/bin/env python3
"""Strict offline audit of a completed Celerity BEST-reference batch run.

Usage:
  python verify_best_reference_v3.py --output-dir results/celerity_all_4k_bestref_bf16 \
      --source-manifest experiments/celerity_all_4k_a6000_bf16/manifest.json
"""
import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def close(a, b, label):
    if not math.isclose(float(a), float(b), rel_tol=1e-8, abs_tol=1e-5):
        raise AssertionError(f"{label}: {a!r} != {b!r}")


def rows_jsonl(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def audit(output, source):
    out = Path(output)
    source = Path(source)
    baseline = json.loads(source.read_text(encoding="utf-8"))
    frozen = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    c = frozen["settings"]["common_context"]
    assert c in (4096, 8192, 16384), c
    assert frozen["version"] == 3
    assert frozen["settings"]["reference_policy"] == "best"
    assert frozen["settings"]["best_reference_selection"] == "minimum_mean_nll_per_token_v1"
    assert frozen["settings"]["dtype"] == "bfloat16"
    assert frozen["settings"]["source_manifest_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    for key in ("longbench_revision", "sample_indices", "sample_id_digest"):
        assert frozen[key] == baseline[key], key
    expected = [m for m in baseline["models"] if m["trained_context_tokens"] >= c]
    assert frozen["models"] == expected
    assert len(expected) == {4096: 189, 8192: 168, 16384: 21}[c]
    specs = {s["model"]: s for s in expected}
    cases = {(t, i) for t, indices in baseline["sample_indices"].items() for i in indices}
    assert len(cases) == 80
    with (out / "summary.csv").open(newline="", encoding="utf-8") as f:
        summary = list(csv.DictReader(f))
    assert len(summary) == len(specs), (len(summary), len(specs))
    assert len({s["model"] for s in summary}) == len(summary)
    dist = Counter()
    for s in summary:
        model = s["model"]
        spec = specs[model]
        assert s["status"] == "ok", (model, s.get("error"))
        assert s["model_revision"] == spec["revision"]
        assert s["reference_policy"] == "best"
        assert int(s["evaluation_context_tokens"]) == c
        assert int(s["example_count"]) == 80
        name = model.split("/", 1)[1]
        examples = rows_jsonl(out / "examples" / (name + ".jsonl"))
        tasks = rows_jsonl(out / "tasks" / (name + ".jsonl"))
        assert len(examples) == 80 and len(tasks) == 16, model
        assert {(e["task"], e["example_index"]) for e in examples} == cases
        assert {t["task"] for t in tasks} == set(baseline["sample_indices"])
        tok_sum = 0
        nll_sum = 0.0
        for e in examples:
            assert e["model"] == model and e["model_revision"] == spec["revision"]
            assert e["reference_policy"] == "best"
            assert e["evaluation_context_tokens"] == c
            refs = e["references"]
            assert refs and len(refs) == e["reference_count_scored"]
            assert e["reference_count_available"] >= len(refs)
            assert all(math.isfinite(r["avg_nll"]) for r in refs)
            winner = min(refs, key=lambda r: r["avg_nll"])
            assert e["reference_index"] == winner["reference_index"]
            assert sum(bool(r["selected"]) for r in refs) == 1
            assert winner["selected"]
            assert e["tokens_scored"] == winner["tokens_scored"]
            close(e["nll_sum"], winner["nll_sum"], model)
            close(e["avg_nll"], winner["avg_nll"], model)
            close(e["perplexity"], math.exp(e["avg_nll"]), model)
            nll_sum += e["nll_sum"]
            tok_sum += e["tokens_scored"]
            dist[len(refs)] += 1
        assert int(s["scored_tokens"]) == tok_sum, model
        close(s["nll_sum"], nll_sum, model)
        close(s["perplexity"], math.exp(nll_sum / tok_sum), model)
        for task in tasks:
            subt = [e for e in examples if e["task"] == task["task"]]
            assert len(subt) == 5
            assert task["reference_policy"] == "best"
            assert task["tokens_scored"] == sum(e["tokens_scored"] for e in subt)
            close(task["nll_sum"], sum(e["nll_sum"] for e in subt), model + task["task"])
            close(task["perplexity"], math.exp(task["nll_sum"] / task["tokens_scored"]), task["task"])
    assert len(list((out / "examples").glob("*.jsonl"))) == len(specs)
    assert len(list((out / "tasks").glob("*.jsonl"))) == len(specs)
    print(f"PASS: {len(specs)} models x 80 examples, 16 tasks/model, context={c}")
    print("PASS: exact frozen checkpoints, dataset revision and sample identities")
    print("PASS: all selected reference indices, candidate traces and summary aggregates")
    print("Distinct reference counts per example:", dict(sorted(dist.items())))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--source-manifest", required=True, type=Path)
    a = p.parse_args()
    audit(a.output_dir, a.source_manifest)
