#!/usr/bin/env python3
"""Reproducible Celerity LongBench gold-answer PPL pilot (HF/Hyperstack).

Deliberately reuses the verified reference implementation:
    longbench_perplexity.answer_conditional_ppl
No second scoring/truncation implementation is maintained here.

Defaults: first 10 most recently updated adpretko/celerity-* HF checkpoints,
5 seeded examples from each of the 16 supported LongBench tasks (80/model),
4096-token common evaluation context capped by the model's native context.
Model revisions, LongBench revision, and sample indices are frozen on disk.

This is NOT the official LongBench generation-accuracy benchmark.

Examples:
  python benchmark_celerity_longbench_v2.py --list-only --output-dir results/pilot10_4k
  python benchmark_celerity_longbench_v2.py --output-dir results/pilot10_4k --device cuda --dtype float16
  python benchmark_celerity_longbench_v2.py --context-mode native --output-dir results/pilot10_native
  python benchmark_celerity_longbench_v2.py --source-manifest experiments/celerity_all_4k_bf16/manifest.json --min-trained-context 8192 --context-mode common --common-context 8192 --dtype bfloat16 --limit-models 0 --output-dir results/celerity_all_8k_bf16 --list-only
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import re
import sys
import tempfile
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from longbench_perplexity import (
    DEFAULT_LONGBENCH_TASKS,
    answer_conditional_ppl,
    make_longbench_prompt,
    tokenize_prompt,
)

OWNER = "adpretko"
PREFIX = f"{OWNER}/celerity-"
DATASET = "THUDM/LongBench"
DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
SCORED_BY = "longbench_perplexity.answer_conditional_ppl"
SUMMARY_FIELDS = [
    "model", "model_revision", "status", "error", "trained_context_tokens",
    "evaluation_context_tokens", "reference_policy", "scorer", "example_count",
    "scored_tokens", "nll_sum", "avg_nll", "perplexity", "example_macro_perplexity",
    "truncated_examples", "prompt_tokens_original_total", "prompt_tokens_retained_first_chunk_total",
    "load_seconds", "evaluation_seconds", "total_seconds", "examples_per_second",
    "peak_allocated_gib", "peak_reserved_gib", "tokenizer_digest",
]


def atomic_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def sha_json(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str, ensure_ascii=False).encode()).hexdigest()


def named_context(repo: str) -> int | None:
    match = re.search(r"-(\d+)k(?:-|$)", repo.lower())
    return int(match.group(1)) * 1024 if match else None


def read_checkpoint_config(repo: str, revision: str) -> dict:
    path = hf_hub_download(repo_id=repo, filename="config.json", revision=revision)
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    if cfg.get("model_type") != "celerity":
        raise ValueError(f"{repo}: config model_type != 'celerity'")
    configured = cfg.get("max_position_embeddings")
    configured = int(configured) if configured is not None else None
    claimed = named_context(repo)
    if configured and claimed and configured != claimed:
        raise ValueError(f"{repo}: name suggests {claimed}, config says {configured}")
    native = configured or claimed
    if not native or native < 2:
        raise ValueError(f"{repo}: no valid native context found")
    return {
        "model": repo, "revision": revision,
        "trained_context_tokens": native, "model_type": cfg["model_type"],
        "vocab_size": cfg.get("vocab_size"),
    }


def discover(api, args) -> tuple[list[dict], list[dict], int]:
    infos = [m for m in api.list_models(author=OWNER, full=True)
             if m.id.lower().startswith(PREFIX)]
    if args.order == "recent":
        infos.sort(key=lambda m: (str(getattr(m, "last_modified", "") or ""), m.id), reverse=True)
    else:
        infos.sort(key=lambda m: m.id)
    subset = infos[args.model_offset:]
    if not subset:
        raise RuntimeError("No Celerity models in the chosen offset")
    models, excluded = [], []
    for info in subset:
        if args.limit_models and len(models) >= args.limit_models:
            break
        try:
            if not info.sha:
                raise ValueError("No immutable model revision SHA from HF metadata")
            models.append(read_checkpoint_config(info.id, info.sha))
        except Exception as exc:
            excluded.append({"model": info.id, "reason": f"{type(exc).__name__}: {exc}"})
            print(f"WARNING metadata skipped {info.id}: {exc}", flush=True)
    if not models:
        raise RuntimeError("No usable model configs found")
    return models, excluded, len(infos)


def settings(args) -> dict:
    cfg = {
        "owner": OWNER,
        "model_prefix": PREFIX,
        "model_order": args.order, "model_offset": args.model_offset,
        "limit_models": args.limit_models,
        "tasks": args.tasks, "examples_per_task": args.examples_per_task,
        "seed": args.seed, "sampling": args.sampling,
        "context_mode": args.context_mode, "common_context": args.common_context,
        "answer_stride": args.answer_stride,
        "dtype": args.dtype, "dataset": DATASET,
        "reference_policy": "first", "use_chat_template": False,
        "scorer": SCORED_BY, "truncation_policy": "longbench_utils.middle_truncate_tokens",
    }
    # Only scientific configuration belongs in the frozen manifest. A pilot's
    # --max-run-models is intentionally NOT recorded, so it can be resumed.
    minimum = getattr(args, "min_trained_context", 0)
    if minimum:
        cfg["min_trained_context"] = minimum
    source = getattr(args, "source_manifest", None)
    if source is not None:
        cfg["source_manifest_sha256"] = hashlib.sha256(Path(source).read_bytes()).hexdigest()
    return cfg


def manifest_from_source(source_path: Path, cfg: dict, args) -> dict:
    """Create the new experiment exclusively from archived immutable inputs.

    Preserve checkpoint revisions, dataset revision, exact sample indices, and
    sample identity digests. Only filtering and evaluation context can change.
    """
    source = json.loads(source_path.read_text(encoding="utf-8"))
    original = source.get("settings", {})
    if (source.get("version") != 2 or original.get("context_mode") != "common"
            or original.get("common_context") != 4096):
        raise ValueError("Source must be the completed version-2 common-4K manifest")

    unchanged = (
        "owner", "model_prefix", "model_order", "model_offset", "limit_models",
        "tasks", "examples_per_task", "seed", "sampling", "answer_stride",
        "dtype", "dataset", "reference_policy", "use_chat_template",
        "scorer", "truncation_policy",
    )
    for key in unchanged:
        if original.get(key) != cfg.get(key):
            raise ValueError(f"Source manifest disagrees on {key}")

    if args.context_mode != "common" or args.min_trained_context < args.common_context:
        raise ValueError("Sourced run needs common context and minimum trained context >= evaluation context")
    if not source.get("longbench_revision") or not source.get("models"):
        raise ValueError("Source manifest has no pinned dataset revision or model list")

    sample_indices = source.get("sample_indices", {})
    sample_id_digest = source.get("sample_id_digest", {})
    for task in cfg["tasks"]:
        indices, digest = sample_indices.get(task), sample_id_digest.get(task)
        if (not isinstance(indices, list) or len(indices) != cfg["examples_per_task"]
                or any(type(i) is not int or i < 0 for i in indices)
                or not isinstance(digest, str)
                or re.fullmatch(r"[0-9a-f]{64}", digest) is None):
            raise ValueError(f"Source manifest lacks frozen samples/digest for {task}")

    # Revisions come directly from the 4K manifest, never current HF HEAD.
    models = [dict(m) for m in source["models"]
              if m["trained_context_tokens"] >= args.min_trained_context]
    if not models:
        raise ValueError("No models meet the minimum trained context")
    if any(m["trained_context_tokens"] < args.common_context for m in models):
        raise ValueError("Selected model context shorter than evaluation context")
    if len({m["model"] for m in models}) != len(models):
        raise ValueError("Source manifest contains duplicate model IDs")

    return {
        "version": 2,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "settings": cfg,
        "models": models,
        "total_celerity_repositories": source["total_celerity_repositories"],
        "excluded_models": source.get("excluded_models", []),
        "longbench_revision": source["longbench_revision"],
        "sample_indices": {task: list(sample_indices[task]) for task in cfg["tasks"]},
        "sample_id_digest": {task: sample_id_digest[task] for task in cfg["tasks"]},
        "source_manifest": str(source_path),
        "source_model_count": len(source["models"]),
    }


def freeze_manifest(path: Path, api, args) -> dict:
    cfg = settings(args)
    source_path = getattr(args, "source_manifest", None)
    if source_path is not None and path.resolve() == Path(source_path).resolve():
        raise ValueError("Source and output manifest must be different files")
    if path.exists():
        doc = json.loads(path.read_text(encoding="utf-8"))
        if doc["settings"] != cfg:
            raise ValueError(f"{path} was created with different settings; use another output directory")
        if source_path is not None:
            expected = manifest_from_source(Path(source_path), cfg, args)
            for key in ("models", "longbench_revision", "sample_indices", "sample_id_digest"):
                if doc.get(key) != expected[key]:
                    raise ValueError(f"{path}: {key} diverged from the frozen source")
        print(f"Reusing frozen manifest: {path}")
        return doc
    if source_path is not None:
        doc = manifest_from_source(Path(source_path), cfg, args)
        atomic_json(path, doc)
        return doc
    models, excluded, total = discover(api, args)
    rev = api.dataset_info(DATASET).sha
    if not rev:
        raise RuntimeError("Could not pin LongBench dataset revision")
    doc = {
        "version": 2, "created_utc": datetime.now(timezone.utc).isoformat(),
        "settings": cfg, "models": models, "total_celerity_repositories": total,
        "excluded_models": excluded, "longbench_revision": rev,
        "sample_indices": {}, "sample_id_digest": {},
    }
    atomic_json(path, doc)
    return doc


def effective_context(spec: dict, args) -> int:
    trained = spec["trained_context_tokens"]
    return trained if args.context_mode == "native" else min(trained, args.common_context)


def select_cases(manifest: dict, path: Path) -> list[dict]:
    import random

    cases = []
    cfg = manifest["settings"]
    for task in cfg["tasks"]:
        ds = load_dataset(DATASET, task, split="test", trust_remote_code=True,
                          revision=manifest["longbench_revision"])
        if task not in manifest["sample_indices"]:
            if cfg.get("source_manifest_sha256"):
                raise RuntimeError(f"Missing frozen sample indices for {task}; refusing to resample")
            valid = [i for i, row in enumerate(ds)
                     if row.get("answers") and str(row["answers"][0]).strip()]
            take = min(cfg["examples_per_task"], len(valid))
            if not take:
                raise RuntimeError(f"{task}: no valid gold-answer examples")
            if cfg["sampling"] == "first":
                indices = valid[:take]
            else:
                salt = int.from_bytes(hashlib.sha256(task.encode()).digest()[:8], "big")
                indices = sorted(random.Random(cfg["seed"] ^ salt).sample(valid, take))
            manifest["sample_indices"][task] = indices
            manifest["sample_id_digest"][task] = sha_json([
                (str(ds[i].get("_id", "")), str(ds[i]["answers"][0])) for i in indices
            ])
            atomic_json(path, manifest)
        indices = manifest["sample_indices"][task]
        current = [(str(ds[i].get("_id", "")), str(ds[i]["answers"][0])) for i in indices]
        if sha_json(current) != manifest["sample_id_digest"][task]:
            raise RuntimeError(f"Frozen LongBench examples changed: {task}")
        for i in indices:
            row = ds[i]
            cases.append({
                "task": task, "example_index": i, "example_id": row.get("_id"),
                "prompt": make_longbench_prompt(row, task),
                "answer": str(row["answers"][0]), "length_group": row.get("length"),
            })
    return cases


def cuda_sync(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def load_model(spec, dtype, device, cache_dir: str | None = None) -> tuple[Any, Any, str]:
    repo, revision = spec["model"], spec["revision"]
    # Keep checkpoint weights in a private temporary directory when requested.
    # Metadata discovery and LongBench datasets retain their normal caches.
    cache_kwargs = {"cache_dir": cache_dir} if cache_dir is not None else {}
    tok = AutoTokenizer.from_pretrained(repo, revision=revision,
                                        trust_remote_code=True, use_fast=True,
                                        **cache_kwargs)
    lm = AutoModelForCausalLM.from_pretrained(repo, revision=revision,
                                             trust_remote_code=True, torch_dtype=DTYPES[dtype],
                                             **cache_kwargs)
    if lm.config.model_type != "celerity":
        raise RuntimeError(f"Unexpected loaded model type: {lm.config.model_type}")
    lm.to(device)
    lm.eval()
    vocab = tok.get_vocab()
    return lm, tok, sha_json({"vocab": vocab, "special_tokens": tok.special_tokens_map})


def score_one(lm, tok, case: dict, context_limit: int, stride: int) -> dict | None:
    # The exact Step 1 scorer performs tokenization, truncation, causal shifting,
    # prompt masking, token-level NLL summation, and answer chunking.
    nll, n_tokens = answer_conditional_ppl(
        model=lm, tokenizer=tok, prompt=case["prompt"], answer=case["answer"],
        max_length=context_limit, stride=stride, use_chat_template=False,
    )
    if not n_tokens:
        return None
    if not math.isfinite(nll) or nll < 0:
        raise RuntimeError("Nonfinite or negative NLL")
    # Extra bookkeeping only. The shared scorer, not this driver, constructs
    # the actual windows and computes loss. These counts flag truncation.
    original_prompt_tokens = len(tokenize_prompt(tok, case["prompt"], use_chat_template=False))
    reference_tokens = len(tok(case["answer"], add_special_tokens=False)["input_ids"])
    if n_tokens != reference_tokens:
        raise RuntimeError("Shared gold scorer did not score every reference token")
    initial_chunk = min(stride, context_limit - 1, reference_tokens)
    retained_first = min(original_prompt_tokens, max(0, context_limit - initial_chunk))
    return {
        "task": case["task"], "example_index": case["example_index"],
        "example_id": case["example_id"], "length_group": case["length_group"],
        "reference_index": 0, "reference_policy": "first",
        "answer_tokens": n_tokens, "nll_sum": nll,
        "prompt_tokens_original": original_prompt_tokens,
        "prompt_tokens_retained_first_chunk": retained_first,
        "prompt_was_truncated_first_chunk": original_prompt_tokens > retained_first,
        "tokens_scored": n_tokens, "avg_nll": nll / n_tokens,
        "perplexity": math.exp(nll / n_tokens),
        "evaluation_context_tokens": context_limit,
    }


def base_row(spec, ctx: int) -> dict:
    return {
        "model": spec["model"], "model_revision": spec["revision"],
        "trained_context_tokens": spec["trained_context_tokens"],
        "evaluation_context_tokens": ctx, "reference_policy": "first", "scorer": SCORED_BY,
        "status": "error", "error": "", "example_count": 0, "scored_tokens": 0,
        "truncated_examples": 0, "prompt_tokens_original_total": 0,
        "prompt_tokens_retained_first_chunk_total": 0,
        "nll_sum": 0.0, "avg_nll": None, "perplexity": None,
        "example_macro_perplexity": None, "load_seconds": None,
        "evaluation_seconds": None, "total_seconds": None,
        "examples_per_second": None, "peak_allocated_gib": None,
        "peak_reserved_gib": None, "tokenizer_digest": None,
    }


def save_summary(out: Path, result: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with (out / "summary.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")
    latest: dict[str, dict] = {}
    for line in (out / "summary.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            latest[row["model"]] = row
    with (out / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(latest.values())


def completed_models(out: Path, manifest: dict) -> set[str]:
    if not (out / "summary.jsonl").exists():
        return set()
    specs = {m["model"]: m for m in manifest["models"]}
    latest = {}
    for line in (out / "summary.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            latest[r["model"]] = r
    return {
        repo for repo, row in latest.items()
        if row.get("status") == "ok" and repo in specs and
           row.get("model_revision") == specs[repo]["revision"] and
           (out / "examples" / (repo.split("/", 1)[1] + ".jsonl")).is_file()
    }


def run_one(spec: dict, cases: list[dict], args, out: Path) -> dict:
    ctx = effective_context(spec, args)
    row = base_row(spec, ctx)
    if args.answer_stride >= ctx:
        raise ValueError("--answer-stride must be < effective context length")
    name = spec["model"].split("/", 1)[1]
    path = out / "examples" / (name + ".jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    lm = tok = None
    model_cache = None
    start = time.perf_counter()
    task_totals: dict[str, dict] = defaultdict(lambda: {"nll_sum": 0., "tokens_scored": 0, "examples": 0})
    means = []
    try:
        load_start = time.perf_counter()
        if getattr(args, "ephemeral_model_cache", False):
            cache_parent = Path(args.model_cache_parent).expanduser()
            cache_parent.mkdir(parents=True, exist_ok=True)
            # Only this new directory will be deleted. Never delete the
            # user's shared HF cache or anything from previous experiments.
            model_cache = tempfile.TemporaryDirectory(
                prefix="celerity-model-", dir=str(cache_parent)
            )
            lm, tok, digest = load_model(
                spec, args.dtype, args.device, cache_dir=model_cache.name
            )
        else:
            lm, tok, digest = load_model(spec, args.dtype, args.device)
        cuda_sync(args.device)
        row["load_seconds"] = time.perf_counter() - load_start
        row["tokenizer_digest"] = digest
        if args.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        cuda_sync(args.device)
        scoring_start = time.perf_counter()
        with path.open("w", encoding="utf-8") as f:
            for j, case in enumerate(cases, 1):
                t0 = time.perf_counter()
                ex = score_one(lm, tok, case, ctx, args.answer_stride)
                cuda_sync(args.device)
                if ex is None:
                    raise RuntimeError(f"No scored gold tokens: {case['task']}#{case['example_index']}")
                ex["model"] = spec["model"]
                ex["model_revision"] = spec["revision"]
                ex["seconds"] = time.perf_counter() - t0
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
                f.flush()
                row["example_count"] += 1
                row["scored_tokens"] += ex["tokens_scored"]
                row["nll_sum"] += ex["nll_sum"]
                row["truncated_examples"] += int(ex["prompt_was_truncated_first_chunk"])
                row["prompt_tokens_original_total"] += ex["prompt_tokens_original"]
                row["prompt_tokens_retained_first_chunk_total"] += ex["prompt_tokens_retained_first_chunk"]
                means.append(ex["avg_nll"])
                part = task_totals[ex["task"]]
                part["nll_sum"] += ex["nll_sum"]
                part["tokens_scored"] += ex["tokens_scored"]
                part["examples"] += 1
                if j % max(1, args.progress_every) == 0 or j == len(cases):
                    print(f"    {j}/{len(cases)} examples: {ex['task']} ({ex['seconds']:.2f}s)", flush=True)
        cuda_sync(args.device)
        row["evaluation_seconds"] = time.perf_counter() - scoring_start
        if row["example_count"] != len(cases) or row["scored_tokens"] == 0:
            raise RuntimeError("Incomplete model evaluation")
        row["avg_nll"] = row["nll_sum"] / row["scored_tokens"]
        row["perplexity"] = math.exp(row["avg_nll"])
        row["example_macro_perplexity"] = math.exp(sum(means) / len(means))
        row["examples_per_second"] = row["example_count"] / row["evaluation_seconds"]
        if args.device.startswith("cuda"):
            row["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
            row["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
        # Task-level results are saved only after the full run succeeds.
        path_summary = out / "tasks" / (name + ".jsonl")
        path_summary.parent.mkdir(parents=True, exist_ok=True)
        with path_summary.open("w", encoding="utf-8") as f:
            for task, s in sorted(task_totals.items()):
                avg = s["nll_sum"] / s["tokens_scored"]
                f.write(json.dumps({"model": spec["model"], "task": task, **s,
                                    "avg_nll": avg, "perplexity": math.exp(avg)}) + "\n")
        row["status"] = "ok"
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
        print(f"ERROR evaluating {spec['model']}: {row['error']}", file=sys.stderr, flush=True)
        traceback.print_exc(limit=5)
    finally:
        # Model VRAM must be released before its on-disk cache is removed.
        # Also remove the temporary weights after a failed model evaluation.
        try:
            cuda_sync(args.device)
        finally:
            row["total_seconds"] = time.perf_counter() - start
            del lm, tok
            gc.collect()
            try:
                if args.device.startswith("cuda") and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            finally:
                if model_cache is not None:
                    model_cache.cleanup()
    return row


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", type=Path, default=Path("results/celerity_pilot10_4k"))
    p.add_argument("--list-only", action="store_true", help="Freeze and show selected models; do not load model weights/data")
    p.add_argument("--limit-models", type=int, default=10, help="0 = all discovered Celerity repos")
    p.add_argument("--model-offset", type=int, default=0)
    p.add_argument("--order", choices=["recent", "name"], default="recent")
    p.add_argument("--tasks", default="all")
    p.add_argument("--examples-per-task", type=int, default=5)
    p.add_argument("--sampling", choices=["random", "first"], default="random")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--context-mode", choices=["common", "native"], default="common")
    p.add_argument("--common-context", type=int, default=4096)
    p.add_argument("--source-manifest", type=Path, default=None,
                   help="Copy frozen model revisions and exact examples from a 4K manifest")
    p.add_argument("--min-trained-context", type=int, default=0,
                   help="With --source-manifest, exclude models trained below this length")
    p.add_argument("--max-run-models", type=int, default=0,
                   help="Maximum new model attempts this invocation; 0 = all (safe pilots)")
    p.add_argument("--answer-stride", type=int, default=128)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=list(DTYPES), default="float16")
    p.add_argument("--progress-every", type=int, default=10)
    p.add_argument("--force", action="store_true", help="Re-run completed models")
    p.add_argument(
        "--ephemeral-model-cache", action="store_true",
        help="Download each model to a private temporary cache, delete after evaluation",
    )
    p.add_argument(
        "--model-cache-parent", type=Path, default=Path(".cache/celerity_models"),
        help="Folder to hold per-model disposable caches (only new subfolders are deleted)",
    )
    args = p.parse_args()
    args.tasks = (list(DEFAULT_LONGBENCH_TASKS) if args.tasks == "all" else
                  [x.strip() for x in args.tasks.split(",") if x.strip()])
    bad = sorted(set(args.tasks) - set(DEFAULT_LONGBENCH_TASKS))
    if bad or not args.tasks:
        p.error(f"Invalid tasks: {bad or args.tasks}")
    if args.limit_models < 0 or args.model_offset < 0 or args.examples_per_task < 1:
        p.error("Model limits/offset must be >=0 and example count >=1")
    if args.common_context < 2 or args.answer_stride < 1 or args.progress_every < 1:
        p.error("Invalid context, stride, or progress frequency")
    if args.min_trained_context < 0 or args.max_run_models < 0:
        p.error("--min-trained-context and --max-run-models must be >= 0")
    if args.source_manifest is not None:
        if not args.source_manifest.is_file():
            p.error(f"Source manifest not found: {args.source_manifest}")
        if args.context_mode != "common" or args.min_trained_context < args.common_context:
            p.error("Sourced runs need common context and --min-trained-context >= --common-context")
        if args.model_offset != 0 or args.limit_models != 0:
            p.error("Sourced runs need --limit-models 0 and --model-offset 0; use --max-run-models for a pilot")
    elif args.min_trained_context:
        p.error("--min-trained-context requires --source-manifest")
    if args.device.startswith("cuda") and not torch.cuda.is_available() and not args.list_only:
        p.error("CUDA unavailable; use a GPU environment or --device cpu --dtype float32")
    return args


def main() -> int:
    args = parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    manifest_file = out / "manifest.json"
    manifest = freeze_manifest(manifest_file, api, args)
    models = manifest["models"]
    print(f"Frozen {len(models)} of {manifest['total_celerity_repositories']} Celerity models")
    for i, spec in enumerate(models, 1):
        print(f"  {i:2}: {spec['model']} @ {spec['revision'][:12]}"
              f" train={spec['trained_context_tokens']} eval={effective_context(spec, args)}")
    if manifest["excluded_models"]:
        print(f"WARNING: {len(manifest['excluded_models'])} models excluded during metadata selection")
    if args.list_only:
        return 0
    cases = select_cases(manifest, manifest_file)
    print(f"Fixed LongBench sample: {len(cases)} examples across {len(args.tasks)} tasks")
    done = completed_models(out, manifest)
    failures = 0
    attempted = 0
    for i, spec in enumerate(models, 1):
        name = spec["model"]
        if name in done and not args.force:
            print(f"[{i}/{len(models)}] SKIP (already successful): {name}")
            continue
        if args.max_run_models and attempted >= args.max_run_models:
            print(f"Reached --max-run-models {args.max_run_models}; resume without this flag")
            break
        attempted += 1
        print(f"[{i}/{len(models)}] START {name}", flush=True)
        result = run_one(spec, cases, args, out)
        save_summary(out, result)
        if result["status"] != "ok":
            failures += 1
            print(f"[{i}/{len(models)}] FAILED {name}")
        else:
            print(f"[{i}/{len(models)}] DONE PPL={result['perplexity']:.5g}"
                  f" load={result['load_seconds']:.1f}s eval={result['evaluation_seconds']:.1f}s", flush=True)
    print(f"Results: {out/'summary.csv'}; failures in this invocation: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
