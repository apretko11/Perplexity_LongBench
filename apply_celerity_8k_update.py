#!/usr/bin/env python3
"""One-time patch for the Celerity LongBench runner at commit 4558a30.

Run from the root of an unchanged checkout of Perplexity_LongBench:
    python ~/apply_celerity_8k_update.py

Only modifies benchmark_celerity_longbench_v2.py and its existing test file.
Refuses to patch unexpected source content, dirty files, or invalid Python.
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

BASE_COMMIT = "4558a30f98ca2d3ef82d76c9d254d837bee455be"
DRIVER = Path("benchmark_celerity_longbench_v2.py")
TESTS = Path("tests/test_benchmark_celerity_longbench_v2.py")


def run(*cmd: str) -> str:
    return subprocess.check_output(cmd, text=True).strip()


def replace_one(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise RuntimeError(f"Expected exactly one anchor, found {source.count(old)}: {old[:90]!r}")
    return source.replace(old, new, 1)


def replace_section(source: str, start: str, end: str, replacement: str) -> str:
    if source.count(start) != 1 or source.count(end) != 1:
        raise RuntimeError(f"Section anchor not unique: {start[:90]!r} ... {end[:90]!r}")
    a = source.index(start)
    b = source.index(end, a + len(start))
    return source[:a] + replacement + source[b:]


MANIFEST_CODE = '''def settings(args) -> dict:
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


'''


NEW_TESTS = '''    def test_frozen_8k_experiment_reuses_4k_revisions_and_examples(self):
        rows = [
            dict(_id='id0', context='alpha', input='Q', answers=['first'], length='4k'),
            dict(_id='id1', context='beta', input='Q', answers=['second'], length='4k'),
            dict(_id='id2', context='gamma', input='Q', answers=['third'], length='4k'),
        ]
        cases_digest = pilot.sha_json([
            (str(rows[i]['_id']), str(rows[i]['answers'][0])) for i in [0, 2]
        ])
        specs = [
            dict(model=f'adpretko/celerity-271m-{size}k-ad0p0',
                 revision=str(size) * 40, trained_context_tokens=size * 1024)
            for size in (4, 8, 16)
        ]
        baseline_args = SimpleNamespace(
            order='recent', model_offset=0, limit_models=0,
            tasks=['narrativeqa'], examples_per_task=2, seed=42,
            sampling='random', context_mode='common', common_context=4096,
            answer_stride=128, dtype='bfloat16',
        )
        baseline = {
            'version': 2, 'settings': pilot.settings(baseline_args),
            'models': specs, 'total_celerity_repositories': 3,
            'excluded_models': [], 'longbench_revision': 'frozen-dataset',
            'sample_indices': {'narrativeqa': [0, 2]},
            'sample_id_digest': {'narrativeqa': cases_digest},
        }
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / '4k_manifest.json'
            dst = Path(td) / '8k_manifest.json'
            pilot.atomic_json(src, baseline)
            args = SimpleNamespace(**vars(baseline_args))
            args.common_context = 8192
            args.source_manifest = src
            args.min_trained_context = 8192
            args.max_run_models = 1  # A pilot limit must not change frozen settings.
            frozen_pilot_settings = pilot.settings(args)
            args.max_run_models = 0
            self.assertEqual(frozen_pilot_settings, pilot.settings(args))
            with patch.object(pilot, 'discover', side_effect=AssertionError('must not rediscover HF')):
                doc = pilot.freeze_manifest(dst, None, args)
            self.assertEqual([m['trained_context_tokens'] for m in doc['models']], [8192, 16384])
            self.assertEqual([m['revision'] for m in doc['models']], [str(8)*40, str(16)*40])
            self.assertEqual(doc['sample_indices'], baseline['sample_indices'])
            self.assertEqual(doc['sample_id_digest'], baseline['sample_id_digest'])
            self.assertEqual(doc['longbench_revision'], baseline['longbench_revision'])
            self.assertEqual(len(doc['models']), 2)
            self.assertEqual(pilot.freeze_manifest(dst, None, args)['models'], doc['models'])
            with patch.object(pilot, 'load_dataset', return_value=FakeDataset(rows)):
                cases = pilot.select_cases(doc, dst)
            self.assertEqual([c['example_index'] for c in cases], [0, 2])
            self.assertEqual([c['answer'] for c in cases], ['first', 'third'])
            baseline['extra'] = 'tampered'
            pilot.atomic_json(src, baseline)
            with self.assertRaisesRegex(ValueError, 'different settings'):
                pilot.freeze_manifest(dst, None, args)

    def test_frozen_8k_experiment_rejects_changed_scoring_policy(self):
        baseline_args = SimpleNamespace(
            order='recent', model_offset=0, limit_models=0,
            tasks=['narrativeqa'], examples_per_task=1, seed=42,
            sampling='random', context_mode='common', common_context=4096,
            answer_stride=128, dtype='bfloat16',
        )
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / 'source.json'
            doc = dict(
                version=2, settings=pilot.settings(baseline_args),
                models=[dict(model='adpretko/celerity-504m-8k-ad0p0',
                             revision='f'*40, trained_context_tokens=8192)],
                total_celerity_repositories=1, excluded_models=[],
                longbench_revision='frozen',
                sample_indices={'narrativeqa': [0]},
                sample_id_digest={'narrativeqa': 'a'*64},
            )
            pilot.atomic_json(src, doc)
            args = SimpleNamespace(**vars(baseline_args))
            args.source_manifest = src
            args.min_trained_context = 8192
            args.common_context = 8192
            args.dtype = 'float16'
            with self.assertRaisesRegex(ValueError, 'dtype'):
                pilot.freeze_manifest(Path(td) / 'new.json', None, args)


'''


def patch_driver(source: str) -> str:
    source = replace_section(source, 'def settings(args) -> dict:\n',
                             'def effective_context(spec: dict, args) -> int:\n',
                             MANIFEST_CODE)
    source = replace_one(source,
        '        if task not in manifest["sample_indices"]:\n            valid = [i for i, row in enumerate(ds)\n',
        '        if task not in manifest["sample_indices"]:\n'
        '            if cfg.get("source_manifest_sha256"):\n'
        '                raise RuntimeError(f"Missing frozen sample indices for {task}; refusing to resample")\n'
        '            valid = [i for i, row in enumerate(ds)\n')
    source = replace_one(source,
        '    p.add_argument("--common-context", type=int, default=4096)\n',
        '    p.add_argument("--common-context", type=int, default=4096)\n'
        '    p.add_argument("--source-manifest", type=Path, default=None,\n'
        '                   help="Copy frozen model revisions and exact examples from a 4K manifest")\n'
        '    p.add_argument("--min-trained-context", type=int, default=0,\n'
        '                   help="With --source-manifest, exclude models trained below this length")\n'
        '    p.add_argument("--max-run-models", type=int, default=0,\n'
        '                   help="Maximum new model attempts this invocation; 0 = all (safe pilots)")\n')
    source = replace_one(source,
        '    if args.device.startswith("cuda") and not torch.cuda.is_available() and not args.list_only:\n',
        '    if args.min_trained_context < 0 or args.max_run_models < 0:\n'
        '        p.error("--min-trained-context and --max-run-models must be >= 0")\n'
        '    if args.source_manifest is not None:\n'
        '        if not args.source_manifest.is_file():\n'
        '            p.error(f"Source manifest not found: {args.source_manifest}")\n'
        '        if args.context_mode != "common" or args.min_trained_context < args.common_context:\n'
        '            p.error("Sourced runs need common context and --min-trained-context >= --common-context")\n'
        '        if args.model_offset != 0 or args.limit_models != 0:\n'
        '            p.error("Sourced runs need --limit-models 0 and --model-offset 0; use --max-run-models for a pilot")\n'
        '    elif args.min_trained_context:\n'
        '        p.error("--min-trained-context requires --source-manifest")\n'
        '    if args.device.startswith("cuda") and not torch.cuda.is_available() and not args.list_only:\n')
    source = replace_one(source,
        '    failures = 0\n    for i, spec in enumerate(models, 1):\n',
        '    failures = 0\n    attempted = 0\n    for i, spec in enumerate(models, 1):\n')
    source = replace_one(source,
        '            print(f"[{i}/{len(models)}] SKIP (already successful): {name}")\n'
        '            continue\n        print(f"[{i}/{len(models)}] START {name}", flush=True)\n',
        '            print(f"[{i}/{len(models)}] SKIP (already successful): {name}")\n'
        '            continue\n'
        '        if args.max_run_models and attempted >= args.max_run_models:\n'
        '            print(f"Reached --max-run-models {args.max_run_models}; resume without this flag")\n'
        '            break\n'
        '        attempted += 1\n'
        '        print(f"[{i}/{len(models)}] START {name}", flush=True)\n')
    source = replace_one(source,
        '  python benchmark_celerity_longbench_v2.py --context-mode native --output-dir results/pilot10_native\n',
        '  python benchmark_celerity_longbench_v2.py --context-mode native --output-dir results/pilot10_native\n'
        '  python benchmark_celerity_longbench_v2.py --source-manifest experiments/celerity_all_4k_bf16/manifest.json --min-trained-context 8192 --context-mode common --common-context 8192 --dtype bfloat16 --limit-models 0 --output-dir results/celerity_all_8k_bf16 --list-only\n')
    return source


def patch_tests(source: str) -> str:
    return replace_one(source, "\n\nif __name__ == '__main__':\n",
                       "\n\n" + NEW_TESTS + "if __name__ == '__main__':\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true',
                        help='Verify that the patch applies and both files compile; do not write')
    args = parser.parse_args()
    if run('git', 'rev-parse', 'HEAD') != BASE_COMMIT:
        raise RuntimeError('Checkout is not at expected commit 4558a30; refusing to patch')
    for target in (DRIVER, TESTS):
        if run('git', 'status', '--porcelain', '--', str(target)):
            raise RuntimeError(f'{target} has uncommitted changes; refusing to overwrite')
        if not target.is_file():
            raise FileNotFoundError(target)
    old_driver = DRIVER.read_text(encoding='utf-8')
    old_tests = TESTS.read_text(encoding='utf-8')
    new_driver = patch_driver(old_driver)
    new_tests = patch_tests(old_tests)
    compile(new_driver, str(DRIVER), 'exec')
    compile(new_tests, str(TESTS), 'exec')
    if args.check_only:
        print('PASS: patch anchors and Python syntax verified; no files modified')
        return 0
    DRIVER.write_text(new_driver, encoding='utf-8')
    TESTS.write_text(new_tests, encoding='utf-8')
    print('Patched benchmark_celerity_longbench_v2.py and tests/test_benchmark_celerity_longbench_v2.py')
    print('Run: python -m unittest discover -s tests -v')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)
