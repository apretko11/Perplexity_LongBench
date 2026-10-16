#!/usr/bin/env python3
"""One-time, guarded extension of the Celerity LongBench repository.

Adds `best` to the standalone gold-answer scorer and verifier, and creates
benchmark_celerity_longbench_v3.py from the exact checked-in v2 runner.
The original v2 runner, archived experiments, and analysis code are unchanged.

Run from ~/Perplexity_LongBench:
    python apply_best_ref_v3.py
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def one(src: str, before: str, after: str, label: str) -> str:
    count = src.count(before)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly 1 matching anchor, got {count}; nothing written")
    return src.replace(before, after, 1)


def section(src: str, begin: str, end: str, replacement: str, label: str) -> str:
    assert src.count(begin) == 1 and src.count(end) == 1, label
    i, j = src.index(begin), src.index(end)
    if j <= i:
        raise RuntimeError(f"Invalid section order: {label}")
    return src[:i] + replacement + src[j:]


def main() -> None:
    scorer_path = ROOT / "longbench_perplexity.py"
    verifier_path = ROOT / "verify_longbench_results.py"
    runner_v2 = ROOT / "benchmark_celerity_longbench_v2.py"
    runner_v3 = ROOT / "benchmark_celerity_longbench_v3.py"
    for p in (scorer_path, verifier_path, runner_v2):
        if not p.is_file():
            raise RuntimeError(f"Missing expected repository file: {p}")
    if runner_v3.exists():
        raise RuntimeError(f"{runner_v3} already exists; refusing to overwrite")

    s = scorer_path.read_text(encoding="utf-8")
    s = one(s, 'choices=["first", "all"],', 'choices=["first", "all", "best"],', "CLI policy")
    s = one(s,
        'help="Answer mode: score a fixed first reference (default) or every distinct reference.",',
        'help="Answer mode: first reference (default), all references, or easiest reference by mean NLL/token.",',
        "CLI description")
    s = one(s,
        '            reference_results = []\n            original_prompt_tokens = None\n',
        '            reference_results = []\n            selected_reference_index = None\n            original_prompt_tokens = None\n',
        "reference trace selection initializer")
    s = one(s,
        '''                nll = sum(r["nll_sum"] for r in reference_results)
                n_tokens = sum(r["tokens_scored"] for r in reference_results)
                # Equal weight for each distinct reference *within this example*.
                # Unlike pooled token NLL, this is not ordinary token-weighted PPL.
                example_mean_nll = sum(
                    r["avg_nll"] for r in reference_results
                ) / len(reference_results)
''',
        '''                if args.reference_policy == "best":
                    # Reference with minimum mean NLL/token (not minimum total
                    # NLL). Exact ties select the earlier original reference.
                    chosen = min(reference_results, key=lambda r: r["avg_nll"])
                    nll = chosen["nll_sum"]
                    n_tokens = chosen["tokens_scored"]
                    example_mean_nll = chosen["avg_nll"]
                    selected_reference_index = chosen["reference_index"]
                else:
                    # Preserve legacy 'first' and 'all' aggregation exactly.
                    nll = sum(r["nll_sum"] for r in reference_results)
                    n_tokens = sum(r["tokens_scored"] for r in reference_results)
                    example_mean_nll = sum(
                        r["avg_nll"] for r in reference_results
                    ) / len(reference_results)
                    if args.reference_policy == "first":
                        selected_reference_index = reference_results[0]["reference_index"]
''',
        "score accumulation policy")
    s = one(s,
        '                    "reference_count_scored": len(reference_results),\n',
        '                    "reference_count_scored": len(reference_results),\n'
        '                    "selected_reference_index": selected_reference_index,\n',
        "best reference index in CLI trace")

    v = verifier_path.read_text(encoding="utf-8")
    v = one(v,
        '''            close(r["nll_sum"], sum(float(x["nll_sum"]) for x in refs), str(key) + ".reference_nll")
            if int(r["tokens_scored"]) != sum(int(x["tokens_scored"]) for x in refs):
                raise ValueError(f"{key}: reference token count mismatch")
            ref_macro = sum(float(x["avg_nll"]) for x in refs) / len(refs)
            close(r["reference_macro_avg_nll"], ref_macro, str(key) + ".reference_macro")
''',
        '''            if r.get("reference_policy") == "best":
                chosen = min(refs, key=lambda x: float(x["avg_nll"]))
                expected_nll = float(chosen["nll_sum"])
                expected_tokens = int(chosen["tokens_scored"])
                ref_macro = float(chosen["avg_nll"])
                if r.get("selected_reference_index") != chosen["reference_index"]:
                    raise ValueError(f"{key}: incorrect best reference index")
            else:
                expected_nll = sum(float(x["nll_sum"]) for x in refs)
                expected_tokens = sum(int(x["tokens_scored"]) for x in refs)
                ref_macro = sum(float(x["avg_nll"]) for x in refs) / len(refs)
            close(r["nll_sum"], expected_nll, str(key) + ".reference_nll")
            if int(r["tokens_scored"]) != expected_tokens:
                raise ValueError(f"{key}: reference token count mismatch")
            close(r["reference_macro_avg_nll"], ref_macro, str(key) + ".reference_macro")
''',
        "gold verifier best policy")

    r = runner_v2.read_text(encoding="utf-8")
    # Replace only the module docstring; preserve v2 loading and resume machinery.
    old_heading = '"""Reproducible Celerity LongBench gold-answer PPL pilot (HF/Hyperstack).'
    if r.count(old_heading) != 1:
        raise RuntimeError("v3 docstring: unexpected v2 source")
    doc_begin = r.index(old_heading)
    doc_end = r.index('"""', doc_begin + 3) + 3
    r = r[:doc_begin] + '"""Celerity LongBench BEST-reference gold-answer perplexity (HF/Hyperstack).\n\nBest-reference mode: score all distinct nonempty references independently\nusing longbench_perplexity.answer_conditional_ppl (unchanged), and select\nthe reference with minimum mean NLL per answer token for each example.\nStore all candidate scores and selected reference IDs.\n\nThis is teacher-forced, oracle best-reference PPL, NOT generation accuracy.\nRequires --source-manifest from the original 4K A6000 campaign to freeze\nmodel revisions, dataset revision and exact 80 LongBench examples.\nThe unchanged v2 script remains the canonical FIRST-reference evaluator.\n\nExample (show frozen 8K selection without evaluating models):\n  python benchmark_celerity_longbench_v3.py --source-manifest experiments/celerity_all_4k_a6000_bf16/manifest.json --min-trained-context 8192 --context-mode common --common-context 8192 --reference-policy best --limit-models 0 --dtype bfloat16 --output-dir results/celerity_all_8k_bestref_bf16 --list-only\n"""' + r[doc_end:]
    r = one(r,
        '        "reference_policy": "first", "use_chat_template": False,',
        '        "reference_policy": "best",\n'
        '        "best_reference_selection": "minimum_mean_nll_per_token_v1",\n'
        '        "use_chat_template": False,',
        "v3 scientific manifest settings")
    r = one(r,
        '"dtype", "dataset", "reference_policy", "use_chat_template",',
        '"dtype", "dataset", "use_chat_template",',
        "v3 source policy compatibility")
    r = one(r,
        '    original = source.get("settings", {})\n',
        '    original = source.get("settings", {})\n'
        '    if original.get("reference_policy") != "first":\n'
        '        raise ValueError("Source manifest must be original first-reference experiment")\n',
        "enforce first-ref source")
    r = one(r, '        "version": 2,\n        "created_utc":',
            '        "version": 3,\n        "created_utc":',
            "v3 sourced manifest schema")
    r = one(r, '        "version": 2, "created_utc":',
            '        "version": 3, "created_utc":',
            "v3 fallback manifest schema")
    r = one(r,
        '''                "answer": str(row["answers"][0]), "length_group": row.get("length"),
''',
        '''                "answer": str(row["answers"][0]),
                "answers": [str(a) for a in row["answers"]],
                "length_group": row.get("length"),
''',
        "preserve all references")

    best_score = '''def score_one(lm, tok, case: dict, context_limit: int, stride: int) -> dict | None:
    """Score all distinct references and select minimum NLL per answer token.

    Uses the identical canonical scorer from longbench_perplexity.py.
    LongBench samples, prompt truncation, causal masking, and stride are unchanged.
    """
    original_prompt_tokens = len(tokenize_prompt(
        tok, case["prompt"], use_chat_template=False,
    ))
    available = case["answers"]
    candidates = []
    seen_answers = set()
    for i, answer in enumerate(available):
        answer = str(answer)
        if not answer.strip() or answer in seen_answers:
            continue
        seen_answers.add(answer)
        nll, tokens = answer_conditional_ppl(
            model=lm, tokenizer=tok, prompt=case["prompt"], answer=answer,
            max_length=context_limit, stride=stride, use_chat_template=False,
        )
        if tokens < 1 or not math.isfinite(nll) or nll < 0:
            raise RuntimeError(f"Invalid reference NLL/tokens for {case['task']}#{case['example_index']} ref={i}")
        expected_tokens = len(tok(answer, add_special_tokens=False)["input_ids"])
        if expected_tokens != tokens:
            raise RuntimeError(f"Reference {i}: canonical scorer did not score all answer tokens")
        chunk = min(stride, context_limit - 1, tokens)
        retained = min(original_prompt_tokens, context_limit - chunk)
        candidates.append({
            "reference_index": i, "reference": answer,
            "nll_sum": nll, "tokens_scored": tokens,
            "avg_nll": nll / tokens, "perplexity": math.exp(nll / tokens),
            "prompt_tokens_retained_first_chunk": retained,
        })
    if not candidates:
        raise RuntimeError(f"No valid reference answers for {case['task']}#{case['example_index']}")
    selected = min(candidates, key=lambda r: r["avg_nll"])
    for r in candidates:
        r["selected"] = r is selected
    return {
        "task": case["task"], "example_index": case["example_index"],
        "example_id": case["example_id"], "length_group": case["length_group"],
        "reference_policy": "best",
        "reference_selection": "minimum_mean_nll_per_token_v1",
        "reference_index": selected["reference_index"],
        "reference_count_available": len(available),
        "reference_count_scored": len(candidates),
        "references": candidates,
        "answer_tokens": selected["tokens_scored"],
        "tokens_scored": selected["tokens_scored"],
        "nll_sum": selected["nll_sum"],
        "avg_nll": selected["avg_nll"],
        "perplexity": selected["perplexity"],
        "prompt_tokens_original": original_prompt_tokens,
        "prompt_tokens_retained_first_chunk": selected["prompt_tokens_retained_first_chunk"],
        "prompt_was_truncated_first_chunk": (
            original_prompt_tokens > selected["prompt_tokens_retained_first_chunk"]
        ),
        "evaluation_context_tokens": context_limit,
    }


'''
    r = section(r, 'def score_one(lm, tok, case: dict, context_limit: int, stride: int) -> dict | None:\n',
                'def base_row(spec, ctx: int) -> dict:\n', best_score, "v3 best reference scoring")
    r = one(r,
        '"evaluation_context_tokens": ctx, "reference_policy": "first", "scorer": SCORED_BY,',
        '"evaluation_context_tokens": ctx, "reference_policy": "best", "scorer": SCORED_BY,',
        "v3 model summary metadata")
    r = one(r,
        '''                f.write(json.dumps({"model": spec["model"], "task": task, **s,
                                    "avg_nll": avg, "perplexity": math.exp(avg)}) + "\\n")''',
        '''                f.write(json.dumps({"model": spec["model"], "task": task, **s,
                                    "reference_policy": "best",
                                    "avg_nll": avg, "perplexity": math.exp(avg)}) + "\\n")''',
        "v3 task metadata")
    r = one(r,
        'default=Path("results/celerity_pilot10_4k")',
        'default=Path("results/celerity_best_ref_4k_bf16")',
        "safe default output name")
    r = one(r,
        '    p.add_argument("--answer-stride", type=int, default=128)\n',
        '    p.add_argument("--reference-policy", choices=["best"], default="best",\n'
        '                   help="Select minimum mean NLL/token among distinct gold answers")\n'
        '    p.add_argument("--answer-stride", type=int, default=128)\n',
        "explicit v3 best flag")
    r = one(r,
        '''    args = parse_args()
    out = args.output_dir
''',
        '''    args = parse_args()
    if args.source_manifest is None:
        raise ValueError("v3 requires --source-manifest to reuse frozen model revisions and examples")
    if args.dtype != "bfloat16":
        raise ValueError("v3 campaign requires --dtype bfloat16 for direct comparability")
    if args.output_dir.resolve() == Path("experiments/celerity_all_4k_a6000_bf16").resolve():
        raise ValueError("Never write new results into the archived baseline")
    out = args.output_dir
''',
        "v3 main validation")
    # Compute every transformation before performing any write. The original
    # v2 driver and all experiment archives remain untouched.
    for name, contents in (
        ("longbench_perplexity.py", s),
        ("verify_longbench_results.py", v),
        ("benchmark_celerity_longbench_v3.py", r),
    ):
        compile(contents, name, "exec")
    scorer_path.write_text(s, encoding="utf-8")
    verifier_path.write_text(v, encoding="utf-8")
    runner_v3.write_text(r, encoding="utf-8")
    print("PATCHED: longbench_perplexity.py (--reference-policy best)")
    print("PATCHED: verify_longbench_results.py (best-policy auditing)")
    print("CREATED: benchmark_celerity_longbench_v3.py (v2 unchanged)")
    print("NEXT: python -m unittest discover -s tests -q")


if __name__ == "__main__":
    main()
