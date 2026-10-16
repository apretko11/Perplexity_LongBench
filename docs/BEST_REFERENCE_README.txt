CELERITY LONGBENCH BEST-REFERENCE V3 — LOCAL SETUP
=================================================

This package is designed for the repository at commit 0f395dd (or its descendants
that retain the original anchor code), before any best-policy modifications.

It leaves benchmark_celerity_longbench_v2.py and all archived results unchanged.
It patches only longbench_perplexity.py and verify_longbench_results.py,
creates benchmark_celerity_longbench_v3.py, and supplies a new offline test suite.

1. In ~/Perplexity_LongBench, after extracting these files:

    python apply_best_ref_v3.py
    python -m unittest discover -s tests -q

   Expected (assuming baseline had 40 tests): 47 tests, OK.
   All unit tests are CPU-only and need no Hugging Face model download.
   If the patcher reports a missing anchor, stop and share the error;
   it intentionally refuses to patch unexpected source revisions.

2. Validate model-selection manifests BEFORE renting the GPU, for example:

    python benchmark_celerity_longbench_v3.py \
      --source-manifest experiments/celerity_all_4k_a6000_bf16/manifest.json \
      --min-trained-context 8192 --context-mode common \
      --common-context 8192 --limit-models 0 --dtype bfloat16 \
      --reference-policy best --output-dir results/celerity_all_8k_bestref_bf16 \
      --list-only

   Expected: Frozen 168 of 189 models, all eval=8192.

3. Then commit code and push to GitHub:

    git add longbench_perplexity.py verify_longbench_results.py \
      benchmark_celerity_longbench_v3.py apply_best_ref_v3.py \
      verify_best_reference_v3.py run_best_reference_campaign.sh \
      tests/test_best_reference_v3.py
    git commit -m "Add best-reference LongBench scoring and frozen v3 batch driver"
    git push origin main

4. On Hyperstack: clone/pull main, install validated dependencies,
   activate venv, then run INSIDE TMUX:

    bash run_best_reference_campaign.sh

   This executes full 4K (189 models), 8K (168) and 16K (21) in series.
   It does NOT use a one-model pilot. Each model is resumable.
   On success it audits every model's reference choices and aggregates.

REFERENCE SEMANTICS
-------------------
  first: original first reference (unchanged)
  all:   score/aggregate all distinct references (unchanged)
  best:  score all distinct nonempty references, select the one with
         minimum mean NLL/token per EXAMPLE and MODEL, and aggregate ONLY
         the selected NLL and number of tokens. Ties choose the earliest
         reference. All candidate scores and the selected index are saved.

  Best-reference is an oracle-style conditional perplexity, NOT an
  accuracy metric. Different models/contexts may select different
  reference texts and token counts. Do not treat the resulting NLLs as
  paired scores for an identical target unless the selected reference agrees.

  Frozen model SHAs, dataset revision and indices come from the archived
  A6000 4K first-reference manifest. No Hugging Face rediscovery occurs.

  Results are written only under results/celerity_all_*_bestref_bf16,
  not into the archived first-reference experiment directories.

IMPORTANT
---------
  Multi-reference scoring may require substantially more time than first-
  reference scoring; the 16K 504M models have tight VRAM headroom.
  The run sets PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True.
  Failures leave successful model results intact for resumption.
  No code has been run on the actual A6000 by this patch package;
  run offline regression tests and check --list-only before GPU execution.
