# LongBench verification and Celerity pilot (after Step 3)

Base Git commit: `2ceaa0a` on `fix/longbench-step3`.

**Status:** All 28 existing regression tests passed on the user's machine; GPT-2 gold, generation, and traced self-PPL have been checked. Phi-3 and a real Celerity HF checkpoint still need end-to-end verification. The audit/driver unit tests in this package use synthetic fixtures and do not download model weights.

## Files

- `verify_longbench_results.py`: independently checks that summary PPL/NLL and token counts match underlying example/reference records; no third-party dependencies.
- `benchmark_celerity_longbench_v2.py`: batches Celerity gold-answer evaluation by calling **the verified** `longbench_perplexity.answer_conditional_ppl`; no separate scoring code or prompt truncation logic. Initially choose up to ten `adpretko/celerity-*` models, freeze model & LongBench revisions and sample indices, record timing and VRAM.
- `test_verify_longbench_results.py`: two offline audit tests.
- `test_benchmark_celerity_longbench_v2.py`: five offline driver tests.

## Apply in a new branch

From the user's existing checkout after Step 3:

```bash
cd ~/Perplexity_LongBench
git switch -c feature/celerity-ppl-verification-batch
cp ~/Downloads/longbench_pilot/verify_longbench_results.py .
cp ~/Downloads/longbench_pilot/benchmark_celerity_longbench_v2.py .
mkdir -p tests
cp ~/Downloads/longbench_pilot/tests/test_verify_longbench_results.py tests/
cp ~/Downloads/longbench_pilot/tests/test_benchmark_celerity_longbench_v2.py tests/
python -m py_compile verify_longbench_results.py benchmark_celerity_longbench_v2.py
python -m unittest discover -s tests -v
```

The additional batch tests temporarily stub external imports during module loading,
then restore the originals. They can coexist with the existing regression suite.
The tests use fake model and dataset objects; only the real Hyperstack smoke run
establishes Celerity checkpoint compatibility.

## Verify existing gold and traced generation outputs

```bash
python verify_longbench_results.py gold --summary results/gpt2_step1_first.jsonl
python verify_longbench_results.py gold --summary results/gpt2_step1_all.jsonl
python verify_longbench_results.py generated --results results/gpt2_step3_selfppl.jsonl
```

Expected three `PASS` messages and no discrepancy in example counts, reference sums, or aggregate PPL.

## Phi-3 real-model smoke test on Hyperstack (CUDA)

Phi-3 Mini is larger than GPT-2; run this on a suitable NVIDIA GPU, not the desktop CPU. In the project's Python environment, install the pinned dependencies from its README and a GPU-enabled PyTorch build appropriate for that machine. Verify `torch.cuda.is_available()` first.

```bash
python longbench_perplexity.py \
  --model microsoft/Phi-3-mini-4k-instruct \
  --tasks narrativeqa --mode answer --reference-policy first \
  --limit 3 --max-length 1024 --stride 512 \
  --trust-remote-code --use-chat-template \
  --device cuda --dtype float16 \
  --output-jsonl results/phi3_step4_gold.jsonl

python verify_longbench_results.py gold --summary results/phi3_step4_gold.jsonl

python generate_longbench.py \
  --model microsoft/Phi-3-mini-4k-instruct \
  --tasks narrativeqa --limit 1 --max-length 1024 \
  --max-new-tokens 6 --generation-chunk-size 3 \
  --trust-remote-code --use-chat-template \
  --device cuda --dtype float16 \
  --output-jsonl results/phi3_step4_generations.jsonl

python generated_perplexity.py \
  --model microsoft/Phi-3-mini-4k-instruct \
  --input-jsonl results/phi3_step4_generations.jsonl \
  --trust-remote-code --device cuda --dtype float16 \
  --output-jsonl results/phi3_step4_selfppl.jsonl

python verify_longbench_results.py generated --results results/phi3_step4_selfppl.jsonl
```

The historical Phi-3 three-example gold PPL was approximately 9.23, but minor numerical differences or differences in exact reference policy may alter that value. The generated self-PPL with only six tokens **is not comparable** to the historical 47-token, three-example result. Focus on completed computation, token counts, verified context policy, and finite NLL/PPL.

## Celerity metadata inspection and one-model compatibility test

On Hyperstack (requires Internet to Hugging Face):

```bash
python benchmark_celerity_longbench_v2.py \
  --list-only --output-dir results/celerity_pilot10_4k
```

Then use a **separate** output directory and matching settings for an initial one-model/one-example smoke test:

```bash
python benchmark_celerity_longbench_v2.py \
  --limit-models 1 --tasks narrativeqa --examples-per-task 1 \
  --device cuda --dtype float16 --context-mode common --common-context 4096 \
  --output-dir results/celerity_smoke1_4k
```

If that works, run the frozen pilot created by `--list-only`:

```bash
/usr/bin/time -p python benchmark_celerity_longbench_v2.py \
  --device cuda --dtype float16 --context-mode common --common-context 4096 \
  --output-dir results/celerity_pilot10_4k
```

Default is **10 models x 5 examples x 16 LongBench tasks = 800 gold-answer evaluations**. Separate `--context-mode native` results should use a different output directory; 16K runs can be much more memory/time intensive.

The standard scorer computes full `[sequence, vocabulary]` logits. The Celerity vocab is approximately 128K, so a 4K FP32 logits tensor alone can take about 2 GiB, before other model memory. First measure a single model on your actual GPU; if memory or runtime is excessive, verify a logits-tail optimization against the canonical scorer before enabling it for the full evaluation. **Do not silently substitute the earlier standalone benchmark's independent score function**, as that violates the shared-scoring requirement.

## Next research quality checks

- Confirm all models use the same tokenizer (the driver records a digest).
- Inspect per-task results and truncation rates; do not compare different evaluation contexts as if equivalent.
- For final experimental inference, freeze stratified task/length samples (LongBench-E can help) and use paired Celerity NLL differences, rather than assuming an 80-example pilot has sufficient statistical power.
- For true generated-answer task accuracy, use LongBench's task-specific evaluation metrics; self-PPL alone does not establish correctness.
