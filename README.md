# LongBench Perplexity Harness

This repository provides a small LongBench evaluation harness centered on **token-level negative log-likelihood (NLL) and perplexity (PPL)** for Hugging Face causal language models.

The main goal is to use perplexity-related measurements as signals for **workload / task complexity**, rather than to reproduce the official LongBench leaderboard.

The repository currently has three scripts:

| Script | Purpose |
|---|---|
| `longbench_perplexity.py` | Measure perplexity on LongBench text or gold answers |
| `generate_longbench.py` | Generate answers from LongBench prompts |
| `generated_perplexity.py` | Measure self-perplexity of the exact generated answer tokens |

This is **not** the official LongBench generation evaluator. Official LongBench evaluation generates answers and scores correctness against references using task-specific metrics. This harness instead focuses on model likelihood, NLL, and perplexity.

---

## Measurements

Perplexity is calculated as:

```text
perplexity = exp(avg_nll)
```

where `avg_nll` is the total negative log-likelihood divided by the total number of scored tokens.

Lower perplexity means the model assigned higher probability to the scored sequence.

The current workflow supports three related measurements.

### 1. Full-text perplexity

`longbench_perplexity.py --mode full`

Scores:

```text
context + input
```

This is a diagnostic language-model likelihood measurement over the LongBench text itself. Gold answers are not included.

For long inputs, the script uses sliding-window scoring so tokens can be evaluated beyond a single model context window.

### 2. Gold-answer conditional perplexity

`longbench_perplexity.py --mode answer`

Scores:

```text
P(gold answer | LongBench prompt)
```

The script:

1. Builds the task-specific LongBench prompt.
2. Tokenizes the prompt and gold answer separately.
3. Masks prompt tokens with `-100`.
4. Scores only the gold-answer tokens.
5. Uses stride-based answer chunking when the answer is longer than one scoring window.
6. Sums NLL and token counts across chunks before calculating perplexity.

The default `--reference-policy first` scores the same fixed `answers[0]` for every model.
`--reference-policy all` scores every **distinct** supplied reference. The ordinary
`avg_nll` and `perplexity` remain token-weighted; `example_macro_avg_nll` and
`example_macro_perplexity` first average normalized NLL across references within
each example, then average across examples. The latter are supplementary
reference-formulation diagnostics, **not** measures of generated answer accuracy.

When `--output-jsonl results/example.jsonl` is supplied, detailed per-example
records are also written to `results/example_examples.jsonl`. Use
`--examples-output-jsonl PATH` to override this path without changing the
backward-compatible task-summary JSONL output.


For raw/base models, the prompt is used directly.

For instruction/chat models, answer mode can use the tokenizer's native chat template:

```bash
--use-chat-template
```

This affects **answer mode only**. `full` mode remains a measurement of the LongBench text itself and is not wrapped in a chat template.

### 3. Generated-answer self-perplexity

The generation workflow is:

```text
official LongBench prompt
        ↓
generate_longbench.py
        ↓
exact generated token IDs
        ↓
generated_perplexity.py
        ↓
P(generated answer | prompt)
```

`generate_longbench.py` saves the exact token IDs produced by `model.generate()`.

`generated_perplexity.py` then scores those exact saved IDs directly. It does **not** decode the generated answer and tokenize it again.

This makes generated-answer scoring analogous to gold-answer scoring while avoiding an unnecessary decode → re-tokenize round trip.

Generated self-perplexity should not be interpreted as answer correctness. Greedy generation intentionally selects high-probability tokens, and repetitive or degenerate generations can therefore have very low self-perplexity even when the answer is wrong.

For example, a model may repeatedly generate a highly predictable `Question:` / `Answer:` pattern and receive very low self-PPL despite failing the task.

---

## Prompt formatting

The harness supports two intended prompt paths.

### Raw prompt path

Appropriate for base models such as:

```text
gpt2
facebook/opt-2.7b
```

Do not pass `--use-chat-template`.

### Chat-template path

Appropriate for instruction/chat models such as:

```text
microsoft/Phi-3-mini-4k-instruct
```

Pass:

```bash
--use-chat-template
```

to `longbench_perplexity.py` in answer mode and to `generate_longbench.py`.

`generate_longbench.py` records whether the chat template was used in every JSONL row:

```json
"use_chat_template": true
```

`generated_perplexity.py` reads this setting automatically from the generation file and reconstructs the same prompt format during self-PPL scoring. There is therefore no separate `--use-chat-template` flag for `generated_perplexity.py`.

Older generation files without this field are treated as raw-prompt generations.

---

## Rolling-context generation

`generate_longbench.py` supports answers that may be longer than a single model context window.

Two separate limits are used:

```text
max_length
    model context available during one generation call

max_new_tokens
    maximum total answer length
```

These are intentionally not coupled.

Generation proceeds in chunks controlled by:

```bash
--generation-chunk-size
```

For example:

```text
max_length = 1024
generation_chunk_size = 64
```

Each generation step reserves room for its next output chunk. It retains the **most
recent generated-answer tokens**, then splits the remaining input budget between
the **beginning and end of the original prompt** (LongBench-style middle
truncation). Unlike the old tail-only policy, this usually preserves both the
task instructions and question. If an answer grows longer than a window,
its recent generated tokens take priority and the prompt can disappear.

Conceptually:

```text
[prompt head][prompt tail][recent generated tokens][room for next chunk]
<--------------------------- max_length ----------------------------->
```

Each output row now includes a compact `generation_trace` containing the
prompt-token SHA256 and, per chunk, the exact input token checksum,
original-prompt head/tail spans, retained generated-history span, and
new-token indices. These spans reconstruct the *actual input IDs* passed
to `model.generate()` without storing those long IDs repeatedly. Step 3 will
use `reconstruct_generation_context` and verify the checksum. The original
four-value return from `generate_answer()` remains supported.

`generated_perplexity.py` now accepts the Step 2 trace format and verifies
exact input contexts before scoring. Older untraced files are still accepted,
but they are labeled `legacy_untraced_tail_window` and should not be mixed
with newly generated outputs.

This means `max_new_tokens` is a stopping cap, not a requirement that the entire answer fit alongside the original prompt in one context window.

---

## Generated-token storage

Generation output contains both:

```json
"generated_token_ids": [...]
```

and:

```json
"generated_content_token_ids": [...]
```

`generated_token_ids` preserves the exact sequence returned by generation, including a trailing EOS token when one is generated.

`generated_content_token_ids` removes trailing EOS/pad tokens so generated-answer PPL matches the gold-answer PPL convention, where EOS is not automatically appended to the reference answer.

`generated_perplexity.py` scores `generated_content_token_ids` directly.

---

## Generated-answer scoring windows

**New traced generations (Step 2 format):** for each recorded chunk, the
scorer verifies the original tokenized prompt SHA256, reconstructs the exact
head/tail prompt plus preceding generated-history input IDs, and verifies the
chunk-context checksum. It then teacher-forces that chunk's original generated
content tokens after its verified context; preceding context is masked from
loss. Every content token is scored once, including across multiple chunks.
Trailing EOS/padding is not scored. The saved chunk boundaries are authoritative:
do not supply `--stride` for traced files or change their `--max-length`.

**Older untraced generations:** retain the previous tail-only sliding-window
scoring behavior for backward compatibility. The output's
`scoring_context_policy` field distinguishes these from verified traces.
Numerical values from these two policies should not be pooled as though they
used the same conditioning contexts.

Teacher-forced full forward calls reproduce the same token contexts as
`model.generate()`, although small floating-point differences versus the
incremental KV-cache generation kernels are possible.

The scorer accumulates:

```text
total_nll = Σ chunk_nll
total_tokens = Σ chunk_tokens

avg_nll = total_nll / total_tokens
perplexity = exp(avg_nll)
```

Chunk perplexities are **not** averaged directly.

For traced files, the original generation chunks determine scoring spans.
For legacy files, `--stride` (or the recorded `generation_chunk_size`) continues
to define tail-window scoring spans.

---

## Environment

The LongBench Hugging Face dataset currently works best with an older `datasets` stack because it uses a dataset loading script.

### CPU environment

```bash
conda create -n lb_ppl python=3.11 -y
conda activate lb_ppl

python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu

python -m pip install \
  "transformers==4.44.2" \
  "datasets==2.19.2" \
  "huggingface_hub==0.25.2" \
  "accelerate==0.34.2" \
  "tqdm"
```

### ROCm environment

```bash
conda create -n lb_ppl_rocm python=3.11 -y
conda activate lb_ppl_rocm

python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/rocm6.3

python -m pip install \
  "transformers==4.44.2" \
  "datasets==2.19.2" \
  "huggingface_hub==0.25.2" \
  "accelerate==0.34.2" \
  "tqdm"
```

Verify package versions:

```bash
python - <<'PY'
import sys
import torch
import transformers
import datasets
import huggingface_hub

print("python:", sys.version)
print("torch:", torch.__version__)
print("torch.version.hip:", torch.version.hip)
print("cuda/rocm available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device 0:", torch.cuda.get_device_name(0))
print("transformers:", transformers.__version__)
print("datasets:", datasets.__version__)
print("huggingface_hub:", huggingface_hub.__version__)
PY
```

### Tested CPU environment

| Package | Version |
|---|---|
| Python | 3.11.16 |
| torch | 2.13.0+cpu |
| transformers | 4.44.2 |
| datasets | 2.19.2 |
| huggingface_hub | 0.25.2 |

### Tested ROCm environment

| Component | Version / value |
|---|---|
| GPU | Radeon RX 7900 XT |
| GPU VRAM | 20 GB |
| ROCm | 6.3.0 |
| torch | 2.9.1+rocm6.3 |
| torch.version.hip | 6.3.42134 |
| device argument | `cuda` |

Even on AMD ROCm, PyTorch uses the `cuda` device string:

```bash
--device cuda
```

---

## Basic syntax checks

```bash
python -m py_compile longbench_perplexity.py
python -m py_compile generate_longbench.py
python -m py_compile generated_perplexity.py
```

---

## Example: raw/base-model workflow

The following three commands exercise the intended raw-prompt path with GPT-2.

### 1. Gold-answer PPL

```bash
python longbench_perplexity.py \
  --model gpt2 \
  --tasks narrativeqa \
  --mode answer \
  --limit 3 \
  --max-length 1024 \
  --stride 512 \
  --device cuda \
  --dtype float16 \
  --output-jsonl results/gpt2_narrativeqa_raw_gold_ppl_limit3.jsonl
```

Expected:

```text
use_chat_template: False
```

### 2. Generate answers

```bash
python generate_longbench.py \
  --model gpt2 \
  --tasks narrativeqa \
  --limit 3 \
  --max-length 1024 \
  --max-new-tokens 128 \
  --generation-chunk-size 64 \
  --device cuda \
  --dtype float16 \
  --output-jsonl results/gpt2_narrativeqa_raw_generated_limit3.jsonl
```

### 3. Generated-answer self-PPL

```bash
python generated_perplexity.py \
  --model gpt2 \
  --input-jsonl results/gpt2_narrativeqa_raw_generated_limit3.jsonl \
  --max-length 1024 \
  --device cuda \
  --dtype float16 \
  --output-jsonl results/gpt2_narrativeqa_raw_generated_ppl_limit3.jsonl
```

Expected:

```text
stride: 64
use_chat_template: False
```

---

## Example: instruction/chat-model workflow

The following three commands exercise the intended chat-template path with Phi-3-Instruct.

### 1. Gold-answer PPL

```bash
python longbench_perplexity.py \
  --model microsoft/Phi-3-mini-4k-instruct \
  --tasks narrativeqa \
  --mode answer \
  --limit 3 \
  --max-length 1024 \
  --stride 512 \
  --device cuda \
  --dtype float16 \
  --trust-remote-code \
  --use-chat-template \
  --output-jsonl results/phi3mini_narrativeqa_chat_gold_ppl_limit3.jsonl
```

Expected:

```text
use_chat_template: True
```

### 2. Generate answers

```bash
python generate_longbench.py \
  --model microsoft/Phi-3-mini-4k-instruct \
  --tasks narrativeqa \
  --limit 3 \
  --max-length 1024 \
  --max-new-tokens 128 \
  --generation-chunk-size 64 \
  --device cuda \
  --dtype float16 \
  --trust-remote-code \
  --use-chat-template \
  --output-jsonl results/phi3mini_narrativeqa_chat_generated_limit3.jsonl
```

### 3. Generated-answer self-PPL

```bash
python generated_perplexity.py \
  --model microsoft/Phi-3-mini-4k-instruct \
  --input-jsonl results/phi3mini_narrativeqa_chat_generated_limit3.jsonl \
  --max-length 1024 \
  --device cuda \
  --dtype float16 \
  --trust-remote-code \
  --output-jsonl results/phi3mini_narrativeqa_chat_generated_ppl_limit3.jsonl
```

Expected:

```text
stride: 64
use_chat_template: True
```

The chat-template setting is inferred from the generation file.

---

## Existing full-mode example

```bash
mkdir -p results

python longbench_perplexity.py \
  --model gpt2 \
  --tasks narrativeqa,qasper,multifieldqa_en \
  --limit 20 \
  --mode full \
  --max-length 1024 \
  --stride 512 \
  --device cpu \
  --dtype float32 \
  --output-jsonl results/gpt2_qa3_limit20_full.jsonl
```

---

## Tested results

### GPT-2 full mode

| Task | Examples | Tokens scored | Avg NLL | Perplexity |
|---|---:|---:|---:|---:|
| narrativeqa | 20 | 544,742 | 4.670750 | 106.7778 |
| qasper | 20 | 122,071 | 3.172258 | 23.8613 |
| multifieldqa_en | 20 | 164,685 | 3.107136 | 22.3569 |
| OVERALL | 60 | 831,498 | 4.141072 | 62.8702 |

### GPT-2 gold-answer mode

| Task | Examples | Tokens scored | Avg NLL | Perplexity |
|---|---:|---:|---:|---:|
| narrativeqa | 100 | 599 | 4.852336 | 128.0392 |
| qasper | 100 | 1,850 | 3.776111 | 43.6460 |
| multifieldqa_en | 100 | 1,449 | 3.111908 | 22.4639 |
| OVERALL | 300 | 3,898 | 3.694589 | 40.2290 |

### End-to-end ROCm smoke tests

These tests use the same first three NarrativeQA examples.

| Model | Prompt path | Measurement | Examples | Tokens scored | Avg NLL | Perplexity |
|---|---|---|---:|---:|---:|---:|
| `gpt2` | raw | gold-answer PPL | 3 | 27 | 3.670232 | 39.2610 |
| `gpt2` | raw | generated self-PPL | 3 | 384 | 0.390641 | 1.4779 |
| `facebook/opt-2.7b` | raw | gold-answer PPL | 3 | 27 | 2.955859 | 19.2182 |
| `facebook/opt-2.7b` | raw | generated self-PPL | 3 | 384 | 0.254890 | 1.2903 |
| `microsoft/Phi-3-mini-4k-instruct` | chat template | gold-answer PPL | 3 | 28 | 2.222677 | 9.2320 |
| `microsoft/Phi-3-mini-4k-instruct` | chat template | generated self-PPL | 3 | 47 | 0.746018 | 2.1086 |

The GPT-2 raw path and Phi-3 chat-template path were both smoke-tested across all three scripts after chat-template support was added.

---

## Output formats

### `longbench_perplexity.py`

When `--output-jsonl` is supplied, the script writes one JSON object per task plus one `OVERALL` row.

Example:

```json
{"task": "narrativeqa", "mode": "answer", "use_chat_template": false, "examples": 3, "tokens_scored": 27, "avg_nll": 3.670231571903935, "perplexity": 39.26099655141135}
```

### `generate_longbench.py`

Writes one row per generated example.

Important fields include:

```json
{
  "task": "narrativeqa",
  "example_index": 0,
  "model": "gpt2",
  "prediction": "...",
  "generated_token_ids": [],
  "generated_content_token_ids": [],
  "generated_tokens": 128,
  "generated_tokens_raw": 128,
  "prompt_tokens": 35112,
  "max_length": 1024,
  "max_new_tokens": 128,
  "generation_chunk_size": 64,
  "use_chat_template": false,
  "answers": ["..."],
  "length": 22692
}
```

### `generated_perplexity.py`

Writes per-example rows followed by per-task and overall summaries.

Example per-example row:

```json
{
  "row_type": "example",
  "task": "narrativeqa",
  "example_index": 0,
  "model": "gpt2",
  "generated_tokens": 128,
  "tokens_scored": 128,
  "nll_sum": 79.77105712890625,
  "avg_nll": 0.6232113838195801,
  "perplexity": 1.8649073691003633
}
```

Example task summary:

```json
{
  "row_type": "task_summary",
  "task": "narrativeqa",
  "model": "gpt2",
  "examples": 3,
  "tokens_scored": 384,
  "avg_nll": 0.3906408653904994,
  "perplexity": 1.4779276431249224
}
```

---

## Warnings seen during testing

### Long tokenizer sequence warning

LongBench examples can be much longer than a model's nominal context window, so the tokenizer may print warnings such as:

```text
Token indices sequence length is longer than the specified maximum sequence length
```

This is expected in the current harness. The full logical prompt/text is tokenized first; model calls are then restricted to the configured scoring or rolling-generation window.

### AMD attention warnings

On the Radeon RX 7900 XT, PyTorch may report that flash-efficient or memory-efficient attention is experimental.

These warnings did not block the tested runs.

### Phi-3 flash-attention warning

Phi-3 may print:

```text
flash-attention package not found
```

and:

```text
You are not running the flash-attention implementation, expect numerical differences.
```

This was a performance / implementation warning during the current tests and did not prevent generation or PPL scoring.

---

## Interpretation notes

The three measurements answer different questions.

### Full-text PPL

```text
How surprising is the LongBench context/input text to the model?
```

### Gold-answer conditional PPL

```text
How much probability does the model assign to the provided correct/reference answer given the prompt?
```

This is currently the most directly task-oriented perplexity measurement in the harness.

### Generated-answer self-PPL

```text
How predictable is the sequence the model itself chose to generate?
```

This can be useful as a secondary complexity/confidence diagnostic, but it is not a correctness score.

A very low generated self-PPL can occur when a model confidently generates an incorrect or repetitive answer. Greedy decoding particularly biases this measurement toward lower perplexity because each token is selected from the model's own high-probability distribution.

Therefore, gold-answer PPL and generated self-PPL should be compared carefully and should not be interpreted as interchangeable measures.

---

## Current scope

The harness has been smoke-tested on CPU and on a Radeon RX 7900 XT using ROCm PyTorch.

Tested model families include:

```text
gpt2
facebook/opt-2.7b
microsoft/Phi-3-mini-4k-instruct
```

All three ran successfully on ROCm with `--dtype float16`.

The intended model range for the current project is approximately 271M–3B parameters, with Phi-3-mini-4k-instruct also used as a larger instruction-model validation case.

Larger 7B+ models may require more careful memory handling and are outside the current scope.
