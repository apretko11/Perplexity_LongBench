# LongBench Perplexity Harness

This repository provides a small evaluation harness for measuring perplexity on LongBench examples using Hugging Face causal language models.

It supports two perplexity modes:

| Mode | What is scored | Purpose |
|---|---|---|
| full | context + input | Diagnostic language-model likelihood over LongBench text |
| answer | Gold answer tokens, conditioned on context + input | More task-like answer likelihood |

This is not the official LongBench generation evaluation.

Official LongBench evaluation generates answers and scores them against references. This script measures token-level negative log-likelihood and perplexity.

## What this script measures

Perplexity is calculated as:

    perplexity = exp(avg_nll)

where avg_nll is the average negative log-likelihood per scored token.

Lower perplexity means the model assigned higher probability to the scored text.

Important distinction:

- full mode is a diagnostic LM likelihood measurement over LongBench text.
- answer mode is a more task-like likelihood measurement over gold answers.
- neither mode is the official LongBench generation score.

## Environment

The LongBench Hugging Face dataset currently works best with an older datasets stack because it uses a dataset loading script.

Recommended setup:

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

Verify package versions:

    python - <<'PY'
    import sys
    import torch
    import transformers
    import datasets
    import huggingface_hub

    print("python:", sys.version)
    print("torch:", torch.__version__)
    print("transformers:", transformers.__version__)
    print("datasets:", datasets.__version__)
    print("huggingface_hub:", huggingface_hub.__version__)
    PY

Tested environment:

| Package | Version |
|---|---|
| Python | 3.11.16 |
| torch | 2.13.0+cpu |
| transformers | 4.44.2 |
| datasets | 2.19.2 |
| huggingface_hub | 0.25.2 |

## Smoke test

Tiny GPT-2 full-mode smoke test:

    python longbench_perplexity.py \
      --model sshleifer/tiny-gpt2 \
      --tasks narrativeqa \
      --limit 3 \
      --mode full \
      --max-length 512 \
      --stride 256 \
      --device cpu \
      --dtype float32

Tiny GPT-2 answer-mode smoke test:

    python longbench_perplexity.py \
      --model sshleifer/tiny-gpt2 \
      --tasks narrativeqa \
      --limit 3 \
      --mode answer \
      --max-length 512 \
      --device cpu \
      --dtype float32

## Example GPT-2 runs

Full-context diagnostic perplexity:

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

Answer-conditioned perplexity:

    python longbench_perplexity.py \
      --model gpt2 \
      --tasks narrativeqa,qasper,multifieldqa_en \
      --limit 100 \
      --mode answer \
      --max-length 1024 \
      --device cpu \
      --dtype float32 \
      --output-jsonl results/gpt2_qa3_limit100_answer.jsonl

## Tested results

### GPT-2, full mode

| Task | Examples | Tokens scored | Avg NLL | Perplexity |
|---|---:|---:|---:|---:|
| narrativeqa | 20 | 544,742 | 4.670750 | 106.7778 |
| qasper | 20 | 122,071 | 3.172258 | 23.8613 |
| multifieldqa_en | 20 | 164,685 | 3.107136 | 22.3569 |
| OVERALL | 60 | 831,498 | 4.141072 | 62.8702 |

### GPT-2, answer mode

| Task | Examples | Tokens scored | Avg NLL | Perplexity |
|---|---:|---:|---:|---:|
| narrativeqa | 100 | 599 | 4.852336 | 128.0392 |
| qasper | 100 | 1,850 | 3.776111 | 43.6460 |
| multifieldqa_en | 100 | 1,449 | 3.111908 | 22.4639 |
| OVERALL | 300 | 3,898 | 3.694589 | 40.2290 |

## Output format

When --output-jsonl is supplied, the script writes one JSON object per task plus one OVERALL row.

Example row:

    {"task": "qasper", "mode": "answer", "examples": 100, "tokens_scored": 1850, "avg_nll": 3.776110503737991, "perplexity": 43.645950404257086}

## Notes

For long examples, the tokenizer may warn that the sequence is longer than the model maximum length. This is expected for LongBench. The script chunks or truncates inputs before model evaluation according to --max-length and --stride.

For AMD ROCm GPUs, PyTorch usually still uses the cuda device string, but a ROCm-enabled PyTorch build is required. The tested environment here used CPU-only PyTorch.
