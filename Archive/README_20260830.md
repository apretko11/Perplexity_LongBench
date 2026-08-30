# LongBench Perplexity Harness

This repository provides a small evaluation harness for measuring perplexity on LongBench examples using Hugging Face causal language models.

It supports two perplexity modes:

| Mode | What is scored | Purpose |
|---|---|---|
| full | context + input | Diagnostic language-model likelihood over LongBench text |
| answer | Gold answer tokens, conditioned on the LongBench task prompt | More task-like answer likelihood |

This is not the official LongBench generation evaluation.

Official LongBench evaluation generates answers and scores them against references. This script measures token-level negative log-likelihood and perplexity.

## What this script measures

Perplexity is calculated as:

    perplexity = exp(avg_nll)

where avg_nll is the average negative log-likelihood per scored token.

Lower perplexity means the model assigned higher probability to the scored text.

Important distinction:

- full mode is a diagnostic LM likelihood measurement over LongBench text.
- answer mode is a more task-like likelihood measurement over gold answers using task-specific LongBench prompts.
- neither mode is the official LongBench generation score.

## Environment

The LongBench Hugging Face dataset currently works best with an older datasets stack because it uses a dataset loading script.

Recommended CPU setup:

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

Recommended ROCm setup:

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

Verify package versions:

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

Tested CPU environment:

| Package | Version |
|---|---|
| Python | 3.11.16 |
| torch | 2.13.0+cpu |
| transformers | 4.44.2 |
| datasets | 2.19.2 |
| huggingface_hub | 0.25.2 |

Tested ROCm environment:

| Component | Version / value |
|---|---|
| GPU | Radeon RX 7900 XT |
| ROCm | 6.3.0 |
| torch | 2.9.1+rocm6.3 |
| torch.version.hip | 6.3.42134 |
| device argument | cuda |

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
      --stride 256 \
      --device cpu \
      --dtype float32

ROCm answer-mode smoke test:

    python longbench_perplexity.py \
      --model sshleifer/tiny-gpt2 \
      --tasks narrativeqa \
      --limit 1 \
      --mode answer \
      --max-length 128 \
      --stride 64 \
      --device cuda \
      --dtype float16 \
      --output-jsonl results/smoke_answer_rocm.jsonl

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
      --stride 512 \
      --device cpu \
      --dtype float32 \
      --output-jsonl results/gpt2_qa3_limit100_answer.jsonl

Example ROCm run:

    python longbench_perplexity.py \
      --model facebook/opt-2.7b \
      --tasks narrativeqa \
      --limit 1 \
      --mode answer \
      --max-length 1024 \
      --stride 512 \
      --device cuda \
      --dtype float16 \
      --output-jsonl results/opt27b_answer_rocm_limit1.jsonl

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

### ROCm smoke tests

| Model | Task | Mode | Examples | Tokens scored | Avg NLL | Perplexity |
|---|---|---|---:|---:|---:|---:|
| gpt2 | narrativeqa | answer | 3 | 27 | 3.670232 | 39.2610 |
| facebook/opt-2.7b | narrativeqa | answer | 1 | 13 | 2.053633 | 7.7962 |
| microsoft/Phi-3-mini-4k-instruct | narrativeqa | answer | 1 | 13 | 2.162411 | 8.6921 |

## Output format

When --output-jsonl is supplied, the script writes one JSON object per task plus one OVERALL row.

Example row:

    {"task": "qasper", "mode": "answer", "examples": 100, "tokens_scored": 1850, "avg_nll": 3.776110503737991, "perplexity": 43.645950404257086}

## Notes

For long examples, the tokenizer may warn that the sequence is longer than the model maximum length. This is expected for LongBench. The script chunks or truncates inputs before model evaluation according to --max-length and --stride.

For AMD ROCm GPUs, PyTorch still uses the cuda device string. Use --device cuda with a ROCm-enabled PyTorch build.

This harness has been smoke-tested on CPU and on a Radeon RX 7900 XT using ROCm PyTorch. The intended model range for this project is approximately 271M-3B parameters. Smoke tests confirmed that GPT-2, facebook/opt-2.7b, and microsoft/Phi-3-mini-4k-instruct run successfully on the ROCm GPU with --dtype float16. Larger 7B+ models may require more careful memory handling and are outside the current scope.
