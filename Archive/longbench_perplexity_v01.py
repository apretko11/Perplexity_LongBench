#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path
from typing import Iterable, Optional

import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_LONGBENCH_TASKS = [
    "narrativeqa",
    "qasper",
    "multifieldqa_en",
    "hotpotqa",
    "2wikimqa",
    "musique",
    "gov_report",
    "qmsum",
    "multi_news",
    "trec",
    "triviaqa",
    "samsum",
    "passage_count",
    "passage_retrieval_en",
    "lcc",
    "repobench-p",
]


def dtype_from_string(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unknown dtype: {name}")


def first_model_device(model) -> torch.device:
    return next(model.parameters()).device


def load_model_and_tokenizer(model_name: str, device: str, dtype: torch.dtype, trust_remote_code: bool):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        use_fast=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
    )

    model.to(device)
    model.eval()

    return model, tokenizer


@torch.no_grad()
def nll_sum_for_labels(model, input_ids: torch.Tensor, labels: torch.Tensor) -> tuple[float, int]:
    """
    Explicit causal-LM NLL.

    labels should be same shape as input_ids.
    Positions with -100 are ignored.
    We shift manually:
      logits[:, :-1, : ] predict labels[:, 1:]
    """
    outputs = model(input_ids=input_ids)
    logits = outputs.logits

    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]

    vocab_size = shift_logits.size(-1)

    loss_sum = F.cross_entropy(
        shift_logits.reshape(-1, vocab_size).float(),
        shift_labels.reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )

    n_tokens = (shift_labels != -100).sum().item()
    return float(loss_sum.item()), int(n_tokens)


@torch.no_grad()
def full_text_ppl(
    model,
    tokenizer,
    text: str,
    max_length: int,
    stride: int,
) -> tuple[float, int]:
    """
    Standard sliding-window PPL over one long text.

    This scores each token once, using as much left context as max_length allows.
    """
    device = first_model_device(model)

    enc = tokenizer(text, return_tensors="pt", add_special_tokens=True)
    input_ids = enc["input_ids"][0]
    seq_len = input_ids.numel()

    if seq_len < 2:
        return 0.0, 0

    total_nll = 0.0
    total_tokens = 0

    # Hugging Face style sliding window:
    # i marks the first newly scored token region.
    for i in range(0, seq_len, stride):
        begin = max(i + stride - max_length, 0)
        end = min(i + stride, seq_len)
        trg_len = end - i

        chunk = input_ids[begin:end].unsqueeze(0).to(device)
        labels = chunk.clone()

        # Ignore everything except the newly introduced trailing trg_len tokens.
        labels[:, :-trg_len] = -100

        nll, n_tokens = nll_sum_for_labels(model, chunk, labels)
        total_nll += nll
        total_tokens += n_tokens

        if end == seq_len:
            break

    return total_nll, total_tokens


@torch.no_grad()
def answer_conditional_ppl(
    model,
    tokenizer,
    prompt: str,
    answer: str,
    max_length: int,
) -> tuple[float, int]:
    """
    Scores answer tokens only, conditioned on prompt.

    If prompt+answer exceeds max_length, we left-truncate the prompt side.
    This is the pragmatic long-context evaluation behavior.
    """
    device = first_model_device(model)

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]

    bos_id = tokenizer.bos_token_id
    if bos_id is not None:
        prompt_ids = [bos_id] + prompt_ids

    if len(answer_ids) == 0:
        return 0.0, 0

    full_ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids

    if len(full_ids) > max_length:
        # Keep the rightmost max_length tokens, so the answer remains present.
        full_ids = full_ids[-max_length:]
        labels = labels[-max_length:]

    input_ids = torch.tensor(full_ids, dtype=torch.long, device=device).unsqueeze(0)
    label_ids = torch.tensor(labels, dtype=torch.long, device=device).unsqueeze(0)

    nll, n_tokens = nll_sum_for_labels(model, input_ids, label_ids)
    return nll, n_tokens


def make_longbench_prompt(example: dict) -> str:
    context = example.get("context", "")
    inp = example.get("input", "")

    # Simple generic format. Later we can swap this for official LongBench task prompts.
    return f"Context:\n{context}\n\nInput:\n{inp}\n\nAnswer:"


def make_longbench_full_text(example: dict) -> str:
    context = example.get("context", "")
    inp = example.get("input", "")

    # For generic text perplexity, do not include gold answer by default.
    return f"{context}\n\n{inp}"


def iter_tasks(tasks_arg: str) -> list[str]:
    if tasks_arg == "all":
        return DEFAULT_LONGBENCH_TASKS
    return [x.strip() for x in tasks_arg.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(description="Measure perplexity on LongBench with a HF causal LM.")
    parser.add_argument("--model", required=True, help="HF model name or local path")
    parser.add_argument("--tasks", default="narrativeqa", help="LongBench task, comma list, or 'all'")
    parser.add_argument("--split", default="test")
    parser.add_argument("--mode", choices=["full", "answer"], default="full")
    parser.add_argument("--limit", type=int, default=None, help="Limit examples per task for smoke tests")
    parser.add_argument("--max-length", type=int, default=None, help="Context window used for scoring")
    parser.add_argument("--stride", type=int, default=None, help="Stride for full-text sliding-window PPL")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--output-jsonl", default=None, help="Optional per-task JSONL output")
    args = parser.parse_args()

    dtype = dtype_from_string(args.dtype)

    print(f"Loading model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Dtype: {args.dtype}")

    model, tokenizer = load_model_and_tokenizer(
        args.model,
        args.device,
        dtype,
        trust_remote_code=args.trust_remote_code,
    )

    inferred_max_length = getattr(model.config, "max_position_embeddings", None)
    if inferred_max_length is None:
        inferred_max_length = getattr(tokenizer, "model_max_length", 2048)

    # Some tokenizers set this to an absurd sentinel.
    if inferred_max_length is None or inferred_max_length > 1_000_000:
        inferred_max_length = 2048

    max_length = args.max_length or int(inferred_max_length)
    stride = args.stride or max_length

    print(f"max_length: {max_length}")
    print(f"stride: {stride}")
    print(f"mode: {args.mode}")

    out_f = None
    if args.output_jsonl:
        out_path = Path(args.output_jsonl)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_f = out_path.open("w", encoding="utf-8")

    grand_nll = 0.0
    grand_tokens = 0
    task_rows = []

    for task in iter_tasks(args.tasks):
        print()
        print(f"===== LongBench task: {task} =====")

        ds = load_dataset("THUDM/LongBench", task, split=args.split, trust_remote_code=True)

        task_nll = 0.0
        task_tokens = 0
        n_examples = 0

        if args.limit is not None:
            ds_iter: Iterable = ds.select(range(min(args.limit, len(ds))))
        else:
            ds_iter = ds

        for ex in tqdm(ds_iter):
            if args.mode == "full":
                text = make_longbench_full_text(ex)
                nll, n_tokens = full_text_ppl(
                    model=model,
                    tokenizer=tokenizer,
                    text=text,
                    max_length=max_length,
                    stride=stride,
                )
            else:
                answers = ex.get("answers", [])
                if not answers:
                    continue
                prompt = make_longbench_prompt(ex)
                answer = str(answers[0])
                nll, n_tokens = answer_conditional_ppl(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    answer=answer,
                    max_length=max_length,
                )

            if n_tokens == 0:
                continue

            task_nll += nll
            task_tokens += n_tokens
            n_examples += 1

        task_avg_nll = task_nll / task_tokens
        task_ppl = math.exp(task_avg_nll)

        row = {
            "task": task,
            "mode": args.mode,
            "examples": n_examples,
            "tokens_scored": task_tokens,
            "avg_nll": task_avg_nll,
            "perplexity": task_ppl,
        }
        task_rows.append(row)

        print(json.dumps(row, indent=2))

        if out_f is not None:
            out_f.write(json.dumps(row) + "\n")
            out_f.flush()

        grand_nll += task_nll
        grand_tokens += task_tokens

    overall_avg_nll = grand_nll / grand_tokens
    overall_ppl = math.exp(overall_avg_nll)

    overall = {
        "task": "OVERALL",
        "mode": args.mode,
        "examples": sum(r["examples"] for r in task_rows),
        "tokens_scored": grand_tokens,
        "avg_nll": overall_avg_nll,
        "perplexity": overall_ppl,
    }

    print()
    print("===== OVERALL =====")
    print(json.dumps(overall, indent=2))

    if out_f is not None:
        out_f.write(json.dumps(overall) + "\n")
        out_f.close()


if __name__ == "__main__":
    main()
