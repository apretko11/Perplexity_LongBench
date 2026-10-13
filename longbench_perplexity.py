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
from longbench_utils import middle_truncate_tokens


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

LONGBENCH_PROMPTS = {
    "narrativeqa": "You are given a story, which can be either a novel or a movie script, and a question. Answer the question asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nStory: {context}\n\nNow, answer the question based on the story asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "qasper": "You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nArticle: {context}\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "multifieldqa_en": "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "hotpotqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "2wikimqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "musique": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "gov_report": "You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:",
    "qmsum": "You are given a meeting transcript and a query containing a question or instruction. Answer the query in one or more sentences.\n\nTranscript:\n{context}\n\nNow, answer the query based on the above meeting transcript in one or more sentences.\n\nQuery: {input}\nAnswer:",
    "multi_news": "You are given several news passages. Write a one-page summary of all news. \n\nNews:\n{context}\n\nNow, write a one-page summary of all the news.\n\nSummary:",
    "trec": "Please determine the type of the question below. Here are some examples of questions.\n\n{context}\n{input}",
    "triviaqa": "Answer the question based on the given passage. Only give me the answer and do not output any other words. The following are some examples.\n\n{context}\n\n{input}",
    "samsum": "Summarize the dialogue into a few short sentences. The following are some examples.\n\n{context}\n\n{input}",
    "passage_count": "There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. Please carefully read these paragraphs and determine how many unique paragraphs there are after removing duplicates. In other words, how many non-repeating paragraphs are there in total?\n\n{context}\n\nPlease enter the final count of unique paragraphs after removing duplicates. The output format should only contain the number, such as 1, 2, 3, and so on.\n\nThe final answer is: ",
    "passage_retrieval_en": "Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is from.\n\n{context}\n\nThe following is an abstract.\n\n{input}\n\nPlease enter the number of the paragraph that the abstract is from. The answer format must be like \"Paragraph 1\", \"Paragraph 2\", etc.\n\nThe answer is: ",
    "lcc": "Please complete the code given below. \n{context}Next line of code:\n",
    "repobench-p": "Please complete the code given below. \n{context}{input}Next line of code:\n",
}


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


def tokenize_prompt(
    tokenizer,
    prompt: str,
    use_chat_template: bool,
) -> list[int]:
    """
    Tokenize an answer-mode prompt.

    Raw mode matches the existing behavior. Chat mode uses the
    tokenizer's native chat template and leaves the assistant
    generation prompt in place before the gold answer.
    """

    if use_chat_template:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
        )

    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=False,
    )["input_ids"]

    bos_id = tokenizer.bos_token_id
    if bos_id is not None:
        prompt_ids = [bos_id] + prompt_ids

    return prompt_ids


@torch.no_grad()
def answer_conditional_ppl(
    model,
    tokenizer,
    prompt: str,
    answer: str,
    max_length: int,
    stride: int,
    use_chat_template: bool,
) -> tuple[float, int]:
    """
    Scores all answer tokens, conditioned on as much prompt + previous answer
    context as fits in max_length.

    If the prompt must be truncated, keep tokens from both its beginning and
    end while preserving the answer prefix used by the current scoring window.
    """
    device = first_model_device(model)

    prompt_ids = tokenize_prompt(
        tokenizer=tokenizer,
        prompt=prompt,
        use_chat_template=use_chat_template,
    )
    answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]

    if len(answer_ids) == 0:
        return 0.0, 0

    # Do not try to score more answer tokens in one window than the model can handle.
    answer_stride = min(stride, max_length - 1)

    total_nll = 0.0
    total_tokens = 0

    for start in range(0, len(answer_ids), answer_stride):
        end = min(start + answer_stride, len(answer_ids))

        # Reserve room for the answer prefix. LongBench keeps the head and tail
        # of an overlength prompt; only answer tokens contribute to NLL.
        answer_prefix = answer_ids[:end]
        prompt_budget = max_length - len(answer_prefix)
        window_prompt = middle_truncate_tokens(
            prompt_ids, max(0, prompt_budget),
        )

        # Model sees the retained prompt plus answer up through the current chunk.
        full_ids = window_prompt + answer_prefix

        # Ignore prompt and previous answer tokens.
        # Score only answer_ids[start:end].
        labels = [-100] * (len(window_prompt) + start) + answer_ids[start:end]

        # Only needed when the answer prefix itself exceeds the context window.
        if len(full_ids) > max_length:
            full_ids = full_ids[-max_length:]
            labels = labels[-max_length:]

        input_ids = torch.tensor(full_ids, dtype=torch.long, device=device).unsqueeze(0)
        label_ids = torch.tensor(labels, dtype=torch.long, device=device).unsqueeze(0)

        nll, n_tokens = nll_sum_for_labels(model, input_ids, label_ids)

        total_nll += nll
        total_tokens += n_tokens

    return total_nll, total_tokens


def make_longbench_prompt(example: dict, task: str) -> str:
    context = example.get("context", "")
    inp = example.get("input", "")

    if task not in LONGBENCH_PROMPTS:
        raise ValueError(f"No LongBench prompt template found for task: {task}")

    return LONGBENCH_PROMPTS[task].format(context=context, input=inp)


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
    parser.add_argument("--stride", type=int, default=None, help="Stride for sliding-window PPL")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--use-chat-template",
        action="store_true",
        help="Use the tokenizer's native chat template in answer mode only",
    )
    parser.add_argument(
        "--reference-policy",
        choices=["first", "all"],
        default="first",
        help="Answer mode: score a fixed first reference (default) or every distinct reference.",
    )
    parser.add_argument("--output-jsonl", default=None, help="Optional per-task summary JSONL output")
    parser.add_argument(
        "--examples-output-jsonl",
        default=None,
        help="Optional per-example JSONL path. Defaults to <output-jsonl stem>_examples.jsonl.",
    )
    args = parser.parse_args()

    dtype = dtype_from_string(args.dtype)

    print(f"Loading model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Dtype: {args.dtype}")

    model, tokenizer = load_model_and_tokenizer(
        args.model, args.device, dtype, trust_remote_code=args.trust_remote_code,
    )

    inferred_max_length = getattr(model.config, "max_position_embeddings", None)
    if inferred_max_length is None:
        inferred_max_length = getattr(tokenizer, "model_max_length", 2048)
    if inferred_max_length is None or inferred_max_length > 1_000_000:
        inferred_max_length = 2048

    max_length = args.max_length if args.max_length is not None else int(inferred_max_length)
    stride = args.stride if args.stride is not None else max_length
    if max_length < 2:
        raise ValueError("max_length must be >= 2")
    if stride <= 0:
        raise ValueError("stride must be > 0")
    if stride > max_length:
        raise ValueError("stride must be <= max_length")

    print(f"max_length: {max_length}")
    print(f"stride: {stride}")
    print(f"mode: {args.mode}")
    if args.mode == "answer":
        print(f"use_chat_template: {args.use_chat_template}")
        print(f"reference_policy: {args.reference_policy}")

    out_path = Path(args.output_jsonl) if args.output_jsonl else None
    examples_path = (
        Path(args.examples_output_jsonl) if args.examples_output_jsonl
        else out_path.with_name(out_path.stem + "_examples.jsonl") if out_path
        else None
    )
    if out_path is not None and examples_path is not None:
        if out_path.resolve() == examples_path.resolve():
            raise ValueError("Summary and per-example JSONL paths must differ")

    out_f = None
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_f = out_path.open("w", encoding="utf-8")

    examples_out_f = None
    if examples_path is not None:
        examples_path.parent.mkdir(parents=True, exist_ok=True)
        examples_out_f = examples_path.open("w", encoding="utf-8")
        print(f"Per-example output: {examples_path}")

    grand_nll = 0.0
    grand_tokens = 0
    grand_example_mean_nll_sum = 0.0
    grand_examples = 0
    task_rows = []

    for task in iter_tasks(args.tasks):
        print()
        print(f"===== LongBench task: {task} =====")

        ds = load_dataset("THUDM/LongBench", task, split=args.split, trust_remote_code=True)
        task_nll = 0.0
        task_tokens = 0
        task_example_mean_nll_sum = 0.0
        n_examples = 0

        if args.limit is not None:
            ds_iter: Iterable = ds.select(range(min(args.limit, len(ds))))
        else:
            ds_iter = ds

        for example_index, ex in enumerate(tqdm(ds_iter)):
            reference_results = []
            original_prompt_tokens = None
            reference_count_available = 0

            if args.mode == "full":
                text = make_longbench_full_text(ex)
                nll, n_tokens = full_text_ppl(
                    model=model, tokenizer=tokenizer, text=text,
                    max_length=max_length, stride=stride,
                )
                example_mean_nll = nll / n_tokens if n_tokens else None
            else:
                answers = ex.get("answers", [])
                if not answers:
                    continue
                reference_count_available = len(answers)
                prompt = make_longbench_prompt(ex, task)
                original_prompt_tokens = len(tokenize_prompt(
                    tokenizer=tokenizer,
                    prompt=prompt,
                    use_chat_template=args.use_chat_template,
                ))
                selected_indices = (
                    range(1) if args.reference_policy == "first"
                    else range(len(answers))
                )
                seen_answers = set()
                for reference_index in selected_indices:
                    answer = str(answers[reference_index])
                    if not answer.strip() or answer in seen_answers:
                        continue
                    seen_answers.add(answer)

                    candidate_nll, candidate_tokens = answer_conditional_ppl(
                        model=model, tokenizer=tokenizer, prompt=prompt,
                        answer=answer, max_length=max_length, stride=stride,
                        use_chat_template=args.use_chat_template,
                    )
                    if candidate_tokens == 0:
                        continue

                    first_answer_chunk = min(candidate_tokens, stride, max_length - 1)
                    first_prompt_budget = max_length - first_answer_chunk
                    reference_results.append({
                        "reference_index": reference_index,
                        "reference": answer,
                        "nll_sum": candidate_nll,
                        "tokens_scored": candidate_tokens,
                        "avg_nll": candidate_nll / candidate_tokens,
                        "perplexity": math.exp(candidate_nll / candidate_tokens),
                        "prompt_tokens_retained_first_chunk": min(
                            original_prompt_tokens, first_prompt_budget,
                        ),
                        "prompt_middle_truncated_first_chunk": (
                            original_prompt_tokens > first_prompt_budget
                        ),
                    })

                if not reference_results:
                    continue
                nll = sum(r["nll_sum"] for r in reference_results)
                n_tokens = sum(r["tokens_scored"] for r in reference_results)
                # Equal weight for each distinct reference *within this example*.
                # Unlike pooled token NLL, this is not ordinary token-weighted PPL.
                example_mean_nll = sum(
                    r["avg_nll"] for r in reference_results
                ) / len(reference_results)

            if n_tokens == 0:
                continue

            task_nll += nll
            task_tokens += n_tokens
            task_example_mean_nll_sum += example_mean_nll
            n_examples += 1

            if examples_out_f is not None:
                example_result = {
                    "row_type": "example",
                    "task": task,
                    "example_index": example_index,
                    "example_id": ex.get("_id"),
                    "model": args.model,
                    "mode": args.mode,
                    "reference_policy": (
                        args.reference_policy if args.mode == "answer" else None
                    ),
                    "reference_count_available": reference_count_available,
                    "reference_count_scored": len(reference_results),
                    "nll_sum": nll,
                    "tokens_scored": n_tokens,
                    "avg_nll": nll / n_tokens,
                    "perplexity": math.exp(nll / n_tokens),
                    "reference_macro_avg_nll": example_mean_nll,
                    "reference_macro_perplexity": math.exp(example_mean_nll),
                    "prompt_tokens_original": original_prompt_tokens,
                    "max_length": max_length,
                    "stride": stride,
                    "references": reference_results,
                }
                examples_out_f.write(json.dumps(example_result, ensure_ascii=False) + "\n")
                examples_out_f.flush()

        if task_tokens == 0:
            print(f"Skipping task {task}: no scored tokens.")
            continue

        # Standard perplexity: total NLL / total scored tokens.
        task_avg_nll = task_nll / task_tokens
        task_ppl = math.exp(task_avg_nll)
        # Supplementary metric: first mean over references, then examples.
        # Every example contributes equally regardless of reference lengths/count.
        task_example_macro_nll = task_example_mean_nll_sum / n_examples

        row = {
            "task": task,
            "mode": args.mode,
            "use_chat_template": args.use_chat_template if args.mode == "answer" else False,
            "reference_policy": args.reference_policy if args.mode == "answer" else None,
            "examples": n_examples,
            "tokens_scored": task_tokens,
            "avg_nll": task_avg_nll,
            "perplexity": task_ppl,
            "example_macro_avg_nll": task_example_macro_nll,
            "example_macro_perplexity": math.exp(task_example_macro_nll),
        }
        task_rows.append(row)

        print(json.dumps(row, indent=2))
        if out_f is not None:
            out_f.write(json.dumps(row) + "\n")
            out_f.flush()

        grand_nll += task_nll
        grand_tokens += task_tokens
        grand_example_mean_nll_sum += task_example_mean_nll_sum
        grand_examples += n_examples

    if grand_tokens == 0:
        raise RuntimeError("No scored tokens across all tasks.")

    micro_avg_nll = grand_nll / grand_tokens
    micro_ppl = math.exp(micro_avg_nll)
    macro_avg_nll = sum(r["avg_nll"] for r in task_rows) / len(task_rows)
    macro_ppl = math.exp(macro_avg_nll)
    example_macro_avg_nll = grand_example_mean_nll_sum / grand_examples

    overall = {
        "task": "OVERALL",
        "mode": args.mode,
        "use_chat_template": args.use_chat_template if args.mode == "answer" else False,
        "reference_policy": args.reference_policy if args.mode == "answer" else None,
        "examples": sum(r["examples"] for r in task_rows),
        "tokens_scored": grand_tokens,
        # Preserve the historical overall micro keys.
        "avg_nll": micro_avg_nll,
        "perplexity": micro_ppl,
        "micro_avg_nll": micro_avg_nll,
        "micro_perplexity": micro_ppl,
        "macro_avg_nll": macro_avg_nll,
        "macro_perplexity": macro_ppl,
        "example_macro_avg_nll": example_macro_avg_nll,
        "example_macro_perplexity": math.exp(example_macro_avg_nll),
    }
    print()
    print("===== OVERALL =====")
    print(json.dumps(overall, indent=2))
    if out_f is not None:
        out_f.write(json.dumps(overall) + "\n")
        out_f.close()
    if examples_out_f is not None:
        examples_out_f.close()


if __name__ == "__main__":
    main()
