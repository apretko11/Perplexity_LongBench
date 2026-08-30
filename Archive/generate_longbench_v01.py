#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Iterable

import torch
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


# Reasonable LongBench task-specific generation limits.
LONGBENCH_MAX_NEW_TOKENS = {
    "narrativeqa": 128,
    "qasper": 128,
    "multifieldqa_en": 64,
    "hotpotqa": 32,
    "2wikimqa": 32,
    "musique": 32,
    "gov_report": 512,
    "qmsum": 512,
    "multi_news": 512,
    "trec": 64,
    "triviaqa": 32,
    "samsum": 128,
    "passage_count": 32,
    "passage_retrieval_en": 32,
    "lcc": 64,
    "repobench-p": 64,
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


def load_model_and_tokenizer(
    model_name: str,
    device: str,
    dtype: torch.dtype,
    trust_remote_code: bool,
):
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


def make_longbench_prompt(example: dict, task: str) -> str:
    context = example.get("context", "")
    inp = example.get("input", "")

    if task not in LONGBENCH_PROMPTS:
        raise ValueError(f"No LongBench prompt template found for task: {task}")

    return LONGBENCH_PROMPTS[task].format(
        context=context,
        input=inp,
    )


def iter_tasks(tasks_arg: str) -> list[str]:
    if tasks_arg == "all":
        return DEFAULT_LONGBENCH_TASKS

    return [
        x.strip()
        for x in tasks_arg.split(",")
        if x.strip()
    ]


def prepare_prompt_ids(
    tokenizer,
    prompt: str,
    prompt_budget: int,
) -> list[int]:
    """
    Tokenize the prompt exactly once.

    As in the existing answer-PPL code:
      - do not automatically add EOS/special tokens
      - manually prepend BOS when the tokenizer has one

    If the prompt is too long, keep the rightmost prompt tokens.
    """

    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=False,
    )["input_ids"]

    bos_id = tokenizer.bos_token_id
    if bos_id is not None:
        prompt_ids = [bos_id] + prompt_ids

    if len(prompt_ids) > prompt_budget:
        prompt_ids = prompt_ids[-prompt_budget:]

    return prompt_ids


@torch.no_grad()
def generate_answer(
    model,
    tokenizer,
    prompt: str,
    max_length: int,
    max_new_tokens: int,
) -> tuple[str, list[int], int]:
    """
    Generate one answer.

    Returns:
        decoded prediction
        exact generated content token IDs
        number of prompt tokens actually given to model
    """

    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be > 0")

    if max_new_tokens >= max_length:
        raise ValueError(
            f"max_new_tokens ({max_new_tokens}) must be smaller than "
            f"max_length ({max_length})"
        )

    device = first_model_device(model)

    # Reserve enough context-window space for the entire generated answer.
    prompt_budget = max_length - max_new_tokens

    prompt_ids = prepare_prompt_ids(
        tokenizer=tokenizer,
        prompt=prompt,
        prompt_budget=prompt_budget,
    )

    if len(prompt_ids) == 0:
        raise RuntimeError("Prompt became empty after tokenization/truncation.")

    input_ids = torch.tensor(
        prompt_ids,
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)

    attention_mask = torch.ones_like(input_ids)

    output_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    generated_ids = output_ids[0, input_ids.shape[1]:].tolist()

    # For perplexity comparison with gold-answer mode, do not treat a final
    # EOS/padding token as part of the answer text.
    while generated_ids and generated_ids[-1] in {
        tokenizer.eos_token_id,
        tokenizer.pad_token_id,
    }:
        generated_ids.pop()

    prediction = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    )

    return prediction, generated_ids, len(prompt_ids)


def main():
    parser = argparse.ArgumentParser(
        description="Generate answers for LongBench with a Hugging Face causal LM."
    )

    parser.add_argument(
        "--model",
        required=True,
        help="HF model name or local path",
    )

    parser.add_argument(
        "--tasks",
        default="narrativeqa",
        help="LongBench task, comma list, or 'all'",
    )

    parser.add_argument(
        "--split",
        default="test",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit examples per task for smoke tests",
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Total model context window including generated answer",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Override task-specific generation length",
    )

    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default="float32",
    )

    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
    )

    parser.add_argument(
        "--output-jsonl",
        required=True,
        help="Output file containing generated predictions",
    )

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

    inferred_max_length = getattr(
        model.config,
        "max_position_embeddings",
        None,
    )

    if inferred_max_length is None:
        inferred_max_length = getattr(
            tokenizer,
            "model_max_length",
            2048,
        )

    if (
        inferred_max_length is None
        or inferred_max_length > 1_000_000
    ):
        inferred_max_length = 2048

    max_length = (
        args.max_length
        if args.max_length is not None
        else int(inferred_max_length)
    )

    if max_length < 2:
        raise ValueError("max_length must be >= 2")

    print(f"max_length: {max_length}")

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    n_total = 0

    with output_path.open("w", encoding="utf-8") as out_f:

        for task in iter_tasks(args.tasks):

            print()
            print(f"===== LongBench task: {task} =====")

            ds = load_dataset(
                "THUDM/LongBench",
                task,
                split=args.split,
                trust_remote_code=True,
            )

            if args.limit is not None:
                ds_iter: Iterable = ds.select(
                    range(min(args.limit, len(ds)))
                )
            else:
                ds_iter = ds

            task_max_new_tokens = (
                args.max_new_tokens
                if args.max_new_tokens is not None
                else LONGBENCH_MAX_NEW_TOKENS[task]
            )

            if task_max_new_tokens >= max_length:
                raise ValueError(
                    f"{task}: max_new_tokens={task_max_new_tokens} "
                    f"must be smaller than max_length={max_length}"
                )

            print(
                f"max_new_tokens: "
                f"{task_max_new_tokens}"
            )

            for example_index, ex in enumerate(
                tqdm(ds_iter)
            ):
                prompt = make_longbench_prompt(
                    ex,
                    task,
                )

                prediction, generated_token_ids, prompt_tokens = (
                    generate_answer(
                        model=model,
                        tokenizer=tokenizer,
                        prompt=prompt,
                        max_length=max_length,
                        max_new_tokens=task_max_new_tokens,
                    )
                )

                row = {
                    "task": task,
                    "example_index": example_index,
                    "model": args.model,
                    "prediction": prediction,
                    "generated_token_ids": generated_token_ids,
                    "generated_tokens": len(generated_token_ids),
                    "prompt_tokens": prompt_tokens,
                    "max_length": max_length,
                    "max_new_tokens": task_max_new_tokens,
                    "answers": ex.get("answers", []),
                    "length": ex.get("length"),
                }

                out_f.write(
                    json.dumps(row, ensure_ascii=False)
                    + "\n"
                )
                out_f.flush()

                n_total += 1

    print()
    print(
        f"Wrote {n_total} generated examples to "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()
