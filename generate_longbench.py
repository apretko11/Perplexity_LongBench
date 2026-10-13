#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Iterable

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from longbench_utils import build_generation_context, token_ids_sha256


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
        raise ValueError(
            f"No LongBench prompt template found for task: {task}"
        )

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


def tokenize_prompt(
    tokenizer,
    prompt: str,
    use_chat_template: bool,
) -> list[int]:
    """
    Tokenize the LongBench prompt.

    Raw-prompt mode matches longbench_perplexity.py: suppress automatic
    special tokens and manually prepend BOS if the tokenizer defines one.

    Chat-template mode uses the tokenizer's native chat template with the
    LongBench prompt as a user message and requests the generation prompt.
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
def generate_answer(
    model,
    tokenizer,
    prompt: str,
    max_length: int,
    max_new_tokens: int,
    generation_chunk_size: int,
    use_chat_template: bool,
    generation_trace: dict | None = None,
) -> tuple[str, list[int], list[int], int]:
    """Generate with LongBench-style head/tail prompt truncation.

    At each chunk, reserve room for ``chunk_new_tokens``. Keep the recent
    generated answer tokens first, then use the remaining context budget for
    the beginning and end of the original tokenized prompt. No decoding and
    re-tokenization takes place. Even when chunks roll, the generated history
    remains in causal order and the total forward length stays <= max_length.

    An optional ``generation_trace`` receives lightweight metadata sufficient
    to reconstruct *exactly* each input to model.generate() in Step 3. This
    retains the original four-value return for existing callers.
    """
    if max_length < 2:
        raise ValueError("max_length must be >= 2")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be > 0")
    if generation_chunk_size <= 0 or generation_chunk_size >= max_length:
        raise ValueError("generation_chunk_size must be in [1, max_length - 1]")

    device = first_model_device(model)
    prompt_ids = tokenize_prompt(
        tokenizer=tokenizer,
        prompt=prompt,
        use_chat_template=use_chat_template,
    )
    original_prompt_tokens = len(prompt_ids)
    if original_prompt_tokens == 0:
        raise RuntimeError("Prompt produced zero tokens.")

    if generation_trace is not None:
        generation_trace.clear()
        generation_trace.update({
            "format": "head_tail_prompt_recent_generation_v1",
            "prompt_token_ids_sha256": token_ids_sha256(prompt_ids),
            "chunks": [],
        })

    generated_token_ids = []
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id

    while len(generated_token_ids) < max_new_tokens:
        remaining = max_new_tokens - len(generated_token_ids)
        chunk_new_tokens = min(generation_chunk_size, remaining)
        context_budget = max_length - chunk_new_tokens

        # Generated history takes precedence. Remaining capacity is split
        # between the start and end of the original LongBench prompt.
        context_ids, context_plan = build_generation_context(
            prompt_ids, generated_token_ids, context_budget,
        )
        if len(context_ids) + chunk_new_tokens > max_length:
            raise RuntimeError("Generation context exceeds max_length")

        input_ids = torch.tensor(
            context_ids, dtype=torch.long, device=device,
        ).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids)

        generated_start = len(generated_token_ids)
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=chunk_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=pad_id,
            eos_token_id=eos_id,
        )
        new_ids = output_ids[0, input_ids.shape[1]:].tolist()
        if not new_ids:
            break
        if len(new_ids) > chunk_new_tokens:
            raise RuntimeError("model.generate() produced more tokens than requested")

        generated_token_ids.extend(new_ids)
        if generation_trace is not None:
            generation_trace["chunks"].append({
                "chunk_index": len(generation_trace["chunks"]),
                "requested_new_tokens": chunk_new_tokens,
                "generated_start": generated_start,
                "generated_end": len(generated_token_ids),
                **context_plan,
            })

        if eos_id is not None and eos_id in new_ids:
            break
        if len(new_ids) < chunk_new_tokens:
            break

    # Preserve exact raw IDs, including EOS, for context reconstruction.
    # Do not score trailing EOS/pad as answer *content*, matching gold PPL.
    generated_content_token_ids = generated_token_ids.copy()
    trailing_special_ids = {
        token_id for token_id in (eos_id, pad_id) if token_id is not None
    }
    while (generated_content_token_ids
           and generated_content_token_ids[-1] in trailing_special_ids):
        generated_content_token_ids.pop()

    prediction = tokenizer.decode(
        generated_content_token_ids, skip_special_tokens=True,
    )
    return (
        prediction,
        generated_token_ids,
        generated_content_token_ids,
        original_prompt_tokens,
    )

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate LongBench answers with a Hugging Face causal LM "
            "using rolling context."
        )
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
        help="Model context window used during rolling generation",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help=(
            "Maximum total answer tokens to generate. "
            "Overrides the LongBench task-specific default."
        ),
    )

    parser.add_argument(
        "--generation-chunk-size",
        type=int,
        default=64,
        help=(
            "Number of new tokens to request per rolling-context "
            "generation step"
        ),
    )

    parser.add_argument(
        "--device",
        default=(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        ),
    )

    parser.add_argument(
        "--dtype",
        choices=[
            "float32",
            "float16",
            "bfloat16",
        ],
        default="float32",
    )

    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
    )

    parser.add_argument(
        "--use-chat-template",
        action="store_true",
        help=(
            "Wrap the official LongBench prompt with the tokenizer's "
            "native chat template before generation"
        ),
    )

    parser.add_argument(
        "--output-jsonl",
        required=True,
        help="Output JSONL containing generated predictions",
    )

    args = parser.parse_args()

    dtype = dtype_from_string(args.dtype)

    print(f"Loading model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Dtype: {args.dtype}")

    model, tokenizer = load_model_and_tokenizer(
        model_name=args.model,
        device=args.device,
        dtype=dtype,
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

    # Some tokenizers use a giant integer as a sentinel meaning
    # "no meaningful tokenizer-side maximum".
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
        raise ValueError(
            "max_length must be >= 2"
        )

    if args.generation_chunk_size <= 0:
        raise ValueError(
            "generation_chunk_size must be > 0"
        )

    if args.generation_chunk_size >= max_length:
        raise ValueError(
            "generation_chunk_size must be smaller than max_length"
        )

    print(f"max_length: {max_length}")
    print(
        "generation_chunk_size: "
        f"{args.generation_chunk_size}"
    )

    output_path = Path(
        args.output_jsonl
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    n_total = 0

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as out_f:

        for task in iter_tasks(args.tasks):

            print()
            print(
                f"===== LongBench task: {task} ====="
            )

            if task not in LONGBENCH_MAX_NEW_TOKENS:
                raise ValueError(
                    "No generation-length default found "
                    f"for LongBench task: {task}"
                )

            ds = load_dataset(
                "THUDM/LongBench",
                task,
                split=args.split,
                trust_remote_code=True,
            )

            if args.limit is not None:
                ds_iter: Iterable = ds.select(
                    range(
                        min(
                            args.limit,
                            len(ds),
                        )
                    )
                )
            else:
                ds_iter = ds

            task_max_new_tokens = (
                args.max_new_tokens
                if args.max_new_tokens is not None
                else LONGBENCH_MAX_NEW_TOKENS[task]
            )

            if task_max_new_tokens <= 0:
                raise ValueError(
                    f"{task}: max_new_tokens must be > 0"
                )

            print(
                "max_new_tokens: "
                f"{task_max_new_tokens}"
            )

            for example_index, ex in enumerate(
                tqdm(ds_iter)
            ):
                prompt = make_longbench_prompt(
                    example=ex,
                    task=task,
                )

                generation_trace = {}
                (
                    prediction,
                    generated_token_ids,
                    generated_content_token_ids,
                    original_prompt_tokens,
                ) = generate_answer(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    max_length=max_length,
                    max_new_tokens=task_max_new_tokens,
                    generation_chunk_size=(
                        args.generation_chunk_size
                    ),
                    use_chat_template=(
                        args.use_chat_template
                    ),
                    generation_trace=generation_trace,
                )

                row = {
                    "task": task,
                    "example_index": example_index,
                    "model": args.model,
                    "prediction": prediction,

                    # Exact IDs returned by generation.
                    "generated_token_ids": (
                        generated_token_ids
                    ),

                    # Same answer without trailing EOS/pad.
                    # This is what we will use for generated-answer
                    # perplexity to match the existing gold-answer
                    # PPL semantics.
                    "generated_content_token_ids": (
                        generated_content_token_ids
                    ),

                    "generated_tokens": len(
                        generated_content_token_ids
                    ),

                    "generated_tokens_raw": len(
                        generated_token_ids
                    ),

                    "prompt_tokens": (
                        original_prompt_tokens
                    ),

                    "max_length": max_length,

                    "max_new_tokens": (
                        task_max_new_tokens
                    ),

                    "generation_chunk_size": (
                        args.generation_chunk_size
                    ),

                    "use_chat_template": (
                        args.use_chat_template
                    ),

                    "generation_trace": generation_trace,

                    "answers": ex.get(
                        "answers",
                        [],
                    ),

                    "length": ex.get(
                        "length",
                    ),
                }

                out_f.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                out_f.flush()

                n_total += 1

    print()
    print(
        f"Wrote {n_total} generated examples "
        f"to {output_path}"
    )


if __name__ == "__main__":
    main()
