#!/usr/bin/env python3

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


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


@torch.no_grad()
def nll_sum_for_labels(
    model,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[float, int]:
    """
    Explicit causal-LM NLL.

    This is the same calculation used by longbench_perplexity.py.

    logits at position i predict token i+1.

    Positions whose label is -100 are ignored.
    """

    outputs = model(
        input_ids=input_ids,
    )

    logits = outputs.logits

    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]

    vocab_size = shift_logits.size(-1)

    loss_sum = F.cross_entropy(
        shift_logits.reshape(
            -1,
            vocab_size,
        ).float(),
        shift_labels.reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )

    n_tokens = (
        shift_labels != -100
    ).sum().item()

    return (
        float(loss_sum.item()),
        int(n_tokens),
    )


@torch.no_grad()
def generated_answer_conditional_ppl(
    model,
    tokenizer,
    prompt: str,
    answer_ids: list[int],
    max_length: int,
    stride: int,
) -> tuple[float, int]:
    """
    Score exact generated answer token IDs conditioned on the LongBench
    prompt and previous generated answer tokens.

    This mirrors answer_conditional_ppl() from longbench_perplexity.py,
    except answer_ids come directly from generate_longbench.py instead
    of being produced by tokenizing a gold-answer string.

    Every answer token is scored exactly once.

    When prompt + answer prefix exceeds max_length, keep the rightmost
    max_length tokens so each chunk uses as much recent context as fits.
    """

    device = first_model_device(model)

    # Tokenize prompt exactly the same way as answer mode in
    # longbench_perplexity.py.
    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=False,
    )["input_ids"]

    bos_id = tokenizer.bos_token_id

    if bos_id is not None:
        prompt_ids = [bos_id] + prompt_ids

    if len(answer_ids) == 0:
        return 0.0, 0

    # Always leave at least one position before the scored region so
    # causal shifting has context from which to predict the first
    # scored token.
    answer_stride = min(
        stride,
        max_length - 1,
    )

    total_nll = 0.0
    total_tokens = 0

    for start in range(
        0,
        len(answer_ids),
        answer_stride,
    ):
        end = min(
            start + answer_stride,
            len(answer_ids),
        )

        # Logical sequence:
        #
        #     full LongBench prompt
        #     +
        #     generated answer through current chunk
        #
        # answer_ids are the exact IDs saved during generation.
        full_ids = (
            prompt_ids
            + answer_ids[:end]
        )

        # Ignore:
        #
        #     prompt
        #     +
        #     already-scored generated tokens
        #
        # Score only:
        #
        #     answer_ids[start:end]
        labels = (
            [-100]
            * (len(prompt_ids) + start)
            + answer_ids[start:end]
        )

        # Rolling scoring window.
        #
        # If logical history is longer than the model window, retain
        # only the most recent max_length tokens.
        if len(full_ids) > max_length:
            full_ids = full_ids[
                -max_length:
            ]

            labels = labels[
                -max_length:
            ]

        input_ids = torch.tensor(
            full_ids,
            dtype=torch.long,
            device=device,
        ).unsqueeze(0)

        label_ids = torch.tensor(
            labels,
            dtype=torch.long,
            device=device,
        ).unsqueeze(0)

        nll, n_tokens = nll_sum_for_labels(
            model=model,
            input_ids=input_ids,
            labels=label_ids,
        )

        total_nll += nll
        total_tokens += n_tokens

    return (
        total_nll,
        total_tokens,
    )


def read_generation_rows(
    input_path: Path,
) -> list[dict]:
    rows = []

    with input_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line_number, line in enumerate(
            f,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line "
                    f"{line_number}: {exc}"
                ) from exc

            rows.append(row)

    if not rows:
        raise RuntimeError(
            f"No generation rows found in {input_path}"
        )

    return rows


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Measure self-perplexity of generated LongBench "
            "answers using their exact saved token IDs."
        )
    )

    parser.add_argument(
        "--model",
        required=True,
        help=(
            "HF model name or local path. Must match the model "
            "that generated the answers."
        ),
    )

    parser.add_argument(
        "--input-jsonl",
        required=True,
        help=(
            "JSONL produced by generate_longbench.py"
        ),
    )

    parser.add_argument(
        "--output-jsonl",
        required=True,
        help=(
            "Output JSONL containing per-example and summary "
            "perplexity results"
        ),
    )

    parser.add_argument(
        "--split",
        default="test",
        help="LongBench split used during generation",
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help=(
            "Context window for PPL scoring. If omitted, use "
            "the max_length stored in the generation file."
        ),
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help=(
            "Answer scoring stride. Defaults to max_length."
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

    args = parser.parse_args()

    input_path = Path(
        args.input_jsonl
    )

    output_path = Path(
        args.output_jsonl
    )

    rows = read_generation_rows(
        input_path
    )

    # ------------------------------------------------------------------
    # Self-perplexity guard
    # ------------------------------------------------------------------
    #
    # Every row must have been generated by the same model we are now
    # using to score it.
    #

    recorded_models = {
        str(row.get("model"))
        for row in rows
    }

    if len(recorded_models) != 1:
        raise ValueError(
            "Input JSONL contains generations from multiple models: "
            f"{sorted(recorded_models)}"
        )

    recorded_model = next(
        iter(recorded_models)
    )

    if recorded_model != args.model:
        raise ValueError(
            "Self-perplexity requires the scoring model to match "
            "the generation model.\n"
            f"Generation model: {recorded_model}\n"
            f"Scoring model:    {args.model}"
        )

    # ------------------------------------------------------------------
    # Determine scoring max_length
    # ------------------------------------------------------------------

    recorded_max_lengths = {
        int(row["max_length"])
        for row in rows
        if row.get("max_length") is not None
    }

    if args.max_length is not None:
        max_length = args.max_length
    else:
        if len(recorded_max_lengths) != 1:
            raise ValueError(
                "Could not infer a single max_length from the "
                "generation file. Specify --max-length explicitly."
            )

        max_length = next(
            iter(recorded_max_lengths)
        )

    recorded_chunk_sizes = {
        int(row["generation_chunk_size"])
        for row in rows
        if row.get("generation_chunk_size") is not None
    }

    if args.stride is not None:
        stride = args.stride
    else:
        if len(recorded_chunk_sizes) != 1:
            raise ValueError(
                "Could not infer a single generation_chunk_size from the "
                "generation file. Specify --stride explicitly."
            )

        stride = next(iter(recorded_chunk_sizes))

    if max_length < 2:
        raise ValueError(
            "max_length must be >= 2"
        )

    if stride <= 0:
        raise ValueError(
            "stride must be > 0"
        )

    if stride > max_length:
        raise ValueError(
            "stride must be <= max_length"
        )

    dtype = dtype_from_string(
        args.dtype
    )

    print(
        f"Input generations: {input_path}"
    )

    print(
        f"Loading model: {args.model}"
    )

    print(
        f"Device: {args.device}"
    )

    print(
        f"Dtype: {args.dtype}"
    )

    print(
        f"max_length: {max_length}"
    )

    print(
        f"stride: {stride}"
    )

    model, tokenizer = (
        load_model_and_tokenizer(
            model_name=args.model,
            device=args.device,
            dtype=dtype,
            trust_remote_code=(
                args.trust_remote_code
            ),
        )
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Load each LongBench task only once.
    datasets = {}

    task_totals = {}

    grand_nll = 0.0
    grand_tokens = 0
    grand_examples = 0

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as out_f:

        for row in tqdm(
            rows,
            desc="Scoring generations",
        ):
            task = row.get("task")

            if task is None:
                raise ValueError(
                    "Generation row is missing 'task'"
                )

            if task not in LONGBENCH_PROMPTS:
                raise ValueError(
                    f"Unknown LongBench task: {task}"
                )

            if "example_index" not in row:
                raise ValueError(
                    "Generation row is missing "
                    "'example_index'"
                )

            example_index = int(
                row["example_index"]
            )

            if (
                "generated_content_token_ids"
                not in row
            ):
                raise ValueError(
                    "Generation row is missing "
                    "'generated_content_token_ids'"
                )

            # These IDs are used DIRECTLY.
            #
            # There is deliberately no:
            #
            #     prediction text -> tokenizer
            #
            # step.
            answer_ids = [
                int(token_id)
                for token_id
                in row[
                    "generated_content_token_ids"
                ]
            ]

            if task not in datasets:
                print()
                print(
                    f"Loading LongBench task: {task}"
                )

                datasets[task] = load_dataset(
                    "THUDM/LongBench",
                    task,
                    split=args.split,
                    trust_remote_code=True,
                )

            ds = datasets[task]

            if (
                example_index < 0
                or example_index >= len(ds)
            ):
                raise IndexError(
                    f"{task}: example_index "
                    f"{example_index} is outside "
                    f"dataset size {len(ds)}"
                )

            ex = ds[
                example_index
            ]

            prompt = make_longbench_prompt(
                example=ex,
                task=task,
            )

            nll, n_tokens = (
                generated_answer_conditional_ppl(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    answer_ids=answer_ids,
                    max_length=max_length,
                    stride=stride,
                )
            )

            # It is possible for a model to immediately emit EOS,
            # producing zero content tokens.
            if n_tokens == 0:
                example_result = {
                    "row_type": "example",
                    "task": task,
                    "example_index": (
                        example_index
                    ),
                    "model": args.model,
                    "generated_tokens": len(
                        answer_ids
                    ),
                    "tokens_scored": 0,
                    "nll_sum": 0.0,
                    "avg_nll": None,
                    "perplexity": None,
                }

                out_f.write(
                    json.dumps(
                        example_result
                    )
                    + "\n"
                )

                out_f.flush()

                continue

            avg_nll = (
                nll
                / n_tokens
            )

            ppl = math.exp(
                avg_nll
            )

            example_result = {
                "row_type": "example",
                "task": task,
                "example_index": example_index,
                "model": args.model,
                "generated_tokens": len(
                    answer_ids
                ),
                "tokens_scored": n_tokens,
                "nll_sum": nll,
                "avg_nll": avg_nll,
                "perplexity": ppl,
            }

            out_f.write(
                json.dumps(
                    example_result
                )
                + "\n"
            )

            out_f.flush()

            if task not in task_totals:
                task_totals[task] = {
                    "nll": 0.0,
                    "tokens": 0,
                    "examples": 0,
                }

            task_totals[task][
                "nll"
            ] += nll

            task_totals[task][
                "tokens"
            ] += n_tokens

            task_totals[task][
                "examples"
            ] += 1

            grand_nll += nll
            grand_tokens += n_tokens
            grand_examples += 1

        print()

        # --------------------------------------------------------------
        # Per-task summaries
        # --------------------------------------------------------------

        for task, totals in task_totals.items():

            if totals["tokens"] == 0:
                continue

            task_avg_nll = (
                totals["nll"]
                / totals["tokens"]
            )

            task_ppl = math.exp(
                task_avg_nll
            )

            task_summary = {
                "row_type": "task_summary",
                "task": task,
                "model": args.model,
                "examples": (
                    totals["examples"]
                ),
                "tokens_scored": (
                    totals["tokens"]
                ),
                "avg_nll": task_avg_nll,
                "perplexity": task_ppl,
            }

            print(
                json.dumps(
                    task_summary,
                    indent=2,
                )
            )

            out_f.write(
                json.dumps(
                    task_summary
                )
                + "\n"
            )

        # --------------------------------------------------------------
        # Overall summary
        # --------------------------------------------------------------

        if grand_tokens == 0:
            raise RuntimeError(
                "No generated answer tokens were scored."
            )

        overall_avg_nll = (
            grand_nll
            / grand_tokens
        )

        overall_ppl = math.exp(
            overall_avg_nll
        )

        overall = {
            "row_type": "overall_summary",
            "task": "OVERALL",
            "model": args.model,
            "examples": grand_examples,
            "tokens_scored": grand_tokens,
            "avg_nll": overall_avg_nll,
            "perplexity": overall_ppl,
        }

        print()
        print(
            "===== OVERALL ====="
        )

        print(
            json.dumps(
                overall,
                indent=2,
            )
        )

        out_f.write(
            json.dumps(
                overall
            )
            + "\n"
        )

    print()
    print(
        f"Wrote perplexity results to "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()
