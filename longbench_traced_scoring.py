"""Verified teacher-forced self-PPL for Step 2 LongBench generation traces.

All scored tokens are the original generated token IDs; the decoded prediction
is never retokenized.  This reconstructs the exact *generation input IDs* for
each model.generate call (including its retained prompt and answer history),
then teacher-forces the continuation from that call.  Numerical values may
still differ slightly from incremental generate() logits because forward-pass
and KV-cache implementations can use different floating-point kernels.
"""

from collections.abc import Callable

import torch

from longbench_utils import reconstruct_generation_context, token_ids_sha256

TRACE_FORMAT = "head_tail_prompt_recent_generation_v1"


@torch.no_grad()
def score_traced_generation(
    *,
    model,
    tokenizer,
    prompt: str,
    row: dict,
    max_length: int,
    use_chat_template: bool,
    nll_fn: Callable,
    tokenize_prompt_fn: Callable,
) -> tuple[float, int, int]:
    """Return (summed NLL, scored content tokens, verified generation chunks).

    Reject missing, damaged, reordered, overlapping or inconsistent trace
    metadata before accepting scores.  The recorded prompt/context SHA256
    values prevent silent scoring against a different tokenized prompt.

    A full teacher-forced call uses:
        input_ids = recorded_generation_context + generated_chunk_content
        labels    = -100 for context + actual chunk-content token IDs
    `nll_fn` handles the causal shift as in the existing scorer.
    """
    trace = row.get("generation_trace")
    if not isinstance(trace, dict) or trace.get("format") != TRACE_FORMAT:
        raise ValueError("Missing or unsupported Step 2 generation trace")
    if max_length < 2 or int(row.get("max_length", -1)) != max_length:
        raise ValueError("Scoring max_length must match the recorded generation max_length")

    prompt_ids = tokenize_prompt_fn(
        tokenizer=tokenizer, prompt=prompt, use_chat_template=use_chat_template,
    )
    if token_ids_sha256(prompt_ids) != trace.get("prompt_token_ids_sha256"):
        raise ValueError("Prompt token checksum mismatch: wrong prompt, dataset or tokenizer")

    if "generated_token_ids" not in row or "generated_content_token_ids" not in row:
        raise ValueError("Traced scoring requires exact raw and content token IDs")
    raw_ids = [int(x) for x in row["generated_token_ids"]]
    content_ids = [int(x) for x in row["generated_content_token_ids"]]
    if raw_ids[:len(content_ids)] != content_ids:
        raise ValueError("Content token IDs must be an unchanged prefix of raw token IDs")
    if len(content_ids) > len(raw_ids):
        raise ValueError("Content token count exceeds raw token count")

    trailing_specials = {
        x for x in (tokenizer.eos_token_id, tokenizer.pad_token_id) if x is not None
    }
    if any(x not in trailing_specials for x in raw_ids[len(content_ids):]):
        raise ValueError("Only trailing EOS/pad IDs may be excluded from self-PPL")
    if row.get("generated_tokens") is not None and int(row["generated_tokens"]) != len(content_ids):
        raise ValueError("Recorded generated_tokens disagrees with content token IDs")
    if row.get("generated_tokens_raw") is not None and int(row["generated_tokens_raw"]) != len(raw_ids):
        raise ValueError("Recorded generated_tokens_raw disagrees with raw token IDs")

    chunks = trace.get("chunks")
    if not isinstance(chunks, list):
        raise ValueError("Generation trace must contain a chunks list")
    if not chunks and raw_ids:
        raise ValueError("Generated tokens exist but trace has no chunks")

    device = next(model.parameters()).device
    total_nll, scored_tokens = 0.0, 0
    next_start = 0
    for chunk_idx, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            raise ValueError("Malformed generation chunk")
        if int(chunk["chunk_index"]) != chunk_idx:
            raise ValueError("Generation chunk indices are not contiguous")

        start = int(chunk["generated_start"])
        end = int(chunk["generated_end"])
        request = int(chunk["requested_new_tokens"])
        if start != next_start or not (0 <= start < end <= len(raw_ids)):
            raise ValueError("Generation token spans are missing, overlapping or out of order")
        if not (0 < end - start <= request < max_length):
            raise ValueError("Generated chunk length is inconsistent with requested tokens")
        if int(chunk["generated_history_end"]) != start:
            raise ValueError("Generated-history end must equal the start of this chunk")

        context_ids = reconstruct_generation_context(prompt_ids, raw_ids, chunk)
        # These were the actual token IDs fed to generate(): the budget was
        # reserved for the *requested* token cap, not the final shorter EOS call.
        if len(context_ids) < 1 or len(context_ids) + request > max_length:
            raise ValueError("Generation context exceeds recorded max_length")

        content_end = min(end, len(content_ids))
        if start < content_end:
            chunk_content = raw_ids[start:content_end]
            input_ids = torch.tensor(
                [context_ids + chunk_content], dtype=torch.long, device=device,
            )
            labels = torch.tensor(
                [[-100] * len(context_ids) + chunk_content],
                dtype=torch.long, device=device,
            )
            nll, n_tokens = nll_fn(model=model, input_ids=input_ids, labels=labels)
            if n_tokens != len(chunk_content):
                raise RuntimeError("Causal NLL scorer omitted or duplicated content tokens")
            total_nll += nll
            scored_tokens += n_tokens
        next_start = end

    if next_start != len(raw_ids):
        raise ValueError("Generation trace fails to cover all raw generated tokens")
    if scored_tokens != len(content_ids):
        raise RuntimeError("Every generated content token must be scored exactly once")
    return total_nll, scored_tokens, len(chunks)
