"""Shared, model-independent helpers for LongBench scoring and generation."""

import hashlib
import json
from typing import Sequence, TypeVar

T = TypeVar("T")


def middle_truncate_tokens(token_ids: Sequence[T], budget: int) -> list[T]:
    """Keep the start and end of a token sequence, dropping its middle.

    The left side receives the extra token when ``budget`` is odd.
    This operates on token IDs, so it does not decode and re-tokenize.
    """
    if budget < 0:
        raise ValueError("budget must be nonnegative")
    if len(token_ids) <= budget:
        return list(token_ids)
    left = (budget + 1) // 2
    right = budget // 2
    if right == 0:
        return list(token_ids[:left])
    return list(token_ids[:left]) + list(token_ids[-right:])


# The next helpers encode the *actual* input sequence passed to model.generate()
# for each generation chunk. Step 3 can reconstruct it from original prompt IDs
# and the exact saved generated_token_ids, and reject a mismatched checksum.
def token_ids_sha256(token_ids: Sequence[int]) -> str:
    """Hash an ordered list of token IDs using stable JSON serialization."""
    encoded = json.dumps(list(token_ids), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_generation_context(
    prompt_ids: Sequence[int],
    generated_ids: Sequence[int],
    context_budget: int,
) -> tuple[list[int], dict]:
    """Preserve prompt head/tail, then the most recent generated history.

    Generated continuation has priority; an extremely long answer may
    eventually occupy the entire window. The returned plan encodes the
    concrete token sequence that will be passed to ``model.generate``.
    """
    if context_budget < 1:
        raise ValueError("context_budget must be >= 1")

    keep_generated = min(len(generated_ids), context_budget)
    generated_start = len(generated_ids) - keep_generated
    prompt_budget = context_budget - keep_generated

    if len(prompt_ids) <= prompt_budget:
        head, tail = len(prompt_ids), 0
    else:
        head = (prompt_budget + 1) // 2
        tail = prompt_budget // 2

    context_ids = (
        list(prompt_ids[:head])
        + (list(prompt_ids[-tail:]) if tail else [])
        + list(generated_ids[generated_start:])
    )
    if not context_ids:
        raise ValueError("No prompt or generated-history tokens to condition on")

    plan = {
        "prompt_head_tokens": head,
        "prompt_tail_tokens": tail,
        "generated_history_start": generated_start,
        "generated_history_end": len(generated_ids),
        "input_context_tokens": len(context_ids),
        "input_context_sha256": token_ids_sha256(context_ids),
        "prompt_middle_truncated": head + tail < len(prompt_ids),
    }
    return context_ids, plan


def reconstruct_generation_context(
    prompt_ids: Sequence[int],
    generated_ids: Sequence[int],
    plan: dict,
) -> list[int]:
    """Rebuild an exact generation input and verify its recorded checksum.

    ``generated_ids`` must be the raw generated_token_ids, including any EOS.
    The caller should separately verify the prompt-token checksum.
    """
    head = int(plan["prompt_head_tokens"])
    tail = int(plan["prompt_tail_tokens"])
    start = int(plan["generated_history_start"])
    end = int(plan["generated_history_end"])
    if not (0 <= head <= len(prompt_ids)
            and 0 <= tail <= len(prompt_ids) - head):
        raise ValueError("Invalid recorded prompt spans")
    if not 0 <= start <= end <= len(generated_ids):
        raise ValueError("Invalid recorded generated-history span")
    result = (
        list(prompt_ids[:head])
        + (list(prompt_ids[-tail:]) if tail else [])
        + list(generated_ids[start:end])
    )
    if len(result) != int(plan["input_context_tokens"]):
        raise ValueError("Recorded generation context length mismatch")
    if token_ids_sha256(result) != plan["input_context_sha256"]:
        raise ValueError("Recorded generation context checksum mismatch")
    return result
