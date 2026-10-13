"""Shared, model-independent helpers for LongBench scoring and generation."""

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
