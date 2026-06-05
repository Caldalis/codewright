from __future__ import annotations

DEFAULT_MAX_CHARS = 10240


def truncate_middle(text: str, max_chars: int = DEFAULT_MAX_CHARS) -> str:

    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if len(text) <= max_chars:
        return text


    marker_budget = 64
    half = max(1, (max_chars - marker_budget) // 2)
    head = text[:half]
    tail = text[-half:]
    dropped = len(text) - len(head) - len(tail)
    marker = f"\n…[{dropped} chars truncated]…\n"
    return head + marker + tail
