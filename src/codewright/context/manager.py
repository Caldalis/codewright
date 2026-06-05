from __future__ import annotations

from codewright.llm.base import CanonicalMessage


def approx_token_count(text: str) -> int:
    return len(text) // 4


class ContextManager:
    def __init__(self, max_context_tokens: int = 128_000, compact_threshold: float = 0.9) -> None:
        if max_context_tokens <= 0:
            raise ValueError("max_context_tokens must be positive")
        if not (0.0 < compact_threshold <= 1.0):
            raise ValueError("compact_threshold must be in (0, 1]")

        from codewright.context.history import History

        self._history = History()
        self._max_context_tokens = max_context_tokens
        self._compact_threshold = compact_threshold

    def append(self, msg: CanonicalMessage) -> None:
        self._history.append(msg)

    def replace_all(self, items: list[CanonicalMessage]) -> None:
        self._history.replace_all(items)

    def snapshot(self) -> tuple[CanonicalMessage, ...]:
        return self._history.snapshot()

    def total_tokens(self) -> int:
        return self._history.total_tokens()

    def should_compact(self) -> bool:
        return self.total_tokens() >= int(self._compact_threshold * self._max_context_tokens)

    @property
    def max_context_tokens(self) -> int:
        return self._max_context_tokens

    @property
    def compact_threshold(self) -> float:
        return self._compact_threshold

    def __len__(self) -> int:
        return len(self._history)
