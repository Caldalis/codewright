

from __future__ import annotations

from codewright.context.manager import approx_token_count
from codewright.llm.base import CanonicalMessage


class History:

    def __init__(self) -> None:
        self._items: list[CanonicalMessage] = []

    def snapshot(self) -> tuple[CanonicalMessage, ...]:
        return tuple(self._items)

    def append(self, msg: CanonicalMessage) -> None:
        self._items.append(msg)

    def replace_all(self, items: list[CanonicalMessage]) -> None:
        self._items = list(items)

    def total_tokens(self) -> int:
        total = 0
        for m in self._items:
            content = m.content
            if isinstance(content, str):
                total += approx_token_count(content)
            else:
                for block in content:
                    if block.text:
                        total += approx_token_count(block.text)
        return total

    def __len__(self) -> int:
        return len(self._items)
