from __future__ import annotations

from codewright.context.manager import approx_token_count
from codewright.llm.base import CanonicalMessage, ContentBlock, ToolCallBlock


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
        return sum(_message_token_count(msg) for msg in self._items)

    def __len__(self) -> int:
        return len(self._items)


def _message_token_count(msg: CanonicalMessage) -> int:
    total = _content_token_count(msg.content)
    if msg.name:
        total += approx_token_count(msg.name)
    if msg.tool_call_id:
        total += approx_token_count(msg.tool_call_id)
    for call in msg.tool_calls or ():
        total += _tool_call_token_count(call)
    return total


def _content_token_count(content: str | tuple[ContentBlock, ...]) -> int:
    if isinstance(content, str):
        return approx_token_count(content)
    total = 0
    for block in content:
        if block.text:
            total += approx_token_count(block.text)
        for call in block.tool_calls or ():
            total += _tool_call_token_count(call)
    return total


def _tool_call_token_count(call: ToolCallBlock) -> int:
    return (
        approx_token_count(call.call_id)
        + approx_token_count(call.tool_name)
        + approx_token_count(call.arguments_json)
    )
