from __future__ import annotations

import abc
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:  
    pass

Role = Literal["system", "developer", "user", "assistant", "tool"]

@dataclass(frozen=True)
class ToolCallBlock:
    call_id: str
    tool_name: str
    arguments_json: str


@dataclass(frozen=True)
class ContentBlock:

    text: str | None = None
    tool_calls: tuple[ToolCallBlock, ...] | None = None


@dataclass(frozen=True)
class CanonicalMessage:

    role: Role
    content: str | tuple[ContentBlock, ...]
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: tuple[ToolCallBlock, ...] | None = None


@dataclass(frozen=True)
class TokenUsage:
    input: int
    output: int
    total: int


StreamEventKind = Literal[
    "text_delta",
    "tool_call_started",
    "tool_call_arguments_delta",
    "tool_call_completed",
    "message_completed",
    "usage",
    "error",
]


@dataclass(frozen=True)
class StreamEvent:
  
    kind: StreamEventKind
    text: str | None = None
    tool_call: ToolCallBlock | None = None
    arguments_delta: str | None = None
    usage: TokenUsage | None = None
    error: str | None = None

    tool_call_index: int | None = None


@dataclass(frozen=True)
class ProviderRequest:

    messages: tuple[CanonicalMessage, ...]
    tools: tuple[Any, ...] = field(default_factory=tuple)
    model: str = ""
    parallel_tool_calls: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


class LLMProvider(abc.ABC):

    @abc.abstractmethod
    def stream(
        self,
        messages: list[CanonicalMessage],
        tools: list[Any],
        turn_context: Any,
    ) -> AsyncIterator[StreamEvent]: ...
