from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Self


_ROOT_SEGMENT = "root"
_CHILD_SEGMENT_RE = re.compile(r"^[a-z][a-z0-9]*_[a-z0-9_]+$")


class AgentPathError(ValueError):
    """Raised when an AgentPath segment violates the naming convention."""


@dataclass(frozen=True, slots=True)
class AgentPath:

    segments: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.segments:
            raise AgentPathError("AgentPath must have at least one segment")
        if self.segments[0] != _ROOT_SEGMENT:
            raise AgentPathError(
                f"AgentPath must start with '{_ROOT_SEGMENT}', got {self.segments[0]!r}"
            )
        for seg in self.segments[1:]:
            if not _CHILD_SEGMENT_RE.match(seg):
                raise AgentPathError(
                    f"AgentPath segment {seg!r} must match <role>_<task_name>"
                )



    @classmethod
    def root(cls) -> Self:
        return cls(segments=(_ROOT_SEGMENT,))

    @classmethod
    def parse(cls, text: str) -> Self:
        if not text.startswith("/"):
            raise AgentPathError(f"AgentPath text must start with '/': {text!r}")
        parts = tuple(part for part in text.split("/") if part)
        return cls(segments=parts)


    def child(self, segment: str) -> Self:
        return type(self)(segments=(*self.segments, segment))

    def parent(self) -> Self | None:
        if len(self.segments) == 1:
            return None
        return type(self)(segments=self.segments[:-1])

    def is_root(self) -> bool:
        return len(self.segments) == 1



    def __str__(self) -> str:
        return "/" + "/".join(self.segments)


@dataclass(frozen=True, slots=True)
class InterAgentMessage:

    author: AgentPath
    recipient: AgentPath
    content: str
    trigger_turn: bool
    created_at: float = field(default_factory=time.time)
