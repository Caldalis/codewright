from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ToolCall:

    call_id: str
    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)
