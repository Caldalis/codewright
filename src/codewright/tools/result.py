from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ToolResult:
    success: bool
    body: str
    structured_data: dict[str, Any] = field(default_factory=dict)
