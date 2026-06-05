from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from codewright.agent.cancellation import CancellationToken
from codewright.agent.turn_context import TurnContext

if TYPE_CHECKING:  # pragma: no cover
    pass

@dataclass(frozen=True)
class ToolInvocation:
    session: Any
    turn_context: TurnContext
    call_id: str
    tool_name: str
    arguments: dict[str, Any]
    cancellation_token: CancellationToken
